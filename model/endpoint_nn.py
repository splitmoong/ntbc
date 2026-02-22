"""
NTBC Endpoint Network for BC1 compression (multi-RGB-texture capable).
Based on: Neural Texture Block Compression (arXiv:2407.09543)

If num_textures=T, the network outputs (6*T) values per block:
    [r0,g0,b0,r1,g1,b1] repeated T times (each in [0,1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Quantization helpers ----------

def _fake_quantize_asymmetric_with_range(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    bits: int = 8,
) -> torch.Tensor:
    """Asymmetric fake quantization with straight-through estimator."""
    qmin = 0.0
    qmax = float((1 << bits) - 1)
    x_clamped = torch.clamp(x, alpha, beta)
    scale = (beta - alpha) / (qmax - qmin)
    zero_point = torch.round(-alpha / scale).clamp(qmin, qmax)
    q = torch.round(x_clamped / scale + zero_point).clamp(qmin, qmax)
    x_q = (q - zero_point) * scale
    return x_clamped + (x_q - x_clamped).detach()


def _fake_quantize_rgb565_ste(endpoints6: torch.Tensor) -> torch.Tensor:
    """Fake-quantize 6-channel endpoints to RGB565 levels with STE.
    Input/output: (..., 6) in [0,1] = [r0,g0,b0,r1,g1,b1]
    """
    levels = torch.tensor([31.0, 63.0, 31.0, 31.0, 63.0, 31.0],
                          device=endpoints6.device, dtype=endpoints6.dtype)
    shape = [1] * (endpoints6.ndim - 1) + [6]
    lv = levels.view(*shape)
    scaled = (endpoints6 * lv).round()
    clamped = torch.min(torch.max(scaled, torch.zeros_like(scaled)), lv)
    quantized = clamped / lv
    return endpoints6 + (quantized - endpoints6).detach()


# ---------- Utilities (BC1) ----------

_BC1_W = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=torch.float32)


def clamp_coords01(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return coords.clamp(0.0, 1.0 - eps)


def endpoints6_to_e0e1(endpoints6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return endpoints6[..., 0:3], endpoints6[..., 3:6]


def bc1_palette_from_endpoints(e0: torch.Tensor, e1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wv = w.view(*([1] * (e0.ndim - 1)), -1, 1).to(device=e0.device, dtype=e0.dtype)
    return (1.0 - wv) * e0.unsqueeze(-2) + wv * e1.unsqueeze(-2)


def pack_rgb565_from_epq01(endpoints6: torch.Tensor) -> torch.Tensor:
    """Converts predicted endpoints to packed RGB565 format. (B,6) -> (B,2) uint16"""
    r0 = (endpoints6[:, 0] * 31.0).round().clamp(0, 31).to(torch.int32)
    g0 = (endpoints6[:, 1] * 63.0).round().clamp(0, 63).to(torch.int32)
    b0 = (endpoints6[:, 2] * 31.0).round().clamp(0, 31).to(torch.int32)
    r1 = (endpoints6[:, 3] * 31.0).round().clamp(0, 31).to(torch.int32)
    g1 = (endpoints6[:, 4] * 63.0).round().clamp(0, 63).to(torch.int32)
    b1 = (endpoints6[:, 5] * 31.0).round().clamp(0, 31).to(torch.int32)
    c0 = (r0 << 11) | (g0 << 5) | b0
    c1 = (r1 << 11) | (g1 << 5) | b1
    return torch.stack([c0, c1], dim=1).to(torch.uint16)


def _infer_num_textures_from_flat(endpoints_flat: torch.Tensor) -> int:
    if endpoints_flat.ndim != 2:
        raise ValueError(f"endpoints_flat must be (B, 6*T). Got {tuple(endpoints_flat.shape)}")
    D = int(endpoints_flat.shape[1])
    if D % 6 != 0:
        raise ValueError(f"Endpoint dimension must be multiple of 6. Got {D}")
    return D // 6


def split_endpoints_flat(endpoints_flat: torch.Tensor) -> torch.Tensor:
    """(B,6*T) -> (B,T,6)"""
    T = _infer_num_textures_from_flat(endpoints_flat)
    return endpoints_flat.view(endpoints_flat.shape[0], T, 6)


# ---------- Multi-resolution Feature Grid ----------

class MultiResFeatureGrid2D(nn.Module):
    """Multi-resolution feature grids with bilinear interpolation."""

    def __init__(
        self,
        num_levels: int = 7,
        base_resolution: int = 16,
        finest_resolution: int = 1024,
        feature_dim: int = 2,
        init_range: float = 1e-4,
        param_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.feature_dim = int(feature_dim)
        self.init_range = float(init_range)
        self.param_dtype = param_dtype

        if self.num_levels == 1:
            self._resolutions = [self.base_resolution]
        else:
            b = math.exp((math.log(self.finest_resolution) - math.log(self.base_resolution)) / (self.num_levels - 1))
            self._resolutions = [int(math.floor(self.base_resolution * (b ** l) + 1e-9)) for l in range(self.num_levels)]
            self._resolutions[-1] = self.finest_resolution

        grids = []
        for r in self._resolutions:
            g = nn.Parameter(torch.empty((r * r, self.feature_dim), dtype=self.param_dtype))
            nn.init.uniform_(g, a=-self.init_range, b=+self.init_range)
            grids.append(g)
        self.grids = nn.ParameterList(grids)
        self.output_dim = self.num_levels * self.feature_dim
        self.qat_enabled = False
        self.qat_bits = 8

    def enable_qat(self, bits: int = 8) -> None:
        self.qat_enabled = True
        self.qat_bits = int(bits)

    def disable_qat(self) -> None:
        self.qat_enabled = False

    @property
    def resolutions(self):
        return list(self._resolutions)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be (B,2), got {tuple(coords.shape)}")
        coords = clamp_coords01(coords).to(dtype=torch.float32)
        feats = []
        x, y = coords[:, 0], coords[:, 1]
        for lvl, r in enumerate(self._resolutions):
            grid = self.grids[lvl]
            alpha = grid.min().detach() if self.qat_enabled else None
            beta = grid.max().detach() if self.qat_enabled else None
            xs, ys = x * (r - 1), y * (r - 1)
            x0 = torch.floor(xs).to(torch.int64).clamp(0, r - 2)
            y0 = torch.floor(ys).to(torch.int64).clamp(0, r - 2)
            fx = (xs - x0.to(xs.dtype)).unsqueeze(1)
            fy = (ys - y0.to(ys.dtype)).unsqueeze(1)
            x1, y1 = x0 + 1, y0 + 1
            idx00, idx10 = x0 + y0 * r, x1 + y0 * r
            idx01, idx11 = x0 + y1 * r, x1 + y1 * r
            f00 = grid[idx00].to(torch.float32)
            f10 = grid[idx10].to(torch.float32)
            f01 = grid[idx01].to(torch.float32)
            f11 = grid[idx11].to(torch.float32)
            if self.qat_enabled:
                f00 = _fake_quantize_asymmetric_with_range(f00, alpha, beta, bits=self.qat_bits)
                f10 = _fake_quantize_asymmetric_with_range(f10, alpha, beta, bits=self.qat_bits)
                f01 = _fake_quantize_asymmetric_with_range(f01, alpha, beta, bits=self.qat_bits)
                f11 = _fake_quantize_asymmetric_with_range(f11, alpha, beta, bits=self.qat_bits)
            f = (f00 * (1.0 - fx) + f10 * fx) * (1.0 - fy) + (f01 * (1.0 - fx) + f11 * fx) * fy
            feats.append(f.to(dtype=grid.dtype))
        return torch.cat(feats, dim=1)


# ---------- Endpoint Network ----------

class EndpointNetwork(nn.Module):
    """Predicts BC1 endpoints from normalized block coordinates."""

    def __init__(
        self,
        num_textures: int = 1,
        num_levels: int = 7,
        base_resolution: int = 16,
        finest_resolution: int = 1024,
        feature_dim: int = 2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 3,
        param_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        if num_textures < 1:
            raise ValueError("num_textures must be >= 1")
        self.num_textures = int(num_textures)
        self.encoding = MultiResFeatureGrid2D(
            num_levels=num_levels, base_resolution=base_resolution,
            finest_resolution=finest_resolution, feature_dim=feature_dim,
            init_range=1e-4, param_dtype=param_dtype,
        )
        out_dim = 6 * self.num_textures
        layers = []
        in_dim = self.encoding.output_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)
        self._init_mlp_he()
        if param_dtype == torch.float16:
            self.mlp = self.mlp.half()
        self.register_buffer("bc1_w", _BC1_W.clone(), persistent=False)

    def _init_mlp_he(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        feats = self.encoding(coords)
        return self.mlp(feats)

    @torch.no_grad()
    def predict_rgb565(self, coords: torch.Tensor) -> torch.Tensor:
        ep_flat = self.forward(coords)
        if self.num_textures == 1:
            return pack_rgb565_from_epq01(ep_flat)
        ep = ep_flat.view(ep_flat.shape[0], self.num_textures, 6)
        return torch.stack([pack_rgb565_from_epq01(ep[:, t, :]) for t in range(self.num_textures)], dim=1)


# ---------- Endpoint Loss ----------

@dataclass
class EndpointLossOutput:
    total: torch.Tensor
    le: torch.Tensor
    lcd: torch.Tensor
    hard_indices: torch.Tensor


def endpoint_loss_bc1(
    pred_endpoints6: torch.Tensor,
    ref_endpoints6: torch.Tensor,
    ref_colors: torch.Tensor,      # (B, 16, 3)
    temperature: float = 0.01,
    reduction: str = "mean",
) -> EndpointLossOutput:
    le = F.mse_loss(pred_endpoints6, ref_endpoints6, reduction=reduction)
    pred_q = _fake_quantize_rgb565_ste(pred_endpoints6)
    pred_e0, pred_e1 = endpoints6_to_e0e1(pred_q)
    w = _BC1_W.to(device=pred_endpoints6.device, dtype=pred_endpoints6.dtype)
    pal_pred = bc1_palette_from_endpoints(pred_e0, pred_e1, w=w)  # (B,4,3)
    diff = ref_colors.unsqueeze(2) - pal_pred.unsqueeze(1)
    dn = -torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
    hard_n = torch.argmax(dn, dim=-1).to(torch.uint8)
    ref_e0, ref_e1 = endpoints6_to_e0e1(ref_endpoints6)
    w_levels = _BC1_W.to(device=ref_endpoints6.device, dtype=ref_endpoints6.dtype)
    w_hard = w_levels[hard_n.long()]
    decoded_hard = (1.0 - w_hard).unsqueeze(-1) * ref_e0.unsqueeze(1) + w_hard.unsqueeze(-1) * ref_e1.unsqueeze(1)
    p = F.softmax(dn / float(temperature), dim=-1)
    w_soft = (p * w_levels.view(1, 1, 4)).sum(dim=-1)
    decoded_soft = (1.0 - w_soft).unsqueeze(-1) * ref_e0.unsqueeze(1) + w_soft.unsqueeze(-1) * ref_e1.unsqueeze(1)
    decoded = decoded_hard + (decoded_soft - decoded_soft.detach())
    lcd = F.mse_loss(decoded, ref_colors, reduction=reduction)
    return EndpointLossOutput(total=le + lcd, le=le, lcd=lcd, hard_indices=hard_n)


def endpoint_loss_bc1_multi(
    pred_endpoints: torch.Tensor,
    ref_endpoints: torch.Tensor,
    ref_colors: torch.Tensor,
    temperature: float = 0.01,
    reduction: str = "mean",
) -> EndpointLossOutput:
    pred_e = split_endpoints_flat(pred_endpoints) if pred_endpoints.ndim == 2 else pred_endpoints
    ref_e = split_endpoints_flat(ref_endpoints) if ref_endpoints.ndim == 2 else ref_endpoints
    B, T, _ = pred_e.shape
    ref_c = ref_colors.unsqueeze(1) if (T == 1 and ref_colors.ndim == 3) else ref_colors
    totals, les, lcds, hards = [], [], [], []
    for t in range(T):
        out_t = endpoint_loss_bc1(pred_e[:, t, :], ref_e[:, t, :], ref_c[:, t, :, :], temperature, reduction)
        totals.append(out_t.total); les.append(out_t.le)
        lcds.append(out_t.lcd); hards.append(out_t.hard_indices)
    return EndpointLossOutput(
        total=torch.stack(totals).mean(), le=torch.stack(les).mean(),
        lcd=torch.stack(lcds).mean(), hard_indices=torch.stack(hards, dim=1),
    )
