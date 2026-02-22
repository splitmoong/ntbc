"""
NTBC Color Network for BC1 compression (multi-RGB-texture capable).
Based on: Neural Texture Block Compression (arXiv:2407.09543)

If num_textures=T, the network outputs (3*T) values per texel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- BC1 helpers ----------

_BC1_W = torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=torch.float32)


def clamp_coords01(coords: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return coords.clamp(0.0, 1.0 - eps)


def endpoints6_to_e0e1(endpoints6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return endpoints6[..., 0:3], endpoints6[..., 3:6]


def bc1_palette_from_endpoints(e0: torch.Tensor, e1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wv = w.view(*([1] * (e0.ndim - 1)), -1, 1).to(device=e0.device, dtype=e0.dtype)
    return (1.0 - wv) * e0.unsqueeze(-2) + wv * e1.unsqueeze(-2)


def _fake_quantize_asymmetric_with_range(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, bits: int = 8,
) -> torch.Tensor:
    qmin, qmax = 0.0, float((1 << bits) - 1)
    x_clamped = torch.clamp(x, alpha, beta)
    scale = (beta - alpha) / (qmax - qmin)
    zero_point = torch.round(-alpha / scale).clamp(qmin, qmax)
    q = torch.round(x_clamped / scale + zero_point).clamp(qmin, qmax)
    x_q = (q - zero_point) * scale
    return x_clamped + (x_q - x_clamped).detach()


def _infer_num_textures_from_flat_colors(colors_flat: torch.Tensor) -> int:
    if colors_flat.ndim != 2:
        raise ValueError(f"colors_flat must be (B, 3*T). Got {tuple(colors_flat.shape)}")
    D = int(colors_flat.shape[1])
    if D % 3 != 0:
        raise ValueError(f"Color dimension must be multiple of 3. Got {D}")
    return D // 3


def split_colors_flat(colors_flat: torch.Tensor) -> torch.Tensor:
    """(B,3*T) -> (B,T,3)"""
    T = _infer_num_textures_from_flat_colors(colors_flat)
    return colors_flat.view(colors_flat.shape[0], T, 3)


# ---------- Multi-resolution Feature Grid ----------

class MultiResFeatureGrid2D(nn.Module):
    """Multi-resolution feature grids with bilinear interpolation."""

    def __init__(
        self,
        num_levels: int = 8,
        base_resolution: int = 16,
        finest_resolution: int = 2048,
        feature_dim: int = 2,
        init_range: float = 1e-4,
        param_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_levels = int(num_levels)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.feature_dim = int(feature_dim)
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
            nn.init.uniform_(g, a=-init_range, b=+init_range)
            grids.append(g)
        self.grids = nn.ParameterList(grids)
        self.output_dim = self.num_levels * self.feature_dim
        self.qat_enabled = False
        self.qat_bits = 8

    def enable_qat(self, bits: int = 8) -> None:
        self.qat_enabled = True; self.qat_bits = int(bits)

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
            f00 = grid[x0 + y0 * r].to(torch.float32)
            f10 = grid[x1 + y0 * r].to(torch.float32)
            f01 = grid[x0 + y1 * r].to(torch.float32)
            f11 = grid[x1 + y1 * r].to(torch.float32)
            if self.qat_enabled:
                f00 = _fake_quantize_asymmetric_with_range(f00, alpha, beta, bits=self.qat_bits)
                f10 = _fake_quantize_asymmetric_with_range(f10, alpha, beta, bits=self.qat_bits)
                f01 = _fake_quantize_asymmetric_with_range(f01, alpha, beta, bits=self.qat_bits)
                f11 = _fake_quantize_asymmetric_with_range(f11, alpha, beta, bits=self.qat_bits)
            f = (f00 * (1.0 - fx) + f10 * fx) * (1.0 - fy) + (f01 * (1.0 - fx) + f11 * fx) * fy
            feats.append(f.to(dtype=grid.dtype))
        return torch.cat(feats, dim=1)


# ---------- Color Network ----------

class ColorNetwork(nn.Module):
    """Predicts uncompressed RGB color(s) from 2D texture coordinates (u,v).
    If num_textures=T, outputs (B, 3*T) with sigmoid activation.
    """

    def __init__(
        self,
        num_textures: int = 1,
        param_dtype: torch.dtype = torch.float32,
        finest_resolution: int = 2048,
        base_resolution: int = 16,
        num_levels: int = 8,
    ):
        super().__init__()
        if num_textures < 1:
            raise ValueError("num_textures must be >= 1")
        self.num_textures = int(num_textures)
        self.encoding = MultiResFeatureGrid2D(
            num_levels=int(num_levels), base_resolution=int(base_resolution),
            finest_resolution=int(finest_resolution), feature_dim=2, init_range=1e-4,
            param_dtype=(torch.float16 if param_dtype == torch.float16 else torch.float32),
        )
        out_dim = 3 * self.num_textures
        self.mlp = nn.Sequential(
            nn.Linear(self.encoding.output_dim, 64), nn.SELU(inplace=True),
            nn.Linear(64, 64), nn.SELU(inplace=True),
            nn.Linear(64, 64), nn.SELU(inplace=True),
            nn.Linear(64, out_dim), nn.Sigmoid(),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoding(uv).to(torch.float32))


# ---------- Color Loss ----------

@dataclass
class ColorLossOutput:
    total: torch.Tensor
    lc: torch.Tensor
    lcd: torch.Tensor
    hard_indices: torch.Tensor


def color_loss_bc1(
    pred_color: torch.Tensor,      # (B,3)
    ref_color: torch.Tensor,       # (B,3)
    ref_endpoints6: torch.Tensor,  # (B,6)
    temperature: float = 0.01,
    reduction: str = "mean",
) -> ColorLossOutput:
    lc = F.mse_loss(pred_color, ref_color, reduction=reduction)
    ref_e0, ref_e1 = endpoints6_to_e0e1(ref_endpoints6)
    w_levels = _BC1_W.to(device=pred_color.device, dtype=pred_color.dtype)
    pal = bc1_palette_from_endpoints(ref_e0, ref_e1, w=w_levels)  # (B,4,3)
    diff = pred_color.unsqueeze(1) - pal
    dn = -torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
    hard_n = torch.argmax(dn, dim=-1).to(torch.uint8)
    w_hard = w_levels[hard_n.long()]
    decoded_hard = (1.0 - w_hard).unsqueeze(-1) * ref_e0 + w_hard.unsqueeze(-1) * ref_e1
    p = F.softmax(dn / float(temperature), dim=-1)
    w_soft = (p * w_levels.view(1, 4)).sum(dim=-1)
    decoded_soft = (1.0 - w_soft).unsqueeze(-1) * ref_e0 + w_soft.unsqueeze(-1) * ref_e1
    decoded = decoded_hard + (decoded_soft - decoded_soft.detach())
    lcd = F.mse_loss(decoded, ref_color, reduction=reduction)
    return ColorLossOutput(total=lc + lcd, lc=lc, lcd=lcd, hard_indices=hard_n)


def color_loss_bc1_multi(
    pred_colors: torch.Tensor,
    ref_colors: torch.Tensor,
    ref_endpoints: torch.Tensor,
    temperature: float = 0.01,
    reduction: str = "mean",
) -> ColorLossOutput:
    pred_c = split_colors_flat(pred_colors) if pred_colors.ndim == 2 else pred_colors
    ref_c = split_colors_flat(ref_colors) if ref_colors.ndim == 2 else ref_colors
    if ref_endpoints.ndim == 2:
        D = int(ref_endpoints.shape[1])
        T = D // 6
        ref_e = ref_endpoints.view(ref_endpoints.shape[0], T, 6)
    else:
        ref_e = ref_endpoints
    B, T, _ = pred_c.shape
    totals, lcs, lcds, hards = [], [], [], []
    for t in range(T):
        out_t = color_loss_bc1(pred_c[:, t, :], ref_c[:, t, :], ref_e[:, t, :], temperature, reduction)
        totals.append(out_t.total); lcs.append(out_t.lc)
        lcds.append(out_t.lcd); hards.append(out_t.hard_indices)
    return ColorLossOutput(
        total=torch.stack(totals).mean(), lc=torch.stack(lcs).mean(),
        lcd=torch.stack(lcds).mean(), hard_indices=torch.stack(hards, dim=1),
    )
