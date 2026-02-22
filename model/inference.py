"""
NTBC BC1 Inference — loads a merged compressed state dict and writes BC1 DDS files.

Usage:
    infer = NTBCInference(
        merged_ckpt="path/to/ntbc_bc1_merged_compressed.pt",
        coords_json="path/to/Inference_input.json",
        output_dir="path/to/output",
    )
    result = infer.run(callback=print)
    # result.out_dds_paths -> list of written DDS files
    # result.out_png_paths -> list of written preview PNGs
"""

from __future__ import annotations

import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from endpoint_nn import (
    EndpointNetwork, pack_rgb565_from_epq01,
    bc1_palette_from_endpoints, _BC1_W,
)
from color_nn import ColorNetwork
from model_compress import decompress_state_dict


# ---------- DDS writer ----------

def _dds_header_dxt1(width: int, height: int) -> bytes:
    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)
    linear_size = blocks_x * blocks_y * 8
    header = struct.pack(
        "<I I I I I I I 11I",
        124, 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000,
        height, width, linear_size, 0, 0, *([0] * 11)
    )
    pixel_format = struct.pack("<I I 4s I I I I I", 32, 0x4, b"DXT1", 0, 0, 0, 0, 0)
    caps = struct.pack("<I I I I I", 0x1000, 0, 0, 0, 0)
    return b"DDS " + header + pixel_format + caps


def _write_dds_dxt1(path: Path, width: int, height: int, bc1_blocks: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_dds_header_dxt1(width, height) + bc1_blocks)


# ---------- State dict helpers ----------

def _split_merged_state_dict(merged: dict) -> Tuple[dict, dict]:
    ep_sd, col_sd = {}, {}
    for k, v in merged.items():
        if k.startswith("endpoint."):
            ep_sd[k[len("endpoint."):]] = v
        elif k.startswith("color."):
            col_sd[k[len("color."):]] = v
    return ep_sd, col_sd


def _infer_grid_params(state: dict, prefix: str = "encoding.grids."):
    grid_keys = sorted([k for k in state if k.startswith(prefix)],
                       key=lambda k: int(k.split(".")[2]) if len(k.split(".")) > 2 else -1)
    if not grid_keys:
        raise ValueError("No grid keys found in state_dict.")
    g0, glast = state[grid_keys[0]], state[grid_keys[-1]]
    num_levels = len(grid_keys)
    base_res = int(round(math.sqrt(g0.shape[0])))
    finest_res = int(round(math.sqrt(glast.shape[0])))
    feature_dim = int(g0.shape[1])
    return num_levels, base_res, finest_res, feature_dim, g0.dtype


def _infer_out_dim_from_mlp(state: dict) -> int:
    best_i, best_out = -1, None
    for k, v in state.items():
        if k.startswith("mlp.") and k.endswith(".weight") and v.ndim == 2:
            try:
                i = int(k.split(".")[1])
            except Exception:
                continue
            if i > best_i:
                best_i, best_out = i, int(v.shape[0])
    if best_out is None:
        raise ValueError("Could not infer MLP output dim from state_dict.")
    return best_out


def _rgb565_to_q01(c: torch.Tensor) -> torch.Tensor:
    c = c.to(torch.int32)
    return torch.stack([
        ((c >> 11) & 31).to(torch.float32) / 31.0,
        ((c >> 5) & 63).to(torch.float32) / 63.0,
        (c & 31).to(torch.float32) / 31.0,
    ], dim=-1)


_OFF_X = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
_OFF_Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.long)


def _pack_indices_u32(indices_16: torch.Tensor) -> torch.Tensor:
    idx = indices_16.to(torch.int64)
    shifts = (2 * torch.arange(16, device=idx.device, dtype=torch.int64)).view(1, 16)
    return torch.sum(((idx & 3) << shifts), dim=1).to(torch.uint32)


def _make_output_paths(base_dds: Path, base_png: Optional[Path], names: List[str]):
    if len(names) == 1:
        return [base_dds], [base_png]
    out_dds, out_png = [], []
    for i, n in enumerate(names):
        safe = (n or f"tex{i:02d}").replace(" ", "_")
        out_dds.append(base_dds.parent / f"{base_dds.stem}_{safe}{base_dds.suffix}")
        out_png.append(
            base_png.parent / f"{base_png.stem}_{safe}{base_png.suffix}" if base_png else None
        )
    return out_dds, out_png


# ---------- Result dataclass ----------

@dataclass
class InferenceResult:
    width: int
    height: int
    out_dds_paths: List[Path]
    out_png_paths: List[Path]


# ---------- NTBCInference class ----------

class NTBCInference:
    """
    Loads a merged compressed checkpoint and runs NTBC BC1 inference.

    Args:
        merged_ckpt:    Path to ntbc_bc1_merged_compressed.pt
        coords_json:    Path to Inference_input.json
        output_dir:     Where to write output DDS and PNG files
        device:         "cuda" or "cpu"
        block_batch:    How many blocks to process at once (default 65536)
        save_preview:   Whether to save preview PNG files
    """

    def __init__(
        self,
        merged_ckpt: str,
        coords_json: str,
        output_dir: str,
        device: Optional[str] = None,
        block_batch: int = 65536,
        save_preview: bool = True,
    ):
        self.merged_ckpt = Path(merged_ckpt).expanduser().resolve()
        self.coords_json = Path(coords_json).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(raw_device).startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available — falling back to CPU.")
            raw_device = "cpu"
        self.device = raw_device
        self.block_batch = block_batch
        self.save_preview = save_preview

    @torch.no_grad()
    def run(self, callback: Optional[Callable[[str], None]] = None) -> InferenceResult:
        def log(msg):
            if callback:
                callback(msg)
            else:
                print(msg)

        device = self.device
        coords = json.loads(self.coords_json.read_text())
        blocks_x = int(coords["blocks_x"])
        blocks_y = int(coords["blocks_y"])
        W, H = blocks_x * 4, blocks_y * 4
        names = coords.get("texture_names") or [f"tex{i:02d}" for i in range(int(coords.get("num_textures", 1)))]

        log(f"[Infer] {W}x{H}, blocks=({blocks_x},{blocks_y}), textures={names}")

        merged = torch.load(self.merged_ckpt, map_location="cpu", weights_only=False)
        ep_comp, col_comp = _split_merged_state_dict(merged)
        ep_state = decompress_state_dict(ep_comp)
        col_state = decompress_state_dict(col_comp)

        ep_out = _infer_out_dim_from_mlp(ep_state)
        col_out = _infer_out_dim_from_mlp(col_state)
        T = ep_out // 6
        if len(names) != T:
            names = [names[i] if i < len(names) else f"tex{i:02d}" for i in range(T)]

        nl, br, fr, fd, dt = _infer_grid_params(ep_state)
        param_dtype = torch.float16 if (dt == torch.float16 and device == "cuda") else torch.float32
        ep_net = EndpointNetwork(num_textures=T, num_levels=nl, base_resolution=br,
                                  finest_resolution=fr, feature_dim=fd,
                                  hidden_dim=64, num_hidden_layers=3, param_dtype=param_dtype).to(device)
        ep_net.load_state_dict(ep_state, strict=True); ep_net.eval()

        nl, br, fr, fd, dt = _infer_grid_params(col_state)
        param_dtype = torch.float16 if (dt == torch.float16 and device == "cuda") else torch.float32
        col_net = ColorNetwork(num_textures=T, param_dtype=param_dtype, finest_resolution=fr,
                                base_resolution=br, num_levels=nl).to(device)
        col_net.load_state_dict(col_state, strict=True); col_net.eval()

        amp_enabled = (device == "cuda")
        autocast_device = "cuda" if device == "cuda" else "cpu"

        bx_all = torch.arange(blocks_x, dtype=torch.int64)
        by_all = torch.arange(blocks_y, dtype=torch.int64)
        grid_by, grid_bx = torch.meshgrid(by_all, bx_all, indexing="ij")
        bxby = torch.stack([grid_bx.reshape(-1), grid_by.reshape(-1)], dim=1)
        N = int(bxby.shape[0])
        denom_x = float(max(1, blocks_x - 1))
        denom_y = float(max(1, blocks_y - 1))
        st = torch.stack([bxby[:, 0].float() / denom_x, bxby[:, 1].float() / denom_y], dim=1)

        bc1_bytes = [bytearray(N * 8) for _ in range(T)]
        previews = [np.zeros((H, W, 3), dtype=np.uint8) for _ in range(T)] if self.save_preview else None

        base_dds = self.output_dir / "ntbc_out.dds"
        base_png = self.output_dir / "ntbc_out_preview.png" if self.save_preview else None
        out_dds_paths, out_png_paths = _make_output_paths(base_dds, base_png, names)
        map_idx = torch.tensor([0, 2, 3, 1], dtype=torch.uint8)

        for start in range(0, N, self.block_batch):
            end = min(N, start + self.block_batch)
            b = end - start
            bxby_b = bxby[start:end]
            st_b = st[start:end].to(device=device)

            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                ep_pred_flat = ep_net(st_b).to(torch.float32)
            ep_pred = ep_pred_flat.view(b, T, 6)

            ep565 = torch.stack([pack_rgb565_from_epq01(ep_pred[:, t, :]).to(torch.int32) for t in range(T)], dim=1)
            c0, c1 = ep565[:, :, 0], ep565[:, :, 1]
            swap = (c0 <= c1)
            c0, c1 = torch.where(swap, c1, c0), torch.where(swap, c0, c1)
            equal = (c0 == c1)
            if equal.any():
                can_inc = (c0 < 0xFFFF)
                c0 = torch.where(equal & can_inc, c0 + 1, c0)
                c1 = torch.where(equal & ~can_inc, c1 - 1, c1)

            e0_q = _rgb565_to_q01(c0)
            e1_q = _rgb565_to_q01(c1)

            base_x = (bxby_b[:, 0] * 4).view(-1, 1)
            base_y = (bxby_b[:, 1] * 4).view(-1, 1)
            x = (base_x + _OFF_X.view(1, 16)).to(torch.int64)
            y = (base_y + _OFF_Y.view(1, 16)).to(torch.int64)

            u = (x.float() / float(max(1, W - 1))).to(device=device)
            v = (y.float() / float(max(1, H - 1))).to(device=device)
            uv = torch.stack([u, v], dim=-1).reshape(-1, 2)

            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                pred_flat = col_net(uv).to(torch.float32)
            pred = pred_flat.view(b * 16, T, 3).view(b, 16, T, 3).permute(0, 2, 1, 3).contiguous()

            w = _BC1_W.to(device=device, dtype=torch.float32)
            pal = bc1_palette_from_endpoints(e0_q.to(device), e1_q.to(device), w=w)
            diff = pred.unsqueeze(3) - pal.unsqueeze(2)
            idx_paper = torch.argmin((diff * diff).sum(dim=-1), dim=-1).to(torch.uint8)
            idx_bc1 = map_idx.to(device=idx_paper.device)[idx_paper.long()]

            packed_idx = _pack_indices_u32(idx_bc1.view(b * T, 16)).to("cpu").numpy().astype(np.uint32).reshape(b, T)
            c0_cpu = c0.to("cpu").to(torch.uint16).numpy()
            c1_cpu = c1.to("cpu").to(torch.uint16).numpy()

            for t in range(T):
                bb = bc1_bytes[t]
                for i in range(b):
                    off = (start + i) * 8
                    struct.pack_into("<HHI", bb, off, int(c0_cpu[i, t]), int(c1_cpu[i, t]), int(packed_idx[i, t]))

            if previews is not None:
                e0_cpu = e0_q.to(torch.float32)
                e1_cpu = e1_q.to(torch.float32)
                pal_bc1 = torch.stack([e0_cpu, e1_cpu, (2*e0_cpu + e1_cpu)/3, (e0_cpu + 2*e1_cpu)/3], dim=2)
                pal_cpu = pal_bc1.to("cpu")
                idx_cpu = idx_bc1.to("cpu").to(torch.int64)
                dec = torch.gather(
                    pal_cpu.unsqueeze(2).expand(-1, -1, 16, -1, -1), 3,
                    idx_cpu.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 3),
                ).squeeze(3)
                dec_u8 = (dec.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).numpy()
                x_np, y_np = x.numpy().reshape(-1), y.numpy().reshape(-1)
                for t in range(T):
                    previews[t][y_np, x_np] = dec_u8[:, t, :, :].reshape(-1, 3)

            if (start // self.block_batch) % 10 == 0:
                log(f"[Infer] Blocks {start}..{end-1} / {N}")

        out_png_written = []
        for t in range(T):
            _write_dds_dxt1(out_dds_paths[t], W, H, bytes(bc1_bytes[t]))
            log(f"[Done] DDS: {out_dds_paths[t]}")
            if previews and out_png_paths[t]:
                Image.fromarray(previews[t], mode="RGB").save(out_png_paths[t])
                out_png_written.append(out_png_paths[t])
                log(f"[Done] PNG: {out_png_paths[t]}")

        return InferenceResult(width=W, height=H, out_dds_paths=out_dds_paths, out_png_paths=out_png_written)
