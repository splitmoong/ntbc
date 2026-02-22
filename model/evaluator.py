"""
NTBC Evaluator — computes PSNR (and optionally SSIM) between source PNGs
and BC1-compressed DDS files (reference + NTBC output).

Usage:
    evaluator = NTBCEvaluator(
        source_images=["albedo.png", "normal.png"],
        ref_dds_list=["albedo_ref.dds", "normal_ref.dds"],
        test_dds_list=["ntbc_out_albedo.dds", "ntbc_out_normal.dds"],
        compute_ssim=True,
    )
    results = evaluator.evaluate(callback=print)
    # results -> list of dicts with psnr_ref, psnr_ntbc, psnr_delta, etc.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image


# ---------- Image I/O ----------

def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _resize_max_side(arr: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if max_side is None:
        return arr
    h, w = arr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return arr
    scale = max_side / float(m)
    img = Image.fromarray(arr, mode="RGB")
    img = img.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                     resample=Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _pad_to_shape_edge(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == th and w == tw:
        return arr
    pad_h, pad_w = max(0, th - h), max(0, tw - w)
    if pad_h == 0 and pad_w == 0:
        return arr[:th, :tw, :]
    return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


# ---------- DDS BC1 Decoder ----------

def _rgb565_to_rgb888(c_u16: np.ndarray) -> np.ndarray:
    c = c_u16.astype(np.uint16)
    r = ((c >> 11) & 31).astype(np.float32) * (255.0 / 31.0)
    g = ((c >> 5) & 63).astype(np.float32) * (255.0 / 63.0)
    b = (c & 31).astype(np.float32) * (255.0 / 31.0)
    return np.clip(np.rint(np.stack([r, g, b], axis=-1)), 0, 255).astype(np.uint8)


def decode_dds_bc1(path: str) -> np.ndarray:
    """Fully vectorized BC1/DXT1 DDS decoder. Returns (H, W, 3) uint8."""
    p = Path(path)
    data = p.read_bytes()
    if len(data) < 128 or data[:4] != b"DDS ":
        raise ValueError(f"{p}: not a valid DDS file")
    header = data[4:4 + 124]
    height = struct.unpack_from("<I", header, 8)[0]
    width = struct.unpack_from("<I", header, 12)[0]
    fourcc = header[80:84]
    offset = 128
    if fourcc == b"DX10":
        offset += 20
    elif fourcc != b"DXT1":
        raise ValueError(f"{p}: FourCC {fourcc!r} not supported (need DXT1/BC1)")
    bw = (width + 3) // 4
    bh = (height + 3) // 4
    N = bw * bh
    buf = np.frombuffer(data, dtype=np.uint8, offset=offset, count=N * 8).reshape(N, 8)
    c0 = buf[:, 0].astype(np.uint16) | (buf[:, 1].astype(np.uint16) << 8)
    c1 = buf[:, 2].astype(np.uint16) | (buf[:, 3].astype(np.uint16) << 8)
    idx_raw = (buf[:, 4].astype(np.uint32) | (buf[:, 5].astype(np.uint32) << 8)
               | (buf[:, 6].astype(np.uint32) << 16) | (buf[:, 7].astype(np.uint32) << 24))
    rgb0 = _rgb565_to_rgb888(c0).astype(np.float32)
    rgb1 = _rgb565_to_rgb888(c1).astype(np.float32)
    mode4 = (c0 > c1)[:, None]
    pal = np.empty((N, 4, 3), dtype=np.float32)
    pal[:, 0] = rgb0; pal[:, 1] = rgb1
    pal[:, 2] = np.where(mode4, (2*rgb0 + rgb1) / 3, (rgb0 + rgb1) / 2)
    pal[:, 3] = np.where(mode4, (rgb0 + 2*rgb1) / 3, 0)
    pal = np.clip(np.rint(pal), 0, 255).astype(np.uint8)
    shifts = (2 * np.arange(16, dtype=np.uint32))[None, :]
    tex_idx = ((idx_raw[:, None] >> shifts) & 3).astype(np.intp)
    colors = pal[np.arange(N)[:, None], tex_idx]
    out = colors.reshape(bh, bw, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)
    return out[:height, :width, :]


# ---------- Metrics ----------

def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def _ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        return float(ssim_fn(a, b, data_range=255, channel_axis=2))
    except ImportError:
        raise ImportError("Install scikit-image to enable SSIM: pip install scikit-image")


# ---------- NTBCEvaluator class ----------

class NTBCEvaluator:
    """
    Evaluates NTBC output quality vs. reference BC1 compression.

    Args:
        source_images:  List of paths to original PNG source images.
        ref_dds_list:   List of paths to reference BC1 DDS files (e.g., from Compressonator).
        test_dds_list:  List of paths to NTBC-generated DDS files.
        compute_ssim:   Whether to also compute SSIM (requires scikit-image).
        eval_max_side:  Max resolution side for PSNR evaluation (None = no resize).
    """

    def __init__(
        self,
        source_images: List[str],
        ref_dds_list: List[str],
        test_dds_list: List[str],
        compute_ssim: bool = False,
        eval_max_side: Optional[int] = None,
    ):
        if not (len(source_images) == len(ref_dds_list) == len(test_dds_list)):
            raise ValueError("source_images, ref_dds_list, and test_dds_list must all have the same length.")
        self.source_images = source_images
        self.ref_dds_list = ref_dds_list
        self.test_dds_list = test_dds_list
        self.compute_ssim = compute_ssim
        self.eval_max_side = eval_max_side

    def evaluate(self, callback: Optional[Callable[[str], None]] = None) -> List[dict]:
        """
        Runs evaluation for all textures.

        Returns:
            List of dicts with keys:
                name, psnr_ref, psnr_ntbc, psnr_delta
                (and optionally ssim_ref, ssim_ntbc, ssim_delta)
        """
        def log(msg):
            if callback:
                callback(msg)
            else:
                print(msg)

        results = []
        names = [Path(s).stem for s in self.source_images]

        for name, src_p, ref_p, test_p in zip(names, self.source_images, self.ref_dds_list, self.test_dds_list):
            if not (src_p and ref_p and test_p):
                log(f"[SKIP] {name}: missing path(s)")
                continue

            for p in [ref_p, test_p]:
                if not Path(p).exists():
                    log(f"[SKIP] {name}: file not found: {p}")
                    continue

            src = _load_rgb(src_p)
            ref = decode_dds_bc1(ref_p)
            test = decode_dds_bc1(test_p)

            th, tw = ref.shape[0], ref.shape[1]
            src_padded = _pad_to_shape_edge(src, th, tw)
            test = _pad_to_shape_edge(test, th, tw)

            src_eval = _resize_max_side(src_padded, self.eval_max_side)
            ref_eval = _resize_max_side(ref, self.eval_max_side)
            test_eval = _resize_max_side(test, self.eval_max_side)

            psnr_ref = _psnr(src_eval, ref_eval)
            psnr_test = _psnr(src_eval, test_eval)

            row = {
                "name": name,
                "psnr_ref": psnr_ref,
                "psnr_ntbc": psnr_test,
                "psnr_delta": psnr_test - psnr_ref,
            }

            if self.compute_ssim:
                try:
                    row["ssim_ref"] = _ssim_rgb(src_padded, ref)
                    row["ssim_ntbc"] = _ssim_rgb(src_padded, test)
                    row["ssim_delta"] = row["ssim_ntbc"] - row["ssim_ref"]
                except ImportError as e:
                    log(f"[WARN] SSIM skipped: {e}")

            results.append(row)

        # Print report
        log("\n======== NTBC EVAL RESULTS ========")
        for r in results:
            base = f"{r['name']:30s}  PSNR ref={r['psnr_ref']:.3f}  ntbc={r['psnr_ntbc']:.3f}  Δ={r['psnr_delta']:+.3f}"
            if "ssim_ref" in r:
                base += f" | SSIM ref={r['ssim_ref']:.4f} ntbc={r['ssim_ntbc']:.4f} Δ={r['ssim_delta']:+.4f}"
            log(base)

        if results:
            avg_delta = sum(r["psnr_delta"] for r in results) / len(results)
            log(f"\n[AVG] PSNR Δ: {avg_delta:+.3f} dB over {len(results)} texture(s)")

        return results
