"""
NTBC Trainer — trains both the Endpoint and Color networks.

Usage:
    trainer = Trainer(
        dataset_json="path/to/Train_dataset.json",
        source_images=["path/to/albedo.png", "path/to/normal.png"],
        output_dir="path/to/model_output",
    )
    trainer.run(callback=lambda msg: print(msg))
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

import torch

# Add parent dir to path so model/* imports work regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from endpoint_nn import EndpointNetwork, endpoint_loss_bc1_multi
from color_nn import ColorNetwork, color_loss_bc1_multi
from model_compress import compress_state_dict, decompress_state_dict, print_size_comparison


# ---------- Utilities ----------

def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> str:
    """Return 'cpu' if CUDA is requested but not available, with a warning."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA requested but not available — falling back to CPU.")
        return "cpu"
    return requested


def _lr_scale_warmup_cos(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 1:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    t = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def _set_lrs(optimizer, lr_scale, lr_grid, lr_mlp, grid_lr_mul=1.0):
    optimizer.param_groups[0]["lr"] = (lr_grid * grid_lr_mul) * lr_scale
    optimizer.param_groups[1]["lr"] = lr_mlp * lr_scale


def _load_image_rgb_u8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _pad_image_to_blocks(img: np.ndarray, blocks_x: int, blocks_y: int) -> np.ndarray:
    H, W, _ = img.shape
    pad_h = max(0, blocks_y * 4 - H)
    pad_w = max(0, blocks_x * 4 - W)
    if pad_h == 0 and pad_w == 0:
        return img
    return np.pad(img, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _load_images_stack_u8(paths: List[Path], blocks_x: int, blocks_y: int) -> torch.Tensor:
    imgs = []
    for p in paths:
        arr = _load_image_rgb_u8(p)
        arr = _pad_image_to_blocks(arr, blocks_x, blocks_y)
        imgs.append(torch.from_numpy(arr))
    return torch.stack(imgs, dim=0)  # (T,H,W,3) uint8 CPU


_OFF_X = torch.tensor([0, 1, 2, 3] * 4, dtype=torch.long)
_OFF_Y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.long)


def _extract_block_colors_batch_u8_multi(imgs_thwc_u8: torch.Tensor, bxby_batch: torch.Tensor) -> torch.Tensor:
    """imgs_thwc_u8: (T,H,W,3) uint8; bxby_batch: (B,2) -> (B,T,16,3) uint8"""
    dev = bxby_batch.device
    bx, by = bxby_batch[:, 0], bxby_batch[:, 1]
    x = (bx * 4)[:, None] + _OFF_X.to(dev)[None, :]  # (B,16)
    y = (by * 4)[:, None] + _OFF_Y.to(dev)[None, :]
    return imgs_thwc_u8[:, y, x].permute(1, 0, 2, 3).contiguous()  # (B,T,16,3)


def _save_checkpoint(out_dir: Path, name: str, net, optimizer, step: int, meta: dict = None):
    ckpt = {"global_step": step, "model_state": net.state_dict(), "optimizer_state": optimizer.state_dict()}
    if meta:
        ckpt["meta"] = meta
    torch.save(ckpt, out_dir / name)


def _merge_compressed_state_dicts(endpoint_sd, color_sd):
    merged = {}
    for k, v in endpoint_sd.items():
        merged["endpoint." + k] = v
    for k, v in color_sd.items():
        merged["color." + k] = v
    return merged


# ---------- Trainer ----------

class Trainer:
    """
    Trains Endpoint and Color networks from a `Train_dataset.json`.

    Args:
        dataset_json:   Path to Train_dataset.json produced by EndpointExtractor.
        source_images:  List of source PNG image paths (one per texture in the dataset).
        output_dir:     Directory where checkpoints and the merged .pt will be saved.
        main_steps:     Number of main training steps (default 20000).
        qat_tail_frac:  Fraction of main_steps for QAT tail (default 0.1 = 10%).
        batch_blocks:   Batch size for endpoint training (default 4096).
        batch_texels:   Batch size for color training (default 131072).
        device:         "cuda" or "cpu".
        seed:           Random seed.
    """

    DEFAULT_CFG = {
        "main_steps": 20_000,
        "qat_tail_fraction": 0.10,
        "warmup_steps": 10,
        "qat_warmup_steps": 10,
        "lr_grid": 1e-2,
        "lr_mlp": 5e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-15,
        "temperature": 0.01,
        "qat_bits": 8,
        "freeze_grids_during_qat": True,
        "batch_size_blocks": 4096,
        "batch_size_texels": 131072,
        "use_amp": True,
        "param_dtype": "float32",
        "log_every_steps": 50,
        "save_every_steps": 5000,
        "seed": 0,
    }

    def __init__(
        self,
        dataset_json: str,
        source_images: List[str],
        output_dir: str,
        **kwargs,
    ):
        self.dataset_json = Path(dataset_json).expanduser().resolve()
        self.source_images = [Path(p).expanduser().resolve() for p in source_images]
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = {**self.DEFAULT_CFG, **kwargs}
        raw_device = self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.cfg["device"] = _resolve_device(str(raw_device))

    def _load_dataset(self):
        d = json.loads(self.dataset_json.read_text())
        bxby = np.asarray(d["inputs"]["bxby"], dtype=np.int64)
        ep = np.asarray(d["targets"]["ep_q01"], dtype=np.float32)
        meta = d.get("meta", {})
        return bxby, ep, meta

    def _infer_num_textures(self, ep_np, meta):
        if "num_textures" in meta:
            return int(meta["num_textures"])
        if ep_np.ndim == 2 and ep_np.shape[1] % 6 == 0:
            return int(ep_np.shape[1] // 6)
        raise ValueError(f"Cannot infer num_textures from ep_q01 shape {ep_np.shape}")

    def run(self, callback: Optional[Callable[[str], None]] = None) -> Path:
        """Run full training. Returns path to the merged .pt checkpoint."""
        def log(msg: str):
            if callback:
                callback(msg)
            else:
                print(msg)

        cfg = self.cfg
        _set_seed(int(cfg["seed"]))
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        device = _resolve_device(str(cfg["device"]))
        use_amp = bool(cfg["use_amp"]) and device.startswith("cuda")
        param_dtype = torch.float16 if cfg["param_dtype"] == "float16" else torch.float32
        autocast_device = "cuda" if device.startswith("cuda") else "cpu"

        if not self.dataset_json.exists():
            raise FileNotFoundError(f"Dataset JSON not found: {self.dataset_json}")

        bxby_np, ep_np, meta = self._load_dataset()
        N = int(bxby_np.shape[0])
        T = self._infer_num_textures(ep_np, meta)

        blocks_x = int(meta.get("blocks_x", int(bxby_np[:, 0].max() + 1)))
        blocks_y = int(meta.get("blocks_y", int(bxby_np[:, 1].max() + 1)))

        log(f"Dataset: N={N} blocks, ({blocks_x}x{blocks_y}), T={T} textures")

        # st coords for endpoint network
        st_np = np.zeros((N, 2), dtype=np.float32)
        st_np[:, 0] = bxby_np[:, 0] / max(blocks_x - 1, 1)
        st_np[:, 1] = bxby_np[:, 1] / max(blocks_y - 1, 1)

        st_t = torch.from_numpy(st_np).to(device)
        bxby_t = torch.from_numpy(bxby_np).to(device)
        ep_t = torch.from_numpy(ep_np).to(device)

        src_paths = self.source_images
        if len(src_paths) != T:
            raise ValueError(f"source_images length ({len(src_paths)}) != dataset num_textures ({T})")
        for p in src_paths:
            if not p.exists():
                raise FileNotFoundError(f"Source image not found: {p}")

        imgs_thwc_u8 = _load_images_stack_u8(src_paths, blocks_x, blocks_y).to(device)
        H_img, W_img = int(imgs_thwc_u8.shape[1]), int(imgs_thwc_u8.shape[2])
        log(f"Loaded {len(src_paths)} images on {device} ({W_img}x{H_img})")

        # ---- Endpoint Training ----
        log("\n======== TRAIN ENDPOINT NET ========")
        ep_sd = self._train_endpoint(
            cfg, device, use_amp, param_dtype, autocast_device,
            N, blocks_x, blocks_y, T, st_t, bxby_t, ep_t, imgs_thwc_u8, meta, log,
        )

        # ---- Color Training ----
        log("\n======== TRAIN COLOR NET ========")
        col_sd = self._train_color(
            cfg, device, use_amp, param_dtype, autocast_device,
            N, blocks_x, blocks_y, T, bxby_t, ep_t, imgs_thwc_u8, W_img, H_img, meta, log,
        )

        # ---- Merge & Save ----
        merged = _merge_compressed_state_dicts(ep_sd, col_sd)
        merged_path = self.output_dir / "ntbc_bc1_merged_compressed.pt"
        torch.save(merged, merged_path)
        total_bytes = sum(t.numel() * t.element_size() for t in merged.values())
        log(f"\n[MERGED] Saved: {merged_path}")
        log(f"[MERGED] Total size: {total_bytes / 1024 / 1024:.2f} MB")

        # Clean up run dirs
        import shutil
        for d_name in ("runs_endpoint", "runs_color"):
            d = self.output_dir / d_name
            if d.exists():
                shutil.rmtree(d)

        return merged_path

    def _train_endpoint(self, cfg, device, use_amp, param_dtype, autocast_device,
                        N, blocks_x, blocks_y, T, st_t, bxby_t, ep_t, imgs_thwc_u8, meta, log):
        out_dir = self.output_dir / "runs_endpoint"
        out_dir.mkdir(parents=True, exist_ok=True)

        net = EndpointNetwork(num_textures=T, param_dtype=param_dtype).to(device)
        net.train()
        optimizer = torch.optim.Adam(
            [{"params": net.encoding.parameters(), "lr": cfg["lr_grid"]},
             {"params": net.mlp.parameters(), "lr": cfg["lr_mlp"]}],
            betas=tuple(cfg["betas"]), eps=float(cfg["eps"]),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        main_steps = int(cfg["main_steps"])
        qat_tail = int(round(main_steps * float(cfg["qat_tail_fraction"])))
        total_steps = main_steps + qat_tail
        batch_size = int(cfg["batch_size_blocks"])
        temperature = float(cfg["temperature"])
        log_every = int(cfg["log_every_steps"])
        save_every = int(cfg["save_every_steps"])
        freeze_grids = bool(cfg["freeze_grids_during_qat"])
        qat_warmup = int(cfg.get("qat_warmup_steps", 10))
        in_qat, grid_lr_mul = False, 1.0

        run_total = run_le = run_lcd = 0.0
        run_count = 0
        t0 = t_start = time.time()

        for step in range(total_steps):
            if step == main_steps and qat_tail > 0:
                log("[Endpoint] >>> Enabling QAT <<<")
                net.encoding.enable_qat(bits=int(cfg["qat_bits"]))
                in_qat = True
                grid_lr_mul = 0.0 if freeze_grids else 1.0
                _save_checkpoint(out_dir, f"endpoint_bc1_step{step:06d}_qat_start.pt", net, optimizer, step, meta)

            didx = torch.randint(0, N, (batch_size,), device=device, dtype=torch.long)
            st = st_t[didx]; bxby = bxby_t[didx]; ref_ep = ep_t[didx]
            ref_cols_u8 = _extract_block_colors_batch_u8_multi(imgs_thwc_u8, bxby)
            ref_cols = ref_cols_u8.to(torch.float32) / 255.0

            if in_qat:
                lr_scale = _lr_scale_warmup_cos(step - main_steps, qat_tail, qat_warmup)
            else:
                lr_scale = _lr_scale_warmup_cos(step, main_steps, int(cfg["warmup_steps"]))
            _set_lrs(optimizer, lr_scale, cfg["lr_grid"], cfg["lr_mlp"], grid_lr_mul)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                pred_ep = net(st)
                loss_out = endpoint_loss_bc1_multi(pred_ep, ref_ep, ref_cols, temperature)
                loss = loss_out.total

            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            run_total += float(loss_out.total.detach()); run_le += float(loss_out.le.detach())
            run_lcd += float(loss_out.lcd.detach()); run_count += 1

            if (step + 1) % log_every == 0:
                msg = (f"[Endpoint {step+1:06d}/{total_steps}] "
                       f"loss={run_total/run_count:.6f} Le={run_le/run_count:.6f} Lcd={run_lcd/run_count:.6f} "
                       f"lr_grid={optimizer.param_groups[0]['lr']:.5g} time={time.time()-t0:.1f}s")
                log(msg)
                run_total = run_le = run_lcd = run_count = 0
                t0 = time.time()

            if save_every > 0 and (step + 1) % save_every == 0:
                _save_checkpoint(out_dir, f"endpoint_bc1_step{step+1:06d}.pt", net, optimizer, step+1, meta)

        _save_checkpoint(out_dir, "endpoint_bc1_final.pt", net, optimizer, total_steps, meta)
        elapsed = time.time() - t_start
        log(f"[Endpoint] Done in {int(elapsed//60)}m {elapsed%60:.1f}s")

        compressed = compress_state_dict(net.state_dict())
        return compressed

    def _train_color(self, cfg, device, use_amp, param_dtype, autocast_device,
                     N, blocks_x, blocks_y, T, bxby_t, ep_t, imgs_thwc_u8, W_img, H_img, meta, log):
        out_dir = self.output_dir / "runs_color"
        out_dir.mkdir(parents=True, exist_ok=True)

        net = ColorNetwork(num_textures=T, param_dtype=param_dtype, finest_resolution=2048).to(device)
        net.train()
        optimizer = torch.optim.Adam(
            [{"params": net.encoding.parameters(), "lr": cfg["lr_grid"]},
             {"params": net.mlp.parameters(), "lr": cfg["lr_mlp"]}],
            betas=tuple(cfg["betas"]), eps=float(cfg["eps"]),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        main_steps = int(cfg["main_steps"])
        qat_tail = int(round(main_steps * float(cfg["qat_tail_fraction"])))
        total_steps = main_steps + qat_tail
        batch_size = int(cfg["batch_size_texels"])
        temperature = float(cfg["temperature"])
        log_every = int(cfg["log_every_steps"])
        save_every = int(cfg["save_every_steps"])
        freeze_grids = bool(cfg["freeze_grids_during_qat"])
        qat_warmup = int(cfg.get("qat_warmup_steps", 10))
        in_qat, grid_lr_mul = False, 1.0

        run_total = run_lc = run_lcd = 0.0
        run_count = 0
        t0 = t_start = time.time()

        for step in range(total_steps):
            if step == main_steps and qat_tail > 0:
                log("[Color] >>> Enabling QAT <<<")
                net.encoding.enable_qat(bits=int(cfg["qat_bits"]))
                in_qat = True
                grid_lr_mul = 0.0 if freeze_grids else 1.0
                _save_checkpoint(out_dir, f"color_bc1_step{step:06d}_qat_start.pt", net, optimizer, step, meta)

            didx = torch.randint(0, N, (batch_size,), device=device, dtype=torch.long)
            bxby = bxby_t[didx]; ref_ep = ep_t[didx]
            ox = torch.randint(0, 4, (batch_size,), device=device, dtype=torch.long)
            oy = torch.randint(0, 4, (batch_size,), device=device, dtype=torch.long)
            px = (bxby[:, 0] * 4 + ox).clamp(0, W_img - 1)
            py = (bxby[:, 1] * 4 + oy).clamp(0, H_img - 1)
            ref_rgb = imgs_thwc_u8[:, py, px].permute(1, 0, 2).to(torch.float32) / 255.0  # (B,T,3)
            u = px.to(torch.float32) / float(max(1, W_img - 1))
            v = py.to(torch.float32) / float(max(1, H_img - 1))
            uv = torch.stack([u, v], dim=1)

            if in_qat:
                lr_scale = _lr_scale_warmup_cos(step - main_steps, qat_tail, qat_warmup)
            else:
                lr_scale = _lr_scale_warmup_cos(step, main_steps, int(cfg["warmup_steps"]))
            _set_lrs(optimizer, lr_scale, cfg["lr_grid"], cfg["lr_mlp"], grid_lr_mul)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                pred = net(uv)
                loss_out = color_loss_bc1_multi(pred, ref_rgb, ref_ep, temperature)
                loss = loss_out.total

            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()

            run_total += float(loss_out.total.detach()); run_lc += float(loss_out.lc.detach())
            run_lcd += float(loss_out.lcd.detach()); run_count += 1

            if (step + 1) % log_every == 0:
                msg = (f"[Color {step+1:06d}/{total_steps}] "
                       f"loss={run_total/run_count:.6f} Lc={run_lc/run_count:.6f} Lcd={run_lcd/run_count:.6f} "
                       f"lr_grid={optimizer.param_groups[0]['lr']:.5g} time={time.time()-t0:.1f}s")
                log(msg)
                run_total = run_lc = run_lcd = run_count = 0
                t0 = time.time()

            if save_every > 0 and (step + 1) % save_every == 0:
                _save_checkpoint(out_dir, f"color_bc1_step{step+1:06d}.pt", net, optimizer, step+1, meta)

        _save_checkpoint(out_dir, "color_bc1_final.pt", net, optimizer, total_steps, meta)
        elapsed = time.time() - t_start
        log(f"[Color] Done in {int(elapsed//60)}m {elapsed%60:.1f}s")

        compressed = compress_state_dict(net.state_dict())
        return compressed
