"""
End-to-End Pipeline Window
Orchestrates: PNG input → Compress to DDS → Extract Endpoints → Train → Infer → Evaluate.
Paths are persisted to / loaded from .env in the project root.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from src.dotenv_config import load_env, save_env


_TITLE_FONT = ("Segoe UI", 14, "bold")
_LABEL_FONT = ("Segoe UI", 10)
_BTN_FONT   = ("Segoe UI", 11, "bold")
_MONO_FONT  = ("Consolas", 9)
_SEC_FONT   = ("Segoe UI", 10, "bold")


def _add_row(parent, label_text: str, var: tk.StringVar, browse_cmd):
    row = tk.Frame(parent)
    row.pack(fill="x", padx=18, pady=3)
    tk.Label(row, text=label_text, font=_LABEL_FONT, anchor="w", width=26).pack(side="left")
    tk.Entry(row, textvariable=var, font=_LABEL_FONT).pack(side="left", fill="x", expand=True)
    tk.Button(row, text="Browse", font=_LABEL_FONT, command=browse_cmd).pack(side="left", padx=(6, 0))
    return row


def _compress_png_to_dds(cli: str, src_png: Path, dst_dds: Path, log_fn) -> bool:
    """
    Call CompressonatorCLI to compress a single PNG to BC1 DDS.
    Returns True on success.
    """
    cmd = [cli, "-fd", "BC1", str(src_png), str(dst_dds)]
    log_fn(f"  compressing: {src_png.name} → {dst_dds.name}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            log_fn(f"  [WARN] Compressonator returned {r.returncode}: {r.stderr.strip()[:200]}")
            return False
        return True
    except FileNotFoundError:
        log_fn(f"  [ERROR] CompressonatorCLI not found: {cli}")
        return False
    except subprocess.TimeoutExpired:
        log_fn(f"  [ERROR] Compressonator timed out for {src_png.name}")
        return False


class EndToEndFrame(tk.Frame):
    """
    Full end-to-end pipeline:
    Phase 0: Compress PNGs → BC1 DDS (via CompressonatorCLI)
    Phase 1: Extract BC1 endpoints → Train_dataset.json
    Phase 2: Train Endpoint + Color networks
    Phase 3: Infer → NTBC DDS files
    Phase 4: Evaluate PSNR (+ optional SSIM)
    """

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._thread: threading.Thread | None = None
        self._running = False

        self.png_folder   = tk.StringVar()
        self.output_dir   = tk.StringVar()
        self.compressor   = tk.StringVar()
        self.var_steps    = tk.StringVar(value="20000")
        self.var_batch    = tk.StringVar(value="4096")
        self.var_device   = tk.StringVar(value="cuda")
        self.var_save_preview  = tk.BooleanVar(value=True)
        self.var_compute_ssim  = tk.BooleanVar(value=False)

        self._load_dotenv()
        self._build()

    # ---------- .env persistence ----------

    def _load_dotenv(self):
        cfg = load_env()
        if cfg.get("NTBC_PNG_FOLDER"):    self.png_folder.set(cfg["NTBC_PNG_FOLDER"])
        if cfg.get("NTBC_OUTPUT_DIR"):    self.output_dir.set(cfg["NTBC_OUTPUT_DIR"])
        if cfg.get("NTBC_STEPS"):         self.var_steps.set(cfg["NTBC_STEPS"])
        if cfg.get("NTBC_BATCH"):         self.var_batch.set(cfg["NTBC_BATCH"])
        if cfg.get("NTBC_DEVICE"):        self.var_device.set(cfg["NTBC_DEVICE"])
        if cfg.get("NTBC_COMPRESSONATOR_CLI"):
            self.compressor.set(cfg["NTBC_COMPRESSONATOR_CLI"])

    def _save_dotenv(self):
        save_env({
            "NTBC_PNG_FOLDER":          self.png_folder.get().strip(),
            "NTBC_OUTPUT_DIR":          self.output_dir.get().strip(),
            "NTBC_STEPS":               self.var_steps.get().strip(),
            "NTBC_BATCH":               self.var_batch.get().strip(),
            "NTBC_DEVICE":              self.var_device.get().strip(),
            "NTBC_COMPRESSONATOR_CLI":  self.compressor.get().strip(),
        })

    # ---------- UI ----------

    def _build(self):
        # Back
        tk.Button(self, text="← Back", font=_LABEL_FONT,
                  command=lambda: self.controller.show_frame("MainMenu")).pack(anchor="nw", padx=10, pady=8)
        tk.Label(self, text="End-to-End Pipeline", font=_TITLE_FONT).pack(pady=(0, 4))

        # Step overview
        steps_outer = tk.Frame(self)
        steps_outer.pack(fill="x")

        def _step_label(p, step_no, text):
            f = tk.Frame(p, relief="groove", bd=1)
            f.pack(fill="x", padx=18, pady=1)
            tk.Label(f, text=f"  PHASE {step_no}", font=_SEC_FONT, fg="#555", anchor="w").pack(side="left", padx=(6, 0))
            tk.Label(f, text=text, font=_LABEL_FONT, anchor="w").pack(side="left", padx=8)

        _step_label(steps_outer, "0", "Compress PNGs → BC1 DDS (CompressonatorCLI)")
        _step_label(steps_outer, "1", "Extract BC1 endpoints → dataset JSON")
        _step_label(steps_outer, "2", "Train Endpoint + Color networks")
        _step_label(steps_outer, "3", "Infer → NTBC DDS files")
        _step_label(steps_outer, "4", "Evaluate PSNR / SSIM")

        form = tk.Frame(self)
        form.pack(fill="x", pady=4)
        _add_row(form, "PNG source folder:", self.png_folder, self._browse_png)
        _add_row(form, "Output directory:", self.output_dir, self._browse_out)
        _add_row(form, "CompressonatorCLI path:", self.compressor, self._browse_cli)

        # Options
        opts = tk.Frame(self)
        opts.pack(fill="x", padx=18, pady=4)
        tk.Label(opts, text="Steps:", font=_LABEL_FONT).pack(side="left")
        tk.Entry(opts, textvariable=self.var_steps, font=_LABEL_FONT, width=8).pack(side="left", padx=(2, 10))
        tk.Label(opts, text="Batch:", font=_LABEL_FONT).pack(side="left")
        tk.Entry(opts, textvariable=self.var_batch, font=_LABEL_FONT, width=8).pack(side="left", padx=(2, 10))
        tk.Label(opts, text="Device:", font=_LABEL_FONT).pack(side="left")
        tk.Entry(opts, textvariable=self.var_device, font=_LABEL_FONT, width=6).pack(side="left", padx=(2, 10))
        tk.Checkbutton(opts, text="Preview PNGs", variable=self.var_save_preview, font=_LABEL_FONT).pack(side="left", padx=4)
        tk.Checkbutton(opts, text="SSIM", variable=self.var_compute_ssim, font=_LABEL_FONT).pack(side="left", padx=4)

        # Buttons
        btn_row = tk.Frame(self)
        btn_row.pack(pady=6)
        self.run_btn = tk.Button(btn_row, text="▶  Run Pipeline", font=_BTN_FONT,
                                  bg="#3498db", fg="white", padx=14, pady=6,
                                  command=self._start_pipeline)
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = tk.Button(btn_row, text="■  Stop", font=_BTN_FONT,
                                   bg="#e74c3c", fg="white", padx=14, pady=6,
                                   state="disabled", command=self._request_stop)
        self.stop_btn.pack(side="left", padx=6)

        # Phase indicator
        self.phase_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.phase_var, font=_SEC_FONT, fg="#2980b9", anchor="w").pack(
            fill="x", padx=18, pady=(2, 0))

        # Log
        tk.Label(self, text="Pipeline Log:", font=_LABEL_FONT, anchor="w").pack(fill="x", padx=18)
        self.log_box = scrolledtext.ScrolledText(self, font=_MONO_FONT, state="disabled",
                                                  bg="#1e1e1e", fg="#d4d4d4",
                                                  height=10, relief="flat")
        self.log_box.pack(fill="both", expand=True, padx=18, pady=(2, 8))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, font=_LABEL_FONT, anchor="w", fg="#555").pack(
            fill="x", padx=18, pady=(0, 6))

    # ---------- Browse helpers ----------

    def _browse_png(self):
        p = filedialog.askdirectory(title="Select PNG source folder")
        if p: self.png_folder.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Select output directory")
        if p: self.output_dir.set(p)

    def _browse_cli(self):
        p = filedialog.askopenfilename(title="Locate CompressonatorCLI",
                                        filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if p: self.compressor.set(p)

    # ---------- Log / state helpers ----------

    def _log(self, msg: str):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _phase(self, msg: str):
        self.after(0, self.phase_var.set, msg)
        self.after(0, self._log, f"\n▶ {msg}")

    def _set_ready(self):
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._running = False

    def _request_stop(self):
        self._running = False
        self.status_var.set("Stop requested…")

    # ---------- Pipeline ----------

    def _start_pipeline(self):
        png_folder = self.png_folder.get().strip()
        out_dir    = self.output_dir.get().strip()
        cli_path   = self.compressor.get().strip()

        if not png_folder or not out_dir:
            messagebox.showwarning("Missing inputs", "Please fill in PNG source folder and Output directory.")
            return
        if not cli_path:
            messagebox.showwarning("Missing CompressonatorCLI",
                                   "Please provide the path to compressonatorcli.exe.\n"
                                   "It is needed to compress PNGs to BC1 DDS format.")
            return

        png_path = Path(png_folder)
        png_files = sorted(png_path.glob("*.png"))
        if not png_files:
            messagebox.showerror("No PNGs", f"No .png files found in:\n{png_folder}")
            return

        try:
            steps = int(self.var_steps.get())
            batch = int(self.var_batch.get())
        except ValueError:
            messagebox.showerror("Invalid", "Steps and Batch must be integers.")
            return

        device       = self.var_device.get().strip() or "cuda"
        save_preview = bool(self.var_save_preview.get())
        compute_ssim = bool(self.var_compute_ssim.get())

        # Persist settings
        self._save_dotenv()

        # Clear UI
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
        self.phase_var.set("")
        self.status_var.set("Running pipeline…")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._running = True

        def worker():
            try:
                out_path = Path(out_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                dds_ref_dir = out_path / "dds_ref"
                dds_ref_dir.mkdir(parents=True, exist_ok=True)

                # ---- PHASE 0: Compress PNGs → DDS ----
                self.after(0, self._phase, "Phase 0/4 — Compressing PNGs → BC1 DDS…")
                dds_files = []
                all_ok = True
                for png_p in png_files:
                    if not self._running:
                        self.after(0, self._log, "Stopped by user.")
                        return
                    dst = dds_ref_dir / (png_p.stem + "_ref_bc1.dds")
                    ok = _compress_png_to_dds(cli_path, png_p, dst,
                                              lambda m: self.after(0, self._log, m))
                    if ok:
                        dds_files.append(dst)
                    else:
                        all_ok = False

                if not dds_files:
                    self.after(0, self._log,
                               "\n❌ No DDS files were created. "
                               "Check that CompressonatorCLI path is correct.")
                    self.after(0, self.status_var.set, "❌ Compression failed")
                    return
                if not all_ok:
                    self.after(0, self._log,
                               "[WARN] Some PNGs failed to compress — continuing with successful ones.")

                self.after(0, self._log, f"  ✓ {len(dds_files)} DDS file(s) written to {dds_ref_dir}")

                if not self._running:
                    self.after(0, self._log, "Stopped by user.")
                    return

                # ---- PHASE 1: Extract endpoints ----
                self.after(0, self._phase, "Phase 1/4 — Extracting BC1 endpoints…")
                from src.extract_endpoints import EndpointExtractor
                extractor = EndpointExtractor(str(dds_ref_dir), str(out_path))
                dataset_json_path = extractor.extract()
                inference_json_path = str(out_path / "Inference_input.json")
                self.after(0, self._log, f"  ✓ Dataset: {dataset_json_path}")

                if not self._running:
                    self.after(0, self._log, "Stopped by user.")
                    return

                # ---- PHASE 2: Train ----
                self.after(0, self._phase, "Phase 2/4 — Training endpoint + color networks…")
                from model.train import Trainer
                trainer = Trainer(
                    dataset_json=dataset_json_path,
                    source_images=[str(p) for p in png_files],
                    output_dir=str(out_path),
                    main_steps=steps,
                    batch_size_blocks=batch,
                    device=device,
                )
                merged_pt = trainer.run(callback=lambda msg: self.after(0, self._log, msg))

                if not self._running:
                    self.after(0, self._log, "Stopped by user.")
                    return

                # ---- PHASE 3: Infer ----
                self.after(0, self._phase, "Phase 3/4 — Generating NTBC DDS files…")
                from model.inference import NTBCInference
                infer = NTBCInference(
                    merged_ckpt=str(merged_pt),
                    coords_json=inference_json_path,
                    output_dir=str(out_path / "inference_output"),
                    device=device,
                    save_preview=save_preview,
                )
                infer_result = infer.run(callback=lambda msg: self.after(0, self._log, msg))

                if not self._running:
                    self.after(0, self._log, "Stopped by user.")
                    return

                # ---- PHASE 4: Evaluate ----
                self.after(0, self._phase, "Phase 4/4 — Evaluating PSNR…")
                from model.evaluator import NTBCEvaluator

                # Match PNG → reference DDS → NTBC DDS by index
                # dds_files and png_files are sorted in the same order
                evaluator = NTBCEvaluator(
                    source_images=[str(p) for p in png_files],
                    ref_dds_list=[str(p) for p in dds_files],
                    test_dds_list=[str(p) for p in infer_result.out_dds_paths],
                    compute_ssim=compute_ssim,
                )
                results = evaluator.evaluate(callback=lambda msg: self.after(0, self._log, msg))

                # ---- DONE ----
                avg_delta = sum(r["psnr_delta"] for r in results) / max(1, len(results))
                done_msg = (f"\n✅ Pipeline complete! "
                            f"Avg PSNR Δ = {avg_delta:+.3f} dB "
                            f"over {len(results)} texture(s)")
                self.after(0, self._log, done_msg)
                self.after(0, self.phase_var.set, "✅ Done")
                self.after(0, self.status_var.set, f"✅ Done — Avg PSNR Δ = {avg_delta:+.3f} dB")

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, self._log, f"\n❌ Pipeline Error:\n{tb}")
                self.after(0, self.status_var.set, "❌ Error — see log")
                self.after(0, self.phase_var.set, "❌ Error")
            finally:
                self.after(0, self._set_ready)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
