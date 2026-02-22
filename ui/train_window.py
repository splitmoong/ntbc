"""
Training Window — spawns a background training thread and streams logs live.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from src.dotenv_config import load_env, save_env


_TITLE_FONT = ("Segoe UI", 14, "bold")
_LABEL_FONT = ("Segoe UI", 10)
_BTN_FONT = ("Segoe UI", 11, "bold")
_MONO_FONT = ("Consolas", 9)


def _add_row(parent, label_text: str, var: tk.StringVar, browse_cmd):
    row = tk.Frame(parent)
    row.pack(fill="x", padx=18, pady=4)
    tk.Label(row, text=label_text, font=_LABEL_FONT, anchor="w", width=22).pack(side="left")
    tk.Entry(row, textvariable=var, font=_LABEL_FONT).pack(side="left", fill="x", expand=True)
    tk.Button(row, text="Browse", font=_LABEL_FONT, command=browse_cmd).pack(side="left", padx=(6, 0))
    return row


class TrainFrame(tk.Frame):
    """UI window for running Endpoint + Color network training."""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._thread: threading.Thread | None = None
        self._running = False

        self.dataset_json = tk.StringVar()
        self.source_folder = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.var_steps = tk.StringVar(value="20000")
        self.var_batch = tk.StringVar(value="4096")
        self.var_device = tk.StringVar(value="cuda")

        self._load_dotenv()
        self._build()

    # ---------- .env persistence ----------

    def _load_dotenv(self):
        cfg = load_env()
        if cfg.get("NTBC_OUTPUT_DIR"):  self.output_dir.set(cfg["NTBC_OUTPUT_DIR"])
        if cfg.get("NTBC_STEPS"):       self.var_steps.set(cfg["NTBC_STEPS"])
        if cfg.get("NTBC_BATCH"):       self.var_batch.set(cfg["NTBC_BATCH"])
        if cfg.get("NTBC_DEVICE"):      self.var_device.set(cfg["NTBC_DEVICE"])

    def _save_dotenv(self):
        save_env({
            "NTBC_OUTPUT_DIR": self.output_dir.get().strip(),
            "NTBC_STEPS":      self.var_steps.get().strip(),
            "NTBC_BATCH":      self.var_batch.get().strip(),
            "NTBC_DEVICE":     self.var_device.get().strip(),
        })

    def _build(self):
        # Back button
        tk.Button(self, text="← Back", font=_LABEL_FONT,
                  command=lambda: self.controller.show_frame("MainMenu")).pack(anchor="nw", padx=10, pady=8)
        tk.Label(self, text="Train NTBC Model", font=_TITLE_FONT).pack(pady=(0, 10))

        form = tk.Frame(self)
        form.pack(fill="x", padx=4)

        _add_row(form, "Dataset JSON:", self.dataset_json, self._browse_json)
        _add_row(form, "Source images folder:", self.source_folder, self._browse_src)
        _add_row(form, "Output directory:", self.output_dir, self._browse_out)

        # Hyperparameter row
        hp = tk.Frame(self)
        hp.pack(fill="x", padx=18, pady=6)
        for label, var, w in [("Steps:", self.var_steps, 8),
                               ("Block batch:", self.var_batch, 8),
                               ("Device:", self.var_device, 8)]:
            tk.Label(hp, text=label, font=_LABEL_FONT).pack(side="left")
            tk.Entry(hp, textvariable=var, font=_LABEL_FONT, width=w).pack(side="left", padx=(2, 14))

        # Buttons
        btn_row = tk.Frame(self)
        btn_row.pack(pady=8)
        self.run_btn = tk.Button(btn_row, text="▶  Train", font=_BTN_FONT,
                                  bg="#2ecc71", fg="white", padx=14, pady=6,
                                  command=self._start_training)
        self.run_btn.pack(side="left", padx=6)
        self.stop_btn = tk.Button(btn_row, text="■  Stop", font=_BTN_FONT,
                                   bg="#e74c3c", fg="white", padx=14, pady=6,
                                   state="disabled", command=self._request_stop)
        self.stop_btn.pack(side="left", padx=6)

        # Log
        tk.Label(self, text="Training Log:", font=_LABEL_FONT, anchor="w").pack(fill="x", padx=18)
        self.log_box = scrolledtext.ScrolledText(self, font=_MONO_FONT, state="disabled",
                                                  bg="#1e1e1e", fg="#d4d4d4",
                                                  height=12, relief="flat")
        self.log_box.pack(fill="both", expand=True, padx=18, pady=(2, 14))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var, font=_LABEL_FONT,
                 anchor="w", fg="#555").pack(fill="x", padx=18, pady=(0, 8))

    def _browse_json(self):
        p = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if p:
            self.dataset_json.set(p)

    def _browse_src(self):
        p = filedialog.askdirectory()
        if p:
            self.source_folder.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory()
        if p:
            self.output_dir.set(p)

    def _log(self, msg: str):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _set_ready(self):
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._running = False

    def _request_stop(self):
        self._running = False
        self.status_var.set("Stopping after current step…")

    def _start_training(self):
        dataset = self.dataset_json.get().strip()
        src_folder = self.source_folder.get().strip()
        out_dir = self.output_dir.get().strip()

        if not dataset or not src_folder or not out_dir:
            messagebox.showwarning("Missing inputs",
                                   "Please fill in Dataset JSON, Source images folder, and Output directory.")
            return

        # Collect source PNG file paths from folder
        src_path = Path(src_folder)
        src_images = sorted(str(p) for p in src_path.glob("*.png"))
        if not src_images:
            src_images = sorted(str(p) for p in src_path.glob("*.jpg"))
        if not src_images:
            messagebox.showerror("No images", f"No PNG/JPG images found in:\n{src_folder}")
            return

        try:
            steps = int(self.var_steps.get())
            batch = int(self.var_batch.get())
        except ValueError:
            messagebox.showerror("Invalid", "Steps and Batch must be integers.")
            return

        device = self.var_device.get().strip() or "cuda"

        self._save_dotenv()
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")
        self.status_var.set("Training…")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._running = True

        def worker():
            try:
                from model.train import Trainer
                trainer = Trainer(
                    dataset_json=dataset,
                    source_images=src_images,
                    output_dir=out_dir,
                    main_steps=steps,
                    batch_size_blocks=batch,
                    device=device,
                )
                result_path = trainer.run(callback=lambda msg: self.after(0, self._log, msg))
                self.after(0, self._log, f"\n✅ Done! Saved: {result_path}")
                self.after(0, self.status_var.set, "✅ Training complete")
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, self._log, f"\n❌ Error:\n{tb}")
                self.after(0, self.status_var.set, "❌ Error")
            finally:
                self.after(0, self._set_ready)

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()
