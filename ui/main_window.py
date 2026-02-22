import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tkinter as tk
from ui.bcn_converter_window import ConverterFrame
from ui.extract_endpoints_window import ExtractEndpointsFrame
from ui.train_window import TrainFrame
from ui.end_to_end_window import EndToEndFrame


_TITLE_FONT  = ("Segoe UI", 22, "bold")
_SUB_FONT    = ("Segoe UI", 10)
_BTN_FONT    = ("Segoe UI", 12)


# ---------- Styled button helper ----------

def _menu_btn(parent, text, description, command):
    outer = tk.Frame(parent, relief="flat", bd=0)
    outer.pack(fill="x", padx=60, pady=6)
    btn = tk.Button(outer, text=text, font=_BTN_FONT,
                    relief="groove", cursor="hand2",
                    anchor="w", padx=18, pady=10,
                    command=command)
    btn.pack(fill="x")
    tk.Label(outer, text=description, font=_SUB_FONT, fg="#888", anchor="w").pack(fill="x", padx=4)
    return btn


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NTBC — Neural Texture Block Compression")
        self.geometry("660x540")
        self.resizable(True, True)

        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MainMenu, ConverterFrame, ExtractEndpointsFrame, TrainFrame, EndToEndFrame):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainMenu")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class MainMenu(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self._build()

    def _build(self):
        # Header
        header = tk.Frame(self, pady=28)
        header.pack(fill="x")
        tk.Label(header, text="NTBC", font=_TITLE_FONT).pack()
        tk.Label(header, text="Neural Texture Block Compression", font=_SUB_FONT, fg="#666").pack()
        tk.Frame(self, height=1, bg="#ddd").pack(fill="x", padx=40, pady=(0, 10))

        c = self.controller

        _menu_btn(self, "🔄  BCn Converter",
                  "Convert images to BC1–BC7 compressed DDS format.",
                  lambda: c.show_frame("ConverterFrame"))

        _menu_btn(self, "📦  Extract BC1 Endpoints",
                  "Parse a folder of BC1 DDS files to generate a training dataset.",
                  lambda: c.show_frame("ExtractEndpointsFrame"))

        _menu_btn(self, "🧠  Train Model",
                  "Train the Endpoint + Color networks from a dataset JSON.",
                  lambda: c.show_frame("TrainFrame"))

        _menu_btn(self, "⚡  End-to-End Pipeline",
                  "Extract → Train → Infer → Evaluate PSNR in one click.",
                  lambda: c.show_frame("EndToEndFrame"))

        # Footer
        tk.Frame(self, height=1, bg="#ddd").pack(fill="x", padx=40, pady=(16, 4))
        tk.Label(self, text="arXiv:2407.09543", font=_SUB_FONT, fg="#aaa").pack(pady=4)


if __name__ == "__main__":
    app = App()
    app.mainloop()
