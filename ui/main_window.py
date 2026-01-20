import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import tkinter as tk
from ui.converter import ConverterFrame

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NTBC")
        self.geometry("300x500")
        
        # Container for screens
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        for F in (MainMenu, ConverterFrame):
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
        
        convert_btn = tk.Button(self, text="Converter",
                                command=lambda: controller.show_frame("ConverterFrame"),
                                font=("Arial", 14), padx=20, pady=10)
        convert_btn.pack(pady=20)

if __name__ == "__main__":
    app = App()
    app.mainloop()
