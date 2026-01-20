import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import sys

# Ensure parent directory is in path to import bcn
sys.path.append(str(Path(__file__).parent.parent))
from bcn import bcn

class ConverterFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Variables
        self.source_path = tk.StringVar()
        self.dest_folder = tk.StringVar()
        self.format_var = tk.StringVar(value="BC1")
        self.file_type_var = tk.StringVar(value="dds")
        self.quality_var = tk.StringVar(value="0.05")
        
        # UI Layout
        self.create_widgets()
        
    def create_widgets(self):
        # Back Button
        back_btn = tk.Button(self, text="< Back", command=lambda: self.controller.show_frame("MainMenu"))
        back_btn.pack(anchor="nw", padx=10, pady=10)
        
        # Title
        tk.Label(self, text="Converter", font=("Arial", 16)).pack(pady=10)
        
        # Source Selection
        source_frame = tk.LabelFrame(self, text="Source", padx=10, pady=10)
        source_frame.pack(fill="x", padx=10, pady=5)
        
        btn_frame = tk.Frame(source_frame)
        btn_frame.pack(fill="x")
        
        tk.Button(btn_frame, text="Select File", command=self.select_source_file).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Select Folder", command=self.select_source_folder).pack(side="left", padx=5)
        
        tk.Label(source_frame, textvariable=self.source_path).pack(fill="x", pady=5)
        
        # Destination Selection
        dest_frame = tk.LabelFrame(self, text="Destination", padx=10, pady=10)
        dest_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(dest_frame, text="Select Output Folder", command=self.select_dest_folder).pack(anchor="w")
        tk.Label(dest_frame, textvariable=self.dest_folder).pack(fill="x", pady=5)
        
        # Parameters
        param_frame = tk.LabelFrame(self, text="Parameters", padx=10, pady=10)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Format (BC1/BC4)
        tk.Label(param_frame, text="Format:").grid(row=0, column=0, sticky="w", padx=5)
        format_options = ["BC1", "BC4"]
        tk.OptionMenu(param_frame, self.format_var, *format_options).grid(row=0, column=1, sticky="w", padx=5)
        
        # File Type (Radio buttons)
        tk.Label(param_frame, text="Output Type:").grid(row=1, column=0, sticky="w", padx=5)
        type_frame = tk.Frame(param_frame)
        type_frame.grid(row=1, column=1, sticky="w")
        for ftype in ["dds", "ktx", "ktx2"]:
            tk.Radiobutton(type_frame, text=ftype, variable=self.file_type_var, value=ftype).pack(side="left")
            
        # Quality
        tk.Label(param_frame, text="Quality (0.05 - 1.0):").grid(row=2, column=0, sticky="w", padx=5)
        tk.Entry(param_frame, textvariable=self.quality_var).grid(row=2, column=1, sticky="w", padx=5)
        
        # Convert Button
        tk.Button(self, text="CONVERT", command=self.convert, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white").pack(pady=20)

    def select_source_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tga")])
        if filename:
            self.source_path.set(filename)

    def select_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_path.set(folder)

    def select_dest_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dest_folder.set(folder)

    def convert(self):
        # Validation
        source = self.source_path.get()
        dest = self.dest_folder.get()
        
        if not source:
            messagebox.showerror("Error", "Please select a source file or folder.")
            return
            
        if not dest:
            messagebox.showerror("Error", "Please select a destination folder.")
            return
            
        try:
            quality = float(self.quality_var.get())
        except ValueError:
            messagebox.showerror("Error", "Quality must be a number.")
            return
            
        # Determine output path
        source_path = Path(source)
        output_path = Path(dest)
        
        if source_path.is_file():
            # If source is file, append name + extension
            new_filename = source_path.stem + "." + self.file_type_var.get()
            output_path = output_path / new_filename
        
        # Create bcn object
        try:
            converter = bcn(
                input_image=str(source_path),
                output_image=str(output_path),
                format=self.format_var.get(),
                quality=quality,
                use_gpu=True # Assuming GPU usage as per bcn.py default/check
            )
            
            # Check logic before running (per user request interpretation "check things before creating", handled via simple constructor validation mostly, but here we can add pre-checks if needed)
            
            result = converter.run()
            
            if result.returncode == 0:
                messagebox.showinfo("Success", "Conversion completed successfully!")
            else:
                messagebox.showerror("Error", f"Conversion failed:\n{result.stderr}")
                
        except Exception as e:
             messagebox.showerror("Error", f"An error occurred: {str(e)}")
