import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import sys

# Ensure parent directory is in path to import bcn
sys.path.append(str(Path(__file__).parent.parent))
from src.bcn import bcn, get_compressonator_path

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
        self.compressonator_path = tk.StringVar(value=str(get_compressonator_path()))
        
        # UI Layout
        self.create_widgets()
        
    def create_widgets(self):
        # Back Button
        back_btn = tk.Button(self, text="< Back", command=lambda: self.controller.show_frame("MainMenu"))
        back_btn.pack(anchor="nw", padx=10, pady=10)
        
        # Title
        tk.Label(self, text="Converter", font=("Arial", 16)).pack(pady=10)
        
        # Row 1: Source and Destination
        row1 = tk.Frame(self)
        row1.pack(fill="x", padx=5, pady=5)

        # Source Selection
        source_frame = tk.LabelFrame(row1, text="Source", padx=10, pady=10)
        source_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        btn_frame = tk.Frame(source_frame)
        btn_frame.pack(fill="x")
        
        tk.Button(btn_frame, text="Select File", command=self.select_source_file).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Select Folder", command=self.select_source_folder).pack(side="left", padx=5)
        
        tk.Label(source_frame, textvariable=self.source_path).pack(fill="x", pady=5)
        
        # Destination Selection
        dest_frame = tk.LabelFrame(row1, text="Destination", padx=10, pady=10)
        dest_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        tk.Button(dest_frame, text="Select Output Folder", command=self.select_dest_folder).pack(anchor="w")
        tk.Label(dest_frame, textvariable=self.dest_folder).pack(fill="x", pady=5)
        
        # Row 2: Parameters and System Configuration
        row2 = tk.Frame(self)
        row2.pack(fill="x", padx=5, pady=5)

        # Parameters
        param_frame = tk.LabelFrame(row2, text="Parameters", padx=10, pady=10)
        param_frame.pack(side="left", fill="both", expand=True, padx=5)
        
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

        # Compressonator Path
        sys_frame = tk.LabelFrame(row2, text="System Configuration", padx=10, pady=10)
        sys_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        tk.Button(sys_frame, text="Change Compressonator Path", command=self.change_compressonator_path).pack(anchor="w")
        tk.Label(sys_frame, textvariable=self.compressonator_path, wraplength=250, justify="left").pack(fill="x", pady=5)
        
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

    def change_compressonator_path(self):
        filename = filedialog.askopenfilename(
            title="Select Compressonator CLI Executable",
            filetypes=[("Executable", "*.exe"), ("All Files", "*.*")]
        )
        if filename:
            self.compressonator_path.set(filename)
            self.update_env_file(filename)

    def update_env_file(self, new_path):
        env_path = Path(__file__).parent.parent / ".env"
        lines = []
        if env_path.exists():
            try:
                with open(env_path, "r") as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"Error reading .env: {e}")
        
        # Remove existing key
        lines = [line for line in lines if not line.strip().startswith("COMPRESSONATOR_PATH=")]
        
        # Add new key
        lines.append(f"COMPRESSONATOR_PATH={new_path}\n")
        
        try:
            with open(env_path, "w") as f:
                f.writelines(lines)
            messagebox.showinfo("Configuration Saved", "Compressonator path updated successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

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
        
        files_to_process = []
        if source_path.is_file():
            files_to_process.append(source_path)
        elif source_path.is_dir():
            valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tga'}
            for item in source_path.iterdir():
                if item.is_file() and item.suffix.lower() in valid_extensions:
                    files_to_process.append(item)
        
        if not files_to_process:
            messagebox.showerror("Error", "No image files found to process.")
            return

        print(f"Found {len(files_to_process)} files to process.") # Debug print

            
        error_messages = []
        success_count = 0
        
        for src_file in files_to_process:
            try:
                # Construct output filename
                quality_str = f"{quality:.2f}"
                new_filename = f"{src_file.stem}-{self.format_var.get()}-{quality_str}.{self.file_type_var.get()}"
                out_file_path = output_path / new_filename
                
                converter = bcn(
                    input_image=str(src_file),
                    output_image=str(out_file_path),
                    format=self.format_var.get(),
                    quality=quality,
                    use_gpu=True
                )
                
                result = converter.run()
                
                if result.returncode == 0:
                    success_count += 1
                else:
                    error_messages.append(f"{src_file.name}: {result.stderr}")
                    
            except Exception as e:
                error_messages.append(f"{src_file.name}: {str(e)}")
        
        if not error_messages:
            messagebox.showinfo("Success", f"Converted {success_count} files successfully!")
        else:
            msg = f"Completed with errors.\nSuccess: {success_count}\nFailures: {len(error_messages)}\n\nErrors:\n" + "\n".join(error_messages[:5])
            if len(error_messages) > 5:
                msg += "\n..."
            messagebox.showerror("Conversion Issues", msg)
