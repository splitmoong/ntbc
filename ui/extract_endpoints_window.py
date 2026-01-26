import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import sys

# Ensure parent directory is in path to import src modules
sys.path.append(str(Path(__file__).parent.parent))
from src.extract_endpoints import extract_endpoints_to_json

class ExtractEndpointsFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Variables
        self.source_path = tk.StringVar()
        self.dest_folder = tk.StringVar()
        
        # UI Layout
        self.create_widgets()
        
    def create_widgets(self):
        # Back Button
        back_btn = tk.Button(self, text="< Back", command=lambda: self.controller.show_frame("MainMenu"))
        back_btn.pack(anchor="nw", padx=10, pady=10)
        
        # Title
        tk.Label(self, text="Extract Endpoints", font=("Arial", 16)).pack(pady=10)
        
        # Source Selection
        source_frame = tk.LabelFrame(self, text="Source File", padx=10, pady=10)
        source_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(source_frame, text="Select File", command=self.select_source_file).pack(anchor="w")
        tk.Label(source_frame, textvariable=self.source_path).pack(fill="x", pady=5)
        
        # Destination Selection
        dest_frame = tk.LabelFrame(self, text="Output Folder", padx=10, pady=10)
        dest_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(dest_frame, text="Select Output Folder", command=self.select_dest_folder).pack(anchor="w")
        tk.Label(dest_frame, textvariable=self.dest_folder).pack(fill="x", pady=5)
        
        # Extract Button
        tk.Button(self, text="Extract", command=self.extract, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white").pack(pady=20)

    def select_source_file(self):
        filename = filedialog.askopenfilename(filetypes=[("DDS files", "*.dds")])
        if filename:
            self.source_path.set(filename)

    def select_dest_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dest_folder.set(folder)

    def extract(self):
        # Validation
        source = self.source_path.get()
        dest = self.dest_folder.get()
        
        if not source:
            messagebox.showerror("Error", "Please select a source .dds file.")
            return
            
        if not dest:
            messagebox.showerror("Error", "Please select an output folder.")
            return
            
        try:
            output_file = extract_endpoints_to_json(source, output_folder=dest)
            messagebox.showinfo("Success", f"Endpoints extracted to:\n{output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract endpoints:\n{str(e)}")
