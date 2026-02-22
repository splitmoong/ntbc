import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import sys

# Ensure parent directory is in path to import src modules
sys.path.append(str(Path(__file__).parent.parent))
from src.extract_endpoints import EndpointExtractor

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
        
        # Container for Source and Destination (Side by Side)
        row1 = tk.Frame(self)
        row1.pack(fill="x", padx=5, pady=5)

        tk.Label(row1, text="Source Folder:", anchor="w").pack(fill="x", padx=15)
        source_frame = tk.Frame(row1)
        source_frame.pack(fill="x", padx=15, pady=5)
        tk.Entry(source_frame, textvariable=self.source_path).pack(side="left", fill="x", expand=True)
        tk.Button(source_frame, text="Browse", command=self.select_source_folder).pack(side="right", padx=5)

        tk.Label(row1, text="Output Folder:", anchor="w").pack(fill="x", padx=15, pady=(10, 0))
        dest_frame = tk.Frame(row1)
        dest_frame.pack(fill="x", padx=15, pady=5)
        tk.Entry(dest_frame, textvariable=self.dest_folder).pack(side="left", fill="x", expand=True)
        tk.Button(dest_frame, text="Browse", command=self.select_dest_folder).pack(side="right", padx=5)
        
        # Extract Button
        tk.Button(self, text="Extract", command=self.extract, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white").pack(pady=20)

    def select_source_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.source_path.set(folder)

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
            extractor = EndpointExtractor(source, dest)
            output_file = extractor.extract()
            messagebox.showinfo("Success", f"Dataset created at:\n{output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract endpoints:\n{str(e)}")
