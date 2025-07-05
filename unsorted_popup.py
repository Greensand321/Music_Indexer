import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

class UnsortedPopup(tk.Toplevel):
    """Dialog to allow user to open the 'Not Sorted' folder."""

    def __init__(self, parent: tk.Misc, folder: str):
        super().__init__(parent)
        self.folder = folder
        self.title("Not Sorted")
        self.resizable(False, False)
        # Center relative to parent
        self.transient(parent)

        msg = (
            "Move any folders you want the indexer to skip into the 'Not Sorted' "
            "folder, then press Continue."
        )
        ttk.Label(self, text=msg, wraplength=320, justify="left").pack(
            padx=10, pady=(10, 5)
        )

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=(0, 10), padx=10, fill="x")
        open_btn = ttk.Button(btn_frame, text="Open Folder", command=self.open_folder)
        open_btn.pack(side="left")
        close_btn = ttk.Button(btn_frame, text="Continue", command=self.destroy)
        close_btn.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def open_folder(self) -> None:
        """Open the Not Sorted folder in the OS file browser."""
        try:
            if os.name == "nt":
                os.startfile(self.folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", self.folder])
            else:
                subprocess.Popen(["xdg-open", self.folder])
        except Exception as e:
            messagebox.showerror("Open Folder Failed", str(e))
