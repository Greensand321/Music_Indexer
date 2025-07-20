import os
import tkinter as tk
from tkinter import ttk, messagebox

from controllers.highlight_controller import play_snippet, PYDUB_AVAILABLE

class CandidatePopup(tk.Toplevel):
    """Popup showing alternative candidate files with play buttons."""

    def __init__(self, parent: tk.Misc, candidates: list[str]):
        super().__init__(parent)
        self.title("Candidate Matches")
        self.resizable(False, False)
        self.transient(parent)

        frame = ttk.Frame(self)
        frame.pack(padx=10, pady=10, fill="both", expand=True)

        for path in candidates:
            row = ttk.Frame(frame)
            ttk.Label(row, text=os.path.basename(path)).pack(side="left", padx=(0, 5))
            if PYDUB_AVAILABLE:
                ttk.Button(row, text="Play", command=lambda p=path: self._play(p)).pack(side="right")
            row.pack(fill="x", pady=2)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 10))

    def _play(self, path: str) -> None:
        try:
            play_snippet(path)
        except Exception as e:
            messagebox.showerror("Playback failed", str(e))

