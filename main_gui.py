# main_gui.py

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from validator import validate_soundvault_structure

# Import the indexer API
from music_indexer_api import run_full_indexer

# Path to remember the last‐used directory
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "last_path.txt")


def load_last_path():
    """Return the last‐used path from last_path.txt, or '' if none."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def save_last_path(path):
    """Save the given path into last_path.txt."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(path)
    except Exception:
        pass


def count_audio_files(root):
    exts = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
    count = 0
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in exts:
                count += 1
    return count


class SoundVaultImporterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SoundVault Importer")
        self.geometry("700x500")

        self.library_path = ""
        self.library_name_var = tk.StringVar(value="No library selected")
        self.library_path_var = tk.StringVar(value="")
        self.library_stats_var = tk.StringVar(value="")

        top = tk.Frame(self)
        top.pack(fill="both", expand=True)

        info_frame = tk.LabelFrame(top, text="Library Info", width=350)
        info_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        button_frame = tk.Frame(top)
        button_frame.pack(side="right", fill="y", padx=10, pady=10)

        tk.Button(info_frame, text="Select Library", command=self.select_library).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.library_name_var).pack(anchor="w", pady=(5, 0))
        tk.Label(
            info_frame, textvariable=self.library_path_var, wraplength=320, justify="left"
        ).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.library_stats_var, justify="left").pack(
            anchor="w", pady=(5, 0)
        )

        actions_frame = tk.LabelFrame(button_frame, text="Library Tasks")
        actions_frame.pack(fill="x", pady=(0, 10))
        for lbl, cmd in [
            ("Validate Library", self.validate_library),
            ("Import New Songs", self.import_songs),
            ("Run Indexer", self.run_indexer),
        ]:
            tk.Button(actions_frame, text=lbl, command=cmd).pack(fill="x", pady=2)

        maint_frame = tk.LabelFrame(button_frame, text="Maintenance")
        maint_frame.pack(fill="x", pady=(0, 10))
        for lbl, cmd in [
            ("Scan for Orphans", self.scan_orphans),
            ("Compare Libraries", self.compare_libraries),
            ("Regenerate Playlists", self.regenerate_playlists),
        ]:
            tk.Button(maint_frame, text=lbl, command=cmd).pack(fill="x", pady=2)

        tk.Button(button_frame, text="Exit", command=self.quit).pack(fill="x", pady=(20, 0))

        self.output = tk.Text(self, wrap="word", state="disabled", height=10)
        self.output.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def select_library(self):
        initial = load_last_path()
        chosen = filedialog.askdirectory(title="Select SoundVault Root", initialdir=initial)
        if not chosen:
            return
        save_last_path(chosen)
        self.library_path = chosen
        self.library_name_var.set(os.path.basename(chosen) or chosen)
        self.library_path_var.set(chosen)
        self.update_library_info()

    def update_library_info(self):
        if not self.library_path:
            self.library_stats_var.set("")
            return
        num = count_audio_files(self.library_path)
        is_valid, _ = validate_soundvault_structure(self.library_path)
        status = "Valid" if is_valid else "Invalid"
        self.library_stats_var.set(f"Songs: {num}\nValidation: {status}")

    def require_library(self):
        if not self.library_path:
            messagebox.showwarning("No Library", "Please select a library first.")
            return None
        return self.library_path

    def validate_library(self):
        """Validate the currently selected library."""
        path = self.require_library()
        if not path:
            return

        is_valid, errors = validate_soundvault_structure(path)
        if is_valid:
            messagebox.showinfo("Validation OK", "This is a valid SoundVault.")
            self._log(f"✔ Valid SoundVault: {path}")
        else:
            messagebox.showerror("Validation Failed", "\n".join(errors))
            self._log(f"✘ Invalid SoundVault: {path}\n" + "\n".join(errors))
        self.update_library_info()

    def import_songs(self):
        """Stub for future 'Add New Songs' functionality."""
        path = self.require_library()
        if not path:
            return
        messagebox.showinfo("Import New Songs", f"[stub] Would import new songs into:\n{path}")
        self._log(f"[stub] Import New Songs → {path}")
        self.update_library_info()

    def run_indexer(self):
        """Run the MusicIndexer on the selected library."""
        path = self.require_library()
        if not path:
            return

        dry_run = messagebox.askyesno(
            "Dry Run?", "Perform a dry-run preview (generate MusicIndex.html only)?"
        )
        output_html = os.path.join(path, "MusicIndex.html")

        try:
            summary = run_full_indexer(
                path, output_html, dry_run_only=dry_run, log_callback=self._log
            )
            if dry_run:
                messagebox.showinfo("Dry Run Complete", f"Preview written to:\n{output_html}")
            else:
                moved = summary.get("moved", 0)
                messagebox.showinfo("Indexing Complete", f"Moved/renamed {moved} files.")
            self._log(f"✓ Run Indexer finished for {path}. Dry run: {dry_run}.")
        except Exception as e:
            messagebox.showerror("Indexing failed", str(e))
            self._log(f"✘ Run Indexer failed for {path}: {e}")
        self.update_library_info()

    def scan_orphans(self):
        """Stub for orphan scan."""
        path = self.require_library()
        if not path:
            return
        messagebox.showinfo("Scan for Orphans", f"[stub] Would scan for orphans in:\n{path}")
        self._log(f"[stub] Scan for Orphans → {path}")
        self.update_library_info()

    def compare_libraries(self):
        """Stub for library comparison."""
        master = self.require_library()
        if not master:
            return
        device = filedialog.askdirectory(title="Select Device Library Root", initialdir=master)
        if not device:
            return
        save_last_path(device)
        messagebox.showinfo(
            "Compare Libraries", f"[stub] Would compare:\nMaster: {master}\nDevice: {device}"
        )
        self._log(f"[stub] Compare Libraries → Master: {master}, Device: {device}")
        self.update_library_info()

    def regenerate_playlists(self):
        """Stub for playlist regeneration."""
        path = self.require_library()
        if not path:
            return
        messagebox.showinfo(
            "Regenerate Playlists", f"[stub] Would regenerate playlists in:\n{path}"
        )
        self._log(f"[stub] Regenerate Playlists → {path}")
        self.update_library_info()

    def _log(self, msg):
        self.output.configure(state="normal")
        self.output.insert("end", msg + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")


if __name__ == "__main__":
    app = SoundVaultImporterApp()
    app.mainloop()
