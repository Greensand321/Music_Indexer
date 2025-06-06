# main_gui.py

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from validator import validate_soundvault_structure

# Import the indexer API
from music_indexer_api import run_full_indexer
from importer_core import scan_and_import

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


class SoundVaultImporterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SoundVault Importer")
        self.geometry("600x400")

        # Menu bar
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # ─── File Menu ─────────────────────────────────────────────────
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Validate Library", command=self.validate_library)
        file_menu.add_command(label="Import New Songs", command=self.import_songs)
        file_menu.add_command(label="Run Indexer", command=self.run_indexer)   # Calls indexer directly
        file_menu.add_command(label="Scan for Orphans", command=self.scan_orphans)
        file_menu.add_command(label="Compare Libraries", command=self.compare_libraries)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # ─── Tools Menu ────────────────────────────────────────────────
        tools_menu = tk.Menu(menubar, tearoff=False)
        tools_menu.add_command(label="Regenerate Playlists", command=self.regenerate_playlists)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # A main text area for logging
        self.output = tk.Text(self, wrap="word", state="disabled")
        self.output.pack(fill="both", expand=True)

    def validate_library(self):
        """Ask user to pick a root, then call validator."""
        initial = load_last_path()
        chosen = filedialog.askdirectory(title="Select SoundVault Root", initialdir=initial)
        if not chosen:
            return

        save_last_path(chosen)

        is_valid, errors = validate_soundvault_structure(chosen)
        if is_valid:
            messagebox.showinfo("Validation OK", "This is a valid SoundVault.")
            self._log(f"✔ Valid SoundVault: {chosen}")
        else:
            messagebox.showerror("Validation Failed", "\n".join(errors))
            self._log(f"✘ Invalid SoundVault: {chosen}\n" + "\n".join(errors))

    def import_songs(self):
        """Import new audio files into an existing SoundVault."""
        initial = load_last_path()
        vault = filedialog.askdirectory(title="Select SoundVault Root", initialdir=initial)
        if not vault:
            return

        save_last_path(vault)

        is_valid, errors = validate_soundvault_structure(vault)
        if not is_valid:
            messagebox.showerror("Invalid SoundVault", "\n".join(errors))
            self._log(f"✘ Invalid SoundVault: {vault}\n" + "\n".join(errors))
            return

        import_folder = filedialog.askdirectory(title="Select Folder of New Songs", initialdir=vault)
        if not import_folder:
            return

        dry_run = messagebox.askyesno("Dry Run?", "Perform a dry-run preview only?")
        estimate = messagebox.askyesno("Estimate BPM?", "Attempt BPM estimation for missing values?")

        try:
            summary = scan_and_import(vault, import_folder, dry_run=dry_run,
                                      estimate_bpm=estimate, log_callback=self._log)
            if summary["dry_run"]:
                messagebox.showinfo("Dry Run Complete", f"Preview written to:\n{summary['html']}")
            else:
                moved = summary.get("moved", 0)
                messagebox.showinfo("Import Complete", f"Imported {moved} files. Preview:\n{summary['html']}")

            if summary.get("errors"):
                self._log("! Some files failed to import. Check log for details.")

            self._log(f"✓ Import finished for {import_folder} → {vault}. Dry run: {dry_run}. BPM: {estimate}.")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))
            self._log(f"✘ Import failed for {import_folder}: {e}")

    def run_indexer(self):
        """
        Run the full MusicIndexer re-organization on whichever folder the user chooses,
        even if it doesn’t already have a SoundVault structure. The indexer will create
        a 'By Artist' and 'By Year' subfolder inside this folder.
        """
        initial = load_last_path()
        chosen = filedialog.askdirectory(
            title="Select Any Folder to Reorganize (becomes your Music root)",
            initialdir=initial
        )
        if not chosen:
            return

        save_last_path(chosen)

        # Ask if dry‐run
        dry_run = messagebox.askyesno("Dry Run?", "Perform a dry‐run preview (generate MusicIndex.html only)?")
        output_html = os.path.join(chosen, "MusicIndex.html")

        try:
            # Call indexer on 'chosen' directly—no prior validation
            summary = run_full_indexer(chosen, output_html, dry_run_only=dry_run, log_callback=self._log)
            if dry_run:
                messagebox.showinfo("Dry Run Complete", f"Preview written to:\n{output_html}")
            else:
                moved = summary.get("moved", 0)
                messagebox.showinfo("Indexing Complete", f"Moved/renamed {moved} files.")
            self._log(f"✓ Run Indexer finished for {chosen}. Dry run: {dry_run}.")
        except Exception as e:
            messagebox.showerror("Indexing failed", str(e))
            self._log(f"✘ Run Indexer failed for {chosen}: {e}")

    def scan_orphans(self):
        """Stub for orphan scan."""
        initial = load_last_path()
        chosen = filedialog.askdirectory(title="Select SoundVault Root", initialdir=initial)
        if not chosen:
            return

        save_last_path(chosen)
        messagebox.showinfo("Scan for Orphans", f"[stub] Would scan for orphans in:\n{chosen}")
        self._log(f"[stub] Scan for Orphans → {chosen}")

    def compare_libraries(self):
        """Stub for library comparison."""
        initial = load_last_path()
        master = filedialog.askdirectory(title="Select Master Library Root", initialdir=initial)
        if not master:
            return
        save_last_path(master)

        device = filedialog.askdirectory(title="Select Device Library Root", initialdir=master)
        if not device:
            return
        save_last_path(device)

        messagebox.showinfo("Compare Libraries", f"[stub] Would compare:\nMaster: {master}\nDevice: {device}")
        self._log(f"[stub] Compare Libraries → Master: {master}, Device: {device}")

    def regenerate_playlists(self):
        """Stub for playlist regeneration."""
        initial = load_last_path()
        chosen = filedialog.askdirectory(title="Select SoundVault Root", initialdir=initial)
        if not chosen:
            return

        save_last_path(chosen)
        messagebox.showinfo("Regenerate Playlists", f"[stub] Would regenerate playlists in:\n{chosen}")
        self._log(f"[stub] Regenerate Playlists → {chosen}")

    def _log(self, msg):
        self.output.configure(state="normal")
        self.output.insert("end", msg + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")


if __name__ == "__main__":
    app = SoundVaultImporterApp()
    app.mainloop()
