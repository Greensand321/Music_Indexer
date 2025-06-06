import os
import tkinter as tk
from tkinter import filedialog, messagebox

from validator import validate_soundvault_structure
from music_indexer_api import run_full_indexer
from importer_core import scan_and_import
from tag_fixer import fix_tags  # AcoustID-based fixer

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "last_path.txt")


def load_last_path():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return ""


def save_last_path(path):
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

        # ─── Menu Bar ─────────────────────────────────────────────────────────
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Validate Library", command=self.validate_library)
        file_menu.add_command(label="Import New Songs", command=self.import_songs)
        file_menu.add_command(label="Run Indexer", command=self.run_indexer)
        file_menu.add_command(label="Scan for Orphans", command=self.scan_orphans)
        file_menu.add_command(label="Compare Libraries", command=self.compare_libraries)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menubar, tearoff=False)
        tools_menu.add_command(label="Regenerate Playlists", command=self.regenerate_playlists)
        tools_menu.add_command(label="Fix Tags via AcoustID", command=self.fix_tags)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # ─── Library Info ─────────────────────────────────────────────────────
        info_frame = tk.LabelFrame(self, text="Library Info")
        info_frame.pack(fill="x", padx=10, pady=10)

        tk.Button(info_frame, text="Select Library", command=self.select_library).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.library_name_var).pack(anchor="w", pady=(5, 0))
        tk.Label(info_frame, textvariable=self.library_path_var, wraplength=650, justify="left").pack(anchor="w")
        tk.Label(info_frame, textvariable=self.library_stats_var, justify="left").pack(anchor="w", pady=(5, 0))

        # ─── Output Log ───────────────────────────────────────────────────────
        self.output = tk.Text(self, wrap="word", state="disabled", height=15)
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
        path = self.require_library()
        if not path:
            return
        messagebox.showinfo("Scan for Orphans", f"[stub] Would scan for orphans in:\n{path}")
        self._log(f"[stub] Scan for Orphans → {path}")
        self.update_library_info()

    def compare_libraries(self):
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
        path = self.require_library()
        if not path:
            return
        messagebox.showinfo(
            "Regenerate Playlists", f"[stub] Would regenerate playlists in:\n{path}"
        )
        self._log(f"[stub] Regenerate Playlists → {path}")
        self.update_library_info()

    def fix_tags(self):
        initial = load_last_path()
        chosen = filedialog.askdirectory(title="Select Folder for Tag Fixer", initialdir=initial)
        if not chosen:
            return
        save_last_path(chosen)
        try:
            summary = fix_tags(chosen, log_callback=self._log)
            messagebox.showinfo("Tag Fixer Complete",
                                f"Processed {summary['processed']} files\nUpdated {summary['updated']} files.")
        except Exception as e:
            messagebox.showerror("Tag Fixer Error", str(e))

    def _log(self, msg):
        self.output.configure(state="normal")
        self.output.insert("end", msg + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")


if __name__ == "__main__":
    app = SoundVaultImporterApp()
    app.mainloop()
