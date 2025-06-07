import os
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from validator import validate_soundvault_structure
from music_indexer_api import run_full_indexer
from importer_core import scan_and_import
from sample_highlight import play_file_highlight, PYDUB_AVAILABLE
import log_manager

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


class ProgressDialog(tk.Toplevel):
    def __init__(self, parent, total, title="Working…"):
        super().__init__(parent)
        self.title(title)
        self.grab_set()
        self.resizable(False, False)

        tk.Label(self, text="Scanning files, please wait…").pack(padx=10, pady=(10, 0))
        self.pb = ttk.Progressbar(
            self,
            orient="horizontal",
            length=300,
            mode="determinate",
            maximum=total,
        )
        self.pb.pack(padx=10, pady=10)

        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.update()

    def update_progress(self, value):
        self.pb["value"] = value
        self.update_idletasks()


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
        file_menu.add_command(label="Open Library…", command=self.select_library)
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
        tools_menu.add_command(label="Fix Tags via AcoustID", command=self.fix_tags_gui)
        if PYDUB_AVAILABLE:
            tools_menu.add_command(label="Sample Song Highlight", command=self.sample_song_highlight)
        else:
            tools_menu.add_command(
                label="Sample Song Highlight (requires pydub)",
                state="disabled"
            )
        tools_menu.add_separator()
        tools_menu.add_command(label="Reset Tag-Fix Log", command=self.reset_tagfix_log)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        # ─── Library Info ─────────────────────────────────────────────────────
        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 0))
        tk.Button(top, text="Choose Library…", command=self.select_library).pack(side="left")
        tk.Label(top, textvariable=self.library_name_var, anchor="w").pack(side="left", padx=(5, 0))
        tk.Label(self, textvariable=self.library_path_var, anchor="w").pack(fill="x", padx=10)
        tk.Label(self, textvariable=self.library_stats_var, justify="left").pack(anchor="w", padx=10, pady=(0, 10))

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

    def _tagfix_filter_dialog(self):
        dlg = tk.Toplevel(self)
        dlg.title("Exclude from this scan")
        dlg.grab_set()
        var_no_diff = tk.BooleanVar()
        var_skipped = tk.BooleanVar()
        tk.Checkbutton(
            dlg,
            text="Songs previously logged as \u201cno differences\u201d",
            variable=var_no_diff,
        ).pack(anchor="w", padx=10, pady=(10, 5))
        tk.Checkbutton(
            dlg,
            text="Songs previously logged as \u201cskipped\u201d",
            variable=var_skipped,
        ).pack(anchor="w", padx=10)
        btn = tk.Frame(dlg)
        btn.pack(pady=10)
        result = {"proceed": False}

        def ok():
            result["proceed"] = True
            dlg.destroy()

        def cancel():
            dlg.destroy()

        tk.Button(btn, text="OK", command=ok).pack(side="left", padx=5)
        tk.Button(btn, text="Cancel", command=cancel).pack(side="left", padx=5)
        dlg.wait_window()
        return result["proceed"], var_no_diff.get(), var_skipped.get()

    def fix_tags_gui(self):
        folder = filedialog.askdirectory(title="Select Folder to Fix Tags")
        if not folder:
            return

        log_data = log_manager.load_log(folder)
        removed = log_manager.prune_missing_entries(log_data, folder)
        if removed:
            msg = "Cleaned up %d log entries for missing files:\n" % len(removed)
            msg += "\n".join("\u2022 " + p for p in removed)
            msg += "\nProceed with scan?"
            if not messagebox.askyesno("Cleanup", msg):
                return
            log_manager.save_log(log_data, folder)

        from tag_fixer import find_files

        files = find_files(folder)
        if not files:
            messagebox.showinfo(
                "No audio files", "No supported audio found in that folder."
            )
            return

        proceed, ex_no_diff, ex_skipped = self._tagfix_filter_dialog()
        if not proceed:
            return

        filtered = []
        for f in files:
            rel = os.path.relpath(f, folder)
            entry = log_data.get(rel)
            if entry:
                if entry.get("status") == "applied":
                            continue
                if ex_no_diff and entry.get("status") == "no_diff":
                    continue
                if ex_skipped and entry.get("status") == "skipped":
                    continue
            filtered.append(f)

        if not filtered:
            messagebox.showinfo("No files", "All files were excluded by your filters.")
            return

        q = queue.Queue()
        progress = ProgressDialog(self, total=len(filtered), title="Fingerprinting…")

        def worker():
            from tag_fixer import collect_tag_proposals

            diff_props, no_diff = collect_tag_proposals(
                filtered,
                log_callback=lambda m: None,
                progress_callback=lambda idx: q.put(("progress", idx)),
            )
            q.put(("done", (diff_props, no_diff)))

        threading.Thread(target=worker, daemon=True).start()

        def poll_queue():
            try:
                while True:
                    msg, payload = q.get_nowait()
                    if msg == "progress":
                        progress.update_progress(payload)
                    elif msg == "done":
                        progress.destroy()
                        diff_props, no_diff_files = payload
                        if not diff_props and (ex_no_diff or not no_diff_files):
                            messagebox.showinfo(
                                "No proposals", "No missing tags above threshold."
                            )
                            return
                        result = self.show_proposals_dialog(
                            diff_props,
                            [] if ex_no_diff else no_diff_files,
                        )
                        if result is not None:
                            selected, fields = result
                            from tag_fixer import apply_tag_proposals

                            count = apply_tag_proposals(
                                selected,
                                diff_props,
                                no_diff_files,
                                folder,
                                fields=fields,
                                log_callback=self._log,
                            )
                            messagebox.showinfo("Done", f"Updated {count} files.")
                        return
            except queue.Empty:
                pass
            self.after(100, poll_queue)

        self.after(100, poll_queue)

    def show_proposals_dialog(self, diff_proposals, no_diff_files):
        # Reset from any prior invocation
        self._proceed = False
        self._selected = ([], [])

        dlg = tk.Toplevel(self)
        dlg.title("Review Tag Fix Proposals")
        dlg.grab_set()

        self.apply_artist = tk.BooleanVar(value=True)
        self.apply_title = tk.BooleanVar(value=True)
        self.apply_album = tk.BooleanVar(value=False)
        self.apply_genres = tk.BooleanVar(value=False)

        chk_frame = ttk.Frame(dlg)
        for var, label in (
            (self.apply_artist, "Artist"),
            (self.apply_title, "Title"),
            (self.apply_album, "Album"),
            (self.apply_genres, "Genres"),
        ):
            ttk.Checkbutton(chk_frame, text=label, variable=var).pack(side="left", padx=5)
        chk_frame.pack(pady=5)


        cols = (
            "File",
            "Score",
            "Old Artist",
            "New Artist",
            "Old Title",
            "New Title",
            "Old Album",
            "New Album",
            "Genres",
        )

        container = tk.Frame(dlg)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        vsb = ttk.Scrollbar(container, orient="vertical")
        vsb.pack(side="right", fill="y")
        hsb = ttk.Scrollbar(container, orient="horizontal")
        hsb.pack(side="bottom", fill="x")

        tv = ttk.Treeview(
            container,
            columns=cols,
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            selectmode="extended",
        )
        vsb.config(command=tv.yview)
        hsb.config(command=tv.xview)
        tv.pack(fill="both", expand=True)

        def treeview_sort_column(tv, col, reverse=False):
            data = [(tv.set(k, col), k) for k in tv.get_children("")]
            try:
                data = [(float(v), k) for v, k in data]
            except ValueError:
                pass
            data.sort(reverse=reverse)
            for idx, (_, k) in enumerate(data):
                tv.move(k, "", idx)
            tv.heading(col, command=lambda: treeview_sort_column(tv, col, not reverse))

        for c in cols:
            tv.heading(c, text=c, command=lambda _c=c: treeview_sort_column(tv, _c, False))
            width = 100
            if c == "File":
                width = 300
            elif c in ("Old Album", "New Album"):
                width = 120
            elif c == "Genres":
                width = 150
            tv.column(c, width=width, anchor="w")

        tv.tag_configure("perfect", background="white")
        tv.tag_configure("changed", background="#fff8c6")

        all_rows = sorted(diff_proposals, key=lambda p: p.score, reverse=True) + list(no_diff_files)
        iid_to_prop = {}
        for p in all_rows:
            row_tag = "perfect" if (p.old_artist == p.new_artist and p.old_title == p.new_title) else "changed"
            iid = tv.insert(
                "",
                "end",
                values=(
                    str(p.file),
                    f"{p.score:.2f}",
                    p.old_artist or "",
                    p.new_artist or "",
                    p.old_title or "",
                    p.new_title or "",
                    p.old_album or "",
                    p.new_album or "",
                    ", ".join(p.new_genres or []),
                ),
                tags=(row_tag,),
            )
            if p in diff_proposals:
                iid_to_prop[iid] = p

        def select_all(event):
            tv.selection_set(tv.get_children(""))
            return "break"

        tv.bind("<Control-a>", select_all)

        sel_label = tk.Label(dlg, text="Selected: 0")
        sel_label.pack(anchor="w", padx=10)

        def update_selection_count(event=None):
            cnt = sum(1 for iid in tv.selection() if iid in iid_to_prop)
            sel_label.config(text=f"Selected: {cnt}")

        tv.bind("<<TreeviewSelect>>", update_selection_count)

        def on_apply():
            selected = [iid_to_prop[iid] for iid in tv.selection() if iid in iid_to_prop]
            fields = []
            if self.apply_artist.get():
                fields.append("artist")
            if self.apply_title.get():
                fields.append("title")
            if self.apply_album.get():
                fields.append("album")
            if self.apply_genres.get():
                fields.append("genres")
            self._selected = (selected, fields)
            dlg.destroy()
            setattr(self, "_proceed", True)

        btn_frame = tk.Frame(dlg)
        tk.Button(btn_frame, text="Apply Selection", command=on_apply).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(side="left", padx=5)
        btn_frame.pack(pady=10)

        update_selection_count()
        dlg.wait_window()
        if getattr(self, "_proceed", False):
            return self._selected
        return None

    def sample_song_highlight(self):
        """Ask the user for an audio file and play its highlight."""
        initial = load_last_path()
        path = filedialog.askopenfilename(
            title="Select Audio File",
            initialdir=initial,
            filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.ogg"), ("All files", "*")],
        )
        if not path:
            return

        save_last_path(os.path.dirname(path))
        try:
            start_sec = play_file_highlight(path)
            self._log(f"Played highlight of '{os.path.basename(path)}' starting at {start_sec:.2f}s")
        except Exception as e:
            messagebox.showerror("Playback failed", str(e))
            self._log(f"✘ Playback failed for {path}: {e}")

    def reset_tagfix_log(self):
        initial = self.library_path or load_last_path()
        folder = filedialog.askdirectory(title="Select Library Root", initialdir=initial)
        if not folder:
            return
        if not messagebox.askyesno("Reset Log", "This will erase all history of prior scans. Continue?"):
            return
        log_manager.reset_log(folder)
        messagebox.showinfo("Reset", "Tag-fix log cleared.")
        self._log(f"Reset tag-fix log for {folder}")

    def _log(self, msg):
        self.output.configure(state="normal")
        self.output.insert("end", msg + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")


if __name__ == "__main__":
    app = SoundVaultImporterApp()
    app.mainloop()
