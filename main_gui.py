import os
import json
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from validator import validate_soundvault_structure
from music_indexer_api import run_full_indexer
from importer_core import scan_and_import
from sample_highlight import play_file_highlight, PYDUB_AVAILABLE
from tag_fixer import MIN_INTERACTIVE_SCORE, FileRecord
from typing import Callable, List

FilterFn = Callable[[FileRecord], bool]
_cached_filters = None

def make_filters(ex_no_diff: bool, ex_skipped: bool, show_all: bool) -> List[FilterFn]:
    global _cached_filters
    key = (ex_no_diff, ex_skipped, show_all)
    if _cached_filters and _cached_filters[0] == key:
        return _cached_filters[1]

    fns: List[FilterFn] = []
    if not show_all:
        fns.append(lambda r: r.status != 'applied')
        if ex_no_diff:
            fns.append(lambda r: r.status != 'no_diff')
        if ex_skipped:
            fns.append(lambda r: r.status != 'skipped')

    _cached_filters = (key, fns)
    return fns

def apply_filters(records: List[FileRecord], filters: List[FilterFn]) -> List[FileRecord]:
    for fn in filters:
        records = [r for r in records if fn(r)]
    return records

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
        self.show_all = False
        self.genre_mapping = {}
        self.mapping_path = ""

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
        file_menu.add_command(label="Show All Files", command=self._on_show_all)
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
        tools_menu.add_command(label="Genre Normalizer", command=self._open_genre_normalizer)
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
        show_all = False
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

        def on_show_all():
            nonlocal show_all
            result["proceed"] = True
            show_all = True
            var_no_diff.set(False)
            var_skipped.set(False)
            dlg.destroy()

        tk.Button(btn, text="OK", command=ok).pack(side="left", padx=5)
        tk.Button(btn, text="Cancel", command=cancel).pack(side="left", padx=5)
        tk.Button(btn, text="Show All Songs", command=on_show_all).pack(side="left", padx=5)
        dlg.wait_window()
        return result["proceed"], var_no_diff.get(), var_skipped.get(), show_all

    def _on_show_all(self):
        """Run tag-fix scan showing every file regardless of prior log."""
        self.show_all = True
        try:
            self.fix_tags_gui()
        finally:
            self.show_all = False

    def fix_tags_gui(self):
        folder = filedialog.askdirectory(title="Select Folder to Fix Tags")
        if not folder:
            return

        from tag_fixer import init_db
        db_path = os.path.join(folder, ".soundvault.db")
        init_db(db_path)

        # Load any saved genre mapping for this library
        self.mapping_path = os.path.join(folder, ".genre_mapping.json")
        self.genre_mapping = {}
        if os.path.isfile(self.mapping_path):
            try:
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    self.genre_mapping = json.load(f)
            except Exception:
                self.genre_mapping = {}

        from tag_fixer import find_files

        files = find_files(folder)
        print(f"[DEBUG] Total discovered files: {len(files)}")
        if not files:
            messagebox.showinfo(
                "No audio files", "No supported audio found in that folder."
            )
            return

        proceed, ex_no_diff, ex_skipped, show_all = self._tagfix_filter_dialog()
        if not proceed:
            return

        show_all = show_all or getattr(self, "show_all", False)

        q = queue.Queue()
        progress = ProgressDialog(self, total=len(files), title="Fingerprinting…")

        def worker():
            from tag_fixer import build_file_records
            import sqlite3

            conn = sqlite3.connect(db_path)

            records = build_file_records(
                folder,
                db_conn=conn,
                show_all=show_all,
                log_callback=lambda m: None,
                progress_callback=lambda idx: q.put(("progress", idx)),
            )

            conn.commit()
            conn.close()

            q.put(("done", records))

        threading.Thread(target=worker, daemon=True).start()

        def poll_queue():
            try:
                while True:
                    msg, payload = q.get_nowait()
                    if msg == "progress":
                        progress.update_progress(payload)
                    elif msg == "done":
                        progress.destroy()
                        all_records = payload
                        self.all_records = all_records

                        # Apply genre normalization if mapping is loaded
                        def normalize_list(lst):
                            return [self.genre_mapping.get(g, g) for g in lst]

                        for rec in all_records:
                            rec.old_genres = normalize_list(rec.old_genres)
                            rec.new_genres = normalize_list(rec.new_genres)

                        from tag_fixer import FileRecord, apply_tag_proposals

                        filters = make_filters(ex_no_diff, ex_skipped, show_all)
                        records = apply_filters(all_records, filters)

                        records = sorted(
                            records,
                            key=lambda r: (r.score is not None, r.score if r.score is not None else 0),
                            reverse=True,
                        )
                        self.filtered_records = records

                        if not records:
                            messagebox.showinfo(
                                'No proposals', 'No missing tags above threshold.'
                            )
                            return

                        result = self.show_proposals_dialog(records)
                        if result is not None:
                            selected, fields = result

                            count = apply_tag_proposals(
                                selected,
                                fields=fields,
                                log_callback=self._log,
                            )

                            selected_set = {rec.path for rec in selected}
                            import sqlite3
                            conn = sqlite3.connect(db_path)
                            for rec in all_records:
                                if rec.path in selected_set:
                                    status = 'applied'
                                elif rec.status == 'no_diff':
                                    status = 'no_diff'
                                else:
                                    status = 'skipped'
                                rec.status = status
                                conn.execute(
                                    "UPDATE files SET status=? WHERE path=?",
                                    (status, str(rec.path)),
                                )
                            conn.commit()
                            conn.close()
                            messagebox.showinfo('Done', f'Updated {count} files.')
                        return
            except queue.Empty:
                pass
            self.after(100, poll_queue)

        self.after(100, poll_queue)

    def show_proposals_dialog(self, records: List[FileRecord]):
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
            "Genres",            # existing embedded genres
            "Suggested Genre",   # fetched from MusicBrainz
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

        self._prop_tv = tv

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
            elif c in ("Genres", "Suggested Genre"):
                width = 150
            tv.column(c, width=width, anchor="w")

        tv.tag_configure("perfect", background="white")
        tv.tag_configure("changed", background="#fff8c6")
        tv.tag_configure("lowconf", background="#f8d7da")

        all_rows = sorted(records, key=lambda p: p.score or 0, reverse=True)
        self._render_table(all_rows)
        iid_to_prop = {iid: rec for iid, rec in zip(tv.get_children(""), all_rows)}

        def select_all(event):
            tv.selection_set(tv.get_children(""))
            return "break"

        tv.bind("<Control-a>", select_all)

        sel_label = tk.Label(dlg, text="Selected: 0")
        sel_label.pack(anchor="w", padx=10)

        def update_selection_count(event=None):
            cnt = len(tv.selection())
            sel_label.config(text=f"Selected: {cnt}")

        tv.bind("<<TreeviewSelect>>", update_selection_count)

        def on_apply():
            selected = [iid_to_prop[iid] for iid in tv.selection()]
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

    def _render_table(self, records: List[FileRecord]):
        tv = self._prop_tv
        tv.delete(*tv.get_children())
        for rec in records:
            if rec.status == 'unmatched' or (
                rec.score is not None and rec.score < MIN_INTERACTIVE_SCORE
            ):
                tag = 'lowconf'
            elif (
                rec.old_artist == rec.new_artist
                and rec.old_title == rec.new_title
                and rec.old_album == rec.new_album
                and sorted(rec.old_genres) == sorted(rec.new_genres)
            ):
                tag = 'perfect'
            else:
                tag = 'changed'
            tv.insert(
                '',
                'end',
                values=(
                    str(rec.path),
                    f"{rec.score:.2f}" if rec.score is not None else '',
                    rec.old_artist or '',
                    rec.new_artist or '',
                    rec.old_title or '',
                    rec.new_title or '',
                    rec.old_album or '',
                    rec.new_album or '',
                    '; '.join(rec.old_genres),
                    '; '.join(rec.new_genres),
                ),
                tags=(tag,),
            )

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

    def _open_genre_normalizer(self):
        """Open a dialog to assist with genre normalization."""
        folder = self.require_library()
        if not folder:
            return
        all_genres = set()
        for rec in getattr(self, "all_records", []):
            all_genres.update(rec.old_genres or [])
            all_genres.update(rec.new_genres or [])
        genre_list = "\n".join(sorted(all_genres))

        ai_prompt = (
            "I will provide a list of raw music genres (one per line). "
            "Your job is to group and map each raw genre into a canonical key in JSON format, "
            "e.g. {\"rock & roll\": \"Rock\", \"hip hop\": \"Hip-Hop\", …}. "
            "If a term is not a genre, list it under \"invalid\". "
            "If you have questions about ambiguous names, ask clarifying questions."
        )

        win = tk.Toplevel(self)
        win.title("Genre Normalization Assistant")
        win.grab_set()

        tk.Label(win, text="LLM Prompt Template:").pack(anchor="w", padx=10, pady=(10, 0))
        text_prompt = ScrolledText(win, width=50, height=6)
        text_prompt.pack(fill="both", padx=10, pady=(0, 10))
        text_prompt.insert("1.0", ai_prompt)
        text_prompt.configure(state="disabled")

        def copy_prompt():
            self.clipboard_clear()
            self.clipboard_append(ai_prompt)

        tk.Button(win, text="Copy Prompt", command=copy_prompt).pack(pady=(0, 10))

        tk.Label(win, text="Raw Genre List:").pack(anchor="w", padx=10)
        text_genres = ScrolledText(win, width=50, height=15)
        text_genres.pack(fill="both", padx=10, pady=(0, 10))
        text_genres.insert("1.0", genre_list)
        text_genres.configure(state="disabled")

        def copy_genres():
            self.clipboard_clear()
            self.clipboard_append(genre_list)

        tk.Button(win, text="Copy Raw List", command=copy_genres).pack(pady=(0, 10))

        tk.Label(win, text="Mapping JSON Input:").pack(anchor="w", padx=10)
        text_map = ScrolledText(win, width=50, height=10)
        text_map.pack(fill="both", padx=10, pady=(0, 10))

        def apply_mapping():
            mapping_json = text_map.get("1.0", "end").strip()
            mapping_path = os.path.join(self.library_path or "", ".genre_mapping.json")
            try:
                mapping = json.loads(mapping_json)
                if not isinstance(mapping, dict):
                    raise ValueError("Mapping must be a JSON object")
            except Exception as e:
                messagebox.showerror("Invalid Mapping", f"Could not parse JSON key:\n{e}")
                return
            try:
                with open(mapping_path, "w", encoding="utf-8") as f:
                    json.dump(mapping, f, indent=2)
            except Exception as e:
                messagebox.showerror("Save Failed", str(e))
                return

            self.mapping_path = mapping_path
            self.genre_mapping = mapping

            def normalize_list(lst):
                return [self.genre_mapping.get(g, g) for g in lst]

            for rec in getattr(self, "all_records", []):
                rec.old_genres = normalize_list(rec.old_genres)
                rec.new_genres = normalize_list(rec.new_genres)

            if hasattr(self, "filtered_records") and hasattr(self, "_prop_tv"):
                self._render_table(self.filtered_records)

            win.destroy()

        tk.Button(win, text="Apply Mapping", command=apply_mapping).pack(side="left", padx=10, pady=(0, 10))
        tk.Button(win, text="Close", command=win.destroy).pack(side="right", padx=10, pady=(0, 10))

    def reset_tagfix_log(self):
        initial = self.library_path or load_last_path()
        folder = filedialog.askdirectory(title="Select Library Root", initialdir=initial)
        if not folder:
            return
        if not messagebox.askyesno("Reset Log", "This will erase all history of prior scans. Continue?"):
            return
        db_path = os.path.join(folder, ".soundvault.db")
        if os.path.exists(db_path):
            os.remove(db_path)
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
