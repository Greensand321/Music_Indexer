import threading
import os, json
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from validator import validate_soundvault_structure
from music_indexer_api import run_full_indexer
from controllers.library_index_controller import generate_index
from controllers.import_controller import import_new_files
from controllers.genre_list_controller import list_unique_genres
from controllers.highlight_controller import play_snippet, PYDUB_AVAILABLE
from tag_fixer import MIN_INTERACTIVE_SCORE, FileRecord
from typing import Callable, List

from controllers.library_controller import (
    load_last_path,
    save_last_path,
    count_audio_files,
    open_library,
)
from controllers.tagfix_controller import (
    prepare_library,
    discover_files,
    gather_records,
    apply_proposals,
)
from controllers.normalize_controller import (
    normalize_genres,
    PROMPT_TEMPLATE,
    load_mapping,
    scan_raw_genres,
)

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
        tools_menu.add_command(
            label="Generate Library Index…",
            command=lambda: generate_index(self.require_library()),
        )
        tools_menu.add_command(
            label="List Unique Genres…",
            command=lambda: list_unique_genres(self.require_library()),
        )
        if PYDUB_AVAILABLE:
            tools_menu.add_command(label="Play Highlight…", command=self.sample_song_highlight)
        else:
            tools_menu.add_command(
                label="Play Highlight… (requires pydub)",
                state="disabled",
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

    def _load_genre_mapping(self):
        """Load genre mapping from ``self.mapping_path`` if possible."""
        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.genre_mapping = json.load(f)
        except Exception:
            self.genre_mapping = {}

    def select_library(self):
        initial = load_last_path()
        chosen = filedialog.askdirectory(
            title="Select SoundVault Root", initialdir=initial
        )
        if not chosen:
            return
        save_last_path(chosen)

        info = open_library(chosen)
        self.library_path = info["path"]
        self.library_name_var.set(info["name"])
        self.library_path_var.set(info["path"])
        self.mapping_path = os.path.join(self.library_path, ".genre_mapping.json")
        self._load_genre_mapping()
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
            summary = import_new_files(
                vault,
                import_folder,
                dry_run=dry_run,
                estimate_bpm=estimate,
                log_callback=self._log,
            )
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

        db_path, _ = prepare_library(folder)
        self.mapping_path = os.path.join(folder, ".genre_mapping.json")
        self._load_genre_mapping()

        files = discover_files(folder)
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
            records = gather_records(
                folder,
                db_path,
                show_all,
                progress_callback=lambda idx: q.put(("progress", idx)),
            )
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

                        # Apply genre normalization before filtering
                        for rec in self.all_records:
                            rec.old_genres = normalize_genres(rec.old_genres, self.genre_mapping)
                            rec.new_genres = normalize_genres(rec.new_genres, self.genre_mapping)

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

                            count = apply_proposals(
                                selected,
                                all_records,
                                db_path,
                                fields,
                                log_callback=self._log,
                            )
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
            start_sec = play_snippet(path)
            self._log(
                f"Played highlight of '{os.path.basename(path)}' starting at {start_sec:.2f}s"
            )
        except Exception as e:
            messagebox.showerror("Playback failed", str(e))
            self._log(f"✘ Playback failed for {path}: {e}")

    def _open_genre_normalizer(self, _show_dialog: bool = False):
        """Open a dialog to assist with genre normalization."""
        folder = self.require_library()
        if not folder:
            return

        # Always refresh mapping path
        self.mapping_path = os.path.join(folder, ".genre_mapping.json")

        if not _show_dialog:
            # ── Show progress bar and run raw-only scan ──
            files = discover_files(folder)
            if not files:
                messagebox.showinfo("No audio files", "No supported audio found.")
                return

            prog_win = tk.Toplevel(self)
            prog_win.title("Scanning Genres…")
            prog_bar = ttk.Progressbar(
                prog_win, orient="horizontal", length=300, mode="determinate"
            )
            prog_bar.pack(padx=20, pady=20)
            prog_bar["maximum"] = len(files)

            def on_progress(idx, total):
                prog_bar["value"] = idx
                prog_win.update_idletasks()

            def scan_task():
                self.raw_genre_list = scan_raw_genres(folder, on_progress)
                prog_win.destroy()
                # reopen dialog to show panels
                self.after(0, lambda: self._open_genre_normalizer(True))

            threading.Thread(target=scan_task, daemon=True).start()
            return

        # ── Now _show_dialog == True: build the three-panel dialog ──
        win = tk.Toplevel(self)
        win.title("Genre Normalization Assistant")
        win.grab_set()

        # 1) LLM Prompt
        tk.Label(win, text="LLM Prompt Template:").pack(anchor="w", padx=10, pady=(10,0))
        self.text_prompt = ScrolledText(win, height=8, wrap="word")
        self.text_prompt.pack(fill="both", padx=10)
        self.text_prompt.insert("1.0", PROMPT_TEMPLATE.strip())
        self.text_prompt.configure(state="disabled")
        ttk.Button(
            win,
            text="Copy Prompt",
            command=lambda: self.clipboard_append(PROMPT_TEMPLATE.strip()),
        ).pack(anchor="e", padx=10, pady=(0,10))

        # 2) Raw Genre List
        tk.Label(win, text="Raw Genre List:").pack(anchor="w", padx=10)
        self.text_raw = ScrolledText(win, width=50, height=15)
        self.text_raw.pack(fill="both", padx=10, pady=(0,10))
        self.text_raw.insert("1.0", "\n".join(self.raw_genre_list))
        self.text_raw.configure(state="disabled")
        ttk.Button(
            win,
            text="Copy Raw List",
            command=lambda: self.clipboard_append("\n".join(self.raw_genre_list)),
        ).pack(anchor="e", padx=10, pady=(0,10))

        # 3) Mapping JSON Input
        tk.Label(win, text="Paste JSON Mapping Here:").pack(anchor="w", padx=10)
        self.text_map = ScrolledText(win, width=50, height=10)
        self.text_map.pack(fill="both", padx=10, pady=(0,10))
        # pre-load existing mapping
        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        self.text_map.insert("1.0", json.dumps(existing, indent=2))

        # Buttons: Apply Mapping & Close
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(0,10))
        ttk.Button(btn_frame, text="Apply Mapping", command=self.apply_mapping).pack(side="right")
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side="right", padx=(0,5))

    def apply_mapping(self):
        """Persist mapping JSON from text box and apply normalization."""
        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                existing_map = json.load(f)
        except Exception:
            existing_map = {}

        raw = self.text_map.get("1.0", "end").strip()
        try:
            new_map = json.loads(raw)
        except json.JSONDecodeError as e:
            messagebox.showerror("Invalid JSON", str(e))
            return

        conflicts = [k for k, v in new_map.items() if k in existing_map and existing_map[k] != v]
        if conflicts:
            msg = (
                "You’re about to change existing mappings for:\n\n"
                + "\n".join(conflicts)
                + "\n\nClick “Yes” to override or “No” to cancel."
            )
            if not messagebox.askyesno("Overwrite Detected", msg):
                return

        os.makedirs(os.path.dirname(self.mapping_path), exist_ok=True)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(new_map, f, indent=2)
        self.genre_mapping = new_map

        for rec in getattr(self, "all_records", []):
            rec.old_genres = normalize_genres(rec.old_genres, self.genre_mapping)
            rec.new_genres = normalize_genres(rec.new_genres, self.genre_mapping)

        if hasattr(self, "filtered_records") and hasattr(self, "_prop_tv"):
            self._render_table(self.filtered_records)

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
