import os
import threading
import sys
import logging

if sys.platform == "win32":
    try:
        # For Windows 8.1+
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # SYSTEM_DPI_AWARE
    except Exception:
        try:
            # Fallback for older Windows
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import tkinter as tk
from tkinter import ttk

Style = ttk.Style  # Default built-in themes
# Optional theme packs:
# from ttkthemes import ThemedStyle as Style
# import ttkbootstrap as tb; Style = tb.Style
import json
import queue
import subprocess
from tkinter import filedialog, messagebox, Text, Scrollbar
from unsorted_popup import UnsortedPopup
from tkinter.scrolledtext import ScrolledText
import textwrap
import time

from validator import validate_soundvault_structure
from music_indexer_api import (
    run_full_indexer,
    find_duplicates as api_find_duplicates,
    get_tags,
)
import simple_duplicate_finder as sdf_mod
import fingerprint_cache
import chromaprint_utils
from controllers.library_index_controller import generate_index
from controllers.import_controller import import_new_files
from controllers.genre_list_controller import list_unique_genres
from controllers.genre_playlist_controller import (
    group_tracks_by_genre,
    write_genre_playlists,
)
from controllers.highlight_controller import play_snippet, PYDUB_AVAILABLE
from controllers.scan_progress_controller import ScanProgressController
from gui.audio_preview import PreviewPlayer
from io import BytesIO
from PIL import Image, ImageTk
from mutagen import File as MutagenFile
from tag_fixer import MIN_INTERACTIVE_SCORE, FileRecord
from typing import Callable, List
from indexer_control import cancel_event, IndexCancelled
import library_sync
import playlist_generator
import crash_watcher

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
from plugins.assistant_plugin import AssistantPlugin
from controllers.cluster_controller import cluster_library
from config import load_config, save_config, DEFAULT_FP_THRESHOLDS
from playlist_engine import bucket_by_tempo_energy, more_like_this, autodj_playlist
from playlist_generator import write_playlist
from controllers.cluster_controller import gather_tracks

FilterFn = Callable[[FileRecord], bool]
_cached_filters = None


def make_filters(ex_no_diff: bool, ex_skipped: bool, show_all: bool) -> List[FilterFn]:
    global _cached_filters
    key = (ex_no_diff, ex_skipped, show_all)
    if _cached_filters and _cached_filters[0] == key:
        return _cached_filters[1]

    fns: List[FilterFn] = []
    if not show_all:
        fns.append(lambda r: r.status != "applied")
        if ex_no_diff:
            fns.append(lambda r: r.status != "no_diff")
        if ex_skipped:
            fns.append(lambda r: r.status != "skipped")

    _cached_filters = (key, fns)
    return fns


def apply_filters(
    records: List[FileRecord], filters: List[FilterFn]
) -> List[FileRecord]:
    for fn in filters:
        records = [r for r in records if fn(r)]
    return records


def create_panel_for_plugin(app, name: str, parent: tk.Widget) -> ttk.Frame:
    """Return a UI panel for the given playlist plugin."""
    frame = ttk.Frame(parent)

    try:
        from cluster_graph_panel import ClusterGraphPanel
    except ModuleNotFoundError as exc:
        ttk.Label(
            frame,
            text=(
                "Missing dependency for interactive clustering.\n"
                "Install requirements with `pip install -r requirements.txt`."
            ),
            justify="center",
        ).pack(padx=10, pady=10)
        app._log(f"\u26a0 {exc}")
        return frame

    cluster_data = getattr(app, "cluster_data", None)
    cluster_cfg = getattr(app, "cluster_params", None)
    if cluster_data is None:
        tracks = features = None
    else:
        tracks, features = cluster_data

    if name == "Interactive – KMeans":
        from sklearn.cluster import KMeans

        def km_func(X, p):
            return KMeans(n_clusters=p["n_clusters"]).fit_predict(X)

        n_clusters = 5
        if cluster_cfg and cluster_cfg.get("method") == "kmeans":
            n_clusters = int(cluster_cfg.get("num", 5))
        engine = "librosa"
        if cluster_cfg and "engine" in cluster_cfg:
            engine = cluster_cfg["engine"]
        params = {"n_clusters": n_clusters, "method": "kmeans", "engine": engine}
    elif name == "Interactive – HDBSCAN":
        from hdbscan import HDBSCAN

        def km_func(X, p):
            kwargs = {"min_cluster_size": p["min_cluster_size"]}
            if "min_samples" in p:
                kwargs["min_samples"] = p["min_samples"]
            if "cluster_selection_epsilon" in p:
                kwargs["cluster_selection_epsilon"] = p["cluster_selection_epsilon"]
            return HDBSCAN(**kwargs).fit_predict(X)

        min_cs = 25
        extras = {}
        if cluster_cfg and cluster_cfg.get("method") == "hdbscan":
            min_cs = int(cluster_cfg.get("min_cluster_size", cluster_cfg.get("num", 25)))
            if "min_samples" in cluster_cfg:
                extras["min_samples"] = int(cluster_cfg["min_samples"])
            if "cluster_selection_epsilon" in cluster_cfg:
                extras["cluster_selection_epsilon"] = float(
                    cluster_cfg["cluster_selection_epsilon"]
                )
        engine = "librosa"
        if cluster_cfg and "engine" in cluster_cfg:
            engine = cluster_cfg["engine"]
        params = {
            "min_cluster_size": min_cs,
            "method": "hdbscan",
            "engine": engine,
            **extras,
        }
    elif name == "Sort by Genre":
        use_mapping_var = tk.BooleanVar(value=True)
        split_multi_var = tk.BooleanVar(value=True)
        include_unknown_var = tk.BooleanVar(value=True)

        opts = ttk.Frame(frame)
        opts.pack(fill="x", padx=10, pady=10)
        ttk.Checkbutton(
            opts,
            text="Normalize with genre mapping",
            variable=use_mapping_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            opts,
            text="Split multiple genres (;,/)",
            variable=split_multi_var,
        ).pack(anchor="w", pady=(5, 0))
        ttk.Checkbutton(
            opts,
            text="Include Unknown playlist",
            variable=include_unknown_var,
        ).pack(anchor="w", pady=(5, 0))

        def _run():
            path = app.require_library()
            if not path:
                return

            app.show_log_tab()
            tracks = gather_tracks(path, getattr(app, "folder_filter", None))
            if not tracks:
                messagebox.showinfo("No Tracks", "No audio files found in the library.")
                return

            mapping = app.genre_mapping if use_mapping_var.get() else {}
            grouped = group_tracks_by_genre(
                tracks,
                mapping=mapping,
                include_unknown=include_unknown_var.get(),
                split_multi=split_multi_var.get(),
                log_callback=app._log,
            )
            if not grouped:
                messagebox.showinfo(
                    "No Genres",
                    "No genre tags found. Update your tags or adjust the options.",
                )
                return

            playlists_dir = os.path.join(path, "Playlists")
            write_genre_playlists(grouped, playlists_dir, log_callback=app._log)
            messagebox.showinfo(
                "Playlists",
                f"Wrote {len(grouped)} genre playlists to {playlists_dir}",
            )

        ttk.Button(frame, text="Generate Playlists", command=_run).pack(
            padx=10, pady=10
        )
        return frame
    elif name == "Tempo/Energy Buckets":
        def _run():
            path = app.require_library()
            if not path:
                return
            app.show_log_tab()
            tracks = gather_tracks(path)
            try:
                bucket_by_tempo_energy(tracks, path, app._log)
                messagebox.showinfo("Buckets", "Tempo/Energy playlists written")
            except Exception as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(frame, text="Generate Buckets", command=_run).pack(padx=10, pady=10)
        return frame
    elif name == "More Like This":
        sel = tk.StringVar()

        def browse():
            f = filedialog.askopenfilename()
            if f:
                sel.set(f)

        def generate():
            path = app.require_library()
            if not path or not sel.get():
                messagebox.showerror("Error", "Select a library and track")
                return
            app.show_log_tab()
            tracks = gather_tracks(path)
            res = more_like_this(sel.get(), tracks, 10, log_callback=app._log)
            outfile = os.path.join(path, "Playlists", "more_like_this.m3u")
            write_playlist(res, outfile)
            messagebox.showinfo("Playlist", f"Written to {outfile}")

        tk.Entry(frame, textvariable=sel, width=40).pack(side="left", padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=browse).pack(side="left")
        ttk.Button(frame, text="Generate", command=generate).pack(side="left", padx=5)
        return frame
    elif name == "Auto-DJ":
        sel = tk.StringVar()
        count_var = tk.StringVar(value="20")

        def browse():
            f = filedialog.askopenfilename()
            if f:
                sel.set(f)

        def generate():
            path = app.require_library()
            if not path or not sel.get():
                messagebox.showerror("Error", "Select a library and track")
                return
            try:
                n = int(count_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid count")
                return
            app.show_log_tab()
            tracks = gather_tracks(path)
            order = autodj_playlist(sel.get(), tracks, n, log_callback=app._log)
            outfile = os.path.join(path, "Playlists", "autodj.m3u")
            write_playlist(order, outfile)
            messagebox.showinfo("Playlist", f"Written to {outfile}")

        row = ttk.Frame(frame)
        row.pack(padx=10, pady=10)
        tk.Entry(row, textvariable=sel, width=40).pack(side="left")
        ttk.Button(row, text="Browse", command=browse).pack(side="left", padx=5)
        ttk.Entry(row, textvariable=count_var, width=5).pack(side="left")
        ttk.Label(row, text="songs").pack(side="left")
        ttk.Button(row, text="Generate", command=generate).pack(side="left", padx=5)
        return frame
    else:
        ttk.Label(frame, text=f"{name} panel coming soon…").pack(padx=10, pady=10)
        return frame

    if tracks is None:
        msg = ttk.Frame(frame)
        msg.pack(padx=10, pady=10)
        ttk.Label(msg, text="Run clustering once first").pack()
        ttk.Button(
            msg,
            text="Run Clustering",
            command=lambda: app.cluster_playlists_dialog(params["method"]),
        ).pack(pady=(5, 0))
        return frame

    # Container ensures the graph resizes while keeping controls visible
    container = ttk.Frame(frame)
    container.pack(fill="both", expand=True)
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)

    panel = ClusterGraphPanel(
        container,
        tracks,
        features,
        cluster_func=km_func,
        cluster_params=params,
        library_path=app.library_path,
        log_callback=app._log,
    )
    panel.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

    ttk.Separator(container, orient="horizontal").grid(row=1, column=0, sticky="ew")

    btn_frame = ttk.Frame(container)
    btn_frame.grid(row=2, column=0, sticky="ew", pady=5)

    panel.lasso_var = tk.BooleanVar(value=False)

    lasso_btn = ttk.Checkbutton(
        btn_frame,
        text="Lasso Mode",
        variable=panel.lasso_var,
        command=panel.toggle_lasso,
    )
    lasso_btn.pack(side="left")
    panel.lasso_btn = lasso_btn

    panel.ok_btn = ttk.Button(
        btn_frame,
        text="OK",
        command=panel.finalize_lasso,
        state="disabled",
    )
    panel.ok_btn.pack(side="left", padx=(5, 0))

    panel.gen_btn = ttk.Button(
        btn_frame,
        text="Generate Playlist",
        command=panel.create_playlist,
        state="disabled",
    )
    panel.gen_btn.pack(side="left", padx=(5, 0))

    def _auto_create_all():
        method = panel.cluster_params.get("method")
        params = {
            k: v
            for k, v in panel.cluster_params.items()
            if k not in {"method", "engine"}
        }
        engine = panel.cluster_params.get("engine", "librosa")
        if not params:
            return
        app.show_log_tab()
        threading.Thread(
            target=app._run_cluster_generation,
            args=(app.library_path, method, params, engine),
            daemon=True,
        ).start()

    auto_btn = ttk.Button(btn_frame, text="Auto-Create", command=_auto_create_all)
    auto_btn.pack(side="left", padx=(5, 0))

    if name == "Interactive – HDBSCAN":
        redo_btn = ttk.Button(
            btn_frame, text="Redo Values", command=panel.open_param_dialog
        )
        redo_btn.pack(side="left", padx=(5, 0))

    # ─── Hover Metadata Panel ────────────────────────────────────────────
    hover_panel = ttk.Frame(panel, relief="solid", borderwidth=1)
    art_lbl = tk.Label(hover_panel)
    art_lbl.pack(side="left")
    text_frame = ttk.Frame(hover_panel)
    text_frame.pack(side="left", padx=5)
    title_lbl = ttk.Label(text_frame, font=("TkDefaultFont", 10, "bold"))
    title_lbl.pack(anchor="w")
    artist_lbl = ttk.Label(text_frame, font=("TkDefaultFont", 8))
    artist_lbl.pack(anchor="w")
    hover_panel.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
    hover_panel.place_forget()

    panel.setup_hover(hover_panel, art_lbl, title_lbl, artist_lbl)

    return frame


class Tooltip:
    """Simple hover tooltip displaying dynamic text."""

    def __init__(self, widget: tk.Widget, text_func: Callable[[], str]):
        self.widget = widget
        self.text_func = text_func
        self.tipwindow: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _=None) -> None:
        text = self.text_func()
        if self.tipwindow or not text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 1
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        ttk.Label(
            tw, text=text, background="#ffffe0", relief="solid", borderwidth=1
        ).pack()

    def _hide(self, _=None) -> None:
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


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


class ScanProgressWindow(tk.Toplevel):
    """Non-modal window showing scan progress and logs."""

    def __init__(self, parent: tk.Widget, cancel_event: threading.Event):
        super().__init__(parent)
        self.title("Scanning…")
        self.cancel_event = cancel_event
        self.resizable(True, True)
        self.transient(parent)

        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(10, 0))
        self.progress.start()

        self.log_widget = ScrolledText(self, width=60, height=15, state="disabled")
        self.log_widget.pack(padx=10, pady=10, fill="both", expand=True)

        btn = ttk.Button(self, text="Cancel", command=self._on_cancel)
        btn.pack(pady=(0, 10))
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_cancel(self) -> None:
        if messagebox.askyesno("Cancel Scan", "Stop scanning?"):
            self.cancel_event.set()

    # Public method used by background threads
    def update_progress(self, kind: str, current: int, total: int, msg: str) -> None:
        self.after(0, lambda: self._do_update(kind, current, total, msg))

    # Actual UI updates executed on main thread
    def _do_update(self, kind: str, current: int, total: int, msg: str) -> None:
        if kind == "walk":
            if self.progress["mode"] != "indeterminate":
                self.progress.config(mode="indeterminate")
                self.progress.start()
            if msg:
                self._append(f"Scanning: {msg}")
        elif kind == "fp_start":
            if self.progress["mode"] != "determinate":
                self.progress.stop()
                self.progress.config(mode="determinate", maximum=total, value=current - 1)
            self._append(f"Fingerprinting file {current} of {total}\n{msg}")
        elif kind == "fp_end":
            if self.progress["mode"] == "determinate":
                self.progress["value"] = current
        elif kind == "log":
            self._append(msg)
        elif kind == "complete":
            self.progress.stop()
            self._append(f"Completed: {current} duplicate pairs found")

    def _append(self, text: str) -> None:
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

class SoundVaultImporterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        # Auto-detect and apply default DPI scaling
        cfg = load_config()
        width, height = self.winfo_screenwidth(), self.winfo_screenheight()
        default_scale = cfg.get("ui_scale") or (
            1.5 if (width >= 1920 and height >= 1080) else 1.25
        )
        self.current_scale = default_scale
        self.tk.call("tk", "scaling", self.current_scale)
        self.title("SoundVault Importer")
        self.geometry("700x500")

        self.library_path = ""
        self.library_name_var = tk.StringVar(value="No library selected")
        self.library_path_var = tk.StringVar(value="")
        # Folder to run Quality Checker - now always uses library_path
        self.dup_folder_var = tk.StringVar(value="")  # retained for compatibility
        self.library_stats_var = tk.StringVar(value="")
        self.show_all = False
        self.genre_mapping = {}
        self.mapping_path = ""
        self.assistant_plugin = AssistantPlugin()

        # Cached tracks and feature vectors for interactive clustering
        self.cluster_data = None
        self.folder_filter = {"include": [], "exclude": []}

        # Library Sync state
        self.sync_debug_var = tk.BooleanVar(value=False)
        self.sync_library_var = tk.StringVar(value="")
        self.sync_incoming_var = tk.StringVar(value="")
        self.sync_auto_var = tk.BooleanVar(value=True)
        self.sync_new = []
        self.sync_existing = []
        self.sync_improved = []

        # ── Tag Fixer state ──
        self.tagfix_folder_var = tk.StringVar(value="")
        self.tagfix_ex_no_diff = tk.BooleanVar(value=False)
        self.tagfix_ex_skipped = tk.BooleanVar(value=False)
        self.tagfix_show_all = tk.BooleanVar(value=False)
        self.tf_apply_artist = tk.BooleanVar(value=True)
        self.tf_apply_title = tk.BooleanVar(value=True)
        self.tf_apply_album = tk.BooleanVar(value=False)
        self.tf_apply_genres = tk.BooleanVar(value=False)
        self.tagfix_db_path = ""
        self.tagfix_debug_var = tk.BooleanVar(value=False)

        self.dup_debug_var = tk.BooleanVar(value=False)

        # Distance threshold for duplicate scanning
        dup_thr = cfg.get("duplicate_threshold", 0.03)
        self.fp_threshold_var = tk.DoubleVar(value=dup_thr)
        dup_pref = cfg.get("duplicate_prefix_len", sdf_mod.FP_PREFIX_LEN)
        self.fp_prefix_var = tk.IntVar(value=dup_pref)

        # Quality Checker state
        self._dup_logging = False
        self._preview_thread = None
        self.preview_player = PreviewPlayer(
            on_done=lambda: self.after(0, self._preview_finished_ui)
        )

        # assume ffmpeg is available without performing checks
        self.ffmpeg_available = True

        # theme and scale variables used across rebuilds
        self.style = Style(self)
        default_theme = cfg.get("ui_theme") or self.style.theme_use()
        self.style.theme_use(default_theme)
        self.theme_var = tk.StringVar(value=default_theme)
        self.scale_var = tk.StringVar(value=str(self.current_scale))

        self._configure_treeview_style()

        # Build initial UI
        self.build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_treeview_style(self):
        """Adjust Treeview row height based on current UI scale."""
        base = 20
        self.style.configure("Treeview", rowheight=int(base * self.current_scale))

    def build_ui(self):
        """Create all menus, frames, and widgets."""
        # Theme selector setup
        themes = self.style.theme_names()

        # ─── Menu Bar ───────────────────────────────────────────────────────
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open Library…", command=self.select_library)
        file_menu.add_command(label="Validate Library", command=self.validate_library)
        file_menu.add_command(label="Import New Songs", command=self.import_songs)
        file_menu.add_command(label="Scan for Orphans", command=self.scan_orphans)
        file_menu.add_command(label="Compare Libraries", command=self.compare_libraries)
        file_menu.add_command(label="Show All Files", command=self._on_show_all)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(
            label="Metadata Services…", command=self.open_metadata_settings
        )
        menubar.add_cascade(label="Settings", menu=settings_menu)

        tools_menu = tk.Menu(menubar, tearoff=False)
        tools_menu.add_command(
            label="Regenerate Playlists", command=self.regenerate_playlists
        )
        tools_menu.add_command(label="Fix Tags via AcoustID", command=self.fix_tags_gui)
        tools_menu.add_command(
            label="Generate Library Index…",
            command=lambda: generate_index(self.require_library()),
        )
        tools_menu.add_command(
            label="List Unique Genres…",
            command=lambda: list_unique_genres(self.require_library()),
        )
        if PYDUB_AVAILABLE and self.ffmpeg_available:
            tools_menu.add_command(
                label="Play Highlight…", command=self.sample_song_highlight
            )
        else:
            tools_menu.add_command(
                label="Play Highlight… (requires pydub & ffmpeg)", state="disabled"
            )
        tools_menu.add_separator()
        tools_menu.add_command(
            label="Genre Normalizer", command=self._open_genre_normalizer
        )
        tools_menu.add_command(label="Reset Tag-Fix Log", command=self.reset_tagfix_log)
        cluster_menu = tk.Menu(tools_menu, tearoff=False)
        cluster_menu.add_command(
            label="K-Means…",
            command=lambda: self.cluster_playlists_dialog("kmeans"),
        )
        cluster_menu.add_command(
            label="HDBSCAN…",
            command=lambda: self.cluster_playlists_dialog("hdbscan"),
        )
        tools_menu.add_cascade(label="Clustered Playlists", menu=cluster_menu)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        debug_menu = tk.Menu(menubar, tearoff=False)
        debug_menu.add_command(
            label="Enable Verbose Logging",
            command=lambda: logging.getLogger().setLevel(logging.DEBUG),
        )
        menubar.add_cascade(label="Debug", menu=debug_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(
            label="View Crash Log…", command=self._view_crash_log
        )
        menubar.add_cascade(label="Help", menu=help_menu)

        # ─── Library Info ───────────────────────────────────────────────────
        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 0))
        tk.Button(top, text="Choose Library…", command=self.select_library).pack(
            side="left"
        )
        tk.Label(top, textvariable=self.library_name_var, anchor="w").pack(
            side="left", padx=(5, 0)
        )
        cb = ttk.Combobox(
            top,
            textvariable=self.theme_var,
            values=themes,
            state="readonly",
            width=20,
        )
        cb.pack(side="right", padx=5)
        cb.bind("<<ComboboxSelected>>", self.on_theme_change)
        scale_choices = ["1.25", "1.5", "1.75", "2.0", "2.25"]
        cb_scale = ttk.Combobox(
            top,
            textvariable=self.scale_var,
            values=scale_choices,
            state="readonly",
            width=5,
        )
        cb_scale.pack(side="right", padx=5)
        cb_scale.bind("<<ComboboxSelected>>", self.on_scale_change)
        tk.Label(self, textvariable=self.library_path_var, anchor="w").pack(
            fill="x", padx=10
        )
        tk.Label(self, textvariable=self.library_stats_var, justify="left").pack(
            anchor="w", padx=10, pady=(0, 10)
        )

        # ─── Output + Help Notebook ─────────────────────────────────────────
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Log")

        self.output = tk.Text(self.log_tab, wrap="word", state="disabled", height=15)
        self.output.pack(fill="both", expand=True)

        # ─── Indexer Tab ──────────────────────────────────────────────────
        self.indexer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.indexer_tab, text="Indexer")

        # (Library path now selected via Choose Library… in the main toolbar)
        options = ttk.Frame(self.indexer_tab)
        options.grid(row=0, column=0, columnspan=2, sticky="w")

        self.dry_run_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Dry Run", variable=self.dry_run_var).pack(
            side="left"
        )

        self.phase_c_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options, text="Enable Cross-Album Scan (Phase 3)", variable=self.phase_c_var
        ).pack(side="left", padx=(5, 0))

        self.flush_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options, text="Flush Cache", variable=self.flush_cache_var
        ).pack(side="left", padx=(5, 0))

        self.playlists_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options, text="Create Playlists", variable=self.playlists_var
        ).pack(side="left", padx=(5, 0))

        ttk.Label(options, text="Max Workers:").pack(side="left", padx=(5, 0))
        self.worker_var = tk.StringVar(value="")
        ttk.Entry(options, textvariable=self.worker_var, width=4).pack(side="left")

        self.start_indexer_btn = ttk.Button(
            self.indexer_tab, text="Start Indexer", command=self.run_indexer
        )
        self.start_indexer_btn.grid(row=1, column=0, pady=10)
        self.open_not_sorted_btn = ttk.Button(
            self.indexer_tab,
            text="Open 'Not Sorted' Folder",
            command=self.open_not_sorted_folder,
        )
        self.open_not_sorted_btn.grid(row=1, column=1, pady=10)
        self.cancel_indexer_btn = ttk.Button(
            self.indexer_tab,
            text="Cancel",
            command=self.cancel_indexer,
            state="disabled",
        )
        self.cancel_indexer_btn.grid(row=1, column=2, pady=10)

        progress_frame = ttk.Frame(self.indexer_tab)
        progress_frame.grid(row=0, column=2, rowspan=3, padx=10, sticky="nsew")
        progress_frame.columnconfigure(1, weight=1)

        ttk.Label(progress_frame, text="Phase A").grid(row=0, column=0, sticky="e")
        self.phase_a_bar = ttk.Progressbar(
            progress_frame, length=160, mode="determinate", orient="horizontal"
        )
        self.phase_a_bar.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(progress_frame, text="Phase B").grid(row=1, column=0, sticky="e")
        self.phase_b_bar = ttk.Progressbar(
            progress_frame, length=160, mode="determinate", orient="horizontal"
        )
        self.phase_b_bar.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(progress_frame, text="Phase C").grid(row=2, column=0, sticky="e")
        self.phase_c_bar = ttk.Progressbar(
            progress_frame, length=160, mode="determinate", orient="horizontal"
        )
        self.phase_c_bar.grid(row=2, column=1, sticky="ew", pady=2)

        self.status_var = tk.StringVar(value="")
        self._full_status = ""
        status_frame = ttk.Frame(self.indexer_tab, width=400)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        status_frame.grid_propagate(False)
        self.status_label = ttk.Label(
            status_frame, textvariable=self.status_var, anchor="w"
        )
        self.status_label.pack(fill="x")
        Tooltip(self.status_label, lambda: self._full_status)

        self.log = ScrolledText(self.indexer_tab, height=10, wrap="word")
        self.log.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=5)
        self.indexer_tab.rowconfigure(3, weight=1)
        self.indexer_tab.columnconfigure((0, 1, 2), weight=1)

        # ─── Playlist Creator Tab ───────────────────────────────────────────
        self.playlist_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.playlist_tab, text="Playlist Creator")

        self.plugin_list = tk.Listbox(
            self.playlist_tab, width=30, exportselection=False
        )
        self.plugin_list.grid(row=0, column=0, sticky="ns")
        for name in [
            "Interactive – KMeans",
            "Interactive – HDBSCAN",
            "Sort by Genre",
            "Tempo/Energy Buckets",
            "Metadata",
            "More Like This",
            "Auto-DJ",
        ]:
            self.plugin_list.insert("end", name)
        self.plugin_list.bind("<<ListboxSelect>>", self.on_plugin_select)

        self.plugin_panel = ttk.Frame(self.playlist_tab)
        self.plugin_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.playlist_tab.columnconfigure(1, weight=1)
        self.playlist_tab.rowconfigure(0, weight=1)


        # ─── Tag Fixer Tab ────────────────────────────────────────────────
        self.tagfix_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tagfix_tab, text="Tag Fixer")

        path_frame = ttk.Frame(self.tagfix_tab)
        path_frame.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Entry(path_frame, textvariable=self.tagfix_folder_var).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(path_frame, text="Browse…", command=self._browse_tagfix_folder).pack(
            side="left", padx=(5, 0)
        )
        ttk.Button(path_frame, text="Scan", command=self.fix_tags_gui).pack(
            side="left", padx=(5, 0)
        )

        opts = ttk.Frame(self.tagfix_tab)
        opts.pack(fill="x", padx=10)
        ttk.Checkbutton(
            opts,
            text="Exclude 'no diff'",
            variable=self.tagfix_ex_no_diff,
            command=self._refresh_tagfix_view,
        ).pack(side="left")
        ttk.Checkbutton(
            opts,
            text="Exclude 'skipped'",
            variable=self.tagfix_ex_skipped,
            command=self._refresh_tagfix_view,
        ).pack(side="left", padx=(5, 0))
        ttk.Checkbutton(
            opts,
            text="Show All",
            variable=self.tagfix_show_all,
            command=self._refresh_tagfix_view,
        ).pack(side="left", padx=(5, 0))
        ttk.Checkbutton(
            opts,
            text="Verbose Debug",
            variable=self.tagfix_debug_var,
        ).pack(side="left", padx=(5, 0))

        self.tagfix_progress = ttk.Progressbar(
            self.tagfix_tab, orient="horizontal", mode="determinate"
        )
        self.tagfix_progress.pack(fill="x", padx=10, pady=(5, 5))

        table_container = ttk.Frame(self.tagfix_tab)
        table_container.pack(fill="both", expand=True, padx=10, pady=5)
        vsb = ttk.Scrollbar(table_container, orient="vertical")
        vsb.pack(side="right", fill="y")
        hsb = ttk.Scrollbar(table_container, orient="horizontal")
        hsb.pack(side="bottom", fill="x")

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
            "Suggested Genre",
        )

        self.tagfix_tree = ttk.Treeview(
            table_container,
            columns=cols,
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            selectmode="extended",
        )
        vsb.config(command=self.tagfix_tree.yview)
        hsb.config(command=self.tagfix_tree.xview)
        self.tagfix_tree.pack(fill="both", expand=True)
        self._prop_tv = self.tagfix_tree

        for c in cols:
            self.tagfix_tree.heading(
                c,
                text=c,
                command=lambda _c=c: self._sort_tagfix_column(_c, False),
            )
            width = 100
            if c == "File":
                width = 300
            elif c in ("Old Album", "New Album"):
                width = 120
            elif c in ("Genres", "Suggested Genre"):
                width = 150
            self.tagfix_tree.column(c, width=width, anchor="w")

        self.tagfix_tree.tag_configure("perfect", background="white")
        self.tagfix_tree.tag_configure("changed", background="#fff8c6")
        self.tagfix_tree.tag_configure("lowconf", background="#f8d7da")

        self.tagfix_tree.bind("<Control-a>", lambda e: self._select_all_tagfix())
        self.tagfix_tree.bind("<<TreeviewSelect>>", self._update_tagfix_selection)

        self.tagfix_sel_label = ttk.Label(self.tagfix_tab, text="Selected: 0")
        self.tagfix_sel_label.pack(anchor="w", padx=10)

        apply_frame = ttk.Frame(self.tagfix_tab)
        apply_frame.pack(fill="x", pady=(0, 10), padx=10)
        for var, label in (
            (self.tf_apply_artist, "Artist"),
            (self.tf_apply_title, "Title"),
            (self.tf_apply_album, "Album"),
            (self.tf_apply_genres, "Genres"),
        ):
            ttk.Checkbutton(apply_frame, text=label, variable=var).pack(
                side="left", padx=5
            )
        ttk.Button(
            apply_frame, text="Apply Selected", command=self._apply_selected_tags
        ).pack(side="right")

        # ─── Quality Checker Tab ─────────────────────────────────────────
        self.dup_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dup_tab, text="Quality Checker")

        df_controls = ttk.Frame(self.dup_tab)
        df_controls.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(df_controls, textvariable=self.library_path_var).pack(side="left")
        self.scan_btn = ttk.Button(
            df_controls,
            text="Scan",
            command=self.scan_duplicates,
            state="disabled",
        )
        self.scan_btn.pack(side="left", padx=(5, 0))
        ttk.Label(df_controls, text="Distance Threshold:").pack(side="left", padx=(10, 0))
        thr_entry = ttk.Entry(df_controls, textvariable=self.fp_threshold_var, width=5)
        thr_entry.pack(side="left")
        ttk.Label(df_controls, text="Prefix Length:").pack(side="left", padx=(10, 0))
        pref_entry = ttk.Entry(df_controls, textvariable=self.fp_prefix_var, width=4)
        pref_entry.pack(side="left")
        self.fp_threshold_var.trace_add("write", lambda *a: self._validate_threshold())
        self.fp_prefix_var.trace_add("write", lambda *a: self._validate_threshold())
        ttk.Checkbutton(
            df_controls,
            text="Verbose Debug",
            variable=self.dup_debug_var,
        ).pack(side="left", padx=(5, 0))

        self.qc_canvas = tk.Canvas(self.dup_tab)
        self.qc_scroll = ttk.Scrollbar(
            self.dup_tab, orient="vertical", command=self.qc_canvas.yview
        )
        self.qc_canvas.configure(yscrollcommand=self.qc_scroll.set)
        self.qc_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        self.qc_scroll.pack(side="right", fill="y", pady=(0, 10))
        self.qc_inner = ttk.Frame(self.qc_canvas)
        self.qc_canvas.create_window((0, 0), window=self.qc_inner, anchor="nw")
        self.qc_inner.bind(
            "<Configure>",
            lambda e: self.qc_canvas.configure(scrollregion=self.qc_canvas.bbox("all")),
        )
        if self.library_path_var.get():
            self.scan_btn.config(state="normal")
        self._validate_threshold()

        # ─── Library Sync Tab ─────────────────────────────────────────────
        self.sync_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sync_tab, text="Library Sync")

        path_row = ttk.Frame(self.sync_tab)
        path_row.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(path_row, text="Library Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(path_row, textvariable=self.sync_library_var, state="readonly").grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(path_row, text="Browse…", command=self._browse_sync_library).grid(
            row=0, column=2, padx=(5, 0)
        )
        path_row.columnconfigure(1, weight=1)

        inc_row = ttk.Frame(self.sync_tab)
        inc_row.pack(fill="x", padx=10, pady=(0, 5))
        ttk.Label(inc_row, text="Incoming Folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(inc_row, textvariable=self.sync_incoming_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(inc_row, text="Browse…", command=self._browse_sync_incoming).grid(
            row=0, column=2, padx=(5, 0)
        )
        inc_row.columnconfigure(1, weight=1)

        ttk.Button(self.sync_tab, text="Scan", command=self._scan_library_sync).pack(
            pady=5
        )

        lists = ttk.Frame(self.sync_tab)
        lists.pack(fill="both", expand=True, padx=10, pady=5)
        lists.columnconfigure((0, 1, 2), weight=1)

        ttk.Label(lists, text="New Tracks").grid(row=0, column=0)
        ttk.Label(lists, text="Existing").grid(row=0, column=1)
        ttk.Label(lists, text="Improvement Candidates").grid(row=0, column=2)

        self.sync_new_list = tk.Listbox(lists, selectmode="extended")
        self.sync_new_list.grid(row=1, column=0, sticky="nsew")
        self.sync_existing_list = tk.Listbox(lists, selectmode="extended")
        self.sync_existing_list.grid(row=1, column=1, sticky="nsew")
        self.sync_improved_list = tk.Listbox(lists, selectmode="extended")
        self.sync_improved_list.grid(row=1, column=2, sticky="nsew")

        actions = ttk.Frame(self.sync_tab)
        actions.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Checkbutton(
            actions, text="Auto-Update Playlists", variable=self.sync_auto_var
        ).pack(side="left")
        ttk.Button(actions, text="Copy New", command=self._copy_new_tracks).pack(
            side="left", padx=(5, 0)
        )
        ttk.Button(
            actions, text="Replace Selected", command=self._replace_selected
        ).pack(side="left", padx=(5, 0))

        # after your other tabs
        help_frame = ttk.Frame(self.notebook)
        self.notebook.add(help_frame, text="Help")

        # Chat history display
        self.chat_history = ScrolledText(
            help_frame, height=15, state="disabled", wrap="word"
        )
        self.chat_history.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        # Entry + send button
        entry_frame = ttk.Frame(help_frame)
        entry_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.chat_input = ttk.Entry(entry_frame)
        self.chat_input.pack(side="left", fill="x", expand=True)
        send_btn = ttk.Button(entry_frame, text="Send", command=self._send_help_query)
        send_btn.pack(side="right", padx=(5, 0))

    def on_theme_change(self, event=None):
        """Apply the selected theme and persist to config."""
        theme = self.theme_var.get()
        self.style.theme_use(theme)
        cfg = load_config()
        cfg["ui_theme"] = theme
        save_config(cfg)

    def on_scale_change(self, event=None):
        """Rebuild UI under the new scaling factor."""
        try:
            scale = float(self.scale_var.get())
            self.tk.call("tk", "scaling", scale)
            self.current_scale = scale
        except ValueError:
            return
        cfg = load_config()
        cfg["ui_scale"] = scale
        save_config(cfg)
        self._configure_treeview_style()
        for widget in self.winfo_children():
            widget.destroy()
        self.build_ui()

    def on_plugin_select(self, event):
        """Swap in the UI panel for the selected playlist plugin."""
        for w in self.plugin_panel.winfo_children():
            w.destroy()
        try:
            sel = self.plugin_list.get(self.plugin_list.curselection())
        except tk.TclError:
            return
        panel = create_panel_for_plugin(self, sel, parent=self.plugin_panel)
        if panel:
            panel.pack(fill="both", expand=True)

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
        cfg = load_config()
        cfg["library_root"] = chosen
        save_config(cfg)

        info = open_library(chosen)
        self.library_path = info["path"]
        self.library_name_var.set(info["name"])
        self.library_path_var.set(info["path"])
        self.mapping_path = os.path.join(self.library_path, ".genre_mapping.json")
        self._load_genre_mapping()
        # Clear any cached clustering data when switching libraries
        self.cluster_data = None
        if hasattr(self, "scan_btn"):
            self.scan_btn.config(state="normal")
            self._validate_threshold()
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

    def _set_status(self, text: str) -> None:
        self._full_status = text
        short = textwrap.shorten(text, width=60, placeholder="…")
        self.status_var.set(short)

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
        vault = filedialog.askdirectory(
            title="Select SoundVault Root", initialdir=initial
        )
        if not vault:
            return

        save_last_path(vault)

        is_valid, errors = validate_soundvault_structure(vault)
        if not is_valid:
            messagebox.showerror("Invalid SoundVault", "\n".join(errors))
            self._log(f"✘ Invalid SoundVault: {vault}\n" + "\n".join(errors))
            return

        import_folder = filedialog.askdirectory(
            title="Select Folder of New Songs", initialdir=vault
        )
        if not import_folder:
            return

        dry_run = messagebox.askyesno("Dry Run?", "Perform a dry-run preview only?")
        estimate = messagebox.askyesno(
            "Estimate BPM?", "Attempt BPM estimation for missing values?"
        )

        stop_on_error = False
        if not dry_run:
            stop_on_error = messagebox.askyesno(
                "Stop on Error?",
                "Stop the import if any file move fails?",
            )

        def log_line(msg: str) -> None:
            self.after(0, lambda m=msg: self._log(m))

        def task() -> None:
            try:
                summary = import_new_files(
                    vault,
                    import_folder,
                    dry_run=dry_run,
                    estimate_bpm=estimate,
                    log_callback=log_line,
                    stop_on_error=stop_on_error,
                )

                def ui_complete() -> None:
                    if summary["dry_run"]:
                        messagebox.showinfo(
                            "Dry Run Complete",
                            f"Preview written to:\n{summary['html']}",
                        )
                    else:
                        moved = summary.get("moved", 0)
                        base_msg = (
                            f"Imported {moved} files. Preview:\n{summary['html']}"
                        )
                        if summary.get("errors"):
                            err_text = "\n".join(summary["errors"])
                            full_msg = f"{base_msg}\n\nErrors:\n{err_text}"
                            if messagebox.askretrycancel(
                                "Import Complete (with errors)", full_msg
                            ):
                                threading.Thread(target=task, daemon=True).start()
                                return
                        else:
                            messagebox.showinfo("Import Complete", base_msg)

                    if summary.get("errors"):
                        for err in summary["errors"]:
                            self._log(f"! {err}")

                    self._log(
                        f"✓ Import finished for {import_folder} → {vault}. Dry run: {dry_run}. BPM: {estimate}."
                    )

                self.after(0, ui_complete)
            except Exception as e:

                def ui_err() -> None:
                    messagebox.showerror("Import failed", str(e))
                    self._log(f"✘ Import failed for {import_folder}: {e}")

                self.after(0, ui_err)

        threading.Thread(target=task, daemon=True).start()

    # ── Quality Checker Actions ────────────────────────────────────────
    def scan_duplicates(self):
        folder = self.library_path_var.get()
        if not folder:
            messagebox.showwarning("No Folder", "Please select a library first.")
            return

        debug_enabled = self.dup_debug_var.get()
        sdf_mod.verbose = debug_enabled
        fingerprint_cache.verbose = debug_enabled
        chromaprint_utils.verbose = debug_enabled
        if debug_enabled:
            self._open_dup_debug_window()

        self.clear_quality_view()
        self.scan_btn.config(state="disabled")
        try:
            thr = float(self.fp_threshold_var.get())
        except Exception:
            thr = load_config().get("duplicate_threshold", 0.03)
            self._log("! Invalid threshold; using saved value")
        else:
            if not (0.0 < thr <= 1.0):
                thr = load_config().get("duplicate_threshold", 0.03)
                self._log("! Threshold out of range; using saved value")
        cfg = load_config()
        cfg["duplicate_threshold"] = thr
        try:
            pref_val = int(self.fp_prefix_var.get())
        except Exception:
            pref_val = sdf_mod.FP_PREFIX_LEN
            self._log("! Invalid prefix length; using saved value")
        cfg["duplicate_prefix_len"] = pref_val
        save_config(cfg)
        controller = ScanProgressController()
        prog_win = ScanProgressWindow(self, controller.cancel_event)
        controller.set_callback(prog_win.update_progress)

        def task():
            self._dup_logging = True

            def cb(msg):
                self.after(0, lambda m=msg: self._log(m))
                controller.update("log", 0, 0, msg)

            try:
                dups, missing = sdf_mod.find_duplicates(
                    folder,
                    threshold=thr,
                    prefix_len=pref_val,
                    log_callback=cb,
                    progress_callback=controller.update,
                    cancel_event=controller.cancel_event,
                )
                self.after(0, lambda: self._log(f"Found {len(dups)} duplicate pairs"))
                self.after(0, lambda: self.populate_quality_table(dups))
                if missing:
                    msg = f"{missing} files could not be fingerprinted."
                    self.after(0, lambda m=msg: self._log(m))
            finally:
                self._dup_logging = False
                self.after(0, lambda: (self.scan_btn.config(state="normal"), prog_win.destroy()))

        threading.Thread(target=task, daemon=True).start()

    def run_indexer(self):
        path = self.require_library()
        if not path:
            return

        dry_run = self.dry_run_var.get()

        docs_dir = os.path.join(path, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        output_html = os.path.join(docs_dir, "MusicIndex.html")

        for bar in (self.phase_a_bar, self.phase_b_bar, self.phase_c_bar):
            bar.config(mode="determinate", maximum=1, value=0)
            bar.stop()
        self._set_status("")
        self.log.delete("1.0", "end")
        self.start_indexer_btn["state"] = "disabled"
        self.open_not_sorted_btn["state"] = "disabled"
        self.cancel_indexer_btn["state"] = "normal"
        cancel_event.clear()

        def log_line(msg):
            def ui():
                self.log.insert("end", msg + "\n")
                self.log.see("end")
                self._set_status(msg)

            self.after(0, ui)

        def progress(idx, total, path_, phase="A"):
            if cancel_event.is_set():
                raise IndexCancelled()

            def ui():
                bar = {
                    "A": self.phase_a_bar,
                    "B": self.phase_b_bar,
                    "C": self.phase_c_bar,
                }.get(phase, self.phase_a_bar)

                if total:
                    if bar["mode"] != "determinate":
                        bar.stop()
                        bar.config(mode="determinate")
                    bar["maximum"] = total
                    bar["value"] = idx
                else:
                    if bar["mode"] != "indeterminate":
                        bar.config(mode="indeterminate", maximum=100)
                        bar.start(10)

                self._set_status(path_)
                self.log.insert("end", f"[{idx}/{total}] {path_}\n")
                self.log.see("end")

            self.after(0, ui)

        def task():
            try:
                dups = api_find_duplicates(path, log_callback=log_line)
                if dups:
                    proceed = []
                    ev = threading.Event()

                    def ask():
                        proceed.append(self._confirm_duplicates(dups))
                        ev.set()

                    self.after(0, ask)
                    ev.wait()
                    if not proceed[0]:
                        log_line("✗ Operation cancelled by user")
                        return

                not_sorted = os.path.join(path, "Not Sorted")
                os.makedirs(not_sorted, exist_ok=True)
                ev = threading.Event()

                def show_popup():
                    dlg = UnsortedPopup(self, not_sorted)
                    dlg.bind(
                        "<Destroy>", lambda e: ev.set() if e.widget is dlg else None
                    )

                self.after(0, show_popup)
                ev.wait()

                workers = self.worker_var.get().strip()
                mw = int(workers) if workers else None
                summary = run_full_indexer(
                    path,
                    output_html,
                    dry_run_only=dry_run,
                    log_callback=log_line,
                    progress_callback=progress,
                    enable_phase_c=self.phase_c_var.get(),
                    flush_cache=self.flush_cache_var.get(),
                    max_workers=mw,
                    create_playlists=self.playlists_var.get(),
                )
                self.after(0, lambda: self.log.insert("end", "✔ Indexing complete\n"))
                if dry_run:
                    self.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Dry Run Complete", f"Preview written to:\n{output_html}"
                        ),
                    )
                else:
                    self.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Indexing Complete",
                            f"Moved/renamed {summary.get('moved', 0)} files.",
                        ),
                    )
                self.after(
                    0,
                    lambda: self._log(
                        f"✓ Run Indexer finished for {path}. Dry run: {dry_run}."
                    ),
                )
            except IndexCancelled:
                self.after(0, lambda: self.log.insert("end", "✘ Indexing cancelled\n"))
            except Exception:
                import traceback

                err_msg = traceback.format_exc().strip()
                self.after(
                    0, lambda m=err_msg: messagebox.showerror("Indexing failed", m)
                )
                self.after(
                    0,
                    lambda m=err_msg: self._log(
                        f"✘ Run Indexer failed for {path}:\n{m}"
                    ),
                )
            finally:
                self.after(0, self.update_library_info)
                self.after(0, lambda: self.start_indexer_btn.config(state="normal"))
                self.after(0, lambda: self.open_not_sorted_btn.config(state="normal"))
                self.after(0, lambda: self.cancel_indexer_btn.config(state="disabled"))
                for bar in (self.phase_a_bar, self.phase_b_bar, self.phase_c_bar):
                    self.after(0, bar.stop)
                self.after(0, lambda: self._set_status(""))

        threading.Thread(target=task, daemon=True).start()

    def open_not_sorted_folder(self):
        path = self.require_library()
        if not path:
            return
        not_sorted = os.path.join(path, "Not Sorted")
        os.makedirs(not_sorted, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(not_sorted)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", not_sorted])
            else:
                subprocess.Popen(["xdg-open", not_sorted])
        except Exception as e:
            messagebox.showerror("Open Folder Failed", str(e))

    def cancel_indexer(self) -> None:
        """Signal the background indexer thread to cancel."""
        cancel_event.set()
        self._log("✘ Cancellation requested…")

    def _confirm_duplicates(self, dups):
        """
        Show modal dialog with duplicate pairs.
        Return True to proceed, False to cancel.
        """
        top = tk.Toplevel(self)
        top.title("Confirm Duplicates")
        top.grab_set()

        cols = ("Original", "Duplicate")
        tv = ttk.Treeview(top, columns=cols, show="headings", height=10)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=400, anchor="w")
        for orig, dup in dups:
            tv.insert("", "end", values=(orig, dup))
        tv.pack(fill="both", expand=True, padx=10, pady=10)

        btn_frame = tk.Frame(top)
        btn_frame.pack(fill="x", pady=(0, 10))
        proceed = tk.BooleanVar(value=False)
        tk.Button(
            btn_frame,
            text="Remove Duplicates",
            command=lambda: (proceed.set(True), top.destroy()),
        ).pack(side="left", padx=20)
        tk.Button(
            btn_frame,
            text="Cancel",
            command=lambda: top.destroy(),
        ).pack(side="right", padx=20)
        top.wait_window()
        return proceed.get()

    def scan_orphans(self):
        path = self.require_library()
        if not path:
            return
        messagebox.showinfo(
            "Scan for Orphans", f"[stub] Would scan for orphans in:\n{path}"
        )
        self._log(f"[stub] Scan for Orphans → {path}")
        self.update_library_info()

    def compare_libraries(self):
        master = self.require_library()
        if not master:
            return
        device = filedialog.askdirectory(
            title="Select Device Library Root", initialdir=master
        )
        if not device:
            return
        save_last_path(device)
        messagebox.showinfo(
            "Compare Libraries",
            f"[stub] Would compare:\nMaster: {master}\nDevice: {device}",
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

    def cluster_playlists_dialog(self, method: str = "kmeans"):
        path = self.require_library()
        if not path:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Clustered Playlists")
        dlg.grab_set()
        dlg.resizable(True, True)

        method_var = tk.StringVar(value=method)
        engine_var = tk.StringVar(value="")

        top = ttk.Frame(dlg)
        top.pack(fill="x", padx=10, pady=(10, 0))

        def _update_fields(*args):
            if method_var.get() == "kmeans":
                hdb_frame.pack_forget()
                km_frame.pack(fill="x")
            else:
                km_frame.pack_forget()
                hdb_frame.pack(fill="x")

        rb_km = ttk.Radiobutton(
            top,
            text="KMeans",
            variable=method_var,
            value="kmeans",
            command=_update_fields,
        )
        rb_hdb = ttk.Radiobutton(
            top,
            text="HDBSCAN",
            variable=method_var,
            value="hdbscan",
            command=_update_fields,
        )
        rb_km.pack(side="left")
        rb_hdb.pack(side="left", padx=(5, 0))

        params_frame = ttk.Frame(dlg)
        params_frame.pack(fill="x", padx=10, pady=(5, 0))

        engine_frame = ttk.LabelFrame(dlg, text="Feature Extraction Engine")
        engine_frame.pack(fill="x", padx=10, pady=(10, 0))

        ttk.Radiobutton(
            engine_frame,
            text="Librosa (current)",
            variable=engine_var,
            value="librosa",
        ).pack(anchor="w", padx=5, pady=(2, 0))
        ttk.Radiobutton(
            engine_frame,
            text="Essentia (coming soon)",
            variable=engine_var,
            value="essentia",
            state="disabled",
        ).pack(anchor="w", padx=5, pady=(0, 5))

        # KMeans params
        km_frame = ttk.Frame(params_frame)
        km_var = tk.StringVar(value="5")
        ttk.Label(km_frame, text="Number of clusters:").pack(side="left")
        ttk.Entry(km_frame, textvariable=km_var, width=10).pack(
            side="left", padx=(5, 0)
        )

        # HDBSCAN params
        hdb_frame = ttk.Frame(params_frame)
        min_size_var = tk.StringVar(value="25")
        ttk.Label(hdb_frame, text="Min cluster size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(hdb_frame, textvariable=min_size_var, width=10).grid(
            row=0, column=1, sticky="w", padx=(5, 0)
        )
        ttk.Label(hdb_frame, text="Min samples (optional):").grid(
            row=1, column=0, sticky="w"
        )
        min_samples_var = tk.StringVar(value="")
        ttk.Entry(hdb_frame, textvariable=min_samples_var, width=10).grid(
            row=1, column=1, sticky="w", padx=(5, 0)
        )
        ttk.Label(hdb_frame, text="Epsilon (advanced, optional):").grid(
            row=2, column=0, sticky="w"
        )
        epsilon_var = tk.StringVar(value="")
        ttk.Entry(hdb_frame, textvariable=epsilon_var, width=10).grid(
            row=2, column=1, sticky="w", padx=(5, 0)
        )
        ttk.Label(
            hdb_frame,
            text=(
                "Suggested min cluster size: 5–20 for <500 tracks, 10–50 for "
                "500–2k, 25–150 for 2k–10k, 50–500 for 10k+. Smaller values "
                "find niche moods; larger values find broad, stable clusters."
            ),
            wraplength=360,
            foreground="gray",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(
            hdb_frame,
            text=(
                "Leave min samples empty to match min cluster size (recommended). "
                "Higher values require tighter musical similarity and may mark "
                "more tracks as noise."
            ),
            wraplength=360,
            foreground="gray",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(
            hdb_frame,
            text=(
                "Epsilon rarely helps HDBSCAN. Leave blank unless you need to "
                "merge nearby clusters: 0.01–0.05 is subtle, 0.05–0.2 is "
                "aggressive; values above 0.2 usually ruin results."
            ),
            wraplength=360,
            foreground="gray",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(
            hdb_frame,
            text=(
                "Why HDBSCAN may show no clusters: it only reports clusters when "
                "tracks form dense musical groupings. A smooth stylistic "
                "continuum can appear as all noise."
            ),
            wraplength=360,
            foreground="gray",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 0))

        engine_var.trace_add("write", _update_fields)
        method_var.trace_add("write", _update_fields)
        _update_fields()

        # ─── Folder Filter UI ────────────────────────────────────────────
        music_root = (
            os.path.join(path, "Music")
            if os.path.isdir(os.path.join(path, "Music"))
            else path
        )

        container = ttk.Frame(dlg)
        container.pack(fill="both", expand=True, padx=10, pady=10)
        container.columnconfigure((0, 1), weight=1)
        container.rowconfigure(1, weight=1)

        tree = ttk.Treeview(container, selectmode="extended")
        tree.heading("#0", text="Library Folders")
        tree.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 5))

        def insert_dir(parent: str, p: str):
            node = tree.insert(parent, "end", iid=p, text=os.path.basename(p) or p)
            try:
                for name in sorted(os.listdir(p)):
                    sub = os.path.join(p, name)
                    if os.path.isdir(sub):
                        insert_dir(node, sub)
            except PermissionError:
                pass

        insert_dir("", music_root)

        lists = ttk.Frame(container)
        lists.grid(row=0, column=1, sticky="nsew")
        lists.columnconfigure((0, 1), weight=1)
        lists.rowconfigure(1, weight=1)

        ttk.Label(lists, text="Include List").grid(row=0, column=0)
        ttk.Label(lists, text="Exclude List").grid(row=0, column=1)
        inc_list = tk.Listbox(lists, selectmode="extended")
        inc_list.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        exc_list = tk.Listbox(lists, selectmode="extended")
        exc_list.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        def add_to(lb: tk.Listbox):
            for sel in tree.selection():
                if sel not in lb.get(0, "end"):
                    lb.insert("end", sel)

        def remove_from(lb: tk.Listbox):
            for idx in reversed(lb.curselection()):
                lb.delete(idx)

        btn_inc = ttk.Frame(lists)
        btn_inc.grid(row=2, column=0, pady=(5, 0))
        ttk.Button(btn_inc, text="Add ➕", command=lambda: add_to(inc_list)).pack(
            side="left"
        )
        ttk.Button(
            btn_inc, text="Remove ➖", command=lambda: remove_from(inc_list)
        ).pack(side="left")

        btn_exc = ttk.Frame(lists)
        btn_exc.grid(row=2, column=1, pady=(5, 0))
        ttk.Button(btn_exc, text="Add ➕", command=lambda: add_to(exc_list)).pack(
            side="left"
        )
        ttk.Button(
            btn_exc, text="Remove ➖", command=lambda: remove_from(exc_list)
        ).pack(side="left")

        for p in self.folder_filter.get("include", []):
            inc_list.insert("end", p)
        for p in self.folder_filter.get("exclude", []):
            exc_list.insert("end", p)

        btns = ttk.Frame(dlg)
        btns.pack(pady=(0, 10))

        def _clamp_min_cluster_size(value: int) -> int:
            return max(5, min(value, 500))

        def _clamp_min_samples(value: int, min_cluster_size: int) -> int:
            clamped = max(1, min(value, 20))
            return min(clamped, min_cluster_size)

        def _clamp_epsilon(value: float) -> float:
            return max(0.0, min(value, 0.2))

        def generate():
            self.folder_filter = {
                "include": list(inc_list.get(0, "end")),
                "exclude": list(exc_list.get(0, "end")),
            }

            engine = engine_var.get()
            if not engine:
                messagebox.showerror(
                    "Select Engine", "Please select a feature extraction engine."
                )
                return

            m = method_var.get()
            params = {}
            if m == "kmeans":
                try:
                    params["n_clusters"] = int(km_var.get())
                except ValueError:
                    messagebox.showerror(
                        "Invalid Value", f"{km_var.get()} is not a valid number"
                    )
                    return
            else:
                try:
                    cs_val = int(min_size_var.get())
                except ValueError:
                    messagebox.showerror(
                        "Invalid Value", f"{min_size_var.get()} is not a valid number"
                    )
                    return
                cs_val = _clamp_min_cluster_size(cs_val)
                min_size_var.set(str(cs_val))
                params["min_cluster_size"] = cs_val

                if min_samples_var.get().strip():
                    try:
                        ms_val = int(min_samples_var.get())
                    except ValueError:
                        messagebox.showerror(
                            "Invalid Value",
                            f"{min_samples_var.get()} is not a valid number",
                        )
                        return
                    ms_val = _clamp_min_samples(ms_val, cs_val)
                    min_samples_var.set(str(ms_val))
                    params["min_samples"] = ms_val
                if epsilon_var.get().strip():
                    try:
                        eps_val = float(epsilon_var.get())
                    except ValueError:
                        messagebox.showerror(
                            "Invalid Value",
                            f"{epsilon_var.get()} is not a valid number",
                        )
                        return
                    eps_val = _clamp_epsilon(eps_val)
                    epsilon_var.set(f"{eps_val}")
                    params["cluster_selection_epsilon"] = eps_val
            self._start_cluster_playlists(m, params, engine, dlg)

        ttk.Button(btns, text="Generate", command=generate).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=5)

    def _start_cluster_playlists(self, method: str, params: dict, engine: str, dlg):
        if dlg is not None:
            dlg.destroy()
        path = self.require_library()
        if not path:
            return
        threading.Thread(
            target=self._run_cluster_generation,
            args=(path, method, params, engine),
            daemon=True,
        ).start()

    def _run_cluster_generation(
        self, path: str, method: str, params: dict, engine: str
    ):
        tracks, feats = cluster_library(
            path, method, params, self._log, self.folder_filter, engine
        )
        self.cluster_data = (tracks, feats)
        self.cluster_params = {"method": method, "engine": engine, **params}

        def done():
            messagebox.showinfo("Clustered Playlists", "Generation complete")
            self._refresh_plugin_panel()

        self.after(0, done)

    def _refresh_plugin_panel(self):
        """Rebuild the current plugin panel if a plugin is selected."""
        if not hasattr(self, "plugin_list") or not hasattr(self, "plugin_panel"):
            return
        try:
            sel = self.plugin_list.get(self.plugin_list.curselection())
        except tk.TclError:
            return
        for w in self.plugin_panel.winfo_children():
            w.destroy()
        panel = create_panel_for_plugin(self, sel, parent=self.plugin_panel)
        if panel:
            panel.pack(fill="both", expand=True)

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
        tk.Button(btn, text="Show All Songs", command=on_show_all).pack(
            side="left", padx=5
        )
        dlg.wait_window()
        return result["proceed"], var_no_diff.get(), var_skipped.get(), show_all

    def _on_show_all(self):
        """Run tag-fix scan showing every file regardless of prior log."""
        self.tagfix_show_all.set(True)
        try:
            self.fix_tags_gui()
        finally:
            self.tagfix_show_all.set(False)

    def fix_tags_gui(self):
        folder = self.tagfix_folder_var.get()
        if not folder:
            folder = filedialog.askdirectory(title="Select Folder to Fix Tags")
            if not folder:
                return
            self.tagfix_folder_var.set(folder)

        self.tagfix_db_path, _ = prepare_library(folder)
        self.mapping_path = os.path.join(folder, ".genre_mapping.json")
        self._load_genre_mapping()

        files = discover_files(folder)
        if not files:
            messagebox.showinfo(
                "No audio files", "No supported audio found in that folder."
            )
            return

        self.tagfix_progress["maximum"] = len(files)
        self.tagfix_progress["value"] = 0

        q = queue.Queue()
        show_all = self.tagfix_show_all.get()
        debug_enabled = self.tagfix_debug_var.get()
        if debug_enabled:
            self._open_tagfix_debug_window()

        def worker():
            records = gather_records(
                folder,
                self.tagfix_db_path,
                show_all,
                progress_callback=lambda idx: q.put(idx),
                log_callback=(lambda m: q.put(("log", m))) if debug_enabled else None,
            )
            q.put(("done", records))

        threading.Thread(target=worker, daemon=True).start()

        def poll_queue():
            try:
                while True:
                    item = q.get_nowait()
                    if isinstance(item, tuple):
                        tag, payload = item
                        if tag == "done":
                            records = payload
                            self.all_records = records
                            for rec in self.all_records:
                                rec.old_genres = normalize_genres(
                                    rec.old_genres, self.genre_mapping
                                )
                                rec.new_genres = normalize_genres(
                                    rec.new_genres, self.genre_mapping
                                )
                            self.tagfix_progress["value"] = self.tagfix_progress[
                                "maximum"
                            ]
                            self._refresh_tagfix_view()
                            return
                        elif tag == "log":
                            self._tagfix_debug(str(payload))
                        continue
                    else:
                        self.tagfix_progress["value"] = item
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
            ttk.Checkbutton(chk_frame, text=label, variable=var).pack(
                side="left", padx=5
            )
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
            "Genres",  # existing embedded genres
            "Suggested Genre",  # fetched from MusicBrainz
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
            tv.heading(
                c, text=c, command=lambda _c=c: treeview_sort_column(tv, _c, False)
            )
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
        tk.Button(btn_frame, text="Apply Selection", command=on_apply).pack(
            side="left", padx=5
        )
        tk.Button(btn_frame, text="Cancel", command=dlg.destroy).pack(
            side="left", padx=5
        )
        btn_frame.pack(pady=10)

        update_selection_count()
        dlg.wait_window()
        if getattr(self, "_proceed", False):
            return self._selected
        return None

    # ── Tag Fixer Embedded Helpers ───────────────────────────────────────
    def _browse_tagfix_folder(self):
        initial = self.tagfix_folder_var.get() or load_last_path()
        folder = filedialog.askdirectory(
            title="Select Folder to Fix Tags", initialdir=initial
        )
        if folder:
            save_last_path(folder)
            self.tagfix_folder_var.set(folder)

    # ── Library Quality Helpers ───────────────────────────────────────

    def _browse_dup_folder(self) -> None:
        """Choose folder to scan for duplicates."""
        initial = (
            self.dup_folder_var.get()
            or self.library_path_var.get()
            or load_last_path()
        )
        folder = filedialog.askdirectory(title="Select Folder", initialdir=initial)
        if folder:
            save_last_path(folder)
            self.dup_folder_var.set(folder)

    def _select_all_tagfix(self):
        self.tagfix_tree.selection_set(self.tagfix_tree.get_children(""))
        return "break"

    def _sort_tagfix_column(self, col: str, reverse: bool = False):
        tv = self.tagfix_tree
        data = [(tv.set(k, col), k) for k in tv.get_children("")]
        try:
            data = [(float(v), k) for v, k in data]
        except ValueError:
            pass
        data.sort(reverse=reverse)
        for idx, (_, k) in enumerate(data):
            tv.move(k, "", idx)
        tv.heading(col, command=lambda: self._sort_tagfix_column(col, not reverse))

    def _update_tagfix_selection(self, event=None):
        cnt = len(self.tagfix_tree.selection())
        self.tagfix_sel_label.config(text=f"Selected: {cnt}")

    def _refresh_tagfix_view(self, *_):
        if not hasattr(self, "all_records"):
            return
        filters = make_filters(
            self.tagfix_ex_no_diff.get(),
            self.tagfix_ex_skipped.get(),
            self.tagfix_show_all.get(),
        )
        records = apply_filters(self.all_records, filters)
        records = sorted(
            records,
            key=lambda r: (r.score is not None, r.score if r.score is not None else 0),
            reverse=True,
        )
        self.filtered_records = records
        self._render_table(records)
        self._iid_to_prop = {
            iid: rec for iid, rec in zip(self.tagfix_tree.get_children(""), records)
        }
        self._update_tagfix_selection()

    def _apply_selected_tags(self):
        if not hasattr(self, "all_records") or not self.tagfix_db_path:
            return
        selected = [self._iid_to_prop[iid] for iid in self.tagfix_tree.selection()]
        if not selected:
            messagebox.showinfo("Tag Fixer", "No rows selected.")
            return
        fields = []
        if self.tf_apply_artist.get():
            fields.append("artist")
        if self.tf_apply_title.get():
            fields.append("title")
        if self.tf_apply_album.get():
            fields.append("album")
        if self.tf_apply_genres.get():
            fields.append("genres")
        count = apply_proposals(
            selected,
            self.all_records,
            self.tagfix_db_path,
            fields,
            log_callback=self._log,
        )
        messagebox.showinfo("Tag Fixer", f"Updated {count} files.")
        self._refresh_tagfix_view()

    def _render_table(self, records: List[FileRecord]):
        tv = self._prop_tv
        tv.delete(*tv.get_children())
        for rec in records:
            if rec.status == "unmatched" or (
                rec.score is not None and rec.score < MIN_INTERACTIVE_SCORE
            ):
                tag = "lowconf"
            elif (
                rec.old_artist == rec.new_artist
                and rec.old_title == rec.new_title
                and rec.old_album == rec.new_album
                and sorted(rec.old_genres) == sorted(rec.new_genres)
            ):
                tag = "perfect"
            else:
                tag = "changed"
            tv.insert(
                "",
                "end",
                values=(
                    str(rec.path),
                    f"{rec.score:.2f}" if rec.score is not None else "",
                    rec.old_artist or "",
                    rec.new_artist or "",
                    rec.old_title or "",
                    rec.new_title or "",
                    rec.old_album or "",
                    rec.new_album or "",
                    "; ".join(rec.old_genres),
                    "; ".join(rec.new_genres),
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

    def _open_tagfix_debug_window(self):
        if (
            getattr(self, "tagfix_debug_win", None)
            and self.tagfix_debug_win.winfo_exists()
        ):
            return
        win = tk.Toplevel(self)
        win.title("Tag Fixer Debug")
        text = ScrolledText(win, height=20, wrap="word")
        text.pack(fill="both", expand=True)
        self.tagfix_debug_win = win
        self.tagfix_debug_text = text

    def _tagfix_debug(self, msg: str):
        widget = getattr(self, "tagfix_debug_text", None)
        if widget:
            widget.insert("end", msg + "\n")
            widget.see("end")

    def _open_dup_debug_window(self) -> None:
        if (
            getattr(self, "dup_debug_win", None)
            and self.dup_debug_win.winfo_exists()
        ):
            return
        win = tk.Toplevel(self)
        win.title("Duplicate Finder Debug")
        text = ScrolledText(win, height=20, wrap="word")
        text.pack(fill="both", expand=True)
        self.dup_debug_win = win
        self.dup_text = text

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
        tk.Label(win, text="LLM Prompt Template:").pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.text_prompt = ScrolledText(win, height=8, wrap="word")
        self.text_prompt.pack(fill="both", padx=10)
        self.text_prompt.insert("1.0", PROMPT_TEMPLATE.strip())
        self.text_prompt.configure(state="disabled")
        ttk.Button(
            win,
            text="Copy Prompt",
            command=lambda: self.clipboard_append(PROMPT_TEMPLATE.strip()),
        ).pack(anchor="e", padx=10, pady=(0, 10))

        # 2) Raw Genre List
        tk.Label(win, text="Raw Genre List:").pack(anchor="w", padx=10)
        self.text_raw = ScrolledText(win, width=50, height=15)
        self.text_raw.pack(fill="both", padx=10, pady=(0, 10))
        self.text_raw.insert("1.0", "\n".join(self.raw_genre_list))
        self.text_raw.configure(state="disabled")
        ttk.Button(
            win,
            text="Copy Raw List",
            command=lambda: self.clipboard_append("\n".join(self.raw_genre_list)),
        ).pack(anchor="e", padx=10, pady=(0, 10))

        # 3) Mapping JSON Input
        tk.Label(win, text="Paste JSON Mapping Here:").pack(anchor="w", padx=10)
        self.text_map = ScrolledText(win, width=50, height=10)
        self.text_map.pack(fill="both", padx=10, pady=(0, 10))
        # pre-load existing mapping
        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
        self.text_map.insert("1.0", json.dumps(existing, indent=2))

        # Buttons: Apply Mapping & Close
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Apply Mapping", command=self.apply_mapping).pack(
            side="right"
        )
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(
            side="right", padx=(0, 5)
        )

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

        conflicts = [
            k for k, v in new_map.items() if k in existing_map and existing_map[k] != v
        ]
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

        # Confirm success and close the normalization dialog
        messagebox.showinfo(
            "Mapping Applied",
            "Your genre mapping has been successfully saved and applied to the library.",
        )
        try:
            self.text_map.winfo_toplevel().destroy()
        except Exception:
            pass

    def reset_tagfix_log(self):
        initial = self.library_path or load_last_path()
        folder = filedialog.askdirectory(
            title="Select Library Root", initialdir=initial
        )
        if not folder:
            return
        if not messagebox.askyesno(
            "Reset Log", "This will erase all history of prior scans. Continue?"
        ):
            return
        docs_dir = os.path.join(folder, "Docs")
        db_path = os.path.join(docs_dir, ".soundvault.db")
        if os.path.exists(db_path):
            os.remove(db_path)
        messagebox.showinfo("Reset", "Tag-fix log cleared.")
        self._log(f"Reset tag-fix log for {folder}")

    # ── Library Sync Helpers ───────────────────────────────────────────
    def _browse_sync_library(self):
        initial = self.sync_library_var.get() or load_last_path()
        folder = filedialog.askdirectory(
            title="Select Library Folder", initialdir=initial
        )
        if folder:
            self.sync_library_var.set(folder)

    def _browse_sync_incoming(self):
        initial = self.sync_incoming_var.get() or load_last_path()
        folder = filedialog.askdirectory(
            title="Select Incoming Folder", initialdir=initial
        )
        if folder:
            self.sync_incoming_var.set(folder)

    def _scan_library_sync(self):
        lib = self.sync_library_var.get()
        inc = self.sync_incoming_var.get()
        if not lib or not inc:
            messagebox.showwarning(
                "Scan", "Please choose library and incoming folders."
            )
            return
        db = os.path.join(lib, "Docs", ".soundvault.db")
        cfg = load_config()
        thresholds = cfg.get("format_fp_thresholds", DEFAULT_FP_THRESHOLDS)

        library_sync.set_debug(self.sync_debug_var.get())

        def task():
            try:
                res = library_sync.compare_libraries(
                    lib, inc, db, thresholds=thresholds
                )
                self.sync_new = res["new"]
                self.sync_existing = res["existing"]
                self.sync_improved = res["improved"]
                self.after(0, self._render_sync_results)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Scan Failed", str(e)))

        threading.Thread(target=task, daemon=True).start()

    def _render_sync_results(self):
        self.sync_new_list.delete(0, "end")
        self.sync_existing_list.delete(0, "end")
        self.sync_improved_list.delete(0, "end")
        for p in self.sync_new:
            self.sync_new_list.insert("end", os.path.basename(p))
        for inc, _lib in self.sync_existing:
            self.sync_existing_list.insert("end", os.path.basename(inc))
        for inc, _lib in self.sync_improved:
            self.sync_improved_list.insert("end", os.path.basename(inc))

    def _copy_new_tracks(self):
        idxs = self.sync_new_list.curselection()
        if not idxs:
            return
        sels = [self.sync_new[int(i)] for i in idxs]
        dests = library_sync.copy_new_tracks(
            sels,
            self.sync_incoming_var.get(),
            self.sync_library_var.get(),
        )
        if self.sync_auto_var.get():
            playlist_generator.update_playlists(dests)
        messagebox.showinfo("Copy New", f"Copied {len(dests)} files")

    def _replace_selected(self):
        idxs = self.sync_improved_list.curselection()
        if not idxs:
            return
        sels = [self.sync_improved[int(i)] for i in idxs]
        dests = library_sync.replace_tracks(sels)
        if self.sync_auto_var.get():
            playlist_generator.update_playlists(dests)
        messagebox.showinfo("Replace", f"Replaced {len(dests)} files")

    # ── Quality Checker Helpers ─────────────────────────────────────────
    def clear_quality_view(self) -> None:
        for w in self.qc_inner.winfo_children():
            w.destroy()

    def _validate_threshold(self) -> None:
        """Enable Scan button only when threshold is valid."""
        try:
            val = float(self.fp_threshold_var.get())
            valid = 0.0 < val <= 1.0
        except Exception:
            valid = False
        try:
            pref = int(self.fp_prefix_var.get())
            valid_pref = pref >= 0
        except Exception:
            valid_pref = False
        folder_selected = self.library_path_var.get()
        if valid and valid_pref and folder_selected:
            self.scan_btn.config(state="normal")
        else:
            self.scan_btn.config(state="disabled")

    def _play_preview(self, path: str) -> None:
        """Play an audio preview, ensuring previous playback is cleaned up."""

        # Stop any current preview immediately (no Tk calls inside)
        self.preview_player.stop_preview()

        if not PYDUB_AVAILABLE:
            messagebox.showerror(
                "Playback failed",
                "pydub/ffmpeg not available. Install requirements to enable preview.",
            )
            return

        def task() -> None:
            try:
                self.preview_player.play_preview(path)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Playback failed", str(e)))

        self._preview_thread = threading.Thread(target=task, daemon=True)
        self._preview_thread.start()

    def _preview_finished_ui(self):
        # Placeholder for UI updates when preview completes
        pass

    def _load_thumbnail(self, path: str, size: int = 100) -> ImageTk.PhotoImage:
        img = None
        try:
            audio = MutagenFile(path)
            img_data = None
            if hasattr(audio, "tags") and audio.tags is not None:
                for key in audio.tags.keys():
                    if str(key).startswith("APIC"):
                        img_data = audio.tags[key].data
                        break
            if img_data is None and getattr(audio, "pictures", None):
                pics = getattr(audio, "pictures", [])
                if pics:
                    img_data = pics[0].data
            if img_data:
                img = Image.open(BytesIO(img_data))
        except Exception:
            img = None
        if img is None:
            img = Image.new("RGB", (size, size), "#777777")
        img.thumbnail((size, size))
        return ImageTk.PhotoImage(img)

    def populate_quality_table(self, matches):
        self.clear_quality_view()
        if not matches:
            ttk.Label(self.qc_inner, text="No duplicates detected in this library").pack(pady=20)
            return

        for keep_path, dup_path in matches:
            row = ttk.Frame(self.qc_inner)
            row.pack(fill="x", padx=10, pady=2)

            play_btn = ttk.Button(row, text="▶", width=1, command=lambda p=dup_path: self._play_preview(p))
            play_btn.pack(side="left")

            thumb = self._load_thumbnail(dup_path, size=40)
            img_label = ttk.Label(row, image=thumb)
            img_label.image = thumb
            img_label.pack(side="left", padx=(5, 10))

            tags = get_tags(dup_path)
            title = tags.get("title") or os.path.basename(dup_path)
            artist = tags.get("artist") or "Unknown"
            year = tags.get("year") or "?"
            ext = os.path.splitext(dup_path)[1].lower()
            size = os.path.getsize(dup_path) / 1024 / 1024

            info = ttk.Frame(row)
            info.pack(side="left", fill="x", expand=True)
            ttk.Label(info, text=title, font=("TkDefaultFont", 9, "bold"), anchor="w").pack(anchor="w")
            ttk.Label(info, text=f"{artist} • {year}", font=("TkDefaultFont", 8), anchor="w").pack(anchor="w")
            ttk.Label(info, text=f"{ext[1:].upper()} {size:.1f} MB", font=("TkDefaultFont", 8), anchor="w").pack(anchor="w")

            del_btn = ttk.Button(row, text="Delete", command=lambda p=dup_path, f=row: self._prompt_delete(p, f))
            del_btn.pack(side="right")

        self.qc_canvas.update_idletasks()

    def _prompt_delete(self, path: str, row: tk.Widget) -> None:
        if getattr(self, "_skip_delete_confirm", False):
            proceed = True
        else:
            proceed, skip = self._confirm_delete(path)
            if skip:
                self._skip_delete_confirm = True
        if not proceed:
            return
        try:
            os.remove(path)
            row.destroy()
            self._log(f"Deleted duplicate {path}")
        except Exception as e:
            messagebox.showerror("Delete Failed", str(e))

    def _confirm_delete(self, path: str) -> tuple[bool, bool]:
        top = tk.Toplevel(self)
        top.title("Confirm Delete")
        top.grab_set()
        ttk.Label(top, text=f"Delete {os.path.basename(path)}?").pack(padx=10, pady=(10, 5))
        skip_var = tk.BooleanVar()
        ttk.Checkbutton(top, text="Don't ask again this session", variable=skip_var).pack(padx=10, pady=(0, 5), anchor="w")
        result = {"ok": False}

        def do_ok() -> None:
            result["ok"] = True
            top.destroy()

        def do_cancel() -> None:
            top.destroy()

        btn_frame = ttk.Frame(top)
        btn_frame.pack(pady=(0, 10))
        ttk.Button(btn_frame, text="Delete", command=do_ok).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=do_cancel).pack(side="left", padx=5)
        top.wait_window()
        return result["ok"], skip_var.get()

    def _send_help_query(self):
        threading.Thread(target=self._do_help_query, daemon=True).start()

    def _do_help_query(self):
        user_q = self.chat_input.get().strip()
        if not user_q:
            return

        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", f"You: {user_q}\n")
        self.chat_history.configure(state="disabled")
        self.chat_input.delete(0, "end")

        try:
            reply = self.assistant_plugin.chat(user_q)
        except Exception as err:
            reply = f"[Error initializing or querying model]\n{err}"

        self.after(0, lambda: self._append_help_response(reply))

    def _append_help_response(self, reply):
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", f"Assistant: {reply}\n\n")
        self.chat_history.configure(state="disabled")
        self.chat_history.see("end")

    def open_metadata_settings(self):
        """Open a non-modal window for configuring metadata services."""
        if getattr(self, "_metadata_win", None) and self._metadata_win.winfo_exists():
            self._metadata_win.focus()
            return
        win = tk.Toplevel(self)
        win.title("Metadata Services")
        from plugins.acoustid_plugin import MetadataServiceConfigFrame

        frame = MetadataServiceConfigFrame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        self._metadata_win = win

    def _on_exit(self) -> None:
        """Triggered by File→Exit to mark a clean shutdown."""
        crash_watcher.mark_clean_shutdown()
        self.quit()

    def _on_close(self):
        """Handle application close event."""
        crash_watcher.record_event("WM_DELETE_WINDOW")
        self.preview_player.stop_preview()
        self.destroy()

    def _view_crash_log(self) -> None:
        """Open a window displaying the last 50 lines of the crash log."""
        path = getattr(self, "log_path", "soundvault_crash.log")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()[-50:]
        except FileNotFoundError:
            messagebox.showinfo("Crash Log", "No crash log found.")
            return

        win = tk.Toplevel(self)
        win.title("Crash Log")
        text = ScrolledText(win, width=80, height=24)
        text.pack(fill="both", expand=True)
        text.insert("end", "".join(lines))
        text.configure(state="disabled")

    def show_log_tab(self) -> None:
        """Switch to the Log tab so users can see background activity."""
        if hasattr(self, "log_tab"):
            self.notebook.select(self.log_tab)

    def _log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        line = f"{timestamp} {msg}"
        self.output.configure(state="normal")
        self.output.insert("end", line + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")
        if getattr(self, "_dup_logging", False) and hasattr(self, "dup_text"):
            self.dup_text.configure(state="normal")
            self.dup_text.insert("end", line + "\n")
            self.dup_text.see("end")
            self.dup_text.configure(state="disabled")


if __name__ == "__main__":
    import logging
    from crash_logger import (
        install as install_crash_logger,
        add_context_provider,
    )

    LOG_PATH = "soundvault_crash.log"
    install_crash_logger(log_path=LOG_PATH, level=logging.DEBUG)
    crash_watcher.start()
    app = SoundVaultImporterApp()
    add_context_provider(lambda: {"library_path": app.library_path})
    app.log_path = LOG_PATH
    app.mainloop()
