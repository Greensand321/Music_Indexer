import os
import threading
import sys

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

from validator import validate_soundvault_structure
from music_indexer_api import run_full_indexer, find_duplicates
from controllers.library_index_controller import generate_index
from controllers.import_controller import import_new_files
from controllers.genre_list_controller import list_unique_genres
from controllers.highlight_controller import play_snippet, PYDUB_AVAILABLE
from tag_fixer import MIN_INTERACTIVE_SCORE, FileRecord
from typing import Callable, List
from indexer_control import cancel_event, IndexCancelled
import tidal_sync

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
from config import load_config, save_config

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
        params = {"n_clusters": n_clusters, "method": "kmeans"}
    elif name == "Interactive – HDBSCAN":
        from hdbscan import HDBSCAN

        def km_func(X, p):
            kwargs = {"min_cluster_size": p["min_cluster_size"]}
            if "min_samples" in p:
                kwargs["min_samples"] = p["min_samples"]
            if "cluster_selection_epsilon" in p:
                kwargs["cluster_selection_epsilon"] = p["cluster_selection_epsilon"]
            return HDBSCAN(**kwargs).fit_predict(X)

        min_cs = 5
        extras = {}
        if cluster_cfg and cluster_cfg.get("method") == "hdbscan":
            min_cs = int(cluster_cfg.get("min_cluster_size", cluster_cfg.get("num", 5)))
            if "min_samples" in cluster_cfg:
                extras["min_samples"] = int(cluster_cfg["min_samples"])
            if "cluster_selection_epsilon" in cluster_cfg:
                extras["cluster_selection_epsilon"] = float(cluster_cfg["cluster_selection_epsilon"])
        params = {"min_cluster_size": min_cs, "method": "hdbscan", **extras}
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
        params = {k: v for k, v in panel.cluster_params.items() if k != "method"}
        if not params:
            return
        threading.Thread(
            target=app._run_cluster_generation,
            args=(app.library_path, method, params),
            daemon=True,
        ).start()

    auto_btn = ttk.Button(btn_frame, text="Auto-Create", command=_auto_create_all)
    auto_btn.pack(side="left", padx=(5, 0))

    if name == "Interactive – HDBSCAN":
        redo_btn = ttk.Button(btn_frame, text="Redo Values", command=panel.open_param_dialog)
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
        ttk.Label(tw, text=text, background="#ffffe0", relief="solid", borderwidth=1).pack()

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
        self.library_stats_var = tk.StringVar(value="")
        self.show_all = False
        self.genre_mapping = {}
        self.mapping_path = ""
        self.assistant_plugin = AssistantPlugin()

        # Cached tracks and feature vectors for interactive clustering
        self.cluster_data = None
        self.folder_filter = {"include": [], "exclude": []}

        # tidal-dl sync state
        self.subpar_path_var = tk.StringVar(value="")
        self.downloads_path_var = tk.StringVar(value="")
        self.sync_status_var = tk.StringVar(value="")
        self.subpar_list = []
        self.downloads_list = []
        self.matches = []
        self.fp_threshold_var = tk.DoubleVar(value=0.3)
        self.sync_debug_var = tk.BooleanVar(value=False)

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
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

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

        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Log")

        self.output = tk.Text(log_frame, wrap="word", state="disabled", height=15)
        self.output.pack(fill="both", expand=True)

        # ─── Indexer Tab ──────────────────────────────────────────────────
        self.indexer_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.indexer_tab, text="Indexer")

        # (Library path now selected via Choose Library… in the main toolbar)
        options = ttk.Frame(self.indexer_tab)
        options.grid(row=0, column=0, columnspan=2, sticky="w")

        self.dry_run_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Dry Run", variable=self.dry_run_var).pack(side="left")

        self.phase_c_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="Enable Cross-Album Scan (Phase 3)", variable=self.phase_c_var).pack(side="left", padx=(5,0))

        self.flush_cache_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text="Flush Cache", variable=self.flush_cache_var).pack(side="left", padx=(5,0))

        self.playlists_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options, text="Create Playlists", variable=self.playlists_var).pack(side="left", padx=(5,0))

        ttk.Label(options, text="Max Workers:").pack(side="left", padx=(5,0))
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
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x")
        Tooltip(self.status_label, lambda: self._full_status)

        self.log = ScrolledText(self.indexer_tab, height=10, wrap="word")
        self.log.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=5)
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
            "BPM Range",
            "Metadata",
            "More Like This",
        ]:
            self.plugin_list.insert("end", name)
        self.plugin_list.bind("<<ListboxSelect>>", self.on_plugin_select)

        self.plugin_panel = ttk.Frame(self.playlist_tab)
        self.plugin_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.playlist_tab.columnconfigure(1, weight=1)
        self.playlist_tab.rowconfigure(0, weight=1)

        # ─── Library Quality Tab ───────────────────────────────────────────
        self.quality_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.quality_tab, text="Library Quality")

        ttk.Button(
            self.quality_tab, text="Scan Quality", command=self.scan_quality
        ).pack(pady=5)

        sync = ttk.LabelFrame(self.quality_tab, text="Tidal-dl Sync")
        sync.pack(fill="x", padx=10, pady=10)

        ttk.Button(
            sync, text="Load Subpar List", command=self.load_subpar_list
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(sync, textvariable=self.subpar_path_var).grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(
            sync,
            text="Select Downloads Folder",
            command=self.select_downloads_folder,
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Label(sync, textvariable=self.downloads_path_var).grid(
            row=1, column=1, sticky="w", pady=(5, 0)
        )

        ttk.Label(sync, text="FP Threshold:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Entry(sync, textvariable=self.fp_threshold_var, width=5).grid(row=2, column=1, sticky="w", pady=(5, 0))
        ttk.Checkbutton(
            sync,
            text="Verbose Debug",
            variable=self.sync_debug_var,
        ).grid(row=2, column=2, sticky="w", pady=(5, 0))

        ttk.Button(
            sync, text="Match & Compare", command=self.build_comparison_table
        ).grid(row=3, column=0, sticky="w", pady=(5, 0))
        ttk.Label(sync, textvariable=self.sync_status_var).grid(
            row=3, column=1, sticky="w", pady=(5, 0)
        )

        self.compare_frame = ttk.Frame(self.quality_tab)
        self.compare_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.quality_tab.rowconfigure(1, weight=1)
        self.quality_tab.columnconfigure(0, weight=1)

        ttk.Button(
            self.quality_tab, text="Apply Changes", command=self.apply_replacements
        ).pack(pady=(0, 10))

        # ─── Tag Fixer Tab ────────────────────────────────────────────────
        self.tagfix_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.tagfix_tab, text="Tag Fixer")
        ttk.Button(
            self.tagfix_tab,
            text="Run Tag Fixer…",
            command=self.fix_tags_gui,
        ).pack(pady=10)

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

        info = open_library(chosen)
        self.library_path = info["path"]
        self.library_name_var.set(info["name"])
        self.library_path_var.set(info["path"])
        self.mapping_path = os.path.join(self.library_path, ".genre_mapping.json")
        self._load_genre_mapping()
        # Clear any cached clustering data when switching libraries
        self.cluster_data = None
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
                )

                def ui_complete() -> None:
                    if summary["dry_run"]:
                        messagebox.showinfo(
                            "Dry Run Complete", f"Preview written to:\n{summary['html']}"
                        )
                    else:
                        moved = summary.get("moved", 0)
                        messagebox.showinfo(
                            "Import Complete",
                            f"Imported {moved} files. Preview:\n{summary['html']}",
                        )

                    if summary.get("errors"):
                        self._log("! Some files failed to import. Check log for details.")

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
                dups = find_duplicates(path, log_callback=log_line)
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
                    dlg.bind("<Destroy>", lambda e: ev.set() if e.widget is dlg else None)
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
                self.after(0, lambda m=err_msg: messagebox.showerror("Indexing failed", m))
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

        # KMeans params
        km_frame = ttk.Frame(params_frame)
        km_var = tk.StringVar(value="5")
        ttk.Label(km_frame, text="Number of clusters:").pack(side="left")
        ttk.Entry(km_frame, textvariable=km_var, width=10).pack(
            side="left", padx=(5, 0)
        )

        # HDBSCAN params
        hdb_frame = ttk.Frame(params_frame)
        min_size_var = tk.StringVar(value="5")
        ttk.Label(hdb_frame, text="Min cluster size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(hdb_frame, textvariable=min_size_var, width=10).grid(
            row=0, column=1, sticky="w", padx=(5, 0)
        )
        ttk.Label(hdb_frame, text="Min samples:").grid(row=1, column=0, sticky="w")
        min_samples_var = tk.StringVar(value="")
        ttk.Entry(hdb_frame, textvariable=min_samples_var, width=10).grid(
            row=1, column=1, sticky="w", padx=(5, 0)
        )
        ttk.Label(hdb_frame, text="Epsilon:").grid(row=2, column=0, sticky="w")
        epsilon_var = tk.StringVar(value="")
        ttk.Entry(hdb_frame, textvariable=epsilon_var, width=10).grid(
            row=2, column=1, sticky="w", padx=(5, 0)
        )

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

        def generate():
            self.folder_filter = {
                "include": list(inc_list.get(0, "end")),
                "exclude": list(exc_list.get(0, "end")),
            }

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
                    params["min_cluster_size"] = int(min_size_var.get())
                except ValueError:
                    messagebox.showerror(
                        "Invalid Value", f"{min_size_var.get()} is not a valid number"
                    )
                    return
                if min_samples_var.get().strip():
                    try:
                        params["min_samples"] = int(min_samples_var.get())
                    except ValueError:
                        messagebox.showerror(
                            "Invalid Value",
                            f"{min_samples_var.get()} is not a valid number",
                        )
                        return
                if epsilon_var.get().strip():
                    try:
                        params["cluster_selection_epsilon"] = float(epsilon_var.get())
                    except ValueError:
                        messagebox.showerror(
                            "Invalid Value",
                            f"{epsilon_var.get()} is not a valid number",
                        )
                        return
            self._start_cluster_playlists(m, params, dlg)

        ttk.Button(btns, text="Generate", command=generate).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=5)

    def _start_cluster_playlists(self, method: str, params: dict, dlg):
        if dlg is not None:
            dlg.destroy()
        path = self.require_library()
        if not path:
            return
        threading.Thread(
            target=self._run_cluster_generation,
            args=(path, method, params),
            daemon=True,
        ).start()

    def _run_cluster_generation(self, path: str, method: str, params: dict):
        tracks, feats = cluster_library(
            path, method, params, self._log, self.folder_filter
        )
        self.cluster_data = (tracks, feats)
        self.cluster_params = {"method": method, **params}

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
                            rec.old_genres = normalize_genres(
                                rec.old_genres, self.genre_mapping
                            )
                            rec.new_genres = normalize_genres(
                                rec.new_genres, self.genre_mapping
                            )

                        filters = make_filters(ex_no_diff, ex_skipped, show_all)
                        records = apply_filters(all_records, filters)

                        records = sorted(
                            records,
                            key=lambda r: (
                                r.score is not None,
                                r.score if r.score is not None else 0,
                            ),
                            reverse=True,
                        )
                        self.filtered_records = records

                        if not records:
                            messagebox.showinfo(
                                "No proposals", "No missing tags above threshold."
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
                            messagebox.showinfo("Done", f"Updated {count} files.")
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

    # ─── Tidal-dl Sync Methods ──────────────────────────────────────────

    def scan_quality(self):
        path = self.require_library()
        if not path:
            return
        out = filedialog.asksaveasfilename(
            title="Save Subpar List",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
        )
        if not out:
            return
        count = tidal_sync.scan_library_quality(path, out)
        messagebox.showinfo("Quality Scan", f"Saved {count} entries to {out}")
        self._log(f"Scan Quality written to {out}")

    def load_subpar_list(self):
        path = filedialog.askopenfilename(
            title="Select Subpar List", filetypes=[("Text", "*.txt")]
        )
        if not path:
            return
        self.subpar_path_var.set(path)
        self.subpar_list = tidal_sync.load_subpar_list(path)
        self.sync_status_var.set(f"Loaded {len(self.subpar_list)} subpar tracks.")
        if self.downloads_path_var.get():
            self.build_comparison_table()

    def select_downloads_folder(self):
        folder = filedialog.askdirectory(title="Select tidal-dl Output Folder")
        if not folder:
            return
        self.downloads_path_var.set(folder)
        tidal_sync.set_debug(self.sync_debug_var.get(), self.library_path or ".")
        self.downloads_list = tidal_sync.scan_downloads(folder, log_callback=self._log)
        self.sync_status_var.set(f"Scanned {len(self.downloads_list)} downloaded tracks.")
        if self.subpar_path_var.get():
            self.build_comparison_table()

    def build_comparison_table(self):
        thr = float(self.fp_threshold_var.get() or 0.3)
        tidal_sync.set_debug(self.sync_debug_var.get(), self.library_path or ".")
        self.matches = tidal_sync.match_downloads(
            self.subpar_list,
            self.downloads_list,
            threshold=thr,
            log_callback=self._log,
        )
        num_tag = sum(1 for m in self.matches if m.get("method") == "Tag")
        num_fp = sum(1 for m in self.matches if m.get("method") == "Fingerprint")
        num_none = sum(1 for m in self.matches if m.get("download") is None)
        self.sync_status_var.set(
            f"{num_tag} tag matches, {num_fp} fingerprint matches, {num_none} no matches"
        )
        self._render_comparison_table()

    def _render_comparison_table(self):
        for w in self.compare_frame.winfo_children():
            w.destroy()
        cols = ("download", "method", "confidence", "note")
        tv = ttk.Treeview(
            self.compare_frame,
            columns=cols,
            show="headings",
            selectmode="extended",
        )
        tv.heading("download", text="Downloaded")
        tv.heading("method", text="Match Method")
        tv.heading("confidence", text="Confidence")
        tv.heading("note", text="Note")
        tv.column("download", width=220)
        tv.column("method", width=80, anchor="center")
        tv.column("confidence", width=80, anchor="e")
        tv.column("note", width=150, anchor="w")
        tv.tag_configure("hi", background="#d4edda")
        tv.tag_configure("med", background="#fff8c6")
        tv.tag_configure("low", background="#f8d7da")
        tv.tag_configure("nomatch", background="#f8d7da")
        tv.tag_configure("ambig", background="#ffe5b4")
        tv.tag_configure("error", background="#f8d7da")
        for m in self.matches:
            score = "" if m["score"] is None else f"{m['score']:.2f}"
            method = m.get("method", "")
            note = m.get("note", "") or ""
            if m.get("download") is None:
                tag = "nomatch"
            elif "Ambiguous" in note:
                tag = "ambig"
            elif "Error" in note:
                tag = "error"
            elif m["score"] is not None and m["score"] >= 0.8:
                tag = "hi"
            elif m["score"] is not None and m["score"] >= 0.5:
                tag = "med"
            else:
                tag = "low"
            tv.insert(
                "",
                "end",
                iid=m["original"],
                values=(m.get("download") or "", method, score, note),
                tags=(tag,),
            )
        tv.pack(fill="both", expand=True)
        self.match_tree = tv

    def apply_replacements(self):
        if not hasattr(self, "match_tree"):
            return
        sels = self.match_tree.selection()
        if not sels:
            return
        replaced = 0
        for iid in sels:
            match = next((m for m in self.matches if m["original"] == iid), None)
            if match and match.get("download") and not (
                match.get("note") and "Ambiguous" in match.get("note")
            ) and not (
                match.get("note") and "Error" in match.get("note")
            ):
                try:
                    tidal_sync.replace_file(match["original"], match["download"])
                    replaced += 1
                except Exception as e:
                    self._log(f"Failed to replace {match['original']}: {e}")
        messagebox.showinfo("Apply Changes", f"Replaced {replaced} files")
        self._log(f"Applied {replaced} replacements")

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

    def _log(self, msg):
        self.output.configure(state="normal")
        self.output.insert("end", msg + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")


if __name__ == "__main__":
    app = SoundVaultImporterApp()
    app.mainloop()
