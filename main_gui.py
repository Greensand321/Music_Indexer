import os
import threading
import sys
import logging
import webbrowser

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
from datetime import datetime

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
from controllers.scan_progress_controller import ScanProgressController
from gui.audio_preview import PlaybackError, VlcPreviewPlayer
from io import BytesIO
from PIL import Image, ImageTk
from mutagen import File as MutagenFile
from tag_fixer import MIN_INTERACTIVE_SCORE, FileRecord
from typing import Any, Callable, List
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
from plugins.acoustid_plugin import AcoustIDService, MusicBrainzService
from controllers.cluster_controller import cluster_library
from controllers.cluster_view_controller import ClusterComputationManager
from config import load_config, save_config, DEFAULT_FP_THRESHOLDS
import playlist_engine
from playlist_engine import bucket_by_tempo_energy, autodj_playlist
from controllers.cluster_controller import gather_tracks
from playlist_generator import write_playlist

FilterFn = Callable[[FileRecord], bool]
_cached_filters = None

logger = logging.getLogger(__name__)


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
    cluster_manager = getattr(app, "cluster_manager", None)
    if cluster_data is None:
        tracks = features = None
    else:
        tracks, features = cluster_data
        if cluster_manager is None or getattr(cluster_manager, "tracks", None) != tracks:
            app.cluster_manager = ClusterComputationManager(tracks, features, app._log)
            cluster_manager = app.cluster_manager

    requires_clustering = name in {"Interactive – KMeans", "Interactive – HDBSCAN"}

    if name == "Interactive – KMeans":
        from sklearn.cluster import KMeans

        def km_func(X, p):
            requested = max(1, int(p["n_clusters"]))
            effective = min(requested, len(X))
            if effective != requested:
                logger.info(
                    "Adjusting kmeans clusters from %s to %s due to dataset size",
                    requested,
                    effective,
                )
            return KMeans(n_clusters=effective).fit_predict(X)

        n_clusters = 5
        if cluster_cfg and cluster_cfg.get("method") == "kmeans":
            n_clusters = int(cluster_cfg.get("n_clusters", 5))
        engine = "serial"
        if cluster_cfg and "engine" in cluster_cfg:
            engine = "serial" if cluster_cfg["engine"] == "librosa" else cluster_cfg["engine"]
        params = {"n_clusters": n_clusters, "method": "kmeans", "engine": engine}
        algo_key = "kmeans"
    elif name == "Interactive – HDBSCAN":
        from hdbscan import HDBSCAN

        def km_func(X, p):
            min_cluster_size = max(2, int(p["min_cluster_size"]))
            if min_cluster_size > len(X):
                logger.info(
                    "Adjusting min_cluster_size from %s to %s due to dataset size",
                    min_cluster_size,
                    len(X),
                )
                min_cluster_size = len(X)

            kwargs = {"min_cluster_size": min_cluster_size}
            if "min_samples" in p:
                min_samples = max(1, int(p["min_samples"]))
                if min_samples > len(X):
                    logger.info(
                        "Adjusting min_samples from %s to %s due to dataset size",
                        min_samples,
                        len(X),
                    )
                    min_samples = len(X)
                kwargs["min_samples"] = min_samples
            if "cluster_selection_epsilon" in p:
                kwargs["cluster_selection_epsilon"] = float(
                    p["cluster_selection_epsilon"]
                )
            return HDBSCAN(**kwargs).fit_predict(X)

        min_cs = 5
        extras = {}
        if cluster_cfg and cluster_cfg.get("method") == "hdbscan":
            min_cs = int(
                cluster_cfg.get("min_cluster_size", cluster_cfg.get("n_clusters", 5))
            )
            if "min_samples" in cluster_cfg:
                extras["min_samples"] = int(cluster_cfg["min_samples"])
            if "cluster_selection_epsilon" in cluster_cfg:
                extras["cluster_selection_epsilon"] = float(
                    cluster_cfg["cluster_selection_epsilon"]
                )
        engine = "serial"
        if cluster_cfg and "engine" in cluster_cfg:
            engine = "serial" if cluster_cfg["engine"] == "librosa" else cluster_cfg["engine"]
        params = {
            "min_cluster_size": min_cs,
            "method": "hdbscan",
            "engine": engine,
            **extras,
        }
        algo_key = "hdbscan"
    elif name == "Genre Normalizer":
        lib_var = tk.StringVar(value=app.library_path or "No library selected")
        norm_status = tk.StringVar(value="Select a library to enable normalization.")
        scan_status = tk.StringVar(value="Scan genres to populate the assistant.")
        scanning = tk.BooleanVar(value=False)

        def update_controls():
            lib_var.set(app.library_path or "No library selected")
            norm_status.set(
                os.path.join(app.library_path, ".genre_mapping.json")
                if app.library_path
                else "Select a library to enable normalization."
            )
            scan_btn.config(state="normal" if app.library_path else "disabled")
            apply_btn.config(state="normal" if app.library_path else "disabled")

        intro = ttk.Frame(frame)
        intro.pack(fill="x", padx=10, pady=10)

        ttk.Label(
            intro,
            text=(
                "Normalize genre labels across your library by generating or "
                "updating the .genre_mapping.json file."
            ),
            wraplength=480,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        status_row = ttk.Frame(intro)
        status_row.pack(fill="x", pady=(0, 6))
        ttk.Label(status_row, text="Library:", width=12).pack(side="left")
        ttk.Label(status_row, textvariable=lib_var).pack(
            side="left", fill="x", expand=True
        )

        map_row = ttk.Frame(intro)
        map_row.pack(fill="x", pady=(0, 6))
        ttk.Label(map_row, text="Mapping file:", width=12).pack(side="left")
        ttk.Label(map_row, textvariable=norm_status, wraplength=360).pack(
            side="left", fill="x", expand=True
        )

        control_row = ttk.Frame(intro)
        control_row.pack(fill="x", pady=(4, 0))
        scan_btn = ttk.Button(control_row, text="Scan Genres")
        scan_btn.pack(side="left")
        ttk.Label(control_row, textvariable=scan_status).pack(
            side="left", padx=(8, 0)
        )
        prog = ttk.Progressbar(
            control_row, orient="horizontal", length=160, mode="determinate"
        )
        prog.pack(side="right")

        prompt_box = ttk.LabelFrame(frame, text="LLM Prompt Template")
        prompt_box.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        app.text_prompt = ScrolledText(prompt_box, height=8, wrap="word")
        app.text_prompt.pack(fill="both", expand=True, padx=8, pady=6)
        app.text_prompt.insert("1.0", PROMPT_TEMPLATE.strip())
        app.text_prompt.configure(state="disabled")
        ttk.Button(
            prompt_box,
            text="Copy Prompt",
            command=lambda: app.clipboard_append(PROMPT_TEMPLATE.strip()),
        ).pack(anchor="e", padx=8, pady=(0, 6))

        raw_box = ttk.LabelFrame(frame, text="Raw Genre List")
        raw_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        app.text_raw = ScrolledText(raw_box, width=50, height=12)
        app.text_raw.pack(fill="both", expand=True, padx=8, pady=6)
        app.text_raw.insert("1.0", "Scan the library to populate raw genres.")
        app.text_raw.configure(state="disabled")
        ttk.Button(
            raw_box,
            text="Copy Raw List",
            command=lambda: app.clipboard_append("\n".join(getattr(app, "raw_genre_list", []))),
        ).pack(anchor="e", padx=8, pady=(0, 6))

        map_box = ttk.LabelFrame(frame, text="Paste JSON Mapping Here")
        map_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        app.text_map = ScrolledText(map_box, width=50, height=10)
        app.text_map.pack(fill="both", expand=True, padx=8, pady=6)

        btn_frame = ttk.Frame(map_box)
        btn_frame.pack(fill="x", padx=8, pady=(0, 6))
        apply_btn = ttk.Button(btn_frame, text="Apply Mapping", command=app.apply_mapping)
        apply_btn.pack(side="right")

        def populate_raw_genres(genres: list[str]):
            app.text_raw.configure(state="normal")
            app.text_raw.delete("1.0", "end")
            if genres:
                app.text_raw.insert("1.0", "\n".join(genres))
            else:
                app.text_raw.insert("1.0", "No genres found.")
            app.text_raw.configure(state="disabled")

        def load_existing_mapping():
            if not app.library_path:
                app.text_map.delete("1.0", "end")
                return
            try:
                with open(norm_status.get(), "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
            app.text_map.delete("1.0", "end")
            app.text_map.insert("1.0", json.dumps(existing, indent=2))

        def on_progress(idx, total):
            try:
                prog["value"] = idx
                prog["maximum"] = max(total, 1)
            except Exception:
                pass

        def scan_task(folder: str):
            try:
                app.raw_genre_list = scan_raw_genres(folder, on_progress)
            finally:
                app.after(0, lambda: finish_scan(folder))

        def finish_scan(folder: str):
            scanning.set(False)
            scan_btn.config(state="normal")
            scan_status.set(f"Found {len(getattr(app, 'raw_genre_list', []))} genres.")
            populate_raw_genres(getattr(app, "raw_genre_list", []))
            prog["value"] = prog["maximum"]
            app.mapping_path = os.path.join(folder, ".genre_mapping.json")

        def start_scan():
            folder = app.require_library()
            if not folder or scanning.get():
                return
            files = discover_files(folder)
            if not files:
                messagebox.showinfo("No audio files", "No supported audio found.")
                return
            scanning.set(True)
            scan_status.set("Scanning genres…")
            prog["value"] = 0
            prog["maximum"] = len(files)
            scan_btn.config(state="disabled")
            threading.Thread(target=scan_task, args=(folder,), daemon=True).start()

        scan_btn.config(command=start_scan)

        def refresh_panel():
            update_controls()
            if getattr(app, "raw_genre_list", None):
                populate_raw_genres(app.raw_genre_list)
            load_existing_mapping()

        frame.refresh_cluster_panel = refresh_panel
        refresh_panel()
        return frame
    elif name == "Tempo/Energy Buckets":
        lib_var = tk.StringVar(value=app.library_path or "No library selected")
        dep_status = tk.StringVar()
        progress_var = tk.StringVar(value="Waiting to start")
        running = tk.BooleanVar(value=False)
        total_tracks = 0
        playlists_dir: str | None = None
        bucket_result: dict | None = None
        bucket_tracks: dict[tuple[str, str], list[str]] = {}
        q: queue.Queue = queue.Queue()
        cancel_event = threading.Event()
        engine_var = getattr(app, "feature_engine_var", tk.StringVar(value="librosa"))
        app.feature_engine_var = engine_var

        def open_path(path: str) -> None:
            if not path:
                return
            try:
                if sys.platform == "win32":
                    os.startfile(path)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.run(["open", path], check=False)
                else:
                    subprocess.run(["xdg-open", path], check=False)
            except Exception as exc:  # pragma: no cover - OS interaction
                messagebox.showerror("Open File", f"Could not open {path}: {exc}")

        def open_instructions(_evt=None):
            readme = os.path.abspath("README.md")
            webbrowser.open_new(f"file://{readme}")

        def dependencies_ok() -> bool:
            engine = engine_var.get() or "librosa"
            missing = []
            if playlist_engine.np is None:
                missing.append("numpy")
            if engine == "librosa" and playlist_engine.librosa is None:
                missing.append("librosa")
            if engine == "essentia" and playlist_engine.essentia is None:
                missing.append("Essentia")

            essentia_note = (
                "Essentia unavailable; install Essentia to enable the Essentia engine."
                if playlist_engine.essentia is None
                else "Essentia available."
            )

            if missing:
                dep_status.set(
                    f"Missing dependencies for {engine}: "
                    + ", ".join(missing)
                    + f". {essentia_note}"
                )
                return False

            deps = ["numpy", "Essentia" if engine == "essentia" else "librosa"]
            dep_status.set(
                f"Dependencies ready ({', '.join(deps)}). {essentia_note}"
            )
            return True

        def append_log(msg: str) -> None:
            log_box.configure(state="normal")
            log_box.insert("end", msg + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")

        def prompt_save_bucket(bucket_key: tuple[str, str]):
            tracks = bucket_tracks.get(bucket_key)
            if not tracks:
                messagebox.showinfo(
                    "Save Playlist",
                    "No tracks available for this bucket yet.",
                )
                return

            playlists_root = playlists_dir or app.library_path or os.getcwd()
            os.makedirs(playlists_root, exist_ok=True)
            default_name = f"{bucket_key[0]}_{bucket_key[1]}.m3u"
            chosen = filedialog.asksaveasfilename(
                parent=frame,
                title="Save Bucket Playlist As",
                defaultextension=".m3u",
                initialdir=playlists_root,
                initialfile=default_name,
                filetypes=[("M3U Playlist", "*.m3u"), ("All Files", "*.*")],
            )
            if not chosen:
                return

            try:
                write_playlist(tracks, chosen)
            except Exception as exc:
                messagebox.showerror("Save Playlist", f"Could not save playlist: {exc}")
                return

            messagebox.showinfo("Playlist Saved", f"Playlist saved to {chosen}")
            open_path(os.path.dirname(chosen))

        def preview_bucket(bucket_key: tuple[str, str]) -> None:
            tracks = bucket_tracks.get(bucket_key)
            if not tracks:
                messagebox.showinfo(
                    "Playlist Preview", "No tracks available for this bucket yet."
                )
                return

            app.preview_tracks_in_player(
                tracks, f"{bucket_key[0].title()} / {bucket_key[1].title()}"
            )

        def render_stats(result: dict | None):
            nonlocal bucket_result, bucket_tracks
            bucket_result = result
            bucket_tracks = (result or {}).get("buckets", {}) if isinstance(result, dict) else {}
            stats = (result or {}).get("stats") if isinstance(result, dict) else None
            for child in stats_rows.winfo_children():
                child.destroy()
            if not stats:
                ttk.Label(stats_rows, text="No playlists generated yet.").pack(
                    anchor="w", padx=5, pady=5
                )
                return
            header = ttk.Frame(stats_rows)
            header.pack(fill="x", padx=5, pady=(0, 4))
            ttk.Label(header, text="Tempo", width=10).pack(side="left")
            ttk.Label(header, text="Energy", width=10).pack(side="left")
            ttk.Label(header, text="Tracks", width=8).pack(side="left")
            ttk.Label(header, text="Playlist").pack(side="left", padx=(10, 0))
            for (tb, eb), info in sorted(stats.items()):
                row = ttk.Frame(stats_rows)
                row.pack(fill="x", padx=5, pady=2)
                ttk.Label(row, text=tb.title(), width=10).pack(side="left")
                ttk.Label(row, text=eb.title(), width=10).pack(side="left")
                ttk.Label(row, text=str(info.get("count", 0)), width=8).pack(side="left")
                playlist_path = info.get("playlist", "")
                ttk.Label(row, text=os.path.basename(playlist_path)).pack(
                    side="left", padx=(10, 5)
                )
                ttk.Button(
                    row,
                    text="Open",
                    command=lambda key=(tb, eb): preview_bucket(key),
                ).pack(side="left", padx=(0, 5))
                ttk.Button(
                    row,
                    text="Save As…",
                    command=lambda key=(tb, eb): prompt_save_bucket(key),
                ).pack(side="left")

        def update_controls():
            lib_var.set(app.library_path or "No library selected")
            ready = bool(app.library_path) and dependencies_ok() and not running.get()
            run_btn.config(state="normal" if ready else "disabled")
            cancel_btn.config(state="normal" if running.get() else "disabled")

        engine_var.trace_add("write", lambda *_: update_controls())

        def start_run():
            nonlocal total_tracks, playlists_dir
            if running.get():
                return
            path = app.require_library()
            if not path:
                return
            if not dependencies_ok():
                messagebox.showerror("Missing Dependencies", dep_status.get())
                return
            engine = engine_var.get() or "librosa"
            tracks = gather_tracks(path, getattr(app, "folder_filter", None))
            if not tracks:
                messagebox.showinfo("No Tracks", "No audio files found in the library.")
                return
            total_tracks = len(tracks)
            playlists_dir = os.path.join(path, "Playlists")
            cancel_event.clear()
            running.set(True)
            progress["value"] = 0
            progress["maximum"] = total_tracks
            progress_var.set(f"0/{total_tracks} processed")
            log_box.configure(state="normal")
            log_box.delete("1.0", "end")
            log_box.configure(state="disabled")
            append_log(f"Found {total_tracks} tracks. Generating tempo/energy buckets…")
            render_stats(None)
            update_controls()

            def worker():
                try:
                    result = bucket_by_tempo_energy(
                        tracks,
                        path,
                        log_callback=lambda m: q.put(("log", m)),
                        progress_callback=lambda c: q.put(("progress", c)),
                        cancel_event=cancel_event,
                        engine=engine,
                    )
                    q.put(("done", result))
                except Exception as exc:  # pragma: no cover - UI surface only
                    q.put(("error", str(exc)))

            threading.Thread(target=worker, daemon=True).start()
            poll_queue()

        def poll_queue():
            try:
                while True:
                    tag, payload = q.get_nowait()
                    if tag == "log":
                        append_log(payload)
                    elif tag == "progress":
                        progress["value"] = payload
                        progress_var.set(f"{payload}/{total_tracks} processed")
                    elif tag == "done":
                        running.set(False)
                        render_stats(payload)
                        status = (
                            "Cancelled"
                            if payload.get("cancelled")
                            else "Completed"
                        )
                        progress_var.set(
                            f"{status} – processed {payload.get('processed')} of {payload.get('total')} tracks"
                        )
                        update_controls()
                    elif tag == "error":
                        running.set(False)
                        update_controls()
                        progress_var.set("Error during bucket generation")
                        messagebox.showerror("Bucket Error", payload)
            except queue.Empty:
                pass
            if running.get():
                frame.after(200, poll_queue)

        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)

        content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(content_window, width=e.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        info = ttk.LabelFrame(content, text="Tempo/Energy Buckets")
        info.pack(fill="x", padx=10, pady=(5, 10))
        ttk.Label(info, textvariable=lib_var).pack(anchor="w", padx=5, pady=(5, 0))
        ttk.Label(info, textvariable=dep_status, wraplength=360).pack(
            anchor="w", padx=5, pady=2
        )
        engine_frame = ttk.LabelFrame(info, text="Tempo Engine")
        engine_frame.pack(fill="x", padx=5, pady=(2, 5))
        ttk.Radiobutton(
            engine_frame,
            text="Librosa (default)",
            variable=engine_var,
            value="librosa",
            command=update_controls,
        ).pack(anchor="w", padx=5, pady=(2, 0))
        ttk.Radiobutton(
            engine_frame,
            text="Essentia",
            variable=engine_var,
            value="essentia",
            command=update_controls,
        ).pack(anchor="w", padx=5, pady=(0, 2))
        link = ttk.Label(
            info,
            text="View installation instructions",
            foreground="blue",
            cursor="hand2",
        )
        link.pack(anchor="w", padx=5, pady=(0, 5))
        link.bind("<Button-1>", open_instructions)

        controls = ttk.Frame(content)
        controls.pack(fill="x", padx=10)
        run_btn = ttk.Button(controls, text="Generate Buckets", command=start_run)
        run_btn.pack(side="left")
        cancel_btn = ttk.Button(
            controls, text="Cancel", command=cancel_event.set, state="disabled"
        )
        cancel_btn.pack(side="left", padx=(5, 0))
        ttk.Button(
            controls,
            text="Refresh Library Path",
            command=update_controls,
        ).pack(side="right")

        progress = ttk.Progressbar(content, mode="determinate")
        progress.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(content, textvariable=progress_var).pack(anchor="w", padx=12)

        log_group = ttk.LabelFrame(content, text="Run Log")
        log_group.pack(fill="both", expand=False, padx=10, pady=(10, 0))
        log_box = ScrolledText(log_group, height=8, state="disabled")
        log_box.pack(fill="both", expand=True, padx=5, pady=5)

        stats_group = ttk.LabelFrame(content, text="Bucket Summary")
        stats_group.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        stats_rows = ttk.Frame(stats_group)
        stats_rows.pack(fill="both", expand=True)

        update_controls()
        render_stats(None)
        return frame
    elif name == "Auto-DJ":
        sel = tk.StringVar()
        count_var = tk.StringVar(value="20")
        engine_var = getattr(app, "feature_engine_var", tk.StringVar(value="librosa"))
        app.feature_engine_var = engine_var
        playlist_name_var = tk.StringVar(value="Auto DJ Mix")
        playlist_file_var = tk.StringVar(value="autodj.m3u")
        progress_var = tk.StringVar(value="Waiting to start")
        running = tk.BooleanVar(value=False)
        cancel_event = threading.Event()

        def resolve_engine() -> str:
            engine = engine_var.get()
            cluster_cfg = getattr(app, "cluster_params", {}) or {}
            if not engine:
                engine = cluster_cfg.get("feature_engine") or cluster_cfg.get("engine")
            if engine in (None, "", "serial", "parallel"):
                engine = "librosa"
            return engine

        def open_path(path: str) -> None:
            if not path:
                return
            try:
                if sys.platform == "win32":
                    os.startfile(path)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.run(["open", path], check=False)
                else:
                    subprocess.run(["xdg-open", path], check=False)
            except Exception as exc:  # pragma: no cover - OS interaction
                messagebox.showerror("Open File", f"Could not open {path}: {exc}")

        def browse():
            f = filedialog.askopenfilename()
            if f:
                sel.set(f)

        def append_log(msg: str) -> None:
            def _append() -> None:
                app._log(msg)
                log_box.configure(state="normal")
                log_box.insert("end", msg + "\n")
                log_box.see("end")
                log_box.configure(state="disabled")

            app.after(0, _append)

        def set_progress(value: int, maximum: int, message: str | None = None) -> None:
            def _update() -> None:
                progress["maximum"] = max(1, maximum)
                progress["value"] = value
                if message:
                    progress_var.set(message)

            app.after(0, _update)

        def reset_ui(status: str) -> None:
            running.set(False)
            cancel_event.clear()
            progress["value"] = 0
            progress_var.set(status)
            generate_btn.config(state="normal")
            cancel_btn.config(state="disabled")

        def cancel_generation() -> None:
            cancel_event.set()
            append_log("⚠ Cancelling Auto-DJ run…")
            progress_var.set("Cancelling…")
            cancel_btn.config(state="disabled")

        def generate():
            if running.get():
                return
            path = app.require_library()
            if not path or not sel.get():
                messagebox.showerror("Error", "Select a library and track")
                return
            try:
                n = int(count_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid count")
                return

            playlist_name = playlist_name_var.get().strip() or "Auto DJ Mix"
            filename = playlist_file_var.get().strip() or "autodj.m3u"
            if not filename.lower().endswith(".m3u"):
                filename += ".m3u"

            app.show_log_tab()
            running.set(True)
            cancel_event.clear()
            generate_btn.config(state="disabled")
            cancel_btn.config(state="normal")
            progress_var.set("Gathering tracks…")
            log_box.configure(state="normal")
            log_box.delete("1.0", "end")
            log_box.configure(state="disabled")

            def worker() -> None:
                try:
                    append_log("→ Gathering tracks from library…")
                    tracks = gather_tracks(path)
                    if cancel_event.is_set():
                        raise IndexCancelled()
                    if not tracks:
                        raise RuntimeError("No tracks found in library")
                    if sel.get() not in tracks:
                        raise RuntimeError("Selected start track is not in the library")

                    engine = resolve_engine()

                    def progress_cb(current: int, total: int, message: str | None = None):
                        set_progress(current, total, message)

                    append_log(
                        f"→ Generating {playlist_name} with {n} songs using {engine} features"
                    )
                    order = autodj_playlist(
                        sel.get(),
                        tracks,
                        n,
                        log_callback=append_log,
                        engine=engine,
                        progress_callback=progress_cb,
                        cancel_event=cancel_event,
                    )
                    if cancel_event.is_set():
                        raise IndexCancelled()

                    outfile = os.path.join(path, "Playlists", filename)
                    write_playlist(order, outfile)
                    append_log(f"✓ {playlist_name} written to {outfile}")
                    set_progress(len(order), max(len(order), 1), "Complete")
                    app.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Playlist", f"{playlist_name} written to {outfile}"
                        ),
                    )
                    app.after(0, lambda: open_path(os.path.dirname(outfile)))
                except IndexCancelled:
                    append_log("✖ Auto-DJ cancelled.")
                    app.after(0, lambda: progress_var.set("Cancelled"))
                except Exception as exc:
                    append_log(f"✖ Error during Auto-DJ: {exc}")
                    app.after(0, lambda: messagebox.showerror("Playlist", str(exc)))
                finally:
                    app.after(0, lambda: reset_ui("Ready"))

            threading.Thread(target=worker, daemon=True).start()

        row = ttk.LabelFrame(frame, text="Auto-DJ Settings")
        row.pack(fill="x", padx=10, pady=(10, 5))
        path_row = ttk.Frame(row)
        path_row.pack(fill="x", pady=5)
        ttk.Label(path_row, text="Start track:").pack(side="left")
        tk.Entry(path_row, textvariable=sel, width=40).pack(side="left", padx=(5, 0))
        ttk.Button(path_row, text="Browse", command=browse).pack(side="left", padx=5)

        controls_row = ttk.Frame(row)
        controls_row.pack(fill="x", pady=5)
        ttk.Label(controls_row, text="Songs:").pack(side="left")
        ttk.Entry(controls_row, textvariable=count_var, width=5).pack(side="left", padx=(5, 10))
        ttk.Label(controls_row, text="Playlist name:").pack(side="left")
        ttk.Entry(controls_row, textvariable=playlist_name_var, width=20).pack(
            side="left", padx=(5, 10)
        )
        ttk.Label(controls_row, text="File name:").pack(side="left")
        ttk.Entry(controls_row, textvariable=playlist_file_var, width=18).pack(
            side="left", padx=(5, 10)
        )
        ttk.Label(controls_row, text="Engine:").pack(side="left")
        ttk.Combobox(
            controls_row,
            textvariable=engine_var,
            values=["librosa", "essentia"],
            width=10,
            state="readonly",
        ).pack(side="left")

        action_row = ttk.Frame(row)
        action_row.pack(fill="x", pady=(5, 0))
        generate_btn = ttk.Button(action_row, text="Generate", command=generate)
        generate_btn.pack(side="left")
        cancel_btn = ttk.Button(
            action_row, text="Cancel", command=cancel_generation, state="disabled"
        )
        cancel_btn.pack(side="left", padx=5)

        progress = ttk.Progressbar(frame, mode="determinate")
        progress.pack(fill="x", padx=10, pady=(5, 0))
        ttk.Label(frame, textvariable=progress_var).pack(anchor="w", padx=12)

        log_group = ttk.LabelFrame(frame, text="Auto-DJ Log")
        log_group.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        log_box = ScrolledText(log_group, height=8, state="disabled")
        log_box.pack(fill="both", expand=True, padx=5, pady=5)

        reset_ui("Ready")
        return frame
    else:
        ttk.Label(frame, text=f"{name} panel coming soon…").pack(padx=10, pady=10)
        return frame

    # Container ensures the graph resizes while keeping controls visible
    container = ttk.Frame(frame)
    container.pack(fill="both", expand=True)
    container.rowconfigure(0, weight=1)
    container.columnconfigure(0, weight=1)
    container.columnconfigure(1, weight=0)

    graph_area = ttk.Frame(container)
    graph_area.grid(row=0, column=0, sticky="nsew")
    graph_area.rowconfigure(0, weight=1)
    graph_area.columnconfigure(0, weight=1)

    # Keep the controls outside the graph container so they remain visible
    # while the graph refreshes.
    graph_stack = ttk.Frame(graph_area)
    graph_stack.grid(row=0, column=0, sticky="nsew")
    graph_stack.rowconfigure(0, weight=1)
    graph_stack.columnconfigure(0, weight=1)

    panel: ClusterGraphPanel | None = None

    message_var = tk.StringVar(value="")
    banner_var = tk.StringVar(value="")
    press_counter = 0

    def _set_message(text: str) -> None:
        message_var.set(text)

    def _clear_message(event: tk.Event | None = None) -> None:  # type: ignore[name-defined]
        message_var.set("")

    def _message_action(text: str, action: Callable[[], Any]) -> Callable[[], Any]:
        def wrapped() -> Any:
            _set_message(text)
            return action()

        return wrapped

    def _right_panel_action(action: Callable[[], Any] | None) -> Callable[[], Any]:
        def wrapped() -> Any:
            nonlocal press_counter
            press_counter += 1
            banner_var.set(str(press_counter))
            if action is not None:
                return action()

        return wrapped

    frame.bind_all("<Escape>", _clear_message)

    side_tools = ttk.Frame(container)
    side_tools.grid(row=0, column=1, rowspan=3, sticky="ns", padx=(10, 0))
    side_tools.columnconfigure(0, weight=1)
    side_tools.rowconfigure(3, weight=1)

    # Keep the control column stable while the graph updates
    side_tools.update_idletasks()
    container.columnconfigure(1, minsize=side_tools.winfo_reqwidth())

    control_banner = ttk.Frame(side_tools)
    control_banner.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    control_banner.columnconfigure(1, weight=1)

    run_btn = ttk.Button(
        control_banner,
        text="Run Clusters",
        command=_right_panel_action(
            _message_action(
                "Running clustering will refresh your groups based on the current settings.",
                lambda: app.cluster_playlists_dialog(params["method"]),
            )
        ),
    )
    run_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    status = ttk.Label(control_banner)
    status.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    playlist_btn = ttk.Button(
        side_tools,
        text="Current Playlists",
        command=_right_panel_action(
            _message_action(
                "Listing the playlists you already have for quick review.",
                lambda: panel and panel.show_current_playlists(),
            )
        ),
    )
    playlist_btn.grid(row=1, column=0, sticky="ew", pady=(0, 10))

    cluster_box = ttk.LabelFrame(side_tools, text="Cluster Loader")
    cluster_box.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
    cluster_box.columnconfigure(1, weight=1)

    ttk.Label(cluster_box, text="Pick cluster:").grid(
        row=0, column=0, sticky="w", padx=5, pady=(5, 2)
    )
    cluster_select_var = tk.StringVar()
    cluster_combo = ttk.Combobox(
        cluster_box,
        textvariable=cluster_select_var,
        width=10,
    )
    cluster_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=(5, 2))

    ttk.Label(cluster_box, text="Or enter #:").grid(
        row=1, column=0, sticky="w", padx=5, pady=(0, 2)
    )
    manual_cluster_var = tk.StringVar()
    manual_entry = ttk.Entry(
        cluster_box,
        textvariable=manual_cluster_var,
        width=10,
    )
    manual_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=(0, 2))

    def _load_cluster():
        _set_message("Loading the selected cluster and preparing lasso controls.")
        if panel is None:
            return
        choice = manual_cluster_var.get().strip() or cluster_select_var.get().strip()
        if not choice:
            messagebox.showinfo("Clusters", "Choose a cluster first.")
            return
        try:
            cid = int(choice)
        except ValueError:
            messagebox.showinfo("Clusters", f"{choice} is not a valid number")
            return
        panel.load_cluster(cid)

    load_btn = ttk.Button(
        cluster_box,
        text="Load Cluster",
        command=_right_panel_action(_load_cluster),
    )
    load_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 5))

    temp_status_var = tk.StringVar(value="Clusters available: 0")
    ttk.Label(cluster_box, textvariable=temp_status_var).grid(
        row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 5)
    )

    temp_box = ttk.LabelFrame(side_tools, text="Temporary Playlist")
    temp_box.grid(row=3, column=0, sticky="nsew")
    temp_box.columnconfigure(0, weight=1)
    temp_box.rowconfigure(0, weight=1)

    list_frame = ttk.Frame(temp_box)
    list_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    list_frame.columnconfigure(0, weight=1)

    temp_scroll = ttk.Scrollbar(list_frame, orient="vertical")
    temp_listbox = tk.Listbox(
        list_frame,
        height=12,
        selectmode="extended",
        yscrollcommand=temp_scroll.set,
    )
    temp_listbox.pack(side="left", fill="both", expand=True)
    temp_scroll.config(command=temp_listbox.yview)
    temp_scroll.pack(side="right", fill="y")

    show_all_btn = ttk.Button(
        temp_box,
        text="Show All",
        command=_right_panel_action(
            _message_action(
                "Highlighting all songs currently in the temporary playlist.",
                lambda: panel and panel.highlight_temp_playlist(),
            )
        ),
    )
    show_all_btn.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))

    add_highlight_btn = ttk.Button(
        temp_box,
        text="Add Highlighted Songs",
        command=_right_panel_action(
            _message_action(
                "Enable lasso mode to add the highlighted graph selection to the temp playlist.",
                lambda: panel and panel.begin_add_highlight_flow(),
            )
        ),
    )
    add_highlight_btn.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 5))

    temp_remove_btn = ttk.Button(
        temp_box,
        text="Remove Selected",
        command=_right_panel_action(
            _message_action(
                "Removing the selected songs from the temporary playlist.",
                lambda: panel and panel.remove_selected_from_temp(),
            )
        ),
    )
    temp_remove_btn.grid(row=3, column=0, sticky="ew", padx=5, pady=(0, 5))

    create_playlist_btn = ttk.Button(
        temp_box,
        text="Create Playlist",
        command=_right_panel_action(
            _message_action(
                "Saving the current temporary playlist as a new playlist.",
                lambda: panel and panel.create_temp_playlist(),
            )
        ),
    )
    create_playlist_btn.grid(row=4, column=0, sticky="ew", padx=5, pady=(0, 5))

    ttk.Separator(container, orient="horizontal").grid(row=1, column=0, sticky="ew")

    btn_frame = ttk.Frame(container)
    btn_frame.grid(row=2, column=0, sticky="ew", pady=5)

    lasso_var = tk.BooleanVar(value=False)

    lasso_btn = ttk.Checkbutton(
        btn_frame,
        text=(
            "To begin, press the \"Run Clusters\" button, select library folders, "
            "and run. You'll then see the music visually."
        ),
        variable=lasso_var,
        command=_message_action(
            "Toggling lasso mode for manual graph selection.",
            lambda: panel and panel.toggle_lasso(),
        ),
    )
    lasso_btn.pack(side="left")

    placeholder: ttk.Label | None = None

    def _sync_cluster_controls(cluster_ready: bool, running: bool) -> None:
        ready = cluster_ready and panel is not None
        state = "normal" if ready and not running else "disabled"

        for widget in (
            playlist_btn,
            cluster_combo,
            manual_entry,
            load_btn,
            temp_listbox,
            show_all_btn,
            add_highlight_btn,
            temp_remove_btn,
            create_playlist_btn,
            lasso_btn,
        ):
            widget.config(state=state)

        run_btn.state(["disabled"] if running else ["!disabled"])

        if running:
            status.config(text="Clustering in progress…")
        elif not ready:
            status.config(text="Run clustering once first")
        else:
            status.config(text="Clusters loaded")

    def _ensure_placeholder(message: str) -> ttk.Label:
        nonlocal placeholder

        if placeholder is None:
            placeholder = ttk.Label(graph_stack, anchor="center")
            placeholder.grid(row=0, column=0, sticky="nsew")
        placeholder.configure(text=message)
        placeholder.lift()
        return placeholder

    def refresh_cluster_panel():
        nonlocal panel, cluster_manager, cluster_data

        cluster_data = getattr(app, "cluster_data", None)
        cluster_manager = getattr(app, "cluster_manager", None)
        if cluster_data is None:
            tracks_local = features_local = None
        else:
            tracks_local, features_local = cluster_data
            if cluster_manager is None or getattr(cluster_manager, "tracks", None) != tracks_local:
                app.cluster_manager = ClusterComputationManager(
                    tracks_local, features_local, app._log
                )
                cluster_manager = app.cluster_manager

        cluster_generation_running = getattr(app, "cluster_generation_running", False)
        cluster_ready = cluster_manager is not None and cluster_data is not None

        if panel is not None and cluster_manager is not None and panel.cluster_manager != cluster_manager:
            panel.destroy()
            panel = None

        if cluster_generation_running:
            _ensure_placeholder("Clustering in progress…")
            if panel is not None:
                panel.grid()
            _sync_cluster_controls(False, True)
            return

        if not cluster_ready:
            _ensure_placeholder("Run clustering once first")
            if panel is not None:
                panel.grid()
            _sync_cluster_controls(False, False)
            return

        if placeholder is not None:
            placeholder.grid_remove()

        if panel is None:
            X, X2 = cluster_manager.get_projection()

            panel = ClusterGraphPanel(
                graph_stack,
                tracks_local,
                X,
                X2,
                cluster_func=km_func,
                cluster_params=params,
                cluster_manager=cluster_manager,
                algo_key=algo_key,
                library_path=app.library_path,
                log_callback=app._log,
            )

            panel.cluster_select_var = cluster_select_var
            panel.cluster_combo = cluster_combo
            panel.manual_cluster_var = manual_cluster_var
            panel.temp_status_var = temp_status_var
            panel.temp_listbox = temp_listbox
            panel.temp_remove_btn = temp_remove_btn
            panel.lasso_var = lasso_var
            panel.lasso_btn = lasso_btn

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

            def _handle_resize(event):
                if panel is not None:
                    panel.on_resize(event.width, event.height)

            graph_stack.bind("<Configure>", _handle_resize, add="+")
            panel.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
            graph_stack.after_idle(
                lambda: panel.on_resize(graph_stack.winfo_width(), graph_stack.winfo_height())
            )
        else:
            panel.grid()

        # Keep panel state in sync with the latest clustering run
        panel.cluster_params = params
        panel.cluster_manager = cluster_manager

        panel.refresh_control_states()
        panel._refresh_cluster_options()
        _sync_cluster_controls(True, False)

    refresh_cluster_panel()

    frame.refresh_cluster_panel = refresh_cluster_panel  # type: ignore[attr-defined]

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
        self.cluster_manager = None
        self.cluster_generation_running = False
        self.folder_filter = {"include": [], "exclude": []}

        # Shared audio feature/analysis engine selection
        self.feature_engine_var = tk.StringVar(value="librosa")

        # Cached plugin panels to avoid teardown/rebuild churn
        self.plugin_views: dict[str, ttk.Frame] = {}
        self.active_plugin: str | None = None

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
        self.tagfix_api_status = tk.StringVar(value="")

        self.dup_debug_var = tk.BooleanVar(value=False)

        # Distance threshold for duplicate scanning
        dup_thr = cfg.get("duplicate_threshold", 0.03)
        self.fp_threshold_var = tk.DoubleVar(value=dup_thr)
        dup_pref = cfg.get("duplicate_prefix_len", sdf_mod.FP_PREFIX_LEN)
        self.fp_prefix_var = tk.IntVar(value=dup_pref)

        # Quality Checker state
        self._dup_logging = False
        self._preview_thread = None
        self.player_status_var = tk.StringVar(
            value="Select a library to load tracks."
        )
        self.preview_player = VlcPreviewPlayer(
            on_done=lambda: self.after(0, self._preview_finished_ui)
        )
        self.preview_backend_available = self.preview_player.available
        self.preview_backend_error = self.preview_player.availability_error
        if not self.preview_backend_available:
            reason = self.preview_backend_error or "Install python-vlc to enable playback."
            logging.error("Preview backend unavailable: %s", reason)
            self.player_status_var.set(f"Preview disabled: {reason}")
        else:
            logging.info("VLC preview backend initialized")
        self._quality_play_buttons: list[ttk.Button] = []
        self._preview_in_progress = False
        self._player_busy_item: str | None = None
        self._ignore_next_preview_finish = False

        # Player tab state
        self.player_tracks: list[dict[str, str]] = []
        self.player_tree_paths: dict[str, str] = {}
        self.player_tree_rows: dict[str, dict[str, str]] = {}
        self._player_load_thread: threading.Thread | None = None
        self.player_art_image: ImageTk.PhotoImage | None = None
        self.player_search_var = tk.StringVar(value="")
        self.player_temp_playlist: list[str] = []
        self.player_playlist_status_var = tk.StringVar(
            value="No songs in the playlist builder yet."
        )
        self.player_playlist_listbox: tk.Listbox | None = None
        self.player_view_label: str | None = None

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
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(
            label="Metadata Services…", command=self.open_metadata_settings
        )
        menubar.add_cascade(label="Settings", menu=settings_menu)

        tools_menu = tk.Menu(menubar, tearoff=False)
        tools_menu.add_command(label="Fix Tags via AcoustID", command=self.fix_tags_gui)
        tools_menu.add_command(
            label="Generate Library Index…",
            command=lambda: generate_index(self.require_library()),
        )
        tools_menu.add_separator()
        tools_menu.add_command(label="Reset Tag-Fix Log", command=self.reset_tagfix_log)
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
            "Genre Normalizer",
            "Tempo/Energy Buckets",
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
        api_status = ttk.Frame(opts)
        api_status.pack(side="left", padx=(10, 0))
        self.tagfix_api_indicator = tk.Label(api_status, text="●", fg="#d68a00")
        self.tagfix_api_indicator.pack(side="left")
        ttk.Label(api_status, textvariable=self.tagfix_api_status).pack(
            side="left", padx=(4, 0)
        )

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
        self._check_tagfix_api_status()

        # ─── Player Tab ────────────────────────────────────────────────
        self.player_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.player_tab, text="Player")

        player_controls = ttk.Frame(self.player_tab)
        player_controls.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(player_controls, textvariable=self.player_status_var).pack(
            side="left"
        )
        ttk.Label(player_controls, text="Search:").pack(side="left", padx=(10, 4))
        player_search = ttk.Entry(
            player_controls, textvariable=self.player_search_var, width=30
        )
        player_search.pack(side="left")
        self.player_search_var.trace_add("write", lambda *_: self._apply_player_filter())
        self.player_reload_btn = ttk.Button(
            player_controls,
            text="Reload",
            command=self._load_player_library_async,
            state="disabled",
        )
        self.player_reload_btn.pack(side="right")

        player_content = ttk.Frame(self.player_tab)
        player_content.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        player_table = ttk.Frame(player_content)
        player_table.pack(side="left", fill="both", expand=True)
        p_vsb = ttk.Scrollbar(player_table, orient="vertical")
        p_vsb.pack(side="right", fill="y")
        p_hsb = ttk.Scrollbar(player_table, orient="horizontal")
        p_hsb.pack(side="bottom", fill="x")

        cols = ("Title", "Artist", "Album", "Length", "Play")
        self.player_tree = ttk.Treeview(
            player_table,
            columns=cols,
            show="headings",
            yscrollcommand=p_vsb.set,
            xscrollcommand=p_hsb.set,
        )
        p_vsb.config(command=self.player_tree.yview)
        p_hsb.config(command=self.player_tree.xview)
        self.player_tree.pack(fill="both", expand=True)

        widths = {"Title": 220, "Artist": 140, "Album": 160, "Length": 70, "Play": 50}
        for c in cols:
            self.player_tree.heading(c, text=c)
            self.player_tree.column(
                c,
                width=widths.get(c, 100),
                anchor="center" if c in {"Length", "Play"} else "w",
                stretch=c != "Play",
            )

        self.player_tree.bind("<ButtonRelease-1>", self._on_player_tree_click)
        self.player_tree.bind("<<TreeviewSelect>>", self._on_player_selection_change)

        art_panel = ttk.Frame(player_content)
        art_panel.pack(side="right", fill="y", padx=(10, 0))
        self.player_art_caption = ttk.Label(
            art_panel,
            text="Select a track to view album art",
            wraplength=220,
            justify="center",
        )
        self.player_art_caption.pack(fill="x", pady=(0, 6))
        self.player_art_label = ttk.Label(art_panel)
        self.player_art_label.pack(fill="x")
        ttk.Button(
            art_panel, text="Load Playlist", command=self._player_load_playlist
        ).pack(fill="x", pady=(6, 6))
        self._update_player_art(None)

        playlist_box = ttk.LabelFrame(art_panel, text="Playlist Builder")
        playlist_box.pack(fill="both", expand=True, pady=(10, 0))
        ttk.Label(
            playlist_box, textvariable=self.player_playlist_status_var
        ).pack(anchor="w", padx=5, pady=(5, 2))

        pl_list_frame = ttk.Frame(playlist_box)
        pl_list_frame.pack(fill="both", expand=True, padx=5)
        pl_scroll = ttk.Scrollbar(pl_list_frame, orient="vertical")
        self.player_playlist_listbox = tk.Listbox(
            pl_list_frame,
            selectmode="extended",
            height=10,
            yscrollcommand=pl_scroll.set,
        )
        self.player_playlist_listbox.pack(side="left", fill="both", expand=True)
        pl_scroll.config(command=self.player_playlist_listbox.yview)
        pl_scroll.pack(side="right", fill="y")

        btn_row1 = ttk.Frame(playlist_box)
        btn_row1.pack(fill="x", padx=5, pady=(5, 0))
        ttk.Button(
            btn_row1, text="Add Selected", command=self._player_add_selection_to_temp
        ).pack(side="left", expand=True, fill="x")
        ttk.Button(
            btn_row1,
            text="Remove Selected",
            command=self._player_remove_selected_from_temp,
        ).pack(side="left", expand=True, fill="x", padx=(5, 0))

        btn_row2 = ttk.Frame(playlist_box)
        btn_row2.pack(fill="x", padx=5, pady=(5, 5))
        ttk.Button(btn_row2, text="Clear", command=self._player_clear_temp_playlist).pack(
            side="left", fill="x"
        )
        ttk.Button(
            btn_row2, text="Save Playlist", command=self._player_save_temp_playlist
        ).pack(side="left", expand=True, fill="x", padx=(5, 0))
        ttk.Button(
            btn_row2,
            text="Current Playlists",
            command=self._player_show_current_playlists,
        ).pack(side="left", expand=True, fill="x", padx=(5, 0))
        self._sync_player_playlist()

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
        switch_start = time.perf_counter()
        try:
            sel = self.plugin_list.get(self.plugin_list.curselection())
        except tk.TclError:
            return
        logging.info("[perf] tool switch -> %s", sel)

        # Ensure any previously displayed plugin panels are hidden before showing the
        # newly selected tool. This avoids orphaned UIs sticking around when
        # switching between tools (e.g., the Tempo/Energy Buckets view).
        for panel in self.plugin_views.values():
            panel.pack_forget()

        panel = self.plugin_views.get(sel)
        if panel is None:
            panel = create_panel_for_plugin(self, sel, parent=self.plugin_panel)
            if panel:
                self.plugin_views[sel] = panel
        if panel:
            logging.info("[perf] show panel %s", sel)
            panel.pack(fill="both", expand=True)
            self.active_plugin = sel
        duration = (time.perf_counter() - switch_start) * 1000
        logging.info("[perf] tool switch complete in %.1f ms", duration)

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
        if hasattr(self, "player_search_var"):
            self.player_search_var.set("")
        if hasattr(self, "player_playlist_listbox"):
            self._player_set_temp_playlist([])
        # Clear any cached clustering data when switching libraries
        self.cluster_data = None
        self.cluster_manager = None
        self.plugin_views.clear()
        self.active_plugin = None
        if hasattr(self, "scan_btn"):
            self.scan_btn.config(state="normal")
            self._validate_threshold()
        self.update_library_info()
        if hasattr(self, "player_reload_btn"):
            self.player_reload_btn.config(state="normal")
        self._load_player_library_async()

    def update_library_info(self):
        if not self.library_path:
            self.library_stats_var.set("")
            if hasattr(self, "player_reload_btn"):
                self.player_reload_btn.config(state="disabled")
            self.player_status_var.set("Select a library to load tracks.")
            return
        num = count_audio_files(self.library_path)
        is_valid, _ = validate_soundvault_structure(self.library_path)
        status = "Valid" if is_valid else "Invalid"
        self.library_stats_var.set(f"Songs: {num}    Validation: {status}")

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
      
    def cluster_playlists_dialog(self, method: str = "kmeans"):
        path = self.require_library()
        if not path:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Clustered Playlists")
        dlg.grab_set()
        dlg.resizable(True, True)

        cluster_cfg = getattr(self, "cluster_params", {}) or {}
        selected_method = cluster_cfg.get("method", method)

        method_var = tk.StringVar(value=selected_method)
        engine_default = cluster_cfg.get("engine", "serial")
        if engine_default == "librosa":
            engine_default = "serial"
        engine_var = tk.StringVar(value=engine_default)
        use_max_workers_var = tk.BooleanVar(
            value=bool(cluster_cfg.get("use_max_workers", False))
        )

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

        engine_frame = ttk.LabelFrame(dlg, text="Processing Engine")
        engine_frame.pack(fill="x", padx=10, pady=(10, 0))

        ttk.Radiobutton(
            engine_frame,
            text="Standard (single process)",
            variable=engine_var,
            value="serial",
        ).pack(anchor="w", padx=5, pady=(2, 0))
        ttk.Radiobutton(
            engine_frame,
            text="Parallel (multi-core)",
            variable=engine_var,
            value="parallel",
        ).pack(anchor="w", padx=5, pady=(0, 5))

        use_max_workers_chk = ttk.Checkbutton(
            engine_frame,
            text="Aggressive parallelism (use all cores minus one)",
            variable=use_max_workers_var,
        )
        use_max_workers_chk.pack(anchor="w", padx=22, pady=(0, 5))

        def _update_engine_state(*_args):
            if engine_var.get() == "parallel":
                use_max_workers_chk.state(["!disabled"])
            else:
                use_max_workers_chk.state(["disabled"])

        engine_var.trace_add("write", _update_engine_state)
        _update_engine_state()

        # KMeans params
        km_frame = ttk.Frame(params_frame)
        km_var = tk.StringVar(value=str(cluster_cfg.get("n_clusters", 5)))
        ttk.Label(km_frame, text="Number of clusters:").pack(side="left")
        ttk.Entry(km_frame, textvariable=km_var, width=10).pack(
            side="left", padx=(5, 0)
        )
        ttk.Label(
            km_frame,
            text="(try ~5-20 for small libraries, 50-200 for large)",
            foreground="gray",
        ).pack(side="left", padx=(5, 0))

        # HDBSCAN params
        hdb_frame = ttk.Frame(params_frame)
        min_size_var = tk.StringVar(value=str(cluster_cfg.get("min_cluster_size", 5)))
        ttk.Label(hdb_frame, text="Min cluster size:").grid(row=0, column=0, sticky="w")
        ttk.Entry(hdb_frame, textvariable=min_size_var, width=10).grid(
            row=0, column=1, sticky="w", padx=(5, 0)
        )
        ttk.Label(
            hdb_frame,
            text="(e.g., 5-15 for small sets, 30-80 for large)",
            foreground="gray",
        ).grid(row=0, column=2, sticky="w", padx=(5, 0))
        ttk.Label(hdb_frame, text="Min samples:").grid(row=1, column=0, sticky="w")
        min_samples_var = tk.StringVar(
            value=str(cluster_cfg.get("min_samples", ""))
            if "min_samples" in cluster_cfg
            else "1"
        )
        ttk.Entry(hdb_frame, textvariable=min_samples_var, width=10).grid(
            row=1, column=1, sticky="w", padx=(5, 0)
        )
        ttk.Label(
            hdb_frame,
            text="(start with 1-5; increase for stricter clusters)",
            foreground="gray",
        ).grid(row=1, column=2, sticky="w", padx=(5, 0))
        ttk.Label(hdb_frame, text="Epsilon:").grid(row=2, column=0, sticky="w")
        epsilon_var = tk.StringVar(
            value=str(cluster_cfg.get("cluster_selection_epsilon", ""))
            if "cluster_selection_epsilon" in cluster_cfg
            else ""
        )
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

            engine = engine_var.get()
            if not engine:
                messagebox.showerror(
                    "Select Engine", "Please select a processing engine."
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
            self._start_cluster_playlists(
                m, params, engine, dlg, use_max_workers_var.get()
            )

        ttk.Button(btns, text="Generate", command=generate).pack(side="left", padx=5)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="left", padx=5)

    def _start_cluster_playlists(
        self, method: str, params: dict, engine: str, dlg, use_max_workers: bool
    ):
        if dlg is not None:
            dlg.destroy()
        path = self.require_library()
        if not path:
            return
        self.cluster_generation_running = True
        self._refresh_plugin_panel()
        self.show_log_tab()
        threading.Thread(
            target=self._run_cluster_generation,
            args=(path, method, params, engine, use_max_workers),
            daemon=True,
        ).start()

    def _run_cluster_generation(
        self, path: str, method: str, params: dict, engine: str, use_max_workers: bool
    ):
        try:
            tracks, feats = cluster_library(
                path,
                method,
                params,
                self._log,
                self.folder_filter,
                engine,
                use_max_workers=use_max_workers,
            )
            self.cluster_data = (tracks, feats)
            self.cluster_params = {
                "method": method,
                "engine": engine,
                "use_max_workers": use_max_workers,
                **params,
            }
            self.cluster_manager = ClusterComputationManager(tracks, feats, self._log)

            def done():
                for name in ("Interactive – KMeans", "Interactive – HDBSCAN"):
                    if name in self.plugin_views:
                        try:
                            self.plugin_views[name].destroy()
                        except Exception:
                            pass
                        self.plugin_views.pop(name, None)

                self.cluster_generation_running = False
                messagebox.showinfo("Clustered Playlists", "Generation complete")
                self._refresh_plugin_panel()

            self.after(0, done)
        except Exception as exc:
            def fail():
                self.cluster_generation_running = False
                messagebox.showerror("Cluster Generation Failed", str(exc))
                self._log(f"✘ Cluster generation failed: {exc}")
                self._refresh_plugin_panel()

            self.after(0, fail)

    def _refresh_plugin_panel(self):
        """Rebuild the current plugin panel if a plugin is selected."""
        if not hasattr(self, "plugin_list") or not hasattr(self, "plugin_panel"):
            return
        try:
            sel = self.plugin_list.get(self.plugin_list.curselection())
        except tk.TclError:
            return
        if sel in self.plugin_views:
            panel = self.plugin_views[sel]
            refresh = getattr(panel, "refresh_cluster_panel", None)
            if callable(refresh):
                refresh()
                return
            try:
                panel.destroy()
            except Exception:
                pass
            self.plugin_views.pop(sel, None)
        self.active_plugin = None
        self.on_plugin_select(None)

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

    def _check_tagfix_api_status(self) -> None:
        cfg = load_config()
        service = cfg.get("metadata_service", "AcoustID")
        self.tagfix_api_indicator.configure(fg="#d68a00")
        self.tagfix_api_status.set(f"{service}: Checking…")

        def worker() -> None:
            ok, msg = self._test_metadata_service(service)

            def done() -> None:
                if not self.winfo_exists():
                    return
                color = "green" if ok else "red"
                label = "Connected" if ok else (msg or "Unavailable")
                self.tagfix_api_indicator.configure(fg=color)
                self.tagfix_api_status.set(f"{service}: {label}")

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _test_metadata_service(self, service: str) -> tuple[bool, str]:
        try:
            if service == "MusicBrainz":
                return MusicBrainzService().test_connection()
            if service == "AcoustID":
                return AcoustIDService().test_connection()
            return False, "Not supported"
        except Exception as exc:  # pragma: no cover - defensive UI guard
            return False, str(exc)

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

    def apply_mapping(self):
        """Persist mapping JSON from text box and apply normalization."""
        if not hasattr(self, "text_map"):
            messagebox.showwarning(
                "No Mapping Editor", "Open the Genre Normalizer to edit mappings."
            )
            return

        if not getattr(self, "mapping_path", None):
            messagebox.showwarning("No Library", "Select a library before applying.")
            return

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

        # Confirm success and keep the editor open
        messagebox.showinfo(
            "Mapping Applied",
            "Your genre mapping has been successfully saved and applied to the library.",
        )

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

    def _play_preview(
        self,
        path: str,
        start_ms: int = 30000,
        duration_ms: int = 15000,
        player_item: str | None = None,
    ) -> None:
        """Play an audio preview while serializing concurrent requests."""

        if not self.preview_backend_available:
            err = self.preview_backend_error or "VLC preview backend unavailable."
            logging.error("Preview backend unavailable: %s", err)
            if hasattr(self, "player_status_var"):
                self.player_status_var.set(f"Preview disabled: {err}")
            messagebox.showerror("Playback failed", err)
            return

        if self._preview_in_progress:
            # Interrupt any existing playback before starting the new preview
            self._ignore_next_preview_finish = True
            self.preview_player.stop_preview()
            self._preview_in_progress = False

        if hasattr(self, "player_status_var"):
            fname = os.path.basename(path)
            suffix = " (30s highlight)" if duration_ms >= 30000 else ""
            self.player_status_var.set(f"Playing {fname}{suffix}…")

        self._preview_in_progress = True
        self._set_player_play_state_busy(player_item)

        def task() -> None:
            try:
                self.preview_player.play_clip(
                    path, start_ms=start_ms, duration_ms=duration_ms
                )
            except PlaybackError as e:
                logging.exception("Preview playback failed")
                self.after(0, lambda: self._preview_finished_ui(error=str(e)))
            except Exception as e:  # pragma: no cover - safety net
                logging.exception("Unexpected preview failure")
                self.after(0, lambda: self._preview_finished_ui(error=str(e)))

        self._preview_thread = threading.Thread(target=task, daemon=True)
        self._preview_thread.start()

    def _preview_finished_ui(self, error: str | None = None):
        if self._ignore_next_preview_finish:
            # Skip UI reset triggered by an intentionally interrupted preview
            self._ignore_next_preview_finish = False
            return

        self._preview_in_progress = False
        self._restore_player_play_icons()
        self._set_preview_controls_state(enabled=self.preview_backend_available)
        if hasattr(self, "player_status_var"):
            if error:
                self.player_status_var.set("Playback failed. Check logs.")
                messagebox.showerror("Playback failed", error)
            else:
                self.player_status_var.set("Ready to play another track.")

    def _set_preview_controls_state(self, enabled: bool) -> None:
        state = "!disabled" if enabled else "disabled"
        for btn in getattr(self, "_quality_play_buttons", []):
            try:
                btn.state([state])
            except Exception:
                logging.exception("Failed to update preview button state")

    def _set_player_play_state_busy(self, item_id: str | None) -> None:
        if not hasattr(self, "player_tree"):
            return
        self._restore_player_play_icons()
        if not item_id or not self.player_tree.exists(item_id):
            return
        values = list(self.player_tree.item(item_id, "values"))
        if len(values) >= 5:
            values[4] = "⏳"
            self.player_tree.item(item_id, values=values)
            self._player_busy_item = item_id

    def _restore_player_play_icons(self) -> None:
        if not hasattr(self, "player_tree"):
            return
        if self._player_busy_item and self.player_tree.exists(self._player_busy_item):
            values = list(self.player_tree.item(self._player_busy_item, "values"))
            if len(values) >= 5:
                values[4] = "▶" if self.preview_backend_available else "—"
                self.player_tree.item(self._player_busy_item, values=values)
        self._player_busy_item = None

    def _player_art_size(self) -> int:
        base = 200
        scale_factor = 0.8 + (0.2 * max(self.current_scale, 1.0))
        return int(base * min(scale_factor, 1.25))

    def _load_thumbnail(self, path: str | None, size: int = 100) -> ImageTk.PhotoImage:
        img = None
        if path:
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
                candidates = []
                folder = os.path.dirname(path)
                for stem in ("cover", "folder", "front", "album"):
                    for ext in ("jpg", "jpeg", "png", "webp"):
                        candidates.append(os.path.join(folder, f"{stem}.{ext}"))
                for candidate in candidates:
                    if not os.path.exists(candidate):
                        continue
                    try:
                        img = Image.open(candidate)
                        break
                    except Exception:
                        img = None
        if img is None:
            img = Image.new("RGB", (size, size), "#777777")
        img.thumbnail((size, size))
        return ImageTk.PhotoImage(img)

    def _update_player_art(
        self, path: str | None, title: str | None = None, artist: str | None = None
    ) -> None:
        if not hasattr(self, "player_art_label"):
            return
        size = self._player_art_size()
        art = self._load_thumbnail(path, size=size)
        self.player_art_image = art
        self.player_art_label.configure(image=art)
        if path:
            caption = title or os.path.basename(path)
            if artist:
                caption = f"{caption}\n{artist}"
        else:
            caption = "Select a track to view album art"
        self.player_art_caption.configure(text=caption, wraplength=size + 20)

    def _get_selected_player_paths(self) -> list[str]:
        if not hasattr(self, "player_tree"):
            return []
        selection = self.player_tree.selection()
        return [
            self.player_tree_paths[item]
            for item in selection
            if item in self.player_tree_paths
        ]

    def _player_set_temp_playlist(self, tracks: list[str]) -> None:
        self.player_temp_playlist = list(dict.fromkeys(tracks))
        self._sync_player_playlist()

    def _sync_player_playlist(self) -> None:
        if self.player_playlist_listbox is None:
            return
        self.player_playlist_listbox.delete(0, tk.END)
        for path in self.player_temp_playlist:
            self.player_playlist_listbox.insert(tk.END, os.path.basename(path))

        count = len(self.player_temp_playlist)
        suffix = "song" if count == 1 else "songs"
        self.player_playlist_status_var.set(f"Playlist builder items: {count} {suffix}")

    def _player_add_selection_to_temp(self) -> None:
        selected_paths = self._get_selected_player_paths()
        if not selected_paths:
            messagebox.showinfo(
                "Playlist Builder", "Select one or more songs to add first."
            )
            return

        combined = list(self.player_temp_playlist)
        for path in selected_paths:
            if path not in combined:
                combined.append(path)
        self._player_set_temp_playlist(combined)

    def _player_remove_selected_from_temp(self) -> None:
        if self.player_playlist_listbox is None:
            return
        selected = list(self.player_playlist_listbox.curselection())
        if not selected:
            messagebox.showinfo("Playlist Builder", "Select songs in the list to remove.")
            return

        remaining = [
            track for i, track in enumerate(self.player_temp_playlist) if i not in selected
        ]
        self._player_set_temp_playlist(remaining)

    def _player_clear_temp_playlist(self) -> None:
        if not self.player_temp_playlist:
            return
        self._player_set_temp_playlist([])

    def _player_load_playlist(self) -> None:
        playlists_dir = os.path.join(self.library_path, "Playlists")
        initial_dir = (
            playlists_dir
            if os.path.isdir(playlists_dir)
            else self.library_path
            if self.library_path
            else os.getcwd()
        )
        chosen = filedialog.askopenfilename(
            title="Load Playlist",
            initialdir=initial_dir,
            defaultextension=".m3u",
            filetypes=[("M3U Playlist", "*.m3u"), ("All Files", "*.*")],
        )
        if not chosen:
            return

        try:
            with open(chosen, "r", encoding="utf-8") as f:
                entries = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            messagebox.showerror("Load Playlist", f"Could not read playlist: {exc}")
            return

        base_dir = os.path.dirname(chosen)
        resolved: list[str] = []
        for entry in entries:
            candidate = entry if os.path.isabs(entry) else os.path.join(base_dir, entry)
            candidate = os.path.normpath(candidate)
            if os.path.exists(candidate):
                resolved.append(candidate)

        if not resolved:
            messagebox.showinfo(
                "Load Playlist", "No valid tracks found in the selected playlist."
            )
            return

        self._player_set_temp_playlist(resolved)

    def _player_save_temp_playlist(self) -> None:
        if not self.player_temp_playlist:
            messagebox.showinfo("Playlist Builder", "Add songs before saving a playlist.")
            return
        if not self.library_path:
            messagebox.showwarning("No Library", "Select a library before saving.")
            return

        outfile = self._player_prompt_playlist_destination()
        if not outfile:
            return

        try:
            write_playlist(self.player_temp_playlist, outfile)
        except Exception as exc:
            messagebox.showerror("Playlist", f"Failed to save playlist: {exc}")
            return

        messagebox.showinfo("Playlist", f"Playlist saved to {outfile}")
        self._open_folder(os.path.dirname(outfile))

    def _player_prompt_playlist_destination(self) -> str | None:
        playlists_dir = os.path.join(self.library_path, "Playlists")
        os.makedirs(playlists_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"PlayerSelection_{ts}.m3u"

        chosen = filedialog.asksaveasfilename(
            parent=self,
            title="Save Playlist As",
            defaultextension=".m3u",
            initialdir=playlists_dir,
            initialfile=default_name,
            filetypes=[("M3U Playlist", "*.m3u"), ("All Files", "*.*")],
        )

        if not chosen:
            return None

        return os.path.join(playlists_dir, os.path.basename(chosen))

    def _player_show_current_playlists(self) -> None:
        if not self.library_path:
            messagebox.showwarning("No Library", "Select a library first.")
            return

        playlists_dir = os.path.join(self.library_path, "Playlists")
        if not os.path.isdir(playlists_dir):
            messagebox.showinfo(
                "Playlists", "No playlists folder found yet. Save one to get started."
            )
            return
        try:
            entries = [
                f
                for f in os.listdir(playlists_dir)
                if os.path.isfile(os.path.join(playlists_dir, f))
            ]
        except OSError as exc:
            messagebox.showerror("Playlists", f"Could not read folder: {exc}")
            return

        if not entries:
            messagebox.showinfo("Playlists", "No playlists found.")
            return

        messagebox.showinfo("Playlists", "\n".join(sorted(entries)))

    def _open_folder(self, path: str) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=False)
            else:
                subprocess.run(["xdg-open", path], check=False)
        except Exception as exc:
            messagebox.showerror("Open Folder", f"Could not open {path}: {exc}")

    def populate_quality_table(self, matches):
        self.clear_quality_view()
        self._quality_play_buttons = []
        if not matches:
            ttk.Label(self.qc_inner, text="No duplicates detected in this library").pack(pady=20)
            return

        for keep_path, dup_path in matches:
            row = ttk.Frame(self.qc_inner)
            row.pack(fill="x", padx=10, pady=2)

            btn_state = "normal"
            if not self.preview_backend_available or self._preview_in_progress:
                btn_state = "disabled"
            play_btn = ttk.Button(
                row,
                text="▶",
                width=1,
                state=btn_state,
                command=lambda p=dup_path: self._play_preview(p),
            )
            play_btn.pack(side="left")
            self._quality_play_buttons.append(play_btn)

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

    # ── Player Tab Helpers ─────────────────────────────────────────────
    def _clear_player_table(self) -> None:
        if not hasattr(self, "player_tree"):
            return
        for row in self.player_tree.get_children():
            self.player_tree.delete(row)
        self.player_tree_paths.clear()
        self.player_tree_rows.clear()
        self._update_player_art(None)

    def _prepare_player_rows(self, tracks: list[str]) -> list[dict[str, str]]:
        rows = []
        for path in tracks:
            tags = get_tags(path)
            title = tags.get("title") or os.path.basename(path)
            artist = tags.get("artist") or "Unknown"
            album = tags.get("album") or ""
            length_sec = self._get_track_length(path)
            rows.append(
                {
                    "title": title,
                    "artist": artist,
                    "album": album,
                    "length": self._format_duration(length_sec),
                    "path": path,
                }
            )
        return rows

    def _format_duration(self, seconds: float | None) -> str:
        if not seconds:
            return "—"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}:{secs:02d}"

    def _get_track_length(self, path: str) -> float | None:
        try:
            audio = MutagenFile(path)
            if audio and getattr(audio, "info", None):
                length = getattr(audio.info, "length", None)
                if length:
                    return float(length)
        except Exception:
            return None
        return None

    def _load_player_library_async(self) -> None:
        if not self.library_path:
            self.player_status_var.set("Select a library to load tracks.")
            if hasattr(self, "player_reload_btn"):
                self.player_reload_btn.config(state="disabled")
            return
        if self._player_load_thread and self._player_load_thread.is_alive():
            return

        self.player_status_var.set("Loading tracks…")
        self.player_reload_btn.config(state="disabled")
        thread = threading.Thread(
            target=self._load_player_library, args=(self.library_path,), daemon=True
        )
        self._player_load_thread = thread
        thread.start()

    def _load_player_library(self, library_path: str) -> None:
        try:
            tracks = gather_tracks(library_path, self.folder_filter)
        except Exception as exc:
            self.after(
                0,
                lambda: self.player_status_var.set(
                    f"Failed to load library: {exc}"
                ),
            )
            self.after(0, lambda: self.player_reload_btn.config(state="normal"))
            return

        rows = self._prepare_player_rows(tracks)

        self.after(0, lambda: self._update_player_table(rows, library_path))

    def preview_tracks_in_player(self, tracks: list[str], label: str) -> None:
        if not tracks:
            messagebox.showinfo("Playlist Preview", "No tracks available to preview.")
            return

        def worker() -> None:
            rows = self._prepare_player_rows(tracks)
            self.after(0, lambda: self._show_player_preview(rows, label))

        threading.Thread(target=worker, daemon=True).start()

    def _show_player_preview(self, rows: list[dict[str, str]], label: str) -> None:
        self.player_view_label = f"Previewing {label}"
        self.player_tracks = rows
        self._apply_player_filter(total_count=len(rows))
        if hasattr(self, "notebook"):
            self.notebook.select(self.player_tab)

    def _update_player_table(self, rows: list[dict[str, str]], library_path: str) -> None:
        if library_path != self.library_path:
            return
        self.player_view_label = f"Library – {os.path.basename(library_path)}"
        self.player_tracks = rows
        self._apply_player_filter(total_count=len(rows))

    def _apply_player_filter(self, *_args, total_count: int | None = None) -> None:
        if not hasattr(self, "player_tree"):
            return
        query = self.player_search_var.get().strip().lower()
        if query:
            rows = [r for r in self.player_tracks if self._matches_player_query(r, query)]
        else:
            rows = list(self.player_tracks)
        self._render_player_rows(rows, total_count=total_count)

    def _matches_player_query(self, row: dict[str, str], query: str) -> bool:
        for key in ("title", "artist", "album", "length", "path"):
            val = row.get(key)
            if val and query in str(val).lower():
                return True
        return False

    def _render_player_rows(
        self, rows: list[dict[str, str]], total_count: int | None = None
    ) -> None:
        self._clear_player_table()
        play_icon = "▶" if self.preview_backend_available else "—"
        for row in rows:
            item = self.player_tree.insert(
                "",
                "end",
                values=(
                    row["title"],
                    row["artist"],
                    row["album"],
                    row["length"],
                    play_icon,
                ),
            )
            self.player_tree_paths[item] = row["path"]
            self.player_tree_rows[item] = row

        if rows:
            first = rows[0]
            self._update_player_art(
                first.get("path"),
                title=first.get("title"),
                artist=first.get("artist"),
            )
        else:
            self._update_player_art(None)

        total = total_count if total_count is not None else len(rows)
        shown = len(rows)
        suffix = "track" if total == 1 else "tracks"
        prefix = f"{self.player_view_label}. " if self.player_view_label else ""
        if not self.preview_backend_available:
            reason = self.preview_backend_error or "Install python-vlc to enable playback."
            status = f"{prefix}Loaded {total} {suffix}. Preview disabled: {reason}"
        else:
            status = f"{prefix}Loaded {total} {suffix}. Click ▶ for a 30s highlight."
        if shown != total:
            status += f" Showing {shown} match{'es' if shown != 1 else ''}."
        self.player_status_var.set(status)
        self.player_reload_btn.config(state="normal")

    def _on_player_tree_click(self, event) -> None:
        region = self.player_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        col = self.player_tree.identify_column(event.x)
        # Play column is the fifth heading (#5 when using ``show='headings'``)
        if col != "#5":
            return
        item = self.player_tree.identify_row(event.y)
        if not item:
            return
        if not self.preview_backend_available:
            self.player_status_var.set(
                f"Preview disabled: {self.preview_backend_error or 'Install python-vlc.'}"
            )
            return
        path = self.player_tree_paths.get(item)
        if path:
            row = self.player_tree_rows.get(item, {})
            self._update_player_art(
                path, title=row.get("title"), artist=row.get("artist")
            )
            self._play_preview(path, duration_ms=30000, player_item=item)

    def _on_player_selection_change(self, _event) -> None:
        if not hasattr(self, "player_tree"):
            return
        selection = self.player_tree.selection()
        if not selection:
            self._update_player_art(None)
            return
        item = selection[0]
        path = self.player_tree_paths.get(item)
        row = self.player_tree_rows.get(item, {})
        self._update_player_art(path, title=row.get("title"), artist=row.get("artist"))

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
        win.bind(
            "<Destroy>",
            lambda e: (
                self._check_tagfix_api_status()
                if e.widget is win and self.winfo_exists()
                else None
            ),
        )
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
