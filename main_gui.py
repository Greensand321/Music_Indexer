import os
import threading
import sys
import logging
import webbrowser
import re
import html
import shutil
from pathlib import Path

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
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from validator import validate_soundvault_structure
from music_indexer_api import (
    run_full_indexer,
    find_duplicates as api_find_duplicates,
    get_tags,
)
from duplicate_consolidation import (
    _is_noop_group,
    _planned_actions,
    build_consolidation_plan,
    build_duplicate_pair_report,
    ConsolidationPlan,
    consolidation_plan_from_dict,
    export_consolidation_preview,
    export_consolidation_preview_html,
    export_duplicate_pair_report_html,
    LOSSLESS_EXTS,
)
from duplicate_consolidation_executor import ExecutionConfig, execute_consolidation_plan
from duplicate_bucketing_poc import run_duplicate_bucketing_poc
from controllers.library_index_controller import generate_index
from gui.audio_preview import PlaybackError, VlcPreviewPlayer
from io import BytesIO
from PIL import Image, ImageTk
from mutagen import File as MutagenFile
from near_duplicate_detector import fingerprint_distance
import chromaprint_utils
from config import (
    ARTWORK_VASTLY_DIFFERENT_THRESHOLD,
    EXACT_DUPLICATE_THRESHOLD,
    FP_DURATION_MS,
    FP_OFFSET_MS,
    FP_SILENCE_MIN_LEN_MS,
    FP_SILENCE_THRESHOLD_DB,
    FP_TRIM_LEAD_MAX_MS,
    FP_TRIM_PADDING_MS,
    FP_TRIM_TRAIL_MAX_MS,
    FP_TRIM_SILENCE,
    ALLOW_MISMATCHED_EDITS,
    MIXED_CODEC_THRESHOLD_BOOST,
    NEAR_DUPLICATE_THRESHOLD,
    PREVIEW_ARTWORK_MAX_DIM,
    PREVIEW_ARTWORK_QUALITY,
    load_config,
)
from fingerprint_cache import (
    ensure_fingerprint_cache,
    get_fingerprint,
    get_cached_fingerprint_metadata,
    store_fingerprint,
)
from simple_duplicate_finder import SUPPORTED_EXTS, _compute_fp
from tag_fixer import MIN_INTERACTIVE_SCORE, FileRecord
from typing import Any, Callable, List, Mapping
from indexer_control import cancel_event, IndexCancelled
import library_sync_review
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
from config import load_config, save_config
import playlist_engine
from playlist_engine import bucket_by_tempo_energy, autodj_playlist
from controllers.cluster_controller import gather_tracks
from playlist_generator import write_playlist
from utils.path_helpers import ensure_long_path, strip_ext_prefix
from utils.audio_metadata_reader import (
    read_metadata,
    read_tags,
    read_sidecar_artwork_bytes,
)
from utils.opus_library_mirror import mirror_library, write_mirror_report
from utils.opus_metadata_reader import read_opus_metadata


@dataclass
class PlanGenerationResult:
    plan: Any | None
    preview_json_path: str | None
    preview_html_path: str | None
    log_messages: list[str]
    review_required_count: int
    group_count: int
    had_tracks: bool

FilterFn = Callable[[FileRecord], bool]
_cached_filters = None

logger = logging.getLogger(__name__)

YEAR_ASSISTANT_PROMPT = """
You will be given a list of music tracks in the format `Artist - Title` (one per line).
For each line, use web search to determine the most likely original release year of the recording
(prefer the earliest official release of the song, not a later remaster unless that is the only
identifiable version). Return ONLY valid JSON in the following exact format:

[
  {"query":"Artist - Title","year":"YYYY" | null,"confidence":"high|medium|low|unknown","notes":"short source hint"},
  ...
]

Rules:
- Include one JSON object for every input line and preserve the input order.
- Do not add commentary outside JSON.
- If uncertain or multiple versions exist, set year to null or confidence low and explain in notes.
"""


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
                os.path.join(app.library_path, "Docs", ".genre_mapping.json")
                if app.library_path
                else "Select a library to enable normalization."
            )
            scan_btn.config(state="normal" if app.library_path else "disabled")
            save_btn.config(state="normal" if app.library_path else "disabled")
            apply_btn.config(state="normal" if app.library_path else "disabled")

        intro = ttk.Frame(frame)
        intro.pack(fill="x", padx=10, pady=10)

        ttk.Label(
            intro,
            text=(
                "Normalize genre labels across your library by generating or "
                "updating the Docs/.genre_mapping.json file."
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

        init_status = tk.StringVar(
            value="Paste JSON and initialize to rewrite library genres."
        )
        apply_status = tk.StringVar(
            value="Apply a saved or pasted mapping to embedded genres."
        )

        action_bar = ttk.Frame(frame)
        action_bar.pack(fill="x", padx=10, pady=(6, 6))
        init_btn = ttk.Button(action_bar, text="Initialize Normalizer")
        init_btn.pack(side="left")
        save_btn = ttk.Button(action_bar, text="Save Mapping", command=app.apply_mapping)
        save_btn.pack(side="left", padx=(6, 0))
        apply_btn = ttk.Button(action_bar, text="Apply To Songs")
        apply_btn.pack(side="left", padx=(6, 0))

        status_col = ttk.Frame(action_bar)
        status_col.pack(side="left", padx=12, expand=True, fill="x")
        ttk.Label(status_col, textvariable=init_status).pack(anchor="w")
        ttk.Label(status_col, textvariable=apply_status).pack(anchor="w")

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
            app.mapping_path = os.path.join(folder, "Docs", ".genre_mapping.json")

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

        def finish_init(changed: int, total: int):
            init_status.set(f"Rewrote genres for {changed} of {total} files.")
            init_btn.config(state="normal")
            save_btn.config(state="normal" if app.library_path else "disabled")
            apply_btn.config(state="normal" if app.library_path else "disabled")
            scan_btn.config(state="normal" if app.library_path else "disabled")
            if getattr(app, "raw_genre_list", None):
                populate_raw_genres(app.raw_genre_list)

        def start_init():
            if not app.library_path:
                messagebox.showwarning("No Library", "Select a library to initialize.")
                return
            raw_json = app.text_map.get("1.0", "end").strip()
            if not raw_json:
                messagebox.showwarning(
                    "No JSON Provided", "Paste JSON mapping from the assistant first."
                )
                return
            try:
                mapping = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                messagebox.showerror("Invalid JSON", str(exc))
                return

            init_status.set("Initializing and rewriting genres…")
            init_btn.config(state="disabled")
            save_btn.config(state="disabled")
            apply_btn.config(state="disabled")
            scan_btn.config(state="disabled")
            prog["value"] = 0

            def worker():
                try:
                    changed, total = app.initialize_genre_normalizer(
                        mapping, on_progress
                    )
                except Exception as exc:
                    app.after(
                        0,
                        lambda: messagebox.showerror(
                            "Initialization Failed", str(exc)
                        ),
                    )
                    app.after(0, lambda: finish_init(0, 0))
                    return
                app.after(0, lambda: finish_init(changed, total))

            threading.Thread(target=worker, daemon=True).start()

        scan_btn.config(command=start_scan)
        init_btn.config(command=start_init)

        def finish_apply(changed: int, total: int):
            apply_status.set(f"Applied mapping to {changed} of {total} files.")
            init_btn.config(state="normal" if app.library_path else "disabled")
            save_btn.config(state="normal" if app.library_path else "disabled")
            apply_btn.config(state="normal" if app.library_path else "disabled")
            scan_btn.config(state="normal" if app.library_path else "disabled")
            if getattr(app, "raw_genre_list", None):
                populate_raw_genres(app.raw_genre_list)

        def start_apply():
            if not app.library_path:
                messagebox.showwarning("No Library", "Select a library to apply mappings.")
                return
            raw_json = app.text_map.get("1.0", "end").strip()
            if not raw_json:
                messagebox.showwarning(
                    "No JSON Provided", "Paste JSON mapping from the assistant first."
                )
                return
            try:
                mapping = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                messagebox.showerror("Invalid JSON", str(exc))
                return

            apply_status.set("Applying mapping to songs…")
            init_btn.config(state="disabled")
            save_btn.config(state="disabled")
            apply_btn.config(state="disabled")
            scan_btn.config(state="disabled")
            prog["value"] = 0

            def worker():
                try:
                    changed, total = app.initialize_genre_normalizer(
                        mapping, on_progress
                    )
                except Exception as exc:
                    app.after(
                        0,
                        lambda: messagebox.showerror("Apply Failed", str(exc)),
                    )
                    app.after(0, lambda: finish_apply(0, 0))
                    return
                app.after(0, lambda: finish_apply(changed, total))

            threading.Thread(target=worker, daemon=True).start()

        apply_btn.config(command=start_apply)

        def refresh_panel():
            update_controls()
            if getattr(app, "raw_genre_list", None):
                populate_raw_genres(app.raw_genre_list)
            load_existing_mapping()

        frame.refresh_cluster_panel = refresh_panel
        refresh_panel()
        return frame
    elif name == "Year Assistant":
        lib_var = tk.StringVar(value=app.library_path or "No library selected")
        status_var = tk.StringVar(value="Paste AI results and parse to review changes.")
        dry_run_var = tk.BooleanVar(value=True)
        app.year_assistant_dry_run_var = dry_run_var

        expected_queries: list[str] = []
        query_to_paths: dict[str, list[str]] = {}
        results_by_iid: dict[str, dict[str, Any]] = {}

        def update_controls():
            lib_var.set(app.library_path or "No library selected")
            apply_selected_btn.config(
                state="normal" if app.library_path else "disabled"
            )
            apply_high_btn.config(
                state="normal" if app.library_path else "disabled"
            )

        def coerce_entry(item: Any) -> dict[str, str | None] | None:
            if isinstance(item, str):
                return {"query": item.strip(), "path": None}
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    artist = str(item[0]).strip()
                    title = str(item[1]).strip()
                    query = f"{artist} - {title}".strip(" -")
                    path = str(item[2]).strip() if len(item) > 2 and item[2] else None
                    return {"query": query, "path": path}
            if isinstance(item, dict):
                query = item.get("query")
                artist = item.get("artist") or item.get("Artist")
                title = item.get("title") or item.get("Title")
                path = (
                    item.get("path")
                    or item.get("filepath")
                    or item.get("file")
                    or item.get("filename")
                )
                if not query and (artist or title):
                    artist_str = str(artist or "").strip()
                    title_str = str(title or "").strip()
                    query = f"{artist_str} - {title_str}".strip(" -")
                if query:
                    return {"query": str(query).strip(), "path": str(path).strip() if path else None}
            return None

        def load_missing_year_entries() -> list[dict[str, str | None]]:
            raw_sources = None
            for attr in (
                "missing_year_records",
                "missing_year_list",
                "missing_year_tracks",
                "missing_year_entries",
            ):
                raw_sources = getattr(app, attr, None)
                if raw_sources:
                    break
            if isinstance(raw_sources, dict):
                raw_sources = list(raw_sources.values())
            if not isinstance(raw_sources, (list, tuple)):
                return []
            entries: list[dict[str, str | None]] = []
            for item in raw_sources:
                entry = coerce_entry(item)
                if entry and entry["query"]:
                    entries.append(entry)
            return entries

        def dedupe_and_sort(lines: list[str]) -> list[str]:
            unique: dict[str, None] = {}
            for line in lines:
                cleaned = line.strip()
                if cleaned and cleaned not in unique:
                    unique[cleaned] = None
            return sorted(unique.keys(), key=lambda s: s.casefold())

        def refresh_track_list():
            nonlocal expected_queries, query_to_paths
            entries = load_missing_year_entries()
            query_to_paths = {}
            for entry in entries:
                query = entry["query"]
                if not query:
                    continue
                query_to_paths.setdefault(query, [])
                if entry.get("path"):
                    query_to_paths[query].append(str(entry["path"]))

            expected_queries = dedupe_and_sort(list(query_to_paths.keys()))
            track_list_box.configure(state="normal")
            track_list_box.delete("1.0", "end")
            if expected_queries:
                track_list_box.insert("1.0", "\n".join(expected_queries))
            else:
                track_list_box.insert(
                    "1.0",
                    "No missing-year entries available yet.",
                )
            track_list_box.configure(state="disabled")

        def copy_instructions():
            app.clipboard_clear()
            app.clipboard_append(YEAR_ASSISTANT_PROMPT.strip())

        def copy_track_list():
            app.clipboard_clear()
            app.clipboard_append("\n".join(expected_queries))

        def set_status(message: str):
            status_var.set(message)

        def normalize_year_value(raw_year: Any) -> str | None:
            if raw_year is None:
                return None
            if isinstance(raw_year, int):
                raw_year = str(raw_year)
            if isinstance(raw_year, str):
                cleaned = raw_year.strip()
                if re.fullmatch(r"\d{4}", cleaned):
                    return cleaned
            return None

        def parse_results():
            raw_json = results_box.get("1.0", "end").strip()
            if not raw_json:
                messagebox.showwarning("No Results", "Paste the AI JSON output first.")
                return
            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                messagebox.showerror("Invalid JSON", str(exc))
                return
            if not isinstance(data, list):
                messagebox.showerror("Invalid JSON", "Expected a JSON array.")
                return

            allowed_confidence = {"high", "medium", "low", "unknown"}
            parsed: dict[str, dict[str, Any]] = {}
            errors: list[str] = []
            for idx, entry in enumerate(data, start=1):
                if not isinstance(entry, dict):
                    errors.append(f"Entry {idx} must be an object.")
                    continue
                query = entry.get("query")
                confidence = entry.get("confidence")
                notes = entry.get("notes", "")
                year_value = normalize_year_value(entry.get("year"))

                if not isinstance(query, str) or not query.strip():
                    errors.append(f"Entry {idx} missing valid query.")
                    continue
                query = query.strip()
                if query in parsed:
                    errors.append(f"Duplicate query found: {query}")
                    continue
                if confidence not in allowed_confidence:
                    errors.append(f"Entry {idx} has invalid confidence '{confidence}'.")
                    continue
                if not isinstance(notes, str):
                    notes = str(notes)

                parsed[query] = {
                    "query": query,
                    "year": year_value,
                    "confidence": confidence,
                    "notes": notes,
                }

            if errors:
                messagebox.showerror("Parse Errors", "\n".join(errors))
                return

            missing = [q for q in expected_queries if q not in parsed]
            extra = [q for q in parsed.keys() if q not in expected_queries]

            results_tree.delete(*results_tree.get_children())
            results_by_iid.clear()

            def add_row(result: dict[str, Any], status: str, notes_suffix: str | None = None):
                notes_text = result.get("notes") or ""
                if notes_suffix:
                    notes_text = f"{notes_text} ({notes_suffix})" if notes_text else notes_suffix
                values = (
                    result.get("query", ""),
                    result.get("year") or "",
                    result.get("confidence") or "unknown",
                    notes_text,
                    status,
                )
                iid = results_tree.insert("", "end", values=values)
                results_by_iid[iid] = {
                    **result,
                    "status": status,
                    "notes": notes_text,
                }

            for query in expected_queries:
                result = parsed.get(
                    query,
                    {
                        "query": query,
                        "year": None,
                        "confidence": "unknown",
                        "notes": "Missing from AI output.",
                    },
                )
                paths = query_to_paths.get(query, [])
                status = "Ready" if result.get("year") else "Unresolved"
                notes_suffix = None
                if not paths:
                    status = "Unresolved"
                    notes_suffix = "No matching files."
                elif len(paths) > 1:
                    notes_suffix = f"Matches {len(paths)} files."
                add_row(result, status, notes_suffix)

            for query in extra:
                result = parsed[query]
                add_row(result, "Skipped", "Query not in input list.")

            message = "Parsed results."
            if missing:
                message += f" Missing {len(missing)} queries from AI output."
            if extra:
                message += f" {len(extra)} extra entries skipped."
            set_status(message)

        def update_row_status(iid: str, status: str, notes: str | None = None):
            row = results_tree.item(iid)
            values = list(row.get("values", []))
            if len(values) < 5:
                return
            values[4] = status
            if notes is not None:
                values[3] = notes
            results_tree.item(iid, values=values)
            if iid in results_by_iid:
                results_by_iid[iid]["status"] = status
                if notes is not None:
                    results_by_iid[iid]["notes"] = notes

        def write_year_tag(path: str, year: str) -> tuple[bool, str | None, bool]:
            try:
                audio = MutagenFile(ensure_long_path(path), easy=True)
            except Exception as exc:
                return False, f"Failed to read {path}: {exc}", False
            if audio is None:
                return False, f"Unsupported file: {path}", False
            old_tags = read_tags(path)
            old_year = str(old_tags.get("year") or old_tags.get("date") or "").strip()
            if old_year == year:
                return True, old_year, False
            changed = False
            for key in ("date", "year"):
                try:
                    audio[key] = [year]
                    changed = True
                except Exception:
                    continue
            if not changed:
                return False, f"Could not write year tags for {path}", False
            try:
                audio.save()
            except Exception as exc:
                return False, f"Failed to save {path}: {exc}", False
            return True, old_year, True

        def log_audit_entry(path: str, old_year: str | None, new_year: str):
            docs_dir = os.path.join(app.library_path, "Docs")
            os.makedirs(docs_dir, exist_ok=True)
            audit_path = os.path.join(docs_dir, "year_assistant_audit.log")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(audit_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"{timestamp} | {path} | old_year={old_year or ''} | new_year={new_year}\n"
                )

        def apply_entries(iids: list[str], *, label: str):
            if not app.library_path:
                messagebox.showwarning("No Library", "Select a library before applying.")
                return
            if not iids:
                messagebox.showinfo("Apply", "No rows selected.")
                return

            dry_run = dry_run_var.get()
            applied = 0
            skipped = 0
            errors: list[str] = []

            for iid in iids:
                result = results_by_iid.get(iid)
                if not result:
                    continue
                query = result.get("query")
                year = result.get("year")
                if result.get("status") != "Ready" or not query or not year:
                    skipped += 1
                    update_row_status(iid, "Skipped")
                    continue
                paths = query_to_paths.get(query, [])
                if not paths:
                    skipped += 1
                    update_row_status(iid, "Unresolved", "No matching files.")
                    continue
                if dry_run:
                    applied += len(paths)
                    continue
                for path in paths:
                    success, info, changed = write_year_tag(path, year)
                    if not success:
                        errors.append(info or f"Failed to update {path}")
                        continue
                    if not changed:
                        skipped += 1
                        continue
                    log_audit_entry(path, info, year)
                    applied += 1

            summary = f"{label}: updated {applied} files, skipped {skipped}."
            if dry_run:
                summary = f"{label} (Dry Run): would update {applied} files."
            set_status(summary)
            if errors:
                messagebox.showerror("Apply Errors", "\n".join(errors))
            else:
                messagebox.showinfo("Apply Results", summary)

        def apply_selected():
            apply_entries(list(results_tree.selection()), label="Apply Selected")

        def apply_high_confidence():
            high_iids = [
                iid
                for iid, result in results_by_iid.items()
                if result.get("status") == "Ready"
                and result.get("confidence") == "high"
            ]
            apply_entries(high_iids, label="Apply High Confidence")

        intro = ttk.Frame(frame)
        intro.pack(fill="x", padx=10, pady=10)
        ttk.Label(
            intro,
            text=(
                "Use the Year Assistant to generate AI instructions for missing year tags, "
                "then import AI results to apply release years."
            ),
            wraplength=520,
            justify="left",
        ).pack(anchor="w")

        status_row = ttk.Frame(intro)
        status_row.pack(fill="x", pady=(6, 0))
        ttk.Label(status_row, text="Library:", width=12).pack(side="left")
        ttk.Label(status_row, textvariable=lib_var).pack(
            side="left", fill="x", expand=True
        )

        step1_box = ttk.LabelFrame(frame, text="Step 1: Generate AI Instructions")
        step1_box.pack(fill="both", expand=False, padx=10, pady=(0, 10))

        prompt_box = ttk.LabelFrame(step1_box, text="Instructions to paste into the AI")
        prompt_box.pack(fill="both", expand=False, padx=8, pady=(8, 6))
        prompt_text = ScrolledText(prompt_box, height=8, wrap="word")
        prompt_text.pack(fill="both", expand=True, padx=6, pady=6)
        prompt_text.insert("1.0", YEAR_ASSISTANT_PROMPT.strip())
        prompt_text.configure(state="disabled")
        ttk.Button(prompt_box, text="Copy AI Instructions", command=copy_instructions).pack(
            anchor="e", padx=6, pady=(0, 6)
        )

        track_box = ttk.LabelFrame(step1_box, text="Tracks missing year")
        track_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        track_list_box = ScrolledText(track_box, height=10, wrap="word")
        track_list_box.pack(fill="both", expand=True, padx=6, pady=6)
        track_list_box.configure(state="disabled")
        ttk.Button(track_box, text="Copy Track List", command=copy_track_list).pack(
            anchor="e", padx=6, pady=(0, 6)
        )

        step2_box = ttk.LabelFrame(frame, text="Step 2: Import Results + Apply")
        step2_box.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        ttk.Label(
            step2_box,
            text="Paste the AI JSON output below and import to review the proposed years.",
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=8, pady=(6, 0))

        results_box = ScrolledText(step2_box, height=8, wrap="word")
        results_box.pack(fill="x", expand=False, padx=8, pady=(6, 6))

        action_bar = ttk.Frame(step2_box)
        action_bar.pack(fill="x", padx=8, pady=(0, 6))
        ttk.Button(action_bar, text="Import Results", command=parse_results).pack(
            side="left"
        )
        ttk.Checkbutton(action_bar, text="Dry Run", variable=dry_run_var).pack(
            side="left", padx=(10, 0)
        )
        apply_selected_btn = ttk.Button(
            action_bar, text="Apply Selected", command=apply_selected
        )
        apply_selected_btn.pack(side="right")
        apply_high_btn = ttk.Button(
            action_bar, text="Apply All High Confidence", command=apply_high_confidence
        )
        apply_high_btn.pack(side="right", padx=(0, 6))

        table_frame = ttk.Frame(step2_box)
        table_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        results_scroll = ttk.Scrollbar(table_frame, orient="vertical")
        results_scroll.pack(side="right", fill="y")
        results_columns = ("Track", "Year", "Confidence", "Notes", "Status")
        results_tree = ttk.Treeview(
            table_frame,
            columns=results_columns,
            show="headings",
            yscrollcommand=results_scroll.set,
            selectmode="extended",
        )
        results_scroll.config(command=results_tree.yview)
        results_tree.pack(fill="both", expand=True)

        for col in results_columns:
            results_tree.heading(col, text=col)
            width = 120
            if col == "Track":
                width = 240
            elif col == "Notes":
                width = 200
            results_tree.column(col, width=width, anchor="w")

        status_label = ttk.Label(step2_box, textvariable=status_var)
        status_label.pack(anchor="w", padx=8, pady=(0, 6))

        def refresh_panel():
            update_controls()
            refresh_track_list()
            set_status("Ready to import AI results.")

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
            initial_dir = load_last_path() or os.getcwd()
            f = filedialog.askopenfilename(initialdir=initial_dir)
            if f:
                sel.set(f)
                save_last_path(os.path.dirname(f))

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


@dataclass
class DuplicatePair:
    left_path: str
    right_path: str


class DuplicatePairReviewTool(tk.Toplevel):
    """Review paired duplicate results one-by-one."""

    def __init__(self, parent: tk.Widget, *, library_path: str, plan: ConsolidationPlan):
        super().__init__(parent)
        self.title("Duplicate Pair Review")
        self.transient(parent)
        self.resizable(True, True)

        self.library_path = library_path
        self.pairs = self._collect_pairs(plan)
        self._pair_index = 0
        self._left_tags: dict[str, object] = {}
        self._right_tags: dict[str, object] = {}
        self._left_cover = None
        self._right_cover = None
        self._status_var = tk.StringVar(value="Ready")
        self._progress_var = tk.StringVar(value="")

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(
            container,
            text="Duplicate Pair Review",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            container,
            text="Review paired duplicates from the Duplicate Finder preview.",
            foreground="#555",
            wraplength=600,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(container, textvariable=self._progress_var).pack(anchor="w")

        grid = ttk.Frame(container)
        grid.pack(fill="both", expand=True, pady=8)
        grid.columnconfigure((0, 1), weight=1)

        self.left_panel = self._build_track_panel(grid, "Left Track")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.right_panel = self._build_track_panel(grid, "Right Track")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        status_row = ttk.Frame(container)
        status_row.pack(fill="x", pady=(0, 8))
        ttk.Label(status_row, textvariable=self._status_var, foreground="#555").pack(
            side="left"
        )

        btns = ttk.Frame(container)
        btns.pack(fill="x")
        self.yes_btn = ttk.Button(btns, text="Yes", command=self._handle_yes)
        self.yes_btn.pack(side="left")
        self.no_btn = ttk.Button(btns, text="No", command=self._handle_no)
        self.no_btn.pack(side="left", padx=(6, 0))
        self.tag_btn = ttk.Button(btns, text="Tag", command=self._handle_tag)
        self.tag_btn.pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Close", command=self.destroy).pack(side="right")

        self._load_pair()

    def _collect_pairs(self, plan: ConsolidationPlan) -> list[DuplicatePair]:
        pairs = []
        for group in plan.groups:
            if len(group.losers) == 1:
                pairs.append(DuplicatePair(group.winner_path, group.losers[0]))
        return pairs

    def _build_track_panel(self, parent: tk.Widget, title: str) -> ttk.Frame:
        frame = ttk.LabelFrame(parent, text=title)
        frame.columnconfigure(0, weight=1)
        art_label = tk.Label(frame)
        art_label.grid(row=0, column=0, pady=(6, 4))
        title_label = ttk.Label(frame, font=("TkDefaultFont", 10, "bold"))
        title_label.grid(row=1, column=0, sticky="w", padx=6)
        meta_label = ttk.Label(frame, justify="left")
        meta_label.grid(row=2, column=0, sticky="w", padx=6, pady=(2, 6))
        path_label = ttk.Label(frame, foreground="#777", wraplength=300)
        path_label.grid(row=3, column=0, sticky="w", padx=6, pady=(0, 6))
        frame.art_label = art_label  # type: ignore[attr-defined]
        frame.title_label = title_label  # type: ignore[attr-defined]
        frame.meta_label = meta_label  # type: ignore[attr-defined]
        frame.path_label = path_label  # type: ignore[attr-defined]
        return frame

    def _load_cover_image(self, path: str) -> ImageTk.PhotoImage:
        max_dim = int(load_config().get("preview_artwork_max_dim", PREVIEW_ARTWORK_MAX_DIM))
        cover = None
        tags, covers, error, _reader = read_metadata(path, include_cover=True)
        if error:
            logger.warning("Failed to read metadata for %s: %s", path, error)
        if covers:
            cover = covers[0]
        if cover:
            try:
                image = Image.open(BytesIO(cover))
            except Exception:
                image = Image.new("RGB", (max_dim, max_dim), "#333")
        else:
            image = Image.new("RGB", (max_dim, max_dim), "#333")
        image.thumbnail((max_dim, max_dim))
        return ImageTk.PhotoImage(image)

    def _format_metadata(self, tags: dict[str, object], path: str) -> str:
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "unknown"
        lines = [f"Extension: {ext}"]
        if tags.get("artist"):
            lines.append(f"Artist: {tags.get('artist')}")
        if tags.get("title"):
            lines.append(f"Title: {tags.get('title')}")
        if tags.get("album"):
            lines.append(f"Album: {tags.get('album')}")
        if tags.get("year"):
            lines.append(f"Year: {tags.get('year')}")
        if tags.get("track"):
            lines.append(f"Track: {tags.get('track')}")
        if tags.get("disc"):
            lines.append(f"Disc: {tags.get('disc')}")
        if tags.get("genre"):
            lines.append(f"Genre: {tags.get('genre')}")
        return "\n".join(str(line) for line in lines)

    def _display_name(self, tags: dict[str, object], path: str) -> str:
        artist = tags.get("artist")
        title = tags.get("title")
        if artist or title:
            artist_str = str(artist or "Unknown Artist")
            title_str = str(title or "Unknown Title")
            return f"{artist_str} - {title_str}"
        return os.path.basename(path)

    def _load_track(self, panel: ttk.Frame, path: str) -> dict[str, object]:
        tags, _covers, _error, _reader = read_metadata(path, include_cover=False)
        panel.title_label.configure(text=self._display_name(tags, path))
        panel.meta_label.configure(text=self._format_metadata(tags, path))
        panel.path_label.configure(text=path)
        return tags

    def _load_pair(self) -> None:
        if not self.pairs:
            self._progress_var.set("No paired duplicates found in the current preview.")
            self._status_var.set("Idle")
            self.yes_btn.configure(state="disabled")
            self.no_btn.configure(state="disabled")
            self.tag_btn.configure(state="disabled")
            return

        if self._pair_index >= len(self.pairs):
            self._progress_var.set("All pairs reviewed.")
            self._status_var.set("Done")
            self.yes_btn.configure(state="disabled")
            self.no_btn.configure(state="disabled")
            self.tag_btn.configure(state="disabled")
            return

        pair = self.pairs[self._pair_index]
        self._progress_var.set(
            f"Pair {self._pair_index + 1} of {len(self.pairs)}"
        )
        self._status_var.set("Ready")

        self._left_tags = self._load_track(self.left_panel, pair.left_path)
        self._right_tags = self._load_track(self.right_panel, pair.right_path)

        self._left_cover = self._load_cover_image(pair.left_path)
        self.left_panel.art_label.configure(image=self._left_cover)
        self.left_panel.art_label.image = self._left_cover
        self._right_cover = self._load_cover_image(pair.right_path)
        self.right_panel.art_label.configure(image=self._right_cover)
        self.right_panel.art_label.image = self._right_cover

    def _advance(self) -> None:
        self._pair_index += 1
        self._load_pair()

    def _delete_inferior(self, left_path: str, right_path: str) -> str | None:
        left_ext = os.path.splitext(left_path)[1].lower()
        right_ext = os.path.splitext(right_path)[1].lower()
        if left_ext == ".flac" and right_ext != ".flac":
            return right_path
        if right_ext == ".flac" and left_ext != ".flac":
            return left_path
        return None

    def _handle_yes(self) -> None:
        if not self.pairs or self._pair_index >= len(self.pairs):
            return
        pair = self.pairs[self._pair_index]
        delete_path = self._delete_inferior(pair.left_path, pair.right_path)
        if not delete_path:
            self._status_var.set("No FLAC in pair; nothing deleted.")
            self._advance()
            return
        try:
            os.remove(ensure_long_path(delete_path))
            self._status_var.set(f"Deleted {os.path.basename(delete_path)}")
        except OSError as exc:
            messagebox.showerror("Delete Failed", str(exc))
            self._status_var.set("Delete failed")
        self._advance()

    def _handle_no(self) -> None:
        self._status_var.set("Skipped")
        self._advance()

    def _handle_tag(self) -> None:
        if not self.pairs or self._pair_index >= len(self.pairs):
            return
        pair = self.pairs[self._pair_index]
        docs_dir = os.path.join(self.library_path, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        tag_path = os.path.join(docs_dir, "duplicate_pair_tags.txt")
        left_name = self._display_name(self._left_tags, pair.left_path)
        right_name = self._display_name(self._right_tags, pair.right_path)
        line = f"{left_name} | {right_name}"
        try:
            with open(tag_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._status_var.set("Tagged pair added")
        except OSError as exc:
            messagebox.showerror("Tag Failed", str(exc))
            self._status_var.set("Tag failed")
            return
        self._advance()


class DuplicateFinderShell(tk.Toplevel):
    """GUI shell for the refreshed Duplicate Finder workflow."""

    def __init__(self, parent: tk.Widget, library_path: str):
        super().__init__(parent)
        self.title("Duplicate Finder")
        self.transient(parent)
        self.resizable(True, True)

        self.status_var = tk.StringVar(value="Idle")
        self.fingerprint_status_var = tk.StringVar(value="Ready.")
        self.progress_var = tk.DoubleVar(value=0)
        self.library_path_var = tk.StringVar(value=library_path or "")
        self.playlist_path_var = tk.StringVar(
            value=self._default_playlist_folder(library_path)
        )
        self.update_playlists_var = tk.BooleanVar(value=False)
        self.quarantine_var = tk.BooleanVar(value=True)
        self.delete_losers_var = tk.BooleanVar(value=False)
        self._execute_confirmation_pending = False
        self.execute_label_var = tk.StringVar(value="Execute")
        cfg = load_config()
        self.show_artwork_variants_var = tk.BooleanVar(
            value=cfg.get("duplicate_finder_show_artwork_variants", True)
        )
        self._preview_trace_enabled = bool(cfg.get("duplicate_finder_debug_trace", False))
        self._preview_trace = deque(maxlen=50) if self._preview_trace_enabled else None
        self.show_noop_groups_var = tk.BooleanVar(value=False)
        self.group_disposition_var = tk.StringVar(value="")
        self.group_disposition_overrides: dict[str, str] = {}
        self._selected_group_id: str | None = None
        self.preview_json_path: str | None = None
        self.preview_html_path: str | None = None
        self.execution_report_path: str | None = None
        self._plan = None
        self._progress_weights = {
            "fingerprinting": 0.7,
            "grouping": 0.2,
            "preview": 0.1,
        }
        self._dup_preview_heartbeat_running = False
        self._dup_preview_heartbeat_start: float | None = None
        self._dup_preview_heartbeat_last_tick: float | None = None
        self._dup_preview_heartbeat_count = 0
        self._dup_preview_heartbeat_interval = 0.1
        self._dup_preview_heartbeat_stall_logged = False
        self._groups_view_update_id = 0
        self._after_ids: set[str] = set()
        self._closing = False

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(
            container,
            text="Duplicate Finder",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            container,
            text=(
                "Generate a preview of duplicate groups and execute the consolidation plan. "
                "Execution writes backups and reports under the library Docs folder."
            ),
            foreground="#555",
            wraplength=520,
        ).pack(anchor="w", pady=(0, 8))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Destroy>", self._on_destroy)

        # Library selection
        lib_frame = ttk.LabelFrame(container, text="Library Selection")
        lib_frame.pack(fill="x", pady=(0, 10))

        lib_row = ttk.Frame(lib_frame)
        lib_row.pack(fill="x", padx=8, pady=6)
        ttk.Label(lib_row, text="Library Root").pack(side="left")
        lib_entry = ttk.Entry(lib_row, textvariable=self.library_path_var, width=60)
        lib_entry.pack(side="left", padx=6, expand=True, fill="x")
        ttk.Button(lib_row, text="Browse…", command=self._browse_library).pack(
            side="left"
        )

        playlist_row = ttk.Frame(lib_frame)
        playlist_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(playlist_row, text="Playlist Folder").pack(side="left")
        playlist_entry = ttk.Entry(
            playlist_row, textvariable=self.playlist_path_var, width=60
        )
        playlist_entry.pack(side="left", padx=6, expand=True, fill="x")
        ttk.Button(playlist_row, text="Browse…", command=self._browse_playlist).pack(
            side="left"
        )

        # Controls
        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 10))
        self.scan_btn = ttk.Button(controls, text="Scan Library", command=self._handle_scan)
        self.scan_btn.pack(side="left", padx=(0, 6))
        self.preview_btn = ttk.Button(controls, text="Preview", command=self._handle_preview)
        self.preview_btn.pack(side="left", padx=(0, 6))
        self.open_report_btn = ttk.Button(
            controls, text="Open Report", command=self._open_execution_report, state="disabled"
        )
        self.open_report_btn.pack(side="left", padx=(0, 6))
        self.execute_btn = ttk.Button(
            controls,
            textvariable=self.execute_label_var,
            command=self._handle_execute,
        )
        self.execute_btn.pack(
            side="left", padx=(0, 12)
        )
        ttk.Button(controls, text="⚙️ Thresholds", command=self._open_threshold_settings).pack(
            side="left", padx=(0, 12)
        )
        ttk.Checkbutton(
            controls,
            text="Show different artwork variants",
            variable=self.show_artwork_variants_var,
            command=self._toggle_show_artwork_variants,
        ).pack(side="left", padx=(0, 10))
        show_noop_toggle = ttk.Checkbutton(
            controls,
            text="Show no-op groups",
            variable=self.show_noop_groups_var,
            command=self._toggle_show_noop_groups,
        )
        show_noop_toggle.pack(side="left", padx=(0, 10))
        Tooltip(
            show_noop_toggle,
            lambda: "When unchecked, hides groups with no planned operations.",
        )
        ttk.Checkbutton(
            controls,
            text="Update Playlists",
            variable=self.update_playlists_var,
            command=self._toggle_update_playlists,
        ).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(
            controls,
            text="Quarantine Duplicates",
            variable=self.quarantine_var,
            command=self._toggle_quarantine,
        ).pack(side="left")
        ttk.Checkbutton(
            controls,
            text="Delete losers (permanent)",
            variable=self.delete_losers_var,
            command=self._toggle_delete_losers,
        ).pack(side="left", padx=(10, 0))

        # Progress + status
        status_frame = ttk.Frame(container)
        status_frame.pack(fill="x", pady=(0, 10))
        ttk.Progressbar(
            status_frame,
            maximum=100,
            variable=self.progress_var,
        ).pack(fill="x", padx=(0, 10), side="left", expand=True)
        ttk.Label(status_frame, textvariable=self.status_var, width=18).pack(
            side="left"
        )

        # Log box
        log_box = ttk.LabelFrame(container, text="Log")
        log_box.pack(fill="both", expand=True, pady=(0, 10))
        self.log_text = ScrolledText(log_box, height=6, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        if self._preview_trace_enabled:
            self._log_action("Debug trace enabled")

        # Results area
        results = ttk.LabelFrame(container, text="Results")
        results.pack(fill="both", expand=True)
        results.columnconfigure(0, weight=1)
        results.columnconfigure(1, weight=1)

        groups_frame = ttk.LabelFrame(results, text="Duplicate Groups")
        groups_frame.grid(row=0, column=0, sticky="nsew", padx=(6, 3), pady=6)
        groups_frame.configure(height=320)
        groups_frame.grid_propagate(False)
        ttk.Label(
            groups_frame,
            textvariable=self.fingerprint_status_var,
            anchor="w",
            foreground="#555",
        ).pack(fill="x", padx=6, pady=(6, 0))
        cols = ("group", "title", "count", "status")
        self.groups_tree = ttk.Treeview(
            groups_frame,
            columns=cols,
            show="headings",
            height=8,
        )
        for cid, heading in zip(
            cols, ("Group ID", "Track Title", "Count", "Status")
        ):
            self.groups_tree.heading(cid, text=heading)
            width = 90 if cid in ("group", "count") else 160
            self.groups_tree.column(cid, width=width, anchor="w")
        self.groups_tree.pack(fill="both", expand=True, padx=6, pady=6)
        self.groups_tree.bind("<<TreeviewSelect>>", self._on_group_select)

        inspector = ttk.LabelFrame(results, text="Group Details")
        inspector.grid(row=0, column=1, sticky="nsew", padx=(3, 6), pady=6)
        disposition_row = ttk.Frame(inspector)
        disposition_row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(disposition_row, text="Group disposition").pack(side="left")
        self.group_disposition_menu = ttk.Combobox(
            disposition_row,
            textvariable=self.group_disposition_var,
            state="disabled",
            width=22,
            values=("Default (global)", "Retain", "Quarantine", "Delete"),
        )
        self.group_disposition_menu.pack(side="left", padx=(6, 0))
        self.group_disposition_menu.bind("<<ComboboxSelected>>", self._on_group_disposition_change)
        self.group_details = ScrolledText(inspector, height=12, state="disabled", wrap="word")
        self.group_details.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._log_action("Duplicate Finder initialized")

    def _clear_plan(self, note: str | None = None) -> None:
        self._plan = None
        self.group_disposition_overrides.clear()
        self.groups_tree.delete(*self.groups_tree.get_children())
        self.group_details.configure(state="normal")
        self.group_details.delete("1.0", "end")
        self.group_details.insert("end", "Generate a plan to view duplicate groups.")
        self.group_details.configure(state="disabled")
        self._reset_group_selection()
        self._set_fingerprint_status("Ready.")
        self._reset_execute_confirmation()
        if note:
            self._log_action(note)

    def _reset_execute_confirmation(self) -> None:
        self._execute_confirmation_pending = False
        self.execute_label_var.set("Execute")

    def _set_scan_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        if hasattr(self, "scan_btn"):
            self.scan_btn.config(state=state)
        if hasattr(self, "preview_btn"):
            self.preview_btn.config(state=state)
        self.execute_btn.config(state=state)

    def _default_playlist_folder(self, library_path: str) -> str:
        if library_path:
            candidate = os.path.join(library_path, "Playlists")
            if os.path.isdir(candidate):
                return candidate
            return library_path
        return ""

    def _schedule_after(self, delay_ms: int, callable_obj, *args) -> str | None:
        if self._closing:
            return None
        after_id: str | None = None

        def _run() -> None:
            if after_id is not None:
                self._after_ids.discard(after_id)
            if self._closing:
                return
            callable_obj(*args)

        after_id = self.after(delay_ms, _run)
        self._after_ids.add(after_id)
        return after_id

    def _cancel_scheduled_callbacks(self) -> None:
        for after_id in list(self._after_ids):
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                pass
            self._after_ids.discard(after_id)

    def _prepare_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_preview_heartbeat()
        self._cancel_scheduled_callbacks()

    def _on_close(self) -> None:
        self._prepare_close()
        self.destroy()

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self._prepare_close()

    def destroy(self) -> None:
        self._prepare_close()
        super().destroy()

    def _widget_alive(self, widget: tk.Widget | None) -> bool:
        if self._closing or widget is None:
            return False
        try:
            return bool(self.winfo_exists()) and bool(widget.winfo_exists())
        except tk.TclError:
            return False

    def _log_action(self, message: str) -> None:
        if not self._widget_alive(self.log_text):
            return
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"{timestamp} {message}"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _record_preview_trace(self, label: str) -> None:
        if not self._preview_trace_enabled or self._preview_trace is None:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        self._preview_trace.append((timestamp, label))

    def _queue_preview_trace(self, label: str) -> None:
        if not self._preview_trace_enabled or self._preview_trace is None:
            return
        self._schedule_after(0, self._record_preview_trace, label)

    def _emit_preview_trace(self, header: str = "Duplicate Finder preview trace") -> None:
        if not self._preview_trace_enabled or not self._preview_trace:
            return
        entries = "\n".join(f"{ts} {label}" for ts, label in self._preview_trace)
        logger.error("%s\n%s", header, entries)

    def _current_threshold_settings(self) -> dict[str, float]:
        cfg = load_config()
        return {
            "exact_duplicate_threshold": float(cfg.get("exact_duplicate_threshold", EXACT_DUPLICATE_THRESHOLD)),
            "near_duplicate_threshold": float(cfg.get("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD)),
            "mixed_codec_threshold_boost": float(cfg.get("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST)),
            "artwork_vastly_different_threshold": float(
                cfg.get("artwork_vastly_different_threshold", ARTWORK_VASTLY_DIFFERENT_THRESHOLD)
            ),
            "preview_artwork_max_dim": float(cfg.get("preview_artwork_max_dim", PREVIEW_ARTWORK_MAX_DIM)),
            "preview_artwork_quality": float(cfg.get("preview_artwork_quality", PREVIEW_ARTWORK_QUALITY)),
        }

    def _current_fingerprint_settings(self) -> dict[str, float | int | bool]:
        cfg = load_config()
        return {
            "trim_silence": bool(cfg.get("trim_silence", FP_TRIM_SILENCE)),
            "fingerprint_offset_ms": int(cfg.get("fingerprint_offset_ms", FP_OFFSET_MS)),
            "fingerprint_duration_ms": int(cfg.get("fingerprint_duration_ms", FP_DURATION_MS)),
            "fingerprint_silence_threshold_db": float(cfg.get("fingerprint_silence_threshold_db", FP_SILENCE_THRESHOLD_DB)),
            "fingerprint_silence_min_len_ms": int(cfg.get("fingerprint_silence_min_len_ms", FP_SILENCE_MIN_LEN_MS)),
            "fingerprint_trim_lead_max_ms": int(cfg.get("fingerprint_trim_lead_max_ms", FP_TRIM_LEAD_MAX_MS)),
            "fingerprint_trim_trail_max_ms": int(cfg.get("fingerprint_trim_trail_max_ms", FP_TRIM_TRAIL_MAX_MS)),
            "fingerprint_trim_padding_ms": int(cfg.get("fingerprint_trim_padding_ms", FP_TRIM_PADDING_MS)),
            "allow_mismatched_edits": bool(cfg.get("allow_mismatched_edits", ALLOW_MISMATCHED_EDITS)),
        }

    def _thresholds_match(
        self,
        plan_settings: Mapping[str, float] | None,
        current_settings: Mapping[str, float],
        *,
        tolerance: float = 1e-6,
    ) -> bool:
        if not plan_settings:
            return False
        for key, value in current_settings.items():
            if key not in plan_settings:
                return False
            try:
                if abs(float(plan_settings[key]) - float(value)) > tolerance:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _fingerprint_settings_match(
        self,
        plan_settings: Mapping[str, float | int | bool] | None,
        current_settings: Mapping[str, float | int | bool],
        *,
        tolerance: float = 1e-6,
    ) -> bool:
        if not plan_settings:
            return False
        for key, value in current_settings.items():
            if key not in plan_settings:
                return False
            try:
                plan_value = plan_settings[key]
                if isinstance(value, bool) or isinstance(plan_value, bool):
                    if bool(plan_value) != bool(value):
                        return False
                else:
                    if abs(float(plan_value) - float(value)) > tolerance:
                        return False
            except (TypeError, ValueError):
                return False
        return True

    def _default_group_disposition(self) -> str:
        if not self.quarantine_var.get():
            return "retain"
        if self.delete_losers_var.get():
            return "delete"
        return "quarantine"

    def _reset_group_selection(self) -> None:
        self._selected_group_id = None
        self.group_disposition_var.set("Default (global)")
        self.group_disposition_menu.configure(state="disabled")

    def _update_groups_view(self, plan) -> None:
        if not self._widget_alive(self.groups_tree):
            return
        self.groups_tree.delete(*self.groups_tree.get_children())
        groups_batch_size = 150
        self._groups_view_update_id += 1
        update_id = self._groups_view_update_id
        prior_fingerprint_status = self.fingerprint_status_var.get()
        show_noop_groups = self.show_noop_groups_var.get()
        group_rows = []
        for group in plan.groups:
            actions = _planned_actions(group)
            if _is_noop_group(actions) and not show_noop_groups:
                continue
            title = plan and group.planned_winner_tags.get("title") if hasattr(group, "planned_winner_tags") else None
            status = "Review" if group.review_flags else "Ready"
            group_rows.append(
                {
                    "iid": group.group_id,
                    "values": (
                        group.group_id,
                        title or os.path.basename(group.winner_path),
                        len(group.losers) + 1,
                        status,
                    ),
                    "tags": ("review",) if group.review_flags else (),
                }
            )

        total_rows = len(group_rows)

        def insert_chunk(start_index: int = 0) -> None:
            if update_id != self._groups_view_update_id:
                return
            if not self._widget_alive(self.groups_tree):
                return
            end_index = min(start_index + groups_batch_size, total_rows)
            for row in group_rows[start_index:end_index]:
                self.groups_tree.insert(
                    "",
                    "end",
                    iid=row["iid"],
                    values=row["values"],
                    tags=row["tags"],
                )
            if total_rows:
                self.fingerprint_status_var.set(f"Loading groups {end_index}/{total_rows}…")
            if end_index < total_rows:
                self._schedule_after(0, insert_chunk, end_index)
                return
            self.groups_tree.tag_configure("review", background="#fff3cd")
            self.group_details.configure(state="normal")
            self.group_details.delete("1.0", "end")
            self.group_details.insert("end", "Select a group to view details.")
            self.group_details.configure(state="disabled")
            self._reset_group_selection()
            self.fingerprint_status_var.set(prior_fingerprint_status)

        self._schedule_after(0, insert_chunk)

    def _toggle_show_noop_groups(self) -> None:
        state = "enabled" if self.show_noop_groups_var.get() else "disabled"
        self._log_action(f"Show no-op groups {state}")
        if self._plan:
            self._update_groups_view(self._plan)

    def _on_group_select(self, event=None) -> None:
        sel = self.groups_tree.selection()
        if not sel or not self._plan:
            return
        group_id = sel[0]
        group = next((g for g in self._plan.groups if g.group_id == group_id), None)
        if group:
            self._selected_group_id = group.group_id
            self.group_disposition_menu.configure(state="readonly")
            override = self.group_disposition_overrides.get(group.group_id)
            if override == "retain":
                self.group_disposition_var.set("Retain")
            elif override == "quarantine":
                self.group_disposition_var.set("Quarantine")
            elif override == "delete":
                self.group_disposition_var.set("Delete")
            else:
                self.group_disposition_var.set("Default (global)")
            self._render_group_details(group)

    def _render_group_details(self, group) -> None:
        default_disp = self._default_group_disposition()
        override = self.group_disposition_overrides.get(group.group_id)
        disposition_label = f"{override} (override)" if override else f"{default_disp} (default)"
        lines = [
            f"Winner: {group.winner_path}",
            f"Losers: {len(group.losers)}",
            f"Group disposition: {disposition_label}",
            "Dispositions:",
        ]
        for loser in group.losers:
            disp = group.loser_disposition.get(loser, default_disp)
            playlist = group.playlist_rewrites.get(loser, "n/a")
            lines.append(f"  - {loser} → {disp} (playlist → {playlist})")

        lines.append("Quality rationale:")
        for reason in group.winner_quality.get("reasons") or []:
            lines.append(f"  • {reason}")

        if group.review_flags:
            lines.append("Review flags:")
            for flag in group.review_flags:
                lines.append(f"  ! {flag}")

        lines.append("Tag changes:")
        if not group.metadata_changes:
            lines.append("  (none)")
        else:
            for key, diff in sorted(group.metadata_changes.items()):
                lines.append(f"  {key}: {diff.get('from')} → {diff.get('to')}")

        lines.append(
            f"Playlist impact: {group.playlist_impact.playlists} playlists, {group.playlist_impact.entries} entries"
        )

        self.group_details.configure(state="normal")
        self.group_details.delete("1.0", "end")
        self.group_details.insert("end", "\n".join(lines))
        self.group_details.configure(state="disabled")

    def _on_group_disposition_change(self, event=None) -> None:
        if not self._plan or not self._selected_group_id:
            return
        group = next(
            (g for g in self._plan.groups if g.group_id == self._selected_group_id),
            None,
        )
        if not group:
            return
        selection = self.group_disposition_var.get()
        choice_map = {
            "Default (global)": None,
            "Retain": "retain",
            "Quarantine": "quarantine",
            "Delete": "delete",
        }
        disposition = choice_map.get(selection)
        if disposition is None:
            self.group_disposition_overrides.pop(group.group_id, None)
            disposition = self._default_group_disposition()
            self._log_action(f"Group {group.group_id} disposition reset to default ({disposition}).")
        else:
            self.group_disposition_overrides[group.group_id] = disposition
            self._log_action(f"Group {group.group_id} disposition override set to {disposition}.")

        for loser in group.losers:
            group.loser_disposition[loser] = disposition

        self._plan.refresh_plan_signature()
        self._render_group_details(group)
        self._reset_preview_if_needed()
        self._reset_execute_confirmation()

    def _count_group_dispositions(self) -> dict[str, int]:
        counts = {"retain": 0, "quarantine": 0, "delete": 0}
        if not self._plan:
            return counts
        default_disposition = self._default_group_disposition()
        for group in self._plan.groups:
            for loser in group.losers:
                disp = group.loser_disposition.get(loser, default_disposition)
                if disp in counts:
                    counts[disp] += 1
        return counts

    def _set_status(self, status: str, progress: float | None = None) -> None:
        self.status_var.set(status)
        if progress is not None:
            self.progress_var.set(progress)

    def _set_fingerprint_status(self, status: str) -> None:
        self.fingerprint_status_var.set(status)

    def _weighted_progress(self, phase: str, ratio: float) -> float:
        order = ("fingerprinting", "grouping", "preview")
        total = 0.0
        for step in order:
            weight = self._progress_weights.get(step, 0.0)
            if step == phase:
                clamped = max(0.0, min(1.0, ratio))
                total += weight * clamped
                break
            total += weight
        return max(0.0, min(100.0, total * 100.0))

    def _format_track_label(self, path: str, audio: object | None) -> str:
        if not audio:
            return os.path.basename(path)
        tags = getattr(audio, "tags", None) or {}

        def _get_tag(keys: tuple[str, ...]) -> str | None:
            for key in keys:
                if key in tags:
                    raw = tags[key]
                    if hasattr(raw, "text"):
                        raw = raw.text
                    if isinstance(raw, (list, tuple)):
                        if raw:
                            return str(raw[0])
                    elif raw is not None:
                        return str(raw)
            return None

        title = _get_tag(("title", "TIT2", "\xa9nam"))
        artist = _get_tag(("artist", "TPE1", "\xa9ART"))
        if artist and title:
            return f"{artist} - {title}"
        if title:
            return title
        return os.path.basename(path)

    def _validate_library_root(self) -> str | None:
        path = self.library_path_var.get().strip()
        if not path:
            messagebox.showwarning(
                "Library Required", "Please select a library root before continuing."
            )
            self._log_action("Library validation failed: no library selected")
            self._set_status("Idle", progress=0)
            return None
        if not os.path.isdir(path):
            messagebox.showwarning(
                "Library Missing", f"The selected library path does not exist:\n{path}"
            )
            self._log_action(f"Library validation failed: path not found ({path})")
            self._set_status("Idle", progress=0)
            return None
        return path

    def _browse_library(self) -> None:
        initial_dir = self.library_path_var.get().strip() or load_last_path() or os.getcwd()
        chosen = filedialog.askdirectory(
            title="Select Library Root",
            initialdir=initial_dir,
        )
        if not chosen:
            return
        self.library_path_var.set(chosen)
        self.playlist_path_var.set(self._default_playlist_folder(chosen))
        self._log_action(f"Library path updated to {chosen}")
        save_last_path(chosen)

    def _browse_playlist(self) -> None:
        initial_dir = self.playlist_path_var.get().strip() or load_last_path() or os.getcwd()
        chosen = filedialog.askdirectory(
            title="Select Playlist Folder",
            initialdir=initial_dir,
        )
        if not chosen:
            return
        self.playlist_path_var.set(chosen)
        self._log_action(f"Playlist folder updated to {chosen}")
        save_last_path(chosen)

    def _open_threshold_settings(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Threshold Settings")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        cfg = load_config()

        exact_var = tk.StringVar(value=str(cfg.get("exact_duplicate_threshold", EXACT_DUPLICATE_THRESHOLD)))
        near_var = tk.StringVar(value=str(cfg.get("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD)))
        mixed_var = tk.StringVar(value=str(cfg.get("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST)))
        offset_var = tk.StringVar(value=str(cfg.get("fingerprint_offset_ms", FP_OFFSET_MS)))
        duration_var = tk.StringVar(value=str(cfg.get("fingerprint_duration_ms", FP_DURATION_MS)))
        silence_db_var = tk.StringVar(value=str(cfg.get("fingerprint_silence_threshold_db", FP_SILENCE_THRESHOLD_DB)))
        silence_len_var = tk.StringVar(value=str(cfg.get("fingerprint_silence_min_len_ms", FP_SILENCE_MIN_LEN_MS)))
        trim_lead_var = tk.StringVar(value=str(cfg.get("fingerprint_trim_lead_max_ms", FP_TRIM_LEAD_MAX_MS)))
        trim_trail_var = tk.StringVar(value=str(cfg.get("fingerprint_trim_trail_max_ms", FP_TRIM_TRAIL_MAX_MS)))
        trim_padding_var = tk.StringVar(value=str(cfg.get("fingerprint_trim_padding_ms", FP_TRIM_PADDING_MS)))

        frame = ttk.Frame(dlg, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Tune duplicate matching and fingerprint windowing thresholds.",
            foreground="#555",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        def _row(label: str, var: tk.StringVar, row: int) -> None:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=var, width=16).grid(row=row, column=1, sticky="ew", pady=2)

        row_idx = 1
        _row("Exact duplicate threshold", exact_var, row_idx)
        row_idx += 1
        _row("Near duplicate threshold", near_var, row_idx)
        row_idx += 1
        _row("Mixed-codec boost", mixed_var, row_idx)
        row_idx += 1
        _row("Fingerprint offset (ms)", offset_var, row_idx)
        row_idx += 1
        _row("Fingerprint duration (ms)", duration_var, row_idx)
        row_idx += 1
        _row("Silence threshold (dB)", silence_db_var, row_idx)
        row_idx += 1
        _row("Silence min length (ms)", silence_len_var, row_idx)
        row_idx += 1
        _row("Trim lead max (ms)", trim_lead_var, row_idx)
        row_idx += 1
        _row("Trim trail max (ms)", trim_trail_var, row_idx)
        row_idx += 1
        _row("Trim padding (ms)", trim_padding_var, row_idx)
        row_idx += 1

        frame.columnconfigure(1, weight=1)

        def _parse_float(raw: str, label: str, *, min_value: float | None = None, max_value: float | None = None) -> float | None:
            try:
                value = float(raw)
            except ValueError:
                messagebox.showwarning("Invalid Value", f"{label} must be a number.")
                return None
            if min_value is not None and value < min_value:
                messagebox.showwarning(
                    "Invalid Value", f"{label} must be at least {min_value}."
                )
                return None
            if max_value is not None and value > max_value:
                messagebox.showwarning(
                    "Invalid Value", f"{label} must be at most {max_value}."
                )
                return None
            return value

        def _parse_int(raw: str, label: str, *, min_value: int | None = None) -> int | None:
            try:
                value = int(float(raw))
            except ValueError:
                messagebox.showwarning("Invalid Value", f"{label} must be a whole number.")
                return None
            if min_value is not None and value < min_value:
                messagebox.showwarning(
                    "Invalid Value", f"{label} must be at least {min_value}."
                )
                return None
            return value

        def _save() -> None:
            exact = _parse_float(exact_var.get().strip(), "Exact duplicate threshold", min_value=0.0)
            if exact is None:
                return
            near = _parse_float(near_var.get().strip(), "Near duplicate threshold", min_value=0.0)
            if near is None:
                return
            if near < exact:
                messagebox.showwarning(
                    "Invalid Value",
                    "Near duplicate threshold must be greater than or equal to the exact threshold.",
                )
                return
            mixed = _parse_float(mixed_var.get().strip(), "Mixed-codec boost", min_value=0.0)
            if mixed is None:
                return
            offset = _parse_int(offset_var.get().strip(), "Fingerprint offset (ms)", min_value=0)
            if offset is None:
                return
            duration = _parse_int(duration_var.get().strip(), "Fingerprint duration (ms)", min_value=0)
            if duration is None:
                return
            silence_db = _parse_float(
                silence_db_var.get().strip(),
                "Silence threshold (dB)",
                min_value=-120.0,
                max_value=0.0,
            )
            if silence_db is None:
                return
            silence_len = _parse_int(
                silence_len_var.get().strip(),
                "Silence min length (ms)",
                min_value=0,
            )
            if silence_len is None:
                return
            trim_lead = _parse_int(
                trim_lead_var.get().strip(),
                "Trim lead max (ms)",
                min_value=0,
            )
            if trim_lead is None:
                return
            trim_trail = _parse_int(
                trim_trail_var.get().strip(),
                "Trim trail max (ms)",
                min_value=0,
            )
            if trim_trail is None:
                return
            trim_padding = _parse_int(
                trim_padding_var.get().strip(),
                "Trim padding (ms)",
                min_value=0,
            )
            if trim_padding is None:
                return

            cfg["exact_duplicate_threshold"] = exact
            cfg["near_duplicate_threshold"] = near
            cfg["mixed_codec_threshold_boost"] = mixed
            cfg["fingerprint_offset_ms"] = offset
            cfg["fingerprint_duration_ms"] = duration
            cfg["fingerprint_silence_threshold_db"] = silence_db
            cfg["fingerprint_silence_min_len_ms"] = silence_len
            cfg["fingerprint_trim_lead_max_ms"] = trim_lead
            cfg["fingerprint_trim_trail_max_ms"] = trim_trail
            cfg["fingerprint_trim_padding_ms"] = trim_padding
            save_config(cfg)

            self.preview_json_path = None
            self.preview_html_path = None
            self.execution_report_path = None
            self.open_report_btn.config(state="disabled")
            self._reset_execute_confirmation()
            library_root = self.library_path_var.get().strip()
            if library_root:
                docs_dir = os.path.join(library_root, "Docs")
                preview_json = os.path.join(docs_dir, "duplicate_preview.json")
                preview_html = os.path.join(docs_dir, "duplicate_preview.html")
                for preview_path in (preview_json, preview_html):
                    try:
                        if os.path.exists(preview_path):
                            os.remove(preview_path)
                    except OSError:
                        pass
            self._clear_plan("Threshold settings updated; regenerate preview or scan to apply changes.")
            dlg.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row_idx, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btn_frame, text="OK", command=_save).pack()
        dlg.wait_window()

    def _gather_tracks(
        self,
        library_root: str,
        *,
        log_callback: Callable[[str], None] | None = None,
        status_callback: Callable[[str, float], None] | None = None,
        fingerprint_status_callback: Callable[[str], None] | None = None,
        idle_callback: Callable[[], None] | None = None,
    ) -> tuple[list[dict[str, object]], int, int]:
        if not library_root:
            return [], 0, 0
        docs_dir = os.path.join(library_root, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        db_path = os.path.join(docs_dir, ".duplicate_fingerprints.db")
        ensure_fingerprint_cache(db_path)
        excluded_dirs = {"not sorted", "playlists"}

        audio_paths: list[str] = []
        for dirpath, _dirs, files in os.walk(library_root):
            rel = os.path.relpath(dirpath, library_root)
            parts = {p.lower() for p in rel.split(os.sep)}
            if excluded_dirs & parts:
                continue
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTS:
                    audio_paths.append(os.path.join(dirpath, fname))

        tracks: list[dict[str, object]] = []
        total = len(audio_paths)
        if total == 0:
            if fingerprint_status_callback:
                fingerprint_status_callback("No eligible audio files found.")
            if status_callback:
                status_callback("Idle", progress=0)
            return tracks, 0, 0
        if fingerprint_status_callback:
            fingerprint_status_callback(f"Fingerprinting 0/{total}")
        if status_callback:
            status_callback("Fingerprinting…", progress=self._weighted_progress("fingerprinting", 0))
        if idle_callback:
            idle_callback()
        last_update = time.monotonic()
        update_interval = 0.2
        def _fingerprint_worker(target_path: str) -> tuple[str, int | None, str | None, str | None]:
            try:
                duration, fp = _compute_fp(target_path)
                return target_path, duration, fp, None
            except Exception as exc:  # pragma: no cover - depends on external files
                return target_path, None, None, str(exc)

        def _normalized_key(text: str | None) -> str | None:
            if not text:
                return None
            return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

        def _extract_metadata(path: str) -> dict[str, object]:
            ext = os.path.splitext(path)[1].lower()
            bitrate = 0
            sample_rate = 0
            bit_depth = 0
            try:
                audio = MutagenFile(path)
                info = getattr(audio, "info", None)
                if info:
                    bitrate = int(getattr(info, "bitrate", 0) or 0)
                    sample_rate = int(
                        getattr(info, "sample_rate", 0)
                        or getattr(info, "samplerate", 0)
                        or 0
                    )
                    bit_depth = int(
                        getattr(info, "bits_per_sample", 0)
                        or getattr(info, "bitdepth", 0)
                        or 0
                    )
            except Exception:
                pass
            tags: dict[str, object] = {}
            try:
                tags = read_tags(path)
            except Exception:
                tags = {}
            artist_value = tags.get("artist") or tags.get("albumartist")
            title_value = tags.get("title")
            album_value = tags.get("album")
            normalized_artist = _normalized_key(artist_value if isinstance(artist_value, str) else None)
            normalized_title = _normalized_key(title_value if isinstance(title_value, str) else None)
            normalized_album = _normalized_key(album_value if isinstance(album_value, str) else None)
            return {
                "ext": ext,
                "bitrate": bitrate,
                "sample_rate": sample_rate,
                "bit_depth": bit_depth,
                "normalized_artist": normalized_artist,
                "normalized_title": normalized_title,
                "normalized_album": normalized_album,
            }

        def _needs_metadata_refresh(metadata: dict[str, object] | None) -> bool:
            if metadata is None:
                return True
            return any(metadata.get(key) is None for key in ("bitrate", "sample_rate", "bit_depth"))

        def _metadata_for_payload(path: str, metadata: dict[str, object] | None) -> dict[str, object]:
            payload = dict(metadata or {})
            ext = payload.get("ext") or os.path.splitext(path)[1].lower()
            if isinstance(ext, str) and ext and not ext.startswith("."):
                ext = f".{ext}"
            payload["ext"] = ext
            return payload

        def _track_payload(
            path: str,
            fp: str,
            fingerprint_trace: dict[str, object],
            metadata: dict[str, object],
        ) -> dict[str, object]:
            ext = metadata.get("ext") or os.path.splitext(path)[1].lower()
            if isinstance(ext, str) and ext and not ext.startswith("."):
                ext = f".{ext}"
            return {
                "path": path,
                "fingerprint": fp,
                "ext": ext,
                "bitrate": int(metadata.get("bitrate") or 0),
                "sample_rate": int(metadata.get("sample_rate") or 0),
                "bit_depth": int(metadata.get("bit_depth") or 0),
                "fingerprint_trace": dict(fingerprint_trace),
                "discovery": {
                    "scan_roots": [library_root],
                    "excluded_dirs": sorted(excluded_dirs),
                    "skipped_by_filter": False,
                },
            }

        tracks_map: dict[str, dict[str, object]] = {}
        pending_paths: list[str] = []
        completed = 0
        cached_count = 0
        computed_count = 0
        failure_count = 0
        missing_count = 0
        metadata_refresh_count = 0
        sorted_paths = sorted(audio_paths)
        for path in sorted_paths:
            fingerprint_trace: dict[str, object] = {}
            fp, cached_metadata = get_cached_fingerprint_metadata(
                path,
                db_path,
                log_callback=log_callback,
                trace=fingerprint_trace,
            )
            if fp:
                metadata = _metadata_for_payload(path, cached_metadata)
                if _needs_metadata_refresh(cached_metadata):
                    metadata = _extract_metadata(path)
                    metadata_refresh_count += 1
                    if not store_fingerprint(
                        path,
                        db_path,
                        None,
                        fp,
                        log_callback=log_callback,
                        ext=str(metadata.get("ext") or ""),
                        bitrate=int(metadata.get("bitrate") or 0),
                        sample_rate=int(metadata.get("sample_rate") or 0),
                        bit_depth=int(metadata.get("bit_depth") or 0),
                        normalized_artist=metadata.get("normalized_artist"),
                        normalized_title=metadata.get("normalized_title"),
                        normalized_album=metadata.get("normalized_album"),
                    ):
                        failure_count += 1
                tracks_map[path] = _track_payload(path, fp, fingerprint_trace, metadata)
                completed += 1
                cached_count += 1
            else:
                source = fingerprint_trace.get("source")
                if source in {"stat_error", "cache_error"}:
                    failure_count += 1
                    completed += 1
                else:
                    pending_paths.append(path)
            now = time.monotonic()
            if completed == total or now - last_update >= update_interval:
                if fingerprint_status_callback:
                    fingerprint_status_callback(f"Fingerprinting {completed}/{total}")
                if status_callback:
                    status_callback(
                        "Fingerprinting…",
                        progress=self._weighted_progress("fingerprinting", completed / total),
                    )
                if idle_callback:
                    idle_callback()
                last_update = now

        if not pending_paths and metadata_refresh_count == 0 and failure_count == 0:
            refresh_message = "Catalog up to date; using cached metadata."
            if log_callback:
                log_callback(refresh_message)
            if fingerprint_status_callback:
                fingerprint_status_callback(refresh_message)
        elif log_callback:
            log_callback(
                "Fingerprint cache scan complete: "
                f"{cached_count} cached, {len(pending_paths)} pending, "
                f"{metadata_refresh_count} metadata refresh, {failure_count} failures."
            )

        max_workers = min(8, os.cpu_count() or 4)
        if pending_paths:
            if log_callback:
                log_callback(f"Computing {len(pending_paths)} missing fingerprints…")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_fingerprint_worker, path): path for path in pending_paths}
                for fut in as_completed(futures):
                    path = futures[fut]
                    error = None
                    try:
                        path, duration, fp, error = fut.result()
                    except Exception as exc:
                        duration = None
                        fp = None
                        error = str(exc)
                    completed += 1
                    now = time.monotonic()
                    if completed == total or now - last_update >= update_interval:
                        if fingerprint_status_callback:
                            fingerprint_status_callback(f"Fingerprinting {completed}/{total}")
                        if status_callback:
                            status_callback(
                                "Fingerprinting…",
                                progress=self._weighted_progress("fingerprinting", completed / total),
                            )
                        if idle_callback:
                            idle_callback()
                        last_update = now
                    if error:
                        failure_count += 1
                        if log_callback:
                            log_callback(f"! Fingerprint failed for {path}: {error}")
                        continue
                    if not fp:
                        missing_count += 1
                        if log_callback:
                            log_callback(f"! Missing fingerprint for {path}")
                        continue
                    computed_count += 1
                    fingerprint_trace = {"source": "computed", "error": ""}
                    metadata = _extract_metadata(path)
                    tracks_map[path] = _track_payload(path, fp, fingerprint_trace, metadata)
                    if not store_fingerprint(
                        path,
                        db_path,
                        duration,
                        fp,
                        log_callback=log_callback,
                        ext=str(metadata.get("ext") or ""),
                        bitrate=int(metadata.get("bitrate") or 0),
                        sample_rate=int(metadata.get("sample_rate") or 0),
                        bit_depth=int(metadata.get("bit_depth") or 0),
                        normalized_artist=metadata.get("normalized_artist"),
                        normalized_title=metadata.get("normalized_title"),
                        normalized_album=metadata.get("normalized_album"),
                        ):
                            failure_count += 1

        tracks = [tracks_map[path] for path in sorted_paths if path in tracks_map]
        if log_callback:
            log_callback(
                "Fingerprinting summary: "
                f"{cached_count} cached, {computed_count} computed, "
                f"{missing_count} missing, {failure_count} failures."
            )
        return tracks, missing_count, failure_count

    def _generate_plan(
        self,
        library_root: str,
        *,
        write_preview: bool,
        threshold_settings: Mapping[str, float],
        fingerprint_settings: Mapping[str, object],
        show_artwork_variants: bool,
        log_callback: Callable[[str], None] | None = None,
        status_callback: Callable[[str, float], None] | None = None,
        fingerprint_status_callback: Callable[[str], None] | None = None,
        idle_callback: Callable[[], None] | None = None,
    ) -> PlanGenerationResult:
        timing_enabled = write_preview
        log_messages: list[str] = []

        def log(msg: str) -> None:
            log_messages.append(msg)
            if log_callback:
                log_callback(msg)

        def report_status(label: str, progress: float) -> None:
            if status_callback:
                status_callback(label, progress=progress)

        def report_fingerprint_status(message: str) -> None:
            if fingerprint_status_callback:
                fingerprint_status_callback(message)

        if timing_enabled:
            plan_start_ts = datetime.now().isoformat(timespec="seconds")
            plan_start_time = time.monotonic()
            logger.info("Preview plan timing start: %s", plan_start_ts)
        try:
            tracks, missing_count, failure_count = self._gather_tracks(
                library_root,
                log_callback=log,
                status_callback=status_callback,
                fingerprint_status_callback=fingerprint_status_callback,
                idle_callback=idle_callback,
            )
            if not tracks:
                return PlanGenerationResult(
                    plan=None,
                    preview_json_path=None,
                    preview_html_path=None,
                    log_messages=log_messages,
                    review_required_count=0,
                    group_count=0,
                    had_tracks=False,
                )
            if missing_count or failure_count:
                log(
                    f"Fingerprinting complete with {missing_count} missing fingerprints and "
                    f"{failure_count} failures."
                )
            log(f"Fingerprinting complete: {len(tracks)} tracks ready for grouping.")
            log("Beginning duplicate grouping…")
            report_fingerprint_status("Grouping duplicates…")
            report_status("Grouping…", progress=self._weighted_progress("grouping", 0))
            exact_threshold = threshold_settings["exact_duplicate_threshold"]
            near_threshold = threshold_settings["near_duplicate_threshold"]
            mixed_codec_boost = threshold_settings["mixed_codec_threshold_boost"]
            log(
                "Duplicate scan thresholds: "
                f"exact={exact_threshold:.3f}, near={near_threshold:.3f}, mixed_codec_boost={mixed_codec_boost:.3f}"
            )
            log(f"Fingerprint settings: {fingerprint_settings}")
            plan = build_consolidation_plan(
                tracks,
                exact_duplicate_threshold=exact_threshold,
                near_duplicate_threshold=near_threshold,
                mixed_codec_threshold_boost=mixed_codec_boost,
                fingerprint_settings=fingerprint_settings,
                threshold_settings=threshold_settings,
                log_callback=log,
            )
            if write_preview:
                self._queue_preview_trace("metadata-read")
                self._queue_preview_trace("artwork-read")
                self._queue_preview_trace("artwork-compress")
            report_status("Grouping…", progress=self._weighted_progress("grouping", 1))

            docs_dir = os.path.join(library_root, "Docs")
            os.makedirs(docs_dir, exist_ok=True)
            preview_json_path = None
            preview_html_path = None
            if write_preview:
                report_fingerprint_status("Generating preview…")
                report_status("Preview…", progress=self._weighted_progress("preview", 0))
                json_path = os.path.join(docs_dir, "duplicate_preview.json")
                export_consolidation_preview(plan, json_path)
                preview_json_path = json_path
                log(f"Audit JSON written to {json_path}")
                html_path = os.path.join(docs_dir, "duplicate_preview.html")
                export_consolidation_preview_html(
                    plan,
                    html_path,
                    show_artwork_variants=show_artwork_variants,
                )
                self._queue_preview_trace("preview-html-write")
                preview_html_path = html_path
                log(f"Preview HTML written to {html_path}")
                report_status("Preview generated", progress=self._weighted_progress("preview", 1))
            else:
                report_status("Plan ready", progress=100)

            log(f"Plan: {len(plan.groups)} groups, review required={plan.review_required_count}")
            if plan.review_required_count:
                log("Review required groups will block execution unless overridden.")
            return PlanGenerationResult(
                plan=plan,
                preview_json_path=preview_json_path,
                preview_html_path=preview_html_path,
                log_messages=log_messages,
                review_required_count=plan.review_required_count,
                group_count=len(plan.groups),
                had_tracks=True,
            )
        finally:
            if timing_enabled:
                plan_elapsed = time.monotonic() - plan_start_time
                plan_end_ts = datetime.now().isoformat(timespec="seconds")
                logger.info(
                    "Preview plan timing end: %s (elapsed %.2fs)",
                    plan_end_ts,
                    plan_elapsed,
                )

    def _load_preview_plan(self, preview_path: str):
        try:
            with open(preview_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self._log_action(f"Preview plan could not be loaded: {exc}")
            return None
        if not isinstance(payload, dict) or "plan" not in payload:
            self._log_action("Preview plan could not be loaded: missing plan payload.")
            return None
        try:
            return consolidation_plan_from_dict(payload["plan"])
        except ValueError as exc:
            self._log_action(f"Preview plan could not be loaded: {exc}")
            return None

    def _handle_scan(self) -> None:
        path = self._validate_library_root()
        if not path:
            return
        self._reset_execute_confirmation()
        self._log_action("Scan clicked")
        self._set_status("Starting scan…", progress=0)
        self._set_fingerprint_status("Starting scan…")
        self._set_scan_controls_enabled(False)
        threshold_settings = self._current_threshold_settings()
        fingerprint_settings = self._current_fingerprint_settings()
        show_artwork_variants = self.show_artwork_variants_var.get()

        def schedule(callable_obj, *args) -> None:
            self._schedule_after(0, callable_obj, *args)

        def log_callback(message: str) -> None:
            schedule(self._log_action, message)

        def status_callback(status: str, progress: float) -> None:
            schedule(self._set_status, status, progress)

        def fingerprint_status_callback(status: str) -> None:
            schedule(self._set_fingerprint_status, status)

        def idle_callback() -> None:
            schedule(self.update_idletasks)

        def finish(result: PlanGenerationResult | None, error: Exception | None = None) -> None:
            if error is not None:
                self._log_action(f"Scan failed: {error}")
                self._set_status("Scan failed", 100)
                messagebox.showerror("Scan Failed", str(error))
                self._set_scan_controls_enabled(True)
                return
            if result is None or not result.had_tracks:
                messagebox.showwarning(
                    "No Tracks",
                    "No audio tracks were found in the selected library.",
                )
                self._set_status("Idle", 0)
                self._set_scan_controls_enabled(True)
                return
            self._plan = result.plan
            self.group_disposition_overrides.clear()
            self._apply_deletion_mode()
            self.preview_json_path = None
            self.preview_html_path = None
            if hasattr(self.master, "_set_duplicate_finder_plan"):
                self.master._set_duplicate_finder_plan(result.plan, path)
            self._set_scan_controls_enabled(True)

        def worker() -> None:
            try:
                result = self._generate_plan(
                    path,
                    write_preview=False,
                    threshold_settings=threshold_settings,
                    fingerprint_settings=fingerprint_settings,
                    show_artwork_variants=show_artwork_variants,
                    log_callback=log_callback,
                    status_callback=status_callback,
                    fingerprint_status_callback=fingerprint_status_callback,
                    idle_callback=idle_callback,
                )
            except Exception as exc:
                logger.exception("Scan generation failed.")
                self._schedule_after(0, finish, None, exc)
                return
            self._schedule_after(0, finish, result, None)

        threading.Thread(target=worker, daemon=True).start()

    def _start_preview_heartbeat(self) -> None:
        self._dup_preview_heartbeat_running = True
        self._dup_preview_heartbeat_start = time.monotonic()
        self._dup_preview_heartbeat_last_tick = None
        self._dup_preview_heartbeat_count = 0
        self._dup_preview_heartbeat_stall_logged = False
        self._schedule_after(
            int(self._dup_preview_heartbeat_interval * 1000),
            self._preview_heartbeat_tick,
        )

    def _preview_heartbeat_tick(self) -> None:
        if not self._dup_preview_heartbeat_running or self._closing:
            return
        now = time.monotonic()
        if self._dup_preview_heartbeat_last_tick is not None:
            gap = now - self._dup_preview_heartbeat_last_tick
            if (
                gap > self._dup_preview_heartbeat_interval * 10
                and not self._dup_preview_heartbeat_stall_logged
            ):
                logger.warning(
                    "Preview heartbeat delayed by %.2fs while preview running.",
                    gap,
                )
                self._dup_preview_heartbeat_stall_logged = True
        self._dup_preview_heartbeat_last_tick = now
        self._dup_preview_heartbeat_count += 1
        self._schedule_after(
            int(self._dup_preview_heartbeat_interval * 1000),
            self._preview_heartbeat_tick,
        )

    def _stop_preview_heartbeat(self) -> None:
        if not self._dup_preview_heartbeat_running:
            return
        self._dup_preview_heartbeat_running = False
        now = time.monotonic()
        elapsed = 0.0
        if self._dup_preview_heartbeat_start is not None:
            elapsed = now - self._dup_preview_heartbeat_start
        if self._dup_preview_heartbeat_count == 0:
            logger.warning(
                "Preview heartbeat did not fire during preview window (%.2fs).",
                elapsed,
            )
        elif (
            self._dup_preview_heartbeat_last_tick is not None
            and now - self._dup_preview_heartbeat_last_tick
            > self._dup_preview_heartbeat_interval * 10
        ):
            stall = now - self._dup_preview_heartbeat_last_tick
            logger.warning(
                "Preview heartbeat stalled for %.2fs before completion.",
                stall,
            )
        logger.info(
            "Preview heartbeat stopped after %.2fs (%s ticks).",
            elapsed,
            self._dup_preview_heartbeat_count,
        )

    def _handle_preview(self) -> None:
        path = self._validate_library_root()
        if not path:
            return
        self._reset_execute_confirmation()
        self._log_action("Preview clicked")
        if self._preview_trace_enabled and self._preview_trace is not None:
            self._preview_trace.clear()
        self._record_preview_trace("preview-start")
        start_ts = datetime.now().isoformat(timespec="seconds")
        start_time = time.monotonic()
        logger.info("Preview timing start: %s", start_ts)
        self._start_preview_heartbeat()
        self._set_fingerprint_status("Starting preview…")
        self._set_status("Preview…", progress=0)
        threshold_settings = self._current_threshold_settings()
        fingerprint_settings = self._current_fingerprint_settings()
        show_artwork_variants = self.show_artwork_variants_var.get()

        def finalize_preview(trace_emit: bool = False) -> None:
            self._record_preview_trace("preview-finish")
            elapsed = time.monotonic() - start_time
            end_ts = datetime.now().isoformat(timespec="seconds")
            self._stop_preview_heartbeat()
            logger.info(
                "Preview timing end: %s (elapsed %.2fs)",
                end_ts,
                elapsed,
            )
            if trace_emit:
                self._emit_preview_trace()

        def finish(result: PlanGenerationResult | None, error: Exception | None = None) -> None:
            def schedule(callable_obj, *args) -> None:
                self._schedule_after(0, callable_obj, *args)

            if error is not None:
                schedule(self._log_action, f"Preview failed: {error}")
                schedule(self._set_status, "Preview failed", 100)
                schedule(messagebox.showerror, "Preview Failed", str(error))
                schedule(finalize_preview, True)
                return

            if result is None or not result.had_tracks:
                schedule(messagebox.showwarning, "No Tracks", "No audio tracks were found in the selected library.")
                schedule(self._set_status, "Idle", 0)
                schedule(finalize_preview)
                return

            self._plan = result.plan
            self.group_disposition_overrides.clear()
            self._apply_deletion_mode()
            self.preview_json_path = result.preview_json_path
            self.preview_html_path = result.preview_html_path
            if hasattr(self.master, "_set_duplicate_finder_plan"):
                self.master._set_duplicate_finder_plan(result.plan, path)

            for message in result.log_messages:
                schedule(self._log_action, message)
            schedule(self._set_status, "Preview generated", self._weighted_progress("preview", 1))
            if self.preview_html_path:
                schedule(self._open_preview)
            schedule(finalize_preview)

        def worker() -> None:
            try:
                result = self._generate_plan(
                    path,
                    write_preview=True,
                    threshold_settings=threshold_settings,
                    fingerprint_settings=fingerprint_settings,
                    show_artwork_variants=show_artwork_variants,
                )
            except Exception as exc:
                logger.exception("Preview generation failed.")
                self._schedule_after(0, finish, None, exc)
                return
            self._schedule_after(0, finish, result, None)

        threading.Thread(target=worker, daemon=True).start()

    def _handle_execute(self) -> None:
        if not self._execute_confirmation_pending:
            self._execute_confirmation_pending = True
            self.execute_label_var.set("Confirm Execute")
            self._log_action("Execute confirmation requested.")
            return
        path = self._validate_library_root()
        if not path:
            return
        self.execution_report_path = None
        self.open_report_btn.config(state="disabled")
        preview_plan_path = self.preview_json_path or os.path.join(path, "Docs", "duplicate_preview.json")
        plan_input = self._plan
        plan_for_checks = self._plan
        current_thresholds = self._current_threshold_settings()
        current_fingerprint_settings = self._current_fingerprint_settings()
        if plan_input and not self._thresholds_match(
            getattr(plan_input, "threshold_settings", None),
            current_thresholds,
        ):
            messagebox.showwarning(
                "Thresholds Changed",
                "Threshold settings changed since the last scan. Regenerate the preview or scan before executing.",
            )
            self._log_action("Execute blocked: threshold settings changed since last scan.")
            return
        if plan_input and not self._fingerprint_settings_match(
            getattr(plan_input, "fingerprint_settings", None),
            current_fingerprint_settings,
        ):
            messagebox.showwarning(
                "Thresholds Changed",
                "Fingerprint settings changed since the last scan. Regenerate the preview or scan before executing.",
            )
            self._log_action("Execute blocked: fingerprint settings changed since last scan.")
            return
        if preview_plan_path and os.path.exists(preview_plan_path):
            if not self.preview_json_path:
                self.preview_json_path = preview_plan_path
            if not self.preview_html_path:
                preview_html_path = os.path.join(path, "Docs", "duplicate_preview.html")
                plan_for_html = plan_for_checks or self._load_preview_plan(preview_plan_path)
                if plan_for_html:
                    export_consolidation_preview_html(
                        plan_for_html,
                        preview_html_path,
                        show_artwork_variants=self.show_artwork_variants_var.get(),
                    )
                    self.preview_html_path = preview_html_path
            if not plan_for_checks:
                plan_for_checks = self._load_preview_plan(preview_plan_path)
                if plan_for_checks:
                    self._plan = plan_for_checks
            if plan_for_checks and not self._thresholds_match(
                getattr(plan_for_checks, "threshold_settings", None),
                current_thresholds,
            ):
                messagebox.showwarning(
                    "Thresholds Changed",
                    "Threshold settings changed since the preview was generated. "
                    "Generate a new preview to execute with the latest thresholds.",
                )
                self._log_action("Execute blocked: preview thresholds do not match current settings.")
                return
            if plan_for_checks and not self._fingerprint_settings_match(
                getattr(plan_for_checks, "fingerprint_settings", None),
                current_fingerprint_settings,
            ):
                messagebox.showwarning(
                    "Thresholds Changed",
                    "Fingerprint settings changed since the preview was generated. "
                    "Generate a new preview to execute with the latest settings.",
                )
                self._log_action("Execute blocked: preview fingerprint settings do not match current settings.")
                return
            if plan_input:
                self._log_action("Using in-memory plan for execution.")
            else:
                plan_input = preview_plan_path
                self._log_action(f"Using preview output for execution: {preview_plan_path}")

        if not plan_input:
            messagebox.showwarning("Preview Required", "Generate a preview before executing.")
            self._log_action("Execute blocked: no plan generated")
            return
        allow_review_required = self._execute_confirmation_pending
        if plan_for_checks and plan_for_checks.review_required_count and not allow_review_required:
            messagebox.showwarning(
                "Review Required",
                "Resolve review-required groups or confirm execution again to bypass the review block.",
            )
            self._log_action("Execute blocked: review required groups pending")
            return
        if self.delete_losers_var.get() and not self.quarantine_var.get():
            messagebox.showwarning(
                "Cleanup Required",
                "Deletion requires cleanup to be enabled. Check Quarantine Duplicates or disable deletion.",
            )
            self._log_action("Execute blocked: deletion enabled while cleanup disabled")
            return

        playlists_dir = self.playlist_path_var.get().strip()
        if not self.update_playlists_var.get():
            playlists_dir = ""
            self._log_action("Playlist updates disabled; playlists will not be rewritten.")
        elif playlists_dir and not os.path.isdir(playlists_dir):
            messagebox.showwarning(
                "Playlist Folder Missing",
                f"The selected playlist folder does not exist:\n{playlists_dir}",
            )
            self._log_action(f"Execute blocked: playlist folder missing ({playlists_dir})")
            return

        docs_dir = os.path.join(path, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        reports_dir = os.path.join(docs_dir, "duplicate_execution_reports")

        def log_callback(msg: str) -> None:
            self._schedule_after(0, self._log_action, msg)

        disposition_counts = self._count_group_dispositions()
        deletions_requested = disposition_counts.get("delete", 0) > 0
        quarantines_requested = disposition_counts.get("quarantine", 0) > 0

        config = ExecutionConfig(
            library_root=path,
            reports_dir=reports_dir,
            playlists_dir=playlists_dir,
            quarantine_dir=os.path.join(path, "Quarantine"),
            quarantine_flatten=True,
            log_callback=log_callback,
            allow_review_required=allow_review_required,
            retain_losers=not self.quarantine_var.get(),
            allow_deletion=deletions_requested,
            confirm_deletion=deletions_requested,
            show_artwork_variants=self.show_artwork_variants_var.get(),
        )

        if not self.quarantine_var.get():
            if quarantines_requested or deletions_requested:
                self._log_action(
                    "Quarantine disabled globally; group overrides will still move or delete selected losers."
                )
            else:
                self._log_action("Quarantine disabled; duplicates will be retained in place.")
        if deletions_requested:
            self._log_action("Deletion enabled; selected losers will be deleted during execution.")

        self._set_status("Executing…", progress=10)
        self._log_action("Execute started")
        self._reset_execute_confirmation()

        def finish(result, error: Exception | None = None) -> None:
            try:
                if error is not None:
                    self._log_action(f"Execution failed: {error}")
                    self._set_status("Execution failed", progress=100)
                    messagebox.showerror("Execution Failed", str(error))
                    self._reset_execute_confirmation()
                    return
                cancelled = any(action.status == "cancelled" for action in result.actions)
                if cancelled:
                    status = "Execution cancelled"
                else:
                    status = "Executed" if result.success else "Execution failed"
                self._set_status(status, progress=100)
                if cancelled:
                    self._log_action("Execution complete: cancelled")
                else:
                    self._log_action(f"Execution complete: {'success' if result.success else 'failed'}")
                report_path = result.report_paths.get("html_report")
                self._log_action(f"Execution report: {report_path}")
                if report_path:
                    if not os.path.exists(ensure_long_path(report_path)):
                        self._log_action("Execution report missing at expected path; enabling Open Report anyway.")
                    self.execution_report_path = report_path
                    self.open_report_btn.config(state="normal")
                else:
                    self.execution_report_path = None
                    self.open_report_btn.config(state="disabled")
                if cancelled:
                    report_line = ""
                    if report_path:
                        if os.path.exists(ensure_long_path(report_path)):
                            report_line = f"\n\nReport (HTML): {report_path}"
                        else:
                            report_line = (
                                "\n\nReport (HTML) was not found at the expected path:\n"
                                f"{report_path}"
                            )
                    messagebox.showinfo(
                        "Execution Cancelled",
                        "Execution was cancelled. Review the report for details."
                        f"{report_line}",
                    )
                elif not result.success:
                    report_line = ""
                    if report_path:
                        if os.path.exists(ensure_long_path(report_path)):
                            report_line = f"\n\nReport (HTML): {report_path}"
                        else:
                            report_line = (
                                "\n\nReport (HTML) was not found at the expected path:\n"
                                f"{report_path}"
                            )
                    messagebox.showwarning(
                        "Execution Failed",
                        "Execution completed with errors. Review the report for details."
                        f"{report_line}",
                    )
                self._reset_execute_confirmation()
            except Exception as exc:
                self._log_action(f"Execution finish handler failed: {exc}")
                self._set_status("Execution failed", progress=100)
                messagebox.showerror(
                    "Execution Failed",
                    "An unexpected error occurred while finalizing execution. "
                    "Check the log for details.",
                )
                self._reset_execute_confirmation()

        def worker() -> None:
            try:
                result = execute_consolidation_plan(plan_input, config)
            except Exception as exc:
                self._schedule_after(0, finish, None, exc)
                return
            self._schedule_after(0, finish, result, None)

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_update_playlists(self) -> None:
        state = "enabled" if self.update_playlists_var.get() else "disabled"
        self._log_action(f"Update Playlists {state}")
        self._reset_execute_confirmation()

    def _toggle_show_artwork_variants(self) -> None:
        enabled = self.show_artwork_variants_var.get()
        cfg = load_config()
        cfg["duplicate_finder_show_artwork_variants"] = bool(enabled)
        save_config(cfg)
        state = "enabled" if enabled else "disabled"
        self._log_action(f"Show different artwork variants {state}")
        self._reset_execute_confirmation()

    def _toggle_quarantine(self) -> None:
        state = "enabled" if self.quarantine_var.get() else "disabled"
        self._log_action(f"Quarantine Duplicates {state}")
        if not self.quarantine_var.get() and self.delete_losers_var.get():
            messagebox.showwarning(
                "Cleanup Required",
                "Deletion requires cleanup to be enabled. Re-enabling Quarantine Duplicates.",
            )
            self.quarantine_var.set(True)
            self._log_action("Quarantine Duplicates enabled to support deletions.")
        self._reset_preview_if_needed()
        self._reset_execute_confirmation()

    def _toggle_delete_losers(self) -> None:
        if self.delete_losers_var.get():
            if not self.quarantine_var.get():
                self.quarantine_var.set(True)
                self._log_action("Quarantine Duplicates enabled to support deletions.")
            self._log_action("Delete losers enabled; preview will mark losers for deletion.")
        else:
            self._log_action("Delete losers disabled; preview will quarantine losers.")
        self._apply_deletion_mode()
        self._reset_preview_if_needed()
        self._reset_execute_confirmation()

    def _apply_deletion_mode(self) -> None:
        if not self._plan:
            return
        default_disposition = self._default_group_disposition()
        for group in self._plan.groups:
            if group.group_id in self.group_disposition_overrides:
                continue
            for loser in group.losers:
                current = group.loser_disposition.get(loser, "quarantine")
                if current in ("retain", "quarantine", "delete"):
                    group.loser_disposition[loser] = default_disposition
        self._plan.refresh_plan_signature()
        self._update_groups_view(self._plan)

    def _reset_preview_if_needed(self) -> None:
        if self.preview_json_path:
            self.preview_json_path = None
            self.preview_html_path = None
            self._log_action("Preview cleared; generate a new preview to reflect delete settings.")

    def _open_preview(self) -> None:
        self._record_preview_trace("preview-open")
        self._open_local_html(
            self.preview_html_path,
            title="Preview Output",
            missing_message="Preview output could not be found. Generate a new preview.",
            empty_message="No preview output is available yet.",
        )

    def _open_execution_report(self) -> None:
        self._open_local_html(
            self.execution_report_path,
            title="Execution Report",
            missing_message=(
                "Execution report could not be found. Run the execution again to generate a new report."
            ),
            empty_message="No execution report is available yet.",
        )

    def _open_local_html(
        self,
        path: str | None,
        *,
        title: str,
        missing_message: str,
        empty_message: str,
    ) -> None:
        if not path:
            messagebox.showinfo(title, empty_message)
            return
        safe_path = ensure_long_path(path)
        if not os.path.exists(safe_path):
            messagebox.showerror(f"{title} Missing", missing_message)
            return
        display_path = strip_ext_prefix(path)
        try:
            uri = Path(display_path).resolve().as_uri()
        except Exception:
            uri = display_path
        webbrowser.open(uri)

class SimilarityInspectorDialog(tk.Toplevel):
    """Inspect fingerprint similarity for two selected audio files."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("Similarity Inspector")
        self.transient(parent)
        self.resizable(True, True)
        self.parent = parent

        cfg = load_config()
        self.song_a_var = tk.StringVar()
        self.song_b_var = tk.StringVar()
        self.trim_silence_var = tk.BooleanVar(value=bool(cfg.get("trim_silence", False)))
        self.offset_var = tk.StringVar(value=str(cfg.get("fingerprint_offset_ms", FP_OFFSET_MS)))
        self.duration_var = tk.StringVar(value=str(cfg.get("fingerprint_duration_ms", FP_DURATION_MS)))
        self.silence_db_var = tk.StringVar(value=str(cfg.get("fingerprint_silence_threshold_db", FP_SILENCE_THRESHOLD_DB)))
        self.silence_len_var = tk.StringVar(value=str(cfg.get("fingerprint_silence_min_len_ms", FP_SILENCE_MIN_LEN_MS)))
        self.exact_var = tk.StringVar(value=str(cfg.get("exact_duplicate_threshold", EXACT_DUPLICATE_THRESHOLD)))
        self.near_var = tk.StringVar(value=str(cfg.get("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD)))
        self.mixed_var = tk.StringVar(value=str(cfg.get("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST)))

        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Compare two tracks using the Duplicate Finder fingerprint pipeline.",
            foreground="#555",
            wraplength=520,
        ).pack(anchor="w")

        file_frame = ttk.LabelFrame(container, text="Tracks")
        file_frame.pack(fill="x", pady=(10, 8))

        def _file_row(label: str, var: tk.StringVar, row: int, command: Callable[[], None]) -> None:
            ttk.Label(file_frame, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=6)
            ttk.Entry(file_frame, textvariable=var, width=60).grid(row=row, column=1, sticky="ew", padx=6, pady=6)
            ttk.Button(file_frame, text="Browse…", command=command).grid(row=row, column=2, sticky="e", padx=6, pady=6)

        _file_row("Song A", self.song_a_var, 0, lambda: self._choose_file(self.song_a_var))
        _file_row("Song B", self.song_b_var, 1, lambda: self._choose_file(self.song_b_var))
        file_frame.columnconfigure(1, weight=1)

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="Run", command=self._run_inspection).pack(side="left")
        ttk.Button(
            controls,
            text="Duplicate Finder Report",
            command=self._run_duplicate_finder_report,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Close", command=self.destroy).pack(side="left", padx=(6, 0))

        self.advanced_visible = False
        self.advanced_btn = ttk.Button(
            container, text="Advanced ▸", command=self._toggle_advanced
        )
        self.advanced_btn.pack(anchor="w", pady=(4, 0))

        self.advanced_frame = ttk.LabelFrame(container, text="Advanced Overrides")
        self._build_advanced_controls()

        results_frame = ttk.LabelFrame(container, text="Results")
        results_frame.pack(fill="both", expand=True, pady=(10, 0))
        self.results_text = ScrolledText(results_frame, height=12, wrap="word", state="disabled")
        self.results_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_advanced_controls(self) -> None:
        frame = self.advanced_frame
        settings_frame = ttk.Frame(frame)
        settings_frame.pack(fill="x", padx=8, pady=8)

        ttk.Checkbutton(
            settings_frame,
            text="Trim leading/trailing silence before fingerprinting",
            variable=self.trim_silence_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        def _row(label: str, var: tk.StringVar, row: int) -> None:
            ttk.Label(settings_frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(settings_frame, textvariable=var, width=16).grid(row=row, column=1, sticky="w", pady=2)

        row_idx = 1
        _row("Fingerprint offset (ms)", self.offset_var, row_idx)
        row_idx += 1
        _row("Fingerprint duration (ms)", self.duration_var, row_idx)
        row_idx += 1
        _row("Silence threshold (dB)", self.silence_db_var, row_idx)
        row_idx += 1
        _row("Silence min length (ms)", self.silence_len_var, row_idx)
        row_idx += 1
        _row("Exact duplicate threshold", self.exact_var, row_idx)
        row_idx += 1
        _row("Near duplicate threshold", self.near_var, row_idx)
        row_idx += 1
        _row("Mixed-codec boost", self.mixed_var, row_idx)

    def _toggle_advanced(self) -> None:
        if self.advanced_visible:
            self.advanced_frame.pack_forget()
            self.advanced_btn.configure(text="Advanced ▸")
            self.advanced_visible = False
        else:
            self.advanced_frame.pack(fill="x", pady=(6, 0))
            self.advanced_btn.configure(text="Advanced ▾")
            self.advanced_visible = True

    def _choose_file(self, var: tk.StringVar) -> None:
        initial_dir = load_last_path() or os.getcwd()
        path = filedialog.askopenfilename(
            title="Select audio file",
            initialdir=initial_dir,
        )
        if path:
            var.set(path)
            save_last_path(os.path.dirname(path))

    def _parse_float(self, raw: str, label: str, *, min_value: float | None = None) -> float | None:
        try:
            value = float(raw)
        except ValueError:
            messagebox.showwarning("Invalid Value", f"{label} must be a number.")
            return None
        if min_value is not None and value < min_value:
            messagebox.showwarning("Invalid Value", f"{label} must be at least {min_value}.")
            return None
        return value

    def _parse_int(self, raw: str, label: str, *, min_value: int | None = None) -> int | None:
        try:
            value = int(float(raw))
        except ValueError:
            messagebox.showwarning("Invalid Value", f"{label} must be a whole number.")
            return None
        if min_value is not None and value < min_value:
            messagebox.showwarning("Invalid Value", f"{label} must be at least {min_value}.")
            return None
        return value

    def _collect_settings(self) -> dict[str, float | int | bool] | None:
        exact = self._parse_float(self.exact_var.get().strip(), "Exact duplicate threshold", min_value=0.0)
        if exact is None:
            return None
        near = self._parse_float(self.near_var.get().strip(), "Near duplicate threshold", min_value=0.0)
        if near is None:
            return None
        if near < exact:
            messagebox.showwarning(
                "Invalid Value",
                "Near duplicate threshold must be greater than or equal to the exact threshold.",
            )
            return None
        mixed = self._parse_float(self.mixed_var.get().strip(), "Mixed-codec boost", min_value=0.0)
        if mixed is None:
            return None
        offset = self._parse_int(self.offset_var.get().strip(), "Fingerprint offset (ms)", min_value=0)
        if offset is None:
            return None
        duration = self._parse_int(self.duration_var.get().strip(), "Fingerprint duration (ms)", min_value=0)
        if duration is None:
            return None
        silence_db = self._parse_float(self.silence_db_var.get().strip(), "Silence threshold (dB)")
        if silence_db is None:
            return None
        silence_len = self._parse_int(self.silence_len_var.get().strip(), "Silence min length (ms)", min_value=0)
        if silence_len is None:
            return None
        return {
            "exact_duplicate_threshold": exact,
            "near_duplicate_threshold": near,
            "mixed_codec_threshold_boost": mixed,
            "fingerprint_offset_ms": offset,
            "fingerprint_duration_ms": duration,
            "fingerprint_silence_threshold_db": silence_db,
            "fingerprint_silence_min_len_ms": silence_len,
            "trim_silence": bool(self.trim_silence_var.get()),
        }

    def _threshold_settings_snapshot(self, settings: dict[str, float | int | bool]) -> dict[str, float]:
        return {
            "exact_duplicate_threshold": float(settings["exact_duplicate_threshold"]),
            "near_duplicate_threshold": float(settings["near_duplicate_threshold"]),
            "mixed_codec_threshold_boost": float(settings["mixed_codec_threshold_boost"]),
            "preview_artwork_max_dim": float(settings.get("preview_artwork_max_dim", PREVIEW_ARTWORK_MAX_DIM)),
            "preview_artwork_quality": float(settings.get("preview_artwork_quality", PREVIEW_ARTWORK_QUALITY)),
        }

    def _fingerprint_settings_snapshot(self, settings: dict[str, float | int | bool]) -> dict[str, object]:
        return {
            "trim_silence": bool(settings["trim_silence"]),
            "fingerprint_offset_ms": int(settings["fingerprint_offset_ms"]),
            "fingerprint_duration_ms": int(settings["fingerprint_duration_ms"]),
            "fingerprint_silence_threshold_db": float(settings["fingerprint_silence_threshold_db"]),
            "fingerprint_silence_min_len_ms": int(settings["fingerprint_silence_min_len_ms"]),
        }

    def _is_lossless(self, path: str) -> bool:
        ext = os.path.splitext(path)[1].lower()
        return ext in LOSSLESS_EXTS

    def _fingerprint_for_path(self, path: str, settings: dict[str, float | int | bool]) -> tuple[str | None, str | None]:
        try:
            start_sec = float(settings["fingerprint_offset_ms"]) / 1000.0
            duration_ms = float(settings["fingerprint_duration_ms"])
            duration_sec = duration_ms / 1000.0 if duration_ms > 0 else 120.0
            silence_db = float(settings["fingerprint_silence_threshold_db"])
            silence_len = float(settings["fingerprint_silence_min_len_ms"]) / 1000.0
            fp = chromaprint_utils.fingerprint_fpcalc(
                ensure_long_path(path),
                trim=bool(settings["trim_silence"]),
                start_sec=start_sec,
                duration_sec=duration_sec,
                threshold_db=silence_db,
                min_silence_duration=silence_len,
            )
            return fp, None
        except chromaprint_utils.FingerprintError as exc:
            return None, str(exc)
        except Exception as exc:  # pragma: no cover - guard against unexpected failures
            return None, str(exc)

    def _format_duration(self, seconds: float | None) -> str:
        if not seconds:
            return "n/a"
        total = int(round(seconds))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _audio_summary(self, path: str) -> dict[str, object]:
        summary: dict[str, object] = {
            "path": path,
            "codec": os.path.splitext(path)[1].lstrip(".").upper() or "n/a",
            "sample_rate": None,
            "channels": None,
            "duration": None,
        }
        try:
            audio = MutagenFile(ensure_long_path(path))
        except Exception:
            audio = None
        if audio and hasattr(audio, "info") and audio.info:
            info = audio.info
            codec = getattr(info, "codec", None)
            if not codec and getattr(audio, "mime", None):
                mime = audio.mime
                if isinstance(mime, (list, tuple)) and mime:
                    codec = mime[0]
                elif isinstance(mime, str):
                    codec = mime
            summary["codec"] = codec or summary["codec"]
            summary["sample_rate"] = getattr(info, "sample_rate", None) or getattr(info, "samplerate", None)
            summary["channels"] = getattr(info, "channels", None)
            summary["duration"] = getattr(info, "length", None)
        return summary

    def _format_summary_line(self, summary: dict[str, object]) -> str:
        sample_rate = summary.get("sample_rate") or "n/a"
        channels = summary.get("channels") or "n/a"
        duration = self._format_duration(summary.get("duration") if isinstance(summary.get("duration"), (int, float)) else None)
        codec = summary.get("codec") or "n/a"
        return f"Codec: {codec} | Sample rate: {sample_rate} Hz | Channels: {channels} | Duration: {duration}"

    def _export_similarity_inspector_html(
        self,
        *,
        report_path: str,
        summary_a: dict[str, object],
        summary_b: dict[str, object],
        settings: dict[str, float | int | bool],
        exact_threshold: float,
        near_threshold: float,
        mixed_boost: float,
        mixed_codec: bool,
        effective_near: float,
        distance: float,
        verdict: str,
        off_by: float | None,
        err_a: str | None,
        err_b: str | None,
    ) -> None:
        def esc(value: object) -> str:
            return html.escape(str(value))

        html_lines = [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8' />",
            "<title>Similarity Inspector Report</title>",
            "<style>",
            "body{font-family:Arial, sans-serif; margin:24px; color:#222;}",
            "h1{font-size:20px; margin-bottom:6px;}",
            "h2{font-size:16px; margin-top:24px;}",
            "table{border-collapse:collapse; width:100%; margin-top:8px;}",
            "th,td{border:1px solid #ddd; padding:8px; text-align:left; vertical-align:top;}",
            "th{background:#f4f4f4; width:200px;}",
            ".path{font-family:monospace; word-break:break-all;}",
            ".meta{color:#555; font-size:12px; margin-top:4px;}",
            ".muted{color:#666; font-size:12px;}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Similarity Inspector Report</h1>",
            f"<div class='muted'>Generated: {esc(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</div>",
            "<h2>Verdict</h2>",
            "<table>",
            f"<tr><th>Verdict</th><td>{esc(verdict)}</td></tr>",
            f"<tr><th>Raw fingerprint distance</th><td>{esc(f'{distance:.4f}')}</td></tr>",
            f"<tr><th>Effective near threshold</th><td>{esc(f'{effective_near:.4f}')}</td></tr>",
            "</table>",
            "<h2>Tracks</h2>",
            "<table>",
            "<tr><th>Song A</th><td>",
            f"<div class='path'>{esc(summary_a.get('path'))}</div>",
            f"<div class='meta'>{esc(self._format_summary_line(summary_a))}</div>",
            "</td></tr>",
            "<tr><th>Song B</th><td>",
            f"<div class='path'>{esc(summary_b.get('path'))}</div>",
            f"<div class='meta'>{esc(self._format_summary_line(summary_b))}</div>",
            "</td></tr>",
            "</table>",
            "<h2>Fingerprint Settings</h2>",
            "<table>",
            f"<tr><th>Trim silence</th><td>{esc(settings['trim_silence'])}</td></tr>",
            f"<tr><th>Fingerprint offset (ms)</th><td>{esc(settings['fingerprint_offset_ms'])}</td></tr>",
            f"<tr><th>Fingerprint duration (ms)</th><td>{esc(settings['fingerprint_duration_ms'])}</td></tr>",
            f"<tr><th>Silence threshold (dB)</th><td>{esc(settings['fingerprint_silence_threshold_db'])}</td></tr>",
            f"<tr><th>Silence min length (ms)</th><td>{esc(settings['fingerprint_silence_min_len_ms'])}</td></tr>",
            "</table>",
            "<h2>Thresholds</h2>",
            "<table>",
            f"<tr><th>Exact threshold</th><td>{esc(f'{exact_threshold:.4f}')}</td></tr>",
            f"<tr><th>Near threshold</th><td>{esc(f'{near_threshold:.4f}')}</td></tr>",
            f"<tr><th>Mixed-codec boost</th><td>{esc(f'{mixed_boost:.4f}')}</td></tr>",
            f"<tr><th>Mixed-codec applied</th><td>{esc('Yes' if mixed_codec else 'No')}</td></tr>",
            "</table>",
        ]
        if off_by is not None:
            html_lines.extend(
                [
                    "<h2>Distance Gap</h2>",
                    "<table>",
                    f"<tr><th>How far off</th><td>{esc(f'{off_by:.4f}')}</td></tr>",
                    "</table>",
                ]
            )
        if err_a or err_b:
            html_lines.extend(["<h2>Fingerprint Errors</h2>", "<table>"])
            if err_a:
                html_lines.append(f"<tr><th>Song A</th><td>{esc(err_a)}</td></tr>")
            if err_b:
                html_lines.append(f"<tr><th>Song B</th><td>{esc(err_b)}</td></tr>")
            html_lines.append("</table>")
        html_lines.extend(["</body>", "</html>"])

        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(html_lines))

    def _set_results(self, text: str) -> None:
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", text)
        self.results_text.configure(state="disabled")
        self.results_text.see("end")

    def _run_inspection(self) -> None:
        path_a = self.song_a_var.get().strip()
        path_b = self.song_b_var.get().strip()
        if not path_a or not path_b:
            messagebox.showwarning("Missing Files", "Please select both Song A and Song B.")
            return
        for path in (path_a, path_b):
            if not os.path.isfile(path):
                messagebox.showwarning("Invalid File", f"File not found:\n{path}")
                return
            if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTS:
                messagebox.showwarning("Unsupported File", f"Unsupported audio file:\n{path}")
                return

        library_root = None
        if hasattr(self.parent, "require_library"):
            library_root = self.parent.require_library()
        if not library_root:
            return

        settings = self._collect_settings()
        if settings is None:
            return

        fp_a, err_a = self._fingerprint_for_path(path_a, settings)
        fp_b, err_b = self._fingerprint_for_path(path_b, settings)
        distance = fingerprint_distance(fp_a, fp_b)

        exact_threshold = float(settings["exact_duplicate_threshold"])
        near_threshold = max(float(settings["near_duplicate_threshold"]), exact_threshold)
        mixed_boost = float(settings["mixed_codec_threshold_boost"])
        mixed_codec = self._is_lossless(path_a) != self._is_lossless(path_b)
        effective_near = near_threshold + mixed_boost if mixed_codec else near_threshold

        if fp_a and fp_b:
            if distance <= exact_threshold:
                verdict = "Exact duplicate"
            elif distance <= effective_near:
                verdict = "Near duplicate"
            else:
                verdict = "Not a match"
        else:
            verdict = "Not a match (fingerprint unavailable)"

        off_by = None
        if distance > effective_near:
            off_by = distance - effective_near

        summary_a = self._audio_summary(path_a)
        summary_b = self._audio_summary(path_b)

        docs_dir = os.path.join(library_root, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(docs_dir, f"similarity_inspector_report_{timestamp}.html")
        try:
            self._export_similarity_inspector_html(
                report_path=report_path,
                summary_a=summary_a,
                summary_b=summary_b,
                settings=settings,
                exact_threshold=exact_threshold,
                near_threshold=near_threshold,
                mixed_boost=mixed_boost,
                mixed_codec=mixed_codec,
                effective_near=effective_near,
                distance=distance,
                verdict=verdict,
                off_by=off_by,
                err_a=err_a,
                err_b=err_b,
            )
        except OSError as exc:
            messagebox.showwarning("Report Save Failed", f"Could not save report:\n{exc}")

        report_lines = [
            "Similarity Inspector Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Verdict: {verdict}",
            f"Raw fingerprint distance: {distance:.4f}",
        ]
        if off_by is not None:
            report_lines.append(f"How far off: {off_by:.4f} above the effective near threshold")
        if err_a or err_b:
            report_lines.append("")
            report_lines.append("Fingerprint Errors:")
            if err_a:
                report_lines.append(f"  Song A: {err_a}")
            if err_b:
                report_lines.append(f"  Song B: {err_b}")
        report_lines.append("")
        report_lines.append(f"Saved HTML report: {report_path}")
        self._set_results("\n".join(report_lines))

    def _run_duplicate_finder_report(self) -> None:
        path_a = self.song_a_var.get().strip()
        path_b = self.song_b_var.get().strip()
        if not path_a or not path_b:
            messagebox.showwarning("Missing Files", "Please select both Song A and Song B.")
            return
        for path in (path_a, path_b):
            if not os.path.isfile(path):
                messagebox.showwarning("Invalid File", f"File not found:\n{path}")
                return
            if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTS:
                messagebox.showwarning("Unsupported File", f"Unsupported audio file:\n{path}")
                return

        library_root = None
        if hasattr(self.parent, "require_library"):
            library_root = self.parent.require_library()
        if not library_root:
            return

        settings = self._collect_settings()
        if settings is None:
            return

        fp_a, err_a = self._fingerprint_for_path(path_a, settings)
        fp_b, err_b = self._fingerprint_for_path(path_b, settings)

        report = build_duplicate_pair_report(
            {"path": path_a, "fingerprint": fp_a, "ext": os.path.splitext(path_a)[1].lower()},
            {"path": path_b, "fingerprint": fp_b, "ext": os.path.splitext(path_b)[1].lower()},
            exact_duplicate_threshold=float(settings["exact_duplicate_threshold"]),
            near_duplicate_threshold=float(settings["near_duplicate_threshold"]),
            mixed_codec_threshold_boost=float(settings["mixed_codec_threshold_boost"]),
            fingerprint_settings=self._fingerprint_settings_snapshot(settings),
            threshold_settings=self._threshold_settings_snapshot(settings),
        )

        docs_dir = os.path.join(library_root, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(docs_dir, f"duplicate_pair_report_{timestamp}.html")
        try:
            export_duplicate_pair_report_html(report, report_path)
        except OSError as exc:
            messagebox.showwarning("Report Save Failed", f"Could not save report:\n{exc}")
            report_path = None

        report_lines = [
            "Duplicate Finder Pair Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Verdict: {report.verdict}",
            f"Match type: {report.match_type}",
        ]
        if report.fingerprint_distance is not None:
            report_lines.append(f"Fingerprint distance: {report.fingerprint_distance:.4f}")
        report_lines.append("")
        report_lines.append("Gate Checks:")
        for step in report.steps:
            report_lines.append(f"  - {step.name}: {step.status} ({step.detail})")
        if err_a or err_b:
            report_lines.append("")
            report_lines.append("Fingerprint Errors:")
            if err_a:
                report_lines.append(f"  Song A: {err_a}")
            if err_b:
                report_lines.append(f"  Song B: {err_b}")
        if report_path:
            report_lines.append("")
            report_lines.append(f"Saved HTML report: {report_path}")

        self._set_results("\n".join(report_lines))


class M4ATesterDialog(tk.Toplevel):
    """Standalone UI for validating M4A metadata/album art parsing."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("M4A Tester")
        self.transient(parent)
        self.resizable(True, True)
        self.parent = parent

        self.file_path_var = tk.StringVar()
        self._cover_photo: ImageTk.PhotoImage | None = None

        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Select an M4A file to validate album art and metadata parsing.",
            foreground="#555",
            wraplength=540,
        ).pack(anchor="w")

        file_frame = ttk.LabelFrame(container, text="M4A File")
        file_frame.pack(fill="x", pady=(10, 8))
        ttk.Label(file_frame, text="File").grid(
            row=0, column=0, sticky="w", padx=6, pady=6
        )
        entry = ttk.Entry(
            file_frame, textvariable=self.file_path_var, width=60, state="readonly"
        )
        entry.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(file_frame, text="Browse…", command=self._choose_file).grid(
            row=0, column=2, sticky="e", padx=6, pady=6
        )
        file_frame.columnconfigure(1, weight=1)

        content = ttk.Frame(container)
        content.pack(fill="both", expand=True)

        art_frame = ttk.LabelFrame(content, text="Album Art")
        art_frame.pack(side="left", fill="both", expand=False, padx=(0, 8))
        self.cover_label = ttk.Label(
            art_frame,
            text="No album art loaded",
            anchor="center",
            width=28,
            padding=8,
        )
        self.cover_label.pack(fill="both", expand=True)

        metadata_frame = ttk.LabelFrame(content, text="Metadata")
        metadata_frame.pack(side="left", fill="both", expand=True)
        self.metadata_text = ScrolledText(
            metadata_frame, height=16, wrap="word", state="disabled"
        )
        self.metadata_text.pack(fill="both", expand=True, padx=6, pady=6)

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Button(controls, text="Close", command=self.destroy).pack(side="left")

    def _choose_file(self) -> None:
        initial_dir = load_last_path() or os.getcwd()
        chosen = filedialog.askopenfilename(
            parent=self,
            title="Select M4A File",
            initialdir=initial_dir,
            filetypes=[("M4A files", "*.m4a")],
        )
        if not chosen:
            return
        if not chosen.lower().endswith(".m4a"):
            messagebox.showerror("M4A Tester", "Please select a .m4a file.")
            return
        self.file_path_var.set(chosen)
        save_last_path(os.path.dirname(chosen))
        self._load_metadata(chosen)

    def _load_metadata(self, path: str) -> None:
        self._set_metadata_text("")
        self._set_cover_image(None)

        tags, _cover_payloads, error, _reader = read_metadata(path, include_cover=False)

        def _format_value(value: object) -> str | None:
            if value in (None, "", []):
                return None
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value if item not in (None, ""))
            return str(value)

        lines = []
        track_value = _format_value(tags.get("tracknumber") or tags.get("track"))
        disc_value = _format_value(tags.get("discnumber") or tags.get("disc"))
        display_fields = [
            ("Title", "title"),
            ("Artist", "artist"),
            ("Album", "album"),
            ("Album Artist", "albumartist"),
            ("Track", track_value),
            ("Disc", disc_value),
            ("Year", "year"),
            ("Date", "date"),
            ("Genre", "genre"),
            ("Compilation", "compilation"),
        ]
        for label, key in display_fields:
            value = _format_value(tags.get(key)) if isinstance(key, str) else key
            if value:
                lines.append(f"{label}: {value}")
        if error and not lines:
            lines.append(f"Metadata error: {error}")
        self._set_metadata_text("\n".join(lines) if lines else "No metadata found.")

        cover_bytes = self._extract_cover_art_bytes(path)
        if cover_bytes:
            try:
                image = Image.open(BytesIO(cover_bytes))
                image.thumbnail((240, 240))
                self._set_cover_image(image)
            except Exception:
                self.cover_label.config(text="Failed to load album art")
        else:
            self.cover_label.config(text="No album art found")

    def _extract_cover_art_bytes(self, path: str) -> bytes | None:
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                ensure_long_path(path),
                "-map",
                "0:v",
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                result = None
            if result and result.returncode == 0 and result.stdout:
                return result.stdout

        return read_sidecar_artwork_bytes(path)

    def _set_metadata_text(self, content: str) -> None:
        self.metadata_text.configure(state="normal")
        self.metadata_text.delete("1.0", tk.END)
        self.metadata_text.insert("1.0", content)
        self.metadata_text.configure(state="disabled")

    def _set_cover_image(self, image: Image.Image | None) -> None:
        if image is None:
            self._cover_photo = None
            self.cover_label.configure(image="", text="No album art loaded")
            return
        self._cover_photo = ImageTk.PhotoImage(image)
        self.cover_label.configure(image=self._cover_photo, text="")


class LibraryCompressionPanel(ttk.Frame):
    """UI panel for converting a library mirror into Opus files."""

    def __init__(self, parent: tk.Widget, controller: tk.Widget):
        super().__init__(parent, padding=12)
        self.controller = controller

        default_path = ""
        if hasattr(controller, "library_path_var"):
            default_path = controller.library_path_var.get()

        self.source_var = tk.StringVar(value=default_path)
        self.destination_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready to mirror library.")
        self.overwrite_var = tk.BooleanVar(value=False)
        self._worker: threading.Thread | None = None
        self.report_path: str | None = None

        ttk.Label(
            self,
            text="Library Compression",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            self,
            text="Create a mirror of your library with FLAC files converted to Opus (96 kbps).",
            foreground="#555",
            wraplength=520,
        ).pack(anchor="w", pady=(0, 8))

        source_frame = ttk.LabelFrame(self, text="Source Library")
        source_frame.pack(fill="x", pady=(10, 6))
        ttk.Entry(source_frame, textvariable=self.source_var, width=60).grid(
            row=0, column=0, sticky="ew", padx=6, pady=6
        )
        ttk.Button(source_frame, text="Browse…", command=self._browse_source).grid(
            row=0, column=1, sticky="e", padx=6, pady=6
        )
        source_frame.columnconfigure(0, weight=1)

        dest_frame = ttk.LabelFrame(self, text="Destination Mirror")
        dest_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(dest_frame, textvariable=self.destination_var, width=60).grid(
            row=0, column=0, sticky="ew", padx=6, pady=6
        )
        ttk.Button(dest_frame, text="Browse…", command=self._browse_destination).grid(
            row=0, column=1, sticky="e", padx=6, pady=6
        )
        dest_frame.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            self,
            text="Overwrite existing files in the destination",
            variable=self.overwrite_var,
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(self, textvariable=self.status_var).pack(anchor="w", pady=(4, 8))
        self.progress = ttk.Progressbar(
            self,
            mode="determinate",
        )
        self.progress.pack(fill="x", pady=(0, 8))

        controls = ttk.Frame(self)
        controls.pack(anchor="e")
        self.run_btn = ttk.Button(controls, text="Start", command=self._start)
        self.run_btn.pack(side="right", padx=(6, 0))
        self.open_report_btn = ttk.Button(
            controls,
            text="Open Report",
            command=self._open_report,
            state="disabled",
        )
        self.open_report_btn.pack(side="right", padx=(0, 6))

    def _browse_source(self) -> None:
        initial = self.source_var.get() or load_last_path() or os.getcwd()
        folder = filedialog.askdirectory(
            parent=self, title="Select Source Library", initialdir=initial
        )
        if folder:
            self.source_var.set(folder)

    def _browse_destination(self) -> None:
        initial = self.destination_var.get() or load_last_path() or os.getcwd()
        folder = filedialog.askdirectory(
            parent=self, title="Select Destination Folder", initialdir=initial
        )
        if folder:
            self.destination_var.set(folder)

    def _start(self) -> None:
        source = self.source_var.get().strip()
        destination = self.destination_var.get().strip()

        if not source or not os.path.isdir(source):
            messagebox.showwarning(
                "Source Required", "Select a valid source library folder."
            )
            return
        if not destination or not os.path.isdir(destination):
            messagebox.showwarning(
                "Destination Required", "Select a valid destination folder."
            )
            return
        if os.path.abspath(source) == os.path.abspath(destination):
            messagebox.showerror(
                "Invalid Destination",
                "The destination folder must be different from the source.",
            )
            return
        try:
            if os.path.commonpath([source, destination]) == os.path.abspath(source):
                messagebox.showerror(
                    "Invalid Destination",
                    "The destination cannot be inside the source library.",
                )
                return
        except ValueError:
            pass
        if not shutil.which("ffmpeg"):
            messagebox.showerror(
                "FFmpeg Required",
                "FFmpeg was not found on your PATH. Install it to convert FLAC files.",
            )
            return

        self._set_running(True, "Scanning library…")
        overwrite = self.overwrite_var.get()
        self.report_path = None
        self.open_report_btn.configure(state="disabled")

        def worker() -> None:
            try:
                summary = self._mirror_library(source, destination, overwrite)
                self.after(0, lambda: self._finished(summary))
            except Exception as exc:  # pragma: no cover - safety net
                self.after(0, lambda: self._failed(str(exc)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _set_running(self, running: bool, status: str) -> None:
        self.status_var.set(status)
        state = "disabled" if running else "normal"
        self.run_btn.configure(state=state)
        if running:
            self.progress.configure(maximum=1, value=0)

    def _log(self, message: str) -> None:
        if hasattr(self.controller, "_log"):
            self.controller._log(message)
            if hasattr(self.controller, "show_log_tab"):
                self.controller.show_log_tab()
        else:
            print(message)

    def _mirror_library(
        self, source: str, destination: str, overwrite: bool
    ) -> dict[str, object]:
        def progress_callback(
            total_tasks: int,
            completed: int,
            converted: int,
            copied: int,
            skipped: int,
            errors: int,
        ) -> None:
            self.after(
                0,
                lambda t=total_tasks, d=completed: (
                    self.status_var.set("Mirroring library…"),
                    self.progress.configure(maximum=max(t, 1)),
                    self.progress.configure(value=d),
                ),
            )

        return mirror_library(
            source,
            destination,
            overwrite,
            progress_callback=progress_callback,
            log_callback=self._log,
        )

    def _finished(self, summary: dict[str, object]) -> None:
        self._set_running(False, "Completed.")
        message = (
            "Mirror complete.\n\n"
            f"Files processed: {summary['total']}\n"
            f"FLAC converted: {summary['converted']}\n"
            f"Other files copied: {summary['copied']}\n"
            f"Skipped: {summary['skipped']}\n"
            f"Errors: {summary['errors']}"
        )
        self.status_var.set(
            f"Done. Converted {summary['converted']} and copied {summary['copied']}."
        )
        messagebox.showinfo("Library Compression", message)
        self._write_report(summary)
        self._log(
            "Library Compression: "
            f"total={summary['total']} converted={summary['converted']} "
            f"copied={summary['copied']} skipped={summary['skipped']} "
            f"errors={summary['errors']}"
        )

    def _failed(self, error: str) -> None:
        self._set_running(False, "Failed.")
        messagebox.showerror("Library Compression", f"Run failed:\n{error}")

    def _write_report(self, summary: dict[str, object]) -> None:
        destination = self.destination_var.get().strip()
        if not destination:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(
            destination,
            "Docs",
            "opus_library_mirror_reports",
        )
        report_path = os.path.join(
            reports_dir,
            f"opus_library_mirror_report_{timestamp}.html",
        )
        try:
            write_mirror_report(report_path, summary)
        except OSError as exc:
            self._log(f"Library Compression: failed to write report: {exc}")
            self.report_path = None
            self.open_report_btn.configure(state="disabled")
            return
        self.report_path = report_path
        self.open_report_btn.configure(state="normal")
        self._log(f"Library Compression: report saved to {report_path}")

    def _open_report(self) -> None:
        if not self.report_path:
            messagebox.showinfo("Library Compression", "No report is available yet.")
            return
        safe_path = ensure_long_path(self.report_path)
        if not os.path.exists(safe_path):
            messagebox.showerror(
                "Report Missing",
                "The report could not be found. Run the mirror again to generate a new report.",
            )
            return
        display_path = strip_ext_prefix(self.report_path)
        try:
            uri = Path(display_path).resolve().as_uri()
        except Exception:
            uri = display_path
        webbrowser.open(uri)


class OpusTesterDialog(tk.Toplevel):
    """Standalone UI for validating Opus metadata/album art parsing."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("Opus Tester")
        self.transient(parent)
        self.resizable(True, True)
        self.parent = parent

        self.file_path_var = tk.StringVar()
        self._cover_photo: ImageTk.PhotoImage | None = None

        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Select an Opus file to validate album art and metadata parsing.",
            foreground="#555",
            wraplength=540,
        ).pack(anchor="w")

        file_frame = ttk.LabelFrame(container, text="Opus File")
        file_frame.pack(fill="x", pady=(10, 8))
        ttk.Label(file_frame, text="File").grid(
            row=0, column=0, sticky="w", padx=6, pady=6
        )
        entry = ttk.Entry(
            file_frame, textvariable=self.file_path_var, width=60, state="readonly"
        )
        entry.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(file_frame, text="Browse…", command=self._choose_file).grid(
            row=0, column=2, sticky="e", padx=6, pady=6
        )
        file_frame.columnconfigure(1, weight=1)

        content = ttk.Frame(container)
        content.pack(fill="both", expand=True)

        art_frame = ttk.LabelFrame(content, text="Album Art")
        art_frame.pack(side="left", fill="both", expand=False, padx=(0, 8))
        self.cover_label = ttk.Label(
            art_frame,
            text="No album art loaded",
            anchor="center",
            width=28,
            padding=8,
        )
        self.cover_label.pack(fill="both", expand=True)

        metadata_frame = ttk.LabelFrame(content, text="Metadata")
        metadata_frame.pack(side="left", fill="both", expand=True)
        self.metadata_text = ScrolledText(
            metadata_frame, height=16, wrap="word", state="disabled"
        )
        self.metadata_text.pack(fill="both", expand=True, padx=6, pady=6)

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Button(controls, text="Close", command=self.destroy).pack(side="left")

    def _choose_file(self) -> None:
        initial_dir = load_last_path() or os.getcwd()
        chosen = filedialog.askopenfilename(
            parent=self,
            title="Select Opus File",
            initialdir=initial_dir,
            filetypes=[("Opus files", "*.opus")],
        )
        if not chosen:
            return
        if not chosen.lower().endswith(".opus"):
            messagebox.showerror("Opus Tester", "Please select a .opus file.")
            return
        self.file_path_var.set(chosen)
        save_last_path(os.path.dirname(chosen))
        self._load_metadata(chosen)

    def _load_metadata(self, path: str) -> None:
        self._set_metadata_text("")
        self._set_cover_image(None)

        tags, cover_payloads, error = read_opus_metadata(path)

        def _format_value(value: object) -> str | None:
            if value in (None, "", []):
                return None
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value if item not in (None, ""))
            return str(value)

        lines = []
        track_value = _format_value(tags.get("tracknumber") or tags.get("track"))
        disc_value = _format_value(tags.get("discnumber") or tags.get("disc"))
        display_fields = [
            ("Title", "title"),
            ("Artist", "artist"),
            ("Album", "album"),
            ("Album Artist", "albumartist"),
            ("Track", track_value),
            ("Disc", disc_value),
            ("Year", "year"),
            ("Date", "date"),
            ("Genre", "genre"),
            ("Compilation", "compilation"),
        ]
        for label, key in display_fields:
            value = _format_value(tags.get(key)) if isinstance(key, str) else key
            if value:
                lines.append(f"{label}: {value}")
        if error and not lines:
            lines.append(f"Metadata error: {error}")
        self._set_metadata_text("\n".join(lines) if lines else "No metadata found.")

        cover_bytes = cover_payloads[0] if cover_payloads else None
        if cover_bytes:
            try:
                image = Image.open(BytesIO(cover_bytes))
                image.thumbnail((240, 240))
                self._set_cover_image(image)
            except Exception:
                self.cover_label.config(text="Failed to load album art")
        else:
            self.cover_label.config(text="No album art found")

    def _set_metadata_text(self, content: str) -> None:
        self.metadata_text.configure(state="normal")
        self.metadata_text.delete("1.0", tk.END)
        self.metadata_text.insert("1.0", content)
        self.metadata_text.configure(state="disabled")

    def _set_cover_image(self, image: Image.Image | None) -> None:
        if image is None:
            self._cover_photo = None
            self.cover_label.configure(image="", text="No album art loaded")
            return
        self._cover_photo = ImageTk.PhotoImage(image)
        self.cover_label.configure(image=self._cover_photo, text="")


class OpusLibraryMirrorDialog(tk.Toplevel):
    """Create a mirrored library with FLAC files converted to Opus."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("Opus Library Mirror")
        self.transient(parent)
        self.resizable(True, False)
        self.parent = parent

        default_path = ""
        if hasattr(parent, "library_path_var"):
            default_path = parent.library_path_var.get()

        self.source_var = tk.StringVar(value=default_path)
        self.destination_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready to mirror library.")
        self.overwrite_var = tk.BooleanVar(value=False)
        self._worker: threading.Thread | None = None
        self.report_path: str | None = None

        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Create a mirror of your library with FLAC files converted to Opus (96 kbps).",
            foreground="#555",
            wraplength=520,
        ).pack(anchor="w")

        source_frame = ttk.LabelFrame(container, text="Source Library")
        source_frame.pack(fill="x", pady=(10, 6))
        ttk.Entry(source_frame, textvariable=self.source_var, width=60).grid(
            row=0, column=0, sticky="ew", padx=6, pady=6
        )
        ttk.Button(source_frame, text="Browse…", command=self._browse_source).grid(
            row=0, column=1, sticky="e", padx=6, pady=6
        )
        source_frame.columnconfigure(0, weight=1)

        dest_frame = ttk.LabelFrame(container, text="Destination Mirror")
        dest_frame.pack(fill="x", pady=(0, 6))
        ttk.Entry(dest_frame, textvariable=self.destination_var, width=60).grid(
            row=0, column=0, sticky="ew", padx=6, pady=6
        )
        ttk.Button(dest_frame, text="Browse…", command=self._browse_destination).grid(
            row=0, column=1, sticky="e", padx=6, pady=6
        )
        dest_frame.columnconfigure(0, weight=1)

        ttk.Checkbutton(
            container,
            text="Overwrite existing files in the destination",
            variable=self.overwrite_var,
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(container, textvariable=self.status_var).pack(anchor="w", pady=(4, 8))
        self.progress = ttk.Progressbar(
            container,
            mode="determinate",
        )
        self.progress.pack(fill="x", pady=(0, 8))

        controls = ttk.Frame(container)
        controls.pack(anchor="e")
        self.run_btn = ttk.Button(controls, text="Start", command=self._start)
        self.run_btn.pack(side="right", padx=(6, 0))
        self.close_btn = ttk.Button(controls, text="Close", command=self.destroy)
        self.close_btn.pack(side="right")
        self.open_report_btn = ttk.Button(
            controls,
            text="Open Report",
            command=self._open_report,
            state="disabled",
        )
        self.open_report_btn.pack(side="right", padx=(0, 6))

    def _browse_source(self) -> None:
        initial = self.source_var.get() or load_last_path() or os.getcwd()
        folder = filedialog.askdirectory(
            parent=self, title="Select Source Library", initialdir=initial
        )
        if folder:
            self.source_var.set(folder)

    def _browse_destination(self) -> None:
        initial = self.destination_var.get() or load_last_path() or os.getcwd()
        folder = filedialog.askdirectory(
            parent=self, title="Select Destination Folder", initialdir=initial
        )
        if folder:
            self.destination_var.set(folder)

    def _start(self) -> None:
        source = self.source_var.get().strip()
        destination = self.destination_var.get().strip()

        if not source or not os.path.isdir(source):
            messagebox.showwarning(
                "Source Required", "Select a valid source library folder."
            )
            return
        if not destination or not os.path.isdir(destination):
            messagebox.showwarning(
                "Destination Required", "Select a valid destination folder."
            )
            return
        if os.path.abspath(source) == os.path.abspath(destination):
            messagebox.showerror(
                "Invalid Destination",
                "The destination folder must be different from the source.",
            )
            return
        try:
            if os.path.commonpath([source, destination]) == os.path.abspath(source):
                messagebox.showerror(
                    "Invalid Destination",
                    "The destination cannot be inside the source library.",
                )
                return
        except ValueError:
            pass
        if not shutil.which("ffmpeg"):
            messagebox.showerror(
                "FFmpeg Required",
                "FFmpeg was not found on your PATH. Install it to convert FLAC files.",
            )
            return

        self._set_running(True, "Scanning library…")
        overwrite = self.overwrite_var.get()
        self.report_path = None
        self.open_report_btn.configure(state="disabled")

        def worker() -> None:
            try:
                summary = self._mirror_library(source, destination, overwrite)
                self.after(0, lambda: self._finished(summary))
            except Exception as exc:  # pragma: no cover - safety net
                self.after(0, lambda: self._failed(str(exc)))

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _set_running(self, running: bool, status: str) -> None:
        self.status_var.set(status)
        state = "disabled" if running else "normal"
        self.run_btn.configure(state=state)
        self.close_btn.configure(state=state)
        if running:
            self.progress.configure(maximum=1, value=0)

    def _log(self, message: str) -> None:
        if hasattr(self.parent, "_log"):
            self.parent._log(message)
            if hasattr(self.parent, "show_log_tab"):
                self.parent.show_log_tab()
        else:
            print(message)

    def _mirror_library(
        self, source: str, destination: str, overwrite: bool
    ) -> dict[str, object]:
        def progress_callback(
            total_tasks: int,
            completed: int,
            converted: int,
            copied: int,
            skipped: int,
            errors: int,
        ) -> None:
            self.after(
                0,
                lambda t=total_tasks, d=completed: (
                    self.status_var.set("Mirroring library…"),
                    self.progress.configure(maximum=max(t, 1)),
                    self.progress.configure(value=d),
                ),
            )

        return mirror_library(
            source,
            destination,
            overwrite,
            progress_callback=progress_callback,
            log_callback=self._log,
        )

    def _finished(self, summary: dict[str, object]) -> None:
        self._set_running(False, "Completed.")
        message = (
            "Mirror complete.\n\n"
            f"Files processed: {summary['total']}\n"
            f"FLAC converted: {summary['converted']}\n"
            f"Other files copied: {summary['copied']}\n"
            f"Skipped: {summary['skipped']}\n"
            f"Errors: {summary['errors']}"
        )
        self.status_var.set(
            f"Done. Converted {summary['converted']} and copied {summary['copied']}."
        )
        messagebox.showinfo("Opus Library Mirror", message)
        self._write_report(summary)
        self._log(
            "Opus Library Mirror: "
            f"total={summary['total']} converted={summary['converted']} "
            f"copied={summary['copied']} skipped={summary['skipped']} "
            f"errors={summary['errors']}"
        )

    def _failed(self, error: str) -> None:
        self._set_running(False, "Failed.")
        messagebox.showerror("Opus Library Mirror", f"Run failed:\n{error}")

    def _write_report(self, summary: dict[str, object]) -> None:
        destination = self.destination_var.get().strip()
        if not destination:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(
            destination,
            "Docs",
            "opus_library_mirror_reports",
        )
        report_path = os.path.join(
            reports_dir,
            f"opus_library_mirror_report_{timestamp}.html",
        )
        try:
            write_mirror_report(report_path, summary)
        except OSError as exc:
            self._log(f"Opus Library Mirror: failed to write report: {exc}")
            self.report_path = None
            self.open_report_btn.configure(state="disabled")
            return
        self.report_path = report_path
        self.open_report_btn.configure(state="normal")
        self._log(f"Opus Library Mirror: report saved to {report_path}")

    def _open_report(self) -> None:
        if not self.report_path:
            messagebox.showinfo("Opus Library Mirror", "No report is available yet.")
            return
        safe_path = ensure_long_path(self.report_path)
        if not os.path.exists(safe_path):
            messagebox.showerror(
                "Report Missing",
                "The report could not be found. Run the mirror again to generate a new report.",
            )
            return
        display_path = strip_ext_prefix(self.report_path)
        try:
            uri = Path(display_path).resolve().as_uri()
        except Exception:
            uri = display_path
        webbrowser.open(uri)


class DuplicateBucketingPocDialog(tk.Toplevel):
    """Minimal UI for the Duplicate Bucketing proof-of-concept tool."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("Duplicate Bucketing POC")
        self.transient(parent)
        self.resizable(False, False)
        self.parent = parent

        default_path = ""
        if hasattr(parent, "library_path_var"):
            default_path = parent.library_path_var.get()

        self.folder_var = tk.StringVar(value=default_path)
        self.status_var = tk.StringVar(value="Idle")
        self.run_btn: ttk.Button | None = None

        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="Select a folder to scan for duplicate bucketing.",
            foreground="#555",
            wraplength=420,
        ).pack(anchor="w")

        row = ttk.Frame(container)
        row.pack(fill="x", pady=(10, 6))
        ttk.Entry(row, textvariable=self.folder_var, width=50).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self._browse_folder).pack(side="left", padx=(6, 0))

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(4, 0))
        self.run_btn = ttk.Button(controls, text="Run", command=self._run)
        self.run_btn.pack(side="left")
        ttk.Button(controls, text="Close", command=self.destroy).pack(side="left", padx=(6, 0))

        ttk.Label(container, textvariable=self.status_var).pack(anchor="w", pady=(8, 0))

    def _browse_folder(self) -> None:
        initial_dir = load_last_path() or os.getcwd()
        chosen = filedialog.askdirectory(
            title="Select Folder for Duplicate Bucketing",
            initialdir=initial_dir,
        )
        if chosen:
            self.folder_var.set(chosen)
            save_last_path(chosen)

    def _set_running(self, running: bool, status: str) -> None:
        self.status_var.set(status)
        if self.run_btn:
            self.run_btn.configure(state=("disabled" if running else "normal"))

    def _log(self, message: str) -> None:
        if hasattr(self.parent, "_log"):
            self.parent._log(message)
            if hasattr(self.parent, "show_log_tab"):
                self.parent.show_log_tab()
        else:
            print(message)

    def _run(self) -> None:
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Folder Required", "Please select a valid folder.")
            return

        self._set_running(True, "Running…")

        def worker() -> None:
            try:
                report_path = run_duplicate_bucketing_poc(folder, log_callback=self._log)
                self.after(0, lambda: self._on_complete(report_path))
            except Exception as exc:  # pragma: no cover - safety net
                self.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_complete(self, report_path: str) -> None:
        self._set_running(False, "Completed")
        open_report = messagebox.askyesno(
            "Duplicate Bucketing POC",
            f"Report saved to:\n{report_path}\n\nOpen it now?",
        )
        if open_report:
            try:
                uri = Path(report_path).resolve().as_uri()
            except Exception:
                uri = report_path
            webbrowser.open(uri)

    def _on_error(self, error: str) -> None:
        self._set_running(False, "Failed")
        messagebox.showerror("Duplicate Bucketing POC", f"Run failed:\n{error}")


class FileCleanupDialog(tk.Toplevel):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("File Clean Up")
        self.transient(parent)
        self.resizable(False, False)
        self.parent = parent
        self.library_path_var = tk.StringVar(
            value=getattr(parent, "library_path", "") or ""
        )
        self.status_var = tk.StringVar(value="Ready to scan.")
        self._worker: threading.Thread | None = None

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        ttk.Label(
            container,
            text="File Clean Up",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            container,
            text=(
                "Remove trailing (numbers) from audio filenames in the library. "
                "Any collisions with existing filenames are skipped."
            ),
            foreground="#555",
            wraplength=420,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(container, text="Library Root").pack(anchor="w")
        lib_row = ttk.Frame(container)
        lib_row.pack(fill="x", pady=(0, 8))
        ttk.Entry(lib_row, textvariable=self.library_path_var, width=60).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(lib_row, text="Browse…", command=self._browse_library).pack(
            side="left", padx=(6, 0)
        )

        ttk.Label(container, textvariable=self.status_var).pack(anchor="w", pady=(4, 8))

        btn_row = ttk.Frame(container)
        btn_row.pack(anchor="e")
        self.execute_btn = ttk.Button(
            btn_row, text="Execute", command=self._execute_cleanup
        )
        self.execute_btn.pack(side="right", padx=(6, 0))
        self.close_btn = ttk.Button(btn_row, text="Cancel", command=self.destroy)
        self.close_btn.pack(side="right")

    def _browse_library(self) -> None:
        initial = self.library_path_var.get() or load_last_path() or os.getcwd()
        folder = filedialog.askdirectory(
            parent=self, title="Select Library Root", initialdir=initial
        )
        if folder:
            self.library_path_var.set(folder)

    def _execute_cleanup(self) -> None:
        library_root = self.library_path_var.get()
        if not library_root:
            messagebox.showwarning(
                "No Library", "Select a library folder before running cleanup."
            )
            return
        if not os.path.isdir(library_root):
            messagebox.showerror(
                "Invalid Library", "The selected folder does not exist."
            )
            return

        self.execute_btn.configure(state="disabled")
        self.status_var.set("Scanning files…")

        def task() -> None:
            summary = self._run_cleanup(library_root)
            self.after(0, lambda: self._cleanup_finished(summary))

        self._worker = threading.Thread(target=task, daemon=True)
        self._worker.start()

    def _run_cleanup(self, library_root: str) -> dict[str, int | list[str]]:
        total = 0
        renamed = 0
        skipped = 0
        conflicts = 0
        errors = 0
        skipped_files: list[str] = []
        pattern = re.compile(r"\s*\(\d+\)$")

        for root, _dirs, files in os.walk(library_root):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                total += 1
                stem = os.path.splitext(filename)[0]
                new_stem = pattern.sub("", stem)
                if new_stem == stem:
                    skipped += 1
                    continue
                new_name = f"{new_stem}{ext}"
                src = os.path.join(root, filename)
                dst = os.path.join(root, new_name)
                if os.path.exists(dst):
                    conflicts += 1
                    skipped_files.append(src)
                    continue
                try:
                    os.rename(src, dst)
                    renamed += 1
                except OSError:
                    errors += 1

        return {
            "total": total,
            "renamed": renamed,
            "skipped": skipped,
            "conflicts": conflicts,
            "errors": errors,
            "skipped_files": skipped_files,
        }

    def _cleanup_finished(self, summary: dict[str, int | list[str]]) -> None:
        self.execute_btn.configure(state="normal")
        self.close_btn.configure(text="Close")
        message = (
            "Finished. "
            f"Renamed {summary['renamed']} of {summary['total']} files. "
            f"Skipped {summary['skipped']}, conflicts {summary['conflicts']}, "
            f"errors {summary['errors']}."
        )
        self.status_var.set(message)
        if hasattr(self.parent, "_log"):
            self.parent._log(
                "File Clean Up: "
                f"renamed={summary['renamed']} total={summary['total']} "
                f"skipped={summary['skipped']} conflicts={summary['conflicts']} "
                f"errors={summary['errors']}"
            )
            skipped_files = summary.get("skipped_files") or []
            if skipped_files:
                skipped_details = "\n".join(
                    f"- {path}" for path in sorted(skipped_files)
                )
                self.parent._log(
                    "File Clean Up: skipped files due to name conflicts:\n"
                    f"{skipped_details}"
                )


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
        # Folder to run Duplicate Finder - now always uses library_path
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

        self.use_review_sync_var = tk.BooleanVar(
            value=cfg.get("use_library_sync_review", False)
        )
        self.sync_review_window: library_sync_review.LibrarySyncReviewWindow | None = None
        self.sync_review_panel: library_sync_review.LibrarySyncReviewPanel | None = None

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

        # Duplicate Finder state
        self.duplicate_finder_window: DuplicateFinderShell | None = None
        self.duplicate_pair_review_window: DuplicatePairReviewTool | None = None
        self.duplicate_finder_plan: ConsolidationPlan | None = None
        self.duplicate_finder_plan_library: str | None = None

        # Shared preview/playback state
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
        self._preview_in_progress = False
        self._player_busy_item: str | None = None
        self._ignore_next_preview_finish = False
        self._dup_preview_heartbeat_running = False
        self._dup_preview_heartbeat_start: float | None = None
        self._dup_preview_heartbeat_last_tick: float | None = None
        self._dup_preview_heartbeat_count = 0
        self._dup_preview_heartbeat_interval = 0.1
        self._dup_preview_heartbeat_stall_logged = False

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
        self.player_playlist_mode = False
        self.player_playlist_rows: list[dict[str, str]] = []
        self.player_playlist_tree_paths: dict[str, str] = {}
        self.player_playlist_tree_rows: dict[str, dict[str, str]] = {}
        self.player_playlist_tree: ttk.Treeview | None = None
        self.player_playlist_frame: ttk.Frame | None = None

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
        tools_menu.add_command(
            label="Library Sync…", command=self._open_library_sync_tool
        )
        tools_menu.add_checkbutton(
            label="Use Library Sync (Review)",
            variable=self.use_review_sync_var,
            command=self._toggle_library_sync_mode,
        )
        tools_menu.add_separator()
        tools_menu.add_command(label="Fix Tags via AcoustID", command=self.fix_tags_gui)
        tools_menu.add_command(
            label="Generate Library Index…",
            command=lambda: generate_index(self.require_library()),
        )
        tools_menu.add_command(
            label="Export Artist/Title List…",
            command=self._export_artist_title_list,
        )
        tools_menu.add_command(
            label="Playlist Artwork", command=self.open_playlist_artwork_folder
        )
        tools_menu.add_command(label="File Clean Up…", command=self._open_file_cleanup_tool)
        tools_menu.add_command(
            label="Similarity Inspector…", command=self._open_similarity_inspector_tool
        )
        tools_menu.add_command(
            label="Duplicate Bucketing POC…",
            command=self._open_duplicate_bucketing_poc_tool,
        )
        tools_menu.add_command(
            label="Duplicate Pair Review…",
            command=self._open_duplicate_pair_review_tool,
        )
        tools_menu.add_command(label="M4A Tester…", command=self._open_m4a_tester_tool)
        tools_menu.add_command(label="Opus Tester…", command=self._open_opus_tester_tool)
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
            "Year Assistant",
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

        # ─── Duplicate Finder Tab ─────────────────────────────────────────
        self.dup_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dup_tab, text="Duplicate Finder")

        df_container = ttk.LabelFrame(self.dup_tab, text="Duplicate Finder")
        df_container.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(
            df_container,
            textvariable=self.library_path_var,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Label(
            df_container,
            text=(
                "Preview duplicate groups and launch the Duplicate Finder workflow. "
                "Execution writes backups and reports under the library Docs folder."
            ),
            wraplength=520,
            justify="left",
        ).pack(anchor="w", padx=10, pady=(0, 6))
        self.scan_btn = ttk.Button(
            df_container,
            text="Open Duplicate Finder",
            command=self.scan_duplicates,
            state="disabled",
        )
        self.scan_btn.pack(anchor="w", padx=10, pady=(4, 10))
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

        self.player_table_region = ttk.Frame(player_content)
        self.player_table_region.pack(side="left", fill="both", expand=True)
        self.player_library_frame = self._create_player_library_table(
            self.player_table_region
        )
        self.player_library_frame.pack(fill="both", expand=True)

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
        self.player_playlist_toggle_btn = ttk.Button(
            art_panel, text="Add Playlist", command=self._toggle_player_playlist_view
        )
        self.player_playlist_toggle_btn.pack(fill="x", pady=(0, 6))
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

        # ─── Library Compression Tab ──────────────────────────────────────
        self.library_compression_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.library_compression_tab, text="Library Compression")
        self.library_compression_panel = LibraryCompressionPanel(
            self.library_compression_tab,
            controller=self,
        )
        self.library_compression_panel.pack(fill="both", expand=True)

        # ─── Library Sync Tab ─────────────────────────────────────────────
        self.sync_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.sync_tab, text="Library Sync")
        self.sync_review_panel = library_sync_review.LibrarySyncReviewPanel(
            self.sync_tab,
            library_root=self.library_path or "",
        )
        self.sync_review_panel.pack(fill="both", expand=True)

        # after your other tabs
        self.help_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.help_tab, text="Help")

        # Chat history display
        self.chat_history = ScrolledText(
            self.help_tab, height=15, state="disabled", wrap="word"
        )
        self.chat_history.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        # Entry + send button
        entry_frame = ttk.Frame(self.help_tab)
        entry_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.chat_input = ttk.Entry(entry_frame)
        self.chat_input.pack(side="left", fill="x", expand=True)
        send_btn = ttk.Button(entry_frame, text="Send", command=self._send_help_query)
        send_btn.pack(side="right", padx=(5, 0))

        self._reorder_tabs()

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
        self.mapping_path = os.path.join(self.library_path, "Docs", ".genre_mapping.json")
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
        self.update_library_info()
        if hasattr(self, "player_reload_btn"):
            self.player_reload_btn.config(state="normal")
        self._load_player_library_async()
        if self.sync_review_panel:
            self.sync_review_panel.set_folders(library_root=self.library_path)

    def update_library_info(self):
        if not self.library_path:
            self.library_stats_var.set("")
            if hasattr(self, "player_reload_btn"):
                self.player_reload_btn.config(state="disabled")
            if hasattr(self, "scan_btn"):
                self.scan_btn.config(state="disabled")
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

    def _open_path(self, path: str) -> None:
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
            messagebox.showerror("Open Folder", f"Could not open {path}: {exc}")

    def open_playlist_artwork_folder(self) -> None:
        """Ensure the playlist artwork folder exists and open it for the user."""

        library = self.require_library()
        if not library:
            return

        playlists_dir = os.path.join(library, "Playlists")
        artwork_dir = os.path.join(playlists_dir, "artwork")

        try:
            os.makedirs(artwork_dir, exist_ok=True)
        except Exception as exc:
            messagebox.showerror(
                "Playlist Artwork",
                f"Could not create the artwork folder:\n{exc}",
            )
            self._log(f"✘ Failed to prepare playlist artwork folder: {exc}")
            return

        info = (
            "Playlist artwork folder is ready.\n\n"
            f"Location:\n{artwork_dir}\n\n"
            "Drop cover images here to accompany your playlists."
        )
        messagebox.showinfo("Playlist Artwork", info)
        self._log(f"🎨 Playlist artwork folder ready at {artwork_dir}")
        self._open_path(artwork_dir)

    def _clean_tag_text(self, value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            for encoding in ("utf-8", "utf-16", "latin-1"):
                try:
                    return value.decode(encoding).strip() or None
                except UnicodeDecodeError:
                    continue
            return value.decode("utf-8", errors="replace").strip() or None
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        try:
            cleaned = str(value).strip()
            return cleaned or None
        except Exception:
            return None

    def _export_artist_title_list(self) -> None:
        library = self.require_library()
        if not library:
            return

        docs_dir = os.path.join(library, "Docs")
        try:
            os.makedirs(docs_dir, exist_ok=True)
        except OSError as exc:
            messagebox.showerror(
                "Export Artist/Title List",
                f"Could not create documentation folder:\n{exc}",
            )
            return

        output_path = os.path.join(docs_dir, "artist_title_list.txt")
        exts = {".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus", ".flac"}
        q: queue.Queue[tuple[str, object]] = queue.Queue()
        running = tk.BooleanVar(value=False)
        exclude_flac_var = tk.BooleanVar(value=False)
        add_album_duplicates_var = tk.BooleanVar(value=False)

        dlg = tk.Toplevel(self)
        dlg.title("Export Artist/Title List")
        dlg.resizable(True, False)

        header = ttk.Frame(dlg)
        header.pack(fill="x", padx=12, pady=(12, 6))
        ttk.Label(
            header,
            text="Export Artist/Title List",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Generate a sorted list of all artist/title pairs in the library.",
            wraplength=520,
        ).pack(anchor="w", pady=(2, 0))

        info_frame = ttk.LabelFrame(dlg, text="Library Path")
        info_frame.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Label(info_frame, text=library, wraplength=520).pack(
            anchor="w", padx=8, pady=6
        )

        output_frame = ttk.LabelFrame(dlg, text="Output File")
        output_frame.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Label(output_frame, text=output_path, wraplength=520).pack(
            anchor="w", padx=8, pady=6
        )

        options_frame = ttk.LabelFrame(dlg, text="Options")
        options_frame.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Checkbutton(
            options_frame,
            text="Exclude flac files",
            variable=exclude_flac_var,
        ).pack(anchor="w", padx=8, pady=6)
        ttk.Checkbutton(
            options_frame,
            text="Add album song duplicates",
            variable=add_album_duplicates_var,
        ).pack(anchor="w", padx=8, pady=0)

        progress_var = tk.StringVar(value="Ready to start.")
        progress = ttk.Progressbar(dlg, mode="determinate")
        progress.pack(fill="x", padx=12)
        ttk.Label(dlg, textvariable=progress_var).pack(
            anchor="w", padx=14, pady=(4, 0)
        )

        log_group = ttk.LabelFrame(dlg, text="Export Log")
        log_group.pack(fill="both", expand=True, padx=12, pady=(10, 0))
        log_box = ScrolledText(log_group, height=8, state="disabled")
        log_box.pack(fill="both", expand=True, padx=6, pady=6)

        button_frame = ttk.Frame(dlg)
        button_frame.pack(fill="x", padx=12, pady=12)

        def append_log(message: str) -> None:
            self._log(message)
            log_box.configure(state="normal")
            log_box.insert("end", message + "\n")
            log_box.see("end")
            log_box.configure(state="disabled")

        def set_controls(active: bool) -> None:
            start_btn.config(state="disabled" if active else "normal")
            open_state = (
                "normal" if not active and open_btn_enabled.get() else "disabled"
            )
            open_btn.config(state=open_state)
            close_btn.config(state="disabled" if active else "normal")

        def start_export() -> None:
            if running.get():
                return
            running.set(True)
            open_btn_enabled.set(False)
            progress_var.set("Scanning library for audio files…")
            progress.config(mode="indeterminate", value=0)
            progress.start(10)
            log_box.configure(state="normal")
            log_box.delete("1.0", "end")
            log_box.configure(state="disabled")
            append_log("Starting artist/title export…")
            append_log(f"Library: {library}")
            set_controls(active=True)

            def worker() -> None:
                try:
                    audio_files: list[str] = []
                    exclude_flac = exclude_flac_var.get()
                    for dirpath, _, files in os.walk(library):
                        for filename in files:
                            ext = os.path.splitext(filename)[1].lower()
                            if exclude_flac and ext == ".flac":
                                continue
                            if ext in exts:
                                audio_files.append(os.path.join(dirpath, filename))
                    q.put(("files", len(audio_files)))

                    entries: list[str] = []
                    entry_data: list[tuple[str, str, str | None, str | None]] = []
                    error_count = 0
                    for idx, full_path in enumerate(audio_files, start=1):
                        ext = os.path.splitext(full_path)[1].lower()
                        filename = os.path.basename(full_path)
                        if ext == ".opus":
                            tags, _covers, error = read_opus_metadata(full_path)
                            if error:
                                error_count += 1
                                q.put(
                                    ("log", f"Skipped unreadable OPUS file: {filename}")
                                )
                        else:
                            tags = read_tags(full_path)
                        artist = self._clean_tag_text(
                            tags.get("artist") or tags.get("albumartist")
                        )
                        title = self._clean_tag_text(tags.get("title"))
                        album = self._clean_tag_text(tags.get("album"))
                        track = self._clean_tag_text(
                            tags.get("tracknumber") or tags.get("track")
                        )
                        if track and "/" in track:
                            track = track.split("/", 1)[0].strip() or None
                        if not title:
                            title = os.path.splitext(filename)[0]
                        if not artist:
                            artist = "Unknown Artist"
                        entry_data.append((artist, title, album, track))
                        if idx == 1 or idx % 50 == 0 or idx == len(audio_files):
                            q.put(("progress", idx, len(audio_files)))

                    add_album_duplicates = add_album_duplicates_var.get()
                    duplicate_counts = Counter(
                        (artist, title) for artist, title, _album, _track in entry_data
                    )
                    album_duplicate_counts = Counter(
                        (artist, title, album)
                        for artist, title, album, _track in entry_data
                    )
                    for artist, title, album, track in entry_data:
                        if add_album_duplicates and duplicate_counts[(artist, title)] > 1:
                            album_label = album or "Unknown Album"
                            if album_duplicate_counts[(artist, title, album)] > 1:
                                track_label = track or "Unknown Track"
                                entries.append(
                                    f"{artist} - {title} - {album_label} - {track_label}"
                                )
                            else:
                                entries.append(f"{artist} - {title} - {album_label}")
                        else:
                            entries.append(f"{artist} - {title}")

                    entries = sorted(set(entries), key=str.lower)
                    with open(output_path, "w", encoding="utf-8") as handle:
                        handle.write("\n".join(entries))
                    q.put(("done", output_path, len(entries), error_count))
                except Exception as exc:  # pragma: no cover - UI surface only
                    q.put(("error", str(exc)))

            threading.Thread(target=worker, daemon=True).start()
            poll_queue()

        def poll_queue() -> None:
            try:
                while True:
                    message = q.get_nowait()
                    tag = message[0]
                    if tag == "log":
                        append_log(message[1])
                    elif tag == "files":
                        total = message[1]
                        progress.stop()
                        progress.config(
                            mode="determinate", maximum=max(1, total), value=0
                        )
                        progress_var.set(f"0/{total} processed")
                        append_log(f"Found {total} audio files to process.")
                    elif tag == "progress":
                        value, total = message[1], message[2]
                        progress["value"] = value
                        progress_var.set(f"{value}/{total} processed")
                    elif tag == "done":
                        _, path, count, error_count = message
                        running.set(False)
                        open_btn_enabled.set(True)
                        progress["value"] = progress["maximum"]
                        note = ""
                        if error_count:
                            note = f" ({error_count} files skipped)"
                        progress_var.set(f"Completed – {count} entries{note}")
                        append_log(f"Export complete: {path}")
                        if error_count:
                            append_log(f"Skipped {error_count} files due to read errors.")
                        set_controls(active=False)
                    elif tag == "error":
                        running.set(False)
                        progress.stop()
                        progress_var.set("Export failed")
                        set_controls(active=False)
                        messagebox.showerror("Export Artist/Title List", message[1])
            except queue.Empty:
                pass
            if running.get():
                dlg.after(200, poll_queue)

        open_btn_enabled = tk.BooleanVar(value=False)

        start_btn = ttk.Button(button_frame, text="Start Export", command=start_export)
        start_btn.pack(side="left")
        open_btn = ttk.Button(
            button_frame,
            text="Open List",
            command=lambda: self._open_path(output_path),
            state="disabled",
        )
        open_btn.pack(side="left", padx=(6, 0))
        close_btn = ttk.Button(button_frame, text="Close", command=dlg.destroy)
        close_btn.pack(side="right")

        def on_close() -> None:
            if running.get():
                messagebox.showinfo(
                    "Export Artist/Title List",
                    "Please wait for the export to finish before closing this window.",
                )
                return
            dlg.destroy()

        dlg.protocol("WM_DELETE_WINDOW", on_close)

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

    def scan_duplicates(self):
        library = self.require_library()
        if not library:
            return

        existing = getattr(self, "duplicate_finder_window", None)
        if existing and existing.winfo_exists():
            existing.lift()
            existing.focus_set()
            return

        win = DuplicateFinderShell(self, library_path=library)
        win.bind(
            "<Destroy>",
            lambda e: (
                setattr(self, "duplicate_finder_window", None)
                if e.widget is win
                else None
            ),
        )
        self.duplicate_finder_window = win

    def _set_duplicate_finder_plan(self, plan: ConsolidationPlan, library_path: str) -> None:
        self.duplicate_finder_plan = plan
        self.duplicate_finder_plan_library = library_path

    def _load_duplicate_plan_from_preview(self, library_path: str) -> ConsolidationPlan | None:
        preview_path = os.path.join(library_path, "Docs", "duplicate_preview.json")
        try:
            with open(preview_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError:
            return None
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or "plan" not in payload:
            return None
        try:
            return consolidation_plan_from_dict(payload["plan"])
        except ValueError:
            return None

    def _open_duplicate_pair_review_tool(self) -> None:
        library = self.require_library()
        if not library:
            return
        existing = getattr(self, "duplicate_pair_review_window", None)
        if existing and existing.winfo_exists():
            existing.lift()
            existing.focus_set()
            return
        plan = self.duplicate_finder_plan
        if plan is None or self.duplicate_finder_plan_library != library:
            plan = self._load_duplicate_plan_from_preview(library)
        if plan is None:
            messagebox.showinfo(
                "Duplicate Pair Review",
                "Run Duplicate Finder preview first to generate paired results.",
            )
            return
        win = DuplicatePairReviewTool(self, library_path=library, plan=plan)
        win.bind(
            "<Destroy>",
            lambda e: (
                setattr(self, "duplicate_pair_review_window", None)
                if e.widget is win
                else None
            ),
        )
        self.duplicate_pair_review_window = win

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
                        lambda: webbrowser.open(
                            Path(output_html).resolve().as_uri()
                        ),
                    )
                    self.after(
                        0,
                        lambda: messagebox.showinfo(
                            "Dry Run Complete",
                            f"Preview opened in your browser:\n{output_html}",
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
        self.mapping_path = os.path.join(folder, "Docs", ".genre_mapping.json")
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

    def initialize_genre_normalizer(self, mapping: dict, progress_callback=None):
        """Persist mapping JSON and rewrite genre tags across the library."""
        if not self.library_path or not getattr(self, "mapping_path", None):
            raise ValueError("Library must be selected before initializing the normalizer.")

        normalized_map: dict[str, list[str]] = {}
        for raw, value in mapping.items():
            if value is None:
                continue
            key = str(raw).strip()
            if not key:
                continue
            if isinstance(value, list):
                cleaned = [str(v).strip() for v in value if str(v).strip()]
            else:
                cleaned = [str(value).strip()] if str(value).strip() else []
            if cleaned:
                normalized_map[key] = cleaned

        os.makedirs(os.path.dirname(self.mapping_path), exist_ok=True)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(normalized_map, f, indent=2)
        self.genre_mapping = normalized_map

        files = discover_files(self.library_path)
        total = len(files)
        if progress_callback:
            progress_callback(0, total)

        changed = 0
        for idx, path in enumerate(files, start=1):
            if progress_callback:
                progress_callback(idx, total)
            tags = read_tags(path)
            raw = tags.get("genre")
            if raw in (None, ""):
                continue
            if isinstance(raw, (list, tuple)):
                existing_genres = [str(v) for v in raw if isinstance(v, str)]
            else:
                existing_genres = [str(raw)]
            rewritten: list[str] = []
            seen: set[str] = set()
            for entry in existing_genres:
                parts = re.split(r"[;,/]", entry)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    mapped = normalized_map.get(part, [part])
                    values = mapped if isinstance(mapped, list) else [mapped]
                    for m in values:
                        val = str(m).strip()
                        if val and val not in seen:
                            seen.add(val)
                            rewritten.append(val)

            if not rewritten or sorted(rewritten) == sorted(existing_genres):
                continue

            try:
                audio["genre"] = rewritten
                audio.save()
                changed += 1
            except Exception:
                continue

        return changed, total

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
    def _toggle_library_sync_mode(self) -> None:
        cfg = load_config()
        cfg["use_library_sync_review"] = bool(self.use_review_sync_var.get())
        save_config(cfg)

    def _open_library_sync_tool(self) -> None:
        if self.use_review_sync_var.get():
            if self.sync_review_window and self.sync_review_window.winfo_exists():
                self.sync_review_window.lift()
                self.sync_review_window.focus_set()
                return
            try:
                self.sync_review_window = library_sync_review.LibrarySyncReviewWindow(
                    self
                )
                self.sync_review_window.bind(
                    "<Destroy>", lambda _e: setattr(self, "sync_review_window", None)
                )
            except Exception as exc:
                messagebox.showerror("Library Sync (Review)", str(exc))
        else:
            try:
                self.notebook.select(self.sync_tab)
            except Exception:
                messagebox.showinfo(
                    "Library Sync", "The Library Sync tab is not available."
                )

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
        if hasattr(self, "player_status_var"):
            if error:
                self.player_status_var.set("Playback failed. Check logs.")
                messagebox.showerror("Playback failed", error)
            else:
                self.player_status_var.set("Ready to play another track.")

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
                _tags, cover_payloads, _error, _reader = read_metadata(path, include_cover=True)
                if cover_payloads:
                    img = Image.open(BytesIO(cover_payloads[0]))
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

        selections: list[str] = []

        # Include selections from the main library table.
        selection = self.player_tree.selection()
        selections.extend(
            [self.player_tree_paths[item] for item in selection if item in self.player_tree_paths]
        )

        # Include selections from the optional playlist preview table when visible.
        if getattr(self, "player_playlist_tree", None):
            pl_selection = self.player_playlist_tree.selection()
            selections.extend(
                [
                    self.player_playlist_tree_paths[item]
                    for item in pl_selection
                    if item in self.player_playlist_tree_paths
                ]
            )

        return selections

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

    def _reorder_tabs(self) -> None:
        desired = [
            getattr(self, "log_tab", None),
            getattr(self, "indexer_tab", None),
            getattr(self, "dup_tab", None),
            getattr(self, "library_compression_tab", None),
            getattr(self, "sync_tab", None),
            getattr(self, "playlist_tab", None),
            getattr(self, "tagfix_tab", None),
            getattr(self, "player_tab", None),
            getattr(self, "help_tab", None),
        ]
        for tab in [t for t in desired if t is not None]:
            if str(tab) in self.notebook.tabs():
                self.notebook.insert("end", tab)

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

    def _open_similarity_inspector_tool(self) -> None:
        SimilarityInspectorDialog(self)

    def _open_file_cleanup_tool(self) -> None:
        FileCleanupDialog(self)

    def _open_duplicate_bucketing_poc_tool(self) -> None:
        DuplicateBucketingPocDialog(self)

    def _open_m4a_tester_tool(self) -> None:
        M4ATesterDialog(self)

    def _open_opus_tester_tool(self) -> None:
        OpusTesterDialog(self)

    def _create_player_library_table(self, parent: tk.Widget) -> ttk.Frame:
        table = ttk.Frame(parent)
        p_vsb = ttk.Scrollbar(table, orient="vertical")
        p_vsb.pack(side="right", fill="y")
        p_hsb = ttk.Scrollbar(table, orient="horizontal")
        p_hsb.pack(side="bottom", fill="x")

        cols = ("Title", "Artist", "Album", "Length", "Play")
        self.player_tree = ttk.Treeview(
            table,
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
        return table

    def _create_playlist_table(self, parent: tk.Widget) -> ttk.Treeview:
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        p_vsb = ttk.Scrollbar(frame, orient="vertical")
        p_vsb.pack(side="right", fill="y")
        p_hsb = ttk.Scrollbar(frame, orient="horizontal")
        p_hsb.pack(side="bottom", fill="x")

        cols = ("Title", "Artist", "Play")
        tree = ttk.Treeview(
            frame,
            columns=cols,
            show="headings",
            yscrollcommand=p_vsb.set,
            xscrollcommand=p_hsb.set,
        )
        p_vsb.config(command=tree.yview)
        p_hsb.config(command=tree.xview)
        tree.pack(fill="both", expand=True)

        widths = {"Title": 220, "Artist": 160, "Play": 50}
        for c in cols:
            tree.heading(c, text=c)
            tree.column(
                c,
                width=widths.get(c, 100),
                anchor="center" if c == "Play" else "w",
                stretch=c != "Play",
            )

        tree.bind("<ButtonRelease-1>", self._on_playlist_tree_click)
        tree.bind("<<TreeviewSelect>>", self._on_playlist_selection_change)
        return tree

    def _toggle_player_playlist_view(self) -> None:
        if self.player_playlist_mode:
            self._unload_player_playlist_view()
            return

        tracks = self._prompt_playlist_tracks()
        if not tracks:
            return
        self.player_playlist_rows = self._prepare_player_rows(tracks)
        self._enter_player_playlist_view()

    def _prompt_playlist_tracks(self) -> list[str] | None:
        playlists_dir = os.path.join(self.library_path, "Playlists")
        initial_dir = (
            playlists_dir
            if os.path.isdir(playlists_dir)
            else self.library_path
            if self.library_path
            else os.getcwd()
        )
        chosen = filedialog.askopenfilename(
            title="Add Playlist",
            initialdir=initial_dir,
            defaultextension=".m3u",
            filetypes=[("M3U Playlist", "*.m3u"), ("All Files", "*.*")],
        )
        if not chosen:
            return None

        try:
            with open(chosen, "r", encoding="utf-8") as f:
                entries = [line.strip() for line in f if line.strip()]
        except Exception as exc:
            messagebox.showerror("Playlist", f"Could not read playlist: {exc}")
            return None

        base_dir = os.path.dirname(chosen)
        resolved: list[str] = []
        for entry in entries:
            candidate = entry if os.path.isabs(entry) else os.path.join(base_dir, entry)
            candidate = os.path.normpath(candidate)
            if os.path.exists(candidate):
                resolved.append(candidate)

        if not resolved:
            messagebox.showinfo(
                "Playlist", "No valid tracks found in the selected playlist."
            )
            return None
        return resolved

    def _enter_player_playlist_view(self) -> None:
        self.player_playlist_mode = True
        self.player_playlist_toggle_btn.config(text="Unload Playlist")
        self.player_library_frame.pack_forget()
        self.player_library_frame.pack(
            side="left", fill="both", expand=True, padx=(0, 10), pady=(0, 0)
        )

        self.player_playlist_frame = ttk.LabelFrame(
            self.player_table_region, text="Playlist Preview"
        )
        self.player_playlist_frame.pack(side="left", fill="both", expand=True)
        self.player_playlist_tree = self._create_playlist_table(
            self.player_playlist_frame
        )
        self._render_playlist_rows(self.player_playlist_rows)

    def _unload_player_playlist_view(self) -> None:
        self.player_playlist_mode = False
        self.player_playlist_toggle_btn.config(text="Add Playlist")
        self.player_playlist_rows = []
        self._clear_playlist_table()
        if getattr(self, "player_playlist_frame", None):
            self.player_playlist_frame.destroy()
            self.player_playlist_frame = None
        self.player_library_frame.pack_forget()
        self.player_library_frame.pack(fill="both", expand=True)
        self._update_player_art(None)

    def _render_playlist_rows(self, rows: list[dict[str, str]]) -> None:
        self._clear_playlist_table()
        if not self.player_playlist_tree:
            return
        play_icon = "▶" if self.preview_backend_available else "—"
        for row in rows:
            item = self.player_playlist_tree.insert(
                "",
                "end",
                values=(
                    row.get("title", ""),
                    row.get("artist", ""),
                    play_icon,
                ),
            )
            path = row.get("path", "")
            self.player_playlist_tree_paths[item] = path
            self.player_playlist_tree_rows[item] = row

    def _clear_playlist_table(self) -> None:
        if self.player_playlist_tree:
            for row in self.player_playlist_tree.get_children():
                self.player_playlist_tree.delete(row)
        self.player_playlist_tree_paths.clear()
        self.player_playlist_tree_rows.clear()

    def _on_playlist_tree_click(self, event) -> None:
        if not self.player_playlist_tree:
            return
        region = self.player_playlist_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        col = self.player_playlist_tree.identify_column(event.x)
        if col != "#3":
            return
        item = self.player_playlist_tree.identify_row(event.y)
        if not item:
            return
        path = self.player_playlist_tree_paths.get(item)
        if not path:
            return
        row = self.player_playlist_tree_rows.get(item, {})
        self._update_player_art(path, title=row.get("title"), artist=row.get("artist"))
        self._play_preview(path, duration_ms=30000, player_item=None)

    def _on_playlist_selection_change(self, _event) -> None:
        if not self.player_playlist_tree:
            return
        selection = self.player_playlist_tree.selection()
        if not selection:
            return
        item = selection[0]
        path = self.player_playlist_tree_paths.get(item)
        row = self.player_playlist_tree_rows.get(item, {})
        self._update_player_art(path, title=row.get("title"), artist=row.get("artist"))

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
