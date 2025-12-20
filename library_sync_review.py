"""Review-first Library Sync entry point and session models."""
from __future__ import annotations

import json
import os
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from config import DEFAULT_FP_THRESHOLDS, FORMAT_PRIORITY, load_config, save_config
import library_sync
from indexer_control import IndexCancelled
from library_sync import MatchResult, MatchStatus, TrackRecord
from library_sync_review_report import DEFAULT_REPORT_VERSION
from library_sync_review_state import ReviewStateStore

REVIEW_CONFIG_KEY = "library_sync_review"
REVIEW_FLAG_KEY = "use_library_sync_review"


class ScanState(str, Enum):
    """States tracked during a review scan lifecycle."""

    IDLE = "idle"
    READY = "ready"
    SCANNING_LIBRARY = "scanning_library"
    SCANNING_INCOMING = "scanning_incoming"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass
class ScanConfig:
    """Global and per-format fingerprint thresholds for a scan."""

    global_threshold: float = DEFAULT_FP_THRESHOLDS.get("default", 0.3)
    per_format_overrides: Dict[str, float] = field(default_factory=dict)
    preset_name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_threshold": float(self.global_threshold),
            "per_format_overrides": dict(self.per_format_overrides),
            "preset_name": self.preset_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ScanConfig":
        data = data or {}
        overrides = data.get("per_format_overrides") or {}
        return cls(
            global_threshold=float(data.get("global_threshold", DEFAULT_FP_THRESHOLDS.get("default", 0.3))),
            per_format_overrides={str(k): float(v) for k, v in overrides.items()},
            preset_name=data.get("preset_name"),
        )


@dataclass
class ScanSession:
    """Persisted state for the review-first Library Sync experience."""

    library_root: str = ""
    incoming_root: str = ""
    scan_config: ScanConfig = field(default_factory=ScanConfig)
    scan_state: ScanState = ScanState.IDLE
    exported_report_version: int = DEFAULT_REPORT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "library_root": self.library_root,
            "incoming_root": self.incoming_root,
            "scan_config": self.scan_config.to_dict(),
            "scan_state": self.scan_state.value,
            "exported_report_version": int(self.exported_report_version),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "ScanSession":
        data = data or {}
        state_value = data.get("scan_state") or ScanState.IDLE.value
        try:
            scan_state = ScanState(state_value)
        except ValueError:
            scan_state = ScanState.IDLE
        return cls(
            library_root=data.get("library_root", "") or "",
            incoming_root=data.get("incoming_root", "") or "",
            scan_config=ScanConfig.from_dict(data.get("scan_config")),
            scan_state=scan_state,
            exported_report_version=int(data.get("exported_report_version", DEFAULT_REPORT_VERSION)),
        )


def load_scan_session() -> ScanSession:
    """Load the persisted review scan session from config."""
    cfg = load_config()
    session = ScanSession.from_dict(cfg.get(REVIEW_CONFIG_KEY))
    cfg[REVIEW_CONFIG_KEY] = session.to_dict()
    save_config(cfg)
    return session


def save_scan_session(session: ScanSession) -> None:
    """Persist the review scan session to config."""
    cfg = load_config()
    cfg[REVIEW_CONFIG_KEY] = session.to_dict()
    save_config(cfg)


def is_review_enabled() -> bool:
    """Return whether the review-first entry point is enabled."""
    cfg = load_config()
    return bool(cfg.get(REVIEW_FLAG_KEY, False))


def set_review_enabled(enabled: bool) -> None:
    """Toggle the review-first entry point flag."""
    cfg = load_config()
    cfg[REVIEW_FLAG_KEY] = bool(enabled)
    save_config(cfg)


class ReportPreviewDialog(tk.Toplevel):
    """Simple dialog that summarizes report counts before export."""

    LABELS = {
        "new": "New",
        "collisions": "Collisions / Exact Matches",
        "low_confidence": "Low Confidence",
        "flagged_copy": "Flagged Copy",
        "flagged_replace": "Flagged Replace",
    }

    def __init__(self, master: tk.Misc | None, summary: dict[str, int]):
        super().__init__(master)
        self.title("Report Preview")
        self.resizable(False, False)
        self.result = False
        self.transient(master)
        self.grab_set()

        ttk.Label(self, text="Review counts before saving the export report:").pack(
            anchor="w", padx=10, pady=(10, 5)
        )

        for key, label in self.LABELS.items():
            row = ttk.Frame(self)
            row.pack(fill="x", padx=10, pady=2)
            ttk.Label(row, text=f"{label}:").pack(side="left")
            ttk.Label(row, text=str(summary.get(key, 0)), font=("TkDefaultFont", 10, "bold")).pack(
                side="right"
            )

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=10, pady=(10, 10))
        ttk.Button(btn_row, text="Save Report", command=self._confirm).pack(side="left")
        ttk.Button(btn_row, text="Cancel", command=self._cancel).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self._cancel)

    @classmethod
    def prompt(cls, master: tk.Misc | None, summary: dict[str, int]) -> bool:
        dialog = cls(master, summary)
        if master is not None:
            master.wait_window(dialog)
        else:
            dialog.wait_window()
        return bool(getattr(dialog, "result", False))

    def _confirm(self) -> None:
        self.result = True
        self.destroy()

    def _cancel(self) -> None:
        self.result = False
        self.destroy()


class LibrarySyncReviewPanel(ttk.Frame):
    """Embeddable preview-first Library Sync UI."""

    def __init__(
        self,
        master: tk.Widget | None = None,
        *,
        library_root: str | None = None,
        incoming_root: str | None = None,
        on_close: Callable[[], None] | None = None,
    ):
        super().__init__(master)
        self.session = load_scan_session()
        if library_root:
            self.session.library_root = library_root
        if incoming_root:
            self.session.incoming_root = incoming_root

        self.library_var = tk.StringVar(value=self.session.library_root)
        self.incoming_var = tk.StringVar(value=self.session.incoming_root)
        self.global_threshold_var = tk.StringVar(value=str(self.session.scan_config.global_threshold))
        self.preset_var = tk.StringVar(value=self.session.scan_config.preset_name or "")
        self.report_version_var = tk.StringVar(value=str(self.session.exported_report_version))
        self.scan_state_var = tk.StringVar(value=self._describe_state(self.session.scan_state))
        self._on_close_callback = on_close
        self._selection_syncing = False
        self.library_var.trace_add("write", lambda *_: self._invalidate_plan("Library folder changed"))
        self.incoming_var.trace_add("write", lambda *_: self._invalidate_plan("Incoming folder changed"))

        # Scan + progress state
        self.library_cancel = threading.Event()
        self.incoming_cancel = threading.Event()
        self.progress_vars: dict[str, tk.DoubleVar] = {
            "library": tk.DoubleVar(value=0),
            "incoming": tk.DoubleVar(value=0),
        }
        self.progress_totals: dict[str, int] = {"library": 0, "incoming": 0}
        self.progress_labels: dict[str, tk.StringVar] = {
            "library": tk.StringVar(value="Idle"),
            "incoming": tk.StringVar(value="Idle"),
        }
        self.partial_labels: dict[str, tk.StringVar] = {
            "library": tk.StringVar(value=""),
            "incoming": tk.StringVar(value=""),
        }
        self.phase_labels: dict[str, tk.StringVar] = {
            "library": tk.StringVar(value=""),
            "incoming": tk.StringVar(value=""),
        }
        self.partial_flags: dict[str, bool] = {"library": False, "incoming": False}

        # Review + matching state
        self.state_store = ReviewStateStore()
        self.library_records: list[TrackRecord] = []
        self.incoming_records: list[TrackRecord] = []
        self.match_results: list[MatchResult] = []
        self.match_by_incoming: dict[str, MatchResult] = {}
        self.match_by_existing: dict[str, list[MatchResult]] = {}
        self.library_index: dict[str, TrackRecord] = {}
        self.incoming_index: dict[str, TrackRecord] = {}

        # Logs
        self.log_entries: list[dict[str, object]] = []
        self.LOG_LIMIT = 400

        # Plan + execution state
        self.plan_progress_var = tk.DoubleVar(value=0)
        self.plan_progress_label = tk.StringVar(value="Idle")
        self.plan_status_var = tk.StringVar(value="No plan built.")
        self.plan_cancel = threading.Event()
        self.plan_running = False
        self.plan_preview_path: str | None = None
        self.active_plan: library_sync.LibrarySyncPlan | None = None
        self._plan_sources: Tuple[str, str] | None = None
        self._plan_version = 0
        self._preview_version = -1

        self._build_ui()

    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 5}

        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        root = ttk.Frame(canvas)
        root_id = canvas.create_window((0, 0), window=root, anchor="nw")

        def _on_frame_config(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_config(event):
            canvas.itemconfigure(root_id, width=event.width)

        root.bind("<Configure>", _on_frame_config)
        canvas.bind("<Configure>", _on_canvas_config)

        folder_frame = ttk.LabelFrame(root, text="Folders")
        folder_frame.pack(fill="x", **padding)
        self._add_browse_row(
            folder_frame, "Existing Library:", self.library_var, self._browse_library
        )
        self._add_browse_row(
            folder_frame, "Incoming Folder:", self.incoming_var, self._browse_incoming
        )

        scan_frame = ttk.LabelFrame(root, text="Scan Configuration")
        scan_frame.pack(fill="x", **padding)

        thr_row = ttk.Frame(scan_frame)
        thr_row.pack(fill="x", padx=5, pady=(5, 2))
        ttk.Label(thr_row, text="Global Threshold:").pack(side="left")
        ttk.Entry(thr_row, textvariable=self.global_threshold_var, width=10).pack(
            side="left", padx=(5, 0)
        )

        preset_row = ttk.Frame(scan_frame)
        preset_row.pack(fill="x", padx=5, pady=2)
        ttk.Label(preset_row, text="Preset Name:").pack(side="left")
        ttk.Entry(preset_row, textvariable=self.preset_var, width=20).pack(
            side="left", padx=(5, 0)
        )

        overrides_frame = ttk.Frame(scan_frame)
        overrides_frame.pack(fill="both", expand=True, padx=5, pady=(2, 5))
        ttk.Label(
            overrides_frame,
            text="Per-format overrides (ext=threshold per line):",
        ).pack(anchor="w")
        self.overrides_text = tk.Text(overrides_frame, height=4)
        self.overrides_text.pack(fill="x", expand=True, pady=(2, 0))
        self.overrides_text.insert("1.0", self._format_overrides())

        meta_row = ttk.Frame(scan_frame)
        meta_row.pack(fill="x", padx=5, pady=(2, 5))
        ttk.Label(meta_row, text="Report Version:").pack(side="left")
        ttk.Entry(meta_row, textvariable=self.report_version_var, width=6).pack(side="left", padx=(5, 10))
        ttk.Label(meta_row, text="Scan State:").pack(side="left")
        ttk.Label(meta_row, textvariable=self.scan_state_var).pack(side="left", padx=(5, 0))

        controls = ttk.LabelFrame(root, text="Scan Progress")
        controls.pack(fill="x", **padding)
        controls.columnconfigure((0, 1), weight=1)
        self._build_scan_column(controls, "Existing Library", "library", 0)
        self._build_scan_column(controls, "Incoming Folder", "incoming", 1)

        plan_frame = ttk.LabelFrame(root, text="Plan & Execution")
        plan_frame.pack(fill="x", **padding)
        self._build_plan_controls(plan_frame)

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, **padding)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=2)
        body.rowconfigure(1, weight=1)

        # Two-list UI
        incoming_frame = ttk.LabelFrame(body, text="Incoming Tracks")
        existing_frame = ttk.LabelFrame(body, text="Existing Tracks")
        incoming_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        existing_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self._build_incoming_tree(incoming_frame)
        self._build_existing_tree(existing_frame)

        # Inspector drawer
        inspector = ttk.LabelFrame(body, text="Inspector")
        inspector.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        inspector.columnconfigure((0, 1), weight=1)
        self.inspector_text = tk.Text(inspector, height=8, state="disabled", wrap="word")
        self.inspector_text.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # Log panel
        log_frame = ttk.LabelFrame(root, text="Logs")
        log_frame.pack(fill="both", expand=True, **padding)
        log_frame.columnconfigure(0, weight=1)
        self.log_widget = tk.Text(log_frame, height=8, state="disabled", wrap="word")
        self.log_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        log_btns = ttk.Frame(log_frame)
        log_btns.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        ttk.Button(log_btns, text="Export Logs…", command=self._export_logs).pack(side="left")
        ttk.Button(log_btns, text="Save Session", command=self._persist_session).pack(
            side="left", padx=(5, 0)
        )
        ttk.Button(log_btns, text="Close", command=self._on_close).pack(side="right")

    def _add_browse_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        command,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=5, pady=2)
        ttk.Label(row, text=label).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=(5, 5))
        ttk.Button(row, text="Browse…", command=command).pack(side="left")

    def _build_scan_column(self, parent: ttk.Frame, title: str, kind: str, column: int) -> None:
        col = ttk.Frame(parent)
        col.grid(row=0, column=column, sticky="nsew", padx=5, pady=5)
        ttk.Label(col, text=title).pack(anchor="w")
        ttk.Label(col, textvariable=self.progress_labels[kind]).pack(anchor="w")
        ttk.Progressbar(col, variable=self.progress_vars[kind], maximum=1).pack(
            fill="x", pady=2
        )
        ttk.Label(col, textvariable=self.phase_labels[kind], foreground="#555").pack(anchor="w")
        ttk.Label(col, textvariable=self.partial_labels[kind], foreground="#a64e00").pack(anchor="w")
        btns = ttk.Frame(col)
        btns.pack(fill="x", pady=(4, 0))
        ttk.Button(btns, text="Scan", command=lambda k=kind: self._start_scan(k)).pack(
            side="left"
        )
        ttk.Button(
            btns,
            text="Cancel",
            command=lambda k=kind: self._cancel_scan(k),
        ).pack(side="left", padx=(5, 0))

    def _build_plan_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, textvariable=self.plan_status_var).pack(anchor="w", padx=5, pady=(5, 2))
        ttk.Progressbar(parent, variable=self.plan_progress_var, maximum=1).pack(fill="x", padx=5)
        ttk.Label(parent, textvariable=self.plan_progress_label, foreground="#555").pack(
            anchor="w", padx=5, pady=(0, 5)
        )
        btns = ttk.Frame(parent)
        btns.pack(fill="x", padx=5, pady=(0, 6))
        self.build_plan_btn = ttk.Button(btns, text="Build Plan", command=self._build_plan)
        self.build_plan_btn.pack(side="left")
        self.preview_plan_btn = ttk.Button(btns, text="Preview", command=self._preview_plan)
        self.preview_plan_btn.pack(side="left", padx=(5, 0))
        self.execute_plan_btn = ttk.Button(btns, text="Execute", command=self._execute_plan)
        self.execute_plan_btn.pack(side="left", padx=(5, 0))
        self.open_preview_btn = ttk.Button(
            btns, text="Open Preview", command=self._open_preview_file, state="disabled"
        )
        self.open_preview_btn.pack(side="right")
        self.cancel_plan_btn = ttk.Button(btns, text="Cancel", command=self._cancel_plan_task, state="disabled")
        self.cancel_plan_btn.pack(side="right", padx=(5, 0))
        self._refresh_plan_actions()

    def _build_incoming_tree(self, parent: ttk.Frame) -> None:
        columns = ("name", "chips", "distance")
        tree = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse")
        tree.heading("name", text="Track")
        tree.heading("chips", text="Chips")
        tree.heading("distance", text="Distance")
        tree.column("name", width=200, anchor="w")
        tree.column("chips", width=200, anchor="w")
        tree.column("distance", width=80, anchor="center")
        tree.pack(fill="both", expand=True, padx=5, pady=5)
        tree.bind("<<TreeviewSelect>>", self._on_incoming_select)
        self.incoming_tree = tree

    def _build_existing_tree(self, parent: ttk.Frame) -> None:
        columns = ("name", "chips", "incoming_count")
        tree = ttk.Treeview(parent, columns=columns, show="headings", selectmode="browse")
        tree.heading("name", text="Track")
        tree.heading("chips", text="Chips")
        tree.heading("incoming_count", text="Best Matches")
        tree.column("name", width=200, anchor="w")
        tree.column("chips", width=200, anchor="w")
        tree.column("incoming_count", width=100, anchor="center")
        tree.pack(fill="both", expand=True, padx=5, pady=5)
        tree.bind("<<TreeviewSelect>>", self._on_existing_select)
        self.existing_tree = tree

    def _browse_library(self) -> None:
        folder = filedialog.askdirectory(title="Select Library Root", initialdir=self.library_var.get() or None)
        if folder:
            self.library_var.set(folder)

    def _browse_incoming(self) -> None:
        folder = filedialog.askdirectory(title="Select Incoming Folder", initialdir=self.incoming_var.get() or None)
        if folder:
            self.incoming_var.set(folder)

    def _format_overrides(self) -> str:
        lines = []
        for ext, thr in self.session.scan_config.per_format_overrides.items():
            lines.append(f"{ext}={thr}")
        return "\n".join(lines)

    def _parse_overrides(self) -> Dict[str, float] | None:
        overrides: Dict[str, float] = {}
        raw = self.overrides_text.get("1.0", "end").strip()
        if not raw:
            return overrides
        for line in raw.splitlines():
            if "=" not in line:
                messagebox.showerror("Invalid Override", f"Expected 'ext=threshold' format, got: {line}")
                return None
            ext, value = line.split("=", 1)
            try:
                overrides[ext.strip()] = float(value.strip())
            except ValueError:
                messagebox.showerror("Invalid Threshold", f"Could not parse threshold for {ext.strip()}: {value}")
                return None
        return overrides

    def _persist_session(self) -> bool:
        overrides = self._parse_overrides()
        if overrides is None:
            return False
        try:
            global_thr = float(self.global_threshold_var.get())
        except ValueError:
            messagebox.showerror("Invalid Threshold", "Global threshold must be a number.")
            return False
        try:
            report_ver = int(self.report_version_var.get())
        except ValueError:
            messagebox.showerror("Invalid Report Version", "Exported report version must be an integer.")
            return False

        self.session = ScanSession(
            library_root=self.library_var.get().strip(),
            incoming_root=self.incoming_var.get().strip(),
            scan_config=ScanConfig(
                global_threshold=global_thr,
                per_format_overrides=overrides,
                preset_name=self.preset_var.get().strip() or None,
            ),
            scan_state=self.session.scan_state,
            exported_report_version=report_ver,
        )
        save_scan_session(self.session)
        self.scan_state_var.set(self._describe_state(self.session.scan_state))
        self._log_event(
            "config_change",
            "Updated scan configuration",
            {
                "global_threshold": global_thr,
                "overrides": overrides,
                "preset": self.preset_var.get().strip() or None,
                "report_version": report_ver,
            },
        )
        return True

    def _invalidate_plan(self, reason: str | None = None) -> None:
        """Clear any cached plan/preview when inputs change."""
        self.active_plan = None
        self._plan_sources = None
        self.plan_preview_path = None
        self._plan_version = 0
        self._preview_version = -1
        self.plan_progress_var.set(0)
        self.plan_progress_label.set("Idle")
        if reason:
            self.plan_status_var.set(reason)
        self._refresh_plan_actions()

    def _refresh_plan_actions(self) -> None:
        """Enable/disable plan controls based on current state."""
        running = self.plan_running
        library_root = self.library_var.get().strip()
        incoming_root = self.incoming_var.get().strip()
        sources_match = self._plan_sources == (library_root, incoming_root)
        execute_ready = (
            not running
            and self.active_plan is not None
            and self._preview_version == self._plan_version
            and sources_match
        )

        for btn in (self.build_plan_btn, self.preview_plan_btn):
            btn_state = "disabled" if running else "normal"
            btn.config(state=btn_state)
        self.cancel_plan_btn.config(state="normal" if running else "disabled")
        self.execute_plan_btn.config(state="normal" if execute_ready else "disabled")
        self.open_preview_btn.config(state="normal" if self.plan_preview_path else "disabled")

    def _plan_inputs(self) -> Tuple[str, str] | None:
        """Validate folder selections for planning/execution."""
        library_root = self.library_var.get().strip()
        incoming_root = self.incoming_var.get().strip()
        if not library_root or not incoming_root:
            messagebox.showerror("Missing Folder", "Select both the Existing Library and Incoming Folder first.")
            return None
        return library_root, incoming_root

    def _update_plan_progress(self, current: int, total: int, path: str, phase: str) -> None:
        if total:
            self.plan_progress_var.set(current / total)
        label = f"{phase.title()}: {os.path.basename(path) if path else ''}".strip()
        self.plan_progress_label.set(label or "Working…")

    def _start_plan_task(self, render_preview: bool) -> None:
        inputs = self._plan_inputs()
        if not inputs or self.plan_running:
            return
        library_root, incoming_root = inputs
        self.plan_cancel.clear()
        self.plan_running = True
        self.plan_progress_var.set(0)
        self.plan_progress_label.set("Starting…")
        self.plan_status_var.set("Building preview…" if render_preview else "Building plan…")
        self._refresh_plan_actions()
        thread = threading.Thread(target=self._plan_worker, args=(library_root, incoming_root, render_preview), daemon=True)
        thread.start()

    def _plan_worker(self, library_root: str, incoming_root: str, render_preview: bool) -> None:
        docs_dir = os.path.join(library_root, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        output_html = os.path.join(docs_dir, "LibrarySyncPreview.html")

        def log(msg: str) -> None:
            self._log_event("plan_log", msg)

        def progress(current: int, total: int, path: str, phase: str) -> None:
            self.after(0, lambda c=current, t=total, p=path, ph=phase: self._update_plan_progress(c, t, p, ph))

        try:
            if render_preview:
                plan = library_sync.build_library_sync_preview(
                    library_root,
                    incoming_root,
                    output_html,
                    log_callback=log,
                    progress_callback=progress,
                    cancel_event=self.plan_cancel,
                )
                preview_path = output_html
            else:
                plan = library_sync.compute_library_sync_plan(
                    library_root,
                    incoming_root,
                    log_callback=log,
                    progress_callback=progress,
                    cancel_event=self.plan_cancel,
                )
                preview_path = None
        except IndexCancelled:
            self.after(0, self._on_plan_cancelled)
            return
        except Exception as exc:  # pragma: no cover - UI behavior
            self.after(0, lambda e=exc: self._handle_plan_error(e))
            return

        self.after(
            0,
            lambda p=plan, preview=preview_path, sources=(library_root, incoming_root), render=render_preview: self._on_plan_ready(
                p, preview, sources, render
            ),
        )

    def _on_plan_ready(
        self,
        plan: library_sync.LibrarySyncPlan,
        preview_path: str | None,
        sources: Tuple[str, str],
        render_preview: bool,
    ) -> None:
        self.plan_running = False
        self.plan_cancel.clear()
        self.active_plan = plan
        self._plan_sources = sources
        self._plan_version += 1
        if render_preview:
            self._preview_version = self._plan_version
            self.plan_preview_path = preview_path
            self.plan_status_var.set(f"Preview ready: {preview_path}")
            self._log_event("plan_preview", "Preview ready", {"path": preview_path, "moves": len(plan.moves)})
            if preview_path:
                try:
                    webbrowser.open(preview_path)
                except Exception:
                    pass
        else:
            self.plan_preview_path = None
            self._preview_version = -1
            self.plan_status_var.set("Plan built. Run Preview to inspect before execution.")
            self._log_event("plan_built", "Plan ready", {"moves": len(plan.moves)})

        self.plan_progress_var.set(1)
        self.plan_progress_label.set("Complete")
        self.session.library_root, self.session.incoming_root = sources
        save_scan_session(self.session)
        self._refresh_plan_actions()

    def _handle_plan_error(self, exc: Exception) -> None:
        self.plan_running = False
        self.plan_cancel.clear()
        self.plan_progress_var.set(0)
        self.plan_progress_label.set("Error")
        self.plan_status_var.set(f"Plan failed: {exc}")
        self._log_event("plan_error", str(exc))
        messagebox.showerror("Library Sync", str(exc))
        self._refresh_plan_actions()

    def _on_plan_cancelled(self) -> None:
        self.plan_running = False
        self.plan_cancel.clear()
        self.plan_progress_var.set(0)
        self.plan_progress_label.set("Cancelled")
        self.plan_status_var.set("Plan cancelled.")
        self._log_event("plan_cancelled", "Plan or preview cancelled")
        self._refresh_plan_actions()

    def _cancel_plan_task(self) -> None:
        if self.plan_running:
            self.plan_cancel.set()
            self.plan_progress_label.set("Cancelling…")
            self.plan_status_var.set("Cancellation requested…")
            self._log_event("plan_cancel_requested", "Cancellation requested")
        else:
            self.plan_cancel.clear()

    def _build_plan(self) -> None:
        self._start_plan_task(render_preview=False)

    def _preview_plan(self) -> None:
        self._start_plan_task(render_preview=True)

    def _open_preview_file(self) -> None:
        if not self.plan_preview_path:
            messagebox.showinfo("Preview", "No preview has been generated yet.")
            return
        if not os.path.exists(self.plan_preview_path):
            messagebox.showerror("Preview Missing", "The preview file could not be found. Build a new preview.")
            return
        try:
            webbrowser.open(self.plan_preview_path)
        except Exception as exc:  # pragma: no cover - OS interaction
            messagebox.showerror("Preview", str(exc))

    def _execute_plan(self) -> None:
        if self.plan_running:
            return
        inputs = self._plan_inputs()
        if not inputs:
            return
        if not self.active_plan:
            messagebox.showerror("Library Sync", "Build and preview a plan before executing.")
            return
        if self._plan_sources != inputs:
            messagebox.showerror("Library Sync", "Plan inputs changed. Rebuild and preview the plan before executing.")
            return
        if self._preview_version != self._plan_version:
            messagebox.showerror("Library Sync", "Generate a preview to lock the current plan before execution.")
            return

        self.plan_cancel.clear()
        self.plan_running = True
        self.plan_progress_var.set(0)
        self.plan_progress_label.set("Executing…")
        self.plan_status_var.set("Executing plan…")
        self._refresh_plan_actions()
        thread = threading.Thread(target=self._execute_plan_worker, args=(self.active_plan,), daemon=True)
        thread.start()

    def _execute_plan_worker(self, plan: library_sync.LibrarySyncPlan) -> None:
        def log(msg: str) -> None:
            self._log_event("execute_log", msg)

        def progress(current: int, total: int, path: str, phase: str) -> None:
            self.after(0, lambda c=current, t=total, p=path, ph=phase: self._update_plan_progress(c, t, p, ph))

        try:
            summary = library_sync.execute_library_sync_plan(
                plan,
                log_callback=log,
                progress_callback=progress,
                cancel_event=self.plan_cancel,
            )
        except Exception as exc:  # pragma: no cover - UI behavior
            self.after(0, lambda e=exc: self._handle_plan_error(e))
            return

        self.after(0, lambda s=summary: self._on_execute_complete(s))

    def _on_execute_complete(self, summary: Dict[str, object]) -> None:
        self.plan_running = False
        self.plan_cancel.clear()
        cancelled = bool(summary.get("cancelled"))
        moved = int(summary.get("moved", 0))
        errors = summary.get("errors", []) or []

        if cancelled:
            status = "Execution cancelled."
            self.plan_progress_var.set(0)
            self.plan_progress_label.set("Cancelled")
            self._log_event("execute_cancelled", status)
        else:
            status = f"Execution complete. Moved {moved} files."
            self.plan_progress_var.set(1)
            self.plan_progress_label.set("Complete")
            self._log_event("execute_complete", status, {"moved": moved})

        self.plan_status_var.set(status)
        if errors:
            messagebox.showerror("Execution Errors", "\n".join(errors))
        elif not cancelled:
            messagebox.showinfo("Execution Complete", status)
        self._refresh_plan_actions()

    def _describe_state(self, state: ScanState) -> str:
        return state.value.replace("_", " ").title()

    def preview_report(self, summary: dict[str, int]) -> bool:
        """Open a modal preview dialog to confirm report export counts."""
        return ReportPreviewDialog.prompt(self, summary)

    def _on_close(self) -> None:
        if self._persist_session() and self._on_close_callback:
            self._on_close_callback()

    def set_folders(self, library_root: str | None = None, incoming_root: str | None = None) -> None:
        """Update the selected folders, persisting them to the scan session."""
        changed = False
        if library_root:
            self.library_var.set(library_root)
            self.session.library_root = library_root
            changed = True
        if incoming_root:
            self.incoming_var.set(incoming_root)
            self.session.incoming_root = incoming_root
            changed = True
        if changed:
            save_scan_session(self.session)

    # ── Scan + Progress -------------------------------------------------
    def _cancel_event_for(self, kind: str) -> threading.Event:
        return self.library_cancel if kind == "library" else self.incoming_cancel

    def _start_scan(self, kind: str) -> None:
        if not self._persist_session():
            return
        folder = self.library_var.get().strip() if kind == "library" else self.incoming_var.get().strip()
        if not folder:
            messagebox.showerror("Missing Folder", f"Please choose a {kind} folder before scanning.")
            return
        cancel_event = self._cancel_event_for(kind)
        cancel_event.clear()
        self.progress_vars[kind].set(0)
        self.progress_totals[kind] = 0
        self.progress_labels[kind].set("Scanning…")
        self.phase_labels[kind].set("")
        self.partial_labels[kind].set("")
        self.partial_flags[kind] = False
        self.session.scan_state = ScanState.SCANNING_LIBRARY if kind == "library" else ScanState.SCANNING_INCOMING
        self.scan_state_var.set(self._describe_state(self.session.scan_state))
        self._log_event("scan_start", f"Scanning {folder}", {"kind": kind, "folder": folder})
        thread = threading.Thread(target=self._scan_worker, args=(kind, folder, cancel_event), daemon=True)
        thread.start()

    def _cancel_scan(self, kind: str) -> None:
        cancel_event = self._cancel_event_for(kind)
        cancel_event.set()
        self.partial_labels[kind].set("Cancelling… Partial results will be kept.")
        self._log_event("scan_cancel", f"Cancellation requested for {kind}", {"kind": kind})

    def _scan_worker(self, kind: str, folder: str, cancel_event: threading.Event) -> None:
        db_root = self.library_var.get().strip() or folder
        docs_dir = os.path.join(db_root, "Docs")
        os.makedirs(docs_dir, exist_ok=True)
        db_path = os.path.join(docs_dir, ".soundvault.db")

        progress_cb = self._make_progress_handler(kind)
        try:
            records = library_sync._scan_folder(
                folder,
                db_path,
                log_callback=lambda msg: self._log_scan_message(kind, msg),
                progress_callback=progress_cb,
                cancel_event=cancel_event,
            )
        except Exception as exc:  # pragma: no cover - UI only
            self._log_event("scan_error", str(exc), {"kind": kind})
            self.after(0, lambda e=exc: messagebox.showerror("Scan Failed", str(e)))
            return

        self.after(0, lambda recs=records: self._on_scan_complete(kind, recs))

    def _make_progress_handler(self, kind: str):
        def handler(current: int, total: int, path: str, phase: str) -> None:
            self.after(0, lambda c=current, t=total, p=path, ph=phase: self._update_progress(kind, c, t, p, ph))

        return handler

    def _update_progress(self, kind: str, current: int, total: int, path: str, phase: str) -> None:
        if total:
            self.progress_totals[kind] = total
            self.progress_vars[kind].set(current / total if total else 0)
        label = f"{phase.title()}: {os.path.basename(path) if path else ''}".strip()
        self.phase_labels[kind].set(label)

    def _on_scan_complete(self, kind: str, records: Sequence[TrackRecord]) -> None:
        cancel_event = self._cancel_event_for(kind)
        partial = cancel_event.is_set()
        self.partial_flags[kind] = partial
        status = "Partial" if partial else "Complete"
        self.progress_labels[kind].set(status)
        self.partial_labels[kind].set("Partial" if partial else "")
        self.phase_labels[kind].set("")
        if kind == "library":
            self.library_records = list(records)
            self.library_index = {r.track_id: r for r in records}
        else:
            self.incoming_records = list(records)
            self.incoming_index = {r.track_id: r for r in records}

        self._log_event(
            "scan_complete",
            f"Finished {kind} scan ({status})",
            {"kind": kind, "count": len(records), "partial": partial},
        )
        if partial:
            self.session.scan_state = ScanState.CANCELLED
        elif self.library_records and self.incoming_records:
            self.session.scan_state = ScanState.COMPLETE
        else:
            self.session.scan_state = ScanState.READY
        self.scan_state_var.set(self._describe_state(self.session.scan_state))
        save_scan_session(self.session)
        self._maybe_match()

    # ── Matching + Display ----------------------------------------------
    def _maybe_match(self) -> None:
        if not self.library_records or not self.incoming_records:
            return
        try:
            overrides = self._parse_overrides() or {}
            global_thr = float(self.global_threshold_var.get())
        except Exception:
            overrides = self._parse_overrides() or {}
            global_thr = DEFAULT_FP_THRESHOLDS.get("default", 0.3)
        thresholds = {"default": global_thr, **overrides}
        self._log_event("match_start", "Computing matches", {"thresholds": thresholds})
        results = library_sync._match_tracks(
            self.incoming_records,
            self.library_records,
            thresholds,
            FORMAT_PRIORITY,
            log_callback=lambda msg: self._log_event("match_log", str(msg)),
        )
        self.match_results = results
        self.match_by_incoming = {res.incoming.track_id: res for res in results}
        match_by_existing: dict[str, list[MatchResult]] = {}
        for res in results:
            if res.existing:
                match_by_existing.setdefault(res.existing.track_id, []).append(res)
        self.match_by_existing = match_by_existing
        self._log_event("match_stats", "Updated matching statistics", self._summarize_matches(results))
        self._render_lists()
        self._update_inspector()

    def _render_lists(self) -> None:
        for tree in (self.incoming_tree, self.existing_tree):
            for iid in tree.get_children():
                tree.delete(iid)

        for res in self.match_results:
            chips = ", ".join(self._chips_for_incoming(res))
            distance = "-" if res.distance is None else f"{res.distance:.3f}"
            name = os.path.basename(res.incoming.path)
            self.incoming_tree.insert(
                "",
                "end",
                iid=res.incoming.track_id,
                values=(name, chips, distance),
            )

        for rec in self.library_records:
            linked = self.match_by_existing.get(rec.track_id, [])
            chips = ", ".join(self._chips_for_existing(linked))
            name = os.path.basename(rec.path)
            self.existing_tree.insert(
                "",
                "end",
                iid=rec.track_id,
                values=(name, chips, len(linked)),
            )

    def _chips_for_incoming(self, res: MatchResult) -> List[str]:
        chips: List[str] = []
        status_map = {
            MatchStatus.NEW: "New",
            MatchStatus.COLLISION: "Collision",
            MatchStatus.EXACT_MATCH: "Exact Match",
            MatchStatus.LOW_CONFIDENCE: "Low Confidence",
        }
        chips.append(status_map.get(res.status, res.status.value.title()))
        if res.quality_label:
            chips.append(res.quality_label)
        if not res.incoming.tags:
            chips.append("Missing Metadata")
        if self.partial_flags.get("incoming") or self.partial_flags.get("library"):
            chips.append("Partial")
        return chips

    def _chips_for_existing(self, linked_matches: Iterable[MatchResult]) -> List[str]:
        chips: List[str] = []
        linked = list(linked_matches)
        if linked:
            chips.append("Best Match")
            # surface aggregate quality direction
            if any(res.quality_label == "Potential Upgrade" for res in linked):
                chips.append("Potential Upgrade")
            elif any(res.quality_label == "Keep Existing" for res in linked):
                chips.append("Keep Existing")
        if self.partial_flags.get("library"):
            chips.append("Partial")
        return chips or ["Unmatched"]

    def _on_incoming_select(self, _event=None) -> None:
        sel = self.incoming_tree.selection()
        if not sel:
            return
        if self._selection_syncing:
            return
        self._selection_syncing = True
        incoming_id = sel[0]
        match = self.match_by_incoming.get(incoming_id)
        if match and match.existing:
            self.existing_tree.selection_set(match.existing.track_id)
            self.existing_tree.see(match.existing.track_id)
        self._selection_syncing = False
        self._update_inspector(incoming_id=incoming_id)

    def _on_existing_select(self, _event=None) -> None:
        sel = self.existing_tree.selection()
        if not sel:
            return
        if self._selection_syncing:
            return
        self._selection_syncing = True
        existing_id = sel[0]
        linked = self.match_by_existing.get(existing_id, [])
        if linked:
            target = linked[0].incoming.track_id
            self.incoming_tree.selection_set(target)
            self.incoming_tree.see(target)
            self._update_inspector(incoming_id=target)
        else:
            self._update_inspector(existing_id=existing_id)
        self._selection_syncing = False

    def _update_inspector(self, incoming_id: str | None = None, existing_id: str | None = None) -> None:
        res: MatchResult | None = None
        if incoming_id:
            res = self.match_by_incoming.get(incoming_id)
        elif existing_id:
            linked = self.match_by_existing.get(existing_id, [])
            res = linked[0] if linked else None

        lines: List[str] = []
        if res:
            inc = res.incoming
            lines.append("Incoming Track")
            lines.append(f" • Path: {inc.path}")
            lines.append(f" • Status: {', '.join(self._chips_for_incoming(res))}")
            lines.append(f" • Ext: {inc.ext} | Bitrate: {inc.bitrate or 'n/a'} | Duration: {inc.duration or 'n/a'}")
            lines.append(f" • Fingerprint Distance: {res.distance if res.distance is not None else 'n/a'}")
            lines.append(f" • Threshold Used: {res.threshold_used}")
            lines.append(f" • Confidence: {res.confidence:.2f}")
            if res.existing:
                lines.append("")
                lines.append("Best Match")
                lines.append(f" • Path: {res.existing.path}")
                lines.append(f" • Quality: {res.quality_label or 'n/a'}")
                lines.append(f" • Match Distance: {res.distance if res.distance is not None else 'n/a'}")
            else:
                lines.append("")
                lines.append("Best Match")
                lines.append(" • None")
        elif existing_id:
            rec = self.library_index.get(existing_id)
            if rec:
                lines.append("Existing Track")
                lines.append(f" • Path: {rec.path}")
                lines.append(f" • Ext: {rec.ext} | Bitrate: {rec.bitrate or 'n/a'} | Duration: {rec.duration or 'n/a'}")
                lines.append(" • No incoming matches yet.")

        self.inspector_text.configure(state="normal")
        self.inspector_text.delete("1.0", "end")
        self.inspector_text.insert("1.0", "\n".join(lines) if lines else "Select a row to inspect details.")
        self.inspector_text.configure(state="disabled")

    def _summarize_matches(self, results: Iterable[MatchResult]) -> Dict[str, int]:
        res_list = list(results)
        counts = {"new": 0, "collision": 0, "exact": 0, "low_confidence": 0}
        for res in res_list:
            if res.status == MatchStatus.NEW:
                counts["new"] += 1
            elif res.status == MatchStatus.COLLISION:
                counts["collision"] += 1
            elif res.status == MatchStatus.EXACT_MATCH:
                counts["exact"] += 1
            elif res.status == MatchStatus.LOW_CONFIDENCE:
                counts["low_confidence"] += 1
        counts["total"] = len(res_list)
        return counts

    # ── Logging ---------------------------------------------------------
    def _log_event(self, event_type: str, message: str, data: Dict[str, object] | None = None) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "message": message,
        }
        if data:
            entry["data"] = data
        self.log_entries.append(entry)
        if len(self.log_entries) > self.LOG_LIMIT:
            self.log_entries = self.log_entries[-self.LOG_LIMIT :]
        self._append_log(entry)

    def _append_log(self, entry: Dict[str, object]) -> None:
        self.log_widget.configure(state="normal")
        line = f"[{entry['timestamp']}] {entry['type']}: {entry['message']}"
        if "data" in entry:
            line += f" {entry['data']}"
        self.log_widget.insert("end", line + "\n")
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _export_logs(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Export Logs",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.log_entries, f, indent=2)
        self._log_event("export_logs", f"Saved logs to {path}")

    def _log_scan_message(self, kind: str, msg: object) -> None:
        if isinstance(msg, str):
            try:
                data = json.loads(msg)
                self._log_event(f"{kind}_scan", data.get("event", "scan"), data)
            except Exception:
                self._log_event(f"{kind}_scan", msg)
        elif isinstance(msg, dict):
            self._log_event(f"{kind}_scan", msg.get("event", "scan"), msg)
        else:
            self._log_event(f"{kind}_scan", str(msg))


class LibrarySyncReviewWindow(tk.Toplevel):
    """Toplevel wrapper that hosts the embeddable review panel."""

    def __init__(self, master: tk.Widget | None = None):
        super().__init__(master)
        self.title("Library Sync (Review)")
        self.resizable(True, True)
        self.panel = LibrarySyncReviewPanel(self, on_close=self._on_close)
        self.panel.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        if self.panel._persist_session():
            self.destroy()
