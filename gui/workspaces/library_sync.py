"""Library Sync workspace — compare two libraries and transfer files.

Flow: scan both folders → review the matches (right-click to flag Copy /
Replace, add notes) → build the plan (it follows the matches and the flags)
→ preview → execute *that* plan. The plan is invalidated whenever one of its
inputs (paths, transfer mode, flags, scan, library) changes, so Execute can
never run a plan the user is not looking at.
"""
from __future__ import annotations

import copy
import json
import os
import threading
import time
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

_STEPS = ("1. Configure paths", "2. Scan & match", "3. Build & execute plan")


def _chip_colour(kind: str) -> str:
    """Status-chip colour from the active theme ("running" | "ok" | "error" | other)."""
    t = get_manager().current
    return {"running": t.warning, "ok": t.success, "error": t.danger}.get(kind, t.text_muted)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path))) if path else ""


def _validate_paths(existing: str, incoming: str) -> str | None:
    """A user-facing problem with the two folders, or None."""
    if not existing or not incoming:
        return "Please set both the existing library and the incoming folder."
    for label, folder in (("Existing library", existing), ("Incoming folder", incoming)):
        if not os.path.isdir(folder):
            return f"{label} not found:\n{folder}\n\nChoose an existing folder."
    lib, inc = _norm(existing), _norm(incoming)
    if lib == inc:
        return "The incoming folder is the existing library itself."
    if inc.startswith(lib + os.sep):
        return "The incoming folder is inside the existing library. Choose a folder outside it."
    if lib.startswith(inc + os.sep):
        return "The existing library is inside the incoming folder. Choose separate folders."
    return None


class _LogBatcher:
    """Collect worker log lines and emit them in batches (at most every
    ``interval`` seconds), so a scan's per-file events cannot flood the GUI
    thread with thousands of cross-thread signals."""

    def __init__(self, emit, interval: float = 0.25) -> None:
        self._emit = emit
        self._interval = interval
        self._buffer: list[str] = []
        self._last = 0.0

    def add(self, line: str) -> None:
        self._buffer.append(line)
        if time.monotonic() - self._last >= self._interval:
            self.flush()

    def flush(self) -> None:
        if self._buffer:
            self._emit("\n".join(self._buffer))
            self._buffer.clear()
        self._last = time.monotonic()


def _humanize(line: str, counters: Counter) -> str | None:
    """Turn the backend's raw JSON event lines into readable text (or drop them)."""
    if line.startswith("DEBUG:"):
        return None
    if not line.startswith("{"):
        return line
    try:
        data = json.loads(line)
    except ValueError:
        return line
    event = data.get("event") if isinstance(data, dict) else None
    if event in ("cache_hit", "cache_miss"):
        counters[event] += 1
        return None
    if event == "fingerprint_start":
        return f"Fingerprinting {data.get('pending', '?')} file(s) in {data.get('folder', '')}…"
    if event == "skip":
        return f"Skipped unreadable file {data.get('path', '')}: {data.get('reason', '')}"
    if event == "scan_cancelled":
        return "Scan cancelled."
    return None  # fingerprint_progress / fingerprint_settings / other internals


def _scan_summary(counts: dict) -> str:
    parts = [
        f"{counts.get('new', 0)} new",
        f"{counts.get('matched', 0)} already in library",
    ]
    if counts.get("upgrades"):
        parts[-1] += f" ({counts['upgrades']} possible upgrade(s))"
    parts.append(f"{counts.get('low_confidence', 0)} low confidence")
    return f"Scan complete — {counts.get('incoming', 0)} incoming track(s): " + ", ".join(parts) + "."


# ── Workers ───────────────────────────────────────────────────────────────────

class SyncScanWorker(QtCore.QThread):
    """Fingerprint both sides and classify incoming files — or, with
    ``rematch_from``, re-match an earlier scan with new thresholds."""
    progress = Signal(int, int, str)       # current, total (0 = unknown), side
    log_line = Signal(str)
    finished = Signal(bool, str, object)   # success, message, result dict

    def __init__(self, existing: str, incoming: str, thresholds: dict, rematch_from=None) -> None:
        super().__init__()
        self.existing = existing
        self.incoming = incoming
        self.thresholds = thresholds
        self.rematch_from = rematch_from
        self.outcome = ""
        self._cancel_event = threading.Event()
        self._last_progress = 0.0

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        batch = _LogBatcher(self.log_line.emit)
        counters: Counter = Counter()

        def finish(ok: bool, message: str, result) -> None:
            if counters:
                batch.add(
                    f"Fingerprints: {counters.get('cache_hit', 0)} from cache, "
                    f"{counters.get('cache_miss', 0)} recomputed after a file changed."
                )
            batch.flush()
            self.finished.emit(ok, message, result)

        try:
            import library_sync
        except Exception as exc:  # noqa: BLE001
            self.outcome = "error"
            finish(False, f"Import error: {exc}", None)
            return

        existing_root = _norm(self.existing)

        def _log(msg: str) -> None:
            if self._cancel_event.is_set():
                return
            line = _humanize(str(msg), counters)
            if line:
                batch.add(line)

        def _progress(current: int, total: int, path: str, _phase: str) -> None:
            now = time.monotonic()
            if (total and current >= total) or now - self._last_progress >= 0.1:
                self._last_progress = now
                side = "existing" if _norm(str(path)).startswith(existing_root + os.sep) else "incoming"
                self.progress.emit(int(current), int(total or 0), side)

        try:
            if self.rematch_from is not None:
                result = library_sync.rematch_scan_result(
                    self.rematch_from, thresholds=self.thresholds, log_callback=_log
                )
            else:
                docs = Path(self.existing) / "Docs"
                docs.mkdir(exist_ok=True)   # the library itself was validated to exist
                result = library_sync.compare_libraries(
                    self.existing,
                    self.incoming,
                    str(docs / ".library_sync_fp.db"),
                    thresholds=self.thresholds,
                    log_callback=_log,
                    progress_callback=_progress,
                    cancel_event=self._cancel_event,
                    include_match_objects=True,
                )
            if self._cancel_event.is_set() or result.get("partial"):
                self.outcome = "cancelled"
                finish(False, "Scan cancelled.", None)
                return
            counts = library_sync.summarize_matches(result.get("match_objects", []))
            self.outcome = "ok"
            finish(True, _scan_summary(counts), result)
        except Exception as exc:  # noqa: BLE001
            self.outcome = "error"
            finish(False, str(exc) or type(exc).__name__, None)


class SyncBuildWorker(QtCore.QThread):
    """Build a Library Sync plan (from the reviewed matches + flags) and write
    the preview HTML."""
    log_line = Signal(str)
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, library_root: str, incoming_folder: str, transfer_mode: str,
                 review_flags=None, match_results=None) -> None:
        super().__init__()
        self.library_root = library_root
        self.incoming_folder = incoming_folder
        self.transfer_mode = transfer_mode
        # A snapshot: the page's store keeps changing on the GUI thread.
        self.review_flags = copy.deepcopy(review_flags)
        self.match_results = list(match_results) if match_results is not None else None
        self.outcome = ""
        self.inputs: tuple = ()
        self._cancel_event = threading.Event()
        # Populated during run()
        self.plan = None
        self.preview_html_path: str = ""

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        batch = _LogBatcher(self.log_line.emit)

        def finish(ok: bool, message: str) -> None:
            batch.flush()
            self.finished.emit(ok, message)

        try:
            import library_sync
        except Exception as exc:  # noqa: BLE001
            self.outcome = "error"
            finish(False, f"Import error: {exc}")
            return
        try:
            docs = Path(self.library_root) / "Docs"
            docs.mkdir(exist_ok=True)
            output_html = str(docs / "LibrarySyncPreview.html")

            def _progress(current: int, total: int, path: str, phase: str) -> None:
                if total:
                    self.progress.emit(int(current / total * 100))

            plan = library_sync.build_library_sync_preview(
                self.library_root,
                self.incoming_folder,
                output_html,
                log_callback=batch.add,
                progress_callback=_progress,
                cancel_event=self._cancel_event,
                transfer_mode=self.transfer_mode,
                match_results=self.match_results,
                review_flags=self.review_flags,
            )
            self.plan = plan
            self.preview_html_path = output_html
            c = plan.counts()
            self.outcome = "ok"
            finish(
                True,
                f"Plan built — {c['copy'] + c['move']} to add ({plan.transfer_mode}), "
                f"{c['replace']} to replace, {c['skip']} skipped, {c['review']} need review. "
                "Preview written.",
            )
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ in ("IndexCancelled", "Cancelled") or self._cancel_event.is_set():
                self.outcome = "cancelled"
                finish(False, "Plan build cancelled.")
            else:
                self.outcome = "error"
                finish(False, str(exc) or type(exc).__name__)


class SyncExecuteWorker(QtCore.QThread):
    """Execute a Library Sync plan (exactly as previewed)."""
    log_line = Signal(str)
    progress = Signal(int)
    finished = Signal(bool, str, str)   # success, message, report_path

    def __init__(self, plan, create_playlist: bool) -> None:
        super().__init__()
        self.plan = plan
        self.create_playlist = create_playlist
        self.summary: dict | None = None
        self.outcome = ""
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        batch = _LogBatcher(self.log_line.emit)

        def finish(ok: bool, message: str, report: str) -> None:
            batch.flush()
            self.finished.emit(ok, message, report)

        try:
            import library_sync
        except Exception as exc:  # noqa: BLE001
            self.outcome = "error"
            finish(False, f"Import error: {exc}", "")
            return
        try:
            def _progress(current: int, total: int, path: str, phase: str) -> None:
                if total:
                    self.progress.emit(int(current / total * 100))

            summary = library_sync.execute_library_sync_plan(
                self.plan,
                log_callback=batch.add,
                progress_callback=_progress,
                cancel_event=self._cancel_event,
                create_playlist=self.create_playlist,
            )
            self.summary = summary
            # The stable "latest run" copy (same content as this run's own
            # timestamped report, which stays in Docs/ for the record).
            report_path = str(summary.get("latest_report_path") or summary.get("report_path") or "")
            audit_name = os.path.basename(str(summary.get("audit_path") or "LibrarySyncAudit.json"))
            moved = summary.get("moved", 0)
            copied = summary.get("copied", 0)
            replaced = summary.get("replaced", 0)
            skipped = len(summary.get("skipped", []) or [])
            stale = len(summary.get("stale", []) or [])
            errors = list(summary.get("errors", []) or [])
            counts = f"{moved} moved, {copied} copied, {replaced} replaced, {skipped} skipped"
            if summary.get("cancelled"):
                self.outcome = "cancelled"
                finish(False, f"Execution cancelled — {counts} before stopping. "
                              f"See Docs/{audit_name}.", report_path)
            elif errors:
                self.outcome = "partial"
                stale_note = (
                    f" {stale} item(s) were not done because the library changed since the plan "
                    "was built — scan again." if stale else ""
                )
                finish(False, f"Execution finished with {len(errors)} problem(s) — {counts}.{stale_note} "
                              f"First problem: {errors[0]} Full record: Docs/{audit_name}.", report_path)
            else:
                self.outcome = "ok"
                finish(True, f"Execution complete — {counts}.", report_path)
        except Exception as exc:  # noqa: BLE001
            self.outcome = "error"
            finish(False, str(exc) or type(exc).__name__, "")


# ── Workspace ─────────────────────────────────────────────────────────────────

class LibrarySyncWorkspace(WorkspaceBase):
    """Library Sync — scan two folders, build plan, preview and execute."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._scan_result = None
        self._scan_inputs: tuple | None = None    # (existing, incoming) the scan belongs to
        self._worker: SyncScanWorker | None = None
        self._build_worker: SyncBuildWorker | None = None
        self._exec_worker: SyncExecuteWorker | None = None
        self._plan = None
        self._plan_inputs: tuple | None = None
        self._plan_preview_path = ""
        self._step_labels: list[QtWidgets.QLabel] = []
        self._current_step = 0

        # Initialize review state store for per-item user flags
        from library_sync_review_state import ReviewStateStore
        self._review_state_store = ReviewStateStore()
        self._current_match_results: list = []  # Store MatchResult objects from scan
        self._match_by_id: dict = {}

        self._build_ui()
        self._load_session()
        self._refresh_controls()

    def _build_ui(self) -> None:
        cl = self.content_layout

        # ── Header card ───────────────────────────────────────────────────
        header_card = self._make_card()
        header_card.setObjectName("headerCard")
        hl = QtWidgets.QVBoxLayout(header_card)
        hl.setContentsMargins(20, 16, 16, 16)
        hl.setSpacing(8)
        hl.addWidget(self._make_section_title("Library Sync"))
        hl.addWidget(self._make_subtitle(
            "Compare an existing library to an incoming folder. "
            "The sync matcher fingerprints both sides, identifies new tracks, "
            "potential upgrades, and collisions, then builds a copy/move plan."
        ))

        stepper = QtWidgets.QFrame()
        stepper.setObjectName("workflowStepper")
        sl = QtWidgets.QHBoxLayout(stepper)
        sl.setContentsMargins(12, 8, 12, 8)
        sl.setSpacing(4)
        for i, step in enumerate(_STEPS):
            lbl = QtWidgets.QLabel(step)
            lbl.setObjectName("stepActive" if i == 0 else "stepInactive")
            sl.addWidget(lbl)
            self._step_labels.append(lbl)
            if i < len(_STEPS) - 1:
                arr = QtWidgets.QLabel("→")
                arr.setObjectName("stepArrow")
                sl.addWidget(arr)
        sl.addStretch(1)
        hl.addWidget(stepper)
        cl.addWidget(header_card)

        # ── Folders card ───────────────────────────────────────────────────
        folders_card = self._make_card()
        folders_layout = QtWidgets.QFormLayout(folders_card)
        folders_layout.setContentsMargins(16, 16, 16, 16)
        folders_layout.setSpacing(10)
        folders_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self._existing_entry, existing_browse = self._make_browse_row(
            "Existing Library:", "Path to your main library…"
        )
        if self._library_path:
            self._existing_entry.setText(self._library_path)
        existing_row = QtWidgets.QHBoxLayout()
        existing_row.addWidget(self._existing_entry, 1)
        existing_row.addWidget(existing_browse)
        existing_browse.clicked.connect(
            lambda: self._browse_folder(self._existing_entry, "Select Existing Library")
        )
        folders_layout.addRow("Existing Library:", existing_row)

        self._incoming_entry, incoming_browse = self._make_browse_row(
            "Incoming Folder:", "Path to new content to compare…"
        )
        incoming_row = QtWidgets.QHBoxLayout()
        incoming_row.addWidget(self._incoming_entry, 1)
        incoming_row.addWidget(incoming_browse)
        incoming_browse.clicked.connect(
            lambda: self._browse_folder(self._incoming_entry, "Select Incoming Folder")
        )
        folders_layout.addRow("Incoming Folder:", incoming_row)
        self._existing_entry.textChanged.connect(self._on_inputs_changed)
        self._incoming_entry.textChanged.connect(self._on_inputs_changed)
        cl.addWidget(folders_card)

        # ── Config card ────────────────────────────────────────────────────
        cfg_card = self._make_card()
        cfg_layout = QtWidgets.QFormLayout(cfg_card)
        cfg_layout.setContentsMargins(16, 16, 16, 16)
        cfg_layout.setSpacing(10)
        cfg_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self._threshold_entry = QtWidgets.QLineEdit("0.3")
        self._threshold_entry.setToolTip(
            "Default fingerprint distance cutoff (0–1). Lower = stricter match."
        )
        cfg_layout.addRow("Global threshold:", self._threshold_entry)

        self._preset_entry = QtWidgets.QLineEdit()
        self._preset_entry.setPlaceholderText("Optional label for this session")
        self._preset_entry.setEnabled(False)
        self._preset_entry.setToolTip("Not used yet: presets are not implemented.")
        cfg_layout.addRow("Preset name:", self._preset_entry)

        self._overrides_text = QtWidgets.QPlainTextEdit()
        self._overrides_text.setPlaceholderText(
            "Per-format overrides, one per line:\n"
            ".flac=0.3\n.mp3=0.35\n.m4a=0.35"
        )
        self._overrides_text.setFixedHeight(80)
        cfg_layout.addRow("Format overrides:", self._overrides_text)

        btn_row_cfg = QtWidgets.QHBoxLayout()
        self._recompute_btn = QtWidgets.QPushButton("↺  Recompute Matches")
        self._recompute_btn.setEnabled(False)
        self._recompute_btn.setToolTip(
            "Re-match the last scan with the current thresholds, without re-reading "
            "or re-fingerprinting any file."
        )
        self._recompute_btn.clicked.connect(self._on_recompute)
        self._save_session_btn = QtWidgets.QPushButton("💾  Save Session")
        self._save_session_btn.setToolTip(
            "Remember the incoming folder and threshold; they are restored next time."
        )
        self._save_session_btn.clicked.connect(self._on_save_session)
        btn_row_cfg.addWidget(self._recompute_btn)
        btn_row_cfg.addWidget(self._save_session_btn)
        btn_row_cfg.addStretch(1)
        cfg_layout.addRow("", btn_row_cfg)
        cl.addWidget(cfg_card)

        # ── Scan card ──────────────────────────────────────────────────────
        scan_card = self._make_card()
        scan_card.setObjectName("actionCard")
        sc_layout = QtWidgets.QVBoxLayout(scan_card)
        sc_layout.setContentsMargins(16, 14, 16, 14)
        sc_layout.setSpacing(10)
        sc_layout.addWidget(self._make_card_title("Scan"))

        scan_row = QtWidgets.QHBoxLayout()
        self._scan_btn = self._make_primary_button("🔍  Scan Both Libraries")
        self._scan_btn.clicked.connect(self._on_scan)
        self._cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._state_lbl = QtWidgets.QLabel("State: Idle")
        self._state_lbl.setObjectName("statusHint")
        scan_row.addWidget(self._scan_btn)
        scan_row.addWidget(self._cancel_btn)
        scan_row.addStretch(1)
        scan_row.addWidget(self._state_lbl)
        sc_layout.addLayout(scan_row)
        cl.addWidget(scan_card)

        # ── Progress ───────────────────────────────────────────────────────
        prog_card = self._make_card()
        prog_layout = QtWidgets.QVBoxLayout(prog_card)
        prog_layout.setContentsMargins(16, 12, 16, 12)
        prog_layout.setSpacing(8)

        sides = QtWidgets.QHBoxLayout()
        for side_name in ("Existing Library", "Incoming Folder"):
            side_col = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel(side_name)
            lbl.setObjectName("phaseLabel")
            bar = QtWidgets.QProgressBar()
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            status = QtWidgets.QLabel("Idle")
            status.setObjectName("statusHint")
            side_col.addWidget(lbl)
            side_col.addWidget(bar)
            side_col.addWidget(status)
            sides.addLayout(side_col)
            if side_name == "Existing Library":
                self._exist_bar = bar
                self._exist_status = status
            else:
                self._incoming_bar = bar
                self._incoming_status = status
        prog_layout.addLayout(sides)
        cl.addWidget(prog_card)

        # ── Results ────────────────────────────────────────────────────────
        results_card = self._make_card()
        results_layout = QtWidgets.QHBoxLayout(results_card)
        results_layout.setContentsMargins(16, 16, 16, 16)
        results_layout.setSpacing(12)

        # Incoming tracks
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self._make_card_title("Incoming Tracks"))
        self._incoming_table = QtWidgets.QTreeWidget()
        self._incoming_table.setHeaderLabels(["Track", "Status", "Distance", "Flag", "Note"])
        self._incoming_table.setColumnWidth(0, 200)
        self._incoming_table.setColumnWidth(1, 80)
        self._incoming_table.setColumnWidth(2, 70)
        self._incoming_table.setColumnWidth(3, 60)
        self._incoming_table.setColumnWidth(4, 100)
        self._incoming_table.setMinimumHeight(200)
        self._incoming_table.setAlternatingRowColors(True)
        self._incoming_table.currentItemChanged.connect(self._on_incoming_selected)
        # Enable context menu for flagging items
        self._incoming_table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self._incoming_table.customContextMenuRequested.connect(self._on_incoming_context_menu)
        left.addWidget(self._incoming_table)

        # Existing tracks the incoming ones matched
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self._make_card_title("Matched Existing Tracks"))
        self._existing_table = QtWidgets.QTreeWidget()
        self._existing_table.setHeaderLabels(["Track", "Status", "Best Matches"])
        self._existing_table.setColumnWidth(0, 260)
        self._existing_table.setColumnWidth(1, 100)
        self._existing_table.setMinimumHeight(200)
        self._existing_table.setAlternatingRowColors(True)
        right.addWidget(self._existing_table)

        results_layout.addLayout(left, 1)
        results_layout.addLayout(right, 1)
        cl.addWidget(results_card)

        # Inspector
        insp_card = self._make_card()
        insp_layout = QtWidgets.QVBoxLayout(insp_card)
        insp_layout.setContentsMargins(16, 16, 16, 16)
        insp_layout.addWidget(self._make_card_title("Match Inspector"))
        self._inspector = QtWidgets.QPlainTextEdit()
        self._inspector.setReadOnly(True)
        self._inspector.setFixedHeight(120)
        self._inspector.setPlaceholderText("Select a track to see match details…")
        insp_layout.addWidget(self._inspector)
        self._match_summary_lbl = QtWidgets.QLabel("No matches yet.")
        self._match_summary_lbl.setObjectName("statusHint")
        self._match_summary_lbl.setWordWrap(True)
        insp_layout.addWidget(self._match_summary_lbl)
        cl.addWidget(insp_card)

        # ── Plan & Execution ───────────────────────────────────────────────
        plan_card = self._make_card()
        plan_layout = QtWidgets.QVBoxLayout(plan_card)
        plan_layout.setContentsMargins(16, 16, 16, 16)
        plan_layout.setSpacing(10)
        plan_layout.addWidget(self._make_card_title("Plan & Execution"))

        plan_btn_row = QtWidgets.QHBoxLayout()
        self._build_plan_btn = QtWidgets.QPushButton("📋  Build Plan")
        self._build_plan_btn.setEnabled(False)
        self._build_plan_btn.clicked.connect(self._on_build_plan)

        self._preview_plan_btn = QtWidgets.QPushButton("👁  Preview Plan")
        self._preview_plan_btn.setEnabled(False)
        self._preview_plan_btn.clicked.connect(self._on_preview_plan)

        self._transfer_toggle = QtWidgets.QPushButton("Copy Originals")
        self._transfer_toggle.setCheckable(True)
        self._transfer_toggle.setChecked(True)
        self._transfer_toggle.setToolTip(
            "Copy keeps the incoming originals; Move takes them out of the incoming folder. "
            "Changing it requires building the plan again."
        )
        self._transfer_toggle.toggled.connect(self._on_transfer_toggled)

        self._execute_plan_btn = QtWidgets.QPushButton("▶  Execute Plan")
        self._execute_plan_btn.setObjectName("dangerBtn")
        self._execute_plan_btn.setEnabled(False)
        self._execute_plan_btn.clicked.connect(self._on_execute_plan)

        self._export_report_btn = QtWidgets.QPushButton("📤  Export Report")
        self._export_report_btn.setEnabled(False)
        self._export_report_btn.setToolTip(
            "Save a report of all match results, flags, and notes as HTML, JSON, or CSV."
        )
        self._export_report_btn.clicked.connect(self._on_export_report)

        self._output_playlist_cb = QtWidgets.QCheckBox("Write transfer playlist (.m3u8)")

        plan_btn_row.addWidget(self._build_plan_btn)
        plan_btn_row.addWidget(self._preview_plan_btn)
        plan_btn_row.addWidget(self._transfer_toggle)
        plan_btn_row.addWidget(self._execute_plan_btn)
        plan_btn_row.addWidget(self._export_report_btn)
        plan_btn_row.addWidget(self._output_playlist_cb)
        plan_btn_row.addStretch(1)
        plan_layout.addLayout(plan_btn_row)

        self._plan_status_lbl = QtWidgets.QLabel("No plan built.")
        self._plan_status_lbl.setObjectName("statusHint")
        self._plan_status_lbl.setWordWrap(True)
        self._plan_bar = QtWidgets.QProgressBar()
        self._plan_bar.setValue(0)
        self._plan_bar.setTextVisible(False)
        self._plan_bar.setFixedHeight(6)
        plan_layout.addWidget(self._plan_status_lbl)
        plan_layout.addWidget(self._plan_bar)
        cl.addWidget(plan_card)

        # ── Log ────────────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_btn_row = QtWidgets.QHBoxLayout()
        log_btn_row.addWidget(self._make_card_title("Log"))
        log_btn_row.addStretch(1)
        self._export_log_btn = QtWidgets.QPushButton("Export Logs…")
        self._export_log_btn.clicked.connect(self._on_export_log)
        log_btn_row.addWidget(self._export_log_btn)
        log_layout.addLayout(log_btn_row)

        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(130)
        self._log_area.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── State ─────────────────────────────────────────────────────────────

    def _paths(self) -> tuple[str, str]:
        return self._existing_entry.text().strip(), self._incoming_entry.text().strip()

    def _mode(self) -> str:
        return "copy" if self._transfer_toggle.isChecked() else "move"

    def _current_plan_inputs(self) -> tuple:
        existing, incoming = self._paths()
        return (_norm(existing), _norm(incoming), self._mode(),
                self._review_state_store.decision_signature())

    def _scan_fresh(self) -> bool:
        if self._scan_result is None or self._scan_inputs is None:
            return False
        existing, incoming = self._paths()
        return self._scan_inputs == (_norm(existing), _norm(incoming))

    def _plan_fresh(self) -> bool:
        return self._plan is not None and self._plan_inputs == self._current_plan_inputs()

    def _busy(self) -> bool:
        return any(w is not None for w in (self._worker, self._build_worker, self._exec_worker))

    def current_step(self) -> int:
        return self._current_step

    def _set_step(self, index: int) -> None:
        self._current_step = index
        for i, lbl in enumerate(self._step_labels):
            name = "stepActive" if i == index else "stepInactive"
            if lbl.objectName() != name:
                lbl.setObjectName(name)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)

    def _refresh_controls(self) -> None:
        scanning = self._worker is not None
        building = self._build_worker is not None
        executing = self._exec_worker is not None
        busy = scanning or building or executing
        scan_fresh = self._scan_fresh()
        plan_fresh = self._plan_fresh()

        self._scan_btn.setEnabled(not busy)
        self._recompute_btn.setEnabled(not busy and scan_fresh)
        self._build_plan_btn.setEnabled(not busy and scan_fresh)
        self._preview_plan_btn.setEnabled(not busy and plan_fresh)
        self._execute_plan_btn.setEnabled(not busy and plan_fresh)
        self._export_report_btn.setEnabled(not scanning and bool(self._current_match_results))
        self._cancel_btn.setEnabled(busy)
        self._transfer_toggle.setEnabled(not executing)

        if not busy and self._plan is not None and not plan_fresh:
            self._plan_status_lbl.setText(
                "Plan out of date — its paths, transfer mode or flags changed. "
                "Build the plan again before executing."
            )
        if executing or plan_fresh:
            self._set_step(2)
        elif scan_fresh or scanning or building:
            self._set_step(1 if not building else 2)
        else:
            self._set_step(0)

    def _drop_plan(self) -> None:
        self._plan = None
        self._plan_inputs = None
        self._plan_preview_path = ""

    @Slot()
    def _on_inputs_changed(self) -> None:
        self._refresh_controls()
        if self._scan_result is not None and not self._scan_fresh() and self._plan is None and not self._busy():
            self._plan_status_lbl.setText("The folders changed since the scan — scan again to build a plan.")

    @Slot(bool)
    def _on_transfer_toggled(self, checked: bool) -> None:
        self._transfer_toggle.setText("Copy Originals" if checked else "Move Originals")
        self._refresh_controls()

    # ── Thresholds ────────────────────────────────────────────────────────

    def _parse_thresholds(self) -> tuple[dict | None, str | None]:
        """``(thresholds, None)`` or ``(None, error message)``. Extension keys
        are normalised to ``.ext`` (``flac=0.1`` used to never match)."""
        text = self._threshold_entry.text().strip() or "0.3"
        try:
            default = float(text)
        except ValueError:
            return None, f"Global threshold “{text}” is not a number (use e.g. 0.3)."
        if not 0 < default <= 1:
            return None, f"Global threshold {default} must be between 0 and 1."
        thresholds = {"default": default}
        for number, raw in enumerate(self._overrides_text.toPlainText().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                return None, f"Format override line {number} (“{line}”) must look like .flac=0.3"
            ext, value = (part.strip() for part in line.split("=", 1))
            ext = ext.lower()
            if not ext or ext == ".":
                return None, f"Format override line {number} (“{line}”) has no extension."
            if not ext.startswith("."):
                ext = "." + ext
            try:
                val = float(value)
            except ValueError:
                return None, f"Format override line {number}: “{value}” is not a number."
            if not 0 < val <= 1:
                return None, f"Format override line {number}: {val} must be between 0 and 1."
            thresholds[ext] = val
        return thresholds, None

    # ── Session ───────────────────────────────────────────────────────────

    def _load_session(self) -> None:
        """Restore what Save Session stored (it used to be write-only)."""
        try:
            from config import load_config
            cfg = load_config()
        except Exception:  # noqa: BLE001
            return
        if cfg.get("sync_incoming"):
            self._incoming_entry.setText(str(cfg["sync_incoming"]))
        if cfg.get("sync_threshold"):
            self._threshold_entry.setText(str(cfg["sync_threshold"]))
        if not self._library_path and cfg.get("sync_existing"):
            self._existing_entry.setText(str(cfg["sync_existing"]))

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if self._busy():
            return
        existing, incoming = self._paths()
        problem = _validate_paths(existing, incoming)
        if problem:
            QtWidgets.QMessageBox.warning(self, "Check the Folders", problem)
            return
        thresholds, error = self._parse_thresholds()
        if error:
            QtWidgets.QMessageBox.warning(self, "Invalid Threshold", error)
            return
        self._start_scan(SyncScanWorker(existing, incoming, thresholds), "Library Sync scan")

    @Slot()
    def _on_recompute(self) -> None:
        """Re-match the last scan with the current thresholds (no rescan)."""
        if self._busy() or not self._scan_fresh():
            return
        thresholds, error = self._parse_thresholds()
        if error:
            QtWidgets.QMessageBox.warning(self, "Invalid Threshold", error)
            return
        existing, incoming = self._paths()
        self._start_scan(
            SyncScanWorker(existing, incoming, thresholds, rematch_from=self._scan_result),
            "Library Sync re-match",
        )

    def _start_scan(self, worker: SyncScanWorker, name: str) -> None:
        # New match results are coming: any plan built from the old ones is void.
        self._drop_plan()
        self._scan_result = None
        self._scan_inputs = None
        self._current_match_results = []
        self._match_by_id = {}
        self._log_area.clear()
        self._incoming_table.clear()
        self._existing_table.clear()
        self._inspector.clear()
        self._plan_status_lbl.setText("No plan built.")
        self._plan_bar.setValue(0)
        self._state_lbl.setText("State: Scanning…" if worker.rematch_from is None else "State: Re-matching…")
        for bar in (self._exist_bar, self._incoming_bar):
            bar.setValue(0)
        self._exist_status.setText("Scanning…")
        self._incoming_status.setText("Waiting…")
        self._log("Starting library sync scan…", "info")
        self.status_changed.emit("Scanning…", _chip_colour("running"))

        self._worker = worker
        worker.log_line.connect(self._on_log_line)
        worker.progress.connect(self._on_scan_progress)
        worker.finished.connect(self._on_scan_finished)
        self._start_worker(worker, name)
        self._refresh_controls()

    @Slot()
    def _on_cancel(self) -> None:
        labels = []
        for worker, label in ((self._worker, "scan"), (self._build_worker, "plan build"),
                              (self._exec_worker, "execution")):
            if worker is not None and worker.isRunning():
                worker.cancel()
                labels.append(label)
        if labels:
            self._cancel_btn.setEnabled(False)
            self._state_lbl.setText(f"State: Cancelling {', '.join(labels)}…")

    @Slot()
    def _on_save_session(self) -> None:
        try:
            from config import load_config, save_config
            cfg = load_config()
            cfg["sync_existing"] = self._existing_entry.text()
            cfg["sync_incoming"] = self._incoming_entry.text()
            cfg["sync_threshold"] = self._threshold_entry.text()
            save_config(cfg)
            self._log("Session saved to config.", "ok")
        except Exception as exc:  # noqa: BLE001
            self._log(str(exc), "error")

    @Slot()
    def _on_build_plan(self) -> None:
        if self._busy():
            return
        library_root, incoming = self._paths()
        problem = _validate_paths(library_root, incoming)
        if problem:
            QtWidgets.QMessageBox.warning(self, "Check the Folders", problem)
            return
        if not self._scan_fresh():
            QtWidgets.QMessageBox.information(
                self, "Scan First", "Scan both folders (with the current paths) before building a plan."
            )
            return

        transfer_mode = self._mode()
        self._drop_plan()
        self._plan_status_lbl.setText("Building plan + preview…")
        self._plan_bar.setValue(0)
        self._log("Building Library Sync plan and preview…", "info")
        self.status_changed.emit("Building plan…", _chip_colour("running"))

        worker = SyncBuildWorker(
            library_root, incoming, transfer_mode,
            review_flags=self._review_state_store,
            match_results=self._current_match_results,
        )
        worker.inputs = self._current_plan_inputs()
        self._build_worker = worker
        worker.log_line.connect(self._on_log_line)
        worker.progress.connect(self._on_plan_progress)
        worker.finished.connect(self._on_build_finished)
        self._start_worker(worker, "Library Sync plan")
        self._refresh_controls()

    @Slot()
    def _on_preview_plan(self) -> None:
        html = Path(self._plan_preview_path) if self._plan_preview_path else None
        if html is not None and html.exists():
            webbrowser.open(html.resolve().as_uri())
        else:
            self._log("No preview HTML found — click Build Plan first.", "warn")

    @Slot()
    def _on_execute_plan(self) -> None:
        if self._busy():
            return
        if not self._plan_fresh():
            QtWidgets.QMessageBox.warning(
                self, "Plan Out of Date",
                "Build the plan again: its paths, transfer mode or flags changed since it was built.",
            )
            self._refresh_controls()
            return
        if self._library_writes_blocked("Library Sync execute"):
            return
        plan = self._plan
        c = plan.counts()
        mode_line = (
            "COPY — the incoming originals are kept"
            if plan.transfer_mode == "copy" else
            "MOVE — the incoming originals are moved out of the incoming folder"
        )
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Execute",
            f"Apply this plan?\n\nLibrary: {plan.library_root}\nIncoming: {plan.incoming_root}\n"
            f"Mode: {mode_line}\n\n"
            f"• {c['copy'] + c['move']} track(s) added\n"
            f"• {c['replace']} existing track(s) replaced (each backed up to "
            "Docs/Backups/LibrarySync/ first)\n"
            f"• {c['skip']} skipped, {c['review']} left for review (not transferred)\n\n"
            "Existing files are never overwritten except by a Replace you flagged, and an "
            "item whose files changed since the plan was built is skipped and reported. "
            "Every action is recorded in this run's own audit in Docs/ "
            "(LibrarySyncAudit_<date_time>.json; LibrarySyncAudit.json is the latest run).",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        create_playlist = self._output_playlist_cb.isChecked()
        self._plan_status_lbl.setText("Executing…")
        self._plan_bar.setValue(0)
        self.status_changed.emit("Executing…", _chip_colour("running"))

        worker = SyncExecuteWorker(plan, create_playlist)
        self._exec_worker = worker
        worker.log_line.connect(self._on_log_line)
        worker.progress.connect(self._on_plan_progress)
        worker.finished.connect(self._on_execute_finished)
        self._start_worker(worker, "Library Sync execute", library_write=True)
        self._refresh_controls()

    @Slot()
    def _on_export_log(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Log", "", "Text files (*.txt);;All files (*)"
        )
        if path:
            try:
                Path(path).write_text(self._log_area.toPlainText())
                self._log(f"Log exported to {path}", "ok")
            except Exception as exc:  # noqa: BLE001
                self._log(str(exc), "error")

    @Slot()
    def _on_export_report(self) -> None:
        if not self._current_match_results:
            return

        existing_root = self._existing_entry.text().strip()
        docs_dir = str(Path(existing_root) / "Docs") if existing_root else ""

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Review Report",
            docs_dir,
            "HTML Report (*.html);;JSON Report (*.json);;CSV Report (*.csv);;All files (*)",
        )
        if not path:
            return

        try:
            from library_sync_review_report import export_report, export_review_report_html
            from config import FORMAT_PRIORITY, MIXED_CODEC_THRESHOLD_BOOST, DEFAULT_FP_THRESHOLDS

            thresholds, error = self._parse_thresholds()
            if error:
                self._log(f"{error} — the report shows the default thresholds.", "warn")
                thresholds = dict(DEFAULT_FP_THRESHOLDS)

            ext = Path(path).suffix.lower()
            if ext == ".html":
                export_review_report_html(
                    path,
                    self._current_match_results,
                    library_root=existing_root,
                    incoming_root=self._incoming_entry.text().strip(),
                    thresholds=thresholds,
                    mixed_codec_threshold_boost=MIXED_CODEC_THRESHOLD_BOOST,
                    format_priority=FORMAT_PRIORITY,
                    flags=self._review_state_store,
                )
            else:
                fmt = "csv" if ext == ".csv" else "json"
                export_report(
                    path,
                    self._current_match_results,
                    self._review_state_store,
                    thresholds,
                    fmt=fmt,
                )

            self._log(f"Report exported to {path}", "ok")
            if ext == ".html":
                webbrowser.open(Path(path).resolve().as_uri())
        except Exception as exc:  # noqa: BLE001
            self._log(f"Export failed: {exc}", "error")

    @Slot(object, object)
    def _on_incoming_selected(self, current, previous) -> None:  # noqa: ANN001
        if not current:
            return
        track_id = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        match = self._match_by_id.get(track_id)
        if match is None:
            self._inspector.setPlainText(
                f"Track: {current.text(0)}\nStatus: {current.text(1)}\nDistance: {current.text(2)}"
            )
            return
        existing = match.existing.path if match.existing else "—"
        dist = f"{match.distance:.3f}" if match.distance is not None else "—"
        flag = current.text(3) or "none"
        lines = [
            f"Incoming: {match.incoming.path}",
            f"Status: {match.status.name}   Distance: {dist}   Confidence: {match.confidence:.0%}",
            f"Existing match: {existing}",
            f"Quality: {match.quality_label or '—'} (incoming score {match.incoming_score}, "
            f"existing {match.existing_score if match.existing_score is not None else '—'})",
            f"Candidates compared: {len(match.candidates)}   Your flag: {flag}",
        ]
        note = self._review_state_store.note_for(track_id)
        if note:
            lines.append(f"Note: {note}")
        if self._plan_fresh():
            for item in self._plan.items:
                if _norm(item.source) == _norm(match.incoming.path):
                    lines.append(f"Planned: {item.decision.value} ({item.action}) — {item.reason}")
                    break
        self._inspector.setPlainText("\n".join(lines))

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        # A worker abandoned after a library change (or replaced by a newer
        # run) still flushes its last batch of lines: never show those in
        # the current library's log.
        sender = self.sender()
        if sender is not None and sender not in (self._worker, self._build_worker, self._exec_worker):
            return
        self._log_area.appendPlainText(line)

    @Slot(int, int, str)
    def _on_scan_progress(self, current: int, total: int, side: str) -> None:
        if self.sender() is not None and self.sender() is not self._worker:
            return
        if side == "existing":
            bar, status = self._exist_bar, self._exist_status
        else:
            bar, status = self._incoming_bar, self._incoming_status
            if self._exist_bar.value() < 100:
                self._exist_bar.setValue(100)
                self._exist_status.setText("Done")
        if total:
            pct = max(0, min(100, int(current * 100 / total)))
            bar.setValue(pct)
            status.setText(f"Fingerprinting {current}/{total}")
        else:
            status.setText(f"{current} file(s) found…")

    @Slot(int)
    def _on_plan_progress(self, pct: int) -> None:
        sender = self.sender()
        if sender is not None and sender not in (self._build_worker, self._exec_worker):
            return
        self._plan_bar.setValue(max(0, min(100, int(pct))))

    @Slot(bool, str, object)
    def _on_scan_finished(self, success: bool, message: str, result) -> None:  # noqa: ANN001
        worker = self.sender() if isinstance(self.sender(), SyncScanWorker) else self._worker
        if worker is None or worker is not self._worker:
            return  # abandoned after a library change
        self._worker = None

        if success and isinstance(result, dict):
            self._scan_result = result
            self._scan_inputs = (_norm(worker.existing), _norm(worker.incoming))
            self._current_match_results = list(result.get("match_objects", []))
            self._match_by_id = {m.incoming.track_id: m for m in self._current_match_results}
            warnings = self._review_state_store.carry_over(self._current_match_results)
            self._exist_bar.setValue(100)
            self._incoming_bar.setValue(100)
            self._exist_status.setText("Done")
            self._incoming_status.setText("Done")
            self._state_lbl.setText("State: Complete")
            self._log(message, "ok")
            self.status_changed.emit("Scan complete", _chip_colour("ok"))
            self._populate_results(result)
            summary = message
            if warnings:
                for warning in warnings:
                    self._log(warning, "warn")
                    self._log_area.appendPlainText(f"⚠ {warning}")
                summary += f" {len(warnings)} earlier flag(s) or note(s) could not be carried over — see the log."
                QtWidgets.QMessageBox.information(self, "Flags Updated", "\n\n".join(warnings[:10]))
            self._match_summary_lbl.setText(summary)
        elif worker.outcome == "cancelled":
            self._state_lbl.setText("State: Cancelled")
            self._exist_status.setText("Cancelled")
            self._incoming_status.setText("Cancelled")
            self._log(message, "warn")
            self.status_changed.emit("Cancelled", _chip_colour("neutral"))
        else:
            self._state_lbl.setText("State: Error")
            self._exist_status.setText("Error")
            self._incoming_status.setText("Error")
            self._log(message, "error")
            self.status_changed.emit("Error", _chip_colour("error"))
            QtWidgets.QMessageBox.warning(self, "Scan Failed", message)
        self._refresh_controls()

    @Slot(bool, str)
    def _on_build_finished(self, success: bool, message: str) -> None:
        worker = self.sender() if isinstance(self.sender(), SyncBuildWorker) else self._build_worker
        if worker is None or worker is not self._build_worker:
            return
        self._build_worker = None
        self._plan_status_lbl.setText(message)

        if success and worker.plan is not None:
            self._plan = worker.plan
            self._plan_inputs = worker.inputs
            self._plan_preview_path = worker.preview_html_path
            self._log(message, "ok")
            self.status_changed.emit("Plan ready", _chip_colour("ok"))
            self._plan_bar.setValue(100)
            html = Path(worker.preview_html_path) if worker.preview_html_path else None
            if html is not None and html.exists():
                webbrowser.open(html.resolve().as_uri())
        elif worker.outcome == "cancelled":
            self._log(message, "warn")
            self.status_changed.emit("Cancelled", _chip_colour("neutral"))
        else:
            self._log(message, "error")
            self.status_changed.emit("Error", _chip_colour("error"))
            QtWidgets.QMessageBox.warning(self, "Plan Failed", message)
        self._refresh_controls()

    @Slot(bool, str, str)
    def _on_execute_finished(self, success: bool, message: str, report_path: str) -> None:
        worker = self.sender() if isinstance(self.sender(), SyncExecuteWorker) else self._exec_worker
        if worker is None or worker is not self._exec_worker:
            return
        self._exec_worker = None
        self._plan_status_lbl.setText(message)
        # A plan is applied at most once, and the library has changed: the
        # match results are now out of date too.
        self._drop_plan()
        self._scan_inputs = None
        outcome = worker.outcome or ("ok" if success else "error")

        if outcome == "ok":
            self._log(message, "ok")
            self.status_changed.emit("Execution complete", _chip_colour("ok"))
            self._plan_bar.setValue(100)
        elif outcome == "cancelled":
            self._log(message, "warn")
            self.status_changed.emit("Cancelled", _chip_colour("neutral"))
            QtWidgets.QMessageBox.information(self, "Execution Cancelled", message)
        elif outcome == "partial":
            self._log(message, "warn")
            self.status_changed.emit("Needs attention", _chip_colour("running"))
            QtWidgets.QMessageBox.warning(self, "Execution Problems", message)
        else:
            self._log(message, "error")
            self.status_changed.emit("Error", _chip_colour("error"))
            QtWidgets.QMessageBox.critical(self, "Execution Failed", message)
        if report_path and Path(report_path).exists():
            webbrowser.open(Path(report_path).resolve().as_uri())
        self._refresh_controls()
        self._plan_status_lbl.setText(f"{message} Scan again before building another plan.")
        self._set_step(2)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _populate_results(self, result) -> None:  # noqa: ANN001
        """Populate the incoming and matched-existing tables from a scan result."""
        if not isinstance(result, dict):
            return

        self._incoming_table.clear()
        self._existing_table.clear()

        matches = result.get("match_objects")
        if isinstance(matches, list):
            by_existing: dict = defaultdict(list)
            for match in matches:
                path = match.incoming.path
                status_label = match.status.name if hasattr(match.status, "name") else str(match.status)
                dist_str = f"{match.distance:.3f}" if match.distance is not None else "—"
                row = QtWidgets.QTreeWidgetItem([Path(path).name, status_label, dist_str, "", ""])
                row.setToolTip(0, path)
                # Store track_id for context menu and flag lookup
                row.setData(0, QtCore.Qt.ItemDataRole.UserRole, match.incoming.track_id)
                self._incoming_table.addTopLevelItem(row)
                if match.existing is not None and match.status.name != "NEW":
                    by_existing[match.existing.path].append(match)
            for match in matches:
                self._update_incoming_item_flags(match.incoming.track_id)
            for existing_path, matched in sorted(by_existing.items()):
                upgrade = any(m.quality_label == "Potential Upgrade" for m in matched)
                row = QtWidgets.QTreeWidgetItem([
                    Path(existing_path).name,
                    "Upgrade available" if upgrade else "Matched",
                    ", ".join(Path(m.incoming.path).name for m in matched),
                ])
                row.setToolTip(0, existing_path)
                self._existing_table.addTopLevelItem(row)

    def _browse_folder(self, entry: QtWidgets.QLineEdit, title: str) -> None:
        start = entry.text() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, title, start)
        if folder:
            entry.setText(folder)

    def _on_library_changed(self, path: str) -> None:
        # Results of a scan or plan build for the old library are useless:
        # stop them and ignore what they return (the JobRegistry keeps the
        # threads alive until they end). A running execute is left to finish.
        for attr, label in (("_worker", "scan"), ("_build_worker", "plan build")):
            worker = getattr(self, attr)
            if worker is not None:
                worker.cancel()
                setattr(self, attr, None)
                self._log(f"Library changed — the running {label} was cancelled.", "warn")
        if self._exec_worker is not None:
            self._log("Library changed — the running execute continues on its own plan.", "warn")
        else:
            self._log_area.clear()      # the old library's log
        self._existing_entry.setText(path)
        self._refresh_controls()

    def _on_incoming_context_menu(self, pos: QtCore.QPoint) -> None:
        """Right-click context menu for incoming track flagging."""
        item = self._incoming_table.itemAt(pos)
        if not item:
            return

        track_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not track_id:
            return

        menu = QtWidgets.QMenu(self)
        menu.addAction("📋 Copy", lambda: self._flag_for_copy(track_id))
        menu.addAction("↻ Replace", lambda: self._flag_for_replace(track_id))
        menu.addAction("✕ Clear flag", lambda: self._flag_clear(track_id))
        menu.addSeparator()
        menu.addAction("📝 Add note", lambda: self._edit_note(track_id))
        menu.exec(self._incoming_table.mapToGlobal(pos))
        menu.deleteLater()

    def _flag_for_copy(self, track_id: str) -> None:
        """Mark track for copy (always added), removing any replace flag."""
        self._review_state_store.clear_flags(track_id)
        self._review_state_store.flag_for_copy(track_id)
        self._update_incoming_item_flags(track_id)
        self._log(f"Flagged {track_id} for copy", "info")
        self._refresh_controls()

    def _flag_for_replace(self, track_id: str) -> None:
        """Mark track to replace existing counterpart."""
        # Find the matching result for validation
        match = self._match_by_id.get(track_id) or next(
            (r for r in self._current_match_results if r.incoming.track_id == track_id), None
        )
        if not match or not match.existing:
            QtWidgets.QMessageBox.warning(self, "Cannot Replace",
                "No existing match found for this track.")
            return

        self._review_state_store.clear_flags(track_id)
        self._review_state_store.flag_for_replace(match)
        self._update_incoming_item_flags(track_id)
        self._log(f"Flagged {track_id} to replace {match.existing.track_id}", "info")
        self._refresh_controls()

    def _flag_clear(self, track_id: str) -> None:
        """Clear both copy and replace flags."""
        self._review_state_store.clear_flags(track_id)
        self._update_incoming_item_flags(track_id)
        self._log(f"Cleared flags for {track_id}", "info")
        self._refresh_controls()

    def _edit_note(self, track_id: str) -> None:
        """Show dialog to edit note for this track."""
        current_note = self._review_state_store.note_for(track_id) or ""
        text, ok = QtWidgets.QInputDialog.getMultiLineText(
            self, "Add Note", "Notes for this track:", current_note
        )
        if ok:
            self._review_state_store.set_note(track_id, text)
            self._update_incoming_item_flags(track_id)
            self._log(f"Added note to {track_id}", "info")

    def _update_incoming_item_flags(self, track_id: str) -> None:
        """Refresh the display for a single incoming item."""
        # Find item by track_id
        for i in range(self._incoming_table.topLevelItemCount()):
            item = self._incoming_table.topLevelItem(i)
            if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == track_id:
                # Update flag column (index 3)
                if self._review_state_store.is_copy_flagged(track_id):
                    item.setText(3, "📋 Copy")
                elif self._review_state_store.replace_target(track_id):
                    item.setText(3, "↻ Replace")
                else:
                    item.setText(3, "")

                # Update note column (index 4)
                note = self._review_state_store.note_for(track_id)
                item.setText(4, note if note else "")
                break
