"""Duplicate Finder workspace — scan → preview → execute."""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase
from utils.reserved_folders import iter_library_files

_AUDIO_EXTS = {'.flac', '.m4a', '.aac', '.mp3', '.wav', '.ogg', '.opus'}

# How many individual fingerprint failures to log before summarising.
_MAX_LOGGED_FAILURES = 20

# Mirrors ExecutionConfig.operation_limit's default in the executor.
_EXECUTOR_OPERATION_LIMIT = 20_000

_DISPOSITION_CHOICES: list[tuple[str, str | None]] = [
    ("Default (global choice)", None),
    ("Keep loser in place (retain)", "retain"),
    ("Quarantine", "quarantine"),
    ("Delete permanently", "delete"),
]

_DELETE_CONFIRM_WORD = "DELETE"


def _missing_fingerprint_tools() -> list[str]:
    """External tools the scan needs (fpcalc + ffmpeg) that are not on PATH."""
    import chromaprint_utils
    return chromaprint_utils.missing_tools()


def _config_thresholds() -> tuple[float, float, float]:
    """(exact, near, mixed-codec boost) as saved by Settings / config defaults.

    Keys are the ones the Settings dialog writes (``exact_duplicate_threshold``,
    ``near_duplicate_threshold``) plus ``mixed_codec_threshold_boost``.
    """
    from config import (
        EXACT_DUPLICATE_THRESHOLD,
        MIXED_CODEC_THRESHOLD_BOOST,
        NEAR_DUPLICATE_THRESHOLD,
        load_config,
    )

    cfg = load_config()

    def _num(key: str, default: float) -> float:
        try:
            return float(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    return (
        _num("exact_duplicate_threshold", EXACT_DUPLICATE_THRESHOLD),
        _num("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD),
        _num("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST),
    )


def _gather_tracks_for_plan(
    library_path: str,
    db_path: str,
    log_callback,
    progress_callback,
    cancel_event: threading.Event,
    *,
    compute_missing: bool = True,
) -> list[dict]:
    """Walk ``library_path``, fingerprint files, and return a list of track dicts.

    Fingerprints come from ``db_path`` when cached; otherwise they are computed
    with ``chromaprint_utils.fingerprint_fpcalc`` (unless ``compute_missing``
    is False, e.g. because fpcalc is not installed) and stored back.
    """
    from fingerprint_cache import (
        ensure_fingerprint_cache,
        flush_fingerprint_writes,
        get_cached_fingerprint_metadata,
        store_fingerprint,
    )
    import chromaprint_utils
    from utils.audio_metadata_reader import read_tags

    ensure_fingerprint_cache(db_path)

    audio_paths = sorted(str(p) for p in iter_library_files(library_path, _AUDIO_EXTS))

    total = len(audio_paths)
    tracks: list[dict] = []
    failures = 0
    skipped_uncached = 0
    computed = 0

    def _failure(message: str) -> None:
        nonlocal failures
        failures += 1
        if failures <= _MAX_LOGGED_FAILURES:
            log_callback(message)

    for i, path in enumerate(audio_paths):
        if cancel_event.is_set():
            break
        pct = int((i + 1) / total * 60)

        fp = None
        cached_meta: dict | None = None
        try:
            fp, cached_meta = get_cached_fingerprint_metadata(path, db_path)
        except Exception:
            pass

        if not fp:
            if not compute_missing:
                skipped_uncached += 1
                progress_callback(pct)
                continue
            try:
                # Returns the fingerprint string (no duration).
                fp = chromaprint_utils.fingerprint_fpcalc(path)
            except chromaprint_utils.FingerprintError as exc:
                if str(exc).startswith("Required tool"):
                    # fpcalc/ffmpeg vanished mid-scan: say so once, not per file.
                    compute_missing = False
                    skipped_uncached += 1
                    log_callback(
                        f"! {exc}. Remaining files can only use fingerprints already in the cache."
                    )
                else:
                    _failure(f"! Fingerprint failed: {Path(path).name}: {exc}")
                progress_callback(pct)
                continue
            except Exception as exc:  # noqa: BLE001
                _failure(f"! Fingerprint failed: {Path(path).name}: {exc}")
                progress_callback(pct)
                continue
            if not fp:
                _failure(f"! No fingerprint (too short or silent?): {Path(path).name}")
                progress_callback(pct)
                continue
            computed += 1
            # Store newly computed fingerprint
            try:
                tags: dict = {}
                try:
                    tags = dict(read_tags(path) or {})
                except Exception:
                    pass
                ext_val = os.path.splitext(path)[1].lower()
                store_fingerprint(path, db_path, None, fp, ext=ext_val, tags=tags)
                cached_meta = {'ext': ext_val, 'tags': tags}
            except Exception:
                pass

        meta = cached_meta or {}
        ext_val = meta.get('ext') or os.path.splitext(path)[1].lower()
        if isinstance(ext_val, str) and ext_val and not ext_val.startswith('.'):
            ext_val = f'.{ext_val}'

        tracks.append({
            'path': path,
            'fingerprint': fp,
            'ext': ext_val,
            'bitrate': int(meta.get('bitrate') or 0),
            'sample_rate': int(meta.get('sample_rate') or 0),
            'bit_depth': int(meta.get('bit_depth') or 0),
            'channels': int(meta.get('channels') or 0),
            'codec': str(meta.get('codec') or ''),
            'container': str(meta.get('container') or ''),
            'tags': meta.get('tags') if isinstance(meta.get('tags'), dict) else {},
            'artwork': meta.get('artwork') if isinstance(meta.get('artwork'), list) else [],
            # Keeps the plan builder's metadata cache in <library>/Docs rather
            # than in <common folder of the tracks>/Docs (an album folder).
            'discovery': {'scan_roots': [library_path]},
        })
        progress_callback(pct)

    if failures > _MAX_LOGGED_FAILURES:
        log_callback(f"… {failures - _MAX_LOGGED_FAILURES} more fingerprint failure(s) not shown.")
    if skipped_uncached:
        log_callback(
            f"{skipped_uncached} file(s) skipped: not in the fingerprint cache and fpcalc is unavailable."
        )
    if computed:
        try:
            flush_fingerprint_writes(db_path, timeout=10)
        except Exception:
            pass
    return tracks


def summarize_execution(result) -> tuple[bool, str]:
    """(success, one-line message) describing an ``ExecutionResult`` honestly.

    Preflight refusals used to surface only as "Execution completed with
    errors." — their reason lived in the report JSON.
    """
    actions = list(getattr(result, "actions", []) or [])
    blocked = [a for a in actions if a.step == "preflight" and a.status in ("blocked", "failed")]
    if blocked:
        details: list[str] = []
        for action in blocked:
            text = action.detail
            if action.target not in ("plan", "execution"):
                text = f"{os.path.basename(action.target)}: {action.detail}"
            if text not in details:
                details.append(text)
        more = f" (+{len(details) - 3} more)" if len(details) > 3 else ""
        return False, "Execution blocked — nothing was changed. " + "; ".join(details[:3]) + more

    cleanup = [a for a in actions if a.step == "loser_cleanup"]
    quarantined = sum(
        1 for a in cleanup if a.status == "success" and a.metadata.get("disposition") == "quarantine"
    )
    deleted = sum(1 for a in cleanup if a.status == "success" and a.metadata.get("disposition") == "delete")
    retained = sum(1 for a in cleanup if a.metadata.get("skip_reason") == "retain")
    counts = f"{quarantined} quarantined, {deleted} deleted, {retained} kept in place"

    if any(a.status == "cancelled" for a in actions):
        return False, f"Execution cancelled at a safe point ({counts} before stopping). See the report."
    failures = [a for a in actions if a.status == "failed"]
    if failures or not getattr(result, "success", False):
        first = failures[0].detail if failures else "see the report"
        return False, (
            f"Execution finished with {len(failures)} error(s): {first} ({counts}). "
            "Some changes may have been applied — see the report."
        )
    return True, f"Execution complete: {counts}."


def _latest_execution_report(library_path: str) -> str:
    if not library_path:
        return ""
    root = Path(library_path) / "Docs" / "duplicate_execution_reports"
    if not root.is_dir():
        return ""
    try:
        reports = sorted(root.rglob("execution_report.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return ""
    return str(reports[0]) if reports else ""


# ── Workers ───────────────────────────────────────────────────────────────────

class DupeBuildWorker(QtCore.QThread):
    """Fingerprint library, build consolidation plan, and write the previews."""
    progress = Signal(int)
    log_line = Signal(str)
    groups_ready = Signal(list)     # list of GroupPlan
    done = Signal(bool, str)

    def __init__(
        self,
        library_path: str,
        exact_threshold: float,
        near_threshold: float,
        codec_boost: float,
        compute_missing: bool = True,
    ) -> None:
        super().__init__()
        self.library_path = library_path
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
        self.codec_boost = codec_boost
        self.compute_missing = compute_missing
        self._cancel_event = threading.Event()
        # Populated during run(); read by workspace after done
        self.plan = None
        self.preview_html_path: str = ""

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            from duplicate_consolidation import (
                build_consolidation_plan,
                export_consolidation_preview,
                export_consolidation_preview_html,
            )
        except ImportError as exc:
            self.done.emit(False, f"Import error: {exc}")
            return

        try:
            def _log(msg: str) -> None:
                self.log_line.emit(msg)

            docs = Path(self.library_path) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            db_path = str(docs / ".duplicate_fingerprints.db")

            _log("Stage 1: fingerprinting library…")
            tracks = _gather_tracks_for_plan(
                self.library_path,
                db_path,
                log_callback=_log,
                progress_callback=lambda pct: self.progress.emit(pct),
                cancel_event=self._cancel_event,
                compute_missing=self.compute_missing,
            )

            if self._cancel_event.is_set():
                self.done.emit(False, "Cancelled.")
                return

            if not tracks:
                self.done.emit(False, "No fingerprinted tracks found.")
                return

            _log(f"Stage 2: building consolidation plan for {len(tracks)} tracks…")
            self.progress.emit(62)
            last_pct = [62]

            def _plan_progress(current: int, total: int, _msg: str) -> None:
                pct = 62 + int(23 * current / total) if total else 62
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    self.progress.emit(pct)

            plan = build_consolidation_plan(
                tracks,
                exact_duplicate_threshold=self.exact_threshold,
                near_duplicate_threshold=self.near_threshold,
                mixed_codec_threshold_boost=self.codec_boost,
                cancel_event=self._cancel_event,
                progress_callback=_plan_progress,
                log_callback=_log,
            )
            if self._cancel_event.is_set():
                self.done.emit(False, "Cancelled.")
                return
            self.plan = plan
            self.progress.emit(85)

            _log("Stage 3: writing preview…")
            html_path = str(docs / "duplicate_preview.html")
            try:
                export_consolidation_preview_html(plan, html_path)
                self.preview_html_path = html_path
                _log(f"Preview written → {html_path}")
            except Exception as exc:
                _log(f"Warning: could not write HTML preview: {exc}")
            # The JSON plan is what Tools → Pair Review reads.
            try:
                export_consolidation_preview(plan, str(docs / "duplicate_preview.json"))
            except Exception as exc:
                _log(f"Warning: could not write JSON preview: {exc}")

            self.progress.emit(100)
            self.groups_ready.emit(plan.groups)
            self.done.emit(True, f"Found {len(plan.groups)} duplicate group(s).")

        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


class DupeExecuteWorker(QtCore.QThread):
    """Execute a consolidation plan (dispositions are already on the plan)."""
    log_line = Signal(str)
    done = Signal(bool, str, str)   # success, message, report_html_path

    def __init__(
        self,
        plan,
        library_path: str,
        *,
        allow_review_required: bool = False,
        allow_deletion: bool = False,
        confirm_operation_overage: bool = False,
    ) -> None:
        super().__init__()
        self.plan = plan
        self.library_path = library_path
        self.allow_review_required = allow_review_required
        self.allow_deletion = allow_deletion
        self.confirm_operation_overage = confirm_operation_overage
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            from duplicate_consolidation_executor import ExecutionConfig, execute_consolidation_plan
        except ImportError as exc:
            self.done.emit(False, f"Import error: {exc}", "")
            return
        try:
            docs = Path(self.library_path) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            reports_dir = str(docs / "duplicate_execution_reports")

            config = ExecutionConfig(
                library_root=self.library_path,
                reports_dir=reports_dir,
                quarantine_dir=str(Path(self.library_path) / "Quarantine"),
                quarantine_flatten=True,
                retain_losers=False,
                allow_review_required=self.allow_review_required,
                # The executor's deletion double-gate: both flags, set only
                # after the user typed the confirmation word.
                allow_deletion=self.allow_deletion,
                confirm_deletion=self.allow_deletion,
                confirm_operation_overage=self.confirm_operation_overage,
                cancel_event=self._cancel_event,
                log_callback=lambda msg: self.log_line.emit(msg),
            )

            result = execute_consolidation_plan(self.plan, config)
            report_html = result.report_paths.get("html_report", "")
            ok, message = summarize_execution(result)
            self.done.emit(ok, message, str(report_html or ""))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc), "")


# ── Workspace ─────────────────────────────────────────────────────────────────

class DuplicatesWorkspace(WorkspaceBase):
    """Duplicate Finder — scan, review groups, preview and execute."""

    _STEP_SCAN, _STEP_REVIEW, _STEP_EXECUTE = range(3)

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._groups: list = []
        self._plan = None
        self._plan_executed = False
        self._group_overrides: dict[str, str] = {}
        self._worker: DupeBuildWorker | None = None
        self._exec_worker: DupeExecuteWorker | None = None
        self._scan_running = False
        self._exec_running = False
        self._preview_html_path: str = ""
        self._last_report_path: str = ""
        self._config_snapshot: tuple[float, float, float] | None = None
        self._step_labels: list[QtWidgets.QLabel] = []
        self._build_ui()
        self._load_thresholds_from_config(force=True)
        self._last_report_path = _latest_execution_report(library_path)
        self._refresh_buttons()

    def _build_ui(self) -> None:
        cl = self.content_layout

        # ── Header card ───────────────────────────────────────────────────
        header_card = self._make_card()
        header_card.setObjectName("headerCard")
        hl = QtWidgets.QVBoxLayout(header_card)
        hl.setContentsMargins(20, 16, 16, 16)
        hl.setSpacing(8)
        hl.addWidget(self._make_section_title("Duplicate Finder"))
        hl.addWidget(self._make_subtitle(
            "Fingerprint your library, group duplicate tracks, preview the plan, "
            "then quarantine or delete losers. "
            "Scan → Preview → Execute."
        ))

        stepper = QtWidgets.QFrame()
        stepper.setObjectName("workflowStepper")
        sl = QtWidgets.QHBoxLayout(stepper)
        sl.setContentsMargins(12, 8, 12, 8)
        sl.setSpacing(4)
        for i, step in enumerate(["1. Scan & fingerprint", "2. Review groups", "3. Execute plan"]):
            lbl = QtWidgets.QLabel(step)
            lbl.setObjectName("stepActive" if i == 0 else "stepInactive")
            self._step_labels.append(lbl)
            sl.addWidget(lbl)
            if i < 2:
                arr = QtWidgets.QLabel("→")
                arr.setObjectName("stepArrow")
                sl.addWidget(arr)
        sl.addStretch(1)
        hl.addWidget(stepper)
        cl.addWidget(header_card)

        # ── Config card ────────────────────────────────────────────────────
        cfg_card = self._make_card()
        cfg_layout = QtWidgets.QVBoxLayout(cfg_card)
        cfg_layout.setContentsMargins(16, 16, 16, 16)
        cfg_layout.setSpacing(12)
        cfg_layout.addWidget(self._make_card_title("Thresholds"))

        thresh_form = QtWidgets.QFormLayout()
        thresh_form.setContentsMargins(0, 0, 0, 0)
        thresh_form.setSpacing(6)

        self._exact_spin = QtWidgets.QDoubleSpinBox()
        self._exact_spin.setRange(0.0, 1.0)
        self._exact_spin.setSingleStep(0.005)
        self._exact_spin.setDecimals(3)
        self._exact_spin.setToolTip(
            "Fingerprint distance ≤ this value → exact duplicate.\n"
            "Starts from the value saved in Settings."
        )

        self._near_spin = QtWidgets.QDoubleSpinBox()
        self._near_spin.setRange(0.0, 1.0)
        self._near_spin.setSingleStep(0.01)
        self._near_spin.setDecimals(3)
        self._near_spin.setToolTip(
            "Fingerprint distance ≤ this value → near duplicate.\n"
            "Starts from the value saved in Settings."
        )

        self._boost_spin = QtWidgets.QDoubleSpinBox()
        self._boost_spin.setRange(0.0, 0.5)
        self._boost_spin.setSingleStep(0.005)
        self._boost_spin.setDecimals(3)
        self._boost_spin.setToolTip(
            "Extra distance allowance when comparing lossless vs. lossy files."
        )

        thresh_form.addRow("Exact duplicate threshold:", self._exact_spin)
        thresh_form.addRow("Near duplicate threshold:", self._near_spin)
        thresh_form.addRow("Mixed-codec boost:", self._boost_spin)
        cfg_layout.addLayout(thresh_form)

        opts_row = QtWidgets.QHBoxLayout()
        self._playlists_cb = QtWidgets.QCheckBox("Update playlists after execution")
        # The executor always rewrites playlist entries that point at a moved
        # loser (it would otherwise leave broken entries), so this is not a
        # choice: show it as always on rather than as a dead checkbox.
        self._playlists_cb.setChecked(True)
        self._playlists_cb.setEnabled(False)
        self._playlists_cb.setToolTip(
            "Always on: playlists that reference a duplicate that is moved or deleted are "
            "rewritten to point at the kept copy (backups are saved with the execution report)."
        )
        opts_row.addWidget(self._playlists_cb)
        opts_row.addStretch(1)
        cfg_layout.addLayout(opts_row)

        action_row = QtWidgets.QHBoxLayout()
        self._quarantine_rb = QtWidgets.QRadioButton("Quarantine duplicates (safe default)")
        self._quarantine_rb.setChecked(True)
        self._quarantine_rb.setToolTip("Losers are moved to the library's Quarantine/ folder.")
        self._delete_rb = QtWidgets.QRadioButton("Delete losers permanently (cannot be undone)")
        self._delete_rb.setToolTip(
            "Losers are deleted, not moved. Execute asks you to type DELETE to confirm."
        )
        action_row.addWidget(self._quarantine_rb)
        action_row.addWidget(self._delete_rb)
        action_row.addStretch(1)
        cfg_layout.addLayout(action_row)
        cl.addWidget(cfg_card)

        # ── Action card ────────────────────────────────────────────────────
        action_card = self._make_card()
        action_card.setObjectName("actionCard")
        ac_layout = QtWidgets.QVBoxLayout(action_card)
        ac_layout.setContentsMargins(16, 14, 16, 14)
        ac_layout.setSpacing(10)
        ac_layout.addWidget(self._make_card_title("Actions"))

        btn_row = QtWidgets.QHBoxLayout()
        self._scan_btn = self._make_primary_button("🔍  Scan Library")
        self._scan_btn.clicked.connect(self._on_scan)

        self._preview_btn = QtWidgets.QPushButton("📄  Preview")
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._on_preview)

        self._execute_btn = QtWidgets.QPushButton("▶  Execute")
        self._execute_btn.setEnabled(False)
        self._execute_btn.setObjectName("dangerBtn")
        self._execute_btn.clicked.connect(self._on_execute)

        self._report_btn = QtWidgets.QPushButton("📊  Open Report")
        self._report_btn.setEnabled(False)
        self._report_btn.setToolTip("Open the most recent execution report.")
        self._report_btn.clicked.connect(self._on_open_report)

        self._cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)

        btn_row.addWidget(self._scan_btn)
        btn_row.addWidget(self._preview_btn)
        btn_row.addWidget(self._execute_btn)
        btn_row.addWidget(self._report_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        ac_layout.addLayout(btn_row)
        cl.addWidget(action_card)

        # ── Progress ───────────────────────────────────────────────────────
        prog_card = self._make_card()
        prog_layout = QtWidgets.QVBoxLayout(prog_card)
        prog_layout.setContentsMargins(16, 12, 16, 12)
        prog_layout.setSpacing(6)

        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setValue(0)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setFixedHeight(6)
        self._prog_status = QtWidgets.QLabel("Idle")
        self._prog_status.setObjectName("statusHint")
        self._prog_status.setWordWrap(True)
        prog_layout.addWidget(self._prog_bar)
        prog_layout.addWidget(self._prog_status)
        cl.addWidget(prog_card)

        # ── Results: groups list + inspector ──────────────────────────────
        results_card = self._make_card()
        results_layout = QtWidgets.QHBoxLayout(results_card)
        results_layout.setContentsMargins(16, 16, 16, 16)
        results_layout.setSpacing(12)

        # Left: groups table
        left = QtWidgets.QVBoxLayout()
        self._fp_status_lbl = QtWidgets.QLabel("")
        self._fp_status_lbl.setObjectName("statusHint")
        left.addWidget(self._fp_status_lbl)

        self._groups_table = QtWidgets.QTreeWidget()
        self._groups_table.setHeaderLabels(["#", "Winner", "Losers", "Review"])
        self._groups_table.setColumnWidth(0, 40)
        self._groups_table.setColumnWidth(1, 220)
        self._groups_table.setColumnWidth(2, 55)
        self._groups_table.setColumnWidth(3, 60)
        self._groups_table.setMinimumHeight(220)
        self._groups_table.setAlternatingRowColors(True)
        self._groups_table.currentItemChanged.connect(self._on_group_selected)
        left.addWidget(self._groups_table)

        # Right: inspector
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Group details"))

        disp_row = QtWidgets.QHBoxLayout()
        disp_row.addWidget(QtWidgets.QLabel("Disposition:"))
        self._disposition_combo = QtWidgets.QComboBox()
        for label, value in _DISPOSITION_CHOICES:
            self._disposition_combo.addItem(label, value)
        self._disposition_combo.setEnabled(False)
        self._disposition_combo.setToolTip(
            "What happens to this group's losers. \"Default\" follows the "
            "Quarantine / Delete choice above."
        )
        self._disposition_combo.currentIndexChanged.connect(self._on_disposition_changed)
        disp_row.addWidget(self._disposition_combo)
        disp_row.addStretch(1)
        right.addLayout(disp_row)

        self._group_details = QtWidgets.QPlainTextEdit()
        self._group_details.setReadOnly(True)
        self._group_details.setMinimumHeight(200)
        self._group_details.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        self._group_details.setPlaceholderText("Select a group to see details…")
        right.addWidget(self._group_details)

        results_layout.addLayout(left, 2)
        results_layout.addLayout(right, 3)
        cl.addWidget(results_card)

        # ── Log ────────────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(self._make_card_title("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(120)
        self._log_area.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── State helpers ─────────────────────────────────────────────────────

    def _status(self, text: str, kind: str = "info") -> None:
        tokens = get_manager().current
        colour = {
            "busy": tokens.warning,
            "ok": tokens.success,
            "warn": tokens.warning,
            "error": tokens.danger,
        }.get(kind, tokens.text_secondary)
        self.status_changed.emit(text, colour)

    def _set_step(self, index: int) -> None:
        for i, lbl in enumerate(self._step_labels):
            name = "stepActive" if i == index else "stepInactive"
            if lbl.objectName() != name:
                lbl.setObjectName(name)
                lbl.style().unpolish(lbl)
                lbl.style().polish(lbl)

    def _current_step(self) -> int:
        for i, lbl in enumerate(self._step_labels):
            if lbl.objectName() == "stepActive":
                return i
        return 0

    def _refresh_buttons(self) -> None:
        busy = self._scan_running or self._exec_running
        self._scan_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        self._preview_btn.setEnabled(
            not busy and bool(self._preview_html_path) and Path(self._preview_html_path).exists()
        )
        self._execute_btn.setEnabled(
            not busy and self._plan is not None and bool(self._groups) and not self._plan_executed
        )
        self._report_btn.setEnabled(
            not self._exec_running and bool(self._last_report_path) and Path(self._last_report_path).exists()
        )

    def _load_thresholds_from_config(self, force: bool = False) -> None:
        """Use the thresholds saved in Settings; re-read when they changed."""
        try:
            values = _config_thresholds()
        except Exception:  # noqa: BLE001 - a broken config must not break the page
            return
        if not force and values == self._config_snapshot:
            return
        self._config_snapshot = values
        exact, near, boost = values
        self._exact_spin.setValue(exact)
        self._near_spin.setValue(near)
        self._boost_spin.setValue(boost)

    def on_settings_saved(self) -> None:
        # Follow thresholds saved in Settings (only reloads when they changed,
        # so a value the user typed here survives an unrelated save).
        self._load_thresholds_from_config(force=False)

    def _global_disposition(self) -> str:
        return "delete" if self._delete_rb.isChecked() else "quarantine"

    def _effective_dispositions(self) -> dict[str, str]:
        if self._plan is None:
            return {}
        default = self._global_disposition()
        out: dict[str, str] = {}
        for group in self._plan.groups:
            disposition = self._group_overrides.get(group.group_id) or default
            for loser in group.losers:
                out[loser] = disposition
        return out

    def _apply_dispositions(self, effective: dict[str, str]) -> bool:
        """Write the chosen dispositions into the plan; True if anything changed.

        The plan's signature is refreshed and the preview re-exported so the
        preview always shows exactly what Execute will do.
        """
        previous = {g.group_id: dict(g.loser_disposition) for g in self._plan.groups}
        changed = False
        for group in self._plan.groups:
            wanted = {loser: effective.get(loser, "quarantine") for loser in group.losers}
            if dict(group.loser_disposition) != wanted:
                group.loser_disposition = wanted
                changed = True
        if not changed:
            return False
        self._plan.refresh_plan_signature()
        try:
            from duplicate_consolidation import (
                export_consolidation_preview,
                export_consolidation_preview_html,
            )
            docs = Path(self._library_path) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            html_path = str(docs / "duplicate_preview.html")
            export_consolidation_preview_html(self._plan, html_path)
            export_consolidation_preview(self._plan, str(docs / "duplicate_preview.json"))
        except Exception:
            # Keep plan and preview in step: undo, so a retry re-exports.
            for group in self._plan.groups:
                group.loser_disposition = previous[group.group_id]
            self._plan.refresh_plan_signature()
            raise
        self._preview_html_path = html_path
        self._log_line("Preview updated with your disposition choices.")
        return True

    def _group_detail_text(self, group) -> str:
        override = self._group_overrides.get(group.group_id)
        default = self._global_disposition()
        lines = [f"Winner: {group.winner_path}"]
        for loser in group.losers:
            disposition = override or default
            lines.append(f"  Loser ({disposition}): {loser}")
        if override:
            lines.append(f"Disposition override: {override}")
        if group.review_flags:
            lines.append(f"Review flags: {', '.join(group.review_flags)}")
        return "\n".join(lines)

    def _log_line(self, line: str) -> None:
        self._log_area.appendPlainText(line)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        if self._scan_running or self._exec_running:
            return

        compute_missing = True
        missing = _missing_fingerprint_tools()
        if missing:
            names = " and ".join(missing)
            reply = QtWidgets.QMessageBox.question(
                self,
                "Fingerprinting tools not found",
                f"{names} could not be found on PATH.\n\n"
                "The Duplicate Finder fingerprints audio with fpcalc (Chromaprint — "
                "https://acoustid.org/chromaprint) and decodes it with FFmpeg "
                "(https://ffmpeg.org). Install them, make sure they are on PATH, "
                "then scan again.\n\n"
                "Scan anyway, using only fingerprints already in this library's cache? "
                "Files that were never fingerprinted will be skipped.",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                self._prog_status.setText(f"Not scanned: {names} not found.")
                self._log(f"Duplicate scan not started: {names} not found on PATH.", "warn")
                return
            compute_missing = False

        self._log_area.clear()
        self._groups_table.clear()
        self._group_details.clear()
        self._groups = []
        self._plan = None
        self._plan_executed = False
        self._group_overrides.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Scanning…")
        self._set_step(self._STEP_SCAN)
        self._log("Starting duplicate scan…", "info")
        self._status("Scanning…", "busy")
        if not compute_missing:
            self._log_line("fpcalc/ffmpeg unavailable: using cached fingerprints only.")

        self._worker = DupeBuildWorker(
            library_path=self._library_path,
            exact_threshold=self._exact_spin.value(),
            near_threshold=self._near_spin.value(),
            codec_boost=self._boost_spin.value(),
            compute_missing=compute_missing,
        )
        self._worker.progress.connect(self._on_scan_progress)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.groups_ready.connect(self._populate_groups)
        self._worker.done.connect(self._on_scan_finished)
        self._scan_running = True
        self._refresh_buttons()
        self._start_worker(self._worker, "Duplicate scan")

    @Slot()
    def _on_cancel(self) -> None:
        if self._scan_running and self._worker is not None:
            self._worker.cancel()
            self._prog_status.setText("Cancelling…")
            self._cancel_btn.setEnabled(False)
        if self._exec_running and self._exec_worker is not None:
            self._exec_worker.cancel()
            self._prog_status.setText("Cancelling at the next safe point…")
            self._cancel_btn.setEnabled(False)

    @Slot()
    def _on_preview(self) -> None:
        if self._preview_html_path and Path(self._preview_html_path).exists():
            webbrowser.open(Path(self._preview_html_path).resolve().as_uri())
        else:
            QtWidgets.QMessageBox.information(self, "No Preview", "Preview not yet generated — run a scan first.")

    def _review_gate(self) -> tuple[bool, bool]:
        """(proceed, allow_review_required). Shows why groups are flagged."""
        plan = self._plan
        flagged = [(i, g) for i, g in enumerate(plan.groups, 1) if g.review_flags]
        if not flagged and not plan.review_flags:
            return True, False
        lines = [
            f"{len(flagged)} of {len(plan.groups)} duplicate group(s) are flagged for review:",
            "",
        ]
        for index, group in flagged[:8]:
            lines.append(f"  #{index} {Path(group.winner_path).name} — {'; '.join(group.review_flags)}")
        if len(flagged) > 8:
            lines.append(f"  … and {len(flagged) - 8} more (see the preview)")
        if plan.review_flags:
            lines.append("")
            lines.append("Plan-wide flags:")
            lines.extend(f"  • {flag}" for flag in plan.review_flags[:5])
        lines += [
            "",
            "The executor refuses a plan with review flags unless you explicitly override.",
            "",
            "Yes — execute ALL groups exactly as previewed, including the flagged ones.",
            "No — stop here; nothing will be moved or changed.",
        ]
        reply = QtWidgets.QMessageBox.question(
            self,
            "Groups flagged for review",
            "\n".join(lines),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return False, False
        return True, True

    def _planned_operation_count(self) -> int:
        plan = self._plan
        losers = sum(len(g.losers) for g in plan.groups)
        artwork = sum(len(g.artwork) for g in plan.groups)
        tags = sum(1 for g in plan.groups if g.metadata_changes)
        rewrites: set[str] = set()
        for group in plan.groups:
            rewrites.update(group.playlist_rewrites)
        return losers + artwork + tags + len(rewrites)

    @Slot()
    def _on_execute(self) -> None:
        if self._plan is None or not self._groups:
            QtWidgets.QMessageBox.warning(self, "No Plan", "No consolidation plan — run a scan first.")
            return
        if self._plan_executed:
            QtWidgets.QMessageBox.information(
                self, "Plan already executed", "This plan has been executed. Scan again to build a fresh plan."
            )
            return
        if self._scan_running or self._exec_running:
            return
        if self._library_writes_blocked("Duplicate execution"):
            return

        proceed, allow_review = self._review_gate()
        if not proceed:
            msg = "Nothing was moved: the plan has groups flagged for review."
            self._prog_status.setText(msg)
            self._log(msg, "warn")
            self._status("Not executed", "warn")
            return

        effective = self._effective_dispositions()
        try:
            self._apply_dispositions(effective)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(
                self, "Preview not updated",
                f"Could not update the preview with your disposition choices:\n{exc}\n\n"
                "Nothing was changed.",
            )
            return

        quarantine = sum(1 for d in effective.values() if d == "quarantine")
        delete = sum(1 for d in effective.values() if d == "delete")
        retain = sum(1 for d in effective.values() if d == "retain")
        tag_updates = sum(1 for g in self._plan.groups if g.metadata_changes)
        artwork = sum(len(g.artwork) for g in self._plan.groups)
        operations = self._planned_operation_count()
        overage = operations > _EXECUTOR_OPERATION_LIMIT

        lines = [f"Execute the duplicate plan for {len(self._plan.groups)} group(s)?", ""]
        if quarantine:
            lines.append(f"• {quarantine} duplicate file(s) will be moved to Quarantine/.")
        if delete:
            lines.append(f"• {delete} duplicate file(s) will be PERMANENTLY DELETED.")
        if retain:
            lines.append(f"• {retain} duplicate file(s) will stay where they are.")
        if tag_updates or artwork:
            lines.append(
                f"• Tags will be updated on {tag_updates} kept file(s); "
                f"{artwork} artwork change(s) will be applied."
            )
        lines.append("• Playlists that reference a moved file are rewritten to the kept copy (backups kept).")
        if overage:
            lines.append(
                f"• This is a large plan: {operations} operations, above the usual "
                f"{_EXECUTOR_OPERATION_LIMIT:,} safety limit. Confirming runs all of them."
            )
        lines += [
            "",
            "The preview (Docs/duplicate_preview.html) shows exactly this plan. "
            "Every action is logged under Docs/duplicate_execution_reports/.",
        ]
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Execute", "\n".join(lines),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        allow_deletion = False
        if delete:
            text, ok = QtWidgets.QInputDialog.getText(
                self,
                "Confirm permanent deletion",
                f"{delete} file(s) will be permanently deleted. They will not go to "
                "Quarantine and cannot be recovered.\n\n"
                f"Type {_DELETE_CONFIRM_WORD} to confirm:",
            )
            if not ok or text.strip() != _DELETE_CONFIRM_WORD:
                msg = "Deletion not confirmed — nothing was changed."
                self._prog_status.setText(msg)
                self._log(msg, "warn")
                return
            allow_deletion = True

        self._prog_status.setText("Executing…")
        self._prog_bar.setValue(0)
        self._set_step(self._STEP_EXECUTE)
        self._status("Executing…", "busy")

        self._exec_worker = DupeExecuteWorker(
            plan=self._plan,
            library_path=self._library_path,
            allow_review_required=allow_review,
            allow_deletion=allow_deletion,
            confirm_operation_overage=overage,
        )
        self._exec_worker.log_line.connect(self._on_log_line)
        self._exec_worker.done.connect(self._on_execute_finished)
        self._exec_running = True
        self._refresh_buttons()
        self._start_worker(self._exec_worker, "Duplicate execution", library_write=True)

    @Slot()
    def _on_open_report(self) -> None:
        report = self._last_report_path
        if not report or not Path(report).exists():
            report = _latest_execution_report(self._library_path)
        if report and Path(report).exists():
            self._last_report_path = report
            webbrowser.open(Path(report).resolve().as_uri())
            return
        QtWidgets.QMessageBox.information(
            self, "No Report",
            "No execution report yet — reports are written when you Execute. "
            "Use Preview to see the plan.",
        )

    @Slot(object, object)
    def _on_group_selected(self, current, previous) -> None:  # noqa: ANN001
        if current is None:
            self._group_details.clear()
            self._disposition_combo.setEnabled(False)
            return
        index = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        group = self._groups[index] if isinstance(index, int) and 0 <= index < len(self._groups) else None
        if group is None:
            self._group_details.setPlainText(f"Group: {current.text(0)}")
            self._disposition_combo.setEnabled(False)
            return
        override = self._group_overrides.get(group.group_id)
        combo_index = next(
            (i for i, (_label, value) in enumerate(_DISPOSITION_CHOICES) if value == override), 0
        )
        self._disposition_combo.blockSignals(True)
        self._disposition_combo.setCurrentIndex(combo_index)
        self._disposition_combo.blockSignals(False)
        self._disposition_combo.setEnabled(not self._plan_executed and not self._exec_running)
        self._group_details.setPlainText(self._group_detail_text(group))

    @Slot(int)
    def _on_disposition_changed(self, combo_index: int) -> None:
        item = self._groups_table.currentItem()
        if item is None:
            return
        index = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(index, int) or not (0 <= index < len(self._groups)):
            return
        group = self._groups[index]
        value = self._disposition_combo.itemData(combo_index)
        if value:
            self._group_overrides[group.group_id] = value
        else:
            self._group_overrides.pop(group.group_id, None)
        self._group_details.setPlainText(self._group_detail_text(group))

    @Slot(int)
    def _on_scan_progress(self, value: int) -> None:
        if self.sender() is not None and self.sender() is not self._worker:
            return
        self._prog_bar.setValue(value)

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._worker and sender is not self._exec_worker:
            return  # a job from before a library change
        self._log_area.appendPlainText(line)

    @Slot(bool, str)
    def _on_scan_finished(self, success: bool, message: str) -> None:
        worker = self.sender()
        if worker is not None and worker is not self._worker:
            return  # stale scan (library changed meanwhile); its results are discarded
        self._scan_running = False
        if self._worker is not None:
            self._preview_html_path = self._worker.preview_html_path
            if success:
                self._plan = self._worker.plan

        self._prog_status.setText(message)

        if success:
            self._log(message, "ok")
            self._status("Scan complete", "ok")
            self._set_step(self._STEP_REVIEW if self._groups else self._STEP_SCAN)
        elif message == "Cancelled.":
            self._log(message, "warn")
            self._status("Cancelled", "warn")
            self._plan = None
        else:
            self._log(message, "error")
            self._status("Error", "error")
            self._plan = None
        self._refresh_buttons()

    @Slot(bool, str, str)
    def _on_execute_finished(self, success: bool, message: str, report_path: str) -> None:
        worker = self.sender()
        if worker is not None and worker is not self._exec_worker:
            # Execution of the previous library's plan, stopped by a library change.
            where = f" Report: {report_path}" if report_path else ""
            self._log(f"Previous library's duplicate execution ended: {message}{where}", "warn")
            return
        self._exec_running = False
        self._plan_executed = True
        self._prog_status.setText(f"{message} Scan again to build a fresh plan.")

        if success:
            self._log(message, "ok")
            self._status("Execution complete", "ok")
            self._prog_bar.setValue(100)
        else:
            self._log(message, "error")
            self._status("Not executed" if message.startswith("Execution blocked") else "Error", "error")

        if report_path and Path(report_path).exists():
            self._last_report_path = report_path
            webbrowser.open(Path(report_path).resolve().as_uri())

        self._disposition_combo.setEnabled(False)
        self._refresh_buttons()

    # ── Helpers ───────────────────────────────────────────────────────────

    @Slot(list)
    def _populate_groups(self, groups: list) -> None:
        if self.sender() is not None and self.sender() is not self._worker:
            return
        self._groups = list(groups)
        self._groups_table.clear()
        for i, group in enumerate(self._groups):
            winner_name = Path(group.winner_path).name
            review = "⚑" if group.review_flags else ""
            item = QtWidgets.QTreeWidgetItem([str(i + 1), winner_name[:80], str(len(group.losers)), review])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, i)
            if group.review_flags:
                item.setToolTip(3, "\n".join(group.review_flags))
            self._groups_table.addTopLevelItem(item)
        flagged = sum(1 for g in self._groups if g.review_flags)
        text = f"{len(self._groups)} duplicate group(s) found"
        if flagged:
            text += f" — {flagged} flagged for review (⚑)"
        self._fp_status_lbl.setText(text)

    def _abandon_running_jobs(self, reason: str) -> None:
        """Ask running jobs to stop; their late results are ignored."""
        if self._scan_running and self._worker is not None:
            self._worker.cancel()
            self._log(f"{reason}: cancelled the running duplicate scan.", "warn")
        if self._exec_running and self._exec_worker is not None:
            self._exec_worker.cancel()
            self._log(
                f"{reason}: asked the running duplicate execution to stop at the next safe "
                "point. Its report is still written to the previous library's Docs/ folder.",
                "warn",
            )
        self._worker = None
        self._exec_worker = None
        self._scan_running = False
        self._exec_running = False

    def _on_library_changed(self, path: str) -> None:
        self._abandon_running_jobs("Library changed")
        self._groups_table.clear()
        self._group_details.clear()
        self._log_area.clear()
        self._fp_status_lbl.setText("")
        self._groups = []
        self._plan = None
        self._plan_executed = False
        self._group_overrides.clear()
        self._preview_html_path = ""
        self._last_report_path = _latest_execution_report(path)
        self._prog_bar.setValue(0)
        self._prog_status.setText("Idle")
        self._disposition_combo.setEnabled(False)
        self._set_step(self._STEP_SCAN)
        self._refresh_buttons()
