"""Duplicate Finder workspace — scan → preview → execute."""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase

_AUDIO_EXTS = {'.flac', '.m4a', '.aac', '.mp3', '.wav', '.ogg', '.opus'}
_SKIP_DIRS = {'not sorted', 'playlists', 'quarantine', 'manual review', 'docs', 'trash'}


def _gather_tracks_for_plan(
    library_path: str,
    db_path: str,
    log_callback,
    progress_callback,
    cancel_event: threading.Event,
) -> list[dict]:
    """Walk ``library_path``, fingerprint files, and return a list of track dicts."""
    from fingerprint_cache import ensure_fingerprint_cache, get_cached_fingerprint_metadata, store_fingerprint
    import chromaprint_utils
    from utils.audio_metadata_reader import read_tags

    ensure_fingerprint_cache(db_path)

    audio_paths: list[str] = []
    for dirpath, dirs, files in os.walk(library_path):
        rel = os.path.relpath(dirpath, library_path)
        parts = {p.lower() for p in rel.split(os.sep)}
        if _SKIP_DIRS & parts:
            dirs.clear()
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in _AUDIO_EXTS:
                audio_paths.append(os.path.join(dirpath, fname))

    total = len(audio_paths)
    tracks: list[dict] = []
    for i, path in enumerate(sorted(audio_paths)):
        if cancel_event.is_set():
            break

        fp = None
        cached_meta: dict | None = None
        try:
            fp, cached_meta = get_cached_fingerprint_metadata(path, db_path)
        except Exception:
            pass

        duration: int | None = None
        if not fp:
            try:
                duration, fp = chromaprint_utils.fingerprint_fpcalc(path)
            except Exception as exc:
                log_callback(f"! Fingerprint failed: {Path(path).name}: {exc}")
                progress_callback(int((i + 1) / total * 60))
                continue
            if not fp:
                log_callback(f"! No fingerprint: {Path(path).name}")
                progress_callback(int((i + 1) / total * 60))
                continue
            # Store newly computed fingerprint
            try:
                tags: dict = {}
                try:
                    tags = dict(read_tags(path) or {})
                except Exception:
                    pass
                ext_val = os.path.splitext(path)[1].lower()
                store_fingerprint(path, db_path, duration, fp, ext=ext_val, tags=tags)
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
        })
        progress_callback(int((i + 1) / total * 60))

    return tracks


# ── Workers ───────────────────────────────────────────────────────────────────

class DupeBuildWorker(QtCore.QThread):
    """Fingerprint library, build consolidation plan, and write preview HTML."""
    progress = Signal(int)
    log_line = Signal(str)
    groups_ready = Signal(list)     # list of GroupPlan
    finished = Signal(bool, str)

    def __init__(
        self,
        library_path: str,
        exact_threshold: float,
        near_threshold: float,
        codec_boost: float,
    ) -> None:
        super().__init__()
        self.library_path = library_path
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
        self.codec_boost = codec_boost
        self._cancel_event = threading.Event()
        # Populated during run(); read by workspace after finished
        self.plan = None
        self.preview_html_path: str = ""

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            from duplicate_consolidation import (
                build_consolidation_plan,
                export_consolidation_preview_html,
            )
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
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
            )

            if self._cancel_event.is_set():
                self.finished.emit(False, "Cancelled.")
                return

            if not tracks:
                self.finished.emit(False, "No fingerprinted tracks found.")
                return

            _log(f"Stage 2: building consolidation plan for {len(tracks)} tracks…")
            self.progress.emit(62)
            plan = build_consolidation_plan(
                tracks,
                exact_duplicate_threshold=self.exact_threshold,
                near_duplicate_threshold=self.near_threshold,
                mixed_codec_threshold_boost=self.codec_boost,
                log_callback=_log,
            )
            self.plan = plan
            self.progress.emit(85)

            if self._cancel_event.is_set():
                self.finished.emit(False, "Cancelled.")
                return

            _log(f"Stage 3: writing HTML preview…")
            html_path = str(docs / "duplicate_preview.html")
            try:
                export_consolidation_preview_html(plan, html_path)
                self.preview_html_path = html_path
                _log(f"Preview written → {html_path}")
            except Exception as exc:
                _log(f"Warning: could not write HTML preview: {exc}")

            self.progress.emit(100)
            self.groups_ready.emit(plan.groups)
            self.finished.emit(True, f"Found {len(plan.groups)} duplicate group(s).")

        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class DupeExecuteWorker(QtCore.QThread):
    """Execute a consolidation plan."""
    log_line = Signal(str)
    finished = Signal(bool, str, str)   # success, message, report_html_path

    def __init__(self, plan, library_path: str, quarantine: bool) -> None:
        super().__init__()
        self.plan = plan
        self.library_path = library_path
        self.quarantine = quarantine
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            from duplicate_consolidation_executor import ExecutionConfig, execute_consolidation_plan
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}", "")
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
                retain_losers=not self.quarantine,
                allow_deletion=False,
                cancel_event=self._cancel_event,
                log_callback=lambda msg: self.log_line.emit(msg),
            )

            result = execute_consolidation_plan(self.plan, config)
            report_html = result.report_paths.get("html_report", "")
            if result.success:
                self.finished.emit(True, "Execution complete.", str(report_html or ""))
            else:
                self.finished.emit(False, "Execution completed with errors.", str(report_html or ""))
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc), "")


# ── Workspace ─────────────────────────────────────────────────────────────────

class DuplicatesWorkspace(WorkspaceBase):
    """Duplicate Finder — scan, review groups, preview and execute."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._groups: list = []
        self._worker: DupeBuildWorker | None = None
        self._exec_worker: DupeExecuteWorker | None = None
        self._preview_html_path: str = ""
        self._build_ui()

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
        self._exact_spin.setValue(0.02)
        self._exact_spin.setToolTip("Fingerprint distance ≤ this value → exact duplicate.")

        self._near_spin = QtWidgets.QDoubleSpinBox()
        self._near_spin.setRange(0.0, 1.0)
        self._near_spin.setSingleStep(0.01)
        self._near_spin.setDecimals(3)
        self._near_spin.setValue(0.1)
        self._near_spin.setToolTip("Fingerprint distance ≤ this value → near duplicate.")

        self._boost_spin = QtWidgets.QDoubleSpinBox()
        self._boost_spin.setRange(0.0, 0.5)
        self._boost_spin.setSingleStep(0.005)
        self._boost_spin.setDecimals(3)
        self._boost_spin.setValue(0.03)
        self._boost_spin.setToolTip(
            "Extra distance allowance when comparing lossless vs. lossy files."
        )

        thresh_form.addRow("Exact duplicate threshold:", self._exact_spin)
        thresh_form.addRow("Near duplicate threshold:", self._near_spin)
        thresh_form.addRow("Mixed-codec boost:", self._boost_spin)
        cfg_layout.addLayout(thresh_form)

        opts_row = QtWidgets.QHBoxLayout()
        self._playlists_cb = QtWidgets.QCheckBox("Update playlists after execution")
        opts_row.addWidget(self._playlists_cb)
        opts_row.addStretch(1)
        cfg_layout.addLayout(opts_row)

        action_row = QtWidgets.QHBoxLayout()
        self._quarantine_rb = QtWidgets.QRadioButton("Quarantine duplicates (safe default)")
        self._quarantine_rb.setChecked(True)
        self._delete_rb = QtWidgets.QRadioButton("Delete losers permanently")
        self._delete_rb.setToolTip("Irreversible — use with caution.")
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
        self._disposition_combo.addItems(["Default (global)", "Retain", "Quarantine", "Delete"])
        self._disposition_combo.setEnabled(False)
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

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return

        self._log_area.clear()
        self._groups_table.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Scanning…")
        self._scan_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._preview_btn.setEnabled(False)
        self._execute_btn.setEnabled(False)
        self._report_btn.setEnabled(False)
        self._log("Starting duplicate scan…", "info")
        self.status_changed.emit("Scanning…", "#f59e0b")

        self._worker = DupeBuildWorker(
            library_path=self._library_path,
            exact_threshold=self._exact_spin.value(),
            near_threshold=self._near_spin.value(),
            codec_boost=self._boost_spin.value(),
        )
        self._worker.progress.connect(self._prog_bar.setValue)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.groups_ready.connect(self._populate_groups)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
        if self._exec_worker and self._exec_worker.isRunning():
            self._exec_worker.cancel()
            self._cancel_btn.setEnabled(False)

    @Slot()
    def _on_preview(self) -> None:
        if self._preview_html_path and Path(self._preview_html_path).exists():
            webbrowser.open(f"file://{self._preview_html_path}")
        else:
            QtWidgets.QMessageBox.information(self, "No Preview", "Preview not yet generated — run a scan first.")

    @Slot()
    def _on_execute(self) -> None:
        if not self._worker or self._worker.plan is None:
            QtWidgets.QMessageBox.warning(self, "No Plan", "No consolidation plan — run a scan first.")
            return

        action = "delete" if self._delete_rb.isChecked() else "quarantine"
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Execute",
            f"This will {action} duplicate losers in your library.\n\n"
            "This action cannot be undone. Review the preview first.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self._execute_btn.setEnabled(False)
        self._scan_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._prog_status.setText("Executing…")
        self._prog_bar.setValue(0)
        self.status_changed.emit("Executing…", "#f59e0b")

        self._exec_worker = DupeExecuteWorker(
            plan=self._worker.plan,
            library_path=self._library_path,
            quarantine=self._quarantine_rb.isChecked(),
        )
        self._exec_worker.log_line.connect(self._on_log_line)
        self._exec_worker.finished.connect(self._on_execute_finished)
        self._exec_worker.start()

    @Slot()
    def _on_open_report(self) -> None:
        # Check execution report first, then preview
        for candidate in [
            Path(self._library_path) / "Docs" / "duplicate_execution_reports",
            Path(self._library_path) / "Docs" / "duplicate_preview.html",
        ]:
            if candidate.is_file():
                webbrowser.open(f"file://{candidate}")
                return
            if candidate.is_dir():
                # Find most recent HTML in reports dir
                reports = sorted(candidate.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
                if reports:
                    webbrowser.open(f"file://{reports[0]}")
                    return
        QtWidgets.QMessageBox.information(self, "No Report", "No duplicate report found.")

    @Slot(object, object)
    def _on_group_selected(self, current, previous) -> None:  # noqa: ANN001
        if current is None:
            self._group_details.setPlaceholderText("Select a group to see details…")
            self._disposition_combo.setEnabled(False)
            return
        self._disposition_combo.setEnabled(True)
        data = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if data:
            self._group_details.setPlainText(str(data))
        else:
            self._group_details.setPlainText(f"Group: {current.text(0)}")

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self._log_area.appendPlainText(line)

    @Slot(bool, str)
    def _on_scan_finished(self, success: bool, message: str) -> None:
        if self._worker:
            self._preview_html_path = self._worker.preview_html_path

        self._scan_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._prog_status.setText(message)

        if success:
            self._log(message, "ok")
            self.status_changed.emit("Scan complete", "#22c55e")
            if self._groups:
                self._preview_btn.setEnabled(True)
                self._execute_btn.setEnabled(True)
        else:
            self._log(message, "error" if message != "Cancelled." else "warn")
            self.status_changed.emit("Error" if message != "Cancelled." else "Idle", "#ef4444")

    @Slot(bool, str, str)
    def _on_execute_finished(self, success: bool, message: str, report_path: str) -> None:
        self._scan_btn.setEnabled(True)
        self._execute_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._prog_status.setText(message)

        if success:
            self._log(message, "ok")
            self.status_changed.emit("Execution complete", "#22c55e")
            self._prog_bar.setValue(100)
        else:
            self._log(message, "error")
            self.status_changed.emit("Error", "#ef4444")

        if report_path and Path(report_path).exists():
            self._report_btn.setEnabled(True)
            webbrowser.open(f"file://{report_path}")

        self._exec_worker = None

    # ── Helpers ───────────────────────────────────────────────────────────

    @Slot(list)
    def _populate_groups(self, groups: list) -> None:
        self._groups = groups
        self._groups_table.clear()
        for i, group in enumerate(groups, 1):
            winner_name = Path(group.winner_path).name if hasattr(group, 'winner_path') else str(group)
            loser_count = str(len(group.losers)) if hasattr(group, 'losers') else "?"
            review = "⚑" if (hasattr(group, 'review_flags') and group.review_flags) else ""
            item = QtWidgets.QTreeWidgetItem([str(i), winner_name[:80], loser_count, review])

            # Build detail text
            if hasattr(group, 'winner_path') and hasattr(group, 'losers'):
                lines = [f"Winner: {group.winner_path}"]
                for loser in group.losers:
                    disp = group.loser_disposition.get(loser, "quarantine") if hasattr(group, 'loser_disposition') else "quarantine"
                    lines.append(f"  Loser ({disp}): {loser}")
                if hasattr(group, 'review_flags') and group.review_flags:
                    lines.append(f"Review flags: {', '.join(group.review_flags)}")
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, "\n".join(lines))
            else:
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(group))

            self._groups_table.addTopLevelItem(item)
        self._fp_status_lbl.setText(f"{len(groups)} duplicate group(s) found")

    def _on_library_changed(self, path: str) -> None:
        self._groups_table.clear()
        self._group_details.clear()
        self._log_area.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Idle")
        self._preview_btn.setEnabled(False)
        self._execute_btn.setEnabled(False)
        self._report_btn.setEnabled(False)
        self._scan_btn.setEnabled(True)
        self._worker = None
        self._exec_worker = None
