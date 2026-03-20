"""Library Sync workspace — compare two libraries and transfer files."""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


# ── Workers ───────────────────────────────────────────────────────────────────

class SyncScanWorker(QtCore.QThread):
    """Quick comparison: fingerprint both sides and classify incoming files."""
    progress = Signal(int, str)    # percent, side ("existing" | "incoming")
    log_line = Signal(str)
    finished = Signal(bool, str, object)  # success, message, result dict

    def __init__(self, existing: str, incoming: str, thresholds: dict) -> None:
        super().__init__()
        self.existing = existing
        self.incoming = incoming
        self.thresholds = thresholds
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            import library_sync
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}", None)
            return
        try:
            def _log(msg: str) -> None:
                if not self._cancel_event.is_set():
                    self.log_line.emit(msg)

            docs = Path(self.existing) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            db_path = str(docs / ".library_sync_fp.db")

            result = library_sync.compare_libraries(
                self.existing,
                self.incoming,
                db_path,
                thresholds=self.thresholds,
                log_callback=_log,
                cancel_event=self._cancel_event,
                include_match_objects=True,  # Request MatchResult objects for review flags
            )
            if self._cancel_event.is_set():
                self.finished.emit(False, "Cancelled.", None)
            else:
                counts = {k: len(v) for k, v in result.items() if isinstance(v, list)} if isinstance(result, dict) else {}
                summary = ", ".join(f"{v} {k}" for k, v in counts.items()) if counts else str(result)
                self.finished.emit(True, f"Scan complete — {summary}.", result)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc), None)


class SyncBuildWorker(QtCore.QThread):
    """Build a Library Sync plan and write the preview HTML."""
    log_line = Signal(str)
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, library_root: str, incoming_folder: str, transfer_mode: str,
                 review_flags=None, match_results=None) -> None:
        super().__init__()
        self.library_root = library_root
        self.incoming_folder = incoming_folder
        self.transfer_mode = transfer_mode
        self.review_flags = review_flags
        self.match_results = match_results
        self._cancel_event = threading.Event()
        # Populated during run()
        self.plan = None
        self.preview_html_path: str = ""

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            import library_sync
            from library_sync import IndexCancelled
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            docs = Path(self.library_root) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            output_html = str(docs / "LibrarySyncPreview.html")

            def _log(msg: str) -> None:
                self.log_line.emit(msg)

            def _progress(current: int, total: int, path: str, phase: str) -> None:
                if total:
                    self.progress.emit(int(current / total * 100))

            # Convert review flags to path overrides for the plan builder
            copy_only_paths = []
            allowed_replacement_paths = []
            if self.review_flags and self.match_results:
                copy_only_paths, allowed_replacement_paths = library_sync.resolve_review_flags_to_paths(
                    self.review_flags, self.match_results
                )

            plan = library_sync.build_library_sync_preview(
                self.library_root,
                self.incoming_folder,
                output_html,
                log_callback=_log,
                progress_callback=_progress,
                cancel_event=self._cancel_event,
                transfer_mode=self.transfer_mode,
                copy_only_paths=copy_only_paths,
                allowed_replacement_paths=allowed_replacement_paths,
            )
            self.plan = plan
            self.preview_html_path = output_html
            moves = len(plan.moves) if hasattr(plan, "moves") else "?"
            self.finished.emit(True, f"Plan built — {moves} file operation(s). Preview written.")
        except Exception as exc:  # noqa: BLE001
            cancelled_names = ("IndexCancelled", "Cancelled")
            if type(exc).__name__ in cancelled_names:
                self.finished.emit(False, "Cancelled.")
            else:
                self.finished.emit(False, str(exc))


class SyncExecuteWorker(QtCore.QThread):
    """Execute a Library Sync plan."""
    log_line = Signal(str)
    finished = Signal(bool, str, str)   # success, message, report_path

    def __init__(self, plan, create_playlist: bool) -> None:
        super().__init__()
        self.plan = plan
        self.create_playlist = create_playlist
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            import library_sync
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}", "")
            return
        try:
            def _log(msg: str) -> None:
                self.log_line.emit(msg)

            summary = library_sync.execute_library_sync_plan(
                self.plan,
                log_callback=_log,
                cancel_event=self._cancel_event,
                create_playlist=self.create_playlist,
            )
            report_path = str(summary.get("executed_report_path") or "")
            moved = summary.get("moved", 0)
            copied = summary.get("copied", 0)
            msg = f"Execution complete — {moved} moved, {copied} copied."
            self.finished.emit(True, msg, report_path)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc), "")


# ── Workspace ─────────────────────────────────────────────────────────────────

class LibrarySyncWorkspace(WorkspaceBase):
    """Library Sync — scan two folders, build plan, preview and execute."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._scan_result = None
        self._worker: SyncScanWorker | None = None
        self._build_worker: SyncBuildWorker | None = None
        self._exec_worker: SyncExecuteWorker | None = None

        # Initialize review state store for per-item user flags
        from library_sync_review_state import ReviewStateStore
        self._review_state_store = ReviewStateStore()
        self._current_match_results: list = []  # Store MatchResult objects from scan

        self._build_ui()

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
        for i, step in enumerate(["1. Configure paths", "2. Scan & match", "3. Build & execute plan"]):
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
            "Default fingerprint distance cutoff. Lower = stricter match."
        )
        cfg_layout.addRow("Global threshold:", self._threshold_entry)

        self._preset_entry = QtWidgets.QLineEdit()
        self._preset_entry.setPlaceholderText("Optional label for this session")
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
        self._recompute_btn.clicked.connect(self._on_recompute)
        self._save_session_btn = QtWidgets.QPushButton("💾  Save Session")
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

        # Existing tracks
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self._make_card_title("Existing Tracks"))
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
        self._transfer_toggle.toggled.connect(
            lambda checked: self._transfer_toggle.setText(
                "Copy Originals" if checked else "Move Originals"
            )
        )

        self._execute_plan_btn = QtWidgets.QPushButton("▶  Execute Plan")
        self._execute_plan_btn.setObjectName("dangerBtn")
        self._execute_plan_btn.setEnabled(False)
        self._execute_plan_btn.clicked.connect(self._on_execute_plan)

        self._output_playlist_cb = QtWidgets.QCheckBox("Write transfer playlist (.m3u8)")

        plan_btn_row.addWidget(self._build_plan_btn)
        plan_btn_row.addWidget(self._preview_plan_btn)
        plan_btn_row.addWidget(self._transfer_toggle)
        plan_btn_row.addWidget(self._execute_plan_btn)
        plan_btn_row.addWidget(self._output_playlist_cb)
        plan_btn_row.addStretch(1)
        plan_layout.addLayout(plan_btn_row)

        self._plan_status_lbl = QtWidgets.QLabel("No plan built.")
        self._plan_status_lbl.setObjectName("statusHint")
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

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        existing = self._existing_entry.text().strip()
        incoming = self._incoming_entry.text().strip()
        if not existing or not incoming:
            QtWidgets.QMessageBox.warning(
                self, "Missing Paths", "Please set both the existing library and incoming folder."
            )
            return

        # Parse threshold overrides
        thresholds: dict = {}
        try:
            default_t = float(self._threshold_entry.text().strip() or "0.3")
            thresholds["default"] = default_t
            for line in self._overrides_text.toPlainText().splitlines():
                line = line.strip()
                if "=" in line:
                    ext, val = line.split("=", 1)
                    thresholds[ext.strip()] = float(val.strip())
        except ValueError:
            pass

        # Clear review state for fresh scan
        self._review_state_store.clear_all()
        self._current_match_results = []
        self._log_area.clear()
        self._incoming_table.clear()
        self._existing_table.clear()
        self._state_lbl.setText("State: Scanning…")
        self._scan_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._build_plan_btn.setEnabled(False)
        self._exist_bar.setValue(0)
        self._incoming_bar.setValue(0)
        self._exist_status.setText("Scanning…")
        self._incoming_status.setText("Scanning…")
        self._log("Starting library sync scan…", "info")
        self.status_changed.emit("Scanning…", "#f59e0b")

        self._worker = SyncScanWorker(existing, incoming, thresholds)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    @Slot()
    def _on_cancel(self) -> None:
        for w in (self._worker, self._build_worker, self._exec_worker):
            if w and w.isRunning():
                w.cancel()
        self._cancel_btn.setEnabled(False)
        self._state_lbl.setText("State: Cancelling…")

    @Slot()
    def _on_recompute(self) -> None:
        """Re-run the scan with current threshold settings."""
        self._on_scan()

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
        library_root = self._existing_entry.text().strip()
        incoming = self._incoming_entry.text().strip()
        if not library_root or not incoming:
            QtWidgets.QMessageBox.warning(self, "Missing Paths", "Set both library paths before building a plan.")
            return

        transfer_mode = "copy" if self._transfer_toggle.isChecked() else "move"
        self._build_plan_btn.setEnabled(False)
        self._preview_plan_btn.setEnabled(False)
        self._execute_plan_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._plan_status_lbl.setText("Building plan + preview…")
        self._plan_bar.setValue(0)
        self._log("Building Library Sync plan and preview…", "info")
        self.status_changed.emit("Building plan…", "#f59e0b")

        self._build_worker = SyncBuildWorker(
            library_root, incoming, transfer_mode,
            review_flags=self._review_state_store,
            match_results=self._current_match_results
        )
        self._build_worker.log_line.connect(self._on_log_line)
        self._build_worker.progress.connect(self._plan_bar.setValue)
        self._build_worker.finished.connect(self._on_build_finished)
        self._build_worker.start()

    @Slot()
    def _on_preview_plan(self) -> None:
        html = Path(self._existing_entry.text()) / "Docs" / "LibrarySyncPreview.html"
        if html.exists():
            webbrowser.open(f"file://{html}")
        else:
            self._log("No preview HTML found — click Build Plan first.", "warn")

    @Slot()
    def _on_execute_plan(self) -> None:
        if not self._build_worker or self._build_worker.plan is None:
            QtWidgets.QMessageBox.warning(self, "No Plan", "Build a plan first.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Execute",
            "This will transfer files to the existing library.\n\nThis cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        create_playlist = self._output_playlist_cb.isChecked()
        self._execute_plan_btn.setEnabled(False)
        self._build_plan_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._plan_status_lbl.setText("Executing…")
        self._plan_bar.setValue(0)
        self.status_changed.emit("Executing…", "#f59e0b")

        self._exec_worker = SyncExecuteWorker(self._build_worker.plan, create_playlist)
        self._exec_worker.log_line.connect(self._on_log_line)
        self._exec_worker.finished.connect(self._on_execute_finished)
        self._exec_worker.start()

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

    @Slot(object, object)
    def _on_incoming_selected(self, current, previous) -> None:  # noqa: ANN001
        if current:
            self._inspector.setPlainText(
                f"Track: {current.text(0)}\n"
                f"Status: {current.text(1)}\n"
                f"Distance: {current.text(2)}"
            )

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self._log_area.appendPlainText(line)

    @Slot(bool, str, object)
    def _on_scan_finished(self, success: bool, message: str, result) -> None:  # noqa: ANN001
        self._scan_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._state_lbl.setText(f"State: {'Complete' if success else 'Error'}")

        if success:
            self._scan_result = result
            # Store match objects for review flag resolution
            if isinstance(result, dict) and "match_objects" in result:
                self._current_match_results = result["match_objects"]
            self._exist_bar.setValue(100)
            self._incoming_bar.setValue(100)
            self._exist_status.setText("Done")
            self._incoming_status.setText("Done")
            self._log(message, "ok")
            self.status_changed.emit("Scan complete", "#22c55e")
            self._build_plan_btn.setEnabled(True)
            self._match_summary_lbl.setText(message)
            self._populate_results(result)
        else:
            self._log(message, "error" if message != "Cancelled." else "warn")
            self.status_changed.emit("Error", "#ef4444")

        self._worker = None

    @Slot(bool, str)
    def _on_build_finished(self, success: bool, message: str) -> None:
        self._cancel_btn.setEnabled(False)
        self._build_plan_btn.setEnabled(True)
        self._plan_status_lbl.setText(message)

        if success:
            self._log(message, "ok")
            self.status_changed.emit("Plan ready", "#22c55e")
            self._plan_bar.setValue(100)
            self._preview_plan_btn.setEnabled(True)
            self._execute_plan_btn.setEnabled(True)
            # Auto-open preview
            if self._build_worker and self._build_worker.preview_html_path:
                html = Path(self._build_worker.preview_html_path)
                if html.exists():
                    webbrowser.open(f"file://{html}")
        else:
            self._log(message, "error" if message != "Cancelled." else "warn")
            self.status_changed.emit("Error", "#ef4444")

    @Slot(bool, str, str)
    def _on_execute_finished(self, success: bool, message: str, report_path: str) -> None:
        self._cancel_btn.setEnabled(False)
        self._build_plan_btn.setEnabled(True)
        self._execute_plan_btn.setEnabled(True)
        self._plan_status_lbl.setText(message)

        if success:
            self._log(message, "ok")
            self.status_changed.emit("Execution complete", "#22c55e")
            self._plan_bar.setValue(100)
            if report_path and Path(report_path).exists():
                webbrowser.open(f"file://{report_path}")
        else:
            self._log(message, "error")
            self.status_changed.emit("Error", "#ef4444")

        self._exec_worker = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _populate_results(self, result) -> None:  # noqa: ANN001
        """Populate incoming tracks table from compare_libraries result dict."""
        if not isinstance(result, dict):
            return

        self._incoming_table.clear()
        self._existing_table.clear()

        # Use match_objects if available (preferred for track_id access)
        if "match_objects" in result and isinstance(result["match_objects"], list):
            for match in result["match_objects"]:
                path = match.incoming.path
                status_label = str(match.status).split(".")[-1] if hasattr(match.status, "name") else str(match.status)
                dist_str = f"{match.distance:.3f}" if match.distance is not None else "—"
                row = QtWidgets.QTreeWidgetItem([Path(path).name, status_label, dist_str, "", ""])
                row.setToolTip(0, path)
                # Store track_id for context menu and flag lookup
                row.setData(0, QtCore.Qt.ItemDataRole.UserRole, match.incoming.track_id)
                self._incoming_table.addTopLevelItem(row)
        else:
            # Fallback to old behavior if match_objects not available
            status_map = {
                "new": "New",
                "upgrade_candidates": "Upgrade",
                "collisions": "Collision",
                "existing_only": "Existing only",
                "matched": "Matched",
            }
            for key, label in status_map.items():
                entries = result.get(key, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    path = str(entry) if isinstance(entry, str) else str(getattr(entry, "incoming_path", entry))
                    dist = getattr(entry, "distance", None)
                    dist_str = f"{dist:.3f}" if dist is not None else "—"
                    row = QtWidgets.QTreeWidgetItem([Path(path).name, label, dist_str, "", ""])
                    row.setToolTip(0, path)
                    self._incoming_table.addTopLevelItem(row)

    def _browse_folder(self, entry: QtWidgets.QLineEdit, title: str) -> None:
        start = entry.text() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, title, start)
        if folder:
            entry.setText(folder)

    def _on_library_changed(self, path: str) -> None:
        self._existing_entry.setText(path)

    def _on_incoming_context_menu(self, pos: QtCore.QPoint) -> None:
        """Right-click context menu for incoming track flagging."""
        item = self._incoming_table.itemAt(pos)
        if not item:
            return

        track_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not track_id:
            return

        menu = QtWidgets.QMenu()
        menu.addAction("📋 Copy", lambda: self._flag_for_copy(track_id))
        menu.addAction("↻ Replace", lambda: self._flag_for_replace(track_id))
        menu.addAction("✕ Clear flag", lambda: self._flag_clear(track_id))
        menu.addSeparator()
        menu.addAction("📝 Add note", lambda: self._edit_note(track_id))
        menu.exec(self._incoming_table.mapToGlobal(pos))

    def _flag_for_copy(self, track_id: str) -> None:
        """Mark track for copy, remove any replace flag."""
        self._review_state_store.flag_for_copy(track_id)
        self._update_incoming_item_flags(track_id)
        self._log(f"Flagged {track_id} for copy", "info")

    def _flag_for_replace(self, track_id: str) -> None:
        """Mark track to replace existing counterpart."""
        # Find the matching result for validation
        match = next((r for r in self._current_match_results
                      if r.incoming.track_id == track_id), None)
        if not match or not match.existing:
            QtWidgets.QMessageBox.warning(self, "Cannot Replace",
                "No existing match found for this track.")
            return

        self._review_state_store.flag_for_replace(match)
        self._update_incoming_item_flags(track_id)
        self._log(f"Flagged {track_id} to replace {match.existing.track_id}", "info")

    def _flag_clear(self, track_id: str) -> None:
        """Clear both copy and replace flags."""
        self._review_state_store.unflag_copy(track_id)
        # Clear replace flag by removing from the replace dict
        if track_id in self._review_state_store.flags.replace:
            del self._review_state_store.flags.replace[track_id]
        self._update_incoming_item_flags(track_id)
        self._log(f"Cleared flags for {track_id}", "info")

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
