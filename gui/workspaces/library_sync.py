"""Library Sync workspace — compare two libraries and transfer files."""
from __future__ import annotations

import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


# ── Worker ────────────────────────────────────────────────────────────────────

class SyncScanWorker(QtCore.QThread):
    progress = Signal(int, str)    # percent, side ("existing" | "incoming")
    log_line = Signal(str)
    finished = Signal(bool, str, object)  # success, message, result

    def __init__(self, existing: str, incoming: str, thresholds: dict) -> None:
        super().__init__()
        self.existing = existing
        self.incoming = incoming
        self.thresholds = thresholds
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import library_sync
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}", None)
            return
        try:
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.log_line.emit(msg)

            result = library_sync.compare_libraries(
                self.existing,
                self.incoming,
                thresholds=self.thresholds,
                log_callback=_log,
            )
            if self._cancelled:
                self.finished.emit(False, "Cancelled.", None)
            else:
                n = len(result) if hasattr(result, "__len__") else 0
                self.finished.emit(True, f"Scan complete — {n} match records.", result)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc), None)


# ── Workspace ─────────────────────────────────────────────────────────────────

class LibrarySyncWorkspace(WorkspaceBase):
    """Library Sync — scan two folders, build plan, preview and execute."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._scan_result = None
        self._worker: SyncScanWorker | None = None
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
        self._incoming_table.setHeaderLabels(["Track", "Status", "Distance"])
        self._incoming_table.setColumnWidth(0, 260)
        self._incoming_table.setColumnWidth(1, 100)
        self._incoming_table.setMinimumHeight(200)
        self._incoming_table.setAlternatingRowColors(True)
        self._incoming_table.currentItemChanged.connect(self._on_incoming_selected)
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
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._state_lbl.setText("State: Cancelling…")

    @Slot()
    def _on_recompute(self) -> None:
        self._log("Recompute not yet wired — re-scan to apply new thresholds.", "warn")

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
        self._plan_status_lbl.setText("Building plan…")
        self._log("Build plan is not yet fully wired to backend.", "warn")
        self._preview_plan_btn.setEnabled(True)

    @Slot()
    def _on_preview_plan(self) -> None:
        html = Path(self._existing_entry.text()) / "Docs" / "sync_preview.html"
        if html.exists():
            webbrowser.open(f"file://{html}")
        else:
            self._log("No preview HTML found — run Build Plan first.", "warn")

    @Slot()
    def _on_execute_plan(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Execute",
            "This will transfer files to the existing library.\n\nProceed?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._log("Execute plan not yet fully wired to backend.", "warn")

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

    # ── Helpers ───────────────────────────────────────────────────────────

    def _populate_results(self, result) -> None:  # noqa: ANN001
        if result is None:
            return
        items = result if hasattr(result, "__iter__") else []
        for item in items:
            if hasattr(item, "incoming_path"):
                row = QtWidgets.QTreeWidgetItem([
                    Path(item.incoming_path).name,
                    str(getattr(item, "status", "?")),
                    f"{getattr(item, 'distance', '?'):.3f}" if hasattr(item, "distance") else "?",
                ])
                self._incoming_table.addTopLevelItem(row)

    def _browse_folder(self, entry: QtWidgets.QLineEdit, title: str) -> None:
        start = entry.text() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, title, start)
        if folder:
            entry.setText(folder)

    def _on_library_changed(self, path: str) -> None:
        self._existing_entry.setText(path)
