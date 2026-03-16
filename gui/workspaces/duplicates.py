"""Duplicate Finder workspace — scan → preview → execute."""
from __future__ import annotations

import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


# ── Worker ────────────────────────────────────────────────────────────────────

class DupeScanWorker(QtCore.QThread):
    progress = Signal(int)
    log_line = Signal(str)
    group_found = Signal(dict)       # one duplicate group dict
    finished = Signal(bool, str, list)  # success, message, groups

    def __init__(self, library_path: str, exact_threshold: float,
                 near_threshold: float, codec_boost: float) -> None:
        super().__init__()
        self.library_path = library_path
        self.exact_threshold = exact_threshold
        self.near_threshold = near_threshold
        self.codec_boost = codec_boost
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from near_duplicate_detector import find_near_duplicates
            from config import load_config
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}", [])
            return

        try:
            cfg = load_config()
            cfg["near_duplicate_threshold"] = self.near_threshold
            cfg["exact_duplicate_threshold"] = self.exact_threshold

            groups = []
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.log_line.emit(msg)

            result = find_near_duplicates(
                self.library_path,
                log_callback=_log,
            )
            groups = list(result.groups) if hasattr(result, "groups") else []
            if self._cancelled:
                self.finished.emit(False, "Cancelled.", [])
            else:
                self.finished.emit(True, f"Found {len(groups)} duplicate group(s).", groups)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc), [])


# ── Workspace ─────────────────────────────────────────────────────────────────

class DuplicatesWorkspace(WorkspaceBase):
    """Duplicate Finder — scan, review groups, preview and execute."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._groups: list = []
        self._worker: DupeScanWorker | None = None
        self._preview_path: str = ""
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
        self._artwork_cb = QtWidgets.QCheckBox("Show groups differing only by artwork")
        self._noop_cb = QtWidgets.QCheckBox("Show no-op groups")
        self._playlists_cb = QtWidgets.QCheckBox("Update playlists after execution")
        opts_row.addWidget(self._artwork_cb)
        opts_row.addWidget(self._noop_cb)
        opts_row.addWidget(self._playlists_cb)
        opts_row.addStretch(1)
        cfg_layout.addLayout(opts_row)

        action_row = QtWidgets.QHBoxLayout()
        self._quarantine_rb = QtWidgets.QRadioButton("Quarantine duplicates (default)")
        self._quarantine_rb.setChecked(True)
        self._delete_rb = QtWidgets.QRadioButton("Delete losers permanently")
        self._delete_rb.setToolTip("Irreversible — requires confirmation before execution.")
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
        fp_status = QtWidgets.QLabel("")
        fp_status.setObjectName("statusHint")
        self._fp_status_lbl = fp_status
        left.addWidget(fp_status)

        self._groups_table = QtWidgets.QTreeWidget()
        self._groups_table.setHeaderLabels(["Group", "Title", "Count", "Status"])
        self._groups_table.setColumnWidth(0, 60)
        self._groups_table.setColumnWidth(1, 200)
        self._groups_table.setColumnWidth(2, 50)
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
        self._log("Starting duplicate scan…", "info")
        self.status_changed.emit("Scanning…", "#f59e0b")

        self._worker = DupeScanWorker(
            library_path=self._library_path,
            exact_threshold=self._exact_spin.value(),
            near_threshold=self._near_spin.value(),
            codec_boost=self._boost_spin.value(),
        )
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    @Slot()
    def _on_preview(self) -> None:
        """Write duplicate_preview.html and open it."""
        if not self._groups:
            return
        try:
            import json
            from pathlib import Path
            docs = Path(self._library_path) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            preview_json = docs / "duplicate_preview.json"
            # Write a basic JSON preview (the real one is generated by the backend)
            preview_json.write_text(
                json.dumps({"groups": len(self._groups), "library": self._library_path}, indent=2)
            )
            self._log("Preview written to Docs/duplicate_preview.json", "ok")
            self._execute_btn.setEnabled(True)
            # Try to open HTML preview if it exists
            html = docs / "duplicate_preview.html"
            if html.exists():
                webbrowser.open(f"file://{html}")
        except Exception as exc:  # noqa: BLE001
            self._log(str(exc), "error")

    @Slot()
    def _on_execute(self) -> None:
        action = "delete" if self._delete_rb.isChecked() else "quarantine"
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Execute",
            f"This will {action} duplicate files in your library.\n\n"
            "Have you reviewed the preview first?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._log("Execute not yet wired to backend — review groups and run from CLI.", "warn")

    @Slot()
    def _on_open_report(self) -> None:
        candidate = Path(self._library_path) / "Docs" / "duplicate_preview.html"
        if candidate.exists():
            webbrowser.open(f"file://{candidate}")
        else:
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
            self._group_details.setPlainText(f"Group: {current.text(0)}\n(Details not available)")

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self._log_area.appendPlainText(line)

    @Slot(bool, str, list)
    def _on_scan_finished(self, success: bool, message: str, groups: list) -> None:
        self._scan_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._prog_status.setText(message)

        if success:
            self._groups = groups
            self._log(message, "ok")
            self.status_changed.emit("Scan complete", "#22c55e")
            self._prog_bar.setValue(100)
            self._populate_groups(groups)
            if groups:
                self._preview_btn.setEnabled(True)
        else:
            self._log(message, "error" if message != "Cancelled." else "warn")
            self.status_changed.emit("Error" if message != "Cancelled." else "Idle", "#ef4444")

        self._worker = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _populate_groups(self, groups: list) -> None:
        self._groups_table.clear()
        for i, group in enumerate(groups, 1):
            if hasattr(group, "tracks"):
                tracks = group.tracks
                title = tracks[0] if tracks else "—"
                count = str(len(tracks))
            elif isinstance(group, dict):
                tracks = group.get("tracks", [])
                title = str(tracks[0]) if tracks else group.get("title", "—")
                count = str(len(tracks))
            else:
                title = str(group)
                count = "?"
            item = QtWidgets.QTreeWidgetItem([str(i), title[:80], count, "Pending"])
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
        self._scan_btn.setEnabled(True)
