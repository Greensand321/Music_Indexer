"""Tag Fixer workspace — repair metadata using AcoustID / MusicBrainz."""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class TagFixWorker(QtCore.QThread):
    progress = Signal(int)
    log_line = Signal(str)
    proposal_ready = Signal(list)   # list of (path, old_tags, new_tags)
    finished = Signal(bool, str)

    def __init__(self, library_path: str, dry_run: bool) -> None:
        super().__init__()
        self.library_path = library_path
        self.dry_run = dry_run
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import tag_fixer
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.log_line.emit(msg)

            proposals = tag_fixer.propose_tag_fixes(
                self.library_path,
                log_callback=_log,
            )
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
                return
            self.proposal_ready.emit(proposals or [])
            self.finished.emit(True, f"{len(proposals or [])} proposal(s) ready.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class TagFixerWorkspace(WorkspaceBase):
    """Repair metadata tags using AcoustID fingerprint lookups."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._proposals: list = []
        self._worker: TagFixWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Tag Fixer"))
        cl.addWidget(self._make_subtitle(
            "Scan your library, look up each track via AcoustID or MusicBrainz, "
            "and propose corrected tags. Review proposals in the table, "
            "approve the ones you want, and apply."
        ))

        # ── Options card ───────────────────────────────────────────────────
        opt_card = self._make_card()
        opt_layout = QtWidgets.QVBoxLayout(opt_card)
        opt_layout.setContentsMargins(16, 16, 16, 16)
        opt_layout.setSpacing(10)
        opt_layout.addWidget(QtWidgets.QLabel("Options"))

        opts_row = QtWidgets.QHBoxLayout()
        self._dry_run_cb = QtWidgets.QCheckBox("Dry run (propose only, do not write tags)")
        self._dry_run_cb.setChecked(True)
        self._overwrite_cb = QtWidgets.QCheckBox("Overwrite existing tags")
        self._acoustid_cb = QtWidgets.QCheckBox("Use AcoustID fingerprint lookup")
        self._acoustid_cb.setChecked(True)
        opts_row.addWidget(self._dry_run_cb)
        opts_row.addWidget(self._overwrite_cb)
        opts_row.addWidget(self._acoustid_cb)
        opts_row.addStretch(1)
        opt_layout.addLayout(opts_row)

        # Field selection
        fields_row = QtWidgets.QHBoxLayout()
        fields_row.addWidget(QtWidgets.QLabel("Fix fields:"))
        for field in ("Title", "Artist", "Album", "Year", "Genre", "Track"):
            cb = QtWidgets.QCheckBox(field)
            cb.setChecked(True)
            fields_row.addWidget(cb)
        fields_row.addStretch(1)
        opt_layout.addLayout(fields_row)
        cl.addWidget(opt_card)

        # ── Action buttons ─────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._scan_btn = self._make_primary_button("🏷  Scan for Tag Issues")
        self._scan_btn.clicked.connect(self._on_scan)

        self._apply_btn = QtWidgets.QPushButton("✓  Apply Selected")
        self._apply_btn.setObjectName("successBtn")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)

        self._select_all_btn = QtWidgets.QPushButton("Select All")
        self._select_all_btn.setEnabled(False)
        self._select_all_btn.clicked.connect(self._on_select_all)

        self._deselect_btn = QtWidgets.QPushButton("Deselect All")
        self._deselect_btn.setEnabled(False)
        self._deselect_btn.clicked.connect(self._on_deselect_all)

        self._cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)

        btn_row.addWidget(self._scan_btn)
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._select_all_btn)
        btn_row.addWidget(self._deselect_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        # ── Progress ───────────────────────────────────────────────────────
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setValue(0)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setFixedHeight(6)
        self._prog_status = QtWidgets.QLabel("Idle")
        self._prog_status.setStyleSheet("color: #64748b; font-size: 12px;")
        cl.addWidget(self._prog_bar)
        cl.addWidget(self._prog_status)

        # ── Proposals table ────────────────────────────────────────────────
        table_card = self._make_card()
        table_layout = QtWidgets.QVBoxLayout(table_card)
        table_layout.setContentsMargins(16, 16, 16, 16)
        table_layout.addWidget(QtWidgets.QLabel("Tag Proposals"))

        self._table = QtWidgets.QTreeWidget()
        self._table.setHeaderLabels([
            "✓", "File", "Field", "Current Value", "Proposed Value", "Source"
        ])
        self._table.setColumnWidth(0, 28)
        self._table.setColumnWidth(1, 240)
        self._table.setColumnWidth(2, 70)
        self._table.setColumnWidth(3, 180)
        self._table.setColumnWidth(4, 180)
        self._table.setColumnWidth(5, 80)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.setMinimumHeight(260)
        table_layout.addWidget(self._table)

        self._summary_lbl = QtWidgets.QLabel("")
        self._summary_lbl.setStyleSheet("color: #64748b; font-size: 12px;")
        table_layout.addWidget(self._summary_lbl)
        cl.addWidget(table_card)

        # ── Log ────────────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(QtWidgets.QLabel("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(100)
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

        self._table.clear()
        self._log_area.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Scanning…")
        self._scan_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._log("Starting tag scan…", "info")
        self.status_changed.emit("Scanning tags…", "#f59e0b")

        self._worker = TagFixWorker(self._library_path, self._dry_run_cb.isChecked())
        self._worker.log_line.connect(self._on_log_line)
        self._worker.proposal_ready.connect(self._on_proposals)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _on_apply(self) -> None:
        selected = []
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            if item and item.checkState(0) == QtCore.Qt.CheckState.Checked:
                selected.append(item)
        if not selected:
            return
        self._log(f"Applying {len(selected)} tag change(s)…", "info")
        # Real apply via tag_fixer.apply_proposals(selected)
        # Not fully wired yet — would need the proposal objects
        self._log("Tag apply not fully wired to backend yet.", "warn")

    @Slot()
    def _on_select_all(self) -> None:
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            if item:
                item.setCheckState(0, QtCore.Qt.CheckState.Checked)

    @Slot()
    def _on_deselect_all(self) -> None:
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            if item:
                item.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    @Slot(list)
    def _on_proposals(self, proposals: list) -> None:
        self._proposals = proposals
        self._table.clear()
        for prop in proposals:
            if isinstance(prop, dict):
                row = QtWidgets.QTreeWidgetItem([
                    "",
                    str(prop.get("path", "")),
                    str(prop.get("field", "")),
                    str(prop.get("old", "")),
                    str(prop.get("new", "")),
                    str(prop.get("source", "AcoustID")),
                ])
            else:
                row = QtWidgets.QTreeWidgetItem(["", str(prop), "", "", "", ""])
            row.setCheckState(0, QtCore.Qt.CheckState.Checked)
            self._table.addTopLevelItem(row)
        count = len(proposals)
        self._summary_lbl.setText(f"{count} proposal(s) found")
        if count:
            self._apply_btn.setEnabled(True)
            self._select_all_btn.setEnabled(True)
            self._deselect_btn.setEnabled(True)

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self._log_area.appendPlainText(line)

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str) -> None:
        self._scan_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._prog_status.setText(message)
        if success:
            self._prog_bar.setValue(100)
            self._log(message, "ok")
            self.status_changed.emit("Scan complete", "#22c55e")
        else:
            self._log(message, "error" if message != "Cancelled." else "warn")
            self.status_changed.emit("Error", "#ef4444")
        self._worker = None
