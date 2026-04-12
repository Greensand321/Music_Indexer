"""Tag Fixer workspace — repair metadata using AcoustID / MusicBrainz."""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase

# Field names the backend understands, mapped from UI labels
_FIELD_MAP = {
    "Title": "title",
    "Artist": "artist",
    "Album": "album",
    "Genre": "genres",
}


class TagFixWorker(QtCore.QThread):
    progress = Signal(int)
    log_line = Signal(str)
    proposal_ready = Signal(list)   # list of FileRecord
    finished = Signal(bool, str)

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self.library_path = library_path
        self._cancelled = False
        # Populated during run(); read by workspace after finished
        self.db_path: str = ""
        self.all_records: list = []

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from controllers import tagfix_controller
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.log_line.emit(msg)

            self.db_path, _ = tagfix_controller.prepare_library(self.library_path)
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
                return

            records = tagfix_controller.gather_records(
                self.library_path,
                self.db_path,
                show_all=True,
                progress_callback=lambda pct: self.progress.emit(int(pct)),
                log_callback=_log,
            )
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
                return

            self.all_records = records
            proposals = [r for r in records if r.status != "no_diff"]
            self.proposal_ready.emit(proposals)
            self.finished.emit(True, f"{len(proposals)} proposal(s) ready.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class TagFixApplyWorker(QtCore.QThread):
    log_line = Signal(str)
    finished = Signal(bool, str)

    def __init__(
        self,
        selected: list,
        all_records: list,
        db_path: str,
        fields: list[str],
    ) -> None:
        super().__init__()
        self.selected = selected
        self.all_records = all_records
        self.db_path = db_path
        self.fields = fields

    def run(self) -> None:
        try:
            from controllers.tagfix_controller import apply_proposals
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            count = apply_proposals(
                self.selected,
                self.all_records,
                self.db_path,
                self.fields,
                log_callback=lambda msg: self.log_line.emit(msg),
            )
            self.finished.emit(True, f"Applied tags to {count} file(s).")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class TagFixerWorkspace(WorkspaceBase):
    """Repair metadata tags using AcoustID fingerprint lookups."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._proposals: list = []
        self._all_records: list = []
        self._db_path: str = ""
        self._item_to_record: dict = {}   # id(QTreeWidgetItem) → FileRecord
        self._worker: TagFixWorker | None = None
        self._apply_worker: TagFixApplyWorker | None = None
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
        opts_row.addWidget(self._dry_run_cb)
        opts_row.addStretch(1)
        opt_layout.addLayout(opts_row)

        # Field selection — store as dict so _on_apply can read them
        fields_row = QtWidgets.QHBoxLayout()
        fields_row.addWidget(QtWidgets.QLabel("Fix fields:"))
        self._field_cbs: dict[str, QtWidgets.QCheckBox] = {}
        for field in ("Title", "Artist", "Album", "Genre"):
            cb = QtWidgets.QCheckBox(field)
            cb.setChecked(True)
            fields_row.addWidget(cb)
            self._field_cbs[field] = cb
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
            "✓", "File", "Field", "Current Value", "Proposed Value", "Score"
        ])
        self._table.setColumnWidth(0, 28)
        self._table.setColumnWidth(1, 240)
        self._table.setColumnWidth(2, 70)
        self._table.setColumnWidth(3, 180)
        self._table.setColumnWidth(4, 180)
        self._table.setColumnWidth(5, 55)
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
        self._item_to_record.clear()
        self._log_area.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Scanning…")
        self._scan_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._select_all_btn.setEnabled(False)
        self._deselect_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._log("Starting tag scan…", "info")
        self.status_changed.emit("Scanning tags…", "#f59e0b")

        self._worker = TagFixWorker(self._library_path)
        self._worker.progress.connect(self._prog_bar.setValue)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.proposal_ready.connect(self._on_proposals)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.start()

    @Slot()
    def _on_apply(self) -> None:
        if self._dry_run_cb.isChecked():
            QtWidgets.QMessageBox.information(
                self, "Dry Run",
                "Dry run is enabled — uncheck it to write tags."
            )
            return

        selected_set: set = set()
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            if item and item.checkState(0) == QtCore.Qt.CheckState.Checked:
                rec = self._item_to_record.get(id(item))
                if rec is not None:
                    selected_set.add(rec)

        if not selected_set:
            QtWidgets.QMessageBox.information(self, "Nothing Selected", "No proposals are checked.")
            return

        fields = [
            _FIELD_MAP[label]
            for label, cb in self._field_cbs.items()
            if cb.isChecked() and label in _FIELD_MAP
        ]
        if not fields:
            QtWidgets.QMessageBox.warning(self, "No Fields", "Select at least one field to fix.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Apply Tags",
            f"Write tags to {len(selected_set)} file(s)?  This cannot be undone.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self._apply_btn.setEnabled(False)
        self._scan_btn.setEnabled(False)
        self._prog_status.setText("Applying…")
        self.status_changed.emit("Applying tags…", "#f59e0b")

        self._apply_worker = TagFixApplyWorker(
            list(selected_set), self._all_records, self._db_path, fields
        )
        self._apply_worker.log_line.connect(self._on_log_line)
        self._apply_worker.finished.connect(self._on_apply_finished)
        self._apply_worker.start()

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
        self._item_to_record.clear()
        self._table.clear()

        row_count = 0
        for rec in proposals:
            # Build one row per changed field
            changed: list[tuple[str, str, str]] = []
            if rec.old_artist != rec.new_artist:
                changed.append(("artist", rec.old_artist or "", rec.new_artist or ""))
            if rec.old_title != rec.new_title:
                changed.append(("title", rec.old_title or "", rec.new_title or ""))
            if rec.old_album != rec.new_album:
                changed.append(("album", rec.old_album or "", rec.new_album or ""))
            if sorted(rec.old_genres or []) != sorted(rec.new_genres or []):
                changed.append((
                    "genres",
                    ", ".join(rec.old_genres or []),
                    ", ".join(rec.new_genres or []),
                ))

            score_str = f"{rec.score:.2f}" if rec.score is not None else "?"
            for field_name, old_val, new_val in changed:
                row = QtWidgets.QTreeWidgetItem([
                    "",
                    Path(rec.path).name,
                    field_name,
                    old_val,
                    new_val,
                    score_str,
                ])
                row.setCheckState(0, QtCore.Qt.CheckState.Checked)
                row.setToolTip(1, str(rec.path))
                self._table.addTopLevelItem(row)
                self._item_to_record[id(row)] = rec
                row_count += 1

        self._summary_lbl.setText(f"{row_count} change(s) across {len(proposals)} file(s)")
        if row_count:
            self._apply_btn.setEnabled(True)
            self._select_all_btn.setEnabled(True)
            self._deselect_btn.setEnabled(True)

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        self._log_area.appendPlainText(line)

    @Slot(bool, str)
    def _on_scan_finished(self, success: bool, message: str) -> None:
        # Capture state from worker before it's cleared
        if self._worker:
            self._db_path = self._worker.db_path
            self._all_records = self._worker.all_records

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

    @Slot(bool, str)
    def _on_apply_finished(self, success: bool, message: str) -> None:
        self._scan_btn.setEnabled(True)
        self._apply_btn.setEnabled(True)
        self._prog_status.setText(message)
        if success:
            self._log(message, "ok")
            self.status_changed.emit("Tags applied", "#22c55e")
            # Re-scan to refresh proposals
            self._on_scan()
        else:
            self._log(message, "error")
            self.status_changed.emit("Error", "#ef4444")
        self._apply_worker = None
