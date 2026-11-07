"""Tag Fixer workspace — repair metadata using AcoustID / MusicBrainz.

Flow: **Scan** (lookups only, never writes) → review grid with one row per
proposed *field* change → tick exactly the changes you want → **Apply
Selected** → confirm → exactly the ticked changes are written, each one
logged with its before/after value in ``Docs/tag_changes_tagfixer_*.jsonl``.

The grid is built from ``tag_fixer.planned_changes`` — the same function the
writer uses — so a row always shows precisely what applying it writes.
Files the lookups could not identify are listed separately ("No match"),
with the reason, instead of appearing as blank proposals.
"""
from __future__ import annotations

import dataclasses
import os
import threading
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

# Field names the backend understands, mapped from UI labels
_FIELD_MAP = {
    "Title": "title",
    "Artist": "artist",
    "Album": "album",
    "Genre": "genres",
}
_FIELD_LABELS = {v: k for k, v in _FIELD_MAP.items()}

_ROW_ROLE = QtCore.Qt.ItemDataRole.UserRole
_SCORE_COL = 5
_WRITE_OWNER = "Tag Fixer apply"


def _fmt(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value) if value else "(none)"
    return "(empty)" if value in (None, "") else str(value)


# Note: no QTreeWidgetItem subclass with ``__lt__`` for numeric sorting —
# PySide6 6.12 segfaults when such an item is re-sorted after a check-state
# change. Scores are fixed two-decimal values in [0, 1], which sort correctly
# as text now that unmatched files (formerly "?") are not rows.


class TagFixWorker(QtCore.QThread):
    progress = Signal(int)          # percent, 0-100
    log_line = Signal(str)
    proposal_ready = Signal(list)   # every FileRecord of the scan
    finished = Signal(bool, str)

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self.library_path = library_path
        self._cancel = threading.Event()
        # Populated during run(); read by workspace after finished
        self.db_path: str = ""
        self.all_records: list = []
        self.diagnostics = None

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            from controllers import tagfix_controller
            from tag_fixer import ScanDiagnostics
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            def _log(msg: str) -> None:
                if not self.is_cancelled():
                    self.log_line.emit(msg)

            self.db_path, _ = tagfix_controller.prepare_library(self.library_path)
            if self.is_cancelled():
                self.finished.emit(False, "Cancelled.")
                return

            self.diagnostics = ScanDiagnostics()
            records = tagfix_controller.gather_records(
                self.library_path,
                self.db_path,
                show_all=True,
                progress_callback=None,
                log_callback=_log,
                # A real percentage (it used to send the file *index* to a
                # 0-100 bar, which sat at 100% from file 100 onwards).
                progress_detail=lambda i, t: self.progress.emit(int(i * 100 / max(t, 1))),
                should_cancel=self.is_cancelled,
                diagnostics=self.diagnostics,
            )
            if self.is_cancelled() or self.diagnostics.cancelled:
                self.finished.emit(False, "Cancelled.")
                return

            self.all_records = records
            self.proposal_ready.emit(records)
            self.finished.emit(True, "Scan complete.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class TagFixApplyWorker(QtCore.QThread):
    log_line = Signal(str)
    progress = Signal(int)
    result_ready = Signal(object)   # tag_fixer.TagApplyResult
    finished = Signal(bool, str)

    def __init__(
        self,
        selections: list,
        all_records: list,
        db_path: str,
        library_root: str,
    ) -> None:
        super().__init__()
        self.selections = selections      # [(FileRecord, [field, ...]), ...]
        self.all_records = all_records
        self.db_path = db_path
        self.library_root = library_root
        self._cancel = threading.Event()
        self.result = None

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set() or self.isInterruptionRequested()

    def run(self) -> None:
        from controllers import tag_write_guard

        try:
            try:
                from controllers.tagfix_controller import apply_selected_fields
            except ImportError as exc:
                self.finished.emit(False, f"Import error: {exc}")
                return
            try:
                result = apply_selected_fields(
                    self.selections,
                    self.all_records,
                    self.db_path,
                    log_callback=self.log_line.emit,
                    should_cancel=self.is_cancelled,
                    progress_callback=lambda i, t: self.progress.emit(int(i * 100 / max(t, 1))),
                    library_root=self.library_root,
                )
            except Exception as exc:  # noqa: BLE001
                self.finished.emit(False, str(exc))
                return
            self.result = result
            self.result_ready.emit(result)
            parts = [f"Applied tags to {len(result.applied)} file(s)"]
            if result.failed:
                parts.append(f"{len(result.failed)} failed")
            if result.skipped:
                parts.append(f"{len(result.skipped)} skipped (changed since the scan)")
            message = ", ".join(parts)
            if result.cancelled:
                message += " — cancelled"
            self.finished.emit(True, message + ".")
        finally:
            # Released here, not in a GUI slot, so the claim can never be
            # left behind if the page is gone before the slot runs.
            tag_write_guard.release(_WRITE_OWNER)


class TagFixerWorkspace(WorkspaceBase):
    """Repair metadata tags using AcoustID fingerprint lookups."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._records: list = []          # every FileRecord of the last scan
        self._all_records: list = []      # (kept name) same list, for the DB update
        self._proposals: list = []        # records with at least one planned change
        self._rows: list = []             # row index -> (FileRecord, field)
        self._db_path: str = ""
        self._scan_library: str = ""      # library the shown proposals belong to
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
        opt_layout.addWidget(self._make_card_title("Options"))

        opts_row = QtWidgets.QHBoxLayout()
        self._dry_run_cb = QtWidgets.QCheckBox("Dry run (propose only, do not write tags)")
        self._dry_run_cb.setChecked(True)
        opts_row.addWidget(self._dry_run_cb)
        opts_row.addStretch(1)
        opt_layout.addLayout(opts_row)

        # Field filter: shows/hides that field's rows. Apply writes exactly
        # the rows that are visible *and* ticked.
        fields_row = QtWidgets.QHBoxLayout()
        fields_row.addWidget(QtWidgets.QLabel("Fix fields:"))
        self._field_cbs: dict[str, QtWidgets.QCheckBox] = {}
        for field in ("Title", "Artist", "Album", "Genre"):
            cb = QtWidgets.QCheckBox(field)
            cb.setChecked(True)
            cb.setToolTip(f"Show {field.lower()} proposals; hidden rows are never written.")
            cb.toggled.connect(self._apply_field_filter)
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
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setFixedHeight(6)
        self._prog_status = QtWidgets.QLabel("Idle")
        self._prog_status.setObjectName("statusHint")
        cl.addWidget(self._prog_bar)
        cl.addWidget(self._prog_status)

        # ── Proposals table ────────────────────────────────────────────────
        table_card = self._make_card()
        table_layout = QtWidgets.QVBoxLayout(table_card)
        table_layout.setContentsMargins(16, 16, 16, 16)
        table_layout.addWidget(self._make_card_title("Tag Proposals"))

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
        self._table.itemChanged.connect(self._on_item_changed)
        table_layout.addWidget(self._table)

        self._summary_lbl = QtWidgets.QLabel("")
        self._summary_lbl.setObjectName("statusHint")
        self._summary_lbl.setWordWrap(True)
        table_layout.addWidget(self._summary_lbl)

        # Why things did or didn't match: active sources, and the reasons.
        self._reason_lbl = QtWidgets.QLabel("")
        self._reason_lbl.setObjectName("statusHint")
        self._reason_lbl.setWordWrap(True)
        table_layout.addWidget(self._reason_lbl)

        # Files with no match are listed here, never as blank proposals.
        self._nomatch_title = QtWidgets.QLabel("No match found")
        self._nomatch_title.setObjectName("cardTitle")
        self._nomatch_list = QtWidgets.QListWidget()
        self._nomatch_list.setMaximumHeight(120)
        self._nomatch_title.setVisible(False)
        self._nomatch_list.setVisible(False)
        table_layout.addWidget(self._nomatch_title)
        table_layout.addWidget(self._nomatch_list)
        cl.addWidget(table_card)

        # ── Log ────────────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(self._make_card_title("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(100)
        self._log_area.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _emit_status(self, text: str, role: str) -> None:
        tokens = get_manager().current
        self.status_changed.emit(text, getattr(tokens, role))

    def _busy(self) -> bool:
        return self._worker is not None or self._apply_worker is not None

    def _update_buttons(self) -> None:
        busy = self._busy()
        has_rows = self._table.topLevelItemCount() > 0
        self._scan_btn.setEnabled(not busy)
        self._apply_btn.setEnabled(has_rows and not busy)
        self._select_all_btn.setEnabled(has_rows and not busy)
        self._deselect_btn.setEnabled(has_rows and not busy)
        self._cancel_btn.setEnabled(busy)

    def _clear_results(self) -> None:
        self._table.clear()
        self._rows = []
        self._records = []
        self._all_records = []
        self._proposals = []
        self._db_path = ""
        self._scan_library = ""
        self._summary_lbl.setText("")
        self._reason_lbl.setText("")
        self._nomatch_list.clear()
        self._nomatch_title.setVisible(False)
        self._nomatch_list.setVisible(False)

    def _selected_changes(self) -> list:
        """``[(FileRecord, [field, ...])]`` for every visible, ticked row.

        Keyed by the record's identity: ``FileRecord`` is an ``eq=True``
        dataclass and therefore unhashable (putting it in a set raised
        ``TypeError`` and Apply could never write anything).
        """
        by_record: dict[int, tuple] = {}
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            if item is None or item.isHidden():
                continue
            if item.checkState(0) != QtCore.Qt.CheckState.Checked:
                continue
            index = item.data(0, _ROW_ROLE)
            if index is None or not (0 <= int(index) < len(self._rows)):
                continue
            rec, field = self._rows[int(index)]
            by_record.setdefault(id(rec), (rec, []))[1].append(field)
        return list(by_record.values())

    def _render_rows(self, keep_unticked: set | None = None) -> None:
        """(Re)build the grid from ``self._records`` via planned_changes."""
        from tag_fixer import planned_changes

        keep_unticked = keep_unticked or set()
        self._table.blockSignals(True)
        sorting = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        self._table.clear()
        self._rows = []
        self._proposals = []
        unmatched = []
        for rec in self._records:
            if rec.status == "unmatched":
                unmatched.append(rec)
                continue
            changes = planned_changes(rec)
            if not changes:
                continue
            self._proposals.append(rec)
            score_str = f"{rec.score:.2f}" if rec.score is not None else ""
            for field_name, (before, after) in changes.items():
                row = QtWidgets.QTreeWidgetItem([
                    "",
                    Path(rec.path).name,
                    _FIELD_LABELS.get(field_name, field_name),
                    _fmt(before),
                    _fmt(after),
                    score_str,
                ])
                index = len(self._rows)
                self._rows.append((rec, field_name))
                row.setData(0, _ROW_ROLE, index)
                row.setData(_SCORE_COL, _ROW_ROLE, float(rec.score or 0))
                ticked = (id(rec), field_name) not in keep_unticked
                row.setCheckState(
                    0, QtCore.Qt.CheckState.Checked if ticked else QtCore.Qt.CheckState.Unchecked
                )
                row.setToolTip(1, str(rec.path))
                row.setToolTip(3, _fmt(before))
                row.setToolTip(4, _fmt(after))
                self._table.addTopLevelItem(row)
        self._table.setSortingEnabled(sorting)
        self._table.blockSignals(False)
        self._apply_field_filter()

        self._nomatch_list.clear()
        for rec in unmatched:
            item = QtWidgets.QListWidgetItem(
                f"{Path(rec.path).name} — {rec.note or 'no match found'}"
            )
            item.setToolTip(str(rec.path))
            self._nomatch_list.addItem(item)
        self._nomatch_title.setText(f"No match found ({len(unmatched)})")
        self._nomatch_title.setVisible(bool(unmatched))
        self._nomatch_list.setVisible(bool(unmatched))
        self._update_summary()

    def _update_summary(self) -> None:
        total_rows = self._table.topLevelItemCount()
        if not self._records:
            self._summary_lbl.setText("")
            return
        selected = self._selected_changes()
        n_fields = sum(len(f) for _r, f in selected)
        unmatched = sum(1 for r in self._records if r.status == "unmatched")
        text = (
            f"{total_rows} change(s) proposed across {len(self._proposals)} file(s) · "
            f"{n_fields} ticked across {len(selected)} file(s)"
        )
        if unmatched:
            text += f" · {unmatched} file(s) with no match"
        self._summary_lbl.setText(text)

    # ── Library change ────────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        # Proposals belong to the library they were scanned in; applying
        # them after a switch would write the *old* library's files. A scan
        # in flight is cancelled and its result discarded. A confirmed apply
        # is allowed to finish (it writes exactly what was confirmed) and its
        # outcome is logged.
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
            self._prog_bar.setValue(0)
        had_results = bool(self._records)
        self._clear_results()
        if self._apply_worker is None:
            self._prog_status.setText(
                "Library changed — scan again." if had_results else "Idle"
            )
        self._update_buttons()

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_scan(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        if self._busy():
            return

        self._clear_results()
        self._log_area.clear()
        self._prog_bar.setValue(0)
        self._prog_status.setText("Scanning…")
        self._scan_library = self._library_path
        self._log("Starting tag scan…", "info")
        self._emit_status("Scanning tags…", "warning")

        worker = TagFixWorker(self._library_path)
        worker.progress.connect(self._prog_bar.setValue)
        worker.log_line.connect(self._on_log_line)
        worker.proposal_ready.connect(self._on_proposals)
        worker.finished.connect(self._on_scan_finished)
        self._worker = worker
        self._update_buttons()
        self._start_worker(worker, "Tag Fixer scan")

    @Slot()
    def _on_apply(self) -> None:
        if self._dry_run_cb.isChecked():
            QtWidgets.QMessageBox.information(
                self, "Dry Run",
                "Dry run is enabled — uncheck it to write tags."
            )
            return
        if self._busy() or not self._db_path:
            return

        selections = self._selected_changes()
        if not selections:
            QtWidgets.QMessageBox.information(self, "Nothing Selected", "No proposals are checked.")
            return

        from controllers import tag_write_guard

        n_fields = sum(len(fields) for _rec, fields in selections)
        reply = QtWidgets.QMessageBox.question(
            self, "Apply Tags",
            f"Write {n_fields} ticked tag change(s) to {len(selections)} file(s), "
            "exactly as shown in the table?\n\n"
            "The files are edited in place. Every change is logged with its "
            "previous value in Docs/tag_changes_tagfixer_….jsonl.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if not tag_write_guard.acquire(_WRITE_OWNER):
            QtWidgets.QMessageBox.information(
                self, "Tags Busy", tag_write_guard.busy_message("Tag Fixer apply")
            )
            return

        self._prog_bar.setValue(0)
        self._prog_status.setText("Applying…")
        self._emit_status("Applying tags…", "warning")

        # The DB status pass rewrites ``status`` on the records it is given;
        # hand it copies so the worker thread never mutates the objects this
        # page displays (it turned "unmatched" into "skipped", emptying the
        # No-match list after an apply).
        status_copies = [dataclasses.replace(rec) for rec in self._records]
        worker = TagFixApplyWorker(
            selections, status_copies, self._db_path, self._scan_library
        )
        worker.log_line.connect(self._on_log_line)
        worker.progress.connect(self._prog_bar.setValue)
        worker.result_ready.connect(self._on_apply_result)
        worker.finished.connect(self._on_apply_finished)
        self._apply_worker = worker
        self._update_buttons()
        self._start_worker(worker, _WRITE_OWNER)

    @Slot()
    def _on_select_all(self) -> None:
        self._set_all_checked(QtCore.Qt.CheckState.Checked)

    @Slot()
    def _on_deselect_all(self) -> None:
        self._set_all_checked(QtCore.Qt.CheckState.Unchecked)

    def _set_all_checked(self, state: QtCore.Qt.CheckState) -> None:
        self._table.blockSignals(True)
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            if item and not item.isHidden():
                item.setCheckState(0, state)
        self._table.blockSignals(False)
        self._update_summary()

    def _apply_field_filter(self, *_args) -> None:
        shown = {_FIELD_MAP[label] for label, cb in self._field_cbs.items() if cb.isChecked()}
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            index = item.data(0, _ROW_ROLE) if item is not None else None
            if index is None or not (0 <= int(index) < len(self._rows)):
                continue
            _rec, field = self._rows[int(index)]
            item.setHidden(field not in shown)
        self._update_summary()

    @Slot(QtWidgets.QTreeWidgetItem, int)
    def _on_item_changed(self, _item, column: int) -> None:
        if column == 0:
            self._update_summary()

    @Slot()
    def _on_cancel(self) -> None:
        for worker in (self._worker, self._apply_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._prog_status.setText("Cancelling…")

    @Slot(list)
    def _on_proposals(self, records: list) -> None:
        if self.sender() is not self._worker:
            return  # a scan cancelled by a library change; its result is stale
        self._records = list(records)
        self._all_records = self._records
        self._render_rows()

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        sender = self.sender()
        if sender is not None and sender is not self._worker and sender is not self._apply_worker:
            return
        self._log_area.appendPlainText(line)

    @Slot(bool, str)
    def _on_scan_finished(self, success: bool, message: str) -> None:
        worker = self.sender()
        if worker is not self._worker:
            return  # stale (cancelled by a library change)
        self._worker = None
        self._db_path = worker.db_path if success else ""
        diag = getattr(worker, "diagnostics", None)
        if success:
            self._prog_bar.setValue(100)
            n_rows = self._table.topLevelItemCount()
            message = (
                f"{n_rows} change(s) proposed for {len(self._proposals)} file(s)."
                if n_rows else "No tag changes to propose."
            )
            self._log(message, "ok")
            self._emit_status("Scan complete", "success")
        else:
            self._clear_results()
            self._log(message, "error" if message != "Cancelled." else "warn")
            self._emit_status("Cancelled" if message == "Cancelled." else "Error",
                              "warning" if message == "Cancelled." else "danger")
        self._prog_status.setText(message)
        if diag is not None and success:
            lines = [diag.sources_line()]
            if diag.unmatched:
                lines.append(f"{diag.unmatched} file(s) unmatched: {diag.unmatched_reason()}")
            self._reason_lbl.setText("\n".join(lines))
        self._update_buttons()

    @Slot(object)
    def _on_apply_result(self, result) -> None:
        if self.sender() is not self._apply_worker:
            return
        worker = self._apply_worker
        if worker is None or worker.library_root != self._scan_library:
            return  # the library changed during the apply; the grid is gone
        # Update the records in place for every field that was written, so
        # those rows drop out; failed / unticked rows stay for review.
        unticked = set()
        for i in range(self._table.topLevelItemCount()):
            item = self._table.topLevelItem(i)
            index = item.data(0, _ROW_ROLE)
            if index is None or not (0 <= int(index) < len(self._rows)):
                continue
            rec, field = self._rows[int(index)]
            if item.checkState(0) != QtCore.Qt.CheckState.Checked:
                unticked.add((id(rec), field))
        from tag_fixer import planned_changes

        written = getattr(result, "written_fields", {}) or {}
        for rec in self._records:
            fields = written.get(str(rec.path))
            if not fields:
                continue
            for field, (_before, after) in planned_changes(rec, fields).items():
                if field == "genres":
                    rec.old_genres = list(after)
                else:
                    setattr(rec, f"old_{field}", after)
            if not planned_changes(rec):
                rec.status = "applied"
        self._render_rows(unticked)
        if result.failed:
            self._reason_lbl.setText(
                f"{len(result.failed)} file(s) could not be written — their rows are "
                "still listed; see the log for why."
            )

    @Slot(bool, str)
    def _on_apply_finished(self, success: bool, message: str) -> None:
        worker = self.sender()
        if worker is not self._apply_worker:
            return
        self._apply_worker = None
        result = getattr(worker, "result", None)
        log_path = getattr(result, "log_path", "") if result is not None else ""
        if log_path:
            message = f"{message} Change log: {os.path.relpath(log_path, worker.library_root)}"
        self._prog_status.setText(message)
        if worker.library_root != self._library_path:
            message = f"{message} (library: {worker.library_root})"
        if success:
            failed = bool(result and (result.failed or result.skipped))
            self._log(message, "warn" if failed else "ok")
            self._emit_status("Tags applied", "warning" if failed else "success")
        else:
            self._log(message, "error")
            self._emit_status("Error", "danger")
        # No automatic rescan: it cleared the log (hiding per-file failures)
        # and re-queried every file online. Written rows were updated above.
        self._update_buttons()
