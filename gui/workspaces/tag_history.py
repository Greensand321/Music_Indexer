"""Metadata → Undo changes: roll back a Tag Fixer or genre apply.

Every tag write the app makes is logged in ``Docs/tag_changes_<tool>_<ts>.jsonl``
with each field's value before and after (``utils.tag_change_log``). This
panel turns one of those logs back into a write, preview-first like every
other file-changing workflow (``controllers.tag_revert``):

1. the list shows the library's change logs that wrote something, newest
   first, and whether each was already undone;
2. **Preview undo** (read-only job) lists, per file and field, what the file
   holds now and what it gets back. Files changed since the apply are left
   alone and listed as conflicts; files that moved are listed as missing;
3. **Undo these changes** asks for confirmation and writes exactly the
   preview, re-checking each file just before writing. The undo is logged
   too (``tag_changes_revert_*.jsonl``), so an undo can itself be undone.

The apply holds ``controllers.tag_write_guard`` (one tag-writing job at a
time) and the library-write lock.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

_WRITE_OWNER = "Undo tag changes"
_TOOL_LABELS = {
    "tagfixer": "Tag Fixer",
    "genre_fill": "Fill genres",
    "genre_normalize": "Normalize genres",
    "revert": "Undo",
}
_LOG_ROLE = QtCore.Qt.ItemDataRole.UserRole


def _tool_label(tool: str) -> str:
    return _TOOL_LABELS.get(tool, tool.replace("_", " ").title() or "—")


def _when(stamp: str) -> str:
    try:
        return datetime.strptime(stamp, "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return stamp or "—"


def _fmt(values) -> str:
    if not values:
        return "(no tag)"
    return "; ".join(str(v) for v in values)


class RevertPlanWorker(QtCore.QThread):
    """Reads only: works out what undoing one change log would write."""

    progress = Signal(int)
    plan_ready = Signal(object)        # controllers.tag_revert.RevertPlan
    finished = Signal(bool, str)

    def __init__(self, log_path: str) -> None:
        super().__init__()
        self.log_path = log_path
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            from controllers.tag_revert import plan_revert

            plan = plan_revert(
                self.log_path,
                progress_callback=lambda i, t: self.progress.emit(int(i * 100 / max(t, 1))),
                cancel_event=self._cancel,
            )
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc) or type(exc).__name__)
            return
        if plan.cancelled or self._cancel.is_set():
            self.finished.emit(False, "Cancelled.")
            return
        self.plan_ready.emit(plan)
        self.finished.emit(True, "")


class RevertApplyWorker(QtCore.QThread):
    """Writes exactly a previewed revert plan (holds the tag-write guard)."""

    progress = Signal(int)
    result_ready = Signal(object)      # controllers.tag_revert.RevertResult
    finished = Signal(bool, str)

    def __init__(self, plan) -> None:
        super().__init__()
        self.plan = plan
        self._cancel = threading.Event()
        self.result = None

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        from controllers import tag_write_guard

        try:
            try:
                from controllers.tag_revert import apply_revert

                result = apply_revert(
                    self.plan,
                    progress_callback=lambda i, t: self.progress.emit(int(i * 100 / max(t, 1))),
                    cancel_event=self._cancel,
                )
            except Exception as exc:  # noqa: BLE001
                self.finished.emit(False, str(exc) or type(exc).__name__)
                return
            self.result = result
            self.result_ready.emit(result)
            parts = [f"Restored {len(result.reverted)} file(s)"]
            if result.skipped:
                parts.append(f"{len(result.skipped)} skipped (changed since the preview)")
            if result.failed:
                parts.append(f"{len(result.failed)} failed")
            message = ", ".join(parts)
            if result.cancelled:
                message += " — cancelled"
            self.finished.emit(True, message + ".")
        finally:
            # Released in the worker, so the claim is never left behind if the
            # page is gone before a slot could run.
            tag_write_guard.release(_WRITE_OWNER)


class TagHistoryWorkspace(WorkspaceBase):
    """Undo a previous tag apply, previewed first."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._logs: list = []
        self._plan = None
        self._plan_worker: RevertPlanWorker | None = None
        self._apply_worker: RevertApplyWorker | None = None
        self._build_ui()
        self.refresh_logs()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_section_title("Undo tag changes"))
        cl.addWidget(self._make_subtitle(
            "Every Tag Fixer and genre apply is logged with each tag's previous value. "
            "Pick one to see exactly what undoing it would restore. Files you have "
            "edited since are left alone, and the undo is logged so it can be undone too."
        ))

        # Change logs
        logs_card = self._make_card()
        ll = QtWidgets.QVBoxLayout(logs_card)
        ll.setContentsMargins(16, 16, 16, 16)
        ll.setSpacing(8)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(self._make_card_title("Tag applies in this library"))
        head.addStretch(1)
        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.setToolTip("Re-read the change logs in Docs/")
        self._refresh_btn.clicked.connect(self.refresh_logs)
        head.addWidget(self._refresh_btn)
        ll.addLayout(head)

        self._log_list = QtWidgets.QTreeWidget()
        self._log_list.setObjectName("tagHistoryLogs")
        self._log_list.setHeaderLabels(["When", "Tool", "Files", "Changes", "Status"])
        self._log_list.setRootIsDecorated(False)
        self._log_list.setUniformRowHeights(True)
        self._log_list.setAlternatingRowColors(True)
        self._log_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self._log_list.setMinimumHeight(160)
        self._log_list.setMaximumHeight(260)
        header = self._log_list.header()
        header.setStretchLastSection(True)
        for col in range(4):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._log_list.itemSelectionChanged.connect(self._on_log_selected)
        self._log_list.itemDoubleClicked.connect(lambda *_: self._on_preview())
        ll.addWidget(self._log_list)
        self._logs_empty = QtWidgets.QLabel(
            "No tag changes logged in this library yet. They appear here after a "
            "Tag Fixer or genre apply."
        )
        self._logs_empty.setObjectName("statusHint")
        self._logs_empty.setWordWrap(True)
        ll.addWidget(self._logs_empty)

        btn_row = QtWidgets.QHBoxLayout()
        self._preview_btn = self._make_primary_button("Preview undo")
        self._preview_btn.setToolTip("Show what undoing the selected apply would restore. Writes nothing.")
        self._preview_btn.clicked.connect(self._on_preview)
        btn_row.addWidget(self._preview_btn)
        btn_row.addStretch(1)
        ll.addLayout(btn_row)
        cl.addWidget(logs_card)

        # Preview
        prev_card = self._make_card()
        pl = QtWidgets.QVBoxLayout(prev_card)
        pl.setContentsMargins(16, 16, 16, 16)
        pl.setSpacing(8)
        pl.addWidget(self._make_card_title("What the undo would do"))
        self._summary_lbl = QtWidgets.QLabel("Select an apply above and preview it.")
        self._summary_lbl.setObjectName("statusHint")
        self._summary_lbl.setWordWrap(True)
        pl.addWidget(self._summary_lbl)

        self._preview = QtWidgets.QTreeWidget()
        self._preview.setObjectName("tagHistoryPreview")
        self._preview.setHeaderLabels(["File", "Tag", "Now", "Restores to"])
        self._preview.setRootIsDecorated(False)
        self._preview.setUniformRowHeights(True)
        self._preview.setAlternatingRowColors(True)
        self._preview.setMinimumHeight(220)
        ph = self._preview.header()
        ph.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        ph.resizeSection(0, 320)
        ph.setStretchLastSection(True)
        pl.addWidget(self._preview)

        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setFixedHeight(6)
        pl.addWidget(self._prog_bar)
        self._status_lbl = QtWidgets.QLabel("Idle")
        self._status_lbl.setObjectName("statusHint")
        self._status_lbl.setWordWrap(True)
        pl.addWidget(self._status_lbl)

        act_row = QtWidgets.QHBoxLayout()
        self._undo_btn = self._make_primary_button("Undo these changes")
        self._undo_btn.setToolTip("Write back exactly the previous values listed above")
        self._undo_btn.clicked.connect(self._on_undo)
        act_row.addWidget(self._undo_btn)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        act_row.addWidget(self._cancel_btn)
        act_row.addStretch(1)
        pl.addLayout(act_row)
        cl.addWidget(prev_card)
        cl.addStretch(1)
        self._update_buttons()

    # ── State ─────────────────────────────────────────────────────────────

    def _emit_status(self, text: str, role: str) -> None:
        self.status_changed.emit(text, getattr(get_manager().current, role))

    def _busy(self) -> bool:
        return self._plan_worker is not None or self._apply_worker is not None

    def _selected_log(self):
        items = self._log_list.selectedItems()
        if not items:
            return None
        index = items[0].data(0, _LOG_ROLE)
        if index is None or not (0 <= int(index) < len(self._logs)):
            return None
        return self._logs[int(index)]

    def _update_buttons(self) -> None:
        busy = self._busy()
        self._refresh_btn.setEnabled(not busy)
        self._log_list.setEnabled(not busy)
        self._preview_btn.setEnabled(not busy and self._selected_log() is not None)
        self._undo_btn.setEnabled(not busy and self._plan is not None and bool(self._plan.items))
        self._cancel_btn.setEnabled(busy)

    # ── Change logs ───────────────────────────────────────────────────────

    def refresh_logs(self) -> None:
        """Re-read the library's change logs (small JSONL files)."""
        if self._busy():
            return
        selected = self._selected_log()
        keep = selected.path if selected is not None else None
        from controllers.tag_revert import list_undoable_logs

        self._logs = list_undoable_logs(self._library_path) if self._library_path else []
        self._log_list.blockSignals(True)
        self._log_list.clear()
        muted = QtGui.QBrush(QtGui.QColor(get_manager().current.text_muted))
        reselect = None
        for i, info in enumerate(self._logs):
            if info.reverted_by:
                status = "Undone"
            elif info.reverts:
                status = "Undo of an earlier apply"
            else:
                status = ""
            extras = []
            if info.failed:
                extras.append(f"{info.failed} failed")
            if info.bad_lines:
                extras.append(f"{info.bad_lines} unreadable line(s)")
            if extras:
                status = ", ".join(([status] if status else []) + extras)
            item = QtWidgets.QTreeWidgetItem([
                _when(info.stamp), _tool_label(info.tool), str(info.files),
                str(info.written), status,
            ])
            item.setData(0, _LOG_ROLE, i)
            item.setToolTip(0, info.path)
            if info.reverted_by:
                for col in range(5):
                    item.setForeground(col, muted)
                item.setToolTip(4, "Already undone by " + ", ".join(
                    os.path.basename(p) for p in info.reverted_by))
            self._log_list.addTopLevelItem(item)
            if info.path == keep:
                reselect = item
        if reselect is None and self._log_list.topLevelItemCount():
            reselect = self._log_list.topLevelItem(0)   # the newest apply
        self._log_list.blockSignals(False)
        if reselect is not None:
            reselect.setSelected(True)
        self._logs_empty.setVisible(not self._logs)
        self._log_list.setVisible(bool(self._logs))
        self._update_buttons()

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        # Another page may have applied tags since this list was read.
        self.refresh_logs()

    @Slot()
    def _on_log_selected(self) -> None:
        if self._plan is not None:
            selected = self._selected_log()
            if selected is None or selected.path != self._plan.log_path:
                self._clear_preview("Select an apply above and preview it.")
        self._update_buttons()

    # ── Preview ───────────────────────────────────────────────────────────

    def _clear_preview(self, message: str) -> None:
        self._plan = None
        self._preview.clear()
        self._summary_lbl.setText(message)

    @Slot()
    def _on_preview(self) -> None:
        info = self._selected_log()
        if info is None or self._busy():
            return
        self._clear_preview("Reading the files…")
        self._prog_bar.setValue(0)
        self._status_lbl.setText("Previewing…")
        worker = RevertPlanWorker(info.path)
        worker.progress.connect(self._prog_bar.setValue)
        worker.plan_ready.connect(self._on_plan_ready)
        worker.finished.connect(self._on_plan_finished)
        self._plan_worker = worker
        self._update_buttons()
        self._start_worker(worker, "Undo preview", ask_before_quit=False)

    @Slot(object)
    def _on_plan_ready(self, plan) -> None:
        if self.sender() is not self._plan_worker:
            return
        self._plan = plan
        self._render_plan(plan)

    @Slot(bool, str)
    def _on_plan_finished(self, ok: bool, message: str) -> None:
        if self.sender() is not self._plan_worker:
            return   # a stale worker (library changed while it ran)
        self._plan_worker = None
        self._prog_bar.setValue(100 if ok else 0)
        if ok:
            self._status_lbl.setText("Preview ready. Nothing has been written.")
        else:
            self._clear_preview("Select an apply above and preview it.")
            self._status_lbl.setText(message or "Preview failed.")
            if message and message != "Cancelled.":
                self._log(f"Undo preview failed: {message}", "error")
        self._update_buttons()

    def _render_plan(self, plan) -> None:
        tokens = get_manager().current
        muted = QtGui.QBrush(QtGui.QColor(tokens.text_muted))
        warn = QtGui.QBrush(QtGui.QColor(tokens.warning))
        root = plan.library_root

        def rel(path: str) -> str:
            try:
                return os.path.relpath(path, root)
            except ValueError:
                return path

        self._preview.clear()
        rows = []
        for item in plan.items:
            for name, fr in sorted(item.fields.items()):
                row = QtWidgets.QTreeWidgetItem([rel(item.path), name, _fmt(fr.current), _fmt(fr.restore)])
                row.setToolTip(0, item.path)
                rows.append(row)
        for conflict in plan.conflicts:
            for name, (_written, now) in sorted(conflict.fields.items()):
                row = QtWidgets.QTreeWidgetItem([
                    rel(conflict.path), name, _fmt(now), "Left alone: edited since the apply",
                ])
                row.setForeground(3, warn)
                row.setToolTip(0, conflict.path)
                rows.append(row)
        for path in plan.missing:
            row = QtWidgets.QTreeWidgetItem([rel(path), "", "", "Left alone: file not found"])
            row.setForeground(3, warn)
            rows.append(row)
        for path, why in plan.unreadable:
            row = QtWidgets.QTreeWidgetItem([rel(path), "", "", f"Left alone: unreadable ({why})"])
            row.setForeground(3, warn)
            rows.append(row)
        for path in plan.already_reverted:
            row = QtWidgets.QTreeWidgetItem([rel(path), "", "", "Already holds the previous values"])
            for col in range(4):
                row.setForeground(col, muted)
            rows.append(row)
        self._preview.addTopLevelItems(rows)

        c = plan.counts()
        if c["files"]:
            text = f"Undoing this would restore {c['fields']} tag(s) in {c['files']} file(s)."
        else:
            text = "Nothing to undo: no file still holds what this apply wrote."
        notes = []
        if c["conflicts"]:
            notes.append(f"{c['conflicts']} edited since (left alone)")
        if c["missing"]:
            notes.append(f"{c['missing']} not found")
        if c["unreadable"]:
            notes.append(f"{c['unreadable']} unreadable")
        if c["already_reverted"]:
            notes.append(f"{c['already_reverted']} already restored")
        if notes:
            text += " " + " · ".join(notes) + "."
        self._summary_lbl.setText(text)

    # ── Undo ──────────────────────────────────────────────────────────────

    @Slot()
    def _on_undo(self) -> None:
        plan = self._plan
        if plan is None or not plan.items or self._busy():
            return
        if self._library_writes_blocked("Undo tag changes"):
            return
        from controllers import tag_write_guard

        c = plan.counts()
        reply = QtWidgets.QMessageBox.question(
            self, "Undo Tag Changes",
            f"Restore {c['fields']} tag(s) in {c['files']} file(s) to their previous values, "
            "exactly as listed?\n\n"
            "Each file is checked again just before writing; a file edited since the "
            "preview is skipped. The undo is logged in Docs/tag_changes_revert_….jsonl, "
            "so it can be undone too.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if not tag_write_guard.acquire(_WRITE_OWNER):
            QtWidgets.QMessageBox.information(
                self, "Tags Busy", tag_write_guard.busy_message("Undo tag changes")
            )
            return
        self._prog_bar.setValue(0)
        self._status_lbl.setText("Restoring…")
        self._emit_status("Restoring tags…", "warning")
        worker = RevertApplyWorker(plan)
        worker.progress.connect(self._prog_bar.setValue)
        worker.result_ready.connect(self._on_apply_result)
        worker.finished.connect(self._on_apply_finished)
        self._apply_worker = worker
        self._update_buttons()
        self._start_worker(worker, _WRITE_OWNER, library_write=True)

    @Slot(object)
    def _on_apply_result(self, result) -> None:
        for path, why in result.skipped:
            self._log(f"Undo skipped {path}: {why}", "warning")
        for path, why in result.failed:
            self._log(f"Undo failed for {path}: {why}", "error")
        if result.log_path:
            self._log(f"Undo logged to {result.log_path}", "info")

    @Slot(bool, str)
    def _on_apply_finished(self, ok: bool, message: str) -> None:
        if self.sender() is not self._apply_worker:
            return
        worker = self._apply_worker
        self._apply_worker = None
        self._prog_bar.setValue(100 if ok else 0)
        result = getattr(worker, "result", None)
        if ok:
            level = "warning" if result is not None and (result.failed or result.skipped) else "success"
            self._log(message, level)
            self._emit_status("Tags restored", level)
        else:
            self._log(f"Undo failed: {message}", "error")
            self._emit_status("Undo failed", "error")
        self._status_lbl.setText(message or "Done.")
        # The undo wrote a new log and marked this one as undone.
        same_library = worker is not None and worker.plan.library_root == os.path.abspath(
            self._library_path or "")
        self._clear_preview("Select an apply above and preview it.")
        if same_library or not self._library_path:
            self.refresh_logs()
        self._update_buttons()

    @Slot()
    def _on_cancel(self) -> None:
        for worker in (self._plan_worker, self._apply_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
        self._status_lbl.setText("Cancelling…")

    # ── Library change ────────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        # A preview belongs to the library it was read from: drop it. A
        # confirmed undo finishes on its own library (it writes exactly what
        # was confirmed) and is logged there.
        if self._plan_worker is not None:
            self._plan_worker.cancel()
            self._plan_worker = None
        self._clear_preview("Select an apply above and preview it.")
        self._prog_bar.setValue(0)
        if self._apply_worker is None:
            self._status_lbl.setText("Idle")
            self.refresh_logs()
        self._update_buttons()
