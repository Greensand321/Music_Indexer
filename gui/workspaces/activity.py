"""Activity page: running jobs, the full log history, and the latest problems.

* **Running jobs** — every background job in ``gui.workers.JobRegistry``
  (live: the list follows ``jobs_changed``), with the page that started it,
  how long it has run, and a Cancel button (``gui.workers.cancel_thread``:
  the job stops at its next safe point, as on quit).
* **Log** — every message any page logged this session (the Activity Log
  drawer keeps working too; it holds the last 500 lines, this keeps more).
* **Problems** — this session's warnings and errors, and the tail of the
  crash logs (``crash_logger``; the same files the old Crash Log viewer read).

Read-only: nothing here changes the library.
"""
from __future__ import annotations

import datetime
import time

from gui.compat import QtCore, QtGui, QtWidgets, Slot
from gui.workspaces.base import WorkspaceBase

_HISTORY_LIMIT = 5000
_PROBLEM_LIMIT = 200
_LEVELS = ("info", "ok", "warn", "error")


def _elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def crash_log_text() -> str:
    """The tail of the crash logs, or a "no crash log" note."""
    from gui.dialogs.crash_log_dialog import _tail, crash_log_paths

    paths = crash_log_paths()
    sections = []
    for path in paths:
        body = _tail(path, 60)
        if body:
            sections.append(f"── {path.name} ──\n{body}")
    if sections:
        return "\n\n".join(sections)
    return f"No crash log found. AlphaDEX records crashes in {paths[0]}."


class ActivityWorkspace(WorkspaceBase):
    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._owner_labels: dict[str, str] = {}
        self._row_threads: list = []
        self._history: list[tuple[str, str, str]] = []   # (time, level, message)
        self._tick = QtCore.QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._refresh_elapsed)
        self._build_ui()
        from gui.workers import job_registry
        job_registry().jobs_changed.connect(self._on_jobs_changed)
        self.refresh_jobs()
        self.refresh_problems()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_section_title("Activity"))
        cl.addWidget(self._make_subtitle(
            "What AlphaDEX is doing now, everything it has reported this session, and "
            "the latest problems. Cancelling a job stops it at its next safe point; "
            "whatever it finished is kept and logged."
        ))

        # Running jobs
        jobs_card = self._make_card()
        jl = QtWidgets.QVBoxLayout(jobs_card)
        jl.setContentsMargins(16, 16, 16, 16)
        jl.setSpacing(8)
        jl.addWidget(self._make_card_title("Running jobs"))
        self._jobs_table = QtWidgets.QTableWidget(0, 4)
        self._jobs_table.setObjectName("activityJobs")
        self._jobs_table.setHorizontalHeaderLabels(["Job", "Started by", "Running for", ""])
        header = self._jobs_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._jobs_table.verticalHeader().setVisible(False)
        self._jobs_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._jobs_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self._jobs_table.setMinimumHeight(120)
        self._jobs_table.setMaximumHeight(220)
        jl.addWidget(self._jobs_table)
        self._jobs_empty = QtWidgets.QLabel("Nothing is running.")
        self._jobs_empty.setObjectName("statusHint")
        jl.addWidget(self._jobs_empty)
        cl.addWidget(jobs_card)

        # Problems
        prob_card = self._make_card()
        pl = QtWidgets.QVBoxLayout(prob_card)
        pl.setContentsMargins(16, 16, 16, 16)
        pl.setSpacing(8)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(self._make_card_title("Problems"))
        head.addStretch(1)
        self._refresh_problems_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_problems_btn.setToolTip("Re-read the crash logs")
        self._refresh_problems_btn.clicked.connect(self.refresh_problems)
        head.addWidget(self._refresh_problems_btn)
        pl.addLayout(head)
        hint = QtWidgets.QLabel("Warnings and errors from this session:")
        hint.setObjectName("statusHint")
        pl.addWidget(hint)
        self._problems = QtWidgets.QPlainTextEdit()
        self._problems.setObjectName("logText")
        self._problems.setReadOnly(True)
        self._problems.setMaximumBlockCount(_PROBLEM_LIMIT)
        self._problems.setPlaceholderText("No warnings or errors so far.")
        self._problems.setFixedHeight(110)
        pl.addWidget(self._problems)
        crash_hint = QtWidgets.QLabel("Latest crash log entries:")
        crash_hint.setObjectName("statusHint")
        pl.addWidget(crash_hint)
        self._crash_text = QtWidgets.QPlainTextEdit()
        self._crash_text.setObjectName("logText")
        self._crash_text.setReadOnly(True)
        self._crash_text.setFixedHeight(130)
        pl.addWidget(self._crash_text)

        # Log history
        log_card = self._make_card()
        ll = QtWidgets.QVBoxLayout(log_card)
        ll.setContentsMargins(16, 16, 16, 16)
        ll.setSpacing(8)
        head = QtWidgets.QHBoxLayout()
        head.addWidget(self._make_card_title("Log"))
        head.addStretch(1)
        self._log_filter = QtWidgets.QComboBox()
        self._log_filter.addItems(["Everything", "Warnings and errors"])
        self._log_filter.currentIndexChanged.connect(self._render_history)
        head.addWidget(self._log_filter)
        self._clear_btn = QtWidgets.QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear_history)
        head.addWidget(self._clear_btn)
        ll.addLayout(head)
        self._history_text = QtWidgets.QPlainTextEdit()
        self._history_text.setObjectName("logText")
        self._history_text.setReadOnly(True)
        self._history_text.setMaximumBlockCount(_HISTORY_LIMIT)
        self._history_text.setPlaceholderText("Nothing logged yet.")
        self._history_text.setMinimumHeight(220)
        ll.addWidget(self._history_text)
        # Order on the page: jobs, the log, then problems.
        cl.addWidget(log_card)
        cl.addWidget(prob_card)
        cl.addStretch(1)

    # ── Jobs ──────────────────────────────────────────────────────────────

    def set_owner_labels(self, labels: dict[str, str]) -> None:
        """Class name of a job's owner → the page name to show for it."""
        self._owner_labels = dict(labels)
        self.refresh_jobs()

    def _owner_label(self, owner_name: str) -> str:
        if owner_name in self._owner_labels:
            return self._owner_labels[owner_name]
        if owner_name == "AlphaDEXWindow":
            return "AlphaDEX"
        for suffix in ("Workspace", "Dialog"):
            if owner_name.endswith(suffix) and owner_name != suffix:
                return owner_name[: -len(suffix)]
        return owner_name or "—"

    @Slot(int)
    def _on_jobs_changed(self, _count: int = 0) -> None:
        # Hidden: rebuild once when shown (jobs come and go often at start-up).
        if self.isVisible():
            self.refresh_jobs()
        else:
            self._jobs_stale = True

    def jobs(self) -> list:
        from gui.workers import job_registry
        return job_registry().jobs()

    def refresh_jobs(self) -> None:
        jobs = self.jobs()
        table = self._jobs_table
        table.setRowCount(len(jobs))
        self._row_threads = []
        now = time.monotonic()
        for row, job in enumerate(jobs):
            self._row_threads.append(job.thread)
            name = QtWidgets.QTableWidgetItem(job.name)
            if job.group:
                name.setToolTip("Changes files in your library")
            table.setItem(row, 0, name)
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(self._owner_label(job.owner_name)))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(_elapsed(now - job.started_at)))
            btn = QtWidgets.QPushButton("Cancel")
            btn.setObjectName("activityCancelBtn")
            btn.setToolTip(f"Stop “{job.name}” at its next safe point")
            btn.clicked.connect(self._on_cancel_clicked)
            btn.setProperty("jobRow", row)
            if self._cancel_requested(job.thread):
                btn.setText("Stopping…")
                btn.setEnabled(False)
            table.setCellWidget(row, 3, btn)
        self._jobs_empty.setVisible(not jobs)
        self._jobs_table.setVisible(bool(jobs))
        if jobs and self.isVisible():
            self._tick.start()
        else:
            self._tick.stop()

    @staticmethod
    def _cancel_requested(thread) -> bool:
        try:
            return bool(thread.isInterruptionRequested())
        except RuntimeError:
            return False

    @Slot()
    def _on_cancel_clicked(self) -> None:
        btn = self.sender()
        row = btn.property("jobRow") if btn is not None else None
        if row is None or not (0 <= int(row) < len(self._row_threads)):
            return
        self.cancel_job(int(row))

    def cancel_job(self, row: int) -> None:
        """Ask the job in ``row`` to stop (cooperatively)."""
        from gui.workers import cancel_thread

        thread = self._row_threads[row]
        name = self._jobs_table.item(row, 0).text() if self._jobs_table.item(row, 0) else "job"
        cancel_thread(thread)
        btn = self._jobs_table.cellWidget(row, 3)
        if btn is not None:
            btn.setText("Stopping…")
            btn.setEnabled(False)
        self._log(f"Cancel requested: {name} (it stops at its next safe point).", "warn")

    def _refresh_elapsed(self) -> None:
        if not self.isVisible():
            self._tick.stop()
            return
        jobs = self.jobs()
        if [j.thread for j in jobs] != self._row_threads:
            self.refresh_jobs()
            return
        now = time.monotonic()
        for row, job in enumerate(jobs):
            item = self._jobs_table.item(row, 2)
            if item is not None:
                item.setText(_elapsed(now - job.started_at))

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self._jobs_stale = False
        self.refresh_jobs()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:  # noqa: N802
        super().hideEvent(event)
        self._tick.stop()

    def on_activated(self) -> None:
        self.refresh_jobs()
        self.refresh_problems()

    # ── Log history ───────────────────────────────────────────────────────

    def append_log(self, message: str, level: str = "info") -> None:
        """Record one log message (the main window sends every page's)."""
        level = level if level in _LEVELS else "info"
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._history.append((stamp, level, message))
        if len(self._history) > _HISTORY_LIMIT:
            del self._history[: len(self._history) - _HISTORY_LIMIT]
        line = self._format(stamp, level, message)
        if self._passes_filter(level):
            self._history_text.appendPlainText(line)
        if level in ("warn", "error"):
            self._problems.appendPlainText(line)

    def history(self) -> list[tuple[str, str, str]]:
        return list(self._history)

    @staticmethod
    def _format(stamp: str, level: str, message: str) -> str:
        tag = {"warn": "WARN ", "error": "ERROR", "ok": "OK   "}.get(level, "     ")
        return f"[{stamp}] {tag} {message}"

    def _passes_filter(self, level: str) -> bool:
        return self._log_filter.currentIndex() == 0 or level in ("warn", "error")

    def _render_history(self, *_args) -> None:
        lines = [self._format(*entry) for entry in self._history if self._passes_filter(entry[1])]
        self._history_text.setPlainText("\n".join(lines))
        self._history_text.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    @Slot()
    def clear_history(self) -> None:
        self._history.clear()
        self._history_text.clear()

    # ── Problems ──────────────────────────────────────────────────────────

    @Slot()
    def refresh_problems(self) -> None:
        try:
            text = crash_log_text()
        except Exception as exc:  # noqa: BLE001 - never break the page over a log
            text = f"The crash log could not be read: {exc}"
        self._crash_text.setPlainText(text)

    def shutdown(self) -> None:
        self._tick.stop()
        super().shutdown()
