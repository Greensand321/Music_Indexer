"""Activity page: running and recent jobs, the full log history, and problems.

* **Jobs** — every background job in ``gui.workers.JobRegistry`` as a glass
  row (live: the list follows ``jobs_changed``): what it is, the page that
  started it, how long it has run, a thin progress bar, and Cancel
  (``gui.workers.cancel_thread``: the job stops at its next safe point, as
  on quit). Below them, the jobs that finished this session (newest first).
  The bar shows real progress when the job reports it in a form that says
  what it is (``progress(int)`` is a percentage, ``progress(int, int[, str])``
  is done and total); otherwise it is a faint full bar: running, amount
  unknown. Nothing is guessed.
* **Log** — every message any page logged this session (the Activity Log
  strip keeps working too; it holds the last 500 lines, this keeps more).
* **Problems** — this session's warnings and errors, and the tail of the
  crash logs (``crash_logger``; the same files the old Crash Log viewer read).

Read-only: nothing here changes the library.
"""
from __future__ import annotations

import datetime
import time
from dataclasses import dataclass

from gui.compat import QtCore, QtGui, QtWidgets, Slot
from gui.workspaces.base import WorkspaceBase

_HISTORY_LIMIT = 5000
_PROBLEM_LIMIT = 200
_RECENT_LIMIT = 8
_LEVELS = ("info", "ok", "warn", "error")

# Progress signatures whose meaning is known (see the module docstring).
_PERCENT = "progress(int)"
_DONE_TOTAL = ("progress(int,int)", "progress(int,int,QString)")


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


def progress_kind(thread) -> str:
    """"percent", "done_total" or "" for a job thread's ``progress`` signal."""
    try:
        meta = thread.metaObject()
    except RuntimeError:
        return ""
    # The most derived class's ``progress`` is the one ``thread.progress``
    # names (a subclass may redeclare it with other arguments).
    while meta is not None:
        for i in range(meta.methodOffset(), meta.methodCount()):
            method = meta.method(i)
            if method.methodType() != QtCore.QMetaMethod.MethodType.Signal:
                continue
            sig = method.methodSignature().data().decode("latin-1")
            if sig.startswith("progress("):
                if sig == _PERCENT:
                    return "percent"
                return "done_total" if sig in _DONE_TOTAL else ""
        meta = meta.superClass()
    return ""


@dataclass
class FinishedJob:
    name: str
    owner: str
    took: float          # seconds
    ended: float         # epoch seconds
    stopped: bool        # cancel was requested before it ended


def _paint_bar(p: QtGui.QPainter, rect: QtCore.QRectF, fraction: float | None) -> None:
    """A thin pill bar: the track, and the accent fill (None: running, amount
    unknown — the whole track faintly tinted)."""
    from gui.themes.glass import current_material, qcolor
    from gui.themes.manager import get_manager

    t = get_manager().current
    r = rect.height() / 2
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.setBrush(qcolor(current_material().track))
    p.drawRoundedRect(rect, r, r)
    if fraction is None:
        p.setBrush(qcolor(t.accent, 0.35))
        p.drawRoundedRect(rect, r, r)
        return
    fraction = max(0.0, min(1.0, fraction))
    if fraction > 0:
        fill = QtCore.QRectF(rect.x(), rect.y(), max(rect.height(), rect.width() * fraction),
                             rect.height())
        p.setBrush(qcolor(t.accent))
        p.drawRoundedRect(fill, r, r)


class _Row(QtWidgets.QWidget):
    """A glass list row (radius 14, the material's hover fill)."""

    def paint_row(self, p: QtGui.QPainter) -> None:
        from gui.themes.glass import current_material, paint_surface
        from gui.themes.tokens import Surface

        paint_surface(p, QtCore.QRectF(self.rect()), Surface(current_material().hover, radius=14))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self.paint_row(p)
        finally:
            p.end()


def _label(text: str, role: str, *, muted: bool = False, weight: int | None = None):
    from gui.themes.glass import apply_font
    from gui.widgets.glass import ElidedLabel

    label = ElidedLabel(text)
    if muted:
        label.setObjectName("mutedLabel")
    apply_font(label, role, **({"weight": weight} if weight else {}))
    return label


class JobRow(_Row):
    """A running job: name, owner and time, a thin progress bar, Cancel."""

    BAR_H = 4

    def __init__(self, job, owner_label: str, on_cancel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from gui.widgets.glyphs import set_glyph

        self.setObjectName("activityJobRow")
        self.job = job
        self.job_thread = job.thread
        self.owner_label = owner_label
        self.fraction: float | None = None
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 10, 10 + self.BAR_H + 8)
        lay.setSpacing(12)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(1)
        self.name_label = _label(job.name, "body", weight=600)
        if job.group:
            self.name_label.setToolTip("Changes files in your library")
        self.detail_label = _label("", "caption", muted=True)
        text.addWidget(self.name_label)
        text.addWidget(self.detail_label)
        lay.addLayout(text, 1)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setObjectName("activityCancelBtn")
        self.cancel_btn.setToolTip(
            f"Stop “{job.name}” at its next safe point; whatever it finished is kept and logged")
        self.cancel_btn.setAccessibleName(f"Cancel {job.name}")
        set_glyph(self.cancel_btn, "stop", size=14)
        self.cancel_btn.clicked.connect(on_cancel)
        lay.addWidget(self.cancel_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._connect_progress()
        self.update_elapsed()

    # Progress: only signatures whose meaning is known (module docstring).
    def _connect_progress(self) -> None:
        kind = progress_kind(self.job_thread)
        try:
            if kind == "percent":
                self.job_thread.progress.connect(self._on_percent)
            elif kind == "done_total":
                self.job_thread.progress.connect(self._on_done_total)
        except (RuntimeError, AttributeError, TypeError):
            pass

    @Slot(int)
    def _on_percent(self, pct: int) -> None:
        self.set_fraction(pct / 100.0)

    @Slot(int, int)
    def _on_done_total(self, done: int, total: int, *_rest) -> None:
        self.set_fraction(done / total if total and total > 0 else None)

    def set_fraction(self, fraction: float | None) -> None:
        self.fraction = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        self.update()

    def update_elapsed(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.elapsed_text = _elapsed(now - self.job.started_at)
        self.detail_label.setText(f"{self.owner_label}  ·  running for {self.elapsed_text}")

    def set_stopping(self) -> None:
        self.cancel_btn.setText("Stopping…")
        self.cancel_btn.setEnabled(False)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self.paint_row(p)
            bar = QtCore.QRectF(16, self.height() - 12 - self.BAR_H,
                                max(0, self.width() - 32), self.BAR_H)
            _paint_bar(p, bar, self.fraction)
        finally:
            p.end()


class FinishedRow(_Row):
    """A job that ended this session."""

    def __init__(self, done: FinishedJob, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("activityFinishedRow")
        self.done = done
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(32, 7, 12, 7)
        lay.setSpacing(10)
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(1)
        self.name_label = _label(done.name, "label", weight=600)
        state = "Stopped on request" if done.stopped else "Ended"
        when = datetime.datetime.fromtimestamp(done.ended).strftime("%H:%M:%S")
        self.detail_label = _label(
            f"{state}  ·  {done.owner}  ·  took {_elapsed(done.took)}  ·  {when}", "caption", muted=True)
        text.addWidget(self.name_label)
        text.addWidget(self.detail_label)
        lay.addLayout(text, 1)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import qcolor
        from gui.themes.manager import get_manager

        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self.paint_row(p)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(qcolor(t.warning if self.done.stopped else t.success))
            p.drawEllipse(QtCore.QPointF(16.0, self.height() / 2.0), 4.0, 4.0)
        finally:
            p.end()


class ActivityWorkspace(WorkspaceBase):
    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._owner_labels: dict[str, str] = {}
        self._rows: list[JobRow] = []
        self._known: dict[int, tuple] = {}      # id(thread) → (job, thread)
        # Threads someone asked to stop (QThread forgets once it has ended).
        self._stop_requested: set[int] = set()
        self._recent_shown: list[FinishedJob] | None = None
        self._finished: list[FinishedJob] = []
        self._history: list[tuple[str, str, str]] = []   # (time, level, message)
        self._jobs_stale = False
        self._tick = QtCore.QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._refresh_elapsed)
        self._build_ui()
        from gui.workers import job_registry
        job_registry().jobs_changed.connect(self._on_jobs_changed)
        self._track_jobs()
        self.refresh_jobs()
        self.refresh_problems()

    # ── UI ────────────────────────────────────────────────────────────────

    def _card(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout]:
        card = self._make_card()
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(10)
        head.addWidget(self._make_card_title(title))
        head.addStretch(1)
        lay.addLayout(head)
        return card, lay, head

    @staticmethod
    def _caption(text: str) -> QtWidgets.QLabel:
        from gui.themes.glass import apply_font

        label = QtWidgets.QLabel(text)
        label.setObjectName("mutedLabel")
        apply_font(label, "caption", weight=600)
        return label

    def _build_ui(self) -> None:
        from gui.widgets.glass_parts import EmptyState
        from gui.widgets.glyphs import set_glyph

        cl = self.content_layout
        cl.addWidget(self._make_subtitle(
            "What AlphaDEX is doing now, and everything it reported this session."))

        # Jobs: running rows, then this session's finished ones
        jobs_card, jl, _head = self._card("Jobs")
        self._jobs_card = jobs_card
        self._jobs_box = QtWidgets.QVBoxLayout()
        self._jobs_box.setSpacing(8)
        jl.addLayout(self._jobs_box)
        self._jobs_empty = EmptyState(
            "Nothing is running",
            "Scans, previews and file operations show here while they run, with Cancel.")
        jl.addWidget(self._jobs_empty)
        self._recent_caption = self._caption("Earlier this session")
        jl.addWidget(self._recent_caption)
        self._recent_box = QtWidgets.QVBoxLayout()
        self._recent_box.setSpacing(6)
        jl.addLayout(self._recent_box)
        cl.addWidget(jobs_card)

        # Log history
        log_card, ll, head = self._card("Log")
        self._log_filter = QtWidgets.QComboBox()
        self._log_filter.addItems(["Everything", "Warnings and errors"])
        self._log_filter.setToolTip("Which lines to show")
        self._log_filter.currentIndexChanged.connect(self._render_history)
        head.addWidget(self._log_filter)
        self._clear_btn = QtWidgets.QPushButton("Clear")
        self._clear_btn.setToolTip("Clear this page's log (the Activity Log strip keeps its own)")
        set_glyph(self._clear_btn, "trash", size=14)
        self._clear_btn.clicked.connect(self.clear_history)
        head.addWidget(self._clear_btn)
        self._history_text = QtWidgets.QPlainTextEdit()
        self._history_text.setObjectName("logText")
        self._history_text.setReadOnly(True)
        self._history_text.setMaximumBlockCount(_HISTORY_LIMIT)
        self._history_text.setPlaceholderText("Nothing logged yet.")
        self._history_text.setMinimumHeight(220)
        ll.addWidget(self._history_text)
        cl.addWidget(log_card)

        # Problems
        prob_card, pl, head = self._card("Problems")
        self._refresh_problems_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_problems_btn.setToolTip("Read the crash logs again")
        set_glyph(self._refresh_problems_btn, "refresh", size=14)
        self._refresh_problems_btn.clicked.connect(self.refresh_problems)
        head.addWidget(self._refresh_problems_btn)
        pl.addWidget(self._caption("Warnings and errors this session"))
        self._problems = QtWidgets.QPlainTextEdit()
        self._problems.setObjectName("logText")
        self._problems.setReadOnly(True)
        self._problems.setMaximumBlockCount(_PROBLEM_LIMIT)
        self._problems.setPlaceholderText("No warnings or errors so far.")
        self._problems.setFixedHeight(110)
        pl.addWidget(self._problems)
        pl.addSpacing(4)
        pl.addWidget(self._caption("Latest crash log entries"))
        self._crash_text = QtWidgets.QPlainTextEdit()
        self._crash_text.setObjectName("logText")
        self._crash_text.setReadOnly(True)
        self._crash_text.setFixedHeight(130)
        pl.addWidget(self._crash_text)
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
        self._track_jobs()
        # Hidden: rebuild once when shown (jobs come and go often at start-up).
        if self.isVisible():
            self.refresh_jobs()
        else:
            self._jobs_stale = True

    def jobs(self) -> list:
        """Running jobs to list (housekeeping such as the backdrop is not)."""
        from gui.workers import job_registry
        return job_registry().listed_jobs()

    def _track_jobs(self) -> None:
        """Note the jobs that ended since the last change (cheap; always runs)."""
        current = {id(j.thread): (j, j.thread) for j in self.jobs()}
        for key, (_job, thread) in current.items():
            if self._cancel_requested(thread):
                self._stop_requested.add(key)
        ended = [(key, entry) for key, entry in self._known.items() if key not in current]
        now_m, now = time.monotonic(), time.time()
        for key, (job, _thread) in ended:
            self._finished.insert(0, FinishedJob(
                job.name, self._owner_label(job.owner_name), now_m - job.started_at, now,
                key in self._stop_requested))
            self._stop_requested.discard(key)
        del self._finished[_RECENT_LIMIT:]
        self._known = current
        if ended and self.isVisible():
            self._render_recent()

    def finished_jobs(self) -> list[FinishedJob]:
        return list(self._finished)

    def job_rows(self) -> list[JobRow]:
        return list(self._rows)

    def refresh_jobs(self) -> None:
        jobs = self.jobs()
        threads = [j.thread for j in jobs]
        if [r.job_thread for r in self._rows] != threads:
            # Keep the rows of jobs still running (their progress), add new ones.
            keep = {id(r.job_thread): r for r in self._rows}
            rows = []
            for job in jobs:
                row = keep.pop(id(job.thread), None)
                if row is None:
                    row = JobRow(job, self._owner_label(job.owner_name), self._on_cancel_clicked)
                rows.append(row)
            for row in keep.values():
                self._jobs_box.removeWidget(row)
                row.setParent(None)
                row.deleteLater()
            for i, row in enumerate(rows):
                self._jobs_box.insertWidget(i, row)
            self._rows = rows
        now = time.monotonic()
        for row in self._rows:
            row.owner_label = self._owner_label(row.job.owner_name)
            row.update_elapsed(now)
            if self._cancel_requested(row.job_thread):
                self._stop_requested.add(id(row.job_thread))
                row.set_stopping()
        self._jobs_empty.setVisible(not jobs)
        self._render_recent()
        if jobs and self.isVisible():
            self._tick.start()
        else:
            self._tick.stop()

    def _render_recent(self) -> None:
        if self._recent_shown == self._finished:
            return
        self._recent_shown = list(self._finished)
        while self._recent_box.count():
            widget = self._recent_box.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        for done in self._finished:
            self._recent_box.addWidget(FinishedRow(done))
        self._recent_caption.setVisible(bool(self._finished))

    @staticmethod
    def _cancel_requested(thread) -> bool:
        try:
            return bool(thread.isInterruptionRequested())
        except RuntimeError:
            return False

    @Slot()
    def _on_cancel_clicked(self) -> None:
        btn = self.sender()
        for i, row in enumerate(self._rows):
            if row.cancel_btn is btn:
                self.cancel_job(i)
                return

    def cancel_job(self, row: int) -> None:
        """Ask the job in ``row`` to stop (cooperatively)."""
        from gui.workers import cancel_thread

        if not 0 <= row < len(self._rows):
            return
        job_row = self._rows[row]
        self._stop_requested.add(id(job_row.job_thread))
        cancel_thread(job_row.job_thread)
        job_row.set_stopping()
        self._log(f"Cancel requested: {job_row.job.name} (it stops at its next safe point).", "warn")

    def _refresh_elapsed(self) -> None:
        if not self.isVisible():
            self._tick.stop()
            return
        if [j.thread for j in self.jobs()] != [r.job_thread for r in self._rows]:
            self.refresh_jobs()
            return
        now = time.monotonic()
        for row in self._rows:
            row.update_elapsed(now)

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
