"""Background-job lifecycle for the Qt app.

The problem this solves
-----------------------
Every workspace runs long operations in ``QThread`` subclasses. Two habits
made the app abort with ``QThread: Destroyed while thread is still running``
(a qFatal — the whole process dies, mid file-move if it is unlucky):

1. Workers declare their own ``finished = Signal(...)``, shadowing
   ``QThread.finished``, and emit it from *inside* ``run()``. The workspace's
   slot then does ``self._worker = None`` — dropping the last Python
   reference while ``run()`` has not returned yet.
2. Nothing stops workers when the main window closes, or when a second run
   replaces the first.

The fix is a single, process-wide ``JobRegistry`` that holds a strong
reference to every started worker until the thread has *really* finished
(the C++ ``QThread::finished()`` signal, which fires after ``run()``
returns and cannot be shadowed from Python), plus a cooperative shutdown
protocol used by the main window's ``closeEvent``.

Usage
-----
In a workspace::

    self._worker = MyWorker(...)
    self._worker.finished.connect(self._on_finished)   # custom signal is fine
    self._start_worker(self._worker, "Indexer preview")  # WorkspaceBase helper

or anywhere else::

    from gui.workers import start_worker
    start_worker(thread, owner=self, name="Album-art scan")

After that, dropping ``self._worker`` at any time is safe.

Cancellation: ``cancel_all()`` calls ``worker.cancel()`` when the worker
defines one (most do — they set a ``threading.Event``) and always calls
``requestInterruption()`` so new workers can poll
``self.isInterruptionRequested()``.

The registry also exposes ``jobs_changed(int)`` — the number of running
jobs — so the redesigned shell can show a global activity indicator.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from gui.compat import QtCore, QtWidgets, Signal


@dataclass
class Job:
    thread: QtCore.QThread
    name: str
    owner_name: str
    started_at: float
    # False for read-only work (scans, previews, art loads, statistics) that
    # can be cancelled on quit without asking the user.
    ask_before_quit: bool = True
    # Mutual-exclusion group, e.g. LIBRARY_WRITES: at most one running job per
    # group is the rule callers enforce via ``group_jobs`` before starting.
    group: str | None = None

    @property
    def running(self) -> bool:
        try:
            return self.thread.isRunning()
        except RuntimeError:  # C++ object already gone
            return False


class JobRegistry(QtCore.QObject):
    """Process-wide owner of every background worker thread."""

    jobs_changed = Signal(int)   # number of jobs still running

    _instance: "JobRegistry | None" = None

    def __init__(self) -> None:
        super().__init__()
        self._jobs: dict[int, Job] = {}
        self._watchers: dict[int, "_FinishWatcher"] = {}

    @classmethod
    def instance(cls) -> "JobRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── registration ───────────────────────────────────────────────────────
    def adopt(
        self,
        thread: QtCore.QThread,
        *,
        owner: QtCore.QObject | None = None,
        name: str = "",
        ask_before_quit: bool = True,
        group: str | None = None,
    ) -> QtCore.QThread:
        """Keep ``thread`` alive until it has finished. Idempotent."""
        key = id(thread)
        if key in self._jobs:
            return thread
        owner_name = type(owner).__name__ if owner is not None else ""
        self._jobs[key] = Job(thread, name or type(thread).__name__, owner_name,
                              time.monotonic(), ask_before_quit, group)
        # The C++ finished() signal, connected by signature so a Python
        # subclass that shadows ``finished`` cannot hide it. The watcher lives
        # in the GUI thread, so delivery is queued there, after run() returns.
        watcher = _FinishWatcher(self, key)
        self._watchers[key] = watcher
        QtCore.QObject.connect(
            thread, QtCore.SIGNAL("finished()"), watcher, QtCore.SLOT("on_finished()")
        )
        self.jobs_changed.emit(self.running_count())
        return thread

    def start(
        self,
        thread: QtCore.QThread,
        *,
        owner: QtCore.QObject | None = None,
        name: str = "",
        ask_before_quit: bool = True,
        group: str | None = None,
    ) -> QtCore.QThread:
        self.adopt(thread, owner=owner, name=name, ask_before_quit=ask_before_quit,
                   group=group)
        thread.start()
        self.jobs_changed.emit(self.running_count())
        return thread

    def _release(self, key: int) -> None:
        job = self._jobs.get(key)
        if job is None:
            return
        try:
            # finished() is emitted just before the thread exits; make sure it
            # has fully stopped before the last reference can go away.
            job.thread.wait(2000)
        except RuntimeError:
            pass
        if not job.running:
            self._jobs.pop(key, None)
            watcher = self._watchers.pop(key, None)
            if watcher is not None:
                watcher.deleteLater()
        self.jobs_changed.emit(self.running_count())

    # ── queries ────────────────────────────────────────────────────────────
    def jobs(self, owner: QtCore.QObject | None = None) -> list[Job]:
        self._prune()
        jobs = [j for j in self._jobs.values() if j.running]
        if owner is not None:
            owner_name = type(owner).__name__
            jobs = [j for j in jobs if j.owner_name == owner_name]
        return jobs

    def group_jobs(self, group: str) -> list[Job]:
        """Running jobs in ``group`` (e.g. ``LIBRARY_WRITES``)."""
        return [j for j in self.jobs() if j.group == group]

    def running_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.running)

    def describe(self) -> list[str]:
        return [j.name for j in self.jobs()]

    def _prune(self) -> None:
        for key, job in list(self._jobs.items()):
            try:
                finished = job.thread.isFinished()
            except RuntimeError:
                finished = True
            if finished:
                self._jobs.pop(key, None)
                watcher = self._watchers.pop(key, None)
                if watcher is not None:
                    watcher.deleteLater()

    # ── shutdown ───────────────────────────────────────────────────────────
    def cancel_all(self, jobs: Iterable[Job] | None = None) -> None:
        for job in list(jobs if jobs is not None else self.jobs()):
            cancel_thread(job.thread)

    def wait_all(self, timeout_ms: int = 5000, process_events: bool = True) -> bool:
        """Wait for every job; True when none is left running."""
        deadline = time.monotonic() + timeout_ms / 1000
        app = QtWidgets.QApplication.instance()
        while self.running_count():
            if time.monotonic() >= deadline:
                return False
            if process_events and app is not None:
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            for job in self.jobs():
                try:
                    job.thread.wait(10)
                except RuntimeError:
                    pass
        self._prune()
        return True


class _FinishWatcher(QtCore.QObject):
    """Receives one thread's C++ ``finished()`` on the GUI thread."""

    def __init__(self, registry: JobRegistry, key: int) -> None:
        super().__init__(registry)
        self._registry = registry
        self._key = key

    @QtCore.Slot()
    def on_finished(self) -> None:
        self._registry._release(self._key)


def cancel_thread(thread: QtCore.QThread) -> None:
    """Ask a worker to stop, whatever cancellation style it implements."""
    try:
        cancel = getattr(thread, "cancel", None)
        if callable(cancel):
            cancel()
        thread.requestInterruption()
    except RuntimeError:
        pass


def job_registry() -> JobRegistry:
    return JobRegistry.instance()


# Jobs that move, rename, delete or rewrite files inside the library. Two of
# them must never run at once (an Indexer execute moving files while a Tag
# Fixer apply rewrites them, or Library Sync and Duplicates both moving files,
# leaves the library in a state neither plan described).
LIBRARY_WRITES = "library-writes"


def start_worker(
    thread: QtCore.QThread,
    *,
    owner: QtCore.QObject | None = None,
    name: str = "",
    ask_before_quit: bool = True,
    group: str | None = None,
) -> QtCore.QThread:
    """Start ``thread`` under the registry's protection (see module docstring).

    Pass ``ask_before_quit=False`` for read-only work (scans, previews, art
    loads): quitting then cancels it silently instead of asking the user.
    Anything that writes files keeps the default, so quitting mid-write asks.
    """
    return job_registry().start(thread, owner=owner, name=name,
                                ask_before_quit=ask_before_quit, group=group)
