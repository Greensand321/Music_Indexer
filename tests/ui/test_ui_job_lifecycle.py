"""Background-job lifecycle: the fix for "QThread: Destroyed while thread is still running".

That message is a qFatal: it kills the interpreter. Before gui/workers.py it
fired whenever a workspace dropped its worker reference inside the worker's
own (shadowed) ``finished`` signal, re-ran a job, or the window closed
mid-run. These tests drive exactly those situations; if the protection
regresses, this module aborts the test process instead of failing politely,
which is loud enough.
"""
from __future__ import annotations

import gc
import threading

from PySide6 import QtCore, QtWidgets

from gui.workers import job_registry, start_worker
from gui.workspaces.base import WorkspaceBase
from ui_harness import process_events


class SlowWorker(QtCore.QThread):
    """Mimics the app's workers: shadows ``finished`` and emits it from run()."""

    finished = QtCore.Signal(bool, str)

    def __init__(self, seconds: float = 0.3) -> None:
        super().__init__()
        self._cancel = threading.Event()
        self._seconds = seconds
        self.cancel_called = False

    def cancel(self) -> None:
        self.cancel_called = True
        self._cancel.set()

    def run(self) -> None:
        self._cancel.wait(self._seconds)
        self.finished.emit(not self._cancel.is_set(), "done")
        # Keep running a little *after* emitting — the window in which the
        # old code dropped its reference and Python destroyed the thread.
        self.msleep(50)


def test_dropping_reference_in_finished_slot_is_safe(qtbot):
    holder = {}

    def on_finished(ok, msg):
        holder["result"] = ok
        holder.pop("worker", None)   # the app's historical pattern
        gc.collect()

    worker = SlowWorker(0.05)
    worker.finished.connect(on_finished)
    holder["worker"] = worker
    start_worker(worker, name="slow")
    del worker
    qtbot.waitUntil(lambda: "result" in holder, timeout=3000)
    assert job_registry().wait_all(3000)
    assert holder["result"] is True


def test_replacing_a_running_worker_is_safe(qtbot):
    """Clicking Run twice used to orphan the first thread."""
    workers = []
    for _ in range(3):
        w = SlowWorker(0.2)
        start_worker(w, name="rerun")
        workers.append(w)
    workers.clear()
    gc.collect()
    assert job_registry().running_count() >= 1
    assert job_registry().wait_all(5000)


class _JobWorkspace(WorkspaceBase):
    def __init__(self) -> None:
        super().__init__()
        self.shutdown_calls = 0

    def run_job(self, seconds: float) -> SlowWorker:
        self._worker = SlowWorker(seconds)
        self._worker.finished.connect(self._on_done)
        self._start_worker(self._worker, "Test job")
        return self._worker

    def _on_done(self, ok, msg):
        self._worker = None

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        super().shutdown()


def test_workspace_job_helpers(qtbot):
    ws = _JobWorkspace()
    qtbot.addWidget(ws)
    worker = ws.run_job(5.0)
    assert ws.has_running_jobs()
    assert [j.name for j in ws.running_jobs()] == ["Test job"]
    ws.cancel_jobs()
    assert worker.cancel_called
    assert job_registry().wait_all(3000)
    assert not ws.has_running_jobs()


def _install_job_workspace(main_window) -> _JobWorkspace:
    ws = _JobWorkspace()
    main_window._stack.addWidget(ws)
    main_window._workspaces["__test__"] = ws
    return ws


def test_close_without_jobs_shuts_workspaces_down_once(main_window, qtbot):
    ws = _install_job_workspace(main_window)
    main_window.close()
    process_events(20)
    assert not main_window.isVisible()
    assert ws.shutdown_calls == 1


def test_close_with_running_job_asks_and_can_be_declined(main_window, dialogs, qtbot):
    ws = _install_job_workspace(main_window)
    worker = ws.run_job(30.0)
    dialogs.answer_no()
    main_window.close()
    process_events(20)
    assert main_window.isVisible(), "declining must keep the app open"
    assert not worker.cancel_called
    assert "Background tasks running" in dialogs.calls_of("question")[0].title
    # Tidy up: stop the job so teardown doesn't flag a running thread.
    ws.cancel_jobs()
    assert job_registry().wait_all(3000)


def test_close_with_running_job_cancels_waits_then_closes(main_window, dialogs, qtbot):
    ws = _install_job_workspace(main_window)
    worker = ws.run_job(30.0)
    dialogs.answer_yes()
    main_window.close()
    assert worker.cancel_called
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=5000)
    assert not worker.isRunning()
    assert ws.shutdown_calls == 1
    # The confirmation is asked once, not on every retry of close().
    assert len(dialogs.calls_of("question")) == 1
