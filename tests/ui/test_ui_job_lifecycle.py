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


def test_read_only_jobs_are_cancelled_on_quit_without_asking(main_window, dialogs, qtbot):
    ws = _install_job_workspace(main_window)
    worker = SlowWorker(30.0)
    ws._start_worker(worker, "Read-only scan", ask_before_quit=False)
    main_window.close()          # no answer queued: asking would fail the test
    assert worker.cancel_called
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=5000)
    assert dialogs.calls_of("question") == []


def test_quit_asks_only_about_jobs_that_may_write(main_window, dialogs, qtbot):
    ws = _install_job_workspace(main_window)
    reader = SlowWorker(30.0)
    ws._start_worker(reader, "Read-only scan", ask_before_quit=False)
    writer = ws.run_job(30.0)    # default: ask before quit
    dialogs.answer_no()
    main_window.close()
    process_events(20)
    assert main_window.isVisible()
    text = dialogs.calls_of("question")[0].text
    assert "Test job" in text and "Read-only scan" not in text
    ws.cancel_jobs()
    assert job_registry().wait_all(3000)
    assert writer.cancel_called and reader.cancel_called


def test_messages_logged_during_construction_reach_the_log(qtbot):
    """SHELL-20: a page's construction-time warnings used to be dropped."""

    class Noisy(WorkspaceBase):
        def __init__(self):
            super().__init__()
            self._log("libVLC not found", "warn")

    ws = Noisy()
    qtbot.addWidget(ws)
    got = []
    ws.log_message.connect(lambda m, lvl: got.append((m, lvl)))
    ws.flush_pending_logs()
    assert got == [("libVLC not found", "warn")]
    ws._log("later", "info")            # once connected, emitted directly
    assert got[-1] == ("later", "info")
    ws.flush_pending_logs()             # nothing replayed twice
    assert len(got) == 2


def test_main_window_shows_construction_time_page_messages(main_window):
    # No libVLC in the test environment: the Player logs this while it is
    # being constructed, before the window connects its log_message signal.
    assert "Playback unavailable" in main_window._log_drawer._text.toPlainText()


def test_library_writes_are_mutually_exclusive_across_pages(qtbot, dialogs):
    """Two pages must never change files in the library at the same time."""
    writer_page = _JobWorkspace()
    other_page = _JobWorkspace()
    qtbot.addWidget(writer_page)
    qtbot.addWidget(other_page)

    assert other_page._library_writes_blocked("Indexer execute") is False

    reader = SlowWorker(30.0)
    writer_page._start_worker(reader, "Player library scan", ask_before_quit=False)
    assert other_page._library_writes_blocked("Indexer execute") is False, \
        "read-only jobs must not block writes"

    writer = SlowWorker(30.0)
    writer_page._start_worker(writer, "Library Sync execute", library_write=True)
    assert other_page._library_writes_blocked("Indexer execute") is True
    info = dialogs.calls_of("information")[-1]
    assert "Library Sync execute" in info.text and "Indexer execute" in info.text

    writer.cancel()
    reader.cancel()
    assert job_registry().wait_all(3000)
    # The group frees itself when the thread really finishes — no manual release.
    assert other_page._library_writes_blocked("Indexer execute") is False


def test_tag_write_guard_also_blocks_file_moves(qtbot, dialogs):
    from controllers import tag_write_guard

    page = _JobWorkspace()
    qtbot.addWidget(page)
    assert tag_write_guard.acquire("Tag Fixer apply")
    try:
        assert page._library_writes_blocked("Duplicate execute") is True
        assert "Tag Fixer apply" in dialogs.calls_of("information")[-1].text
    finally:
        tag_write_guard.release("Tag Fixer apply")


class StubbornWorker(SlowWorker):
    """Ignores cancel() — like a job stuck on a hung network drive."""

    def cancel(self) -> None:
        self.cancel_called = True     # but don't stop


def test_closing_locks_the_ui_and_cancels_jobs_started_later(main_window, qtbot):
    ws = _install_job_workspace(main_window)
    reader = SlowWorker(30.0)
    ws._start_worker(reader, "Read-only scan", ask_before_quit=False)
    main_window.close()
    assert not main_window.centralWidget().isEnabled(), "no new work while stopping"
    late = SlowWorker(30.0)          # e.g. a job chained by the cancelled one
    ws._start_worker(late, "Late writer")
    qtbot.waitUntil(lambda: late.cancel_called, timeout=3000)
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=5000)


def test_force_quit_is_offered_when_a_job_ignores_cancel(main_window, dialogs, monkeypatch, qtbot):
    ws = _install_job_workspace(main_window)
    stubborn = StubbornWorker(30.0)
    ws._start_worker(stubborn, "Stuck copy")
    monkeypatch.setattr(type(main_window), "CLOSE_PATIENCE_S", 0.2)
    forced = []
    monkeypatch.setattr(type(main_window), "_force_quit", staticmethod(lambda names: forced.append(names)))
    dialogs.answer_yes()          # "Stop them and quit?"
    dialogs.answer_no()           # "Quit anyway?" → keep waiting
    main_window.close()
    qtbot.waitUntil(lambda: len(dialogs.calls_of("question")) == 2, timeout=3000)
    assert forced == [] and main_window.isVisible()
    assert "Stuck copy" in dialogs.calls_of("question")[1].text
    process_events(400)
    assert len(dialogs.calls_of("question")) == 2, "asked once, not on every retry"
    stubborn._cancel.set()        # let the thread end so teardown is clean
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=5000)


def test_force_quit_yes_exits_without_interpreter_teardown(main_window, dialogs, monkeypatch, qtbot):
    ws = _install_job_workspace(main_window)
    stubborn = StubbornWorker(30.0)
    ws._start_worker(stubborn, "Stuck copy")
    monkeypatch.setattr(type(main_window), "CLOSE_PATIENCE_S", 0.0)
    forced = []
    monkeypatch.setattr(type(main_window), "_force_quit", staticmethod(lambda names: forced.append(names)))
    dialogs.answer_yes().answer_yes()
    main_window.close()
    qtbot.waitUntil(lambda: forced == ["Stuck copy"], timeout=3000)
    stubborn._cancel.set()
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=5000)


def test_registry_ignores_a_stale_finish_for_a_reused_key(qtbot):
    registry = job_registry()
    current = SlowWorker(30.0)
    start_worker(current, name="current")
    key = id(current)
    stale = QtCore.QThread()      # a different thread claiming the same key
    registry._release(key, stale)
    assert [j.name for j in registry.jobs()] == ["current"], "stale release must be ignored"
    current.cancel()
    assert registry.wait_all(3000)


def test_start_detaches_worker_from_a_widget_parent(qtbot):
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    worker = SlowWorker(0.05)
    worker.setParent(host)
    start_worker(worker, name="parented")
    assert worker.parent() is None, "deleting the widget must not destroy a running thread"
    assert job_registry().wait_all(3000)
