"""Stability: garbage collection stays on the GUI thread; windows close cleanly.

The R2 native crash (``QUnifiedTimer::updateAnimationTimers`` ->
``QAbstractAnimation::setCurrentTime`` on a freed animation) came from
Python's cyclic garbage collector running in a worker thread: a collection
that a worker's allocations trigger frees whatever cycles exist, including
GUI objects from an earlier window, and their C++ destructors then run in
that worker. A running animation unregisters from the *worker's* animation
timer (Qt keeps one per thread), so the GUI thread's timer keeps a dangling
pointer. ``gui.gc_guard`` moves every collection onto the GUI thread.
"""
from __future__ import annotations

import gc
import threading

import pytest
from PySide6 import QtCore, QtWidgets

from gui import gc_guard
from ui_harness import process_events


class _Allocator(QtCore.QThread):
    """A worker doing ordinary Python work: enough cyclic allocations to
    cross the collector's thresholds many times over."""

    def run(self) -> None:
        junk = []
        for i in range(120_000):
            d = {"i": i}
            d["self"] = d
            junk.append(d)


def test_the_gui_test_session_runs_with_the_guard(qapp):
    guard = gc_guard.collector()
    assert guard is not None, "conftest installs gui.gc_guard for the session"
    assert not gc.isenabled(), "automatic (any-thread) collection is off"
    assert guard.thread() is qapp.thread()


def test_install_is_idempotent_and_gui_thread_only(qapp):
    assert gc_guard.install() is gc_guard.collector()
    errors = []

    def other_thread():
        try:
            gc_guard.install()
        except RuntimeError as exc:
            errors.append(exc)

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()
    assert errors, "installing from a worker thread is refused"


def test_cycles_are_collected_on_the_gui_thread_only(qapp):
    """A worker that allocates heavily never runs a collection itself; the
    guard collects the garbage on the GUI thread afterwards. (Without the
    guard this test is the crash: a segfault on the next animation tick.)"""
    assert not gc.isenabled(), "gui.gc_guard must be installed (it is the fix under test)"
    gui_ident = threading.get_ident()
    seen: list[int] = []

    def on_gc(phase, _info):
        if phase == "start":
            seen.append(threading.get_ident())

    class Owner(QtCore.QObject):
        def __init__(self):
            super().__init__()
            self.anim = QtCore.QVariantAnimation(self)
            self.anim.setDuration(60_000)
            self.anim.setStartValue(0.0)
            self.anim.setEndValue(1.0)
            self.anim.valueChanged.connect(self._tick)
            self.me = self                    # a cycle: only the collector frees it
            self.anim.start()

        def _tick(self, _v):
            pass

    owners = [Owner() for _ in range(5)]
    freed: list[int] = []
    for o in owners:
        o.anim.destroyed.connect(lambda *_: freed.append(1))
    del owners, o
    gc.callbacks.append(on_gc)
    try:
        worker = _Allocator()
        worker.start()
        assert worker.wait(30_000)
        assert seen == [], "no collection ran while only the worker allocated"
        guard = gc_guard.collector()
        deadline = QtCore.QDeadlineTimer(5_000)
        while len(freed) < 5 and not deadline.hasExpired():
            process_events(50)
            guard.check()
    finally:
        gc.callbacks.remove(on_gc)
    assert seen and set(seen) == {gui_ident}, "every collection ran on the GUI thread"
    assert len(freed) == 5, "the running animations' owners were freed"
    process_events(50)        # a dangling animation would crash the next tick


def test_alpha_dex_gui_installs_the_guard_at_startup(qapp, monkeypatch):
    import alpha_dex_gui

    calls = []
    monkeypatch.setattr(gc_guard, "install", lambda *a, **k: calls.append(1))
    monkeypatch.setattr("gui.themes.manager.ThemeManager.load_persisted", lambda self: None)
    alpha_dex_gui._prepare_app(qapp)
    assert calls == [1]


_QUIT_SCRIPT = r"""
import os, sys
sys.path[:0] = [{repo!r}, os.path.join({repo!r}, "tests"), os.path.join({repo!r}, "tests", "ui")]
from PySide6 import QtWidgets
app = QtWidgets.QApplication(sys.argv[:1])
from gui import gc_guard
gc_guard.install()
from gui.main_window import AlphaDEXWindow
from gui.workers import job_registry
from ui_harness import process_events
window = AlphaDEXWindow()
window.resize(1300, 860)
window.show()
window.set_library({library!r})
job_registry().wait_all(20000)
for key in ("player", "playlists", "sound_map", "home"):
    window._on_nav_changed(key)
    process_events(40)
job_registry().wait_all(20000)
window._closing = True
window.close()
job_registry().wait_all(20000)
print("closed", flush=True)
"""


def test_quitting_the_app_raises_nothing_on_the_way_out(tmp_path, library):
    """Quit with the window still referenced, as ``main()`` does: the
    interpreter's shutdown then destroys the window, and its C++ teardown
    used to call into the Player's event filter and the Playlists repair
    list's model hook after their widgets' wrappers were gone, printing a
    RuntimeError traceback per event on every quit."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    script = tmp_path / "quit.py"
    script.write_text(_QUIT_SCRIPT.format(repo=str(repo), library=str(library.root)))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", HOME=str(tmp_path / "home"),
               USERPROFILE=str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    proc = subprocess.run([sys.executable, str(script)], cwd=str(repo), env=env,
                          capture_output=True, text=True, timeout=120)
    assert "closed" in proc.stdout, proc.stderr[-2000:]
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Traceback" not in proc.stderr and "RuntimeError" not in proc.stderr, \
        proc.stderr[-2000:]
