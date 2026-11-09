"""Python's cyclic garbage collector runs on the GUI thread only.

Why
---
CPython runs a collection in whichever thread allocates the object that
crosses the collector's threshold. In this app that is often a worker thread
(a tag loader, a scan, the backdrop build), and a collection frees *every*
unreachable cycle, including Qt objects that live on the GUI thread: a dialog
dropped with a lambda that captures it, a widget from an earlier page build,
a window a test let go of. Their C++ destructors then run in the worker.

Deleting a GUI-thread QObject from another thread is undefined behaviour in
Qt, and for a running animation it is a crash: the destructor unregisters the
animation from the *worker's* animation timer (Qt keeps one per thread), so
the GUI thread's timer keeps a dangling pointer, and its next frame segfaults
in ``QUnifiedTimer::updateAnimationTimers`` ->
``QAbstractAnimation::setCurrentTime``. That is the native crash seen while
the Now Playing capsule faded in as a tag-loader thread finished (R2); a
worker that merely allocates while a collectable GUI object animates
reproduces it every time (``tests/ui/test_ui_stability.py`` is that
scenario, and segfaults without this guard).

What
----
``install()`` switches automatic collection off and collects from a timer on
the GUI thread instead, one generation at a time with Python's own
thresholds, the way the interpreter would have (pyqtgraph ships the same
guard for the same reason). Objects freed by reference counting are
unaffected: only cycles wait, at most ``INTERVAL_MS``. Call it once, on the
GUI thread, after the QApplication exists (``alpha_dex_gui._prepare_app``,
and the GUI test harness for the whole session).
"""
from __future__ import annotations

import gc

from gui.compat import QtCore

INTERVAL_MS = 500
_OBJECT_NAME = "alphadexGcGuard"


class GuiThreadCollector(QtCore.QObject):
    """Runs Python's cyclic collector from a GUI-thread timer."""

    def __init__(self, parent: QtCore.QObject, interval_ms: int = INTERVAL_MS) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)
        self._threshold = gc.get_threshold()
        self._was_enabled = gc.isenabled()
        self.collections = 0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.check)
        gc.disable()
        self._timer.start()

    def check(self) -> int:
        """Collect the oldest generation over its threshold (like CPython's
        own trigger); returns the generation collected, or -1."""
        counts = gc.get_count()
        for gen in range(min(len(counts), len(self._threshold)) - 1, -1, -1):
            threshold = self._threshold[gen]
            if threshold and counts[gen] > threshold:
                gc.collect(gen)
                self.collections += 1
                return gen
        return -1

    def stop(self) -> None:
        """Back to automatic collection (tests)."""
        self._timer.stop()
        if self._was_enabled:
            gc.enable()


def collector() -> GuiThreadCollector | None:
    app = QtCore.QCoreApplication.instance()
    if app is None:
        return None
    return app.findChild(GuiThreadCollector, _OBJECT_NAME)


def install(interval_ms: int = INTERVAL_MS) -> GuiThreadCollector | None:
    """Collect cycles on the GUI thread only (idempotent). Needs the
    application object and must be called from its thread; returns the
    collector, or ``None`` when there is no application yet."""
    app = QtCore.QCoreApplication.instance()
    if app is None:
        return None
    if QtCore.QThread.currentThread() is not app.thread():
        raise RuntimeError("gc_guard.install() must run on the GUI thread")
    existing = collector()
    if existing is not None:
        return existing
    return GuiThreadCollector(app, interval_ms)


def uninstall() -> None:
    guard = collector()
    if guard is not None:
        guard.stop()
        guard.setParent(None)
        guard.deleteLater()
