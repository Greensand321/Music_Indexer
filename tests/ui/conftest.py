"""pytest-qt harness for the AlphaDEX Qt app.

Every test in ``tests/ui/`` automatically gets:

* an isolated HOME / config (from ``tests/conftest.py``),
* ``dialogs`` — a DialogGuard: blocking dialogs never hang the run; a
  decision dialog the test didn't queue an answer for fails the test,
* ``threads`` — a ThreadRecorder: tests fail if they leave a QThread running,
* ``motion`` — a MotionRecorder: animations are recorded and, by default,
  compressed to ~1 ms so animated flows reach their end state immediately.
  Use ``@pytest.mark.motion(time_scale=1.0)`` for real-time motion checks,
* a Qt-log policy: a test fails if Qt logs a thread-safety / painting /
  bad-animation warning (see ``QtWarningPolicy``),
* theme-manager state reset between tests.

Fixtures for building the app: ``main_window``, ``make_workspace``, and
``library`` (a fresh synthetic, tagged music library on disk).

The whole directory is skipped when PySide6 or pytest-qt is missing, so the
backend suite still runs on minimal installs.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:  # Skip the GUI suite cleanly on machines without Qt.
    import PySide6  # noqa: F401
    import pytestqt  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    collect_ignore_glob = ["test_*.py"]
    HAVE_QT = False
else:
    HAVE_QT = True

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Threads that outlived their test. Holding a reference stops Python from
# destroying a still-running QThread (which would abort the interpreter with
# "QThread: Destroyed while thread is still running"); the test is failed.
_THREAD_GRAVEYARD: list = []


def pytest_unconfigure(config):
    """Before the interpreter exits, give threads that outlived their test a
    chance to finish: destroying a still-running QThread at exit is a qFatal
    abort, which would hide the real (already reported) test failure."""
    for thread in _THREAD_GRAVEYARD:
        try:
            thread.wait(60_000)
        except RuntimeError:
            pass


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "motion(time_scale=0.0): animation time scale for this test "
        "(0 = finish instantly, 1.0 = real time)",
    )
    config.addinivalue_line(
        "markers", "allow_qt_warnings(*patterns): regexes of Qt warnings this test tolerates"
    )
    config.addinivalue_line(
        "markers", "allow_running_threads: do not fail the test for threads still running at teardown"
    )


if HAVE_QT:
    from PySide6 import QtCore, QtWidgets
    from ui_harness import DialogGuard, MotionRecorder, QtWarningPolicy, ThreadRecorder

    class _UI:
        """Bundle of the per-test harness objects."""

        def __init__(self, dialogs, threads, motion):
            self.dialogs = dialogs
            self.threads = threads
            self.motion = motion

    @pytest.fixture(autouse=True)
    def _ui(request, qapp, qtlog, monkeypatch, _isolated_home):
        marker = request.node.get_closest_marker("motion")
        time_scale = marker.kwargs.get("time_scale", 0.0) if marker else 0.0

        dialogs = DialogGuard().install(monkeypatch)
        threads = ThreadRecorder().install(monkeypatch)
        motion = MotionRecorder(time_scale=time_scale).install(monkeypatch)

        # Modules that captured HOME-relative paths at import time.
        landing = sys.modules.get("gui.widgets.landing")
        if landing is not None and hasattr(landing, "_ArtHistory"):
            monkeypatch.setattr(
                landing._ArtHistory, "_PATH",
                str(Path(_isolated_home) / ".soundvault_art_history.json"),
            )

        ui = _UI(dialogs, threads, motion)
        yield ui

        # ── teardown ───────────────────────────────────────────────────────
        problems: list[str] = []

        # Close any windows the test left open (pytest-qt closes the ones
        # registered with qtbot.addWidget; this catches the rest).
        for w in list(QtWidgets.QApplication.topLevelWidgets()):
            try:
                if w.isVisible():
                    w.close()
            except RuntimeError:
                pass

        still_running = threads.wait_all(timeout_ms=8000)
        if still_running and not request.node.get_closest_marker("allow_running_threads"):
            _THREAD_GRAVEYARD.extend(threads.alive())
            problems.append(f"QThreads still running after the test: {still_running}")

        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete.value)
        QtWidgets.QApplication.processEvents()

        allow = request.node.get_closest_marker("allow_qt_warnings")
        policy = QtWarningPolicy()
        messages = [rec.message for rec in qtlog.records]
        bad = policy.violations(messages)
        if allow:
            import re
            allowed = re.compile("|".join(allow.args))
            bad = [m for m in bad if not allowed.search(m)]
        if bad:
            problems.append("Qt logged thread-safety/painting/animation warnings:\n  "
                            + "\n  ".join(bad[:20]))

        dialogs.uninstall()
        _reset_theme_manager()
        _reset_appearance_caches()
        gc.collect()

        if problems:
            pytest.fail("\n".join(problems), pytrace=False)

    def _settle_housekeeping(timeout_ms: int = 10_000) -> None:
        """Let the window's quiet housekeeping jobs (the backdrop build) finish,
        so a test starts from a settled window: job lists and thread counts
        are then the test's own."""
        import time

        from gui.workers import job_registry

        app = QtWidgets.QApplication.instance()
        deadline = time.monotonic() + timeout_ms / 1000
        while any(j.quiet for j in job_registry().jobs()) and time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
            time.sleep(0.005)
        app.processEvents()

    def _reset_appearance_caches() -> None:
        """Settings read once and cached (backdrop mode, reduced motion) must
        not carry one test's config into the next (results would depend on
        test order)."""
        appearance = sys.modules.get("gui.themes.appearance")
        if appearance is not None:
            appearance.invalidate()
        motion_mod = sys.modules.get("gui.themes.motion")
        if motion_mod is not None:
            motion_mod.set_reduced_motion(None)
            motion_mod.refresh_reduced_motion()
        backdrop = sys.modules.get("gui.widgets.backdrop")
        if backdrop is not None:
            backdrop.clear_caches()

    def _reset_theme_manager() -> None:
        """Put the ThemeManager singleton back to defaults without emitting."""
        mgr_mod = sys.modules.get("gui.themes.manager")
        if mgr_mod is None:
            return
        mgr = mgr_mod.ThemeManager._instance
        if mgr is None:
            return
        from gui.themes.tokens import (
            DEFAULT_ACCENT, DEFAULT_DARK, DEFAULT_LIGHT, DEFAULT_THEME, THEMES,
        )
        mgr._current = THEMES[DEFAULT_THEME]
        mgr._base = THEMES[DEFAULT_THEME]
        mgr._auto_mode = False
        mgr._auto_dark = DEFAULT_DARK
        mgr._auto_light = DEFAULT_LIGHT
        mgr._accent = DEFAULT_ACCENT
        mgr._reduce_transparency = False
        mgr._migration_note = None

    @pytest.fixture
    def dialogs(_ui) -> DialogGuard:
        return _ui.dialogs

    @pytest.fixture
    def threads(_ui) -> ThreadRecorder:
        return _ui.threads

    @pytest.fixture
    def motion(_ui) -> MotionRecorder:
        return _ui.motion

    @pytest.fixture
    def library(synthetic_library):
        """Alias: the synthetic library (``SyntheticLibrary``) for GUI tests."""
        return synthetic_library

    @pytest.fixture
    def main_window(qtbot, _ui):
        """A fully constructed, shown ``AlphaDEXWindow`` (no library selected)."""
        from gui.main_window import AlphaDEXWindow

        window = AlphaDEXWindow()
        qtbot.addWidget(window)
        window.resize(1300, 860)
        window.show()
        qtbot.waitExposed(window)
        _settle_housekeeping()
        yield window
        window.close()
        _ui.threads.wait_all(timeout_ms=8000)

    @pytest.fixture
    def make_workspace(qtbot):
        """Factory: ``make_workspace("indexer", library_path=...)`` → shown workspace."""
        from gui.main_window import _WORKSPACE_MAP

        created = []

        def factory(key: str, library_path: str | os.PathLike = "", show: bool = True):
            cls = _WORKSPACE_MAP[key]
            ws = cls(library_path=str(library_path) if library_path else "")
            qtbot.addWidget(ws)
            if show:
                ws.resize(1100, 760)
                ws.show()
            created.append(ws)
            return ws

        yield factory
