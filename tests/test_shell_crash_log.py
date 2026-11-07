"""crash_logger.install_for_qt: the Qt app's crash log (SHELL-21), and gui.compat laziness (SHELL-24)."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import types

import pytest

import crash_logger


@pytest.fixture
def qt_crash_log(tmp_path, monkeypatch):
    enabled = []
    monkeypatch.setattr(
        crash_logger, "faulthandler",
        types.SimpleNamespace(enable=lambda **kw: enabled.append(kw)),
    )
    monkeypatch.setattr(crash_logger, "_qt_state", {})
    prev_hooks = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: prev_hooks.append(("sys", a[0])))
    monkeypatch.setattr(threading, "excepthook", lambda args: prev_hooks.append(("thread", args.exc_type)))
    path = crash_logger.install_for_qt()
    yield path, prev_hooks, enabled
    handler = crash_logger._qt_state.get("handler")
    if handler is not None:
        import logging
        logging.getLogger("alphadex.crash").removeHandler(handler)
        handler.close()
    f = crash_logger._qt_state.get("fault_file")
    if f is not None:
        f.close()


def test_default_paths_live_under_home(tmp_path):
    home = os.path.expanduser("~")
    assert crash_logger.default_log_path() == os.path.join(home, ".alphadex", "crash.log")
    assert crash_logger.default_fault_log_path() == os.path.join(home, ".alphadex", "faults.log")


def test_uncaught_exception_is_written_and_chained(qt_crash_log):
    path, prev_hooks, enabled = qt_crash_log
    assert path == crash_logger.default_log_path()
    try:
        raise ValueError("kaboom in a slot")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    text = open(path, encoding="utf-8").read()
    assert "ValueError: kaboom in a slot" in text
    assert "Unhandled exception" in text
    assert prev_hooks == [("sys", ValueError)]          # stderr / pytest-qt still see it
    assert enabled and enabled[0]["all_threads"] is True


def test_worker_thread_exception_is_written(qt_crash_log):
    path, prev_hooks, _enabled = qt_crash_log

    def boom():
        raise RuntimeError("worker died")

    t = threading.Thread(target=boom, name="ScanWorker")
    t.start()
    t.join()
    text = open(path, encoding="utf-8").read()
    assert "RuntimeError: worker died" in text and "[ScanWorker]" in text
    assert ("thread", RuntimeError) in prev_hooks


def test_install_is_idempotent(qt_crash_log):
    path, _prev, _enabled = qt_crash_log
    hook = sys.excepthook
    assert crash_logger.install_for_qt() == path
    assert sys.excepthook is hook


def test_crash_logger_does_not_import_tkinter():
    code = "import sys, crash_logger, crash_watcher; print('tkinter' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.stdout.strip() == "False", out.stderr


def test_compat_does_not_import_webengine_eagerly():
    pytest.importorskip("PySide6")
    code = (
        "import sys, gui.compat as c; "
        "print(any(m.endswith(('QtWebEngineWidgets', 'QtWebChannel')) for m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    assert out.stdout.strip() == "False", out.stderr
