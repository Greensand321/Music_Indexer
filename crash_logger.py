"""Enhanced crash logger with context and threaded exception support."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import platform
import functools
import time
try:
    import resource
except ModuleNotFoundError:  # resource module unavailable on Windows
    resource = None
import shutil
import sys
import threading
import traceback

from typing import Callable, Dict, List, TypeVar
from typing_extensions import ParamSpec

import crash_watcher

import faulthandler


_context_providers: List[Callable[[], Dict[str, object]]] = []


P = ParamSpec("P")
R = TypeVar("R")


class _WatcherHelper:
    """Helper exposing decorators for crash event instrumentation."""

    def traced(self, func: Callable[P, R]) -> Callable[P, R]:
        """Decorator logging entry and exit of ``func``."""

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            name = func.__qualname__
            crash_watcher.record_event(f"enter {name}")
            try:
                return func(*args, **kwargs)
            finally:
                crash_watcher.record_event(f"exit {name}")

        return wrapper


watcher = _WatcherHelper()


def add_context_provider(func: Callable[[], Dict[str, object]]) -> None:
    """Register a callable returning additional context for crash logs."""

    _context_providers.append(func)


def install(
    log_path: str = "crash.log",
    *,
    level: int = logging.INFO,
    show_dialog: bool = True,
) -> None:
    """Install global crash logging.

    Parameters
    ----------
    log_path:
        File path for the rotating crash log.
    level:
        Logging level for the root logger. Can be changed later via
        :func:`toggle_debug_mode`.
    show_dialog:
        Show a Tkinter error box on a crash (legacy CLI / Tk app only). The
        Qt app uses :func:`install_for_qt` instead.
    """

    logger = logging.getLogger()
    logger.setLevel(level)
    handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    fmt = "%(asctime)s [%(name)s] %(levelname)s %(funcName)s:%(lineno)d - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    _log_startup_context(logger)

    def _handle(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        stack = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        thread_name = threading.current_thread().name
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        events = crash_watcher.dump_events()
        context = _gather_context()
        summary = (
            f"{ts} [{thread_name}] Unhandled exception:\n{stack}\nRecent events:\n{events}\nContext: {context}"
        )
        logger.critical(summary)

        tk, messagebox = crash_watcher._tk_messagebox() if show_dialog else (None, None)
        if tk and messagebox:
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Crash Report", summary)
                root.destroy()
            except Exception:
                pass

        crash_watcher.mark_clean_shutdown()
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _handle_thread_exception(args: threading.ExceptHookArgs) -> None:
        _handle(args.exc_type, args.exc_value, args.exc_traceback)

    sys.excepthook = _handle
    threading.excepthook = _handle_thread_exception


def toggle_debug_mode() -> None:
    """Raise logging level to DEBUG at runtime."""

    logging.getLogger().setLevel(logging.DEBUG)


def _gather_context() -> Dict[str, object]:
    """Collect runtime context information."""

    ctx: Dict[str, object] = {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "memory_kb": (
            getattr(resource.getrusage(resource.RUSAGE_SELF), "ru_maxrss", "N/A")
            if resource is not None
            else "N/A"
        ),
        "fpcalc": shutil.which("fpcalc"),
        "ffmpeg": shutil.which("ffmpeg"),
    }
    for fn in _context_providers:
        try:
            ctx.update(fn())
        except Exception:
            pass
    return ctx


def _log_startup_context(logger: logging.Logger) -> None:
    ctx = _gather_context()
    logger.info("Application start")
    for k, v in ctx.items():
        logger.info("%s: %s", k, v)



# ─────────────────────────────────────────────────────────────────────────────
# Qt app crash log
# ─────────────────────────────────────────────────────────────────────────────
#
# The Qt app (``alpha_dex_gui.py``) records every uncaught exception — on the
# GUI thread (including exceptions raised inside Qt slots, which PySide routes
# through ``sys.excepthook``) and on worker threads — plus hard crashes
# (segfaults, aborts) via ``faulthandler``, to a fixed per-user location that
# the Utilities → "View Crash Log…" dialog reads.

CRASH_LOG_NAME = "crash.log"
FAULT_LOG_NAME = "faults.log"
_CRASH_LOGGER_NAME = "alphadex.crash"
_qt_state: Dict[str, object] = {}


def crash_log_dir() -> str:
    """Folder holding the Qt app's crash logs (computed from HOME at call time)."""
    return os.path.join(os.path.expanduser("~"), ".alphadex")


def default_log_path() -> str:
    """The Python-exception crash log the Crash Log viewer reads."""
    return os.path.join(crash_log_dir(), CRASH_LOG_NAME)


def default_fault_log_path() -> str:
    """Where ``faulthandler`` writes native crash tracebacks."""
    return os.path.join(crash_log_dir(), FAULT_LOG_NAME)


def format_crash(exc_type, exc_value, exc_tb, thread_name: str | None = None) -> str:
    stack = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    name = thread_name or threading.current_thread().name
    events = crash_watcher.dump_events()
    text = f"{ts} [{name}] Unhandled exception:\n{stack}"
    if events:
        text += f"Recent events:\n{events}\n"
    return text


def install_for_qt(log_path: str | None = None, fault_path: str | None = None) -> str:
    """Record uncaught exceptions and hard crashes for the Qt app.

    Idempotent. Chains to the previously installed hooks, so tracebacks still
    reach stderr (and test harnesses that wrap ``sys.excepthook`` keep
    working). Writes only to a dedicated logger, never the root logger, so
    ordinary app logging does not flood the crash log. Returns the log path.
    """
    log_path = log_path or default_log_path()
    fault_path = fault_path or default_fault_log_path()
    if _qt_state.get("log_path") == log_path:
        return log_path
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    logger = logging.getLogger(_CRASH_LOGGER_NAME)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    old = _qt_state.get("handler")
    if isinstance(old, logging.Handler):
        logger.removeHandler(old)
        old.close()
    handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=2, encoding="utf-8", delay=True
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    _qt_state["handler"] = handler

    prev_excepthook = sys.excepthook
    prev_thread_hook = threading.excepthook

    def _excepthook(exc_type, exc_value, exc_tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            try:
                logger.error(format_crash(exc_type, exc_value, exc_tb))
                handler.flush()
            except Exception:  # never let crash logging raise
                pass
        prev_excepthook(exc_type, exc_value, exc_tb)

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not SystemExit:
            try:
                name = args.thread.name if args.thread is not None else None
                logger.error(format_crash(args.exc_type, args.exc_value, args.exc_traceback, name))
                handler.flush()
            except Exception:
                pass
        prev_thread_hook(args)

    sys.excepthook = _excepthook
    threading.excepthook = _thread_hook

    try:
        os.makedirs(os.path.dirname(os.path.abspath(fault_path)), exist_ok=True)
        fault_file = open(fault_path, "a", encoding="utf-8")
        faulthandler.enable(file=fault_file, all_threads=True)
        old_file = _qt_state.get("fault_file")
        _qt_state["fault_file"] = fault_file  # must stay open for faulthandler
        if old_file is not None:
            try:
                old_file.close()  # type: ignore[union-attr]
            except Exception:
                pass
    except Exception:
        pass

    _qt_state["log_path"] = log_path
    return log_path
