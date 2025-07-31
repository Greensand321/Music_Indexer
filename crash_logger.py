"""Enhanced crash logger with context and threaded exception support."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
import platform
try:
    import resource
except ModuleNotFoundError:  # resource module unavailable on Windows
    resource = None
import shutil
import sys
import threading
import traceback

from typing import Callable, Dict, List


_context_providers: List[Callable[[], Dict[str, object]]] = []


def add_context_provider(func: Callable[[], Dict[str, object]]) -> None:
    """Register a callable returning additional context for crash logs."""

    _context_providers.append(func)


def install(log_path: str = "crash.log", *, level: int = logging.INFO) -> None:
    """Install global crash logging.

    Parameters
    ----------
    log_path:
        File path for the rotating crash log.
    level:
        Logging level for the root logger. Can be changed later via
        :func:`toggle_debug_mode`.
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
        context = _gather_context()
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical("Unhandled exception:\n%s\nContext: %s", msg, context)
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

