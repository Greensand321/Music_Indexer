import logging
import sys
import traceback


def install(log_path: str = "crash.log") -> None:
    """Log unhandled exceptions to ``log_path``.

    The log file is overwritten on each run.
    """
    logger = logging.getLogger("crash")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)

    def _handle(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical("Unhandled exception:\n%s", msg)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _handle
