"""Crash log viewer dialog.

Shows the tail of the crash log the Qt app writes (see
``crash_logger.install_for_qt``, installed by ``alpha_dex_gui.main``):
uncaught Python exceptions in ``~/.alphadex/crash.log`` and native crash
tracebacks (faulthandler) in ``~/.alphadex/faults.log``.
"""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets

_TAIL_LINES = 200


def crash_log_paths() -> list[Path]:
    """The files this viewer reads, in display order."""
    import crash_logger

    return [Path(crash_logger.default_log_path()), Path(crash_logger.default_fault_log_path())]


def _tail(path: Path, lines: int = _TAIL_LINES) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:]).strip()


class CrashLogDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crash Log")
        self.resize(700, 460)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        self._path_label = QtWidgets.QLabel()
        self._path_label.setWordWrap(True)
        self._path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self._path_label)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        root.addWidget(self._text)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)

        self._load_log()

    def _load_log(self) -> None:
        paths = crash_log_paths()
        self._path_label.setText(
            "Crash logs: " + "  ·  ".join(str(p) for p in paths)
        )
        sections = []
        for path in paths:
            body = _tail(path)
            if body:
                sections.append(f"── {path.name} ──\n{body}")
        if sections:
            self._text.setPlainText("\n\n".join(sections))
            self._text.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        else:
            self._text.setPlainText(f"No crash log found.\n\nAlphaDEX records crashes in {paths[0]}.")
