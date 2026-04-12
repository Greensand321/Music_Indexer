"""Crash log viewer dialog."""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtWidgets, Slot


class CrashLogDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Crash Log")
        self.resize(600, 400)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)

        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        root.addWidget(self._text)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)

        self._load_log()

    def _load_log(self) -> None:
        log_path = Path.home().parent.parent / "soundvault_crash.log"
        # Try several common locations
        candidates = [
            Path.home() / "soundvault_crash.log",
            Path("soundvault_crash.log"),
        ]
        for c in candidates:
            if c.exists():
                try:
                    lines = c.read_text(errors="replace").splitlines()
                    self._text.setPlainText("\n".join(lines[-50:]))
                    return
                except Exception:
                    pass
        self._text.setPlainText("No crash log found.")
