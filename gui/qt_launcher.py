"""Standalone Qt preview window launcher."""
from __future__ import annotations

import sys


QtWidgets = None
QtCore = None

try:  # Prefer PySide6 when available.
    from PySide6 import QtCore, QtWidgets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    try:
        from PyQt6 import QtCore, QtWidgets  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "PySide6 or PyQt6 is required to launch the Qt preview window. "
            "Install one of them and try again."
        ) from exc


class QtPreviewWindow(QtWidgets.QMainWindow):
    """Simple placeholder Qt window for incremental migration."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AlphaDEX Qt Preview")
        self.resize(900, 600)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)

        heading = QtWidgets.QLabel("Qt Preview Window")
        heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")

        body = QtWidgets.QLabel(
            "This window is launched from the Tkinter app to begin a staged Qt migration.\n"
            "Add new Qt panels here as they are ported from the Tk interface."
        )
        body.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        body.setWordWrap(True)

        layout.addStretch(1)
        layout.addWidget(heading)
        layout.addSpacing(12)
        layout.addWidget(body)
        layout.addStretch(2)

        self.setCentralWidget(container)


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = QtPreviewWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
