"""AlphaDEX Qt GUI — entry point.

Run with:
    python alpha_dex_gui.py

This launches the PySide6 Qt Widgets rebuild of the AlphaDEX desktop app
using the Option A Navigator + Workspace layout.

The original Tkinter app remains available at:
    python main_gui.py
"""
from __future__ import annotations

import sys
import os

# Ensure the repo root is on sys.path so backend modules are importable
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    from gui.compat import QtWidgets, QtGui
    from gui.main_window import AlphaDEXWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("AlphaDEX")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AlphaDEX")

    window = AlphaDEXWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
