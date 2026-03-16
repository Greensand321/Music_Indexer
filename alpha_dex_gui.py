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
    from gui.compat import QtWidgets, QtGui, QtCore

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("AlphaDEX")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AlphaDEX")

    # High-DPI support
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Register Inter + JetBrains Mono before any widget is created
    from gui.fonts import load_fonts
    load_fonts()

    # Set the application icon (Windows taskbar, title bar, Alt-Tab).
    # Done after QApplication is constructed so the icon engine is ready.
    from gui.icons import make_app_icon
    app.setWindowIcon(make_app_icon())

    # Load the persisted theme now so the splash uses the correct palette.
    # AlphaDEXWindow will call load_persisted() again, which is harmless.
    from gui.themes.manager import get_manager
    get_manager().load_persisted()

    # Show themed splash immediately, build main window in background.
    from gui.widgets.splash import SplashScreen, _FADE_MS
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    from gui.main_window import AlphaDEXWindow
    window = AlphaDEXWindow()

    # Cross-fade: when the splash begins fading out, simultaneously fade the
    # main window *in* over the same duration so both transitions overlap.
    def _start_cross_fade() -> None:
        window.setWindowOpacity(0.0)
        window.show()
        fade_in = QtCore.QVariantAnimation(window)   # parented → won't be GC'd
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setDuration(_FADE_MS)
        fade_in.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        fade_in.valueChanged.connect(
            lambda v: window.setWindowOpacity(float(v))
        )
        fade_in.start()

    splash.reveal_ready.connect(_start_cross_fade)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
