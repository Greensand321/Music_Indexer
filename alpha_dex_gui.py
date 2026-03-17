"""AlphaDEX Qt GUI — entry point.

Run with:
    python alpha_dex_gui.py

This launches the PySide6 Qt Widgets rebuild of the AlphaDEX desktop app
using the Option A Navigator + Workspace layout.

The original Tkinter app remains available at:
    python main_gui.py

Start-up sequence
-----------------
1. ``SplashScreen`` — branded loading bar (1 500 ms fill + 450 ms fade-out).
2. ``MosaicLanding`` — full-window animated mosaic of album-art tiles with a
   frosted-glass CTA card.  The user selects (or confirms) their music library
   here.  Tiles scatter and the landing cross-fades into the main window.
3. ``AlphaDEXWindow`` — main application interface.
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

    # Load the persisted theme so the splash and landing use the correct palette.
    # AlphaDEXWindow calls load_persisted() again — harmless.
    from gui.themes.manager import get_manager
    get_manager().load_persisted()

    # ── Splash ────────────────────────────────────────────────────────────────
    from gui.widgets.splash import SplashScreen, _FADE_MS
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    # ── Main window (built now, shown only after the landing is done) ─────────
    from gui.main_window import AlphaDEXWindow
    window = AlphaDEXWindow()

    # Compute a centred geometry shared by the landing and the main window so
    # the cross-fade from landing → main is perfectly seamless (same rect).
    screen = app.primaryScreen()
    sg = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
    lw, lh = 1300, 860
    lx = sg.left() + (sg.width()  - lw) // 2
    ly = sg.top()  + (sg.height() - lh) // 2
    shared_geo = QtCore.QRect(lx, ly, lw, lh)
    window.setGeometry(shared_geo)

    # ── Load saved library path for the landing's "Continue" button ───────────
    saved_lib = ""
    try:
        from config import load_config
        saved_lib = load_config().get("library_root", "")
    except Exception:
        pass

    # ── Landing page ──────────────────────────────────────────────────────────
    from gui.widgets.landing import MosaicLanding, FADE_OUT_MS
    landing = MosaicLanding(shared_geo, saved_lib)

    # ── Cross-fade: splash → landing ──────────────────────────────────────────
    def _splash_to_landing() -> None:
        """Splash has begun its own fade-out; simultaneously fade the landing in.
        Once the landing reaches full opacity the tiles fly in automatically."""
        landing.setWindowOpacity(0.0)
        landing.show()

        fade = QtCore.QVariantAnimation(landing)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setDuration(_FADE_MS)              # match splash fade duration
        fade.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        fade.valueChanged.connect(lambda v: landing.setWindowOpacity(float(v)))
        fade.finished.connect(landing._fly_in)
        # Keep a reference so the animation is not GC-collected
        _splash_to_landing._anim = fade         # type: ignore[attr-defined]
        fade.start()

    splash.reveal_ready.connect(_splash_to_landing)

    # ── Cross-fade: landing → main window ─────────────────────────────────────
    def _landing_to_main(path: str) -> None:
        """Called when ``library_selected`` fires at the *start* of the landing's
        fade-out.  Fade the main window in over the same duration so both
        windows cross-dissolve simultaneously."""
        if path:
            window.set_library(path)

        window.setWindowOpacity(0.0)
        window.show()

        fade = QtCore.QVariantAnimation(window)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setDuration(FADE_OUT_MS)           # match landing fade-out
        fade.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        fade.valueChanged.connect(lambda v: window.setWindowOpacity(float(v)))
        # Keep a reference
        _landing_to_main._anim = fade           # type: ignore[attr-defined]
        fade.start()

    landing.library_selected.connect(_landing_to_main)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
