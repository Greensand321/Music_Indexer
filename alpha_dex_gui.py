"""AlphaDEX Qt GUI — entry point.

Run with:
    python alpha_dex_gui.py

This launches the PySide6 Qt Widgets rebuild of the AlphaDEX desktop app
using the Option A Navigator + Workspace layout.

The original Tkinter app remains available at:
    python main_gui.py

Start-up sequence
-----------------
1. ``SplashScreen`` — progressive loading bar:
   - Phase 1 (750ms): Fast fill to 50% while UI loads.
   - Phase 2: Monitored wait while images load; bar tracks real progress from
     50% to 100%, or times out after 5 seconds.
   - Fade-out: 450ms fade.
2. ``MosaicLanding`` — full-window animated mosaic of album-art tiles with a
   frosted-glass CTA card.  The user selects (or confirms) their music library
   here.  Tiles scatter and the landing cross-fades into the main window.
   Image loading progress is reported back to the splash in real-time.
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

    # ── Deferred main window construction ──────────────────────────────────────
    # Window construction is deferred until after the splash has shown,
    # preventing blocking I/O from delaying the splash animation.
    # The window is constructed on the next event loop iteration.
    window: object = None  # Will be set by _construct_main_window

    # ── Compute shared geometry ──────────────────────────────────────────────────
    # The landing and main window use the same geometry so cross-fade is seamless.
    screen = app.primaryScreen()
    sg = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
    lw, lh = 1300, 860
    lx = sg.left() + (sg.width()  - lw) // 2
    ly = sg.top()  + (sg.height() - lh) // 2
    shared_geo = QtCore.QRect(lx, ly, lw, lh)

    # ── Load saved library path for the landing's "Continue" button ───────────
    saved_lib = ""
    try:
        from config import load_config
        saved_lib = load_config().get("library_root", "")
    except FileNotFoundError:
        # Config doesn't exist yet; normal on first startup
        pass
    except Exception as e:
        # Log real errors for debugging, but continue with defaults
        print(f"[Warning] Failed to load saved config: {e}", file=sys.stderr)

    # ── Landing page ──────────────────────────────────────────────────────────
    from gui.widgets.landing import MosaicLanding, FADE_OUT_MS
    landing = MosaicLanding(shared_geo, saved_lib)

    # ── Wire splash to landing for progressive image loading ────────────────
    # The splash bar will now show real progress: 0-50% for UI load, then
    # 50-100% as images are loaded, with a 5-second timeout if images take
    # longer than expected.
    landing.wire_splash_progress(splash)

    # ── Deferred main window construction ──────────────────────────────────────
    def _construct_main_window() -> None:
        """Build the main window after the splash has shown.

        This is scheduled for the next event loop iteration to prevent the
        window construction from blocking the splash animation. The window is
        not shown until the landing cross-fade completes.
        """
        nonlocal window
        from gui.main_window import AlphaDEXWindow
        window = AlphaDEXWindow()
        window.setGeometry(shared_geo)
        # Don't show yet; will be shown by _landing_to_main

    # Schedule window construction for next idle moment (after splash.show() completes)
    QtCore.QTimer.singleShot(0, _construct_main_window)

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
        # Self-cleanup: delete animation when finished to prevent memory leak
        fade.finished.connect(fade.deleteLater)
        fade.start()

    splash.reveal_ready.connect(_splash_to_landing)

    # ── Cross-fade: landing → main window ─────────────────────────────────────
    def _landing_to_main(path: str) -> None:
        """Called when ``library_selected`` fires at the *start* of the landing's
        fade-out.  Fade the main window in over the same duration so both
        windows cross-dissolve simultaneously.

        The main window may still be under construction if landing selection
        happens very quickly; wait if needed.
        """
        # Wait for window to be constructed if needed
        if window is None:
            # Window construction should be nearly complete by now (scheduled at
            # QTimer.singleShot(0)). If it's not ready, schedule this callback again.
            QtCore.QTimer.singleShot(10, lambda: _landing_to_main(path))
            return

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
        # Self-cleanup: delete animation when finished to prevent memory leak
        fade.finished.connect(fade.deleteLater)
        fade.start()

    landing.library_selected.connect(_landing_to_main)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
