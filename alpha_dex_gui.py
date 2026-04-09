"""AlphaDEX Qt GUI — entry point.

Run with:
    python alpha_dex_gui.py

This launches the PySide6 Qt Widgets rebuild of the AlphaDEX desktop app
using the Option A Navigator + Workspace layout.

The original Tkinter app remains available at:
    python main_gui.py

Start-up sequence
-----------------
1. ``MosaicLanding`` — shown immediately.
   - Window fades in with only the CTA card visible (logo moment, ~320 ms).
   - Brief pause (~600 ms) so the user reads the brand name.
   - Tiles fly in from off-screen; album art populates them in the background.
   - User selects (or confirms) their music library via the CTA card.
   - Tiles scatter; landing cross-fades into the main window.
2. ``AlphaDEXWindow`` — main application interface.
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

    # Load the persisted theme so the landing uses the correct palette.
    # AlphaDEXWindow calls load_persisted() again — harmless.
    from gui.themes.manager import get_manager
    get_manager().load_persisted()

    # ── Compute shared geometry ───────────────────────────────────────────────
    # Landing and main window share the same rect so the cross-fade is seamless.
    screen = app.primaryScreen()
    sg = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
    lw, lh = 1300, 860
    lx = sg.left() + (sg.width()  - lw) // 2
    ly = sg.top()  + (sg.height() - lh) // 2
    shared_geo = QtCore.QRect(lx, ly, lw, lh)

    # ── Load saved library path for the landing's "Go" button ────────────────
    saved_lib = ""
    try:
        from config import load_config
        saved_lib = load_config().get("library_root", "")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[Warning] Failed to load saved config: {e}", file=sys.stderr)

    # ── Landing page ──────────────────────────────────────────────────────────
    from gui.widgets.landing import MosaicLanding, FADE_OUT_MS
    landing = MosaicLanding(shared_geo, saved_lib)

    # ── Deferred main window construction ────────────────────────────────────
    # Scheduled for the next event loop tick so the landing can appear and begin
    # its fade-in before the heavier AlphaDEXWindow import runs.
    window: object = None
    _play_dir: str = ""

    def _construct_main_window() -> None:
        nonlocal window
        from gui.main_window import AlphaDEXWindow
        window = AlphaDEXWindow()
        window.setGeometry(shared_geo)

    QtCore.QTimer.singleShot(0, _construct_main_window)

    # Show the landing immediately — it handles its own fade-in + logo pause
    # + tile fly-in sequence internally via show_animated().
    landing.show_animated()

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

        # If the user clicked a tile, navigate to Player and start playing
        # before the window fades in so it's ready when it appears.
        if _play_dir:
            window.play_directory(_play_dir)

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

    def _on_tile_clicked(dirpath: str) -> None:
        nonlocal _play_dir
        _play_dir = dirpath

    landing.library_selected.connect(_landing_to_main)
    landing.tile_clicked.connect(_on_tile_clicked)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
