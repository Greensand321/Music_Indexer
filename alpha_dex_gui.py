"""AlphaDEX Qt GUI — entry point.

Run with:
    python alpha_dex_gui.py

This launches the PySide6 Qt Widgets rebuild of the AlphaDEX desktop app
using the Option A Navigator + Workspace layout.

The original Tkinter app remains available at:
    python main_gui.py

Start-up sequence (``build_startup``; ``main`` adds the event loop)
---------------------------------------------------------------------
1. ``MosaicLanding`` — shown immediately.
   - Window fades in with only the CTA card visible (logo moment, ~320 ms).
   - Once it is fully visible (``faded_in``) the main window is built, during
     the ~600 ms logo pause, so the first landing frames are never blocked.
   - Tiles fly in; album art populates them in the background.
   - The user confirms the saved library ("Initialize"), picks a folder
     ("Choose another folder"), clicks an album tile, or quits (the ✕). A
     saved library that no longer exists is shown as "not found" with
     "Choose Library Folder" instead of "Initialize".
   - Tiles scatter; the landing cross-fades into the main window.
2. ``AlphaDEXWindow`` — main application interface. The chosen library is
   applied **once**, when the cross-fade has finished (so the page fan-out
   and background scans never stutter the transition).

If the main window cannot be built, an error dialog is shown, the error is
written to the crash log, and the app exits with status 1.
"""
from __future__ import annotations

if __name__ == "__main__":
    # First thing: clustering uses a process pool, and in a frozen
    # (PyInstaller) build each spawned worker re-executes this script.
    # freeze_support() turns those re-executions into pool workers instead
    # of extra app instances (it never returns in a worker).
    import multiprocessing
    multiprocessing.freeze_support()

import os
import sys
import traceback

# Ensure the repo root is on sys.path so backend modules are importable
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gui.compat import QtCore, QtWidgets, Signal  # noqa: E402

_DEFAULT_SIZE = (1300, 860)


def _shared_geometry(app) -> QtCore.QRect:
    """Landing and main window share one rect so the cross-fade is seamless."""
    screen = app.primaryScreen()
    sg = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1920, 1080)
    lw, lh = _DEFAULT_SIZE
    lx = sg.left() + (sg.width() - lw) // 2
    ly = sg.top() + (sg.height() - lh) // 2
    return QtCore.QRect(lx, ly, lw, lh)


def _saved_library() -> str:
    try:
        from config import load_config
        return str(load_config().get("library_root", "") or "")
    except Exception as e:  # noqa: BLE001
        print(f"[Warning] Failed to load saved config: {e}", file=sys.stderr)
        return ""


def _default_window_factory():
    from gui.main_window import AlphaDEXWindow
    return AlphaDEXWindow()


class Startup(QtCore.QObject):
    """Landing → main window hand-off, driveable without ``app.exec()``.

    Attributes tests (and ``main``) use: ``landing``, ``window`` (None until
    built), ``play_dir`` (the album folder of a clicked tile), ``error`` (why
    the window could not be built), ``exit_code``.
    """

    window_ready = Signal(object)     # the AlphaDEXWindow, once built
    library_applied = Signal(str)     # after the deferred set_library ran
    failed = Signal(str)              # window construction failed

    def __init__(
        self,
        app,
        *,
        geometry: QtCore.QRect | None = None,
        saved_library: str | None = None,
        window_factory=None,
        on_fatal=None,
    ) -> None:
        super().__init__()
        self._app = app
        self._geometry = geometry or _shared_geometry(app)
        self._window_factory = window_factory or _default_window_factory
        self._on_fatal = on_fatal or (lambda code: app.exit(code))
        self.window = None
        self.play_dir = ""
        self.error: str | None = None
        self.exit_code = 0
        self.selected_path = ""
        self._library_applied = False
        self._fade = None

        saved = _saved_library() if saved_library is None else saved_library
        valid = bool(saved) and os.path.isdir(saved)

        from gui.widgets.landing import MosaicLanding
        self.landing = MosaicLanding(
            self._geometry,
            saved if valid else "",
            missing_path="" if valid else saved,
        )
        self.landing.faded_in.connect(self.ensure_window)
        self.landing.library_selected.connect(self._landing_to_main)
        self.landing.tile_clicked.connect(self._on_tile_clicked)
        self.landing.quit_requested.connect(self._on_quit_requested)

    # ── steps ─────────────────────────────────────────────────────────────

    def start(self) -> "Startup":
        self.landing.show_animated()
        return self

    def ensure_window(self):
        """Build the main window if it isn't built yet. None on failure."""
        if self.window is not None or self.error is not None:
            return self.window
        try:
            window = self._window_factory()
            window.setGeometry(self._geometry)
        except Exception as exc:  # noqa: BLE001 - reported, then we exit
            self._fail(exc)
            return None
        self.window = window
        self.window_ready.emit(window)
        return window

    def _fail(self, exc: BaseException) -> None:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.error = f"{type(exc).__name__}: {exc}"
        self.exit_code = 1
        print(f"[AlphaDEX] The main window could not be built:\n{details}", file=sys.stderr)
        try:
            import logging
            logging.getLogger("alphadex.crash").error(
                "Main window construction failed:\n%s", details
            )
        except Exception:
            pass
        try:
            self.landing.hide()
        except RuntimeError:
            pass
        QtWidgets.QMessageBox.critical(
            None,
            "AlphaDEX could not start",
            "The main window could not be built, so AlphaDEX has to close.\n\n"
            f"{self.error}\n\n"
            "Details were written to the crash log and to the console.",
        )
        self.failed.emit(self.error)
        self._on_fatal(1)

    def _on_quit_requested(self) -> None:
        """The landing's ✕: leave without opening the main window."""
        self.exit_code = 0
        self._app.quit()

    def _on_tile_clicked(self, dirpath: str) -> None:
        if not self.play_dir:   # the first click wins; the landing ignores later ones
            self.play_dir = dirpath

    def _landing_to_main(self, path: str) -> None:
        """``library_selected`` fires at the *start* of the landing's fade-out.
        Fade the main window in over the same duration so both windows
        cross-dissolve, then apply the library once."""
        self.selected_path = path
        window = self.ensure_window()   # built now if the user was very fast
        if window is None:
            return

        # The album art the landing found becomes the window's backdrop
        # (no second scan of the same library).
        seed = getattr(window, "seed_backdrop", None)
        if callable(seed):
            seed(path, self.landing.cover_images())

        # A clicked tile: go to the Player and start playing before the
        # window fades in, so it is ready when it appears.
        if self.play_dir:
            window.play_directory(self.play_dir)

        from gui.widgets.landing import FADE_OUT_MS

        window.setWindowOpacity(0.0)
        window.show()

        fade = QtCore.QVariantAnimation(window)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setDuration(FADE_OUT_MS)            # match landing fade-out
        fade.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        fade.valueChanged.connect(lambda v: window.setWindowOpacity(float(v)))
        fade.finished.connect(self._on_cross_fade_done)
        fade.finished.connect(fade.deleteLater)
        self._fade = fade
        fade.start()

    def _on_cross_fade_done(self) -> None:
        self._fade = None
        if self.window is not None:
            self.window.setWindowOpacity(1.0)
        self.apply_library()

    def apply_library(self) -> None:
        """The deferred, one-time library apply (after the cross-fade)."""
        if self._library_applied or self.window is None:
            return
        self._library_applied = True
        if self.selected_path:
            self.window.set_library(self.selected_path)
        self.library_applied.emit(self.selected_path)


def _prepare_app(app) -> None:
    app.setApplicationName("AlphaDEX")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("AlphaDEX")

    # High-DPI scaling is always on in Qt 6 (the old AA_* attributes are
    # deprecated no-ops), so nothing to set here.

    # Register Inter + JetBrains Mono before any widget is created
    from gui.fonts import load_fonts
    load_fonts()

    # Set the application icon (Windows taskbar, title bar, Alt-Tab).
    from gui.icons import make_app_icon
    app.setWindowIcon(make_app_icon())

    # Load the persisted theme so the landing uses the correct palette.
    from gui.themes.manager import get_manager
    get_manager().load_persisted()


def build_startup(app, **kwargs) -> Startup:
    """Prepare ``app`` and show the landing. Returns the ``Startup`` driving
    the rest of the sequence (``main`` then runs ``app.exec()``)."""
    _prepare_app(app)
    return Startup(app, **kwargs).start()


def _install_crash_log() -> None:
    try:
        import crash_logger
        crash_logger.install_for_qt()
    except Exception as exc:  # noqa: BLE001 - never block startup over logging
        print(f"[Warning] Crash log unavailable: {exc}", file=sys.stderr)


def main() -> int:
    _install_crash_log()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    startup = build_startup(app)
    exit_code = app.exec()

    # Never let interpreter shutdown destroy a still-running QThread (that is
    # a qFatal abort). The main window already waits for its jobs on close;
    # this catches anything else (e.g. the landing's art scanner).
    from gui.workers import job_registry
    registry = job_registry()
    registry.cancel_all()
    registry.wait_all(timeout_ms=10_000)
    return startup.exit_code or exit_code


if __name__ == "__main__":
    raise SystemExit(main())
