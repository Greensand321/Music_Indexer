"""AlphaDEX main window — Option A layout: Sidebar + Workspace + Log drawer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.widgets.top_bar import TopBar
from gui.widgets.sidebar import NAV_STRUCTURE, Sidebar
from gui.widgets.log_drawer import LogDrawer
from gui.widgets.now_playing_bar import NowPlayingBar

# ── Workspace imports ─────────────────────────────────────────────────────────
from gui.workspaces.indexer import IndexerWorkspace
from gui.workspaces.library_sync import LibrarySyncWorkspace
from gui.workspaces.duplicates import DuplicatesWorkspace
from gui.workspaces.similarity import SimilarityWorkspace
from gui.workspaces.tag_fixer import TagFixerWorkspace
from gui.workspaces.genres import GenresWorkspace
from gui.workspaces.playlists import PlaylistsWorkspace
from gui.workspaces.clustered_enhanced import EnhancedClusteredWorkspace as ClusteredWorkspace
from gui.workspaces.graph import GraphWorkspace
from gui.workspaces.player import PlayerWorkspace
from gui.workspaces.compression import CompressionWorkspace
from gui.workspaces.tools import ToolsWorkspace
from gui.workspaces.help import HelpWorkspace
from gui.workspaces.base import WorkspaceBase


# Map sidebar keys → workspace class
_WORKSPACE_MAP: dict[str, type[WorkspaceBase]] = {
    "indexer":      IndexerWorkspace,
    "library_sync": LibrarySyncWorkspace,
    "duplicates":   DuplicatesWorkspace,
    "similarity":   SimilarityWorkspace,
    "tag_fixer":    TagFixerWorkspace,
    "genres":       GenresWorkspace,
    "playlists":    PlaylistsWorkspace,
    "clustered":    ClusteredWorkspace,
    "graph":        GraphWorkspace,
    "player":       PlayerWorkspace,
    "compression":  CompressionWorkspace,
    "tools":        ToolsWorkspace,
    "help":         HelpWorkspace,
}

# Sidebar order and labels — the single source for Ctrl+1..9 and the status
# bar, so shortcuts can never drift from what the sidebar shows.
_NAV_ORDER: list[str] = [
    item.key for section in NAV_STRUCTURE for item in section.items
    if item.key in _WORKSPACE_MAP
] + [key for key in _WORKSPACE_MAP if key not in {
    item.key for section in NAV_STRUCTURE for item in section.items
}]
_NAV_LABELS: dict[str, str] = {
    item.key: item.label for section in NAV_STRUCTURE for item in section.items
}

# Focus widgets that own the arrow keys: keyboard mode never steals them.
_ARROW_KEY_OWNERS = (
    QtWidgets.QLineEdit,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QComboBox,
    QtWidgets.QTextEdit,
    QtWidgets.QPlainTextEdit,
    QtWidgets.QKeySequenceEdit,
)
_ARROW_KEYS = {
    QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down,
    QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right,
}


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


class AlphaDEXWindow(QtWidgets.QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AlphaDEX")
        self.resize(1300, 860)
        self.setMinimumSize(900, 600)

        self._library_path: str = ""
        self._workspaces: dict[str, WorkspaceBase] = {}
        self._theme_picker = None   # keep reference so dialog stays open

        self._stats_worker = None
        self._stats_generation = 0

        # Apply persisted theme before building UI. (Card shadows follow theme
        # changes through WorkspaceBase's own theme_changed listener.)
        get_manager().load_persisted()

        self._keyboard_mode_active = False
        self._keyboard_filter_installed = False
        self._build_ui()
        self._setup_shortcuts()
        self._report_config_backup()
        # The library is NOT applied here: the startup sequence (landing →
        # set_library) or the user chooses it, exactly once.

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ────────────────────────────────────────────────────────
        self._top_bar = TopBar()
        self._top_bar.library_changed.connect(self._on_library_changed)
        self._top_bar.settings_requested.connect(self._on_settings)
        self._top_bar.theme_requested.connect(self._on_theme_requested)
        root.addWidget(self._top_bar)

        # ── Body: sidebar + content ────────────────────────────────────────
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.nav_changed.connect(self._on_nav_changed)
        self._sidebar.exit_requested.connect(self.close)
        body_layout.addWidget(self._sidebar)

        # Content area with stacked widget
        content_frame = QtWidgets.QFrame()
        content_frame.setObjectName("contentFrame")
        content_layout = QtWidgets.QVBoxLayout(content_frame)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._stack = QtWidgets.QStackedWidget()
        content_layout.addWidget(self._stack, stretch=1)

        # ── Now Playing bar (persistent, above log drawer) ─────────────────
        self._now_playing_bar = NowPlayingBar()
        content_layout.addWidget(self._now_playing_bar)

        # ── Log drawer (inside content area, bottom) ───────────────────────
        self._log_drawer = LogDrawer()
        content_layout.addWidget(self._log_drawer)

        body_layout.addWidget(content_frame, stretch=1)
        root.addWidget(body, stretch=1)

        # ── Status bar ─────────────────────────────────────────────────────
        self.statusBar().showMessage("Ready")

        # Pre-create and register all workspaces
        self._init_workspaces()

        # Show the first workspace
        self._on_nav_changed("indexer")

    def _init_workspaces(self) -> None:
        for key, cls in _WORKSPACE_MAP.items():
            ws = cls(library_path=self._library_path)
            ws.log_message.connect(self._on_log_message)
            ws.status_changed.connect(self._on_status_changed)
            ws.navigate_requested.connect(self._on_nav_changed)
            ws.play_tracks_requested.connect(self.play_tracks)
            ws.flush_pending_logs()   # anything logged during construction
            self._stack.addWidget(ws)
            self._workspaces[key] = ws

        # ── Wire PlayerWorkspace ↔ NowPlayingBar ────────────────────────
        player_ws = self._workspaces.get("player")
        if isinstance(player_ws, PlayerWorkspace):
            npb = self._now_playing_bar
            # Player → bar
            player_ws.now_playing_changed.connect(npb.update_now_playing)
            player_ws.playback_state_changed.connect(npb.set_playing)
            player_ws.position_changed.connect(npb.update_position)
            # Bar → player
            npb.play_pause_requested.connect(player_ws._on_play_pause)
            npb.next_requested.connect(player_ws.play_next)
            npb.prev_requested.connect(player_ws.play_prev)
            npb.seek_requested.connect(self._on_bar_seek)
            npb.volume_changed.connect(self._on_bar_volume)
            npb.keyboard_mode_toggled.connect(self.set_keyboard_mode_global)
            # Keep bar volume knob in sync with the workspace slider
            # Any volume change on the page (drag, wheel, keys, groove click)
            # updates the bar's knob; set_volume() there doesn't echo back.
            player_ws.volume_changed.connect(npb.set_volume)

    # ── Keyboard mode (Player arrow keys) ─────────────────────────────────
    #
    # The application-level filter is installed only while keyboard mode is
    # on, and it only takes arrow keys when the Player page is showing (or
    # the Now Playing bar has focus) in this window, no modal dialog or popup
    # is open, and the focused widget is not one that uses arrows itself
    # (text fields, spin boxes, combo boxes).

    def _set_keyboard_filter(self, on: bool) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None or on == self._keyboard_filter_installed:
            return
        if on:
            app.installEventFilter(self)
        else:
            app.removeEventFilter(self)
        self._keyboard_filter_installed = on

    def _keyboard_mode_applies(self, obj) -> bool:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return False
        if app.activeModalWidget() is not None or app.activePopupWidget() is not None:
            return False
        player_ws = self._workspaces.get("player")
        if not isinstance(player_ws, PlayerWorkspace):
            return False
        # Key events reach the QWindow first and are then re-sent to the
        # focus widget; decide on the widget delivery only.
        if not isinstance(obj, QtWidgets.QWidget):
            return False
        target = obj
        if target.window() is not self:
            return False
        if isinstance(target, _ARROW_KEY_OWNERS):
            return False
        if self._stack.currentWidget() is player_ws:
            return True
        bar = self._now_playing_bar
        return target is bar or bar.isAncestorOf(target)

    def eventFilter(self, obj, event):  # noqa: N802
        """Route arrow keys to the Player while keyboard mode applies.

        Dispatches to library navigation or queue navigation based on which
        panel the user last clicked (tracked by PlayerWorkspace._kb_target).
        """
        if (
            self._keyboard_mode_active
            and event.type() == QtCore.QEvent.Type.KeyPress
            and event.key() in _ARROW_KEYS
            and not (
                event.modifiers()
                & ~QtCore.Qt.KeyboardModifier.KeypadModifier
            )
            and self._keyboard_mode_applies(obj)
        ):
            key = event.key()
            player_ws = self._workspaces.get("player")
            if isinstance(player_ws, PlayerWorkspace):
                target = player_ws.kb_target()
                if target == "library":
                    if key == QtCore.Qt.Key.Key_Down:
                        player_ws.kb_lib_down()
                        return True
                    elif key == QtCore.Qt.Key.Key_Right:
                        player_ws.kb_lib_right()
                        return True
                    elif key == QtCore.Qt.Key.Key_Left:
                        player_ws.kb_lib_left()
                        return True
                    elif key == QtCore.Qt.Key.Key_Up:
                        player_ws.kb_lib_up()
                        return True
                else:  # queue
                    if key == QtCore.Qt.Key.Key_Down:
                        player_ws._preview_next()
                        return True
                    elif key == QtCore.Qt.Key.Key_Right:
                        player_ws.play_next()
                        return True
                    elif key == QtCore.Qt.Key.Key_Left:
                        player_ws.play_prev()
                        return True
                    elif key == QtCore.Qt.Key.Key_Up:
                        player_ws._restart_current()
                        return True
        return super().eventFilter(obj, event)

    def set_keyboard_mode_global(self, active: bool) -> None:
        """Toggle keyboard mode (from the Now Playing bar's ⌨ pill)."""
        self._keyboard_mode_active = bool(active)
        self._set_keyboard_filter(self._keyboard_mode_active)
        player_ws = self._workspaces.get("player")
        if isinstance(player_ws, PlayerWorkspace):
            player_ws.set_keyboard_mode(active)

    # ── Shortcuts ─────────────────────────────────────────────────────────

    def _setup_shortcuts(self) -> None:
        # Ctrl+O → Change library
        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+O"), self
        ).activated.connect(self._top_bar._on_change_library)

        # Ctrl+, → Settings
        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+,"), self
        ).activated.connect(self._on_settings)

        # Ctrl+L → Toggle log drawer
        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+L"), self
        ).activated.connect(self._log_drawer.toggle)

        # Ctrl+W → Clear log
        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+W"), self
        ).activated.connect(self._log_drawer._on_clear)

        # Ctrl+1–9 → Switch workspaces by sidebar order
        for i, key in enumerate(_NAV_ORDER[:9], 1):
            QtGui.QShortcut(
                QtGui.QKeySequence(f"Ctrl+{i}"), self
            ).activated.connect(lambda k=key: self._on_nav_changed(k))

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot(str)
    def _on_nav_changed(self, key: str) -> None:
        if key not in self._workspaces:
            return
        self._sidebar.activate(key)
        ws = self._workspaces[key]
        self._stack.setCurrentWidget(ws)
        self.statusBar().showMessage(_NAV_LABELS.get(key, key.replace("_", " ").title()))
        # Optional hook: a page can refresh itself when it is shown (e.g. the
        # Music Graph reloading a newer cluster payload).
        hook = getattr(ws, "on_activated", None)
        if callable(hook):
            hook()

    # ── Library selection: the one validated path ─────────────────────────
    #
    # Landing (via set_library), TopBar "Change Library" / Ctrl+O and
    # Settings → Default library all end in _apply_library(), which rejects
    # missing folders, ignores the library that is already active, and only
    # then fans the path out to every page, refreshes the stats and saves it.

    @Slot(str)
    def _on_library_changed(self, path: str) -> None:
        """TopBar ``library_changed`` slot."""
        self._apply_library(path)

    def _apply_library(self, path: str) -> bool:
        """Make *path* the active library. True when it is (now) active."""
        if not path:
            return False
        norm = os.path.normpath(os.path.abspath(os.path.expanduser(path)))
        if self._library_path and _same_path(norm, self._library_path):
            return True   # unchanged: don't reset every page again
        if not os.path.isdir(norm):
            self._on_log_message(f"Library folder not found: {path}", "warn")
            QtWidgets.QMessageBox.warning(
                self,
                "Library folder not found",
                f"The folder\n{path}\ndoes not exist (or is not reachable).\n\n"
                "The current library was kept.",
            )
            return False

        self._library_path = norm
        self._top_bar.set_library(norm)
        for ws in self._workspaces.values():
            ws.set_library_path(norm)
        self._on_log_message(f"Library set to: {norm}", "ok")
        self.statusBar().showMessage(f"Library: {norm}")

        # Update stats in background
        self._update_library_stats(norm)
        self._persist_library(norm)
        return True

    def _persist_library(self, path: str) -> None:
        try:
            from config import load_config, save_config
            cfg = load_config()
            if cfg.get("library_root") != path:
                cfg["library_root"] = path
                save_config(cfg)
        except Exception as exc:  # noqa: BLE001 - never break the UI over this
            self._on_log_message(f"Could not save the library choice: {exc}", "warn")

    def _report_config_backup(self) -> None:
        try:
            import config
            backup = config.last_backup_path()
        except Exception:
            return
        if backup:
            self._on_log_message(
                f"Your settings file could not be read; a copy was saved to {backup}",
                "warn",
            )

    @Slot(str, str)
    def _on_log_message(self, message: str, level: str) -> None:
        self._log_drawer.append(message, level)

    @Slot(str, str)
    def _on_status_changed(self, text: str, colour: str) -> None:
        self._log_drawer.set_status(text, colour)
        self.statusBar().showMessage(text)

    @Slot()
    def _on_settings(self) -> None:
        from gui.dialogs.settings_drawer import SettingsDrawer
        dlg = SettingsDrawer(self)
        dlg.settings_saved.connect(lambda: self._on_settings_saved(dlg))
        try:
            dlg.exec()
        finally:
            dlg.deleteLater()   # one dialog per open, not one leaked per open

    def _on_settings_saved(self, dlg=None) -> None:
        # Cached appearance settings must pick up the new values now, not on
        # the next theme switch.
        from gui.themes.motion import refresh_reduced_motion
        from gui.widgets.gradient_bg import refresh_gradient_setting
        refresh_gradient_setting()   # also repaints every window
        refresh_reduced_motion()
        for ws in self._workspaces.values():
            try:
                ws.on_settings_saved()
            except Exception as exc:  # noqa: BLE001 - one page must not block the rest
                self._on_log_message(f"{type(ws).__name__}: could not apply settings ({exc})", "warn")

        # Switch library only if the default library was changed in the dialog
        if dlg is not None and not getattr(dlg, "library_changed", True):
            return
        try:
            from config import load_config
            lib = load_config().get("library_root", "")
        except Exception:
            return
        if lib:
            self._apply_library(lib)

    @Slot()
    def _on_theme_requested(self) -> None:
        picker = self._theme_picker
        try:
            if picker is not None and picker.isVisible():
                picker.raise_()
                picker.activateWindow()
                return
            if picker is not None:
                picker.deleteLater()
        except RuntimeError:  # already deleted
            pass
        from gui.themes.picker import open_theme_picker
        self._theme_picker = open_theme_picker(self)

    # ── Close / shutdown ──────────────────────────────────────────────────
    #
    # Closing while a worker thread runs used to abort the whole process
    # ("QThread: Destroyed while thread is still running") — possibly half
    # way through moving files. Now: ask, cancel cooperatively, keep the
    # window alive until every job has stopped, then shut workspaces down.

    # Seconds to wait for cancelled jobs before offering "Quit anyway".
    CLOSE_PATIENCE_S = 20.0

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        from gui.workers import job_registry

        registry = job_registry()
        jobs = registry.jobs()
        if jobs:
            if not getattr(self, "_closing", False):
                # Read-only jobs (scans, previews, art loads) are cancelled
                # without asking; only work that may be writing files needs OK.
                asking = [j for j in jobs if j.ask_before_quit]
                if asking:
                    names = ", ".join(sorted({j.name for j in asking}))
                    answer = QtWidgets.QMessageBox.question(
                        self,
                        "Background tasks running",
                        f"{len(asking)} background task(s) are still running:\n{names}\n\n"
                        "Stop them and quit? AlphaDEX will close as soon as they "
                        "reach a safe stopping point.",
                        QtWidgets.QMessageBox.StandardButton.Yes
                        | QtWidgets.QMessageBox.StandardButton.No,
                        QtWidgets.QMessageBox.StandardButton.No,
                    )
                    if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                        event.ignore()
                        return
                self._begin_closing()
            # Cancel on every retry: also catches anything that started after
            # closing began (nothing can be started from the UI now, but a
            # job may chain another).
            registry.cancel_all()
            self.statusBar().showMessage(
                f"Stopping {len(jobs)} background task(s) before closing…"
            )
            event.ignore()
            self._maybe_offer_force_quit(jobs)
            QtCore.QTimer.singleShot(100, self.close)
            return

        if not getattr(self, "_shut_down", False):
            self._shut_down = True
            for ws in self._workspaces.values():
                try:
                    ws.shutdown()
                except Exception as exc:  # noqa: BLE001 - never block quitting
                    print(f"[shutdown] {type(ws).__name__}: {exc}", file=sys.stderr)
            self._set_keyboard_filter(False)
        super().closeEvent(event)

    def _begin_closing(self) -> None:
        """Enter the stopping state: no new work can be started from the UI."""
        import time

        self._closing = True
        self._closing_since = time.monotonic()
        central = self.centralWidget()
        if central is not None:
            central.setEnabled(False)

    def _maybe_offer_force_quit(self, jobs) -> None:
        """A job that ignores cancel (a hung network drive, a long native
        clustering stage) must not trap the user in a window that never
        closes. After CLOSE_PATIENCE_S, ask once whether to quit anyway."""
        import time

        since = getattr(self, "_closing_since", None)
        if since is None or getattr(self, "_force_quit_asked", False):
            return
        if time.monotonic() - since < self.CLOSE_PATIENCE_S:
            return
        self._force_quit_asked = True
        names = ", ".join(sorted({j.name for j in jobs}))
        answer = QtWidgets.QMessageBox.question(
            self,
            "Tasks are not stopping",
            f"Still waiting for: {names}\n\n"
            "They have not reached a safe stopping point. Quit anyway? A file "
            "that is being copied or rewritten right now may be left incomplete.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._force_quit(names)

    @staticmethod
    def _force_quit(names: str) -> None:
        """Leave immediately. Interpreter shutdown would destroy the still
        running QThreads (a qFatal abort), so skip it explicitly."""
        try:
            import crash_logger
            log = getattr(crash_logger, "log_event", None)
            if callable(log):
                log(f"Forced quit while tasks were running: {names}")
        except Exception:  # noqa: BLE001
            pass
        print(f"[AlphaDEX] Forced quit while tasks were running: {names}", file=sys.stderr)
        sys.stderr.flush()
        os._exit(3)

    # ── Public API ────────────────────────────────────────────────────────
    # ── Public API ────────────────────────────────────────────────────────

    def set_library(self, path: str) -> bool:
        """Make *path* the active library (called once by the startup
        sequence after the landing, and by scripts/tests). Validated: a
        missing folder is rejected with a message, the already-active
        library is a no-op. Returns True when *path* is the active library."""
        return self._apply_library(path)

    def play_directory(self, dirpath: str) -> None:
        """Switch to the Player workspace and start playing all tracks in *dirpath*."""
        self._on_nav_changed("player")
        player_ws = self._workspaces.get("player")
        if isinstance(player_ws, PlayerWorkspace):
            player_ws.load_directory_and_play(dirpath)

    @Slot(list, str)
    def play_tracks(self, paths: list, label: str = "") -> None:
        """Switch to the Player workspace and queue *paths*.

        Lets a workspace hand a set of tracks to the player — the Music Graph
        uses it to send a lasso'd selection straight to playback.
        """
        if not paths:
            return
        self._on_nav_changed("player")
        player_ws = self._workspaces.get("player")
        if isinstance(player_ws, PlayerWorkspace):
            player_ws.load_tracks_and_play(list(paths), label)

    @Slot(int)
    def _on_bar_seek(self, ms: int) -> None:
        player_ws = self._workspaces.get("player")
        if isinstance(player_ws, PlayerWorkspace):
            player_ws.seek_to_ms(ms)

    @Slot(int)
    def _on_bar_volume(self, value: int) -> None:
        player_ws = self._workspaces.get("player")
        if isinstance(player_ws, PlayerWorkspace):
            player_ws.set_volume(value)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _update_library_stats(self, path: str) -> None:
        """Scan library stats in a background thread and update the top bar.

        Each scan carries a generation number; results from an older scan
        (the library changed again meanwhile) are ignored, and the older
        scan is asked to stop."""
        from gui.workers import cancel_thread, start_worker

        old = self._stats_worker
        if old is not None:
            cancel_thread(old)
        self._stats_generation += 1
        worker = _StatsWorker(path, self._stats_generation)
        worker.stats_ready.connect(self._on_stats_ready)
        self._stats_worker = worker
        start_worker(worker, owner=self, name="Library statistics", ask_before_quit=False)

    @Slot(int, int, float, int)
    def _on_stats_ready(self, generation: int, tracks: int, size_gb: float, artists: int) -> None:
        if generation != self._stats_generation:
            return  # stale: a newer library's scan is (or was) running
        self._stats_worker = None
        self._top_bar.set_stats(tracks, size_gb, artists)


_AUDIO_EXTS = frozenset({".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"})


def library_stats(library_path: str, should_stop=lambda: False) -> tuple[int, int, int] | None:
    """(tracks, size_bytes, artists) for the top bar; None if stopped early.

    Skips the reserved folders (Quarantine/, Trash/, Docs/, ...), never
    follows directory symlinks, and skips files it cannot stat (broken
    symlinks, permission errors) instead of failing the whole count.
    "Artists" is the folder two levels above a track (``Artist/Album/track``),
    or the folder above a track that sits one level deep (``Artist/track``);
    tracks at the library root count no artist.
    """
    from utils.reserved_folders import walk_library

    tracks = 0
    size_bytes = 0
    artists: set[str] = set()
    for dirpath, _dirnames, filenames in walk_library(library_path):
        if should_stop():
            return None
        rel = os.path.relpath(dirpath, library_path)
        parts = () if rel in (".", "") else Path(rel).parts
        artist = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in _AUDIO_EXTS:
                continue
            try:
                size = os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
            tracks += 1
            size_bytes += size
            if artist:
                artists.add(artist.casefold())
    return tracks, size_bytes, len(artists)


class _StatsWorker(QtCore.QThread):
    # Not called ``finished``: that would shadow QThread.finished.
    stats_ready = Signal(int, int, float, int)  # generation, tracks, GB, artists

    def __init__(self, library_path: str, generation: int = 0) -> None:
        super().__init__()
        self.library_path = library_path
        self.generation = generation

    def run(self) -> None:
        try:
            result = library_stats(self.library_path, self.isInterruptionRequested)
        except Exception:  # noqa: BLE001 - stats are best-effort
            result = (0, 0, 0)
        if result is None:
            return  # cancelled: a newer scan replaces this one
        tracks, size_bytes, artists = result
        self.stats_ready.emit(self.generation, tracks, size_bytes / (1024 ** 3), artists)
