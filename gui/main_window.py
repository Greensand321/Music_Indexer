"""AlphaDEX main window — Option A layout: Sidebar + Workspace + Log drawer."""
from __future__ import annotations

import sys
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.widgets.top_bar import TopBar
from gui.widgets.sidebar import Sidebar
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

_LIBRARY_STATS_DELAY_MS = 3_000


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
        self._pending_stats_path: str = ""
        self._stats_timer = QtCore.QTimer(self)
        self._stats_timer.setSingleShot(True)
        self._stats_timer.timeout.connect(self._run_scheduled_stats)

        # Apply persisted theme before building UI
        get_manager().load_persisted()
        get_manager().theme_changed.connect(self._on_theme_changed)

        self._build_ui()
        self._setup_shortcuts()
        self._load_persisted_library()

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
            # Keep bar volume knob in sync with the workspace slider
            player_ws._vol_slider.sliderMoved.connect(npb.set_volume)
            player_ws._vol_slider.sliderReleased.connect(
                lambda: npb.set_volume(player_ws._vol_slider.value())
            )

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
        keys = list(_WORKSPACE_MAP.keys())
        for i, key in enumerate(keys[:9], 1):
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
        self.statusBar().showMessage(key.replace("_", " ").title())

    @Slot(str)
    def _on_library_changed(self, path: str) -> None:
        self._library_path = path
        self._top_bar.set_library(path)
        for ws in self._workspaces.values():
            ws.set_library_path(path)
        self._on_log_message(f"Library set to: {path}", "ok")
        self.statusBar().showMessage(f"Library: {path}")

        # Update stats in background
        self._schedule_library_stats(path)

        # Persist
        try:
            from config import load_config, save_config
            cfg = load_config()
            cfg["library_root"] = path
            save_config(cfg)
        except Exception:
            pass

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
        dlg.settings_saved.connect(self._on_settings_saved)
        dlg.exec()

    @Slot()
    def _on_settings_saved(self) -> None:
        # Reload library if it changed in settings
        try:
            from config import load_config
            cfg = load_config()
            lib = cfg.get("library_root", "")
            if lib and lib != self._library_path:
                self._top_bar.set_library(lib)
                self._on_library_changed(lib)
        except Exception:
            pass

    @Slot()
    def _on_theme_requested(self) -> None:
        from gui.themes.picker import open_theme_picker
        self._theme_picker = open_theme_picker(self)

    @Slot(object)
    def _on_theme_changed(self, tokens) -> None:
        """Refresh card shadows on all workspaces after a theme switch."""
        for ws in self._workspaces.values():
            ws.refresh_shadows()

    # ── Public API ────────────────────────────────────────────────────────

    def set_library(self, path: str) -> None:
        """Pre-load a library path (called from the landing page before the
        window is shown).  Updates the top bar label and fires the full
        library-changed chain (workspace updates, stats, persistence)."""
        if not path:
            return
        self._top_bar.set_library(path)
        self._on_library_changed(path)

    def play_directory(self, dirpath: str) -> None:
        """Switch to the Player workspace and start playing all tracks in *dirpath*."""
        self._on_nav_changed("player")
        player_ws = self._workspaces.get("player")
        if isinstance(player_ws, PlayerWorkspace):
            player_ws.load_directory_and_play(dirpath)

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

    def _load_persisted_library(self) -> None:
        try:
            from config import load_config
            cfg = load_config()
            lib = cfg.get("library_root", "")
            if lib and Path(lib).is_dir():
                self._top_bar.set_library(lib)
                self._on_library_changed(lib)
        except Exception:
            pass

    def _update_library_stats(self, path: str) -> None:
        """Scan library stats in a background thread and update the top bar."""
        worker = _StatsWorker(path)
        worker.finished.connect(self._on_stats_ready)
        worker.setParent(self)
        # Keep reference so it doesn't get GC'd
        self._stats_worker = worker
        worker.start(QtCore.QThread.Priority.LowPriority)

    def _schedule_library_stats(self, path: str) -> None:
        """Delay expensive full-library stat walk to keep startup animation smooth."""
        self._pending_stats_path = path
        self._stats_timer.start(_LIBRARY_STATS_DELAY_MS)

    @Slot()
    def _run_scheduled_stats(self) -> None:
        if self._pending_stats_path:
            self._update_library_stats(self._pending_stats_path)

    @Slot(int, float, int)
    def _on_stats_ready(self, tracks: int, size_gb: float, artists: int) -> None:
        self._top_bar.set_stats(tracks, size_gb, artists)


class _StatsWorker(QtCore.QThread):
    finished = Signal(int, float, int)

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self.library_path = library_path

    def run(self) -> None:
        try:
            tracks = 0
            size_bytes = 0
            artists: set[str] = set()
            exts = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
            for p in Path(self.library_path).rglob("*"):
                if p.suffix.lower() in exts:
                    tracks += 1
                    size_bytes += p.stat().st_size
                    parent = p.parent.parent.name
                    if parent:
                        artists.add(parent)
            self.finished.emit(tracks, size_bytes / (1024 ** 3), len(artists))
        except Exception:
            self.finished.emit(0, 0.0, 0)
