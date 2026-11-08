"""Base class shared by all workspace panels."""
from __future__ import annotations

from gui.compat import QtCore, QtWidgets, Signal
from gui.themes.manager import get_manager
from gui.widgets.gradient_bg import GradientWidget


class WorkspaceBase(QtWidgets.QWidget):
    """Common base for every workspace panel.

    Provides:
      - log_message(str, level)  signal routed to the LogDrawer
      - status_changed(str)      signal for the status chip
      - navigate_requested(key)  signal asking the main window to switch page/section
      - A scroll-area wrapper so content can grow freely
      - Helper: _make_card() — returns a styled QFrame card
      - Helper: _make_card_title(text) — bold card section heading
      - Helper: _make_section_title(text) — bold workspace heading
      - Helper: _make_run_button(text) — primary blue action button
    """

    log_message = Signal(str, str)   # (message, level)
    status_changed = Signal(str, str)  # (text, colour)
    # Emitted with a navigation key (a panel key from _WORKSPACE_MAP, a page
    # or section key, or "page/section"; see gui/navigation.py) when a
    # workspace wants the main window to navigate elsewhere (e.g. "graph":
    # open this result on the Sound Map's Map). Workspaces never switch
    # themselves; the main window owns navigation.
    navigate_requested = Signal(str)
    # Emitted with (paths, label) to hand a set of tracks to the Player. Same
    # rule as above: the workspace asks, the main window routes.
    play_tracks_requested = Signal(list, str)

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._library_path = library_path
        self._cards: list[QtWidgets.QFrame] = []
        self._setup_scroll()
        get_manager().theme_changed.connect(self._on_theme_changed_base)

    # ── Scroll wrapper ────────────────────────────────────────────────────

    def _setup_scroll(self) -> None:
        """Wrap self in a scroll area so tall content is accessible."""
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        # GradientWidget paints the themed background; plain QWidget would be transparent.
        self._inner = GradientWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._inner)
        self._content_layout.setContentsMargins(24, 20, 24, 20)
        self._content_layout.setSpacing(16)

        scroll.setWidget(self._inner)
        outer.addWidget(scroll)

    @property
    def content_layout(self) -> QtWidgets.QVBoxLayout:
        return self._content_layout

    # ── Builder helpers ───────────────────────────────────────────────────

    def _make_card(self) -> QtWidgets.QFrame:
        """A themed card. Its shadow is painted by the scroll surface
        (``GradientWidget.add_elevated``), not by a graphics effect, so the
        card's one effect slot stays free for fades (DS-02)."""
        card = QtWidgets.QFrame()
        card.setObjectName("workspaceCard")
        self._cards.append(card)
        surface = getattr(self, "_inner", None)
        if hasattr(surface, "add_elevated"):
            surface.add_elevated(card)
        return card

    def refresh_shadows(self) -> None:
        """Repaint card shadows (they are painted from the live theme, so a
        repaint is all a theme change needs). Kept for callers; cheap."""
        surface = getattr(self, "_inner", None)
        if surface is not None:
            surface.update()

    def _make_section_title(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _make_subtitle(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("sectionSubtitle")
        lbl.setWordWrap(True)
        return lbl

    def _make_card_title(self, text: str) -> QtWidgets.QLabel:
        """Bold heading used at the top of a card."""
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("cardTitle")
        return lbl

    def _make_primary_button(self, text: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setObjectName("primaryBtn")
        btn.setMinimumHeight(34)
        return btn

    def _make_browse_row(
        self,
        label_text: str,
        placeholder: str = "",
    ) -> tuple[QtWidgets.QLineEdit, QtWidgets.QPushButton]:
        """Return (entry, browse_button) — caller layouts them."""
        entry = QtWidgets.QLineEdit()
        entry.setPlaceholderText(placeholder)
        browse = QtWidgets.QPushButton("Browse…")
        browse.setFixedWidth(80)
        return entry, browse

    # ── Library path ──────────────────────────────────────────────────────

    def set_library_path(self, path: str) -> None:
        self._library_path = path
        self._on_library_changed(path)

    def _on_library_changed(self, path: str) -> None:
        """Override in subclasses to react to library path updates."""

    # ── Theme handling ────────────────────────────────────────────────────

    def _on_theme_changed_base(self, tokens: object) -> None:
        """Called on every theme change. Nothing to rebuild here: the
        GradientWidget surface repaints itself (background and card shadows)
        via its own theme_changed connection. Subclasses extend this."""

    # ── Background jobs ───────────────────────────────────────────────────
    #
    # Always start worker threads through ``_start_worker``: the JobRegistry
    # keeps each thread alive until it has really finished, so dropping
    # ``self._worker`` in a "finished" slot (or starting a second run) can no
    # longer destroy a running QThread and abort the process. See gui/workers.py.

    def _start_worker(
        self, worker: QtCore.QThread, name: str = "", *,
        ask_before_quit: bool = True, library_write: bool = False,
    ) -> QtCore.QThread:
        """Start ``worker`` under the JobRegistry.

        ``ask_before_quit=False`` for read-only jobs (scans, previews) so
        quitting cancels them silently. ``library_write=True`` for jobs that
        change files in the library; check ``_library_writes_blocked`` first.
        """
        from gui.workers import LIBRARY_WRITES, start_worker
        return start_worker(worker, owner=self, name=name or type(worker).__name__,
                            ask_before_quit=ask_before_quit,
                            group=LIBRARY_WRITES if library_write else None)

    def _library_writes_blocked(self, action: str) -> bool:
        """True (after telling the user) if another job is already changing
        files in the library. Call before confirming a write; start the write
        with ``_start_worker(..., library_write=True)``."""
        from gui.workers import LIBRARY_WRITES, job_registry
        busy = [j.name for j in job_registry().group_jobs(LIBRARY_WRITES)]
        try:
            from controllers import tag_write_guard
            owner = tag_write_guard.current_owner()
        except Exception:  # noqa: BLE001 - guard is optional
            owner = None
        if owner and owner not in busy:
            busy.append(owner)
        if not busy:
            return False
        QtWidgets.QMessageBox.information(
            self, "Library Busy",
            f"{', '.join(busy)} is changing files in your library right now.\n\n"
            f"{action} can start when it has finished — running both at once "
            "could leave the library in a state neither plan described.",
        )
        return True

    def running_jobs(self) -> list:
        """Jobs this workspace started that are still running."""
        from gui.workers import job_registry
        return job_registry().jobs(owner=self)

    def has_running_jobs(self) -> bool:
        return bool(self.running_jobs())

    def cancel_jobs(self) -> None:
        """Ask every running job of this workspace to stop (non-blocking)."""
        from gui.workers import job_registry
        job_registry().cancel_all(self.running_jobs())

    def on_settings_saved(self) -> None:
        """Called by the main window after the Settings dialog saves. Pages
        that cache config values (e.g. thresholds) re-read them here — not on
        every show or paint."""

    def shutdown(self) -> None:
        """Called once by the main window when the app is closing, after all
        jobs have stopped. Override to stop timers / release resources;
        call ``super().shutdown()``."""

    # ── Logging helpers ───────────────────────────────────────────────────

    def _log(self, message: str, level: str = "info") -> None:
        # Messages logged while the page is being constructed (before the main
        # window has connected log_message) used to vanish — e.g. the Player's
        # "libVLC not found" warning. Buffer them until someone listens.
        if self.receivers(QtCore.SIGNAL("log_message(QString,QString)")) > 0:
            self.log_message.emit(message, level)
        else:
            self.__dict__.setdefault("_pending_logs", []).append((message, level))

    def flush_pending_logs(self) -> None:
        """Re-emit messages logged before ``log_message`` was connected."""
        pending = self.__dict__.pop("_pending_logs", [])
        for message, level in pending:
            self.log_message.emit(message, level)

