"""Base class shared by all workspace panels."""
from __future__ import annotations

from gui.compat import QtCore, QtWidgets, Signal
from gui.themes.effects import card_shadow
from gui.themes.manager import get_manager
from gui.widgets.gradient_bg import GradientWidget


class WorkspaceBase(QtWidgets.QWidget):
    """Common base for every workspace panel.

    Provides:
      - log_message(str, level)  signal routed to the LogDrawer
      - status_changed(str)      signal for the status chip
      - A scroll-area wrapper so content can grow freely
      - Helper: _make_card() — returns a styled QFrame card
      - Helper: _make_card_title(text) — bold card section heading
      - Helper: _make_section_title(text) — bold workspace heading
      - Helper: _make_run_button(text) — primary blue action button
    """

    log_message = Signal(str, str)   # (message, level)
    status_changed = Signal(str, str)  # (text, colour)

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
        card = QtWidgets.QFrame()
        card.setObjectName("workspaceCard")
        card.setGraphicsEffect(card_shadow(get_manager().current))
        self._cards.append(card)
        return card

    def refresh_shadows(self) -> None:
        """Re-apply drop shadows after a theme change."""
        tokens = get_manager().current
        for card in self._cards:
            card.setGraphicsEffect(card_shadow(tokens))

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
        """Called on every theme change; refreshes shadows then gradient."""
        self.refresh_shadows()
        # GradientWidget repaints itself via its own theme_changed connection.

    # ── Logging helpers ───────────────────────────────────────────────────

    def _log(self, message: str, level: str = "info") -> None:
        self.log_message.emit(message, level)

