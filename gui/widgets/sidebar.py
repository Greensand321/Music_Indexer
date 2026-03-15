"""Left navigation sidebar with grouped section items."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from gui.compat import QtCore, QtGui, QtWidgets, Signal

# ── Navigation model ──────────────────────────────────────────────────────────

@dataclass
class NavItem:
    key: str
    label: str
    icon: str = ""


@dataclass
class NavSection:
    title: str
    items: list[NavItem] = field(default_factory=list)


# Full navigation structure for AlphaDEX
NAV_STRUCTURE: list[NavSection] = [
    NavSection("ORGANIZE", [
        NavItem("indexer",      "Indexer",        "🗂"),
        NavItem("library_sync", "Library Sync",   "🔄"),
    ]),
    NavSection("CLEAN UP", [
        NavItem("duplicates",   "Duplicates",     "🔍"),
        NavItem("similarity",   "Similarity Inspector", "⚖"),
        NavItem("tag_fixer",    "Tag Fixer",      "🏷"),
        NavItem("genres",       "Genre Normalizer","🎸"),
    ]),
    NavSection("PLAYLISTS", [
        NavItem("playlists",    "Playlist Generator", "🎵"),
        NavItem("clustered",    "Clustered Playlists", "📊"),
        NavItem("graph",        "Visual Music Graph",  "🌐"),
    ]),
    NavSection("PLAYER", [
        NavItem("player",       "Player",         "▶"),
        NavItem("compression",  "Compression",    "📦"),
    ]),
    NavSection("TOOLS", [
        NavItem("tools",        "Export & Utilities", "🛠"),
        NavItem("help",         "Help",           "?"),
    ]),
]


# ── Sidebar widget ────────────────────────────────────────────────────────────

class Sidebar(QtWidgets.QWidget):
    """Dark left navigation panel.

    Emits:
        nav_changed(str)  — key of the newly activated item
    """

    nav_changed = Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(220)

        self._buttons: dict[str, QtWidgets.QPushButton] = {}
        self._active_key: str = ""

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(10, 16, 10, 16)
        root_layout.setSpacing(0)

        # ── Logo area ─────────────────────────────────────────────────────
        logo_lbl = QtWidgets.QLabel("AlphaDEX")
        logo_lbl.setStyleSheet(
            "color: #f8fafc; font-size: 16px; font-weight: 700; "
            "padding: 0 6px 12px 6px; letter-spacing: -0.02em;"
        )
        root_layout.addWidget(logo_lbl)

        # ── Scrollable nav ────────────────────────────────────────────────
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        nav_widget = QtWidgets.QWidget()
        nav_widget.setStyleSheet("background: transparent;")
        nav_layout = QtWidgets.QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(2)

        for section in NAV_STRUCTURE:
            # Section header
            hdr = QtWidgets.QLabel(section.title)
            hdr.setObjectName("sidebarSectionLabel")
            if nav_layout.count() > 0:
                spacer = QtWidgets.QWidget()
                spacer.setFixedHeight(10)
                nav_layout.addWidget(spacer)
            nav_layout.addWidget(hdr)

            for item in section.items:
                btn = QtWidgets.QPushButton(f"  {item.icon}  {item.label}")
                btn.setObjectName("navButton")
                btn.setCheckable(False)
                btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
                btn.setProperty("active", "false")
                btn.clicked.connect(
                    lambda checked=False, k=item.key: self._on_nav_click(k)
                )
                nav_layout.addWidget(btn)
                self._buttons[item.key] = btn

        nav_layout.addStretch(1)
        scroll.setWidget(nav_widget)
        root_layout.addWidget(scroll, stretch=1)

        # ── Activate first item ───────────────────────────────────────────
        first_key = NAV_STRUCTURE[0].items[0].key if NAV_STRUCTURE else ""
        if first_key:
            self.activate(first_key)

    # ── Public API ────────────────────────────────────────────────────────

    def activate(self, key: str) -> None:
        """Programmatically activate a nav item (no signal emitted)."""
        if self._active_key and self._active_key in self._buttons:
            prev = self._buttons[self._active_key]
            prev.setProperty("active", "false")
            prev.style().unpolish(prev)
            prev.style().polish(prev)

        self._active_key = key
        if key in self._buttons:
            btn = self._buttons[key]
            btn.setProperty("active", "true")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def set_badge(self, key: str, count: int) -> None:
        """Show a badge count next to a nav item (0 hides it)."""
        if key not in self._buttons:
            return
        btn = self._buttons[key]
        # Find label text without badge
        base_text = btn.text().rstrip()
        # Strip existing badge
        if "(" in base_text:
            base_text = base_text[: base_text.rfind("(")].rstrip()
        if count > 0:
            btn.setText(f"{base_text}  ({count})")
        else:
            btn.setText(base_text)

    # ── Private ───────────────────────────────────────────────────────────

    def _on_nav_click(self, key: str) -> None:
        self.activate(key)
        self.nav_changed.emit(key)
