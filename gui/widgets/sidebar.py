"""The floating glass sidebar: brand, grouped pages, Activity · Settings · Help.

Geometry follows the Liquid Glass mockup: 250 px wide (220 minimum), radius
30, padding 20 / 14; a 34 px brand tile with accent bars and "AlphaDEX" at
19/700; group labels 11 px / 600 / +0.12em; items 40 px high with radius 14
(they shrink, never below 32, when the window is short); the untitled group
(Activity, Settings, Help) sits at the bottom.

The active indicator is one painted pill (``NavContainer``) that slides
between items by animating a *painter offset* (``motion.animate_value``,
``slow`` + ``emphasis``): no widget geometry is ever animated. The panel
paints its own glass; its shadow is painted by the window backdrop
underneath (``Backdrop.add_floating``), so it falls *under* the pages as in
the mockup.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.animations import AnimatedNavButton

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


def _build_nav_structure() -> list[NavSection]:
    """Sidebar groups from the information architecture (gui/navigation.py).
    A group with an empty title is the bottom group (Activity · Settings ·
    Help), pushed to the foot of the panel."""
    from gui.navigation import nav_groups

    return [
        NavSection(title, [NavItem(p.key, p.label, p.icon) for p in pages])
        for title, pages in nav_groups()
    ]


NAV_STRUCTURE: list[NavSection] = _build_nav_structure()

SIDEBAR_WIDTH = 250
SIDEBAR_MIN_WIDTH = 220
ITEM_HEIGHT = 40
ITEM_MIN_HEIGHT = 32
ITEM_GAP = 2


def _nav_tooltips() -> dict[str, str]:
    from gui.navigation import PAGES
    return {p.key: p.intent for p in PAGES}


# ── Sliding pill container ─────────────────────────────────────────────────────

class NavContainer(QtWidgets.QWidget):
    """Transparent nav content container that paints the sliding active pill."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("navContainer")
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._pill_y: float = 0.0
        self._pill_h: float = float(ITEM_HEIGHT)
        self._pill_visible: bool = False

    def _on_pill_value(self, value: object) -> None:
        self._pill_y = float(value)  # type: ignore[arg-type]
        self.update()

    @property
    def _pill_anim(self):
        """The pill's helper animation (None until the first slide)."""
        from gui.themes import motion
        return motion.animation_for(self, "pill")

    def move_pill(self, y: float, h: float, *, instant: bool = False) -> None:
        from gui.themes import motion
        self._pill_h = h
        self._pill_visible = True
        motion.animate_value(
            self, "pill", float(self._pill_y), float(y), self._on_pill_value,
            duration="instant" if instant else "slow", easing="emphasis",
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        if not self._pill_visible:
            return
        from gui.themes.glass import current_material, paint_surface

        p = QtGui.QPainter(self)
        try:
            pill = QtCore.QRectF(0.0, self._pill_y, float(self.width()), self._pill_h)
            paint_surface(p, pill, current_material().raised, radius=14)
        finally:
            p.end()


class _GroupLabel(QtWidgets.QLabel):
    """An all-caps group heading ("ORGANIZE"), 11 px / 600 / +0.12em."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setObjectName("sidebarSectionLabel")
        self.setContentsMargins(12, 12, 12, 6)


class _BrandMark(QtWidgets.QWidget):
    """34 px glass tile with three accent bars."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(34, 34)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import paint_brand_tile
        p = QtGui.QPainter(self)
        try:
            paint_brand_tile(p, QtCore.QRectF(self.rect()), (9, 16, 12))
        finally:
            p.end()


# ── Sidebar widget ─────────────────────────────────────────────────────────────

class Sidebar(QtWidgets.QWidget):
    """Floating glass navigation panel.

    Signals:
        nav_changed(str)     — key of the newly activated item
        exit_requested()     — kept for callers; the panel has no Exit item
                               any more (quit with the window's close button
                               or Ctrl+Q)
    """

    nav_changed    = Signal(str)
    exit_requested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setMinimumWidth(SIDEBAR_MIN_WIDTH)
        self.setMaximumWidth(SIDEBAR_WIDTH)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

        self._buttons: dict[str, AnimatedNavButton] = {}
        self._labels: list[QtWidgets.QLabel] = []
        self._active_key: str = ""

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(14, 20, 14, 20)
        root_layout.setSpacing(0)

        # ── Brand ─────────────────────────────────────────────────────────
        brand = QtWidgets.QHBoxLayout()
        brand.setContentsMargins(10, 4, 10, 18)
        brand.setSpacing(10)
        brand.addWidget(_BrandMark())
        # No inline colour: the label paints with the palette's WindowText,
        # i.e. the active theme's text_primary (DS-12 / SHELL-18).
        logo_lbl = QtWidgets.QLabel("AlphaDEX")
        logo_lbl.setObjectName("brandName")
        from gui.themes.glass import font
        logo_font = font("heading", px=19, weight=700)
        logo_font.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, -0.02 * 19)
        logo_lbl.setFont(logo_font)
        brand.addWidget(logo_lbl, 1)
        root_layout.addLayout(brand)
        self._logo = logo_lbl

        # ── Scrollable nav (only scrolls in very short windows) ───────────
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("sidebarScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)
        self._scroll = scroll

        self._nav_container = NavContainer()
        nav_layout = QtWidgets.QVBoxLayout(self._nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(ITEM_GAP)

        tooltips = _nav_tooltips()
        for section in NAV_STRUCTURE:
            if section.title:
                hdr = _GroupLabel(section.title)
                nav_layout.addWidget(hdr)
                self._labels.append(hdr)
            else:
                # The bottom group: pushed to the foot of the panel.
                nav_layout.addStretch(1)
                nav_layout.addSpacing(16)
            for item in section.items:
                btn = AnimatedNavButton(item.label, item.key)
                btn.clicked_key.connect(self._on_nav_click)
                if tooltips.get(item.key):
                    btn.setToolTip(tooltips[item.key])
                nav_layout.addWidget(btn)
                self._buttons[item.key] = btn

        scroll.setWidget(self._nav_container)
        root_layout.addWidget(scroll, stretch=1)

        # ── Activate first item ───────────────────────────────────────────
        first_key = NAV_STRUCTURE[0].items[0].key if NAV_STRUCTURE else ""
        if first_key:
            self.activate(first_key)

        QtCore.QTimer.singleShot(0, self._snap_pill)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(SIDEBAR_WIDTH, 600)

    # ── Public API ────────────────────────────────────────────────────────

    def activate(self, key: str) -> None:
        """Programmatically activate a nav item (no signal emitted)."""
        if self._active_key and self._active_key in self._buttons:
            self._buttons[self._active_key].active = False

        self._active_key = key
        if key in self._buttons:
            btn = self._buttons[key]
            btn.active = True
            self._nav_container.move_pill(float(btn.y()), float(btn.height()))

    def set_badge(self, key: str, count: int) -> None:
        if key not in self._buttons:
            return
        self._buttons[key].badge = count

    def active_key(self) -> str:
        return self._active_key

    # ── Painting: the glass panel ─────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface

        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().sidebar)
        finally:
            p.end()

    # ── Layout ────────────────────────────────────────────────────────────

    def _snap_pill(self) -> None:
        key = self._active_key
        if key and key in self._buttons:
            btn = self._buttons[key]
            self._nav_container.move_pill(
                float(btn.y()), float(btn.height()), instant=True
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Defer one tick so Qt has finished updating the scroll-area viewport
        # height before we read it.
        QtCore.QTimer.singleShot(0, self._relayout_buttons)

    def _relayout_buttons(self) -> None:
        """Items are 40 px high; in a short window they shrink to fit (never
        below 32 px; below that the nav scrolls)."""
        vp_h = self._scroll.viewport().height()
        if vp_h <= 0 or not self._buttons:
            return
        labels = sum(lbl.sizeHint().height() for lbl in self._labels)
        n = len(self._buttons)
        gaps = (n + len(self._labels)) * ITEM_GAP + 16
        btn_h = int(max(ITEM_MIN_HEIGHT, min(ITEM_HEIGHT, (vp_h - labels - gaps) // n)))
        for btn in self._buttons.values():
            if btn.height() != btn_h or btn.minimumHeight() != btn_h:
                btn.setFixedHeight(btn_h)
        # Re-anchor the pill after heights change
        QtCore.QTimer.singleShot(0, self._snap_pill)

    def _on_nav_click(self, key: str) -> None:
        self.activate(key)
        self.nav_changed.emit(key)
