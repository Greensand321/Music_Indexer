"""Left navigation sidebar with grouped section items."""
from __future__ import annotations

from dataclasses import dataclass, field

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.animations import AnimatedNavButton
from gui.widgets.gradient_bg import _gradient_enabled, paint_window_gradient

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
    A group with an empty title is drawn under a separator instead of a
    heading (Activity · Settings · Help)."""
    from gui.navigation import nav_groups

    return [
        NavSection(title, [NavItem(p.key, p.label, p.icon) for p in pages])
        for title, pages in nav_groups()
    ]


NAV_STRUCTURE: list[NavSection] = _build_nav_structure()


def _nav_tooltips() -> dict[str, str]:
    from gui.navigation import PAGES
    return {p.key: p.intent for p in PAGES}


# ── Sliding pill container ─────────────────────────────────────────────────────

class NavContainer(QtWidgets.QWidget):
    """Transparent nav content container that paints the sliding accent pill."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")

        self._pill_y: float = 0.0
        self._pill_h: float = 44.0
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

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if not self._pill_visible:
            return
        try:
            from gui.themes.manager import get_manager
            t = get_manager().current
        except Exception:
            return

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        pill = QtCore.QRectF(
            3.0, self._pill_y + 1.0,
            float(self.width()) - 6.0, self._pill_h - 2.0,
        )
        path = QtGui.QPainterPath()
        path.addRoundedRect(pill, 8.0, 8.0)
        p.fillPath(path, QtGui.QBrush(QtGui.QColor(t.sidebar_active)))
        p.end()


class _Separator(QtWidgets.QWidget):
    """1 px rule painted from the active theme's ``card_border`` token."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebarSeparator")
        self.setFixedHeight(1)

    def color(self) -> QtGui.QColor:
        from gui.themes.manager import get_manager
        return QtGui.QColor(get_manager().current.card_border)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self.color())
        p.end()


# ── Sidebar widget ─────────────────────────────────────────────────────────────

class Sidebar(QtWidgets.QWidget):
    """Dark left navigation panel.

    Signals:
        nav_changed(str)     — key of the newly activated item
        exit_requested()     — user clicked the Exit button
    """

    nav_changed    = Signal(str)
    exit_requested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(360)

        self._buttons: dict[str, AnimatedNavButton] = {}
        self._active_key: str = ""

        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(12, 16, 12, 12)
        root_layout.setSpacing(0)

        # ── Logo area ─────────────────────────────────────────────────────
        # No inline colour: the label paints with the palette's WindowText,
        # i.e. the active theme's text_primary, so it stays legible on light
        # themes too (DS-12 / SHELL-18; it used to be a fixed #f8fafc).
        logo_lbl = QtWidgets.QLabel("AlphaDEX")
        logo_lbl.setObjectName("sidebarLogo")
        logo_lbl.setFixedHeight(44)
        logo_lbl.setContentsMargins(8, 0, 8, 0)
        logo_font = QtGui.QFont(logo_lbl.font())
        logo_font.setPixelSize(22)
        logo_font.setWeight(QtGui.QFont.Weight.Bold)
        logo_font.setLetterSpacing(QtGui.QFont.SpacingType.PercentageSpacing, 98.0)
        logo_lbl.setFont(logo_font)
        root_layout.addWidget(logo_lbl)
        self._logo = logo_lbl

        # ── Scrollable nav ────────────────────────────────────────────────
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        self._scroll = scroll

        self._nav_container = NavContainer()
        nav_layout = QtWidgets.QVBoxLayout(self._nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(1)

        tooltips = _nav_tooltips()
        self._group_separators: list[_Separator] = []
        for section in NAV_STRUCTURE:
            if nav_layout.count() > 0:
                spacer = QtWidgets.QWidget()
                spacer.setFixedHeight(2)
                spacer.setStyleSheet("background: transparent;")
                nav_layout.addWidget(spacer)
            if section.title:
                hdr = QtWidgets.QLabel(section.title)
                hdr.setObjectName("sidebarSectionLabel")
                hdr.setFixedHeight(20)
                nav_layout.addWidget(hdr)
            else:
                # The untitled bottom group (Activity · Settings · Help) sits
                # under a hairline instead of a heading; same 20 px slot.
                holder = QtWidgets.QWidget()
                holder.setFixedHeight(20)
                hl = QtWidgets.QVBoxLayout(holder)
                hl.setContentsMargins(8, 9, 8, 10)
                rule = _Separator()
                hl.addWidget(rule)
                self._group_separators.append(rule)
                nav_layout.addWidget(holder)

            for item in section.items:
                btn = AnimatedNavButton(item.label, item.key, item.icon)
                btn.clicked_key.connect(self._on_nav_click)
                if tooltips.get(item.key):
                    btn.setToolTip(tooltips[item.key])
                nav_layout.addWidget(btn)
                self._buttons[item.key] = btn

        nav_layout.addStretch(1)
        scroll.setWidget(self._nav_container)
        root_layout.addWidget(scroll, stretch=1)

        # ── Separator ─────────────────────────────────────────────────────
        sep = _Separator()
        root_layout.addWidget(sep)
        self._separator = sep

        # ── Exit button (pinned at bottom) ─────────────────────────────────
        self._exit_btn = AnimatedNavButton("Exit", "exit", "⏻", is_exit=True)
        self._exit_btn.clicked_key.connect(lambda _: self.exit_requested.emit())
        root_layout.addWidget(self._exit_btn)

        # ── Activate first item ───────────────────────────────────────────
        first_key = NAV_STRUCTURE[0].items[0].key if NAV_STRUCTURE else ""
        if first_key:
            self.activate(first_key)

        QtCore.QTimer.singleShot(0, self._snap_pill)

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

    # ── Private ───────────────────────────────────────────────────────────

    # ── Background painting ───────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Unified Canvas: sidebar is a barely-there glass pane over the shared gradient.

        The window-spanning gradient is painted at the same coordinates used by
        GradientWidget, so the two glows read as a single canvas that stretches
        across both the sidebar and the workspace.  A very thin sidebar_bg tint
        (alpha ≈ 28, ~11 %) is then laid on top — just enough to hint at the
        panel boundary without breaking the illusion of one unified background.
        A 1 px right-edge separator is drawn last to keep the nav content
        visually anchored without a heavy divider line.
        """
        from gui.themes.manager import get_manager
        t = get_manager().current
        p = QtGui.QPainter(self)
        rect = self.rect()
        gradient = _gradient_enabled()   # cached flag: no config I/O in paint
        try:
            # ── Step 1: window-spanning gradient (identical to workspace) ──
            if gradient and rect.width() > 0 and rect.height() > 0:
                paint_window_gradient(
                    p, self, t,
                    radius_scale=1.05,
                    alpha_tr=45 if t.variant == "dark" else 28,
                    alpha_bl=60 if t.variant == "dark" else 38,
                )
            else:
                # Gradient disabled — fall back to a solid sidebar background
                p.fillRect(rect, QtGui.QColor(t.sidebar_bg))

            # ── Step 2: barely-there sidebar tint (~11 % opacity) ──────────
            if gradient:
                tint = QtGui.QColor(t.sidebar_bg)
                tint.setAlpha(28 if t.variant == "dark" else 20)
                p.fillRect(rect, tint)

            # ── Step 3: 1 px right-edge separator ──────────────────────────
            sep_color = QtGui.QColor(t.card_border)
            sep_color.setAlpha(80)
            p.setPen(QtGui.QPen(sep_color, 1))
            p.drawLine(rect.right(), rect.top(), rect.right(), rect.bottom())
        finally:
            p.end()

    def _snap_pill(self) -> None:
        key = self._active_key
        if key and key in self._buttons:
            btn = self._buttons[key]
            self._nav_container.move_pill(
                float(btn.y()), float(btn.height()), instant=True
            )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Defer one tick so Qt has finished updating the scroll-area viewport
        # height before we read it.
        QtCore.QTimer.singleShot(0, self._relayout_buttons)

    def _relayout_buttons(self) -> None:
        """Resize all nav buttons to fill the scroll-area viewport exactly."""
        vp_h = self._scroll.viewport().height()
        if vp_h <= 0:
            return

        n_btns     = len(self._buttons)
        n_sections = len(NAV_STRUCTURE)
        n_spacers  = n_sections - 1
        n_items    = n_sections + n_btns + n_spacers   # items before stretch
        overhead   = (n_sections * 20        # section header fixed heights
                      + n_spacers  * 2       # inter-section spacer heights
                      + (n_items - 1) * 1)   # layout spacing gaps

        btn_h = max(32, min(68, (vp_h - overhead) // n_btns))

        for btn in self._buttons.values():
            btn.setFixedHeight(btn_h)
        self._exit_btn.setFixedHeight(btn_h)

        # Re-anchor the pill after heights change
        QtCore.QTimer.singleShot(0, self._snap_pill)

    def _on_nav_click(self, key: str) -> None:
        self.activate(key)
        self.nav_changed.emit(key)
