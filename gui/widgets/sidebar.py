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


NAV_STRUCTURE: list[NavSection] = [
    NavSection("ORGANIZE", [
        NavItem("indexer",      "Indexer",        "🗂"),
        NavItem("library_sync", "Library Sync",   "🔄"),
    ]),
    NavSection("CLEAN UP", [
        NavItem("duplicates",   "Duplicates",     "🔍"),
        NavItem("similarity",   "Similarity",     "⚖"),
        NavItem("tag_fixer",    "Tag Fixer",      "🏷"),
        NavItem("genres",       "Genre Normalizer","🎸"),
    ]),
    NavSection("PLAYLISTS", [
        NavItem("playlists",    "Playlists",      "🎵"),
        NavItem("clustered",    "Clustered",      "📊"),
        NavItem("graph",        "Music Graph",    "🌐"),
    ]),
    NavSection("PLAYER", [
        NavItem("player",       "Player",         "▶"),
        NavItem("compression",  "Compression",    "📦"),
    ]),
    NavSection("TOOLS", [
        NavItem("tools",        "Utilities",      "🛠"),
        NavItem("help",         "Help",           "?"),
    ]),
]


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

        self._pill_anim = QtCore.QVariantAnimation(self)
        self._pill_anim.setDuration(320)
        self._pill_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutBack)
        self._pill_anim.valueChanged.connect(self._on_pill_value)

    def _on_pill_value(self, value: object) -> None:
        self._pill_y = float(value)  # type: ignore[arg-type]
        self.update()

    def move_pill(self, y: float, h: float, *, instant: bool = False) -> None:
        self._pill_h = h
        self._pill_visible = True
        if instant:
            self._pill_anim.stop()
            self._pill_y = y
            self.update()
            return
        self._pill_anim.stop()
        self._pill_anim.setStartValue(float(self._pill_y))
        self._pill_anim.setEndValue(float(y))
        self._pill_anim.start()

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
        logo_lbl = QtWidgets.QLabel("AlphaDEX")
        logo_lbl.setFixedHeight(44)
        logo_lbl.setStyleSheet(
            "color: #f8fafc; font-size: 22px; font-weight: 700; "
            "padding: 0 8px; letter-spacing: -0.02em;"
        )
        root_layout.addWidget(logo_lbl)

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

        for section in NAV_STRUCTURE:
            hdr = QtWidgets.QLabel(section.title)
            hdr.setObjectName("sidebarSectionLabel")
            hdr.setFixedHeight(20)
            if nav_layout.count() > 0:
                spacer = QtWidgets.QWidget()
                spacer.setFixedHeight(2)
                spacer.setStyleSheet("background: transparent;")
                nav_layout.addWidget(spacer)
            nav_layout.addWidget(hdr)

            for item in section.items:
                btn = AnimatedNavButton(item.label, item.key, item.icon)
                btn.clicked_key.connect(self._on_nav_click)
                nav_layout.addWidget(btn)
                self._buttons[item.key] = btn

        nav_layout.addStretch(1)
        scroll.setWidget(self._nav_container)
        root_layout.addWidget(scroll, stretch=1)

        # ── Separator ─────────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(255,255,255,0.08); border: none;")
        root_layout.addWidget(sep)

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
        """Paint sidebar_bg at ~75 % opacity over the window-spanning gradient.

        The same two radial glows used by GradientWidget (workspace canvas) are
        reproduced here with window-relative origins, but at higher alpha and
        with a larger radius so they reach the far-left panel.  A translucent
        sidebar_bg overlay (alpha ≈ 190, ~75 %) is applied on top, giving the
        "glass panel over a shared gradient" appearance without relying on any
        platform compositing.
        """
        from gui.themes.manager import get_manager
        t = get_manager().current
        p = QtGui.QPainter(self)
        rect = self.rect()

        # ── Step 1: solid sidebar base ─────────────────────────────────────
        p.fillRect(rect, QtGui.QColor(t.sidebar_bg))

        # ── Step 2: window-spanning gradient glows ─────────────────────────
        # Use full diagonal as radius so both glows reach the sidebar even
        # though it sits in the far-left corner of the window.
        # Higher alpha than the workspace so the colour tint is visible over
        # the dark sidebar_bg, simulating the gradient "shining through".
        if _gradient_enabled() and rect.width() > 0 and rect.height() > 0:
            paint_window_gradient(
                p, self, t,
                radius_scale=1.05,
                alpha_tr=45 if t.variant == "dark" else 28,
                alpha_bl=60 if t.variant == "dark" else 38,
            )

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
