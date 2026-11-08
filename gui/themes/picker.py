"""ThemePickerPanel, ThemePickerDialog and AutoThemeDialog — runtime theme UI.

The panel (Settings → Appearance) leads with **Liquid Glass**, the default
look: its swatch, the four accents and "Reduce transparency". Below it,
Auto (on a well row), then the 14 **Classic** themes (dark, then light) in a
responsive grid: as many equal columns as fit the width, the same cell width
in both groups. Everything here applies instantly through the ThemeManager
(which persists it).

A swatch is a miniature of its theme: a preview (sidebar strip, top bar,
card) above a name strip, both on the theme's own background, so the name is
drawn in the theme's own text colour and never overlaps the preview.
"""
from __future__ import annotations

import math

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.tokens import (
    ACCENTS, CLASSIC_THEMES, DARK_THEMES, GLASS_KEY, LIGHT_THEMES, THEMES, ThemeTokens,
)
from gui.themes.manager import get_manager


# ── Swatch card ───────────────────────────────────────────────────────────────

class _SwatchCard(QtWidgets.QAbstractButton):
    """Single theme swatch: a preview of the theme over its name."""

    SIZE = QtCore.QSize(120, 88)
    MIN_WIDTH = 104
    NAME_H = 22

    def __init__(self, tokens: ThemeTokens, parent: QtWidgets.QWidget | None = None,
                 *, fixed: bool = True) -> None:
        super().__init__(parent)
        self._tokens = tokens
        self._selected = False
        if fixed:
            self.setFixedSize(self.SIZE)
        else:
            # In a _SwatchGrid: the grid sets the width (all cells equal).
            self.setMinimumSize(self.MIN_WIDTH, self.SIZE.height())
            self.setMaximumHeight(self.SIZE.height())
        self.setCheckable(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setToolTip(tokens.name)
        self.setAccessibleName(f"{tokens.name} theme")

    @property
    def tokens(self) -> ThemeTokens:
        return self._tokens

    def setSelected(self, val: bool) -> None:
        self._selected = val
        self.setChecked(val)
        self.update()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(self.SIZE)

    def label_color(self) -> QtGui.QColor:
        """The name sits on the swatch's own ``content_bg``, so it is drawn in
        the swatch's own ``text_primary`` (DS-15: it used the *active*
        theme's text colour, illegible on dark swatches under a light theme)."""
        return QtGui.QColor(self._tokens.text_primary)

    def preview_rect(self) -> QtCore.QRectF:
        """Where the miniature is drawn (above the name strip)."""
        card = QtCore.QRectF(self.rect()).adjusted(3, 3, -3, -3)
        return card.adjusted(6, 6, -6, -self.NAME_H)

    def name_rect(self) -> QtCore.QRectF:
        card = QtCore.QRectF(self.rect()).adjusted(3, 3, -3, -3)
        return QtCore.QRectF(card.left() + 4, card.bottom() - self.NAME_H,
                             card.width() - 8, self.NAME_H)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        t = self._tokens
        current = get_manager().current
        selected = self._selected or self.isChecked()

        # ── Card background ───────────────────────────────────────────────
        card_r = QtCore.QRectF(self.rect()).adjusted(3, 3, -3, -3)
        path = QtGui.QPainterPath()
        path.addRoundedRect(card_r, 12, 12)
        if t.is_glass:
            # A hint of the glass look: an accent glow behind the panels.
            glass_accent = current.accent if current.is_glass else t.accent
            grad = QtGui.QRadialGradient(card_r.topRight(), card_r.width())
            glow = QtGui.QColor(glass_accent)
            glow.setAlpha(110)
            grad.setColorAt(0.0, glow)
            grad.setColorAt(1.0, QtGui.QColor(t.content_bg))
            p.setBrush(QtGui.QBrush(grad))
        else:
            p.setBrush(QtGui.QColor(t.content_bg))
        if selected:
            p.setPen(QtGui.QPen(QtGui.QColor(current.accent), 2.0))
        elif self.underMouse():
            p.setPen(QtGui.QPen(QtGui.QColor(current.text_secondary), 1.0))
        else:
            p.setPen(QtGui.QPen(QtGui.QColor(current.card_border), 1.0))
        p.drawPath(path)

        # ── Preview (sidebar strip + top bar + card), above the name ──────
        inner = self.preview_rect()
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        sb_rect = QtCore.QRectF(inner.left(), inner.top(), 24, inner.height())
        p.setBrush(QtGui.QColor(t.sidebar_bg))
        p.drawRoundedRect(sb_rect, 5, 5)
        dot_x = sb_rect.center().x()
        for i, colour in enumerate([t.sidebar_accent, t.accent, t.text_muted]):
            p.setBrush(QtGui.QColor(colour))
            p.drawEllipse(QtCore.QPointF(dot_x, sb_rect.top() + 8 + i * 9), 2.6, 2.6)

        ct = QtCore.QRectF(inner.left() + 28, inner.top(), inner.width() - 28, inner.height())
        bar = QtCore.QRectF(ct.left(), ct.top(), ct.width(), 12)
        p.setBrush(QtGui.QColor(t.card_bg))
        p.drawRoundedRect(bar, 4, 4)
        btn = QtCore.QRectF(ct.right() - 24, ct.top() + 2, 22, 8)
        p.setBrush(QtGui.QColor(t.accent))
        p.drawRoundedRect(btn, 3, 3)
        mock = QtCore.QRectF(ct.left(), ct.top() + 16, ct.width(), ct.height() - 16)
        p.setBrush(QtGui.QColor(t.card_bg))
        p.drawRoundedRect(mock, 4, 4)
        p.setBrush(QtGui.QColor(t.text_primary))
        p.drawRoundedRect(QtCore.QRectF(mock.left() + 5, mock.top() + 6, mock.width() * 0.62, 3),
                          1.5, 1.5)
        p.setBrush(QtGui.QColor(t.text_muted))
        p.drawRoundedRect(QtCore.QRectF(mock.left() + 5, mock.top() + 13, mock.width() * 0.42, 3),
                          1.5, 1.5)

        # ── Name strip ────────────────────────────────────────────────────
        from gui.themes.glass import font
        name_font = font("caption", weight=600 if selected else 500)
        p.setFont(name_font)
        p.setPen(self.label_color())
        rect = self.name_rect()
        p.drawText(rect, int(QtCore.Qt.AlignmentFlag.AlignCenter),
                   QtGui.QFontMetrics(name_font).elidedText(
                       t.name, QtCore.Qt.TextElideMode.ElideRight, int(rect.width())))

        # ── Checkmark if selected ─────────────────────────────────────────
        if selected:
            ck = QtCore.QRectF(inner.right() - 16, inner.top() - 2, 18, 18)
            p.setPen(QtGui.QPen(QtGui.QColor(t.content_bg), 2.0))
            p.setBrush(QtGui.QColor(current.accent))
            p.drawEllipse(ck)
            pen = QtGui.QPen(QtGui.QColor(current.text_inverse), 1.6)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(pen)
            cx, cy = ck.center().x(), ck.center().y()
            p.drawLine(QtCore.QPointF(cx - 3.5, cy), QtCore.QPointF(cx - 1, cy + 2.5))
            p.drawLine(QtCore.QPointF(cx - 1, cy + 2.5), QtCore.QPointF(cx + 3.5, cy - 2.5))

        if self.hasFocus():
            p.setPen(QtGui.QPen(QtGui.QColor(current.input_focus), 1.5))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(card_r.adjusted(-1.5, -1.5, 1.5, 1.5), 13, 13)
        p.end()

    def enterEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.update()
        super().leaveEvent(event)


# ── Responsive swatch grid ────────────────────────────────────────────────────

class _GridFlow(QtWidgets.QLayout):
    """Equal cells in as many columns as fit ``cell_min``; the cells share
    the width (up to ``cell_max``) and wrap onto more rows when narrow.
    Height-for-width, so a scroll area or a card gives it exactly the rows it
    needs."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, *, cell_min: int = 104,
                 cell_max: int = 150, cell_h: int = 88, spacing: int = 10) -> None:
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._cell_min, self._cell_max, self._cell_h = cell_min, cell_max, cell_h
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # QLayout API
    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:  # noqa: N802
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QtCore.QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        m = self.contentsMargins()
        cols = max(1, min(len(self._items), 5))
        w = cols * self._cell_min + (cols - 1) * self.spacing()
        return QtCore.QSize(w + m.left() + m.right(), self.heightForWidth(w))

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        m = self.contentsMargins()
        return QtCore.QSize(self._cell_min + m.left() + m.right(),
                            self._cell_h + m.top() + m.bottom())

    # Geometry
    def columns(self, width: int) -> int:
        sp = self.spacing()
        return max(1, (width + sp) // (self._cell_min + sp))

    def _arrange(self, rect: QtCore.QRect, *, apply: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        n = len(self._items)
        if n == 0:
            return m.top() + m.bottom()
        sp = self.spacing()
        cols = self.columns(r.width())
        cell_w = min(float(self._cell_max), (r.width() - sp * (cols - 1)) / cols)
        rows = math.ceil(n / cols)
        if apply:
            for i, item in enumerate(self._items):
                row, col = divmod(i, cols)
                x = r.x() + col * (cell_w + sp)
                y = r.y() + row * (self._cell_h + sp)
                item.setGeometry(QtCore.QRect(int(round(x)), int(y), int(cell_w), self._cell_h))
        return rows * self._cell_h + (rows - 1) * sp + m.top() + m.bottom()


def _swatch_grid(swatches: list[_SwatchCard]) -> QtWidgets.QWidget:
    box = QtWidgets.QWidget()
    box.setObjectName("swatchGrid")
    flow = _GridFlow(box)
    for sw in swatches:
        flow.addWidget(sw)
    policy = box.sizePolicy()
    policy.setHeightForWidth(True)
    box.setSizePolicy(policy)
    return box


# ── Small painted parts ───────────────────────────────────────────────────────

class _AccentSwatch(QtWidgets.QAbstractButton):
    """A 24 px accent disc; the chosen one carries a ring."""

    SIZE = 34

    def __init__(self, key: str, name: str, colour: str,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self._colour = colour
        self.setCheckable(True)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setToolTip(f"{name} accent")
        self.setAccessibleName(f"{name} accent")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(self.rect())
            if self.isChecked():
                p.setPen(QtGui.QPen(QtGui.QColor(t.text_primary), 2.0))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(r.adjusted(1.5, 1.5, -1.5, -1.5))
            elif self.hasFocus():
                p.setPen(QtGui.QPen(QtGui.QColor(t.input_focus), 1.5))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(r.adjusted(1.5, 1.5, -1.5, -1.5))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(self._colour))
            p.drawEllipse(r.adjusted(5, 5, -5, -5))
        finally:
            p.end()


class _AutoGlyph(QtWidgets.QWidget):
    """Auto's mark: a circle half filled (day / night), painted."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(28, 28)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(4, 4, 20, 20)
            p.setPen(QtGui.QPen(QtGui.QColor(t.text_primary), 1.6))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawEllipse(r)
            half = QtGui.QPainterPath()
            half.moveTo(r.center())
            half.arcTo(r, 90, 180)
            half.closeSubpath()
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(t.text_primary))
            p.drawPath(half)
        finally:
            p.end()


class _WellRow(QtWidgets.QWidget):
    """A row on the material's well (radius 16)."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface

        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), current_material().well)
        finally:
            p.end()


# ── Section header label ──────────────────────────────────────────────────────

def _section_label(text: str, parent: QtWidgets.QWidget | None = None) -> QtWidgets.QLabel:
    from gui.themes.glass import font

    lbl = QtWidgets.QLabel(text, parent)
    lbl.setObjectName("sectionHeader")
    lbl.setFont(font("label", weight=600))
    return lbl


def _group_label(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("mutedLabel")
    return lbl


# ── AutoThemeDialog ───────────────────────────────────────────────────────────

class AutoThemeDialog(QtWidgets.QDialog):
    """Configure the dark / light pair for Auto mode."""

    pair_selected = Signal(str, str)   # dark_key, light_key

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auto Theme — Select Pair")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint
        )

        mgr = get_manager()
        self._dark_key = mgr.auto_dark()
        self._light_key = mgr.auto_light()

        self._dark_swatches: list[_SwatchCard] = []
        self._light_swatches: list[_SwatchCard] = []

        self._build_ui()
        self._refresh_selections()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(14)

        desc = QtWidgets.QLabel(
            "Auto switches between these two themes with your system's day and night "
            "(or the time of day where the system does not say)."
        )
        desc.setWordWrap(True)
        desc.setObjectName("mutedLabel")
        root.addWidget(desc)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(inner)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(10)

        vbox.addWidget(_section_label("Night theme (dark)"))
        vbox.addWidget(self._make_swatch_grid(DARK_THEMES, self._dark_swatches, is_dark=True))
        vbox.addSpacing(8)
        vbox.addWidget(_section_label("Day theme (light)"))
        vbox.addWidget(self._make_swatch_grid(LIGHT_THEMES, self._light_swatches, is_dark=False))
        vbox.addStretch(1)

        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        buttons = QtWidgets.QDialogButtonBox()
        cancel_btn = buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setAutoDefault(False)
        ok_btn = buttons.addButton("Apply Auto Theme", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        ok_btn.setObjectName("primaryBtn")
        ok_btn.setDefault(True)
        cancel_btn.clicked.connect(self.reject)
        ok_btn.clicked.connect(self._accept)
        root.addWidget(buttons)

    def _make_swatch_grid(
        self,
        keys: list[str],
        swatch_list: list[_SwatchCard],
        is_dark: bool,
    ) -> QtWidgets.QWidget:
        cards = []
        for key in keys:
            card = _SwatchCard(THEMES[key], fixed=False)
            card.clicked.connect(lambda checked, k=key, d=is_dark: self._on_swatch_clicked(k, d))
            swatch_list.append(card)
            cards.append(card)
        return _swatch_grid(cards)

    def _on_swatch_clicked(self, key: str, is_dark: bool) -> None:
        if is_dark:
            self._dark_key = key
            for sw in self._dark_swatches:
                sw.setSelected(sw.tokens.key == key)
        else:
            self._light_key = key
            for sw in self._light_swatches:
                sw.setSelected(sw.tokens.key == key)

    def _refresh_selections(self) -> None:
        for sw in self._dark_swatches:
            sw.setSelected(sw.tokens.key == self._dark_key)
        for sw in self._light_swatches:
            sw.setSelected(sw.tokens.key == self._light_key)

    def _accept(self) -> None:
        self.pair_selected.emit(self._dark_key, self._light_key)
        self.accept()


# ── ThemePickerPanel ──────────────────────────────────────────────────────────

class ThemePickerPanel(QtWidgets.QWidget):
    """Liquid Glass, Auto and the 14 classic themes. Changes apply
    instantly. Embedded in Settings → Appearance; ``ThemePickerDialog`` wraps
    it in a window."""

    theme_applied = Signal(str)   # key or "auto"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        mgr = get_manager()
        self._current_key = "auto" if mgr.is_auto() else mgr.current.key
        self._all_swatches: list[_SwatchCard] = []

        self._build_ui()
        self._refresh_selection()
        # Follow switches made elsewhere (another picker, Auto's day/night flip).
        mgr.theme_changed.connect(self._on_theme_changed)

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        # Liquid Glass (the default look) with its accents
        root.addWidget(self._make_glass_row())

        # Auto card (special)
        root.addWidget(self._make_auto_card())

        # The classic themes, opaque, through the same components
        root.addSpacing(4)
        root.addWidget(_section_label("Classic themes"))
        for label, keys in (("Dark", DARK_THEMES), ("Light", LIGHT_THEMES)):
            root.addWidget(_group_label(label))
            root.addWidget(self._make_swatch_grid([k for k in keys if k in CLASSIC_THEMES]))

    def _make_glass_row(self) -> QtWidgets.QWidget:
        """The Liquid Glass swatch, its four accents and Reduce transparency."""
        mgr = get_manager()
        row = QtWidgets.QWidget()
        row.setObjectName("glassRow")
        hl = QtWidgets.QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(18)

        glass = _SwatchCard(THEMES[GLASS_KEY])
        glass.clicked.connect(lambda checked=False: self._apply_theme(GLASS_KEY))
        self._all_swatches.append(glass)
        hl.addWidget(glass, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        col = QtWidgets.QVBoxLayout()
        col.setSpacing(6)
        title = QtWidgets.QLabel("Liquid Glass")
        title.setObjectName("cardTitle")
        desc = QtWidgets.QLabel(
            "Translucent glass over a blurred wall of your own album covers. The default look."
        )
        desc.setObjectName("mutedLabel")
        desc.setWordWrap(True)
        col.addWidget(title)
        col.addWidget(desc)

        accents = QtWidgets.QHBoxLayout()
        accents.setSpacing(6)
        accents.addWidget(_group_label("Accent"))
        accents.addSpacing(4)
        self._accent_buttons: dict[str, _AccentSwatch] = {}
        for key, accent in ACCENTS.items():
            sw = _AccentSwatch(accent.key, accent.name, accent.base)
            sw.clicked.connect(lambda checked=False, k=key: self._apply_accent(k))
            accents.addWidget(sw)
            self._accent_buttons[key] = sw
        accents.addStretch(1)
        col.addLayout(accents)

        self._reduce_transparency_cb = QtWidgets.QCheckBox(
            "Reduce transparency — solid surfaces instead of glass")
        self._reduce_transparency_cb.setToolTip(
            "Opaque panels and cards: easier to read over busy album art, and lighter "
            "on the graphics card. Classic themes are always opaque.")
        self._reduce_transparency_cb.setChecked(mgr.reduce_transparency())
        self._reduce_transparency_cb.toggled.connect(mgr.set_reduce_transparency)
        col.addWidget(self._reduce_transparency_cb)
        hl.addLayout(col, 1)
        self._sync_accents()
        return row

    def _apply_accent(self, key: str) -> None:
        """An accent belongs to Liquid Glass: choosing one shows it."""
        mgr = get_manager()
        mgr.set_accent(key)
        if self._current_key != GLASS_KEY:
            self._apply_theme(GLASS_KEY)
        self._sync_accents()

    def _sync_accents(self) -> None:
        current = get_manager().accent()
        for key, sw in getattr(self, "_accent_buttons", {}).items():
            sw.setChecked(key == current)

    def _make_auto_card(self) -> QtWidgets.QWidget:
        """Build the special Auto-mode row (on a well)."""
        from gui.widgets.glyphs import set_glyph

        container = _WellRow()
        container.setObjectName("autoCard")
        hl = QtWidgets.QHBoxLayout(container)
        hl.setContentsMargins(14, 10, 10, 10)
        hl.setSpacing(12)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setSpacing(1)
        name_lbl = QtWidgets.QLabel("Auto")
        from gui.themes.glass import font
        name_lbl.setFont(font("body", weight=600))
        desc_lbl = QtWidgets.QLabel("Follows your system's day and night")
        desc_lbl.setObjectName("mutedLabel")

        mgr = get_manager()
        # Always the configured night theme — in Auto mode during the day,
        # mgr.current is the *light* theme (DS-16).
        dark_name = THEMES[mgr.auto_dark()].name if mgr.auto_dark() in THEMES else "—"
        light_name = THEMES[mgr.auto_light()].name if mgr.auto_light() in THEMES else "—"
        pair_lbl = QtWidgets.QLabel(f"Night: {dark_name}  ·  Day: {light_name}")
        pair_lbl.setObjectName("mutedLabel")
        self._auto_pair_label = pair_lbl

        text_col.addWidget(name_lbl)
        text_col.addWidget(desc_lbl)
        text_col.addWidget(pair_lbl)

        hl.addWidget(_AutoGlyph(), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        hl.addLayout(text_col, 1)

        cfg_btn = QtWidgets.QPushButton("Configure…")
        cfg_btn.setToolTip("Choose the night and day themes")
        cfg_btn.clicked.connect(self._open_auto_config)
        hl.addWidget(cfg_btn)

        self._auto_select_btn = QtWidgets.QPushButton("Select")
        self._auto_select_btn.setObjectName("primaryBtn")
        set_glyph(self._auto_select_btn, "check", role="inverse")
        self._auto_select_btn.clicked.connect(self._apply_auto)
        hl.addWidget(self._auto_select_btn)

        return container

    def _make_swatch_grid(self, keys: list[str]) -> QtWidgets.QWidget:
        cards = []
        for key in keys:
            card = _SwatchCard(THEMES[key], fixed=False)
            card.clicked.connect(lambda checked, k=key: self._apply_theme(k))
            self._all_swatches.append(card)
            cards.append(card)
        return _swatch_grid(cards)

    # ── Theme application ─────────────────────────────────────────────────

    def _apply_theme(self, key: str) -> None:
        self._current_key = key
        get_manager().apply(key)
        self._refresh_selection()
        self.theme_applied.emit(key)

    def _apply_auto(self) -> None:
        self._current_key = "auto"
        # Check if auto pair is configured, otherwise open config first
        mgr = get_manager()
        if not mgr.is_auto():
            mgr.apply("auto")
        self._refresh_selection()
        self.theme_applied.emit("auto")

    def _open_auto_config(self) -> None:
        dlg = AutoThemeDialog(self)
        dlg.pair_selected.connect(self._on_auto_pair_selected)
        dlg.exec()

    def _on_auto_pair_selected(self, dark_key: str, light_key: str) -> None:
        mgr = get_manager()
        mgr.configure_auto(dark_key, light_key)
        # Update pair label
        dark_name = THEMES[dark_key].name if dark_key in THEMES else dark_key
        light_name = THEMES[light_key].name if light_key in THEMES else light_key
        self._auto_pair_label.setText(f"Night: {dark_name}  ·  Day: {light_name}")
        # Apply auto mode
        self._apply_auto()

    # ── Selection refresh ─────────────────────────────────────────────────

    def _on_theme_changed(self, _tokens=None) -> None:
        mgr = get_manager()
        key = "auto" if mgr.is_auto() else mgr.current.key
        if key != self._current_key:
            self._current_key = key
            self._refresh_selection()
        self._sync_accents()
        cb = getattr(self, "_reduce_transparency_cb", None)
        if cb is not None and cb.isChecked() != mgr.reduce_transparency():
            was = cb.blockSignals(True)
            cb.setChecked(mgr.reduce_transparency())
            cb.blockSignals(was)
        for sw in self._all_swatches:
            sw.update()

    def _refresh_selection(self) -> None:
        is_auto = self._current_key == "auto"
        self._auto_select_btn.setText("Active" if is_auto else "Select")
        self._auto_select_btn.setEnabled(not is_auto)
        for sw in self._all_swatches:
            sw.setSelected(not is_auto and sw.tokens.key == self._current_key)


# ── ThemePickerDialog ─────────────────────────────────────────────────────────

class ThemePickerDialog(QtWidgets.QDialog):
    """``ThemePickerPanel`` in its own non-modal window (the app itself shows
    the panel in Settings → Appearance)."""

    theme_applied = Signal(str)   # key or "auto"

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose Theme")
        self.setModal(False)          # allow interacting with the main window
        self.setMinimumWidth(680)
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint
        )
        from gui.themes.glass import apply_font

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18)
        root.setSpacing(16)

        title = QtWidgets.QLabel("Choose Theme")
        title.setObjectName("dialogTitle")
        apply_font(title, "title")
        subtitle = QtWidgets.QLabel("Changes apply instantly")
        subtitle.setObjectName("mutedLabel")
        title_row = QtWidgets.QVBoxLayout()
        title_row.setSpacing(2)
        title_row.addWidget(title)
        title_row.addWidget(subtitle)
        root.addLayout(title_row)

        self.panel = ThemePickerPanel(self)
        self.panel.theme_applied.connect(self.theme_applied)
        root.addWidget(self.panel)
        # Names callers used before the panel was split out.
        self._auto_pair_label = self.panel._auto_pair_label
        self._all_swatches = self.panel._all_swatches

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        root.addWidget(buttons)


# ── Convenience launcher ──────────────────────────────────────────────────────

def open_theme_picker(parent: QtWidgets.QWidget | None = None) -> ThemePickerDialog:
    """Create and show a non-modal ThemePickerDialog.  Returns the dialog."""
    dlg = ThemePickerDialog(parent)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
