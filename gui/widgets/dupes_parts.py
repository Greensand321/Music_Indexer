"""Small painted parts used by the Duplicates page and the dialogs it opens.

Everything here paints from the active theme at paint time (tokens and the
glass material), so a theme, accent or transparency change only needs the
repaint the theme manager already triggers. No stylesheets, no literals.

``paint_tone_pill`` / ``tone_pill_size``
    A small rounded label in a tone ("" neutral, "accent", "success",
    "warning", "danger"): a translucent tone fill, a tone border and
    primary text, like the mockup's Experimental pill. Item delegates use
    these directly.

``ToneBadge``
    The same pill as a widget whose text and tone change in place (the
    Compare verdict, the Pair review winner badge).

``GlyphLabel``
    A non-interactive line glyph (``gui.widgets.glyphs``) for rows that need
    an icon but are not buttons.

``PathLabel``
    An ``ElidedLabel`` for file paths: middle elide, the full path in the
    tooltip, and a muted placeholder while it is empty.

``TileGrid``
    ``StatTile``s that reflow into balanced rows when the card is too narrow
    for one row (5 tiles become 3 + 2, never 4 + 1), so a summary never
    forces horizontal scrolling at the minimum window size.
"""
from __future__ import annotations

import math

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import current_material, font, qcolor
from gui.widgets.glass import ElidedLabel
from gui.widgets.glass_parts import StatTile

TONES = ("", "accent", "success", "warning", "danger")

_PAD_X = 8
_PAD_Y = 2


def _tokens():
    from gui.themes.manager import get_manager
    return get_manager().current


def pill_colours(tone: str) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
    """``(fill, border, text)`` of a pill in ``tone`` for the active theme."""
    t = _tokens()
    value = {"accent": t.accent, "success": t.success, "warning": t.warning,
             "danger": t.danger}.get(tone)
    if value is None:
        well = current_material().well
        return (qcolor(well.fill), qcolor(well.border or current_material().hairline),
                qcolor(t.text_secondary))
    return qcolor(value, 0.18), qcolor(value, 0.45), qcolor(t.text_primary)


def tone_pill_size(text: str, pill_font: QtGui.QFont, *, pad_x: int = _PAD_X,
                   pad_y: int = _PAD_Y) -> QtCore.QSize:
    fm = QtGui.QFontMetrics(pill_font)
    return QtCore.QSize(fm.horizontalAdvance(text) + 2 * pad_x + 2, fm.height() + 2 * pad_y + 2)


def paint_tone_pill(painter: QtGui.QPainter, rect: QtCore.QRect | QtCore.QRectF, text: str,
                    tone: str, pill_font: QtGui.QFont) -> None:
    """Paint a pill filling ``rect`` (fully rounded) with ``text`` centred."""
    fill, border, colour = pill_colours(tone)
    r = QtCore.QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5)
    radius = r.height() / 2
    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(border, 1.0))
        painter.setBrush(fill)
        painter.drawRoundedRect(r, radius, radius)
        painter.setPen(colour)
        painter.setFont(pill_font)
        fm = QtGui.QFontMetrics(pill_font)
        painter.drawText(QtCore.QRectF(rect), int(QtCore.Qt.AlignmentFlag.AlignCenter),
                         fm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight,
                                       max(0, int(r.width()) - 2 * _PAD_X)))
    finally:
        painter.restore()


def hero_title(text: str) -> QtWidgets.QLabel:
    """A hero card's title: the title role (20/650). Not ``cardTitle``,
    whose stylesheet font size (16) would override the explicit font."""
    from gui.themes.glass import apply_font

    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("dupHeroTitle")
    apply_font(lbl, "title")
    return lbl


def section_card(title: str, *, badge: tuple[str, str] | None = None) -> QtWidgets.QFrame:
    """A glass card for a dialog section: a heading row (``layout().itemAt(0)``,
    an ``QHBoxLayout`` callers may append to) and 16 px margins.

    Dialogs have no scroll surface to paint card shadows, so these cards are
    fill, border and highlight only."""
    from gui.themes.tokens import SPACE
    from gui.widgets.glass import GlassCard

    card = GlassCard()
    lay = QtWidgets.QVBoxLayout(card)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(SPACE.sm)
    head = QtWidgets.QHBoxLayout()
    head.setSpacing(SPACE.sm)
    lbl = QtWidgets.QLabel(title)
    lbl.setObjectName("cardTitle")
    head.addWidget(lbl)
    if badge:
        head.addWidget(ToneBadge(badge[0], badge[1]), 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
    head.addStretch(1)
    lay.addLayout(head)
    return card


def fields_row(fields: list[tuple[str, QtWidgets.QWidget]], *, spacing: int = 14) -> QtWidgets.QHBoxLayout:
    """Compact fields side by side, each a muted caption above its editor."""
    from gui.themes.glass import apply_font
    from gui.themes.tokens import SPACE

    row = QtWidgets.QHBoxLayout()
    row.setSpacing(spacing)
    for caption, editor in fields:
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(SPACE.xxs)
        lbl = QtWidgets.QLabel(caption)
        lbl.setObjectName("mutedLabel")
        apply_font(lbl, "caption")
        lbl.setBuddy(editor)
        if editor.toolTip():
            lbl.setToolTip(editor.toolTip())
        if not editor.accessibleName():
            editor.setAccessibleName(caption)
        col.addWidget(lbl)
        col.addWidget(editor)
        row.addLayout(col, 1)
    return row


def pill_font() -> QtGui.QFont:
    """The small pill font: caption size, semibold, no extra tracking."""
    return font("caption", weight=600)


class ToneBadge(QtWidgets.QWidget):
    """A pill whose text and tone change in place."""

    def __init__(self, text: str = "", tone: str = "", *, role: str = "caption",
                 weight: int = 600, pad_x: int = _PAD_X, pad_y: int = _PAD_Y,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = text
        self._tone = tone if tone in TONES else ""
        self._font = font(role, weight=weight)
        self._pad = (pad_x, pad_y)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setAccessibleName(text)

    def text(self) -> str:
        return self._text

    def tone(self) -> str:
        return self._tone

    def set_text(self, text: str, tone: str | None = None) -> None:
        self._text = text
        if tone is not None:
            self._tone = tone if tone in TONES else ""
        self.setAccessibleName(text)
        self.updateGeometry()
        self.update()

    def set_tone(self, tone: str) -> None:
        self._tone = tone if tone in TONES else ""
        self.update()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        if not self._text:
            return QtCore.QSize(0, 0)
        return tone_pill_size(self._text, self._font, pad_x=self._pad[0], pad_y=self._pad[1])

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.sizeHint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        if not self._text:
            return
        p = QtGui.QPainter(self)
        try:
            paint_tone_pill(p, self.rect(), self._text, self._tone, self._font)
        finally:
            p.end()


class GlyphLabel(QtWidgets.QWidget):
    """A line glyph that is not a button (decoration next to a row's text)."""

    def __init__(self, name: str, *, role: str = "muted", size: int = 16,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from gui.widgets.glyphs import GLYPHS

        if name not in GLYPHS:
            raise KeyError(f"unknown glyph {name!r}")
        self._name = name
        self._role = role
        self.setFixedSize(size, size)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def glyph(self) -> str:
        return self._name

    def set_glyph(self, name: str, role: str | None = None) -> None:
        self._name = name
        if role is not None:
            self._role = role
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.widgets.glyphs import paint_glyph

        t = _tokens()
        colour = {"text": t.text_primary, "muted": t.text_secondary, "accent": t.accent,
                  "success": t.success, "warning": t.warning, "danger": t.danger,
                  }.get(self._role, t.text_secondary)
        p = QtGui.QPainter(self)
        try:
            paint_glyph(p, QtCore.QRectF(self.rect()), self._name, colour)
        finally:
            p.end()


class PathLabel(ElidedLabel):
    """A file path: middle-elided, full text in the tooltip, muted placeholder."""

    def __init__(self, text: str = "", placeholder: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, QtCore.Qt.TextElideMode.ElideMiddle, parent)
        self._placeholder = placeholder
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        self.setToolTip(text)

    def setText(self, text: str) -> None:  # noqa: N802
        super().setText(text)
        self.setToolTip(text)
        self.setAccessibleName(text or self._placeholder)

    def clear(self) -> None:
        self.setText("")

    def placeholder(self) -> str:
        return self._placeholder

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        if self.text():
            super().paintEvent(event)
            return
        if not self._placeholder:
            return
        p = QtGui.QPainter(self)
        try:
            r = self.contentsRect()
            fm = QtGui.QFontMetrics(self.font())
            p.setFont(self.font())
            p.setPen(self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText))
            p.drawText(r, int(self.alignment()),
                       fm.elidedText(self._placeholder, QtCore.Qt.TextElideMode.ElideRight, r.width()))
        finally:
            p.end()


class TileGrid(QtWidgets.QWidget):
    """``StatTile``s in balanced rows: as many columns as fit at
    ``min_tile_width``, then the same number of rows with the tiles spread
    evenly (5 → 3 + 2). ``tile(key)`` returns one to update."""

    def __init__(self, tiles: list[tuple[str, str]], *, min_tile_width: int = 128,
                 spacing: int = 10, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._min_tile = min_tile_width
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(spacing)
        self._tiles: dict[str, StatTile] = {}
        for key, label in tiles:
            tile = StatTile("–", label, parent=self)
            tile.setObjectName(f"stat_{key}")
            tile.setToolTip(label)
            self._tiles[key] = tile
        self._columns = 0
        self._place(len(self._tiles))
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def tile(self, key: str) -> StatTile:
        return self._tiles[key]

    def keys(self) -> list[str]:
        return list(self._tiles)

    def columns(self) -> int:
        return self._columns

    def columns_for(self, width: int) -> int:
        n = len(self._tiles)
        if n == 0:
            return 1
        spacing = self._grid.spacing()
        fit = max(1, min(n, (width + spacing) // (self._min_tile + spacing)))
        rows = math.ceil(n / fit)
        return math.ceil(n / rows)

    def _place(self, columns: int) -> None:
        if columns == self._columns:
            return
        for tile in self._tiles.values():
            self._grid.removeWidget(tile)
        for c in range(max(columns, self._columns)):
            self._grid.setColumnStretch(c, 1 if c < columns else 0)
        for i, tile in enumerate(self._tiles.values()):
            self._grid.addWidget(tile, i // columns, i % columns)
        self._columns = columns
        self.updateGeometry()

    def _height(self) -> int:
        """Height of the current arrangement. (Not height-for-width: a widget
        with a layout reports its layout's height-for-width, which a grid of
        fixed tiles does not have, so the arrangement is re-made on resize
        and ``updateGeometry`` asks the parent for the new height.)"""
        if not self._tiles:
            return 0
        rows = math.ceil(len(self._tiles) / max(1, self._columns))
        tile_h = next(iter(self._tiles.values())).sizeHint().height()
        return rows * tile_h + (rows - 1) * self._grid.spacing()

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(min(self._min_tile, 96), self._height())

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        n = max(1, len(self._tiles))
        return QtCore.QSize(n * self._min_tile + (n - 1) * self._grid.spacing(), self._height())

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place(self.columns_for(event.size().width()))
