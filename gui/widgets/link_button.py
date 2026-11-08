"""A text link in the accent colour, as a real button.

The mockup's "Open" links (Home's Recent runs) and the Help page's document
links are text in the accent colour (hover: the lighter accent, underlined).
``QPushButton`` cannot show that: the proxy style always paints its label in
the text colour. ``LinkButton`` paints itself from the active theme's tokens
at paint time (a theme, accent or transparency change repaints it through
the palette change), so it needs no theme connection.

It is a ``QAbstractButton``: ``clicked``, ``click()``, keyboard focus (Space
activates it; the focus ring is the accent), an accessible name and a
tooltip work as for any button. An optional leading glyph comes from
``gui.widgets.glyphs``. Long text elides instead of widening its row.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets

_GLYPH = 14
_GLYPH_GAP = 6
_PAD_X = 4
_PAD_Y = 4


class LinkButton(QtWidgets.QAbstractButton):
    def __init__(self, text: str = "", glyph: str = "", parent: QtWidgets.QWidget | None = None,
                 *, role: str = "label", weight: int = 600) -> None:
        super().__init__(parent)
        self._glyph = glyph
        self._role = role
        self._weight = weight
        self.setText(text)
        self.setAccessibleName(text)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)

    # ── Geometry ──────────────────────────────────────────────────────────
    def _font(self) -> QtGui.QFont:
        from gui.themes.glass import font
        return font(self._role, weight=self._weight)

    def _lead(self) -> int:
        return _GLYPH + _GLYPH_GAP if self._glyph else 0

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        fm = QtGui.QFontMetrics(self._font())
        w = fm.horizontalAdvance(self.text()) + self._lead() + 2 * _PAD_X + 1
        h = max(fm.height(), _GLYPH) + 2 * _PAD_Y
        return QtCore.QSize(w, h)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        hint = self.sizeHint()
        return QtCore.QSize(min(hint.width(), self._lead() + 2 * _PAD_X + 40), hint.height())

    # ── Painting ──────────────────────────────────────────────────────────
    def colour(self) -> QtGui.QColor:
        """The colour the link paints in now (accent; lighter when hovered)."""
        from gui.themes.glass import qcolor
        from gui.themes.manager import get_manager

        t = get_manager().current
        if not self.isEnabled():
            return qcolor(t.text_muted)
        hot = self.isDown() or self.underMouse()
        return qcolor(t.accent_hover if hot else t.accent)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.widgets.glyphs import paint_glyph

        colour = self.colour()
        f = self._font()
        if self.isEnabled() and (self.isDown() or self.underMouse()):
            f.setUnderline(True)
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(self.rect())
            if self.hasFocus():
                ring = QtGui.QColor(colour)
                ring.setAlphaF(0.6)
                p.setPen(QtGui.QPen(ring, 1.5))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(r.adjusted(0.75, 0.75, -0.75, -0.75), 8, 8)
            x = r.left() + _PAD_X
            if self._glyph:
                paint_glyph(p, QtCore.QRectF(x, r.center().y() - _GLYPH / 2, _GLYPH, _GLYPH),
                            self._glyph, colour)
                x += _GLYPH + _GLYPH_GAP
            p.setFont(f)
            p.setPen(colour)
            text_rect = QtCore.QRectF(x, r.top(), max(0.0, r.right() - _PAD_X - x), r.height())
            fm = QtGui.QFontMetrics(f)
            text = fm.elidedText(self.text(), QtCore.Qt.TextElideMode.ElideRight,
                                 int(text_rect.width()))
            p.drawText(text_rect, int(QtCore.Qt.AlignmentFlag.AlignLeft
                                      | QtCore.Qt.AlignmentFlag.AlignVCenter), text)
        finally:
            p.end()

    def enterEvent(self, event) -> None:  # noqa: N802
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        super().leaveEvent(event)
        self.update()
