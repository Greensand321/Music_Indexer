"""Small rounded pills, painted from the active theme's tokens.

Two looks, one module:

``Pill`` / ``ExperimentalPill``
    A fixed label on an opaque fill: the tone colour blended over
    ``card_bg``, the tone colour as the border, ``text_primary`` text, so it
    stays readable in every theme (checked by ``tests/test_design_contrast.py``).
    ``ExperimentalPill`` marks a prototype (Fuzzy finder, Scan engine,
    Bucketing): offered so the owner can see where it would live, not ready
    to test. ``compact=True`` paints it as a small dot (section tabs at
    narrow widths); the tooltip still says what it means.

``ToneBadge`` / ``paint_tone_pill`` / ``tone_pill_size`` / ``pill_font``
    The glass pill (a translucent tone fill, a tone border, primary text;
    neutral is the material's well) whose text and tone change in place:
    verdicts, "Kept" / "Would be quarantined", "Cannot be undone", and the
    pills item delegates paint inside rows (``gui.widgets.glass_views``).
    Tones: "" (neutral), "accent", "success", "warning", "danger".
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager

# How much of the tone colour goes into the fill (over card_bg).
FILL_STRENGTH = 0.18


def _blend(fg: str, bg: str, amount: float) -> str:
    """``fg`` laid over ``bg`` at ``amount`` (0..1), as an opaque ``#rrggbb``."""
    def rgb(value: str) -> list[int]:
        h = value.lstrip("#")
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]

    mix = [round(cb + (ca - cb) * amount) for ca, cb in zip(rgb(fg), rgb(bg))]
    return "#" + "".join(f"{c:02x}" for c in mix)


def pill_colours(tokens, tone: str = "warning") -> tuple[str, str, str]:
    """``(fill, border, text)`` hex colours of a pill in ``tone``."""
    accent = getattr(tokens, tone)
    return _blend(accent, tokens.card_bg, FILL_STRENGTH), accent, tokens.text_primary


class Pill(QtWidgets.QWidget):
    """A compact, rounded label (e.g. "Experimental").

    ``set_dot(True)`` paints it as a small dot in its tone instead (where
    there is no room for the word, e.g. a section tab at a narrow width);
    the tooltip still says what it means."""

    _PAD_Y = 2
    DOT = 8

    def __init__(
        self, text: str, tone: str = "warning", tooltip: str = "",
        parent: QtWidgets.QWidget | None = None, *, clickthrough: bool = False,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("statusPill")
        self._text = text
        self._tone = tone
        self._dot = False
        # Compact: inside a section tab, where width is scarce.
        self._pad_x = 5 if compact else 7
        font = QtGui.QFont(self.font())
        font.setPointSize(7 if compact else 8)
        font.setWeight(QtGui.QFont.Weight.Bold)
        font.setLetterSpacing(QtGui.QFont.SpacingType.PercentageSpacing, 100.0 if compact else 104.0)
        self.setFont(font)
        if tooltip:
            self.setToolTip(tooltip)
        self.setAccessibleName(text)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        # Inside a tab or button the click (and the tooltip) belong to the
        # host, so the pill lets the mouse through.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, clickthrough)
        get_manager().theme_changed.connect(self._repaint_on_theme)

    def text(self) -> str:
        return self._text

    def tone(self) -> str:
        return self._tone

    def colours(self) -> tuple[str, str, str]:
        return pill_colours(get_manager().current, self._tone)

    def is_dot(self) -> bool:
        return self._dot

    def set_dot(self, on: bool) -> None:
        on = bool(on)
        if on != self._dot:
            self._dot = on
            self.updateGeometry()
            self.update()

    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        if self._dot:
            return QtCore.QSize(self.DOT + 2, self.DOT + 2)
        fm = QtGui.QFontMetrics(self.font())
        return QtCore.QSize(
            fm.horizontalAdvance(self._text) + 2 * self._pad_x + 2,
            fm.height() + 2 * self._PAD_Y + 2,
        )

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.sizeHint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        fill, border, text = self.colours()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self._dot:
            c = QtCore.QRectF(self.rect()).center()
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(border))
            p.drawEllipse(c, self.DOT / 2.0, self.DOT / 2.0)
            p.end()
            return
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2
        p.setPen(QtGui.QPen(QtGui.QColor(border), 1.0))
        p.setBrush(QtGui.QColor(fill))
        p.drawRoundedRect(rect, radius, radius)
        p.setPen(QtGui.QColor(text))
        p.setFont(self.font())
        p.drawText(self.rect(), int(QtCore.Qt.AlignmentFlag.AlignCenter), self._text)
        p.end()


EXPERIMENTAL_TIP = "A prototype, kept so you can see where it would live. Not ready to test."


class ExperimentalPill(Pill):
    """The "Experimental" marker shown next to a prototype's name."""

    def __init__(
        self, parent: QtWidgets.QWidget | None = None, *, clickthrough: bool = False,
        compact: bool = False,
    ) -> None:
        super().__init__("Experimental", "warning", EXPERIMENTAL_TIP, parent,
                         clickthrough=clickthrough, compact=compact)


# ── Glass tone pills (ToneBadge, and pills painted inside rows) ──────────────

TONES = ("", "accent", "success", "warning", "danger")

_TONE_PAD_X = 8
_TONE_PAD_Y = 2


def tone_colours(tone: str) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
    """``(fill, border, text)`` of a glass pill in ``tone`` for the active theme."""
    from gui.themes.glass import current_material, qcolor

    t = get_manager().current
    value = {"accent": t.accent, "success": t.success, "warning": t.warning,
             "danger": t.danger}.get(tone)
    if value is None:
        m = current_material()
        return qcolor(m.well.fill), qcolor(m.well.border or m.hairline), qcolor(t.text_secondary)
    return qcolor(value, 0.18), qcolor(value, 0.45), qcolor(t.text_primary)


def pill_font() -> QtGui.QFont:
    """The small pill font: caption size, semibold, no extra tracking."""
    from gui.themes.glass import font

    return font("caption", weight=600)


def tone_pill_size(text: str, font: QtGui.QFont, *, pad_x: int = _TONE_PAD_X,
                   pad_y: int = _TONE_PAD_Y) -> QtCore.QSize:
    fm = QtGui.QFontMetrics(font)
    return QtCore.QSize(fm.horizontalAdvance(text) + 2 * pad_x + 2, fm.height() + 2 * pad_y + 2)


def paint_tone_pill(painter: QtGui.QPainter, rect: QtCore.QRect | QtCore.QRectF, text: str,
                    tone: str, font: QtGui.QFont, *, opacity: float = 1.0) -> None:
    """Paint a glass pill filling ``rect`` (fully rounded), ``text`` centred
    and elided to fit."""
    fill, border, colour = tone_colours(tone)
    r = QtCore.QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5)
    radius = r.height() / 2
    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if opacity < 1.0:
            painter.setOpacity(painter.opacity() * opacity)
        painter.setPen(QtGui.QPen(border, 1.0))
        painter.setBrush(fill)
        painter.drawRoundedRect(r, radius, radius)
        painter.setPen(colour)
        painter.setFont(font)
        fm = QtGui.QFontMetrics(font)
        painter.drawText(QtCore.QRectF(rect), int(QtCore.Qt.AlignmentFlag.AlignCenter),
                         fm.elidedText(text, QtCore.Qt.TextElideMode.ElideRight,
                                       max(0, int(r.width()) - 2 * _TONE_PAD_X)))
    finally:
        painter.restore()


class ToneBadge(QtWidgets.QWidget):
    """A glass pill whose text and tone change in place."""

    def __init__(self, text: str = "", tone: str = "", *, role: str = "caption",
                 weight: int = 600, pad_x: int = _TONE_PAD_X, pad_y: int = _TONE_PAD_Y,
                 parent: QtWidgets.QWidget | None = None) -> None:
        from gui.themes.glass import font

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
