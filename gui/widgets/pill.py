"""Small rounded status pills, painted from the active theme's tokens.

``ExperimentalPill`` marks a prototype (Fuzzy finder, Scan engine, Bucketing
on the Duplicates page): it is offered so the owner can see where it would
live, but it is not ready to test.

Colours come only from tokens (no literals): the fill is the tone colour
blended over ``card_bg``, the border is the tone colour itself and the text
is ``text_primary``, so the label stays readable in every theme (checked by
``tests/test_design_contrast.py``).
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager

# How much of the tone colour goes into the fill (over card_bg).
FILL_STRENGTH = 0.18


def _blend(fg: str, bg: str, amount: float) -> str:
    """``fg`` laid over ``bg`` at ``amount`` (0..1), as an opaque hex colour."""
    a = QtGui.QColor(fg)
    b = QtGui.QColor(bg)
    mix = [round(cb + (ca - cb) * amount) for ca, cb in
           ((a.red(), b.red()), (a.green(), b.green()), (a.blue(), b.blue()))]
    return QtGui.QColor(*mix).name()


def pill_colours(tokens, tone: str = "warning") -> tuple[str, str, str]:
    """``(fill, border, text)`` hex colours of a pill in ``tone``."""
    accent = getattr(tokens, tone)
    return _blend(accent, tokens.card_bg, FILL_STRENGTH), accent, tokens.text_primary


class Pill(QtWidgets.QWidget):
    """A compact, rounded label (e.g. "Experimental")."""

    _PAD_Y = 2

    def __init__(
        self, text: str, tone: str = "warning", tooltip: str = "",
        parent: QtWidgets.QWidget | None = None, *, clickthrough: bool = False,
        compact: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("statusPill")
        self._text = text
        self._tone = tone
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

    def _repaint_on_theme(self, _tokens=None) -> None:
        self.update()

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
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
