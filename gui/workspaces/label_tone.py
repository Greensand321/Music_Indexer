"""Theme-aware text colour ("tone") for status labels, without inline stylesheets.

Pages used to colour status text with an inline ``color:`` stylesheet and a
hard-coded hex value (a near-white slate, for instance) that is unreadable on light themes (DS-14 / TOOL-16) and goes
stale after a theme switch. Colour flashes restyled the label on every
animation frame and overlapping flashes fought each other (DS-20).

``set_tone(label, "success")`` stores the *role* on the widget and paints it
through the widget palette from the current theme tokens; ``refresh_tones``
re-applies every tone under a widget after a theme change. ``flash_tone``
animates from a colour to the label's tone with exactly one, reused animation
per label, touching only the palette (no stylesheet re-parse per frame).
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager

_PROP = "alphadexTone"
_FLASH_NAME = "alphadexToneFlash"

# tone -> ThemeTokens attribute
_TONE_TOKEN = {
    "primary": "text_primary",
    "secondary": "text_secondary",
    "muted": "text_muted",
    "success": "success",
    "warning": "warning",
    "danger": "danger",
}


def tone_color(tone: str, tokens=None) -> QtGui.QColor:
    tokens = tokens or get_manager().current
    return QtGui.QColor(getattr(tokens, _TONE_TOKEN.get(tone, "text_primary")))


def _paint(widget: QtWidgets.QWidget, color: QtGui.QColor) -> None:
    pal = widget.palette()
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, color)
    pal.setColor(QtGui.QPalette.ColorRole.Text, color)
    widget.setPalette(pal)


def set_tone(widget: QtWidgets.QWidget, tone: str, tokens=None) -> None:
    """Colour ``widget``'s text with a theme role and remember the role."""
    if tone not in _TONE_TOKEN:
        raise ValueError(f"unknown tone {tone!r}")
    anim = widget.findChild(QtCore.QVariantAnimation, _FLASH_NAME)
    if anim is not None:
        anim.stop()
    widget.setProperty(_PROP, tone)
    _paint(widget, tone_color(tone, tokens))


def tone_of(widget: QtWidgets.QWidget) -> str | None:
    value = widget.property(_PROP)
    return str(value) if value else None


def refresh_tones(root: QtWidgets.QWidget, tokens=None) -> None:
    """Re-apply the stored tone of ``root`` and every descendant (theme change)."""
    widgets = [root, *root.findChildren(QtWidgets.QWidget)]
    for w in widgets:
        tone = tone_of(w)
        if tone:
            anim = w.findChild(QtCore.QVariantAnimation, _FLASH_NAME)
            if anim is not None and anim.state() == QtCore.QAbstractAnimation.State.Running:
                anim.setEndValue(tone_color(tone, tokens))
                continue
            _paint(w, tone_color(tone, tokens))


def flash_tone(
    widget: QtWidgets.QWidget,
    from_tone: str,
    settle_tone: str = "secondary",
    duration: int = 1600,
) -> QtCore.QVariantAnimation:
    """Fade ``widget``'s text from ``from_tone`` to ``settle_tone``.

    One animation per widget, reused: a new flash restarts it instead of
    starting a second one that would fight the first.
    """
    anim = widget.findChild(QtCore.QVariantAnimation, _FLASH_NAME)
    if anim is None:
        anim = QtCore.QVariantAnimation(widget)
        anim.setObjectName(_FLASH_NAME)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(lambda c, w=widget: _paint(w, QtGui.QColor(c)))
    anim.stop()
    widget.setProperty(_PROP, settle_tone)
    anim.setDuration(duration)
    anim.setStartValue(tone_color(from_tone))
    anim.setEndValue(tone_color(settle_tone))
    anim.start()
    return anim
