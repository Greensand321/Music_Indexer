"""Subtle gradient background for workspace panels.

``GradientWidget`` replaces the plain ``QWidget`` used as the scroll-area
inner widget in ``WorkspaceBase``.  It paints two small radial glows anchored
to opposite corners using the theme's accent colours, giving each theme a
distinct atmospheric tint without competing with the cards laid on top.

The effect is controlled by the ``bg_gradient_enabled`` key in
``~/.soundvault_config.json`` (defaults to ``True``).  Set it to ``false``
to revert to a flat ``content_bg`` fill.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager


def _gradient_enabled() -> bool:
    try:
        from config import load_config
        return bool(load_config().get("bg_gradient_enabled", True))
    except Exception:
        return True


class GradientWidget(QtWidgets.QWidget):
    """Inner scroll widget that paints a theme-aware radial gradient background.

    Two radial glows are drawn:
    - Top-right corner  → ``accent``  colour at low alpha (~9 % dark / ~6 % light)
    - Bottom-left corner → ``accent_hover`` colour at slightly lower alpha

    Both glows fade to transparent, so the middle of the workspace stays clean
    and the cards remain the focal point.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tokens = get_manager().current
        get_manager().theme_changed.connect(self._on_theme_changed)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_theme_changed(self, tokens: object) -> None:
        self._tokens = tokens  # type: ignore[assignment]
        self.update()

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = self._tokens
        p = QtGui.QPainter(self)
        rect = self.rect()
        w = float(rect.width())
        h = float(rect.height())

        # Solid base — always painted so the widget is never transparent.
        p.fillRect(rect, QtGui.QColor(t.content_bg))

        if _gradient_enabled() and w > 0 and h > 0:
            radius = max(w, h) * 0.85

            # ── Top-right glow (accent) ────────────────────────────────────
            alpha_tr = 22 if t.variant == "dark" else 14
            g1 = QtGui.QRadialGradient(w, 0.0, radius * 0.90)
            c1_on  = QtGui.QColor(t.accent);  c1_on.setAlpha(alpha_tr)
            c1_off = QtGui.QColor(t.accent);  c1_off.setAlpha(0)
            g1.setColorAt(0.0, c1_on)
            g1.setColorAt(1.0, c1_off)
            p.fillRect(rect, QtGui.QBrush(g1))

            # ── Bottom-left glow (accent_hover) ───────────────────────────
            alpha_bl = 15 if t.variant == "dark" else 9
            g2 = QtGui.QRadialGradient(0.0, h, radius * 0.72)
            c2_on  = QtGui.QColor(t.accent_hover);  c2_on.setAlpha(alpha_bl)
            c2_off = QtGui.QColor(t.accent_hover);  c2_off.setAlpha(0)
            g2.setColorAt(0.0, c2_on)
            g2.setColorAt(1.0, c2_off)
            p.fillRect(rect, QtGui.QBrush(g2))

        p.end()
