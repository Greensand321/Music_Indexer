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


def paint_window_gradient(
    painter: "QtGui.QPainter",
    widget: "QtWidgets.QWidget",
    tokens: object,
    *,
    radius_scale: float = 1.0,
    alpha_tr: int = 22,
    alpha_bl: int = 15,
) -> None:
    """Paint a window-spanning radial gradient in widget-local coordinates.

    The gradient origins are computed in *window* space then mapped into the
    widget's local coordinate space.  This makes the gradient read as one
    seamless canvas across the sidebar and the workspace, even though they are
    separate widgets with independent paintEvents.

    Parameters
    ----------
    painter:      Active QPainter on *widget*.
    widget:       The widget being painted.
    tokens:       ThemeTokens object with ``accent`` and ``accent_hover`` attrs.
    radius_scale: Scales the glow radius relative to ``max(win_w, win_h)``.
    alpha_tr:     Alpha (0-255) for the top-right accent glow.
    alpha_bl:     Alpha (0-255) for the bottom-left accent_hover glow.
    """
    win = widget.window()
    ww = float(win.width())
    wh = float(win.height())
    radius = max(ww, wh) * radius_scale

    # Map window-space corners to widget-local coordinates so the gradient
    # origin is consistent across widgets that share the same window.
    tr_local = widget.mapFrom(win, QtCore.QPoint(int(ww), 0))
    bl_local = widget.mapFrom(win, QtCore.QPoint(0, int(wh)))
    tr_x, tr_y = float(tr_local.x()), float(tr_local.y())
    bl_x, bl_y = float(bl_local.x()), float(bl_local.y())

    rect = widget.rect()

    # Top-right glow — accent colour
    g1 = QtGui.QRadialGradient(tr_x, tr_y, radius * 0.90)
    c1_on = QtGui.QColor(tokens.accent)   # type: ignore[attr-defined]
    c1_on.setAlpha(alpha_tr)
    c1_off = QtGui.QColor(tokens.accent)  # type: ignore[attr-defined]
    c1_off.setAlpha(0)
    g1.setColorAt(0.0, c1_on)
    g1.setColorAt(1.0, c1_off)
    painter.fillRect(rect, QtGui.QBrush(g1))

    # Bottom-left glow — accent_hover colour
    g2 = QtGui.QRadialGradient(bl_x, bl_y, radius * 0.72)
    c2_on = QtGui.QColor(tokens.accent_hover)   # type: ignore[attr-defined]
    c2_on.setAlpha(alpha_bl)
    c2_off = QtGui.QColor(tokens.accent_hover)  # type: ignore[attr-defined]
    c2_off.setAlpha(0)
    g2.setColorAt(0.0, c2_on)
    g2.setColorAt(1.0, c2_off)
    painter.fillRect(rect, QtGui.QBrush(g2))


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
