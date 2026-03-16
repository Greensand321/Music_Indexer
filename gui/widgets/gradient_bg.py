"""Subtle gradient background for workspace panels.

``GradientWidget`` replaces the plain ``QWidget`` used as the scroll-area
inner widget in ``WorkspaceBase``.

Both ``GradientWidget`` and ``Sidebar`` independently compute the same two
radial glows using **window-relative** coordinates — the origins live at the
true top-right and bottom-left corners of the top-level window.  Because both
widgets reference the same two origin points, the gradient reads as a single
canvas that spans the full window behind every panel.

The effect is controlled by the ``bg_gradient_enabled`` key in
``~/.soundvault_config.json`` (defaults to ``True``).
"""
from __future__ import annotations

import math

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager


def _gradient_enabled() -> bool:
    try:
        from config import load_config
        return bool(load_config().get("bg_gradient_enabled", True))
    except Exception:
        return True


def paint_window_gradient(
    painter: QtGui.QPainter,
    widget: QtWidgets.QWidget,
    tokens: object,
    *,
    radius_scale: float = 0.85,
    alpha_tr: int | None = None,
    alpha_bl: int | None = None,
) -> None:
    """Paint the two radial glows using window-relative origin points.

    The top-right glow uses ``tokens.accent``, the bottom-left uses
    ``tokens.accent_hover``.  Both origins are expressed in the *window*'s
    coordinate system, then transformed into the caller widget's local coords
    via ``mapTo``.  This makes the gradient appear continuous across all panels
    that call this function.

    Args:
        painter:       Active QPainter for *widget*.
        widget:        The widget being painted.
        tokens:        Current ``ThemeTokens`` instance.
        radius_scale:  Multiplier on ``max(win_w, win_h)`` for the glow radii.
                       Use a value > 1 (e.g. 1.1) to make glows reach panels
                       far from the origin corner (e.g. the sidebar).
        alpha_tr:      Override the top-right glow alpha (0-255).  Defaults to
                       22 (dark) / 14 (light).
        alpha_bl:      Override the bottom-left glow alpha.  Defaults to
                       15 (dark) / 9 (light).
    """
    t = tokens  # type: ignore[assignment]
    rect = widget.rect()

    win = widget.window()
    offset = widget.mapTo(win, QtCore.QPoint(0, 0))
    win_w = float(win.width())
    win_h = float(win.height())
    diagonal = math.hypot(win_w, win_h)
    radius   = diagonal * radius_scale

    # ── Top-right glow (accent) ────────────────────────────────────────────
    # Origin in window space: (win_w, 0).  In local space: subtract offset.
    tr_x = win_w - float(offset.x())
    tr_y =         -float(offset.y())

    _atr = alpha_tr if alpha_tr is not None else (22 if t.variant == "dark" else 14)
    g1 = QtGui.QRadialGradient(tr_x, tr_y, radius * 0.90)
    c1_hi = QtGui.QColor(t.accent); c1_hi.setAlpha(_atr)
    c1_lo = QtGui.QColor(t.accent); c1_lo.setAlpha(0)
    g1.setColorAt(0.0, c1_hi)
    g1.setColorAt(1.0, c1_lo)
    painter.fillRect(rect, QtGui.QBrush(g1))

    # ── Bottom-left glow (accent_hover) ───────────────────────────────────
    # Origin in window space: (0, win_h).
    bl_x = -float(offset.x())
    bl_y = win_h - float(offset.y())

    _abl = alpha_bl if alpha_bl is not None else (15 if t.variant == "dark" else 9)
    g2 = QtGui.QRadialGradient(bl_x, bl_y, radius * 0.72)
    c2_hi = QtGui.QColor(t.accent_hover); c2_hi.setAlpha(_abl)
    c2_lo = QtGui.QColor(t.accent_hover); c2_lo.setAlpha(0)
    g2.setColorAt(0.0, c2_hi)
    g2.setColorAt(1.0, c2_lo)
    painter.fillRect(rect, QtGui.QBrush(g2))


class GradientWidget(QtWidgets.QWidget):
    """Inner scroll widget that paints a theme-aware, window-spanning gradient.

    Gradient origins are expressed in window coordinates so the glow appears
    continuous with the sidebar's gradient — together they look like a single
    background canvas behind the whole application.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tokens = get_manager().current
        get_manager().theme_changed.connect(self._on_theme_changed)

    def _on_theme_changed(self, tokens: object) -> None:
        self._tokens = tokens  # type: ignore[assignment]
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = self._tokens
        p = QtGui.QPainter(self)
        rect = self.rect()

        # Solid base — always opaque so the workspace is never see-through.
        p.fillRect(rect, QtGui.QColor(t.content_bg))

        if _gradient_enabled() and rect.width() > 0 and rect.height() > 0:
            paint_window_gradient(p, self, t)

        p.end()
