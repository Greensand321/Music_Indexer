"""The scroll surface of every workspace, and the legacy gradient helpers.

``GradientWidget`` is the inner widget of ``WorkspaceBase``'s scroll area.
It is **transparent**: the window-level backdrop (``gui/widgets/backdrop.py``:
album art, a gradient or a plain fill, chosen in Settings → Appearance)
shows through it and through the glass cards on it. Its one job left is to
paint the *shadow* of the cards registered with ``add_elevated()``: a card
cannot paint outside its own rect, the surface beneath it can. Shadows are
painted, never a ``QGraphicsDropShadowEffect`` (see
``gui.themes.glass.paint_shadow``).

The background setting is read once and cached (``paintEvent`` must never do
file I/O, DS-03 / SHELL-14): ``_gradient_enabled()`` is now "the backdrop is
not Plain" (``gui.themes.appearance``), still honouring the old
``bg_gradient_enabled`` key. ``refresh_gradient_setting()`` (called after
Settings saves) re-reads it and tells every backdrop to rebuild.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager


def _gradient_enabled() -> bool:
    """Cached: True unless the backdrop is "Plain" (safe to call from paint)."""
    from gui.themes import appearance
    return appearance.backdrop_mode() != "plain"


def invalidate_gradient_setting() -> None:
    """Forget the cached setting; the next read loads it once."""
    from gui.themes import appearance
    appearance.invalidate()


def refresh_gradient_setting() -> bool:
    """Re-read the background setting now, rebuild every backdrop and repaint
    every window. Call after the settings are saved. Returns True unless the
    backdrop is "Plain"."""
    from gui.themes import appearance
    appearance.refresh()
    value = _gradient_enabled()
    app = QtWidgets.QApplication.instance()
    if app is not None:
        for w in app.topLevelWidgets():
            for child in [w, *w.findChildren(QtWidgets.QWidget)]:
                rebuild = getattr(child, "on_appearance_changed", None)
                if callable(rebuild):
                    rebuild()
                child.update()
    return value


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

    The classic themes' "Gradient" backdrop: an accent glow from the top-right
    corner and an ``accent_hover`` glow from the bottom-left, mapped from
    *window* space so separate widgets read as one canvas.
    """
    win = widget.window()
    ww = float(win.width())
    wh = float(win.height())
    radius = max(ww, wh) * radius_scale

    tr_local = widget.mapFrom(win, QtCore.QPoint(int(ww), 0))
    bl_local = widget.mapFrom(win, QtCore.QPoint(0, int(wh)))
    tr_x, tr_y = float(tr_local.x()), float(tr_local.y())
    bl_x, bl_y = float(bl_local.x()), float(bl_local.y())

    rect = widget.rect()

    g1 = QtGui.QRadialGradient(tr_x, tr_y, radius * 0.90)
    c1_on = QtGui.QColor(tokens.accent)   # type: ignore[attr-defined]
    c1_on.setAlpha(alpha_tr)
    c1_off = QtGui.QColor(tokens.accent)  # type: ignore[attr-defined]
    c1_off.setAlpha(0)
    g1.setColorAt(0.0, c1_on)
    g1.setColorAt(1.0, c1_off)
    painter.fillRect(rect, QtGui.QBrush(g1))

    g2 = QtGui.QRadialGradient(bl_x, bl_y, radius * 0.72)
    c2_on = QtGui.QColor(tokens.accent_hover)   # type: ignore[attr-defined]
    c2_on.setAlpha(alpha_bl)
    c2_off = QtGui.QColor(tokens.accent_hover)  # type: ignore[attr-defined]
    c2_off.setAlpha(0)
    g2.setColorAt(0.0, c2_on)
    g2.setColorAt(1.0, c2_off)
    painter.fillRect(rect, QtGui.QBrush(g2))


class GradientWidget(QtWidgets.QWidget):
    """Transparent scroll surface that paints the shadows of its cards."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tokens = get_manager().current
        self._elevated: list[QtWidgets.QWidget] = []
        self.setAutoFillBackground(False)
        get_manager().theme_changed.connect(self._on_theme_changed)

    # ── Elevation ─────────────────────────────────────────────────────────

    def add_elevated(self, widget: QtWidgets.QWidget) -> None:
        """Paint a soft card shadow beneath ``widget`` (a descendant)."""
        if widget in self._elevated:
            return
        self._elevated.append(widget)
        widget.installEventFilter(self)

    def elevated_widgets(self) -> list[QtWidgets.QWidget]:
        self._elevated = [w for w in self._elevated if _alive(w)]
        return list(self._elevated)

    def _shadow_rect(self, widget: QtWidgets.QWidget) -> QtCore.QRect | None:
        if not widget.isVisibleTo(self):
            return None
        top_left = widget.mapTo(self, QtCore.QPoint(0, 0))
        return QtCore.QRect(top_left, widget.size())

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        if event.type() in _GEOMETRY_EVENTS and isinstance(obj, QtWidgets.QWidget):
            # The shadow extends outside the card, so the area the card
            # leaves or enters must be repainted here, not only by the card.
            self.update()
        return False

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_theme_changed(self, tokens: object) -> None:
        self._tokens = tokens  # type: ignore[assignment]
        self.update()

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        if not self._elevated:
            return
        from gui.themes.effects import elevation_margin, paint_elevation
        from gui.widgets.glass import card_radius

        t = self._tokens
        p = QtGui.QPainter(self)
        try:
            dirty = event.rect().adjusted(*elevation_margin(t))
            for card in self.elevated_widgets():
                r = self._shadow_rect(card)
                if r is not None and r.intersects(dirty):
                    paint_elevation(p, r, t, radius=card_radius(card))
        finally:
            p.end()


_GEOMETRY_EVENTS = {
    QtCore.QEvent.Type.Move, QtCore.QEvent.Type.Resize,
    QtCore.QEvent.Type.Show, QtCore.QEvent.Type.Hide,
}


def _alive(obj: QtCore.QObject) -> bool:
    try:
        import shiboken6
        return shiboken6.isValid(obj)
    except Exception:  # pragma: no cover - non-PySide binding
        return True
