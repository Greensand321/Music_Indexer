"""Line glyphs for buttons and labels, painted (no image assets, no emoji).

One small icon set for every page, drawn on a 24-unit grid with round caps
like the shell's search magnifier, so buttons stop carrying emoji
("🔍", "📄", "▶", "💾") that render differently on every platform.

    from gui.widgets.glyphs import glyph_icon, set_glyph
    set_glyph(button, "folder")          # icon follows the theme from now on
    icon = glyph_icon("play", tokens.text_inverse)  # a one-off icon in a fixed colour

``set_glyph`` keeps the icon in the theme's text colour: buttons are held in
a WeakSet and re-tinted by one module-level slot on ``theme_changed`` (no
per-button connection to the ThemeManager singleton).

Names: see ``GLYPHS``.
"""
from __future__ import annotations

import weakref

from gui.compat import QtCore, QtGui, QtWidgets

Point = QtCore.QPointF


def _p(x: float, y: float) -> QtCore.QPointF:
    return Point(x, y)


def _poly(path: QtGui.QPainterPath, pts: list[tuple[float, float]], close: bool = False) -> None:
    path.moveTo(*pts[0])
    for x, y in pts[1:]:
        path.lineTo(x, y)
    if close:
        path.closeSubpath()


# Each painter draws on a 24×24 grid. ``stroke`` paths are drawn with the
# pen; ``fill`` paths are filled (play/pause/prev/next are solid).
def _search(s, f):
    s.addEllipse(_p(11, 11), 7, 7)
    _poly(s, [(16.5, 16.5), (20, 20)])


def _play(s, f):
    _poly(f, [(8, 5), (19, 12), (8, 19)], close=True)


def _pause(s, f):
    f.addRoundedRect(QtCore.QRectF(6.5, 5, 4, 14), 1.2, 1.2)
    f.addRoundedRect(QtCore.QRectF(13.5, 5, 4, 14), 1.2, 1.2)


def _prev(s, f):
    _poly(f, [(18, 6), (9, 12), (18, 18)], close=True)
    f.addRoundedRect(QtCore.QRectF(6, 6, 2.6, 12), 1, 1)


def _next(s, f):
    _poly(f, [(6, 6), (15, 12), (6, 18)], close=True)
    f.addRoundedRect(QtCore.QRectF(15.4, 6, 2.6, 12), 1, 1)


def _stop(s, f):
    f.addRoundedRect(QtCore.QRectF(6.5, 6.5, 11, 11), 2, 2)


def _shuffle(s, f):
    _poly(s, [(4, 7), (8, 7), (16, 17), (20, 17)])
    _poly(s, [(4, 17), (8, 17), (16, 7), (20, 7)])
    _poly(s, [(17.5, 4.5), (20, 7), (17.5, 9.5)])
    _poly(s, [(17.5, 14.5), (20, 17), (17.5, 19.5)])


def _repeat(s, f):
    _poly(s, [(5, 11), (5, 8), (18, 8)])
    _poly(s, [(15.5, 5.5), (18, 8), (15.5, 10.5)])
    _poly(s, [(19, 13), (19, 16), (6, 16)])
    _poly(s, [(8.5, 13.5), (6, 16), (8.5, 18.5)])


def _folder(s, f):
    _poly(s, [(3.5, 7), (3.5, 18.5), (20.5, 18.5), (20.5, 8.5), (11.5, 8.5),
              (9.5, 5.5), (3.5, 5.5), (3.5, 7)])


def _file(s, f):
    _poly(s, [(6, 3.5), (14, 3.5), (18.5, 8), (18.5, 20.5), (6, 20.5)], close=True)
    _poly(s, [(13.5, 3.5), (13.5, 8.5), (18.5, 8.5)])


def _refresh(s, f):
    s.arcMoveTo(QtCore.QRectF(5, 5, 14, 14), 60)
    s.arcTo(QtCore.QRectF(5, 5, 14, 14), 60, 280)
    _poly(s, [(15.2, 3.2), (15.5, 6.9), (19, 6)])


def _close(s, f):
    _poly(s, [(6.5, 6.5), (17.5, 17.5)])
    _poly(s, [(17.5, 6.5), (6.5, 17.5)])


def _check(s, f):
    _poly(s, [(5, 12.5), (10, 17.5), (19, 7)])


def _plus(s, f):
    _poly(s, [(12, 5), (12, 19)])
    _poly(s, [(5, 12), (19, 12)])


def _minus(s, f):
    _poly(s, [(5, 12), (19, 12)])


def _warning(s, f):
    _poly(s, [(12, 4), (21, 19.5), (3, 19.5)], close=True)
    _poly(s, [(12, 10), (12, 14)])
    f.addEllipse(_p(12, 16.8), 1.1, 1.1)


def _info(s, f):
    s.addEllipse(_p(12, 12), 8.5, 8.5)
    _poly(s, [(12, 11), (12, 16.5)])
    f.addEllipse(_p(12, 7.8), 1.1, 1.1)


def _external(s, f):
    _poly(s, [(13, 4.5), (19.5, 4.5), (19.5, 11)])
    _poly(s, [(19.5, 4.5), (11, 13)])
    _poly(s, [(17, 14), (17, 19.5), (4.5, 19.5), (4.5, 7), (10, 7)])


def _save(s, f):
    _poly(s, [(4.5, 4.5), (16.5, 4.5), (19.5, 7.5), (19.5, 19.5), (4.5, 19.5)], close=True)
    _poly(s, [(8, 4.5), (8, 9), (15, 9), (15, 4.5)])
    s.addRoundedRect(QtCore.QRectF(7.5, 13, 9, 6.5), 1, 1)


def _trash(s, f):
    _poly(s, [(4, 7), (20, 7)])
    _poly(s, [(9.5, 7), (9.5, 4.5), (14.5, 4.5), (14.5, 7)])
    _poly(s, [(6, 7), (7, 20), (17, 20), (18, 7)])


def _list(s, f):
    for y in (7, 12, 17):
        _poly(s, [(9, y), (20, y)])
        f.addEllipse(_p(5, y), 1.3, 1.3)


def _music(s, f):
    _poly(s, [(9, 17.5), (9, 5.5), (19, 3.5), (19, 15.5)])
    f.addEllipse(_p(6.8, 17.5), 2.6, 2.2)
    f.addEllipse(_p(16.8, 15.5), 2.6, 2.2)


def _chevron_right(s, f):
    _poly(s, [(9, 5.5), (15.5, 12), (9, 18.5)])


def _chevron_down(s, f):
    _poly(s, [(5.5, 9), (12, 15.5), (18.5, 9)])


def _undo(s, f):
    _poly(s, [(8.5, 5), (4.5, 9), (8.5, 13)])
    s.moveTo(4.5, 9)
    s.lineTo(14, 9)
    s.cubicTo(_p(18, 9), _p(20, 11.5), _p(20, 14.25))
    s.cubicTo(_p(20, 17), _p(18, 19.5), _p(14, 19.5))
    s.lineTo(9, 19.5)


def _download(s, f):
    _poly(s, [(12, 4), (12, 15)])
    _poly(s, [(7.5, 10.5), (12, 15), (16.5, 10.5)])
    _poly(s, [(4.5, 19.5), (19.5, 19.5)])


def _keyboard(s, f):
    s.addRoundedRect(QtCore.QRectF(3, 6.5, 18, 11), 2.5, 2.5)
    for x in (7, 10.3, 13.7, 17):
        f.addEllipse(_p(x, 10), 0.9, 0.9)
    _poly(s, [(8, 14), (16, 14)])


def _move(s, f):
    # Four-way arrows: drag to pan.
    _poly(s, [(12, 3.5), (12, 20.5)])
    _poly(s, [(3.5, 12), (20.5, 12)])
    _poly(s, [(9.5, 6), (12, 3.5), (14.5, 6)])
    _poly(s, [(9.5, 18), (12, 20.5), (14.5, 18)])
    _poly(s, [(6, 9.5), (3.5, 12), (6, 14.5)])
    _poly(s, [(18, 9.5), (20.5, 12), (18, 14.5)])


def _marquee(s, f):
    # A dashed selection rectangle.
    for a, b in (((4.5, 5), (8, 5)), ((10.25, 5), (13.75, 5)), ((16, 5), (19.5, 5)),
                 ((4.5, 19), (8, 19)), ((10.25, 19), (13.75, 19)), ((16, 19), (19.5, 19)),
                 ((4.5, 8), (4.5, 10.5)), ((4.5, 13.5), (4.5, 16)),
                 ((19.5, 8), (19.5, 10.5)), ((19.5, 13.5), (19.5, 16))):
        _poly(s, [a, b])


def _lasso(s, f):
    # A freehand loop with a short tail.
    s.addEllipse(_p(12.5, 9.5), 8, 5.5)
    s.moveTo(7.5, 13.8)
    s.cubicTo(_p(6, 16), _p(8.5, 17.5), _p(8, 20.5))


def _fit(s, f):
    # Four corner brackets: fit everything in view.
    _poly(s, [(4.5, 9), (4.5, 4.5), (9, 4.5)])
    _poly(s, [(15, 4.5), (19.5, 4.5), (19.5, 9)])
    _poly(s, [(19.5, 15), (19.5, 19.5), (15, 19.5)])
    _poly(s, [(9, 19.5), (4.5, 19.5), (4.5, 15)])


GLYPHS = {
    "search": _search, "play": _play, "pause": _pause, "prev": _prev, "next": _next,
    "stop": _stop, "shuffle": _shuffle, "repeat": _repeat, "folder": _folder,
    "file": _file, "refresh": _refresh, "close": _close, "check": _check,
    "plus": _plus, "minus": _minus, "warning": _warning, "info": _info,
    "external": _external, "save": _save, "trash": _trash, "list": _list,
    "music": _music, "chevron_right": _chevron_right, "chevron_down": _chevron_down,
    "undo": _undo, "download": _download, "keyboard": _keyboard,
    "move": _move, "marquee": _marquee, "lasso": _lasso, "fit": _fit,
}


def paint_glyph(painter: QtGui.QPainter, rect: QtCore.QRectF | QtCore.QRect, name: str,
                colour: QtGui.QColor | str, *, weight: float = 1.8) -> None:
    """Paint glyph ``name`` scaled into ``rect`` (square, centred)."""
    draw = GLYPHS.get(name)
    if draw is None:
        raise KeyError(f"unknown glyph {name!r}; known: {', '.join(sorted(GLYPHS))}")
    rect = QtCore.QRectF(rect)
    side = min(rect.width(), rect.height())
    stroke, fill = QtGui.QPainterPath(), QtGui.QPainterPath()
    draw(stroke, fill)
    colour = QtGui.QColor(colour)
    painter.save()
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.translate(rect.center().x() - side / 2, rect.center().y() - side / 2)
        painter.scale(side / 24.0, side / 24.0)
        if not stroke.isEmpty():
            pen = QtGui.QPen(colour, weight, QtCore.Qt.PenStyle.SolidLine,
                             QtCore.Qt.PenCapStyle.RoundCap, QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawPath(stroke)
        if not fill.isEmpty():
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            painter.drawPath(fill)
    finally:
        painter.restore()


_ICON_CACHE: dict[tuple[str, int, int], QtGui.QIcon] = {}


def glyph_icon(name: str, colour: QtGui.QColor | str | None = None, size: int = 16) -> QtGui.QIcon:
    """A QIcon of glyph ``name`` in ``colour`` (default: the theme's text).

    Cached per (name, colour, size): a theme switch re-tints ~370 glyph
    buttons, which share a few dozen distinct icons (QIcon is implicitly
    shared, so handing out the same one is safe)."""
    if colour is None:
        from gui.themes.manager import get_manager
        colour = get_manager().current.text_primary
    key = (name, QtGui.QColor(colour).rgba(), int(size))
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return QtGui.QIcon(cached)
    if len(_ICON_CACHE) > 512:
        _ICON_CACHE.clear()
    icon = _render_icon(name, colour, size)
    _ICON_CACHE[key] = icon
    return QtGui.QIcon(icon)


def _render_icon(name: str, colour: QtGui.QColor | str, size: int) -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        pm = QtGui.QPixmap(int(size * dpr), int(size * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(pm)
        try:
            paint_glyph(p, QtCore.QRectF(0, 0, size, size), name, colour)
        finally:
            p.end()
        icon.addPixmap(pm)
    return icon


# ── Theme-following icons ─────────────────────────────────────────────────────

_FOLLOWERS: "weakref.WeakSet[QtWidgets.QAbstractButton]" = weakref.WeakSet()
_CONNECTED = False


def _retint(_tokens=None) -> None:
    for button in list(_FOLLOWERS):
        try:
            _apply(button)
        except RuntimeError:  # C++ object already deleted
            _FOLLOWERS.discard(button)


def _apply(button: QtWidgets.QAbstractButton) -> None:
    from gui.themes.manager import get_manager

    t = get_manager().current
    role = button.property("glyphRole") or "text"
    colour = {"text": t.text_primary, "muted": t.text_secondary,
              "accent": t.accent, "inverse": t.text_inverse,
              "danger": t.danger}.get(role, t.text_primary)
    size = int(button.property("glyphSize") or 16)
    name = str(button.property("glyphName"))
    key = f"{name}|{colour}|{size}"
    if button.property("glyphKey") == key:
        return            # already this icon (e.g. a theme switch that kept the colour)
    button.setIcon(glyph_icon(name, colour, size))
    button.setIconSize(QtCore.QSize(size, size))
    button.setProperty("glyphKey", key)


def set_glyph(button: QtWidgets.QAbstractButton, name: str, *, role: str = "text",
              size: int = 16) -> QtWidgets.QAbstractButton:
    """Give ``button`` glyph ``name`` and keep it tinted with the theme.

    ``role``: "text" (default), "muted", "accent", "inverse" (dark, for a
    light primary button) or "danger"."""
    global _CONNECTED
    if name not in GLYPHS:
        raise KeyError(f"unknown glyph {name!r}")
    button.setProperty("glyphName", name)
    button.setProperty("glyphRole", role)
    button.setProperty("glyphSize", size)
    button.setProperty("glyphKey", None)     # set_glyph always (re)sets the icon
    _apply(button)
    _FOLLOWERS.add(button)
    if not _CONNECTED:
        from gui.themes.manager import get_manager
        get_manager().theme_changed.connect(_retint)
        _CONNECTED = True
    return button


# ── A glyph that is not a button ──────────────────────────────────────────────

class GlyphLabel(QtWidgets.QWidget):
    """A decorative line glyph (before a path, beside a warning), tinted by
    ``role``: "muted" (default), "text", "accent", "success", "warning" or
    "danger". Transparent for the mouse; it reads the theme at paint time."""

    def __init__(self, name: str, size: int = 16, *, role: str = "muted",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        if name not in GLYPHS:
            raise KeyError(f"unknown glyph {name!r}")
        self._name = name
        self._role = role
        self.setFixedSize(size, size)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def glyph(self) -> str:
        return self._name

    def role(self) -> str:
        return self._role

    def set_glyph(self, name: str, role: str | None = None) -> None:
        if name not in GLYPHS:
            raise KeyError(f"unknown glyph {name!r}")
        self._name = name
        if role is not None:
            self._role = role
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager

        t = get_manager().current
        colour = {"text": t.text_primary, "muted": t.text_secondary, "accent": t.accent,
                  "success": t.success, "warning": t.warning, "danger": t.danger,
                  }.get(self._role, t.text_secondary)
        p = QtGui.QPainter(self)
        try:
            paint_glyph(p, QtCore.QRectF(self.rect()), self._name, colour)
        finally:
            p.end()
