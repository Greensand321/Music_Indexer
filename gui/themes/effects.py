"""Drop-shadow and colour utility helpers."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.tokens import ThemeTokens


# ── Radius constants ──────────────────────────────────────────────────────────

class R:
    """Shared border-radius values used everywhere in the style."""
    button   = 8
    input    = 7
    card     = 12
    tab      = 6
    checkbox = 4
    radio    = 8    # half of 16px → full circle
    scroll   = 4
    tooltip  = 6
    swatch   = 10
    nav_item = 8


# ── Elevation (painted card shadows) ─────────────────────────────────────────
#
# Cards get their shadow *painted* by the surface beneath them
# (``GradientWidget``) from a cached 9-slice pixmap, instead of a
# ``QGraphicsDropShadowEffect``:
#
# * A graphics effect re-renders the whole card off-screen and blurs it every
#   time *any* child repaints: every hover frame, progress tick and caret
#   blink inside the card.
# * A widget has exactly one graphics-effect slot. Fades
#   (``gui.themes.motion.fade``) need that slot; a shadow effect would block
#   them, or be silently replaced by them.
# * The effect also never showed: passed as a temporary, PySide drops it
#   (DS-02). Keeping a reference would fix that, but not the two problems
#   above, so painting it is.

class Elevation:
    """Shadow geometry for level-1 elevation (cards)."""
    sigma = 5.0      # gaussian blur, px
    extent = 15      # how far the shadow reaches beyond the card, px (3 sigma)
    offset_y = 3     # shadow drops slightly below the card, px
    radius = R.card


def elevation_margin() -> tuple[int, int, int, int]:
    """``QRect.adjusted`` margins covering the shadow around a card."""
    e = Elevation.extent
    return (-e, -e, e, e + Elevation.offset_y)


def parse_rgba(value: str, fallback: tuple[int, int, int, int] = (0, 0, 0, 60)) -> QtGui.QColor:
    """``"rgba(r,g,b,a)"`` (a in 0..1) or any QColor name -> QColor."""
    try:
        s = value.strip()
        if s.startswith("rgba("):
            parts = s[5:-1].split(",")
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            a = int(round(float(parts[3].strip()) * 255))
            return QtGui.QColor(r, g, b, a)
        c = QtGui.QColor(s)
        if c.isValid():
            return c
    except Exception:
        pass
    return QtGui.QColor(*fallback)


_SHADOW_CACHE: dict[tuple, QtGui.QPixmap] = {}


def _shadow_pixmap(color: QtGui.QColor, radius: int) -> QtGui.QPixmap:
    """9-slice source: a blurred rounded square with a 1 px stretchable middle."""
    key = (color.rgba(), radius, Elevation.sigma, Elevation.extent)
    pm = _SHADOW_CACHE.get(key)
    if pm is not None:
        return pm
    import math

    e, r, sigma = Elevation.extent, radius, Elevation.sigma
    corner = e + r
    size = 2 * corner + 1
    half = r + 0.5                       # body is (2r+1) px square, centred
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    base_a = color.alphaF()
    centre = size / 2.0
    root2 = math.sqrt(2.0) * sigma
    for y in range(size):
        qy = abs(y + 0.5 - centre) - half + r
        for x in range(size):
            qx = abs(x + 0.5 - centre) - half + r
            outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
            d = outside + min(max(qx, qy), 0.0) - r      # signed distance
            a = base_a * 0.5 * math.erfc(d / root2)      # gaussian-blurred edge
            if a > 0.002:
                c = QtGui.QColor(color)
                c.setAlphaF(min(1.0, a))
                img.setPixelColor(x, y, c)
    pm = QtGui.QPixmap.fromImage(img)
    _SHADOW_CACHE[key] = pm
    return pm


def paint_elevation(painter: QtGui.QPainter, rect: QtCore.QRect, tokens: ThemeTokens,
                    radius: int | None = None) -> None:
    """Paint a soft card shadow for a card occupying ``rect`` (painter coords).

    Call it *before* the card itself is painted (i.e. from the parent
    surface's paintEvent). Only the ring around the card is drawn.
    """
    r = Elevation.radius if radius is None else radius
    if rect.width() < 2 * r + 2 or rect.height() < 2 * r + 2:
        return
    pm = _shadow_pixmap(parse_rgba(tokens.card_shadow), r)
    c = Elevation.extent + r
    ox = rect.x() - Elevation.extent
    oy = rect.y() - Elevation.extent + Elevation.offset_y
    ow = rect.width() + 2 * Elevation.extent
    oh = rect.height() + 2 * Elevation.extent
    mid_w, mid_h = ow - 2 * c, oh - 2 * c
    QRF = QtCore.QRectF
    slices = (
        (QRF(ox, oy, c, c),                     QRF(0, 0, c, c)),
        (QRF(ox + c, oy, mid_w, c),             QRF(c, 0, 1, c)),
        (QRF(ox + ow - c, oy, c, c),            QRF(c + 1, 0, c, c)),
        (QRF(ox, oy + c, c, mid_h),             QRF(0, c, c, 1)),
        (QRF(ox + ow - c, oy + c, c, mid_h),    QRF(c + 1, c, c, 1)),
        (QRF(ox, oy + oh - c, c, c),            QRF(0, c + 1, c, c)),
        (QRF(ox + c, oy + oh - c, mid_w, c),    QRF(c, c + 1, 1, c)),
        (QRF(ox + ow - c, oy + oh - c, c, c),   QRF(c + 1, c + 1, c, c)),
    )
    for target, source in slices:
        painter.drawPixmap(target, pm, source)


def card_shadow(tokens: ThemeTokens,
                parent: QtWidgets.QWidget | None = None) -> QtWidgets.QGraphicsDropShadowEffect:
    """Return a drop-shadow *effect* tuned to the theme (legacy helper).

    Prefer painted elevation (``paint_elevation`` / ``GradientWidget.add_elevated``).
    If you do use this, pass ``parent``: PySide drops an unparented effect
    passed straight to ``setGraphicsEffect`` and the widget gets nothing.
    """
    fx = QtWidgets.QGraphicsDropShadowEffect(parent)
    fx.setBlurRadius(18)
    fx.setOffset(0, 3)
    fx.setColor(parse_rgba(tokens.card_shadow))
    return fx


# ── Colour math ───────────────────────────────────────────────────────────────

def lerp_color(hex_a: str, hex_b: str, t: float) -> QtGui.QColor:
    """Linear interpolation between two hex colours.  t=0 → a, t=1 → b."""
    a = QtGui.QColor(hex_a)
    b = QtGui.QColor(hex_b)
    t = max(0.0, min(1.0, t))
    return QtGui.QColor(
        int(a.red()   + (b.red()   - a.red())   * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue()  + (b.blue()  - a.blue())  * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


def with_alpha(hex_color: str, alpha: int) -> QtGui.QColor:
    """Return a QColor with the given alpha (0-255)."""
    c = QtGui.QColor(hex_color)
    c.setAlpha(alpha)
    return c


def build_palette(tokens: ThemeTokens) -> QtGui.QPalette:
    """Build a QPalette from the token set for selection / window colours."""
    p = QtGui.QPalette()
    C = QtGui.QColor

    def s(role: QtGui.QPalette.ColorRole, hex_col: str) -> None:
        c = C(hex_col)
        p.setColor(role, c)
        p.setColor(QtGui.QPalette.ColorGroup.Disabled, role,
                   lerp_color(hex_col, tokens.content_bg, 0.55))

    R = QtGui.QPalette.ColorRole
    s(R.Window,          tokens.content_bg)
    s(R.WindowText,      tokens.text_primary)
    s(R.Base,            tokens.input_bg)
    s(R.AlternateBase,   tokens.card_bg)
    s(R.Text,            tokens.text_primary)
    s(R.BrightText,      tokens.text_inverse)
    s(R.Button,          tokens.card_bg)
    s(R.ButtonText,      tokens.text_primary)
    s(R.Highlight,       tokens.accent)
    s(R.HighlightedText, tokens.text_inverse)
    s(R.Link,            tokens.accent)
    s(R.LinkVisited,     tokens.accent_hover)
    s(R.ToolTipBase,     tokens.log_bg)
    s(R.ToolTipText,     tokens.log_text)
    s(R.PlaceholderText, tokens.text_muted)
    # The 3-D roles carry the remaining tokens the static stylesheet needs
    # (manager._static_qss reads them with palette(mid) etc.). AlphaDEXStyle
    # never paints from them; Fusion fallbacks get themed lines instead of
    # the system's light-grey defaults.
    s(R.Mid,             tokens.card_border)
    s(R.Dark,            tokens.input_border)
    s(R.Midlight,        tokens.sidebar_hover)
    s(R.Shadow,          tokens.text_secondary)
    return p
