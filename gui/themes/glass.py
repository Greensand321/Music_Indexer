"""The glass panel primitive: one painter routine for every surface.

Cards, the floating sidebar, header pills, the segmented control's thumb,
the Now Playing capsule and the landing card are all painted by
``paint_surface`` from a ``Surface`` of the active theme's material
(``gui/themes/tokens.py``):

1. a soft shadow (``paint_shadow``) — painted, never a graphics effect. It is
   drawn by whoever owns the area *around* the surface (the scroll surface
   for cards, the backdrop for the sidebar and the capsule), because a
   widget cannot paint outside its own rect;
2. a rounded translucent fill;
3. a 1 px border;
4. a 1 px inner top highlight that fades out along the corners (the
   ``inset 0 1px 0`` line of the mockup).

On the web the glass blurs what is behind it. Here the backdrop is already
blurred and static, so a translucent fill over it reads as glass; nothing is
blurred at paint time. Classic themes pass opaque surfaces with no highlight
through the very same calls.

Shadows come from a cached 9-slice pixmap per (colour, blur, radius), built
with numpy when available (a few ms once), so painting one is eight
``drawPixmap`` calls.

``font(role)`` builds a QFont for a role of the type ramp (``TYPE``) with the
preferred families (Segoe UI Variable first), weight and tracking.
"""
from __future__ import annotations

import math
from functools import lru_cache

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.tokens import (
    DISPLAY_FAMILIES, MONO_FAMILIES, PILL, TYPE, UI_FAMILIES, Material, Surface, parse_colour,
)

QRF = QtCore.QRectF


# ── Colours ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=512)
def _rgba_tuple(value: str) -> tuple[int, int, int, int]:
    try:
        r, g, b, a = parse_colour(value)
        return r, g, b, max(0, min(255, int(round(a * 255))))
    except (ValueError, IndexError):
        c = QtGui.QColor(value)
        return c.red(), c.green(), c.blue(), c.alpha()


def qcolor(value: str, opacity: float = 1.0) -> QtGui.QColor:
    """``"#rrggbb"`` / ``"rgba(r,g,b,a)"`` → QColor (``opacity`` scales alpha)."""
    r, g, b, a = _rgba_tuple(value)
    if opacity != 1.0:
        a = int(round(a * max(0.0, min(1.0, opacity))))
    return QtGui.QColor(r, g, b, a)


def current_material() -> Material:
    from gui.themes.manager import get_manager
    return get_manager().current.material


def surface(name: str) -> Surface:
    """The active theme's surface for an elevation ("card", "raised", …)."""
    return current_material().surface(name)


# ── Fonts ─────────────────────────────────────────────────────────────────────

_FAMILIES = {"ui": UI_FAMILIES, "display": DISPLAY_FAMILIES, "mono": MONO_FAMILIES}


def font(role: str, *, weight: int | None = None, px: int | None = None,
         tabular: bool | None = None, tracking: float | None = None) -> QtGui.QFont:
    """A QFont for a role of the type ramp: display · title · heading · body ·
    label · caption · micro · mono · stat. Keyword overrides adjust one
    property (``tracking`` in em). Built once per combination (polishing
    asks for the same few fonts thousands of times); callers get a copy."""
    return QtGui.QFont(_font(role, weight, px, tabular, tracking))


@lru_cache(maxsize=128)
def _font(role: str, weight: int | None, px: int | None, tabular: bool | None,
          tracking: float | None) -> QtGui.QFont:
    spec = TYPE[role]
    if tabular is not None or tracking is not None:
        from dataclasses import replace
        spec = replace(spec, tabular=spec.tabular if tabular is None else tabular,
                       tracking=spec.tracking if tracking is None else tracking)
    f = QtGui.QFont()
    f.setFamilies(list(_FAMILIES.get(spec.family, UI_FAMILIES)))
    size = px or spec.px
    f.setPixelSize(size)
    f.setWeight(QtGui.QFont.Weight(weight or spec.weight))
    if spec.tracking:
        f.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, spec.tracking * size)
    f.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
    if spec.tabular:
        try:
            f.setFeature(QtGui.QFont.Tag("tnum"), 1)
        except (AttributeError, TypeError):  # pragma: no cover - older Qt
            pass
    return f


def apply_font(widget: QtWidgets.QWidget, role: str, **kw) -> QtWidgets.QWidget:
    """Give ``widget`` a type-ramp role font (an explicit font: the style's
    role polishing never overrides it)."""
    widget.setFont(font(role, **kw))
    return widget


# ── Geometry ──────────────────────────────────────────────────────────────────

def clamp_radius(rect: QtCore.QRectF, radius: float) -> float:
    return max(0.0, min(float(radius), rect.height() / 2.0, rect.width() / 2.0))


def rounded_path(rect: QtCore.QRectF, radius: float) -> QtGui.QPainterPath:
    path = QtGui.QPainterPath()
    r = clamp_radius(rect, radius)
    path.addRoundedRect(rect, r, r)
    return path


# ── Shadow ────────────────────────────────────────────────────────────────────

def shadow_extent(blur: int) -> int:
    """How far a shadow reaches beyond its surface: ~3 sigma, with the CSS
    convention sigma = blur / 2."""
    return int(math.ceil(1.5 * blur)) if blur > 0 else 0


def shadow_margins(s: Surface) -> tuple[int, int, int, int]:
    """``QRect.adjusted`` margins covering the shadow of ``s`` around a rect."""
    e = shadow_extent(s.blur)
    return (-e, -e + min(s.offset_y, e), e, e + s.offset_y)


_SHADOW_CACHE: dict[tuple, QtGui.QPixmap] = {}


def _erfc(x):
    """Complementary error function (Abramowitz & Stegun 7.1.26, |err| < 2e-7)."""
    import numpy as np
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    y = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (
        0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (
            1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, y, 2.0 - y)


def _shadow_image(rgba: tuple[int, int, int, int], blur: int, radius: int) -> QtGui.QImage:
    """9-slice source: a blurred rounded square with a 1 px stretchable middle."""
    e = shadow_extent(blur)
    r = radius
    corner = e + r
    size = 2 * corner + 1
    sigma = max(0.5, blur / 2.0)
    half = r + 0.5
    centre = size / 2.0
    base_a = rgba[3] / 255.0
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core requirement
        np = None
    if np is not None:
        coords = np.arange(size, dtype=np.float64) + 0.5 - centre
        q = np.abs(coords) - half + r
        qx, qy = np.meshgrid(q, q)
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        d = outside + np.minimum(np.maximum(qx, qy), 0.0) - r
        alpha = np.clip(base_a * 0.5 * _erfc(d / (math.sqrt(2.0) * sigma)), 0.0, 1.0)
        a8 = np.round(alpha * 255.0).astype(np.uint8)
        buf = np.zeros((size, size, 4), dtype=np.uint8)
        # Premultiplied BGRA (little-endian ARGB32).
        buf[..., 0] = np.round(rgba[2] * alpha).astype(np.uint8)
        buf[..., 1] = np.round(rgba[1] * alpha).astype(np.uint8)
        buf[..., 2] = np.round(rgba[0] * alpha).astype(np.uint8)
        buf[..., 3] = a8
        out = QtGui.QImage(buf.tobytes(), size, size, size * 4,
                           QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        return out.copy()
    root2 = math.sqrt(2.0) * sigma  # pragma: no cover - fallback path
    for y in range(size):  # pragma: no cover
        qy_ = abs(y + 0.5 - centre) - half + r
        for x in range(size):
            qx_ = abs(x + 0.5 - centre) - half + r
            out_ = math.hypot(max(qx_, 0.0), max(qy_, 0.0))
            d_ = out_ + min(max(qx_, qy_), 0.0) - r
            a = base_a * 0.5 * math.erfc(d_ / root2)
            if a > 0.002:
                img.setPixelColor(x, y, QtGui.QColor(rgba[0], rgba[1], rgba[2], int(min(1.0, a) * 255)))
    return img


def _shadow_pixmap(colour: str, blur: int, radius: int) -> QtGui.QPixmap:
    rgba = _rgba_tuple(colour)
    key = (rgba, blur, radius)
    pm = _SHADOW_CACHE.get(key)
    if pm is None:
        pm = QtGui.QPixmap.fromImage(_shadow_image(rgba, blur, radius))
        if len(_SHADOW_CACHE) > 64:
            _SHADOW_CACHE.clear()
        _SHADOW_CACHE[key] = pm
    return pm


def paint_shadow(painter: QtGui.QPainter, rect: QtCore.QRectF | QtCore.QRect, s: Surface, *,
                 radius: float | None = None, strength: float = 1.0) -> None:
    """Paint the soft shadow of a surface occupying ``rect`` (painter coords).

    Call it *before* the surface is painted, from the widget that owns the
    area around it. ``strength`` (0..1) fades it (e.g. while the surface
    fades in)."""
    if not s.shadow or s.blur <= 0 or strength <= 0.0:
        return
    rect = QRF(rect)
    r = int(round(clamp_radius(rect, s.radius if radius is None else radius)))
    if rect.width() < 2 * r + 2 or rect.height() < 2 * r + 2:
        r = max(0, int(min(rect.width(), rect.height()) // 2) - 1)
    pm = _shadow_pixmap(s.shadow, s.blur, r)
    e = shadow_extent(s.blur)
    c = e + r
    ox = rect.x() - e
    oy = rect.y() - e + s.offset_y
    ow = rect.width() + 2 * e
    oh = rect.height() + 2 * e
    mid_w, mid_h = ow - 2 * c, oh - 2 * c
    if mid_w < 0 or mid_h < 0:
        return
    painter.save()
    if strength < 1.0:
        painter.setOpacity(painter.opacity() * strength)
    # Like a CSS box-shadow, never paint under the surface itself: through
    # translucent glass the ring's inner part would show as a darker band.
    clip = QtGui.QPainterPath()
    clip.setFillRule(QtCore.Qt.FillRule.OddEvenFill)
    clip.addRect(QRF(ox, min(oy, rect.y()), ow, oh + abs(s.offset_y)))
    clip.addRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), max(0.0, r - 0.5), max(0.0, r - 0.5))
    painter.setClipPath(clip, QtCore.Qt.ClipOperation.IntersectClip)
    slices = (
        (QRF(ox, oy, c, c),                   QRF(0, 0, c, c)),
        (QRF(ox + c, oy, mid_w, c),           QRF(c, 0, 1, c)),
        (QRF(ox + ow - c, oy, c, c),          QRF(c + 1, 0, c, c)),
        (QRF(ox, oy + c, c, mid_h),           QRF(0, c, c, 1)),
        (QRF(ox + ow - c, oy + c, c, mid_h),  QRF(c + 1, c, c, 1)),
        (QRF(ox, oy + oh - c, c, c),          QRF(0, c + 1, c, c)),
        (QRF(ox + c, oy + oh - c, mid_w, c),  QRF(c, c + 1, 1, c)),
        (QRF(ox + ow - c, oy + oh - c, c, c), QRF(c + 1, c + 1, c, c)),
    )
    for target, source in slices:
        if target.width() > 0 and target.height() > 0:
            painter.drawPixmap(target, pm, source)
    painter.restore()


# ── Surface ───────────────────────────────────────────────────────────────────

def _highlight(painter: QtGui.QPainter, rect: QtCore.QRectF, r: float, colour: QtGui.QColor) -> None:
    """The inner top highlight: a 1 px line just inside the border that
    follows the top corners and fades out as they turn vertical."""
    left, top, right = rect.left(), rect.top(), rect.right()
    y = top + 1.5
    rr = max(0.0, r - 1.5)
    path = QtGui.QPainterPath()
    if rr > 0.5:
        path.moveTo(left + 1.5, y + rr)
        path.arcTo(QRF(left + 1.5, y, 2 * rr, 2 * rr), 180.0, -90.0)
        path.lineTo(right - 1.5 - rr, y)
        path.arcTo(QRF(right - 1.5 - 2 * rr, y, 2 * rr, 2 * rr), 90.0, -90.0)
    else:
        path.moveTo(left + 1.5, y)
        path.lineTo(right - 1.5, y)
    width = max(1.0, right - left)
    fade = min(0.5, max(r, 1.0) / width)
    grad = QtGui.QLinearGradient(QtCore.QPointF(left, 0), QtCore.QPointF(right, 0))
    clear = QtGui.QColor(colour)
    clear.setAlpha(0)
    grad.setColorAt(0.0, clear)
    grad.setColorAt(fade, colour)
    grad.setColorAt(1.0 - fade, colour)
    grad.setColorAt(1.0, clear)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.setPen(QtGui.QPen(QtGui.QBrush(grad), 1.0))
    painter.drawPath(path)


def paint_surface(
    painter: QtGui.QPainter,
    rect: QtCore.QRectF | QtCore.QRect,
    s: Surface,
    *,
    radius: float | None = None,
    shadow: bool = False,
    fill: str | QtGui.QColor | QtGui.QBrush | None = None,
    opacity: float = 1.0,
) -> None:
    """Paint one glass (or classic) surface filling ``rect``.

    ``shadow=True`` also paints its shadow first (only when the painter may
    draw outside ``rect``). ``fill`` overrides the surface fill (hover/pressed
    states); ``opacity`` fades the whole surface."""
    rect = QRF(rect)
    if rect.width() <= 1 or rect.height() <= 1:
        return
    r = clamp_radius(rect, s.radius if radius is None else radius)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    if opacity < 1.0:
        painter.setOpacity(painter.opacity() * max(0.0, opacity))
    if shadow:
        paint_shadow(painter, rect, s, radius=r)
    body = rect.adjusted(0.5, 0.5, -0.5, -0.5)
    path = QtGui.QPainterPath()
    path.addRoundedRect(body, max(0.0, r - 0.5), max(0.0, r - 0.5))
    if isinstance(fill, QtGui.QBrush):
        brush = fill
    elif isinstance(fill, QtGui.QColor):
        brush = QtGui.QBrush(fill)
    else:
        brush = QtGui.QBrush(qcolor(fill if isinstance(fill, str) else s.fill))
    painter.fillPath(path, brush)
    if s.border:
        painter.setPen(QtGui.QPen(qcolor(s.border), 1.0))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
    if s.highlight:
        _highlight(painter, rect, r, qcolor(s.highlight))
    painter.restore()


def paint_pill(painter: QtGui.QPainter, rect: QtCore.QRectF | QtCore.QRect, fill: str,
               border: str = "") -> None:
    """A fully rounded chip (status pills, badges)."""
    paint_surface(painter, rect, Surface(fill, border, radius=PILL))


def paint_brand_tile(painter: QtGui.QPainter, rect: QtCore.QRectF | QtCore.QRect,
                     bars: tuple[int, ...] = (16, 30, 22, 12)) -> None:
    """The AlphaDEX mark: a glass tile holding accent level bars, bottom-aligned.

    64 px (landing): radius 20, bars 5 px wide, 5 apart, 15 px from the
    bottom. 34 px (sidebar): radius 11, bars 3 px, 3 apart, 8 px up."""
    from gui.themes.manager import get_manager

    t = get_manager().current
    m = t.material
    rect = QRF(rect)
    big = rect.width() >= 48
    radius = 20 if big else 11
    bar_w, gap, pad = (5.0, 5.0, 15.0) if big else (3.0, 3.0, 8.0)
    if m.is_glass:
        tile = Surface("rgba(255,255,255,0.14)" if big else "rgba(255,255,255,0.16)",
                       "rgba(255,255,255,0.30)" if big else "rgba(255,255,255,0.28)",
                       "rgba(255,255,255,0.40)" if big else "", radius=radius)
    else:
        tile = Surface(m.raised.fill, m.raised.border, radius=radius)
    paint_surface(painter, rect, tile)
    painter.save()
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(qcolor(t.accent))
    total = len(bars) * bar_w + (len(bars) - 1) * gap
    x = rect.center().x() - total / 2
    bottom = rect.bottom() - pad
    for h in bars:
        painter.drawRoundedRect(QRF(x, bottom - h, bar_w, h), bar_w / 2 + 0.5, bar_w / 2 + 0.5)
        x += bar_w + gap
    painter.restore()
