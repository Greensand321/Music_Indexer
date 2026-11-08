"""Drop-shadow and colour utility helpers."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.tokens import ThemeTokens


# ── Radius constants ──────────────────────────────────────────────────────────

class R:
    """Shared border-radius values (``tokens.RADIUS`` is the full scale)."""
    button   = 999   # pill
    input    = 999   # pill
    card     = 26
    hero     = 30
    tab      = 999
    checkbox = 6
    radio    = 9     # half of 18px → full circle
    scroll   = 4
    tooltip  = 8
    swatch   = 14
    nav_item = 14


# ── Elevation (painted card shadows) ─────────────────────────────────────────
#
# Cards get their shadow *painted* by the surface beneath them
# (``GradientWidget``), with ``gui.themes.glass.paint_shadow`` from the
# active material's ``card`` surface, instead of a
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

def _card_surface(tokens: ThemeTokens | None = None):
    from gui.themes.tokens import classic_material
    if tokens is None:
        from gui.themes.manager import get_manager
        tokens = get_manager().current
    return (tokens.material or classic_material(tokens)).card


def elevation_margin(tokens: ThemeTokens | None = None) -> tuple[int, int, int, int]:
    """``QRect.adjusted`` margins covering the shadow around a card."""
    from gui.themes.glass import shadow_margins
    return shadow_margins(_card_surface(tokens))


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


def paint_elevation(painter: QtGui.QPainter, rect: QtCore.QRect, tokens: ThemeTokens,
                    radius: int | None = None) -> None:
    """Paint a soft card shadow for a card occupying ``rect`` (painter coords).

    Call it *before* the card itself is painted (i.e. from the parent
    surface's paintEvent). Only the ring around the card is visible.
    """
    from gui.themes.glass import paint_shadow
    paint_shadow(painter, rect, _card_surface(tokens), radius=radius)


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
    """Build a QPalette from the token set for selection / window colours.

    Roles the static stylesheet reads for *in-window* surfaces come from the
    material and may be translucent (Liquid Glass): item views, inputs,
    hover, selection, hairlines. Roles that fill *top-level* windows
    (``Window`` for dialogs, menus and combo popups, ``ToolTipBase``) stay
    opaque: a translucent top-level background has nothing behind it.
    """
    from gui.themes.glass import qcolor
    from gui.themes.tokens import classic_material

    p = QtGui.QPalette()
    m = tokens.material or classic_material(tokens)
    glass = m.is_glass
    disabled = QtGui.QPalette.ColorGroup.Disabled

    def s(role: QtGui.QPalette.ColorRole, value: str) -> None:
        c = qcolor(value)
        p.setColor(role, c)
        if c.alpha() < 255:
            faded = QtGui.QColor(c)
            faded.setAlpha(c.alpha() // 2)
            p.setColor(disabled, role, faded)
        else:
            p.setColor(disabled, role, lerp_color(c.name(), tokens.content_bg, 0.55))

    R = QtGui.QPalette.ColorRole
    s(R.Window,          tokens.content_bg)
    s(R.WindowText,      tokens.text_primary)
    s(R.Base,            m.well.fill if glass else tokens.input_bg)
    s(R.AlternateBase,   m.view_alt)
    s(R.Text,            tokens.text_primary)
    s(R.BrightText,      tokens.text_inverse)
    s(R.Button,          m.view)
    s(R.ButtonText,      tokens.text_primary)
    s(R.Highlight,       m.selection)
    s(R.HighlightedText, m.selection_text)
    s(R.Link,            tokens.accent)
    s(R.LinkVisited,     tokens.accent_hover)
    s(R.ToolTipBase,     tokens.log_bg)
    s(R.ToolTipText,     tokens.log_text)
    s(R.PlaceholderText, tokens.text_muted)
    # The 3-D roles carry the remaining tokens the static stylesheet needs
    # (manager._static_qss reads them with palette(mid) etc.). AlphaDEXStyle
    # paints from the tokens; Fusion fallbacks get themed lines instead of
    # the system's light-grey defaults.
    s(R.Mid,             m.card.border or tokens.card_border)
    s(R.Dark,            m.raised.border if glass else tokens.input_border)
    s(R.Midlight,        m.hover)
    s(R.Light,           m.hairline)
    s(R.Shadow,          tokens.text_secondary)
    accent_role = getattr(R, "Accent", None)
    if accent_role is not None:
        s(accent_role, tokens.accent)
    return p
