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


# ── Drop shadow ───────────────────────────────────────────────────────────────

def card_shadow(tokens: ThemeTokens) -> QtWidgets.QGraphicsDropShadowEffect:
    """Return a drop-shadow effect tuned to the current theme."""
    fx = QtWidgets.QGraphicsDropShadowEffect()
    fx.setBlurRadius(18)
    fx.setOffset(0, 3)
    # Parse "rgba(r,g,b,a)" or fallback
    try:
        s = tokens.card_shadow.strip()
        if s.startswith("rgba("):
            parts = s[5:-1].split(",")
            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
            a = int(float(parts[3].strip()) * 255)
            fx.setColor(QtGui.QColor(r, g, b, a))
        else:
            c = QtGui.QColor(s)
            fx.setColor(c)
    except Exception:
        fx.setColor(QtGui.QColor(0, 0, 0, 60))
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
    return p
