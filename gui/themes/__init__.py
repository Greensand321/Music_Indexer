"""gui.themes — AlphaDEX theme engine.

Public surface
--------------
ThemeTokens          – frozen dataclass of colour tokens per theme
THEMES               – dict[str, ThemeTokens] of all 14 named themes
DARK_THEMES          – list of dark theme keys
LIGHT_THEMES         – list of light theme keys
DEFAULT_DARK         – default dark theme key  ("midnight")
DEFAULT_LIGHT        – default light theme key ("pearl")

ThemeManager         – singleton; apply / persist / auto-switch themes
get_manager()        – convenience accessor for ThemeManager.instance()

AlphaDEXStyle        – QProxyStyle subclass; full QPainter rendering engine
build_palette()      – build QPalette from ThemeTokens

card_shadow()        – QGraphicsDropShadowEffect tuned to current theme
lerp_color()         – linear interpolation between two hex colours
with_alpha()         – return QColor with given alpha

R                    – border-radius constants (button, card, input, …)

HoverMixin           – hover animation mixin for QWidget subclasses
AnimatedButton       – QPushButton with hover animation
AnimatedNavButton    – sidebar nav button with badge + hover animation

ThemePickerDialog    – full-screen swatch picker dialog
AutoThemeDialog      – dark/light pair selector for Auto mode
open_theme_picker()  – convenience: create + show ThemePickerDialog
"""

from gui.themes.tokens import (
    ThemeTokens,
    THEMES,
    DARK_THEMES,
    LIGHT_THEMES,
    DEFAULT_DARK,
    DEFAULT_LIGHT,
)
from gui.themes.manager import ThemeManager, get_manager
from gui.themes.effects import build_palette, card_shadow, lerp_color, with_alpha, R
from gui.themes.animations import (
    HoverMixin,
    AnimatedButton,
    AnimatedNavButton,
    AnimatedTabButton,
)
from gui.themes.picker import ThemePickerDialog, AutoThemeDialog, open_theme_picker

__all__ = [
    # tokens
    "ThemeTokens", "THEMES", "DARK_THEMES", "LIGHT_THEMES", "DEFAULT_DARK", "DEFAULT_LIGHT",
    # manager
    "ThemeManager", "get_manager",
    # effects
    "build_palette", "card_shadow", "lerp_color", "with_alpha", "R",
    # animations
    "HoverMixin", "AnimatedButton", "AnimatedNavButton", "AnimatedTabButton",
    # picker
    "ThemePickerDialog", "AutoThemeDialog", "open_theme_picker",
]
