"""ThemeManager singleton — apply, persist and auto-switch themes."""
from __future__ import annotations

import datetime

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.tokens import (
    ThemeTokens, THEMES, DARK_THEMES, LIGHT_THEMES,
    DEFAULT_DARK, DEFAULT_LIGHT,
)
from gui.themes.effects import build_palette


# ── Residual QSS ─────────────────────────────────────────────────────────────
# AlphaDEXStyle handles ALL widget painting.  This residual is only for the
# handful of things QProxyStyle cannot reach: tooltip sizing, plain-text area
# font, and explicit selection-colour anchoring.

def _residual_qss(t: ThemeTokens) -> str:
    try:
        from gui.fonts.loader import UI_STACK, MONO_STACK
    except ImportError:
        UI_STACK   = '"Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif'
        MONO_STACK = '"Consolas", "Menlo", monospace'

    return f"""
QToolTip {{
    background: {t.log_bg};
    color: {t.log_text};
    border: 1px solid {t.card_border};
    border-radius: 6px;
    padding: 4px 8px;
    font-family: {UI_STACK};
    font-size: 9pt;
}}
QPlainTextEdit, QTextEdit {{
    background: {t.input_bg};
    color: {t.text_primary};
    border: 1px solid {t.input_border};
    border-radius: 7px;
    selection-background-color: {t.accent};
    selection-color: {t.text_inverse};
    font-family: {UI_STACK};
    font-size: 10pt;
}}
QPlainTextEdit#logText, QTextEdit#logText {{
    background: {t.log_bg};
    color: {t.log_text};
    border: none;
    font-family: {MONO_STACK};
    font-size: 10pt;
    line-height: 1.4;
}}
QAbstractScrollArea {{
    background: {t.content_bg};
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QHeaderView {{
    background: {t.card_bg};
}}
QHeaderView::section {{
    background: {t.card_bg};
    color: {t.text_secondary};
    border: none;
    border-bottom: 1px solid {t.card_border};
    padding: 5px 8px;
    font-weight: 600;
    font-size: 12px;
}}
QAbstractItemView {{
    background: {t.card_bg};
    alternate-background-color: {t.content_bg};
    color: {t.text_primary};
    selection-background-color: {t.accent};
    selection-color: {t.text_inverse};
    border: none;
    outline: 0;
}}
QAbstractItemView::item:hover {{
    background: {t.sidebar_hover};
}}
QSplitter::handle {{
    background: {t.card_border};
    width: 1px;
    height: 1px;
}}
QStatusBar {{
    background: {t.card_bg};
    color: {t.text_secondary};
    border-top: 1px solid {t.card_border};
    font-size: 12px;
}}
QMenuBar {{
    background: {t.card_bg};
    color: {t.text_primary};
}}
QMenuBar::item:selected {{
    background: {t.sidebar_hover};
}}
QMenu {{
    background: {t.card_bg};
    color: {t.text_primary};
    border: 1px solid {t.card_border};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item:selected {{
    background: {t.accent};
    color: {t.text_inverse};
    border-radius: 4px;
}}
"""


# ── ThemeManager ──────────────────────────────────────────────────────────────

class ThemeManager(QtCore.QObject):
    """Singleton.  Apply, persist, and auto-switch themes at runtime."""

    theme_changed = Signal(object)   # emits ThemeTokens

    _instance: "ThemeManager | None" = None

    def __init__(self) -> None:
        super().__init__()
        self._current: ThemeTokens = THEMES[DEFAULT_DARK]
        self._auto_mode: bool = False
        self._auto_dark:  str = DEFAULT_DARK
        self._auto_light: str = DEFAULT_LIGHT
        # Qt 6.5+ color-scheme change signal
        self._connect_system_theme()

    @classmethod
    def instance(cls) -> "ThemeManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def current(self) -> ThemeTokens:
        return self._current

    def apply(self, key: str) -> None:
        """Install a named theme on the running QApplication."""
        if key == "auto":
            self._auto_mode = True
            self._apply_tokens(self._auto_tokens())
        else:
            self._auto_mode = False
            if key not in THEMES:
                return
            self._apply_tokens(THEMES[key])
        self._persist()

    def configure_auto(self, dark_key: str, light_key: str) -> None:
        """Set the dark/light pair used by Auto mode."""
        self._auto_dark  = dark_key  if dark_key  in THEMES else DEFAULT_DARK
        self._auto_light = light_key if light_key in THEMES else DEFAULT_LIGHT
        try:
            from config import load_config, save_config
            cfg = load_config()
            cfg["auto_dark_theme"]  = self._auto_dark
            cfg["auto_light_theme"] = self._auto_light
            save_config(cfg)
        except Exception:
            pass
        if self._auto_mode:
            self._apply_tokens(self._auto_tokens())

    def load_persisted(self) -> None:
        """Load the saved theme from config and apply it."""
        try:
            from config import load_config
            cfg = load_config()
            self._auto_dark  = cfg.get("auto_dark_theme",  DEFAULT_DARK)
            self._auto_light = cfg.get("auto_light_theme", DEFAULT_LIGHT)
            key = cfg.get("theme", DEFAULT_DARK)
            self.apply(key)
        except Exception:
            self.apply(DEFAULT_DARK)

    def is_auto(self) -> bool:
        return self._auto_mode

    def auto_dark(self)  -> str: return self._auto_dark
    def auto_light(self) -> str: return self._auto_light

    # ── Internal ──────────────────────────────────────────────────────────

    def _apply_tokens(self, tokens: ThemeTokens) -> None:
        from gui.themes.style import AlphaDEXStyle
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        self._current = tokens
        app.setStyle(AlphaDEXStyle(tokens))
        app.setPalette(build_palette(tokens))
        app.setStyleSheet(_residual_qss(tokens))
        self._set_base_font(app)
        self.theme_changed.emit(tokens)

    @staticmethod
    def _set_base_font(app: QtWidgets.QApplication) -> None:
        """Set Inter as the application-wide default font."""
        try:
            from gui.fonts.loader import UI_FAMILY, TypeScale
            f = QtGui.QFont(UI_FAMILY, TypeScale.BODY)
            f.setWeight(QtGui.QFont.Weight(TypeScale.W_REGULAR))
            f.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
            app.setFont(f)
        except ImportError:
            pass

    def _persist(self) -> None:
        try:
            from config import load_config, save_config
            cfg = load_config()
            cfg["theme"] = "auto" if self._auto_mode else self._current.key
            save_config(cfg)
        except Exception:
            pass

    def _auto_tokens(self) -> ThemeTokens:
        return THEMES[self._auto_light if self._is_daytime() else self._auto_dark]

    @staticmethod
    def _is_daytime() -> bool:
        # Try Qt 6.5+ color scheme first
        try:
            hints = QtGui.QGuiApplication.styleHints()
            scheme = hints.colorScheme()
            Qt = QtCore.Qt
            if hasattr(Qt, "ColorScheme"):
                return scheme == Qt.ColorScheme.Light
        except Exception:
            pass
        # Fallback: 07:00–20:00 = day
        h = datetime.datetime.now().hour
        return 7 <= h < 20

    def _connect_system_theme(self) -> None:
        try:
            hints = QtGui.QGuiApplication.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        except Exception:
            pass

    def _on_system_scheme_changed(self) -> None:
        if self._auto_mode:
            self._apply_tokens(self._auto_tokens())


# ── Module-level convenience ──────────────────────────────────────────────────

def get_manager() -> ThemeManager:
    return ThemeManager.instance()
