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
# handful of things QProxyStyle cannot reach: tooltips, plain-text areas,
# item views, cards, the stepper and the workspace typography roles.
#
# The stylesheet is STATIC: it is installed once and never replaced. Any
# change to the application stylesheet makes Qt re-polish every widget in
# the app (~0.7 s for the ~1000-widget main window, whatever the rules say),
# which was the ~1 s freeze on every theme switch (DS-04). Colours therefore
# come from palette() references, and a theme switch only swaps the palette
# and re-polishes the widgets these rules match (``_is_themed``, ~15 ms).
# The palette roles carry these tokens (see effects.build_palette):
#
#   window -> content_bg        window-text -> text_primary
#   button -> card_bg           base -> input_bg       text -> text_primary
#   mid -> card_border          dark -> input_border   midlight -> sidebar_hover
#   shadow -> text_secondary    placeholder-text -> text_muted
#   highlight -> accent         highlighted-text -> text_inverse
#   tool-tip-base -> log_bg     tool-tip-text -> log_text
#
# Palette changes a QSS rule makes on a widget are not inherited by its
# children (verified on PySide6 6.12), so a role means the same thing inside
# a card as outside it.

def _static_qss() -> str:
    from gui.fonts.loader import UI_STACK, MONO_STACK, TypeScale

    section_px = TypeScale.px(TypeScale.SECTION_HDR)
    card_px = TypeScale.px(TypeScale.CARD_TITLE)
    label_px = TypeScale.LABEL_PX
    caption_px = TypeScale.CAPTION_PX

    return f"""
QToolTip {{
    background: palette(tool-tip-base);
    color: palette(tool-tip-text);
    border: 1px solid palette(mid);
    border-radius: 6px;
    padding: 4px 8px;
    font-family: {UI_STACK};
    font-size: 9pt;
}}
QPlainTextEdit, QTextEdit {{
    background: palette(base);
    color: palette(window-text);
    border: 1px solid palette(dark);
    border-radius: 7px;
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
    font-family: {UI_STACK};
    font-size: 10pt;
}}
QPlainTextEdit#logText, QTextEdit#logText {{
    background: palette(tool-tip-base);
    color: palette(tool-tip-text);
    border: none;
    font-family: {MONO_STACK};
    font-size: 10pt;
    line-height: 1.4;
}}
QAbstractScrollArea {{
    background: palette(window);
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QHeaderView {{
    background: palette(button);
}}
QHeaderView::section {{
    background: palette(button);
    color: palette(shadow);
    border: none;
    border-bottom: 1px solid palette(mid);
    padding: 5px 8px;
    font-weight: 600;
    font-size: {label_px}px;
}}
QAbstractItemView {{
    background: palette(button);
    alternate-background-color: palette(window);
    color: palette(window-text);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
    border: none;
    outline: 0;
}}
QAbstractItemView::item:hover {{
    background: palette(midlight);
}}
QSplitter::handle {{
    background: palette(mid);
    width: 1px;
    height: 1px;
}}
QStatusBar {{
    background: palette(button);
    color: palette(shadow);
    border-top: 1px solid palette(mid);
    font-size: {label_px}px;
}}

/* ── Workspace cards ── */
QFrame#workspaceCard {{
    background: palette(button);
    border: 1px solid palette(mid);
    border-radius: 12px;
}}
QFrame#headerCard {{
    background: palette(button);
    border-top: 1px solid palette(mid);
    border-right: 1px solid palette(mid);
    border-bottom: 1px solid palette(mid);
    border-left: 3px solid palette(highlight);
    border-radius: 10px;
}}
QFrame#actionCard {{
    background: palette(button);
    border: 1px solid palette(mid);
    border-top: 2px solid palette(highlight);
    border-radius: 10px;
}}

/* ── Workflow stepper ── */
QFrame#workflowStepper {{
    background: transparent;
    border: 1px solid palette(mid);
    border-radius: 6px;
}}
QLabel#stepActive {{
    color: palette(highlight);
    font-weight: 700;
    font-size: {label_px}px;
}}
QLabel#stepInactive {{
    color: palette(placeholder-text);
    font-size: {label_px}px;
}}
QLabel#stepArrow {{
    color: palette(placeholder-text);
    font-size: {label_px}px;
}}

/* ── Workspace typography (one ramp: gui.fonts.loader.TypeScale) ── */
QLabel#sectionTitle {{
    color: palette(window-text);
    font-size: {section_px}px;
    font-weight: 700;
}}
QLabel#sectionSubtitle {{
    color: palette(shadow);
    font-size: {label_px}px;
}}
QLabel#cardTitle {{
    color: palette(window-text);
    font-weight: 600;
    font-size: {card_px}px;
}}
QLabel#phaseLabel {{
    color: palette(window-text);
    font-weight: 600;
    font-size: {label_px}px;
}}
QLabel#phaseDesc {{
    color: palette(placeholder-text);
    font-size: {caption_px}px;
}}
QLabel#statusHint {{
    color: palette(placeholder-text);
    font-size: {label_px}px;
}}
QLabel#notesHint {{
    color: palette(placeholder-text);
    font-size: {caption_px}px;
}}
QMenuBar {{
    background: palette(button);
    color: palette(window-text);
}}
QMenuBar::item:selected {{
    background: palette(midlight);
}}
QMenu {{
    background: palette(button);
    color: palette(window-text);
    border: 1px solid palette(mid);
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item:selected {{
    background: palette(highlight);
    color: palette(highlighted-text);
    border-radius: 4px;
}}
"""


def _residual_qss(t: ThemeTokens | None = None) -> str:
    """Backward-compatible name: the stylesheet no longer depends on ``t``."""
    return _static_qss()


# Widgets whose look comes from a colour rule above. Only these need a
# re-polish when the palette changes (QSS resolves palette() at polish time).
_THEMED_OBJECT_NAMES = frozenset({
    "workspaceCard", "headerCard", "actionCard", "workflowStepper",
    "stepActive", "stepInactive", "stepArrow", "sectionTitle",
    "sectionSubtitle", "cardTitle", "phaseLabel", "phaseDesc",
    "statusHint", "notesHint", "logText",
})
_THEMED_TYPES = (
    QtWidgets.QAbstractScrollArea,   # item views, headers, text edits, scroll areas
    QtWidgets.QStatusBar,
    QtWidgets.QMenuBar,
    QtWidgets.QMenu,
    QtWidgets.QSplitterHandle,
)


def _is_scroll_content(w: QtWidgets.QWidget) -> bool:
    """Matches ``QScrollArea > QWidget > QWidget``. QSS gives these an
    explicit palette at polish time, and their children (a whole workspace)
    inherit it, so they must be re-polished too or hidden pages keep the
    previous theme's palette."""
    viewport = w.parentWidget()
    return (
        viewport is not None
        and isinstance(viewport.parentWidget(), QtWidgets.QScrollArea)
        and viewport.parentWidget().viewport() is viewport
    )


def _is_themed(w: QtWidgets.QWidget) -> bool:
    return (
        isinstance(w, _THEMED_TYPES)
        or w.objectName() in _THEMED_OBJECT_NAMES
        or w.inherits("QTipLabel")
        or _is_scroll_content(w)
    )


AUTO_RECHECK_MS = 15 * 60 * 1000   # Auto mode re-checks day/night this often


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
        self._style = None            # the one AlphaDEXStyle instance
        self._qss: str | None = None  # the static stylesheet, once installed
        # Auto mode re-checks day/night periodically (needed for the clock
        # fallback; the colour-scheme signal covers platforms that know).
        self._auto_timer = QtCore.QTimer(self)
        self._auto_timer.setInterval(AUTO_RECHECK_MS)
        self._auto_timer.timeout.connect(self._on_auto_tick)
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
            if key not in THEMES:
                return
            self._auto_mode = False
            self._apply_tokens(THEMES[key])
        self._sync_auto_timer()
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
        self._sync_auto_timer()

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
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        self._current = tokens
        if self._installed(app):
            # Fast path (DS-04): same style object, same stylesheet. Swap the
            # tokens the style paints from and the palette the stylesheet
            # reads, then re-polish only the widgets coloured by the sheet.
            self._style.set_tokens(tokens)
            app.setPalette(build_palette(tokens))
            self._repolish_themed(app)
        else:
            from gui.themes.style import AlphaDEXStyle
            self._style = AlphaDEXStyle(tokens)
            app.setStyle(self._style)
            app.setPalette(build_palette(tokens))
            self._qss = _static_qss()
            app.setStyleSheet(self._qss)
            self._set_base_font(app)
        self.theme_changed.emit(tokens)

    def _installed(self, app: QtWidgets.QApplication) -> bool:
        """True when our style and static stylesheet are still in place."""
        style = self._style
        if style is None or self._qss is None:
            return False
        try:
            import shiboken6
            if not shiboken6.isValid(style):
                return False   # the app style was replaced; Qt deleted ours
        except ImportError:  # pragma: no cover - PySide6 ships shiboken6
            pass
        return app.styleSheet() == self._qss

    @staticmethod
    def _repolish_themed(app: QtWidgets.QApplication) -> None:
        """Bring every widget onto the new palette, parents before children.

        With an application stylesheet installed, Qt does not propagate an
        application palette change to stylesheet-styled widgets (i.e. all of
        them), so each widget is told explicitly: widgets coloured by the
        stylesheet are re-polished (which re-resolves their palette()
        references); every other widget without an explicit palette is
        reset to inherit. Widgets that set their own palette keep it (their
        owners refresh it on ``theme_changed``)."""
        import shiboken6

        empty = QtGui.QPalette()
        set_palette = QtCore.Qt.WidgetAttribute.WA_SetPalette
        stack = list(reversed(app.topLevelWidgets()))
        while stack:
            w = stack.pop()
            # Re-polishing can delete and rebuild children (a combo box's
            # popup view, for one); skip wrappers whose C++ side is gone.
            if not shiboken6.isValid(w):
                continue
            if _is_themed(w):
                st = w.style()
                st.unpolish(w)
                st.polish(w)
            elif not w.testAttribute(set_palette):
                w.setPalette(empty)
            # Style-painted widgets read the tokens at paint time.
            w.update()
            stack.extend(reversed(w.findChildren(
                QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)))

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
        dark = THEMES.get(self._auto_dark, THEMES[DEFAULT_DARK])
        light = THEMES.get(self._auto_light, THEMES[DEFAULT_LIGHT])
        return light if self._is_daytime() else dark

    @staticmethod
    def _is_daytime() -> bool:
        # The Qt 6.5+ colour scheme first, but only when the platform knows
        # it. Many Linux desktops report Unknown, which used to make Auto
        # permanently dark (DS-10); fall back to the clock then.
        try:
            hints = QtGui.QGuiApplication.styleHints()
            scheme = hints.colorScheme()
            cs = getattr(QtCore.Qt, "ColorScheme", None)
            if cs is not None:
                if scheme == cs.Light:
                    return True
                if scheme == cs.Dark:
                    return False
        except Exception:
            pass
        # Fallback: 07:00–20:00 = day
        h = ThemeManager._now().hour
        return 7 <= h < 20

    @staticmethod
    def _now() -> datetime.datetime:
        return datetime.datetime.now()

    def _sync_auto_timer(self) -> None:
        if self._auto_mode:
            if not self._auto_timer.isActive():
                self._auto_timer.start()
        else:
            self._auto_timer.stop()

    def _on_auto_tick(self) -> None:
        """Periodic Auto re-check: re-apply only when day/night flipped."""
        if not self._auto_mode:
            self._auto_timer.stop()
            return
        tokens = self._auto_tokens()
        if tokens.key != self._current.key:
            self._apply_tokens(tokens)

    def _connect_system_theme(self) -> None:
        try:
            hints = QtGui.QGuiApplication.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                hints.colorSchemeChanged.connect(self._on_system_scheme_changed)
        except Exception:
            pass

    def _on_system_scheme_changed(self) -> None:
        if self._auto_mode:
            tokens = self._auto_tokens()
            if tokens.key != self._current.key:
                self._apply_tokens(tokens)


# ── Module-level convenience ──────────────────────────────────────────────────

def get_manager() -> ThemeManager:
    return ThemeManager.instance()
