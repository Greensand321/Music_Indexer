"""ThemeManager singleton — apply, persist and auto-switch themes.

The *effective* tokens (``ThemeManager.current``) are the selected theme with
the Liquid Glass accent applied and, when "Reduce transparency" is on, every
translucent surface flattened (``tokens.with_accent`` /
``tokens.reduced_transparency``). Accent and transparency changes go through
the same fast path as a theme switch.

One-time migration: Liquid Glass is the default look. The first time a
config that saved another theme is loaded, the theme is switched to Liquid
Glass once, the old key is kept under ``theme_before_glass`` and the marker
``theme_glass_migrated`` stops it from ever happening again (choosing a
classic theme afterwards sticks). The main window logs the switch.
"""
from __future__ import annotations

import datetime
import logging

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.tokens import (
    ACCENTS, DEFAULT_ACCENT, DEFAULT_DARK, DEFAULT_LIGHT, DEFAULT_THEME, GLASS_KEY, THEMES,
    ThemeTokens, reduced_transparency, with_accent,
)
from gui.themes.effects import build_palette

_log = logging.getLogger(__name__)

# Config keys owned by the manager.
THEME_KEY = "theme"
ACCENT_KEY = "glass_accent"
REDUCE_TRANSPARENCY_KEY = "reduce_transparency"
MIGRATION_MARKER = "theme_glass_migrated"
PREVIOUS_THEME_KEY = "theme_before_glass"


# ── Static QSS ───────────────────────────────────────────────────────────────
# AlphaDEXStyle paints the widgets (buttons, fields, checkboxes, tabs,
# scrollbars, sliders, progress bars). This stylesheet covers what a
# QProxyStyle cannot reach: tooltips, text areas, item views and headers,
# menus and popups, card frames, the stepper and the typography roles.
#
# The stylesheet is STATIC: it is installed once and never replaced. Any
# change to the application stylesheet makes Qt re-polish every widget in
# the app (~0.7 s for the ~1000-widget main window, whatever the rules say),
# which was the ~1 s freeze on every theme switch (DS-04). Colours therefore
# come from palette() references, and a theme switch only swaps the palette
# and re-polishes the widgets these rules match (``_is_themed``, ~15 ms).
# The palette roles carry these tokens (see effects.build_palette); under
# Liquid Glass the in-window roles are translucent:
#
#   window -> content_bg (opaque: dialogs, menus, popups)
#   window-text / text -> text_primary
#   button -> material.view (item views)    alternate-base -> material.view_alt
#   base -> input / well fill    mid -> card border    dark -> field border
#   midlight -> material.hover   light -> material.hairline
#   shadow -> text_secondary     placeholder-text -> text_muted
#   highlight -> material.selection    highlighted-text -> its text
#   link -> accent               tool-tip-base / -text -> log_bg / log_text
#
# Order matters: rules of equal specificity resolve by position, so the
# general scroll-area rule comes before the item-view and text-area rules.
#
# Palette changes a QSS rule makes on a widget are not inherited by its
# children (verified on PySide6 6.12), so a role means the same thing inside
# a card as outside it.

def _static_qss() -> str:
    from gui.fonts.loader import MONO_STACK, UI_STACK
    from gui.themes.tokens import TYPE

    title = TYPE["title"].px
    heading = TYPE["heading"].px
    body = TYPE["body"].px
    label = TYPE["label"].px
    caption = TYPE["caption"].px

    return f"""
QToolTip {{
    background: palette(tool-tip-base);
    color: palette(tool-tip-text);
    border: 1px solid palette(mid);
    border-radius: 8px;
    padding: 5px 9px;
    font-family: {UI_STACK};
    font-size: {caption}px;
}}
QAbstractScrollArea {{
    background: transparent;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QAbstractItemView {{
    background: palette(button);
    alternate-background-color: palette(alternate-base);
    color: palette(window-text);
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
    gridline-color: palette(light);
    border: none;
    outline: 0;
}}
QAbstractItemView::item:hover {{
    background: palette(midlight);
}}

/* ── Dialogs (QDialog, QMessageBox, QFileDialog, QInputDialog,
   QProgressDialog). Top-level windows stay opaque: palette(window) is the
   material's backdrop_base. AlphaDEXStyle paints the rest of the chrome
   (gui/themes/style.py): the raised content surface over that fill with the
   button row on a footer band, pill buttons with the default one primary,
   and the message-box icons. These rules add the type and the wells. They
   sit before the combo-popup rule so a popup inside a dialog keeps its own
   look (equal specificity: the later rule wins). ── */
QDialog QAbstractItemView {{
    border: 1px solid palette(mid);
    border-radius: 12px;
}}
QMessageBox QLabel#qt_msgbox_label {{
    color: palette(window-text);
    font-size: {body}px;
    padding: 9px 2px 2px 0;
}}
QMessageBox QLabel#qt_msgbox_informativelabel {{
    color: palette(shadow);
    font-size: {label}px;
    padding: 0 2px 2px 0;
}}
QInputDialog QLabel, QProgressDialog QLabel {{
    color: palette(window-text);
    font-size: {body}px;
}}
QFileDialog QLabel {{
    color: palette(shadow);
}}
QDialogButtonBox {{
    dialogbuttonbox-buttons-have-icons: 0;
}}

QComboBox QAbstractItemView, QCompleter QAbstractItemView {{
    background: palette(window);
    border: 1px solid palette(mid);
    padding: 4px;
}}
QPlainTextEdit, QTextEdit {{
    background: palette(base);
    color: palette(window-text);
    border: 1px solid palette(dark);
    border-radius: 14px;
    padding: 2px 4px;
    selection-background-color: palette(highlight);
    selection-color: palette(highlighted-text);
    font-family: {UI_STACK};
    font-size: {label}px;
}}
QPlainTextEdit#logText, QTextEdit#logText {{
    background: palette(tool-tip-base);
    color: palette(tool-tip-text);
    border: none;
    border-radius: 12px;
    font-family: {MONO_STACK};
    font-size: {caption}px;
    line-height: 1.4;
}}
QHeaderView {{
    background: transparent;
    border: none;
}}
QHeaderView::section {{
    background: transparent;
    color: palette(shadow);
    border: none;
    border-bottom: 1px solid palette(mid);
    padding: 6px 10px;
    font-weight: 600;
    font-size: {caption}px;
}}
QTableCornerButton::section {{
    background: transparent;
    border: none;
}}
QSplitter::handle {{
    background: palette(light);
    width: 1px;
    height: 1px;
}}
QStatusBar {{
    background: transparent;
    color: palette(shadow);
    border: none;
    font-size: {caption}px;
}}

/* ── Card frames ──
   Cards made with WorkspaceBase._make_card() are GlassCards (glass="true"):
   they paint themselves with gui.themes.glass.paint_surface. The rule below
   is the fallback for plain frames that only borrow the name (dialogs). */
QFrame#workspaceCard, QFrame#headerCard, QFrame#actionCard {{
    background: palette(button);
    border: 1px solid palette(mid);
    border-radius: 18px;
}}
QFrame#workspaceCard[glass="true"], QFrame#headerCard[glass="true"],
QFrame#actionCard[glass="true"] {{
    background: transparent;
    border: none;
}}

/* ── Workflow stepper ── */
QFrame#workflowStepper {{
    background: palette(alternate-base);
    border: 1px solid palette(light);
    border-radius: 12px;
}}
QLabel#stepActive {{
    color: palette(link);
    font-weight: 700;
    font-size: {caption}px;
}}
QLabel#stepInactive {{
    color: palette(placeholder-text);
    font-size: {caption}px;
}}
QLabel#stepArrow {{
    color: palette(placeholder-text);
    font-size: {caption}px;
}}

/* ── Workspace typography (one ramp: gui.themes.tokens.TYPE) ── */
QLabel#sectionTitle {{
    color: palette(window-text);
    font-size: {title}px;
    font-weight: 700;
}}
QLabel#sectionSubtitle {{
    color: palette(shadow);
    font-size: {label}px;
}}
QLabel#cardTitle {{
    color: palette(window-text);
    font-weight: 600;
    font-size: {heading}px;
}}
QLabel#phaseLabel {{
    color: palette(window-text);
    font-weight: 600;
    font-size: {label}px;
}}
QLabel#phaseDesc {{
    color: palette(placeholder-text);
    font-size: {caption}px;
}}
QLabel#statusHint {{
    color: palette(placeholder-text);
    font-size: {label}px;
}}
QLabel#notesHint {{
    color: palette(placeholder-text);
    font-size: {caption}px;
}}

/* ── Shell text roles (fonts come from gui.themes.glass.font) ── */
QLabel#mutedLabel, QLabel#headerStats, QLabel#capsuleMeta, QLabel#capsuleTime,
QLabel#landingHint, QLabel#landingFolderStats {{
    color: palette(shadow);
}}
QLabel#pageTitle, QLabel#brandName, QLabel#capsuleTitle, QLabel#landingTitle {{
    color: palette(window-text);
}}
QLabel#sidebarSectionLabel {{
    color: palette(placeholder-text);
}}

/* ── Page roles (R2). One block per area so parallel edits merge cleanly;
   add rules for your area's objectNames inside your own block only. ── */

/* [area:home] begin */

/* [area:home] end */


/* [area:organize] begin */
/* Hero card titles (font: the title role, set in code). */
QLabel#organizeHeroTitle {{
    color: palette(window-text);
}}
/* Run logs on the Organize pages and Exports: mono, on the normal field. */
QPlainTextEdit#organizeLog {{
    font-family: {MONO_STACK};
    font-size: {caption}px;
}}
/* [area:organize] end */


/* [area:duplicates] begin */
/* Find duplicates: the Scan · Review · Execute steps as a compact pill row. */
QFrame#dupStepRow QLabel#stepActive, QFrame#dupStepRow QLabel#stepInactive {{
    border-radius: 9px;
    padding: 3px 10px;
    font-size: {caption}px;
}}
QFrame#dupStepRow QLabel#stepActive {{
    background: palette(highlight);
    color: palette(highlighted-text);
    border: 1px solid palette(link);
    font-weight: 600;
}}
QFrame#dupStepRow QLabel#stepInactive {{
    background: palette(midlight);
    color: palette(placeholder-text);
    border: 1px solid palette(light);
}}
/* The group list paints its own glass rows (a delegate): no view fill. */
QTreeWidget#dupGroupList {{
    background: transparent;
    border: none;
}}
/* Hero card titles (title role font; cardTitle's 16 px rule would override it). */
QLabel#dupHeroTitle {{
    color: palette(window-text);
}}
/* Prototype launchers: the "not ready to test" note inside its well. */
QLabel#experimentalNote {{
    color: palette(window-text);
}}
/* [area:duplicates] end */


/* [area:metadata] begin */
QAbstractItemView[metaGlass="true"] {{
    background: transparent;
}}
QLabel#metaHeroTitle {{
    color: palette(window-text);
}}
QPlainTextEdit#genreMapEditor {{
    font-family: {MONO_STACK};
    font-size: {caption}px;
}}
/* [area:metadata] end */


/* [area:player] begin */
/* Player + Playlists item views sit on their glass card: no view fill (the
   row delegates paint hover, selection and the playing row). */
QTableWidget#playerLibraryTable, QListWidget#playerQueueList,
QListWidget#playerRecentList, QListWidget#playlistResults,
QListWidget#playlistRepairList {{
    background: transparent;
}}
/* [area:player] end */


/* [area:sound_map] begin */
QLabel#soundMapHeroTitle, QLabel#clusterLegendName, QLabel#clusterLegendCount {{
    color: palette(window-text);
}}
QLabel#clusterLegendName:disabled, QLabel#clusterLegendCount:disabled {{
    color: palette(placeholder-text);
}}
QLabel#clusterPlaylistsHeader, QLabel#qualityReportNote {{
    color: palette(shadow);
}}
/* [area:sound_map] end */


/* [area:takeaway_settings] begin */
/* Text on a painted well (a GlassPanel "well" behind it): the panel draws
   the rounded surface, so the editor itself stays transparent (an editor's
   own background fills its viewport square, past its rounded border). */
QPlainTextEdit#wellText, QPlainTextEdit#wellMono {{
    background: transparent;
    border: none;
    padding: 0;
}}
QPlainTextEdit#wellMono {{
    font-family: {MONO_STACK};
    font-size: {caption}px;
}}
/* [area:takeaway_settings] end */


/* ── Menus and the menu bar (popups: opaque) ── */
QMenuBar {{
    background: transparent;
    color: palette(window-text);
}}
QMenuBar::item:selected {{
    background: palette(midlight);
    border-radius: 8px;
}}
QMenu {{
    background: palette(window);
    color: palette(window-text);
    border: 1px solid palette(mid);
    border-radius: 12px;
    padding: 6px;
}}
QMenu::item {{
    padding: 6px 18px 6px 12px;
    border-radius: 8px;
}}
QMenu::item:selected {{
    background: palette(highlight);
    color: palette(highlighted-text);
}}
QMenu::separator {{
    height: 1px;
    background: palette(light);
    margin: 5px 8px;
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
    "mutedLabel", "headerStats", "capsuleMeta", "capsuleTime", "landingHint",
    "landingFolderStats", "pageTitle", "brandName", "capsuleTitle", "landingTitle",
    "sidebarSectionLabel",
}) | frozenset().union(*(
    # Page roles (R2): add an area's objectNames to its own set only, so
    # parallel edits merge cleanly. Keep in step with the QSS blocks above.

    # [area:home]
    frozenset(),

    # [area:organize]
    frozenset({"organizeHeroTitle"}),

    # [area:duplicates]
    frozenset({"dupStepRow", "dupGroupList", "dupHeroTitle", "experimentalNote"}),

    # [area:metadata]
    frozenset({"metaHeroTitle", "genreMapEditor"}),

    # [area:player]
    frozenset(),

    # [area:sound_map]
    frozenset({"soundMapHeroTitle", "clusterLegendName", "clusterLegendCount",
               "clusterPlaylistsHeader", "qualityReportNote"}),

    # [area:takeaway_settings]  (message-box labels: the global dialog rules)
    frozenset({"qt_msgbox_label", "qt_msgbox_informativelabel"}),
))
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

    theme_changed = Signal(object)   # emits the effective ThemeTokens

    _instance: "ThemeManager | None" = None

    def __init__(self) -> None:
        super().__init__()
        self._base: ThemeTokens = THEMES[DEFAULT_THEME]   # selected, before accent/transparency
        self._current: ThemeTokens = THEMES[DEFAULT_THEME]
        self._auto_mode: bool = False
        self._auto_dark:  str = DEFAULT_DARK
        self._auto_light: str = DEFAULT_LIGHT
        self._accent: str = DEFAULT_ACCENT
        self._reduce_transparency: bool = False
        self._migration_note: str | None = None
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
        """The effective tokens: the theme, its accent, transparency."""
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

    def selected_key(self) -> str:
        """The theme the user chose ("auto" in Auto mode)."""
        return "auto" if self._auto_mode else self._base.key

    def accent(self) -> str:
        return self._accent

    def set_accent(self, key: str) -> None:
        """Liquid Glass accent (blue · peach · mint · lilac); applies at once."""
        if key not in ACCENTS or key == self._accent:
            return
        self._accent = key
        self._save({ACCENT_KEY: key})
        self._apply_tokens(self._base)

    def reduce_transparency(self) -> bool:
        return self._reduce_transparency

    def set_reduce_transparency(self, on: bool) -> None:
        """Opaque surfaces for accessibility / performance; applies at once."""
        on = bool(on)
        if on == self._reduce_transparency:
            return
        self._reduce_transparency = on
        self._save({REDUCE_TRANSPARENCY_KEY: on})
        self._apply_tokens(self._base)

    def take_migration_note(self) -> str | None:
        """The one-time "switched to Liquid Glass" message (once), for the log."""
        note, self._migration_note = self._migration_note, None
        return note

    def configure_auto(self, dark_key: str, light_key: str) -> None:
        """Set the dark/light pair used by Auto mode."""
        self._auto_dark  = dark_key  if dark_key  in THEMES else DEFAULT_DARK
        self._auto_light = light_key if light_key in THEMES else DEFAULT_LIGHT
        self._save({"auto_dark_theme": self._auto_dark, "auto_light_theme": self._auto_light})
        if self._auto_mode:
            self._apply_tokens(self._auto_tokens())
        self._sync_auto_timer()

    def load_persisted(self) -> None:
        """Load the saved theme (migrating it once) and apply it."""
        try:
            from config import load_config, save_config
            cfg = load_config()
            if self._migrate(cfg):
                save_config(cfg)
            self._auto_dark  = cfg.get("auto_dark_theme",  DEFAULT_DARK)
            self._auto_light = cfg.get("auto_light_theme", DEFAULT_LIGHT)
            accent = cfg.get(ACCENT_KEY, DEFAULT_ACCENT)
            self._accent = accent if accent in ACCENTS else DEFAULT_ACCENT
            self._reduce_transparency = bool(cfg.get(REDUCE_TRANSPARENCY_KEY, False))
            key = cfg.get(THEME_KEY, DEFAULT_THEME)
            self.apply(key)
        except Exception:
            self.apply(DEFAULT_THEME)

    def is_auto(self) -> bool:
        return self._auto_mode

    def auto_dark(self)  -> str: return self._auto_dark
    def auto_light(self) -> str: return self._auto_light

    # ── Internal ──────────────────────────────────────────────────────────

    def _migrate(self, cfg: dict) -> bool:
        """Switch a saved theme to Liquid Glass once. True when ``cfg`` changed."""
        if cfg.get(MIGRATION_MARKER):
            return False
        saved = cfg.get(THEME_KEY)
        if saved and saved != GLASS_KEY:
            cfg[PREVIOUS_THEME_KEY] = saved
            cfg[THEME_KEY] = GLASS_KEY
            name = "Auto" if saved == "auto" else (THEMES[saved].name if saved in THEMES else saved)
            self._migration_note = (
                f"Appearance: Liquid Glass is the new look, so your theme ({name}) was "
                f"switched to it once. {name} is still in Settings → Appearance."
            )
            _log.info("Theme migrated once from %r to %r", saved, GLASS_KEY)
        cfg[MIGRATION_MARKER] = True
        return True

    def _effective(self, base: ThemeTokens) -> ThemeTokens:
        tokens = with_accent(base, self._accent)
        if self._reduce_transparency:
            tokens = reduced_transparency(tokens)
        return tokens

    def _apply_tokens(self, tokens: ThemeTokens) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        self._base = THEMES.get(tokens.key, tokens)
        tokens = self._effective(self._base)
        self._current = tokens
        try:
            from gui.themes import appearance
            appearance.invalidate()
        except Exception:  # pragma: no cover - never block a switch
            pass
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
            # A widget with its own inline stylesheet had its palette
            # snapshotted by Qt's stylesheet engine at polish time, so it (and
            # every child inheriting from it) keeps the old theme unless it is
            # re-polished too — e.g. the sidebar's section headers stayed
            # Midnight-white on Pearl. tests/ui/test_ui_theme_switch_equivalence.py
            # The same applies to descendants of such a widget (an inline
            # sheet cascades): Qt pins their palette too (WA_SetPalette). Owners
            # that set a palette deliberately re-apply it on theme_changed,
            # which is emitted after this pass, so re-polishing them is safe.
            if _is_themed(w) or w.styleSheet() or w.testAttribute(set_palette):
                st = w.style()
                st.unpolish(w)
                st.polish(w)
            else:
                w.setPalette(empty)
            # Style-painted widgets read the tokens at paint time.
            w.update()
            stack.extend(reversed(w.findChildren(
                QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)))

    @staticmethod
    def _set_base_font(app: QtWidgets.QApplication) -> None:
        """The application-wide default font: the type ramp's body role."""
        try:
            from gui.themes.glass import font
            app.setFont(font("body"))
        except ImportError:  # pragma: no cover
            pass

    def _save(self, values: dict) -> None:
        try:
            from config import load_config, save_config
            cfg = load_config()
            cfg.update(values)
            save_config(cfg)
        except Exception:
            pass

    def _persist(self) -> None:
        self._save({THEME_KEY: self.selected_key()})

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
