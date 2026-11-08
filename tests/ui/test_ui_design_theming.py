"""Design-system regressions: theme switching, shadows, gradient flag,
fonts, sidebar/log-drawer theming, theme picker, Auto mode, type ramp.

Bug IDs refer to docs/redesign/pages/02_design_system_motion.md.
"""
from __future__ import annotations

import datetime

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from gui.main_window import _WORKSPACE_MAP
from gui.themes.manager import ThemeManager, get_manager
from gui.themes.tokens import THEMES
from ui_harness import FrameGapMeter, process_events

R = QtGui.QPalette.ColorRole

# Measured on the dev container after the DS-04 fix: ~35-65 ms to apply plus
# ~30-45 ms to repaint (was 1,100-1,200 ms). The ceiling leaves room for slow
# CI; the structural test below is the strict guard.
THEME_SWITCH_CEILING_MS = 400


def _contrast(a: str, b: str) -> float:
    def lum(h):
        h = h.lstrip("#")
        c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    la, lb = lum(a), lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# ── DS-04 / SHELL-22: theme switch cost ──────────────────────────────────────

@pytest.mark.parametrize("theme", ["pearl", "midnight", "aurora"])
def test_theme_switch_freeze_ceiling(main_window, theme):
    get_manager().apply("blossom" if theme != "blossom" else "ember")
    process_events(30)
    with FrameGapMeter() as meter:
        get_manager().apply(theme)
        process_events(30)
    print(f"\n[perf] theme -> {theme}: {meter.summary()}")
    assert meter.max_gap_ms < THEME_SWITCH_CEILING_MS


def test_theme_switch_keeps_style_and_stylesheet(main_window, monkeypatch):
    """The ~1 s freeze came from re-installing the app style and stylesheet
    on every switch (each re-polishes ~1000 widgets). Neither may happen."""
    app = QtWidgets.QApplication.instance()
    get_manager().apply("midnight")
    calls = []
    monkeypatch.setattr(QtWidgets.QApplication, "setStyleSheet",
                        lambda self, *a: calls.append("setStyleSheet"))
    monkeypatch.setattr(QtWidgets.QApplication, "setStyle",
                        lambda *a: calls.append("setStyle"))
    for key in ("pearl", "ember", "azure"):
        get_manager().apply(key)
        assert get_manager()._style.t is THEMES[key]
    assert calls == []
    assert app.palette().color(R.Window).name() == THEMES["azure"].content_bg


def test_switch_reaches_hidden_pages(main_window):
    """Pages hidden during a switch must show the new theme when opened
    (stylesheet-styled widgets do not get palette propagation from Qt)."""
    mgr = get_manager()
    mgr.apply("midnight")
    process_events(10)
    mgr.apply("pearl")
    process_events(10)
    t = THEMES["pearl"]
    stale = []
    expected_role = {
        "cardTitle": t.text_primary, "sectionTitle": t.text_primary,
        "sectionSubtitle": t.text_secondary, "statusHint": t.text_muted,
        "phaseDesc": t.text_muted, "stepInactive": t.text_muted,
    }
    for key in _WORKSPACE_MAP:
        main_window._on_nav_changed(key)
        process_events(5)
    for w in main_window.findChildren(QtWidgets.QWidget):
        name = w.objectName()
        if name in expected_role and w.palette().color(R.WindowText).name() != expected_role[name]:
            stale.append(f"{type(w).__name__}#{name}")
        elif isinstance(w, QtWidgets.QTabWidget) and w.palette().color(R.Window).name() != t.content_bg:
            stale.append(f"QTabWidget in {type(w.window()).__name__}")
    assert stale == []


def test_static_stylesheet_carries_no_literal_colours():
    from gui.themes.manager import _static_qss
    import re
    assert re.findall(r"#[0-9a-fA-F]{3,8}\b", _static_qss()) == []


# ── DS-02 / DS-21: painted card shadows ──────────────────────────────────────

def test_cards_get_a_painted_shadow_and_keep_their_effect_slot(make_workspace, monkeypatch):
    get_manager().apply("midnight")
    ws = make_workspace("indexer")
    process_events(20)
    assert ws._cards, "indexer builds its cards with _make_card"
    assert all(c.graphicsEffect() is None for c in ws._cards)
    assert set(map(id, ws._inner.elevated_widgets())) == set(map(id, ws._cards))

    card = ws._cards[0]
    top_left = card.mapTo(ws._inner, QtCore.QPoint(0, 0))
    probe = QtCore.QPoint(top_left.x() + card.width() // 2, top_left.y() + card.height() + 4)
    base = QtGui.QColor(THEMES["midnight"].content_bg)

    def over_page() -> QtGui.QColor:
        # The scroll surface is transparent (the window backdrop shows
        # through), so composite it over the page colour, as on screen.
        shot = ws._inner.grab()
        img = QtGui.QImage(shot.size(), QtGui.QImage.Format.Format_ARGB32)
        img.fill(base)
        p = QtGui.QPainter(img)
        p.drawPixmap(0, 0, shot)
        p.end()
        return img.pixelColor(probe)

    shaded = over_page()
    assert shaded.lightness() < base.lightness(), "no visible shadow under the card"

    import gui.themes.effects as fx
    monkeypatch.setattr(fx, "paint_elevation", lambda *a, **k: None)
    flat = over_page()
    assert flat.name() == base.name()


def test_theme_switch_does_not_rebuild_shadows_in_the_base(make_workspace, monkeypatch):
    """DS-21: the base class no longer rebuilds shadows on theme change (the
    surface repaints itself), so a switch runs refresh_shadows at most once
    per page (from the main window, until the Shell drops that call too)."""
    ws = make_workspace("indexer")
    calls = []
    monkeypatch.setattr(ws, "refresh_shadows", lambda: calls.append(1))
    get_manager().apply("pearl")
    assert calls == []


# ── DS-03 / SHELL-14: no config I/O in paint ─────────────────────────────────

def test_no_config_reads_while_painting(main_window, monkeypatch):
    import config
    import gui.widgets.gradient_bg as gb
    from gui.themes import appearance

    monkeypatch.setattr(appearance, "_mode_cache", None)   # restored after the test
    main_window.repaint()           # warm the cache
    reads = []
    real = config.load_config
    monkeypatch.setattr(config, "load_config", lambda *a, **k: reads.append(1) or real(*a, **k))
    for key in ("indexer", "duplicates", "tools", "player"):
        main_window._on_nav_changed(key)
        main_window.repaint()
        process_events(5)
    assert reads == []

    get_manager().apply("pearl")    # persists the theme (1 read) and drops the cache
    reads.clear()
    for _ in range(3):
        main_window.repaint()
    assert len(reads) <= 1, "the gradient flag is re-read at most once per switch"

    cfg = real()
    cfg["bg_gradient_enabled"] = False
    config.save_config(cfg)
    assert gb.refresh_gradient_setting() is False
    assert gb._gradient_enabled() is False


# ── DS-11: explicit fonts survive polish and theme switches ──────────────────

def test_explicit_fonts_are_not_overridden(qtbot):
    btn = QtWidgets.QPushButton("Big")
    big = QtGui.QFont(btn.font())
    big.setPointSize(20)
    btn.setFont(big)
    edit = QtWidgets.QLineEdit()
    edit.setFont(big)
    plain = QtWidgets.QPushButton("Plain")
    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    for w in (btn, edit, plain):
        lay.addWidget(w)
    qtbot.addWidget(host)
    get_manager().apply("midnight")
    host.show()
    process_events(10)
    assert btn.font().pointSize() == 20 and edit.font().pointSize() == 20
    for key in ("pearl", "ember"):
        get_manager().apply(key)
        process_events(5)
        assert btn.font().pointSize() == 20 and edit.font().pointSize() == 20
    from gui.themes.tokens import TYPE
    assert plain.font().pixelSize() == TYPE["body"].px, "role fonts still apply by default"


# ── DS-24: the type ramp reads top-down ──────────────────────────────────────

def test_type_ramp_is_not_inverted(main_window):
    main_window._on_nav_changed("indexer")
    process_events(10)
    ws = main_window._workspaces["indexer"] if hasattr(main_window, "_workspaces") else None
    root = ws or main_window

    def size(font):
        # The *requested* size in px: QSS roles set pixel sizes (pointSize()
        # is -1 then), TypeScale sets points (1pt = 4/3 px at 96 dpi). Not
        # QFontInfo: it reports -1 when no installed font matches, which is
        # the case on the Windows offscreen platform.
        return font.pixelSize() if font.pixelSize() > 0 else font.pointSizeF() * 4 / 3

    def px(name):
        w = next(w for w in root.findChildren(QtWidgets.QLabel) if w.objectName() == name)
        return size(w.font())

    body = size(QtWidgets.QApplication.font())
    # R2: the page no longer repeats its title in a "sectionTitle" label (the
    # shell header shows it); the hero card's title is the top in-page heading.
    section, card = px("organizeHeroTitle"), px("cardTitle")
    nav = size(main_window.findChild(QtWidgets.QPushButton, "navButton").font())
    assert min(section, card, body, nav) > 0, (section, card, body, nav)
    assert section > card >= body, (section, card, body)
    assert nav < section, (nav, section)


# ── DS-12 / SHELL-18: sidebar and log drawer follow the theme ────────────────

@pytest.mark.parametrize("theme", ["pearl", "sunset", "midnight", "glass"])
def test_sidebar_brand_and_group_labels_use_tokens(qtbot, theme):
    """The brand name and the group headings follow the theme through the
    palette (no inline colours), and stay legible on the sidebar."""
    from gui.widgets.sidebar import Sidebar

    get_manager().apply(theme)
    sb = Sidebar()
    qtbot.addWidget(sb)
    sb.show()
    process_events(5)
    t = THEMES[theme]
    logo = sb._logo
    assert logo.styleSheet() == ""
    colour = logo.palette().color(R.WindowText).name()
    assert colour == t.text_primary
    assert _contrast(colour, t.sidebar_bg) >= 4.5
    for label in sb._labels:
        assert label.styleSheet() == ""
        assert label.palette().color(R.WindowText).name() == t.text_muted


def test_log_drawer_follows_theme(qtbot):
    from gui.widgets.log_drawer import LogDrawer

    get_manager().apply("midnight")
    drawer = LogDrawer()
    qtbot.addWidget(drawer)
    drawer.resize(900, drawer.sizeHint().height())
    drawer.show()
    drawer.append("all good", "ok")
    drawer.append("broken", "error")
    drawer.set_status("Scanning…", "#f59e0b")      # a workspace's Midnight hex
    for key in ("pearl", "meadow", "ember"):
        get_manager().apply(key)
        process_events(5)
        t = THEMES[key]
        assert drawer.status_colour() == t.log_warn
        # The status dot carries the colour; the captions stay muted.
        assert drawer._status_dot.colour().name() == t.log_warn
        assert drawer._status_chip.palette().color(R.WindowText).name() == t.text_secondary
        assert drawer._toggle_icon.palette().color(R.WindowText).name() == t.text_secondary
        # The drawer is one surface (the material's card): opaque in classic themes.
        assert drawer.grab().toImage().pixelColor(drawer.width() // 2, 17).name() == t.card_bg
        doc = drawer._text.document()
        first, second = doc.findBlockByNumber(0), doc.findBlockByNumber(1)
        assert first.begin().fragment().charFormat().foreground().color().name() == t.log_ok
        assert second.begin().fragment().charFormat().foreground().color().name() == t.log_err
    drawer.set_status("Custom", "#123456")           # unknown colours pass through
    assert drawer.status_colour() == "#123456"


# ── DS-15 / DS-16: theme picker ──────────────────────────────────────────────

def test_swatch_names_use_the_swatch_colours(qtbot):
    from gui.themes.picker import _SwatchCard

    get_manager().apply("pearl")
    for key, t in THEMES.items():
        card = _SwatchCard(t)
        qtbot.addWidget(card)
        colour = card.label_color().name()
        assert colour == t.text_primary
        assert _contrast(colour, t.content_bg) >= 4.5, key


def test_auto_card_names_the_night_theme_during_the_day(qtbot, monkeypatch):
    from gui.themes.picker import ThemePickerDialog

    mgr = get_manager()
    mgr.configure_auto("navy", "azure")
    monkeypatch.setattr(ThemeManager, "_is_daytime", staticmethod(lambda: True))
    mgr.apply("auto")
    assert mgr.current.key == "azure"
    dlg = ThemePickerDialog()
    qtbot.addWidget(dlg)
    assert dlg._auto_pair_label.text() == "Night: Navy  ·  Day: Azure"
    mgr.apply("midnight")


# ── DS-10: Auto mode on platforms without a colour scheme ───────────────────

class _Hints:
    def __init__(self, scheme):
        self._scheme = scheme

    def colorScheme(self):  # noqa: N802
        return self._scheme


@pytest.mark.parametrize("hour, expected", [(12, "pearl"), (23, "midnight")])
def test_auto_unknown_scheme_falls_back_to_the_clock(monkeypatch, hour, expected):
    mgr = get_manager()
    mgr.configure_auto("midnight", "pearl")
    monkeypatch.setattr(QtGui.QGuiApplication, "styleHints",
                        staticmethod(lambda: _Hints(QtCore.Qt.ColorScheme.Unknown)))
    monkeypatch.setattr(ThemeManager, "_now",
                        staticmethod(lambda: datetime.datetime(2026, 1, 1, hour, 0)))
    mgr.apply("auto")
    assert mgr.current.key == expected
    mgr.apply("midnight")


def test_auto_rechecks_periodically_and_flips_at_dusk(monkeypatch):
    mgr = get_manager()
    mgr.configure_auto("midnight", "pearl")
    monkeypatch.setattr(QtGui.QGuiApplication, "styleHints",
                        staticmethod(lambda: _Hints(QtCore.Qt.ColorScheme.Unknown)))
    clock = {"h": 19}
    monkeypatch.setattr(ThemeManager, "_now",
                        staticmethod(lambda: datetime.datetime(2026, 1, 1, clock["h"], 59)))
    mgr.apply("auto")
    assert mgr.current.key == "pearl"
    assert mgr._auto_timer.isActive()
    clock["h"] = 20
    mgr._on_auto_tick()
    assert mgr.current.key == "midnight"
    mgr.apply("ember")
    assert not mgr._auto_timer.isActive(), "the re-check timer stops outside Auto"


def test_known_scheme_still_wins(monkeypatch):
    mgr = get_manager()
    mgr.configure_auto("navy", "azure")
    monkeypatch.setattr(QtGui.QGuiApplication, "styleHints",
                        staticmethod(lambda: _Hints(QtCore.Qt.ColorScheme.Dark)))
    monkeypatch.setattr(ThemeManager, "_now",
                        staticmethod(lambda: datetime.datetime(2026, 1, 1, 12, 0)))
    mgr.apply("auto")
    assert mgr.current.key == "navy"
    mgr.apply("midnight")
