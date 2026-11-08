"""Liquid Glass (redesign phase R1): tokens, backdrop, glass surfaces, shell,
landing and the appearance settings.

What these guard, beyond "it paints":

* every theme has a complete material (glass translucent, classic opaque);
* the backdrop is composed in a worker, never on the GUI thread, then only
  blitted from a cache; with no covers it falls back to the gradient;
* the theme migration to Liquid Glass happens exactly once;
* Reduce transparency makes every surface and palette role opaque;
* the floating capsule still drives the Player, floats over the page area,
  and its shadow never takes clicks;
* the sidebar indicator and the section thumb animate a painter value only;
* a page switch stays within a frame-gap budget;
* the landing keeps its behaviour (Initialize, choose a folder, tile play,
  quit) and its hover lift never moves a tile.
"""
from __future__ import annotations

import threading

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from gui.themes import motion as mo
from gui.themes.manager import get_manager
from gui.themes.tokens import (
    ACCENTS, CLASSIC_THEMES, GLASS_KEY, RADIUS, SPACE, SURFACE_NAMES, THEMES, TYPE,
    is_opaque, parse_colour, reduced_transparency, with_accent,
)
from player_helpers import FakeBackend, make_tracks
from shell_helpers import make_album_tree, wait_until, write_config
from ui_harness import FrameGapMeter, process_events

R = QtGui.QPalette.ColorRole


@pytest.fixture(autouse=True)
def _motion_state(monkeypatch):
    monkeypatch.delenv(mo.ENV_VAR, raising=False)
    mo.set_reduced_motion(None)
    mo.refresh_reduced_motion()
    yield
    mo.set_reduced_motion(None)
    mo.refresh_reduced_motion()


@pytest.fixture(autouse=True)
def _fresh_backdrop_caches():
    from gui.widgets import backdrop
    backdrop.clear_caches()
    yield
    backdrop.clear_caches()


# ── Tokens ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", list(THEMES))
def test_every_theme_has_a_complete_material(key):
    t = THEMES[key]
    m = t.material
    assert m is not None, key
    for name in SURFACE_NAMES:
        s = m.surface(name)
        for colour in (s.fill, s.border, s.highlight, s.shadow):
            if colour:
                r, g, b, a = parse_colour(colour)
                assert 0 <= min(r, g, b) and max(r, g, b) <= 255 and 0.0 <= a <= 1.0, (key, name)
        assert s.radius > 0, (key, name)
        if s.shadow:
            assert s.blur > 0, (key, name)
    for colour in (m.hover, m.pressed, m.selection, m.hairline, m.knob, m.track,
                   m.view, m.view_alt, m.backdrop_base, m.backdrop_tone):
        parse_colour(colour)
    assert m.backdrop in ("art", "gradient", "plain")
    if key in CLASSIC_THEMES:
        # Classic: opaque, no highlight, never the album-art wall.
        assert not m.is_glass
        assert all(is_opaque(m.surface(n).fill) and not m.surface(n).highlight
                   for n in SURFACE_NAMES)
        assert m.backdrop != "art"
    else:
        assert m.is_glass and m.backdrop == "art"
        assert not is_opaque(m.card.fill) and m.card.highlight
        # The mockup's numbers.
        assert (m.card.radius, m.card.blur, m.card.offset_y) == (26, 50, 18)
        assert (m.sidebar.radius, m.sidebar.blur, m.sidebar.offset_y) == (30, 70, 24)
        assert (m.capsule.blur, m.capsule.offset_y) == (60, 20)
        assert m.backdrop_scrim == "rgba(8,10,18,0.58)"
        assert (m.wall_columns, m.wall_gap, m.wall_radius, m.blur_px, m.bleed) == (8, 18, 24, 46, 80)


def test_scales_and_accents():
    ramp = [TYPE[r].px for r in ("display", "title", "heading", "body", "caption", "micro")]
    assert ramp == [34, 20, 16, 14, 12, 11]
    assert TYPE["mono"].family == "mono"
    assert (SPACE.gutter, SPACE.page_bottom, SPACE.capsule_bottom) == (18, 104, 22)
    assert (RADIUS.card, RADIUS.hero, RADIUS.sidebar, RADIUS.nav) == (26, 30, 30, 14)
    assert [a.base for a in ACCENTS.values()] == ["#a9c7ff", "#ffb37a", "#9be3b5", "#e8a6ff"]
    peach = with_accent(THEMES[GLASS_KEY], "peach")
    assert peach.accent == "#ffb37a" and peach.key == GLASS_KEY
    assert with_accent(THEMES["midnight"], "peach") is THEMES["midnight"], "classic keeps its accent"


def test_reduce_transparency_flattens_every_surface():
    flat = reduced_transparency(THEMES[GLASS_KEY])
    m = flat.material
    assert all(is_opaque(m.surface(n).fill) for n in SURFACE_NAMES)
    assert all(is_opaque(c) for c in (m.hover, m.selection, m.view, m.view_alt, m.track))
    assert reduced_transparency(THEMES["pearl"]) is THEMES["pearl"]


def test_reduce_transparency_gives_an_opaque_palette_and_persists(qtbot):
    import config

    mgr = get_manager()
    mgr.apply(GLASS_KEY)
    assert QtWidgets.QApplication.palette().color(R.Button).alpha() < 255
    mgr.set_reduce_transparency(True)
    try:
        assert config.load_config()["reduce_transparency"] is True
        pal = QtWidgets.QApplication.palette()
        for role in (R.Window, R.Base, R.Button, R.AlternateBase, R.Highlight, R.Mid, R.Midlight):
            assert pal.color(role).alpha() == 255, role
        from gui.widgets.glass import GlassCard
        card = GlassCard()
        card.resize(200, 120)
        qtbot.addWidget(card)
        img = QtGui.QImage(card.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(QtCore.Qt.GlobalColor.transparent)
        card.render(img, QtCore.QPoint(), QtGui.QRegion(), QtWidgets.QWidget.RenderFlag.DrawChildren)
        assert img.pixelColor(100, 60).alpha() == 255, "the card shows nothing through"
    finally:
        mgr.set_reduce_transparency(False)
    assert QtWidgets.QApplication.palette().color(R.Button).alpha() < 255


# ── Migration ─────────────────────────────────────────────────────────────────

def test_theme_migration_happens_once(qtbot):
    import config
    from gui.themes import manager as mgr_mod

    write_config(theme="midnight")
    mgr = get_manager()
    mgr.load_persisted()
    cfg = config.load_config()
    assert mgr.current.key == GLASS_KEY
    assert cfg["theme"] == GLASS_KEY
    assert cfg[mgr_mod.MIGRATION_MARKER] is True
    assert cfg[mgr_mod.PREVIOUS_THEME_KEY] == "midnight"
    note = mgr.take_migration_note()
    assert note and "Midnight" in note and "Liquid Glass" in note
    assert mgr.take_migration_note() is None, "the note is logged once"

    # Choosing a classic theme afterwards sticks: no second migration.
    mgr.apply("midnight")
    mgr.load_persisted()
    assert mgr.current.key == "midnight"
    assert mgr.take_migration_note() is None


def test_fresh_config_gets_glass_and_the_marker_without_a_note():
    import config
    from gui.themes import manager as mgr_mod

    mgr = get_manager()
    mgr.load_persisted()
    assert mgr.current.key == GLASS_KEY
    assert config.load_config()[mgr_mod.MIGRATION_MARKER] is True
    assert mgr.take_migration_note() is None


def test_main_window_logs_the_migration(qtbot):
    from gui.main_window import AlphaDEXWindow

    write_config(theme="ember")
    window = AlphaDEXWindow()
    qtbot.addWidget(window)
    text = window._log_drawer._text.toPlainText()
    assert "Liquid Glass" in text and "Ember" in text


# ── Backdrop ──────────────────────────────────────────────────────────────────

def _host_backdrop(qtbot, library: str = ""):
    from gui.widgets.backdrop import Backdrop

    bd = Backdrop()
    qtbot.addWidget(bd)
    bd.resize(900, 600)
    if library:
        bd.set_library(library)
    bd.show()
    return bd


def _cover_library(root, n: int = 6) -> str:
    root.mkdir(parents=True, exist_ok=True)
    colours = ("red", "green", "blue", "yellow", "magenta", "cyan")
    for i in range(n):
        d = root / f"Artist {i}" / f"Album {i}"
        d.mkdir(parents=True)
        img = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB32)
        img.fill(QtGui.QColor(colours[i % len(colours)]))
        assert img.save(str(d / "cover.png"))
    return str(root)


def test_backdrop_is_composed_in_a_quiet_worker_never_on_the_gui_thread(qtbot, threads,
                                                                         monkeypatch, tmp_path):
    from gui.widgets import backdrop
    from gui.workers import job_registry

    get_manager().apply(GLASS_KEY)
    gui_thread = threading.get_ident()
    calls = []
    real = backdrop.compose

    def spy(spec, covers):
        calls.append((threading.get_ident(), spec.mode, len(covers or [])))
        return real(spec, covers)

    monkeypatch.setattr(backdrop, "compose", spy)
    lib = _cover_library(tmp_path / "Lib")
    bd = _host_backdrop(qtbot, lib)
    assert wait_until(lambda: bd.is_building(), 3000) or calls
    assert "Backdrop" not in [j.name for j in job_registry().listed_jobs()], "housekeeping is quiet"
    assert threads.wait_all(15_000) == []
    assert wait_until(lambda: bd.pixmap() is not None, 5000)
    assert calls and all(tid != gui_thread for tid, _m, _n in calls)
    assert calls[-1][1] == "art" and calls[-1][2] >= 1, "the wall used the library's covers"
    assert bd.spec().library == lib
    assert "BackdropWorker" in threads.started


def test_backdrop_repaints_from_the_cached_pixmap_only(qtbot, threads, monkeypatch, tmp_path):
    import config
    from gui.widgets import backdrop

    get_manager().apply(GLASS_KEY)
    bd = _host_backdrop(qtbot, _cover_library(tmp_path / "Lib"))
    assert threads.wait_all(15_000) == []
    assert wait_until(lambda: bd.pixmap() is not None, 5000)
    shown = bd.pixmap()
    started = list(threads.started)

    def boom(*_a, **_k):
        raise AssertionError("composed again")

    monkeypatch.setattr(backdrop, "compose", boom)
    reads = []
    real = config.load_config
    monkeypatch.setattr(config, "load_config", lambda *a, **k: reads.append(1) or real(*a, **k))
    for _ in range(3):
        bd.repaint()
        bd.grab()
    bd.resize(910, 610)               # same size bucket: nothing to rebuild
    process_events(bd.DEBOUNCE_MS + 60)
    bd.rebuild()
    assert bd.pixmap() is shown
    assert threads.started == started, "no worker for a repaint or an in-bucket resize"
    assert reads == [], "painting never reads the config"

    # A new size bucket builds once; going back is straight from the cache.
    monkeypatch.setattr(backdrop, "compose", lambda spec, covers: QtGui.QImage(
        8, 8, QtGui.QImage.Format.Format_RGB32))
    bd.resize(1400, 900)
    process_events(bd.DEBOUNCE_MS + 60)
    assert threads.wait_all(15_000) == []
    assert wait_until(lambda: bd.pixmap() is not shown, 3000)
    monkeypatch.setattr(backdrop, "compose", boom)
    started = list(threads.started)
    bd.resize(900, 600)
    bd.rebuild()
    assert bd.pixmap() is shown
    assert threads.started == started


def test_backdrop_falls_back_to_the_gradient_then_plain(qtbot, threads, tmp_path):
    from gui.themes import appearance
    from gui.widgets import backdrop

    get_manager().apply(GLASS_KEY)
    empty = tmp_path / "Empty"
    empty.mkdir()
    bd = _host_backdrop(qtbot, str(empty))
    assert threads.wait_all(15_000) == []
    assert wait_until(lambda: bd.pixmap() is not None, 5000)
    spec = bd.spec()
    assert spec.mode == "art"
    gradient = backdrop.compose(backdrop.BackdropSpec(**{**spec.__dict__, "mode": "gradient",
                                                          "library": ""}), None)
    assert bd.pixmap().toImage().pixelColor(40, 40) == \
        QtGui.QPixmap.fromImage(gradient).toImage().pixelColor(40, 40), "no covers: the gradient"

    # Plain: no image at all, the base colour.
    write_config(backdrop_mode="plain")
    appearance.refresh()
    bd.on_appearance_changed()
    process_events(30)
    assert bd.spec().mode == "plain" and bd.pixmap() is None
    px = bd.grab().toImage().pixelColor(10, 10)
    assert px.name() == THEMES[GLASS_KEY].material.backdrop_base

    # Classic themes have no album-art wall: "art" shows their gradient.
    write_config(backdrop_mode="art")
    appearance.refresh()
    get_manager().apply("midnight")
    process_events(30)
    assert bd.desired_spec().mode == "gradient"
    assert threads.wait_all(15_000) == []


def test_legacy_gradient_switch_still_works():
    from gui.themes import appearance

    assert appearance.mode_from_config({}) == "art"
    assert appearance.mode_from_config({"bg_gradient_enabled": False}) == "plain"
    assert appearance.mode_from_config({"bg_gradient_enabled": True}) == "art"
    assert appearance.mode_from_config({"backdrop_mode": "gradient",
                                        "bg_gradient_enabled": False}) == "gradient"
    assert appearance.config_values("plain") == {"backdrop_mode": "plain",
                                                 "bg_gradient_enabled": False}


def test_blur_is_a_close_gaussian():
    import numpy as np
    from gui.imaging import gaussian_blur

    a = np.zeros((101, 101, 1), dtype=np.float32)
    a[50, 50, 0] = 1000.0
    out = gaussian_blur(a, 6.0)
    assert out.sum() == pytest.approx(1000.0, rel=1e-3), "energy is preserved"
    row = out[50, :, 0]
    # Spread of a Gaussian: sigma from the second moment.
    xs = np.arange(101) - 50
    sigma = float(np.sqrt((row * xs * xs).sum() / row.sum()))
    assert sigma == pytest.approx(6.0, abs=0.6)


# ── Glass surfaces ────────────────────────────────────────────────────────────

def test_cards_are_glass_and_the_backdrop_shows_through(main_window):
    get_manager().apply(GLASS_KEY)
    main_window._on_nav_changed("indexer")
    process_events(20)
    ws = main_window._workspaces["indexer"]
    from gui.widgets.glass import GlassCard
    assert ws._cards and all(isinstance(c, GlassCard) for c in ws._cards)
    scroll = ws._scroll_area
    assert not scroll.viewport().autoFillBackground()
    assert not ws._inner.autoFillBackground()
    # Translucent: the card's own paint lets what is behind it through.
    card = ws._cards[0]
    img = QtGui.QImage(card.size(), QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    card.render(img, QtCore.QPoint(), QtGui.QRegion(), QtWidgets.QWidget.RenderFlag(0))
    alpha = img.pixelColor(card.width() // 2, card.height() - 4).alpha()
    assert 0 < alpha < 255


def test_shadow_is_never_painted_under_its_surface():
    from gui.themes.glass import paint_shadow
    from gui.themes.tokens import Surface

    img = QtGui.QImage(300, 300, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(img)
    paint_shadow(p, QtCore.QRectF(100, 100, 100, 100),
                 Surface("", shadow="rgba(0,0,0,0.5)", blur=40, offset_y=18, radius=20))
    p.end()
    assert img.pixelColor(150, 150).alpha() == 0, "inside: like a CSS box-shadow"
    assert img.pixelColor(150, 212).alpha() > 0, "below: the shadow"


# ── Shell ─────────────────────────────────────────────────────────────────────

def test_shell_is_glass_over_the_backdrop(main_window):
    from gui.widgets.backdrop import Backdrop

    assert isinstance(main_window.centralWidget(), Backdrop)
    assert not main_window.statusBar().isVisible()
    sidebar = main_window._sidebar
    assert 220 <= sidebar.width() <= 250
    assert sidebar.geometry().topLeft() == QtCore.QPoint(SPACE.gutter, SPACE.gutter)
    main_window._on_nav_changed("duplicates/compare")
    assert main_window._top_bar.title() == "Duplicates"


@pytest.mark.motion(time_scale=1.0)
def test_sidebar_indicator_is_a_parented_painter_slide_that_settles(qtbot, motion):
    from gui.widgets.sidebar import Sidebar

    sb = Sidebar()
    qtbot.addWidget(sb)
    sb.resize(250, 860)
    sb.show()
    process_events(30)
    geometries = {k: b.geometry() for k, b in sb._buttons.items()}
    for key in ("duplicates", "player", "settings"):
        sb.activate(key)
    anim = sb._nav_container._pill_anim
    assert anim.parent() is sb._nav_container
    assert anim.duration() == mo.SLOW
    assert motion.settle(3000) == []
    btn = sb._buttons["settings"]
    assert sb._nav_container._pill_y == pytest.approx(float(btn.y()))
    assert {k: b.geometry() for k, b in sb._buttons.items()} == geometries
    assert all(r.prop not in ("geometry", "pos", "size") for r in motion.records)
    assert motion.orphans() == []


@pytest.mark.motion(time_scale=1.0)
def test_section_thumb_slides_by_a_painter_value(main_window, motion):
    page = main_window._pages["duplicates"]
    main_window._on_nav_changed("duplicates/find")
    process_events(20)
    tabs = page.tab_bar()
    from gui.pages import SegmentedTabBar
    assert isinstance(tabs, SegmentedTabBar)
    before = tabs.geometry()
    tabs.setCurrentIndex(2)
    anim = tabs._thumb_anim
    assert anim is not None and anim.parent() is tabs and anim.duration() == mo.BASE
    process_events(60)
    mid = tabs.thumb_rect()
    assert motion.settle(3000) == []
    process_events(10)
    assert tabs.thumb_rect() == tabs._segment(2)
    assert mid is not None and mid != tabs._segment(2)
    assert tabs.geometry() == before
    assert page.current_section() == "pair_review"


def _player_with_fake(main_window, monkeypatch):
    ws = main_window._workspaces["player"]
    fake = FakeBackend()
    monkeypatch.setattr(ws, "_backend", fake)
    ws._init_backend()
    return ws, fake


def test_capsule_drives_the_player_and_floats_over_the_pages(main_window, monkeypatch, tmp_path,
                                                             threads, qtbot):
    ws, fake = _player_with_fake(main_window, monkeypatch)
    bar = main_window._now_playing_bar
    assert not bar.isVisible(), "hidden until something plays"
    main_window._on_nav_changed("home")
    ws.load_tracks_and_play(make_tracks(tmp_path, 3), "Album")
    process_events(20)
    assert bar.isVisible()
    assert bar._play_btn.text() == "⏸"

    # Floating over the page area, bottom centre, 22 px up, at most 880 wide.
    stack = main_window._stack
    area = QtCore.QRect(stack.mapTo(main_window._backdrop, QtCore.QPoint(0, 0)), stack.size())
    geo = bar.geometry()
    assert geo.width() <= 880 and area.contains(geo)
    assert abs(geo.center().x() - area.center().x()) <= 1
    assert area.bottom() + 1 - geo.bottom() - 1 == SPACE.capsule_bottom
    shadow = bar._shadow
    assert shadow is not None and shadow.isVisible()
    assert shadow.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    first = fake.plays()[-1]
    qtbot.mouseClick(bar._next_btn, QtCore.Qt.MouseButton.LeftButton)
    assert fake.plays()[-1] != first
    qtbot.mouseClick(bar._prev_btn, QtCore.Qt.MouseButton.LeftButton)
    assert fake.plays()[-1] == first
    pauses = fake.count("pause")
    qtbot.mouseClick(bar._play_btn, QtCore.Qt.MouseButton.LeftButton)
    assert fake.count("pause") == pauses + 1
    assert bar._play_btn.text() == "▶"

    bar.update_position(10_000, 100_000)
    bar._seek_slider.triggerAction(QtWidgets.QAbstractSlider.SliderAction.SliderPageStepAdd)
    assert fake.calls[-1][0] == "set_time"
    bar._vol_slider.setValue(42)
    assert ws._vol_slider.value() == 42 and fake.calls[-1] == ("volume", 42)

    ws._clear_queue()
    assert not bar.isVisible() and not shadow.isVisible()
    assert threads.wait_all(5000) == []


def test_pages_keep_room_for_the_capsule(main_window):
    for key in ("home", "indexer", "player"):
        ws = main_window._workspaces[key]
        layout = ws.layout() if key == "player" else ws.content_layout
        assert layout.contentsMargins().bottom() >= SPACE.page_bottom, key


def test_header_search_filters_the_player(main_window, monkeypatch):
    seen = []
    player = main_window._workspaces["player"]
    monkeypatch.setattr(player, "_apply_filter", lambda text: seen.append(text))
    player._search.textChanged.disconnect()
    player._search.textChanged.connect(player._apply_filter)
    search = main_window._top_bar.search_field()
    search.setText("aurora")
    search.returnPressed.emit()
    assert main_window.current_panel() is player
    assert player._search.text() == "aurora"
    assert seen[-1] == "aurora"


@pytest.mark.motion(time_scale=1.0)
def test_page_switch_stays_within_the_frame_budget(main_window, motion):
    """Page transitions fade the page in over the backdrop: the switch itself
    must not freeze the window (offscreen raster, so the budget is generous
    for CI; a real GPU-composited desktop is much faster)."""
    keys = ["home", "indexer", "duplicates", "settings", "help", "exports"]
    for key in keys:                      # first visits build lazy content
        main_window._on_nav_changed(key)
        motion.settle(2000)
    worst = 0.0
    for key in keys * 2:
        with FrameGapMeter() as meter:
            main_window._on_nav_changed(key)
            process_events(260)
        worst = max(worst, meter.max_gap_ms)
    print(f"\n[perf] page switch worst gap: {worst:.0f} ms")
    assert worst < 150      # measured ~25-30 ms offscreen; generous for slow CI
    assert motion.orphans() == []
    page = main_window.current_page()
    assert motion.settle(2000) == []
    eff = page.graphicsEffect()
    assert eff is None or (not eff.isEnabled() and eff.opacity() == 1.0), "ends fully opaque"


def test_reduced_motion_switches_pages_without_a_fade(main_window):
    mo.set_reduced_motion(True)
    main_window._on_nav_changed("indexer")
    main_window._on_nav_changed("home")
    page = main_window.current_page()
    eff = page.graphicsEffect()
    assert eff is None or not eff.isEnabled()


# ── Landing ───────────────────────────────────────────────────────────────────

GEO = QtCore.QRect(0, 0, 1300, 860)


@pytest.fixture
def landing_factory(qtbot):
    from gui.widgets.landing import MosaicLanding
    made = []

    def make(saved: str = "", missing: str = ""):
        landing = MosaicLanding(GEO, saved, missing_path=missing)
        qtbot.addWidget(landing)
        made.append(landing)
        return landing

    yield make
    for landing in made:
        landing._stop_scanner()


def test_landing_initialize_choose_another_folder_and_quit(landing_factory, tmp_path, dialogs,
                                                         threads, motion):
    lib = make_album_tree(tmp_path / "Lib")
    landing = landing_factory(saved=lib)
    landing.show()
    card = landing._cta
    assert card.main_button.text() == "Initialize"
    assert card.other_button.text() == "Choose another folder"
    assert card.tagline.text() == "Your library is ready."

    other = tmp_path / "Other"
    other.mkdir()
    selected = []
    landing.library_selected.connect(selected.append)
    dialogs.accept_file_dialog(other)
    card.other_button.click()
    motion.settle(3000)
    assert wait_until(lambda: bool(selected), 3000)
    assert selected == [str(other)]
    assert threads.wait_all(10_000) == []

    again = landing_factory(saved=lib)
    again.show()
    chosen = []
    again.library_selected.connect(chosen.append)
    again._cta.main_button.click()
    motion.settle(3000)
    assert wait_until(lambda: bool(chosen), 3000)
    assert chosen == [lib]

    third = landing_factory(saved=lib)
    third.show()
    quits = []
    third.quit_requested.connect(lambda: quits.append(1))
    third._quit_btn.click()
    assert quits == [1] and not third.isVisible()
    assert threads.wait_all(10_000) == []


def test_landing_without_a_library_offers_the_folder_choice(landing_factory):
    landing = landing_factory()
    assert landing._cta.main_button.text() == "Choose Library Folder"
    assert landing._cta.other_button is None and landing._cta.folder_row is None
    assert landing._scanner is None


def test_landing_tile_click_plays_and_counts_the_library(landing_factory, tmp_path, threads,
                                                        motion):
    lib = make_album_tree(tmp_path / "Lib", artists=2, albums=1, tracks=3)
    landing = landing_factory(saved=lib)
    landing.show()
    landing._start_counter()
    assert threads.wait_all(10_000) == []
    process_events(20)
    assert landing._cta.folder_row.stats_label.text() == "6 tracks · 2 artists"

    tile = landing._tiles[0]
    tile.set_dirpath(str(tmp_path / "Lib" / "Artist0" / "Album0"))
    clicked, selected = [], []
    landing.tile_clicked.connect(clicked.append)
    landing.library_selected.connect(selected.append)
    landing._on_tile_clicked(tile._dirpath)
    motion.settle(3000)
    assert wait_until(lambda: bool(selected), 3000)
    assert clicked == [tile._dirpath] and selected == [lib]


@pytest.mark.motion(time_scale=1.0)
def test_landing_hover_lifts_without_moving_the_tile(landing_factory, motion):
    landing = landing_factory()
    landing.show()
    for tile in landing._tiles:
        tile.move(tile._target)
    tile = landing._tiles[3]
    tile.set_info("Hymns", "Cobalt Choir")
    geo = tile.geometry()
    landing._on_tile_hovered(tile, True)
    lift = landing._lift
    assert lift.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert motion.settle(2000) == []
    assert lift.isVisible() and lift.progress() == pytest.approx(1.0)
    lift.grab()                                   # paints (scaled cover, ring, chip)
    landing._on_tile_hovered(tile, False)
    assert motion.settle(2000) == []
    assert not lift.isVisible() and lift.tile() is None
    assert tile.geometry() == geo
    assert all(r.prop not in ("geometry", "pos", "size") for r in motion.records)
    assert motion.orphans() == []


def test_landing_reduced_motion_skips_the_fly_in_and_scatter(landing_factory, threads):
    mo.set_reduced_motion(True)
    landing = landing_factory()
    landing.show()
    landing._fly_in()
    assert all(t.pos() == t._target for t in landing._tiles)
    selected = []
    landing.library_selected.connect(selected.append)
    landing._accept("/lib")
    assert selected == ["/lib"] and landing._fly_out_grp is None


def test_startup_quit_from_the_landing(qtbot, monkeypatch, threads):
    import alpha_dex_gui

    app = QtWidgets.QApplication.instance()
    quits = []
    monkeypatch.setattr(app, "quit", lambda: quits.append(1))
    startup = alpha_dex_gui.Startup(app, geometry=GEO, saved_library="",
                                    window_factory=lambda: QtWidgets.QWidget())
    qtbot.addWidget(startup.landing)
    startup.landing.show()
    startup.landing._quit_btn.click()
    assert quits == [1]
    assert startup.window is None, "quitting never builds the main window"


# ── Appearance settings ───────────────────────────────────────────────────────

def test_appearance_accents_and_transparency_apply_instantly(make_workspace):
    import config

    ws = make_workspace("settings")
    picker = ws.theme_picker
    get_manager().apply("midnight")
    picker._accent_buttons["mint"].click()
    mgr = get_manager()
    assert mgr.current.key == GLASS_KEY, "an accent belongs to Liquid Glass"
    assert mgr.current.accent == ACCENTS["mint"].base
    assert config.load_config()["glass_accent"] == "mint"
    assert picker._accent_buttons["mint"].isChecked()
    picker._reduce_transparency_cb.setChecked(True)
    assert mgr.reduce_transparency() and is_opaque(mgr.current.material.card.fill)
    picker._reduce_transparency_cb.setChecked(False)
    assert not is_opaque(mgr.current.material.card.fill)
    keys = [sw.tokens.key for sw in picker._all_swatches]
    assert keys[0] == GLASS_KEY and set(keys[1:]) == set(CLASSIC_THEMES)


def test_accent_survives_a_restart(qtbot):
    from gui.main_window import AlphaDEXWindow

    mgr = get_manager()
    mgr.apply(GLASS_KEY)
    mgr.set_accent("lilac")
    mgr._accent = "blue"                      # forget it, as a new process would
    window = AlphaDEXWindow()                 # load_persisted reads it back
    qtbot.addWidget(window)
    assert mgr.current.accent == ACCENTS["lilac"].base


def test_reduce_motion_setting_reaches_the_motion_switch(main_window):
    settings = main_window._workspaces["settings"]
    main_window._on_nav_changed("settings/appearance")
    settings._reduce_motion_cb.setChecked(True)
    assert settings._on_save()
    assert mo.reduced_motion() is True
    settings._reduce_motion_cb.setChecked(False)
    assert settings._on_save()
    assert mo.reduced_motion() is False
