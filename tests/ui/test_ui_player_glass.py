"""The Player in Liquid Glass (redesign R2): the capsule is the only
transport, the page is library + queue on glass, and nothing overflows at
the minimum window size.

See docs/redesign/pages/07_player.md ("R2: the bar is the only transport").
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from gui import playback_backend as pb
from gui.widgets.now_playing_bar import NowPlayingBar
from gui.workspaces import player as player_mod
from player_helpers import FakeBackend, make_player_with_bar, make_tracks
from ui_harness import process_events

LEFT = QtCore.Qt.MouseButton.LeftButton


def _player_in(main_window, monkeypatch):
    ws = main_window._workspaces["player"]
    fake = FakeBackend()
    monkeypatch.setattr(ws, "_backend", fake)
    ws._init_backend()
    return ws, fake


# ── The page has no transport of its own ────────────────────────────────────

def test_the_player_page_has_no_transport_of_its_own(qtbot):
    ws, _, _ = make_player_with_bar(qtbot)
    for name in ("_play_btn", "_prev_btn", "_next_btn", "_stop_btn", "_shuffle_btn",
                 "_repeat_btn", "_seek_slider", "_vol_slider"):
        assert not hasattr(ws, name), name
    assert ws.findChildren(QtWidgets.QSlider) == []


def test_shuffle_and_repeat_live_on_the_capsule(qtbot, tmp_path):
    ws, fake, bar = make_player_with_bar(qtbot)
    tracks = make_tracks(tmp_path, 4)
    ws.load_tracks_and_play(tracks)

    qtbot.mouseClick(bar._shuffle_btn, LEFT)
    assert ws.shuffle_enabled() and bar.shuffle()
    assert bar._shuffle_btn.toolTip() == "Shuffle: On"
    qtbot.mouseClick(bar._shuffle_btn, LEFT)
    assert not ws.shuffle_enabled() and not bar.shuffle()

    # Off → Track → All → Off, the order the page's button had.
    seen = []
    for _ in range(3):
        qtbot.mouseClick(bar._repeat_btn, LEFT)
        seen.append((ws.repeat_mode(), bar.repeat(), bar._repeat_btn.badge()))
    assert seen == [("track", "track", "1"), ("all", "all", ""), ("off", "off", "")]

    # Repeat: Track set from the capsule really repeats at the end of a track.
    qtbot.mouseClick(bar._repeat_btn, LEFT)
    fake.finish()
    ws._tick_position()
    assert fake.plays() == [tracks[0], tracks[0]]


def test_the_player_announces_its_modes_to_the_capsule(qtbot):
    ws, _, bar = make_player_with_bar(qtbot)
    ws.toggle_shuffle()
    assert bar.shuffle() and bar._shuffle_btn.is_active()
    ws._cycle_repeat()
    ws._cycle_repeat()
    assert bar.repeat() == "all" and bar._repeat_btn.is_active()


def test_without_a_backend_the_capsule_says_so_and_offers_nothing(qtbot):
    ws, _, bar = make_player_with_bar(qtbot, backend=pb.NullBackend("python-vlc is missing"))
    assert ws._notice.isVisible()
    assert ws._vlc_status_lbl.text() == "Playback unavailable"
    assert "python-vlc is missing" in ws._vlc_reason_lbl.text()
    assert bar.isVisible(), "the Player page pins its transport"
    assert bar._title_lbl.text() == "Playback unavailable"
    for w in (bar._play_btn, bar._prev_btn, bar._next_btn, bar._shuffle_btn,
              bar._repeat_btn, bar._seek_slider):
        assert not w.isEnabled()
    assert "python-vlc is missing" in bar._prev_btn.toolTip()


def test_with_a_backend_the_notice_stays_hidden(qtbot):
    ws, _, bar = make_player_with_bar(qtbot)
    assert not ws._notice.isVisible()
    assert bar._title_lbl.text() == "Nothing playing"
    assert bar._next_btn.isEnabled() and not bar._play_btn.isEnabled()


# ── When the capsule shows ──────────────────────────────────────────────────

def test_the_capsule_is_pinned_on_the_player_page_only(main_window, monkeypatch, tmp_path):
    bar = main_window._now_playing_bar
    assert not bar.isVisible()
    main_window._on_nav_changed("player")
    process_events(10)
    assert bar.isVisible() and bar.is_pinned() and not bar.is_loaded()
    main_window._on_nav_changed("home")
    process_events(10)
    assert not bar.isVisible(), "idle and not on the Player page: hidden"

    ws, _ = _player_in(main_window, monkeypatch)
    ws.load_tracks_and_play(make_tracks(tmp_path, 2), "Album")
    process_events(10)
    assert bar.isVisible(), "once something plays it stays on every page"
    main_window._on_nav_changed("indexer")
    process_events(10)
    assert bar.isVisible()


def test_the_capsule_shuffle_reaches_the_player_in_the_window(main_window, monkeypatch, tmp_path):
    ws, fake = _player_in(main_window, monkeypatch)
    main_window._on_nav_changed("player")
    bar = main_window._now_playing_bar
    ws.load_tracks_and_play(make_tracks(tmp_path, 3), "Album")
    bar._shuffle_btn.click()
    assert ws.shuffle_enabled()
    bar._repeat_btn.click()
    assert ws.repeat_mode() == "track"


# ── Widths: the capsule and the page at the minimum window ──────────────────

def _assert_row_fits(bar: NowPlayingBar) -> None:
    lay = bar.layout()
    lay.activate()
    assert lay.minimumSize().width() <= bar.width()
    inside = bar.rect()
    for child in bar.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly):
        if child.isVisibleTo(bar) and not child.isWindow():
            assert inside.contains(child.geometry()), child


@pytest.mark.parametrize("width,tier", [(880, "full"), (760, "compact"), (600, "narrow")])
def test_capsule_tiers_never_overflow(qtbot, width, tier):
    bar = NowPlayingBar()
    qtbot.addWidget(bar)
    bar.update_now_playing("A rather long track title", "Some Artist", "Some Album", None)
    bar.resize(width, NowPlayingBar.HEIGHT)
    process_events(5)
    assert bar.tier() == tier
    _assert_row_fits(bar)
    # Every control stays reachable in every tier.
    for w in (bar._shuffle_btn, bar._prev_btn, bar._play_btn, bar._next_btn,
              bar._repeat_btn, bar._seek_slider, bar._vol_icon, bar._kb_btn):
        assert w.isVisibleTo(bar)
    assert bar._vol_slider.isVisibleTo(bar) == (tier == "full")
    assert bar._vol_icon.is_interactive() == (tier != "full")
    assert bar._pos_lbl.isVisibleTo(bar) == (tier != "narrow")


def test_compact_capsule_volume_popup_drives_the_volume(qtbot):
    bar = NowPlayingBar()
    qtbot.addWidget(bar)
    bar.resize(700, NowPlayingBar.HEIGHT)
    bar.show()
    seen = []
    bar.volume_changed.connect(seen.append)
    qtbot.mouseClick(bar._vol_icon, LEFT)
    pop = bar._vol_popup
    assert pop is not None and pop.isVisible()
    pop.slider.setValue(33)
    assert seen == [33] and bar._vol_slider.value() == 33
    bar.set_volume(70)                      # player → capsule: the popup follows
    assert pop.slider.value() == 70 and seen == [33]
    pop.hide()


def test_narrow_capsule_puts_the_times_in_the_seek_tooltip(qtbot):
    bar = NowPlayingBar()
    qtbot.addWidget(bar)
    bar.resize(600, NowPlayingBar.HEIGHT)
    bar.show()
    process_events(5)
    assert bar.tier() == "narrow" and not bar._pos_lbl.isVisibleTo(bar)
    bar.update_position(102_000, 245_000)
    assert bar._seek_slider.toolTip() == "1:42 / 4:05"


def test_capsule_fits_the_minimum_window(main_window, monkeypatch, tmp_path):
    main_window.resize(900, 600)
    ws, _ = _player_in(main_window, monkeypatch)
    main_window._on_nav_changed("player")
    ws.load_tracks_and_play(make_tracks(tmp_path, 2), "Album")
    process_events(30)
    bar = main_window._now_playing_bar
    assert bar.isVisible()
    _assert_row_fits(bar)


@pytest.mark.parametrize("size", [(900, 600), (1300, 860), (1600, 1000)])
def test_player_page_fits_without_horizontal_scrolling(main_window, size):
    main_window.resize(*size)
    main_window._on_nav_changed("player")
    process_events(30)
    ws = main_window._workspaces["player"]
    assert ws.minimumSizeHint().width() <= ws.width()
    assert ws._lib_table.horizontalScrollBar().maximum() == 0
    for card in (ws._bg_art, ws._queue_card):
        assert ws.rect().contains(card.mapTo(ws, card.rect().bottomRight())), card
    # The carousel only shows when the column has room for it and the queue.
    side = ws._side
    if ws._np_card.isVisibleTo(side):
        assert ws._queue_card.height() >= ws._queue_card.minimumSizeHint().height()


def test_narrow_table_hides_album_then_number_but_still_searches_them(qtbot, tmp_path):
    ws, _, _ = make_player_with_bar(qtbot)
    rows = [player_mod._load_row(p) for p in make_tracks(tmp_path, 3)]
    rows[1]["album"] = "Hidden Album"
    ws._on_tracks_ready(ws._scan_gen, rows)
    ws._on_scan_complete(ws._scan_gen, len(rows))
    ws.resize(760, 600)
    process_events(10)
    assert ws._lib_table.isColumnHidden(player_mod._COL_ALBUM)
    ws._search.setText("hidden album")
    shown = [r for r in range(ws._lib_table.rowCount()) if not ws._lib_table.isRowHidden(r)]
    assert len(shown) == 1


# ── Look: glass item views, glyph buttons, empty states ─────────────────────

def test_no_inline_sheets_and_glyph_buttons_without_emoji(qtbot):
    ws, _, _ = make_player_with_bar(qtbot)
    assert all(not w.styleSheet() for w in ws.findChildren(QtWidgets.QWidget))
    for btn in (ws._reload_btn, ws._browse_btn, ws._add_queue_btn, ws._load_m3u_btn,
                ws._save_m3u_btn, ws._clear_queue_btn, ws._rp_toggle_btn):
        assert btn.property("glyphName"), btn.toolTip()
        assert btn.text() == "" and btn.toolTip() and btn.accessibleName()
    for btn in ws.findChildren(QtWidgets.QAbstractButton):
        assert not any(ch in btn.text() for ch in "💾📂⟳🔀🔁🔂▶⏮⏭■"), btn.text()


def test_table_rows_are_34px_and_the_playing_row_is_marked(qtbot, tmp_path):
    ws, fake, _ = make_player_with_bar(qtbot)
    paths = make_tracks(tmp_path, 3)
    rows = [player_mod._load_row(p) for p in paths]
    ws._on_tracks_ready(ws._scan_gen, rows)
    ws._on_scan_complete(ws._scan_gen, len(rows))
    t = ws._lib_table
    assert t.verticalHeader().defaultSectionSize() == 34
    assert not t.showGrid()
    assert ws._playing_marker() == ("", False)
    ws._play_directly(paths[1])
    assert ws._playing_marker() == (paths[1], True)
    ws._on_play_pause()
    assert ws._playing_marker() == (paths[1], False)
    # The playing row is painted with the accent tint; the others are not.
    r = next(r for r in range(t.rowCount()) if ws._get_lib_path_at(r) == paths[1])
    other = next(rr for rr in range(t.rowCount()) if rr != r)
    img = t.viewport().grab().toImage()
    rect = t.visualItemRect(t.item(r, 1))
    orect = t.visualItemRect(t.item(other, 1))
    px = QtGui.QColor(img.pixel(rect.right() - 4, rect.center().y()))
    plain = QtGui.QColor(img.pixel(orect.right() - 4, orect.center().y()))
    assert px.rgba() != plain.rgba()


def test_empty_states_before_and_after(qtbot, tmp_path):
    ws, _, _ = make_player_with_bar(qtbot)
    assert ws._lib_empty.isVisible() and ws._lib_empty.title() == "No library selected"
    assert ws._queue_empty.isVisible()
    rows = [player_mod._load_row(p) for p in make_tracks(tmp_path, 2)]
    ws._on_tracks_ready(ws._scan_gen, rows)
    ws._on_scan_complete(ws._scan_gen, len(rows))
    assert not ws._lib_empty.isVisible()
    ws._search.setText("zzz-no-such-track")
    assert ws._lib_empty.isVisible() and ws._lib_empty.title() == "No matches"
    ws._search.clear()
    ws._add_paths_to_queue([rows[0]["path"]])
    assert not ws._queue_empty.isVisible()
    assert ws._save_m3u_btn.isEnabled()


def test_recently_played_toggles_with_a_chevron(qtbot, tmp_path):
    ws, _, _ = make_player_with_bar(qtbot)
    ws._play_directly(make_tracks(tmp_path, 1)[0])
    assert ws._recent_list.height() == 0
    ws._rp_toggle_btn.click()
    assert ws._recent_list.height() > 0
    assert ws._rp_toggle_btn.property("glyphName") == "chevron_down"
    assert ws._rp_hdr.text() == "Recently played  ·  1"
    ws._rp_toggle_btn.click()
    assert ws._recent_list.height() == 0


def test_folder_art_is_used_when_a_track_has_none(qtbot, tmp_path, threads):
    """Libraries keep covers as cover.jpg next to the tracks; the Player
    showed none of them (only embedded art)."""
    img = QtGui.QImage(40, 40, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(200, 30, 30))
    img.save(str(tmp_path / "cover.jpg"))
    track = make_tracks(tmp_path, 1)[0]
    ws, _, bar = make_player_with_bar(qtbot)
    ws.load_tracks_and_play([track])
    qtbot.waitUntil(lambda: not bar._art_lbl.pixmap().isNull(), timeout=5000)
    assert ws._bg_art.has_art
    assert ws._covers._current_pm is not None
    assert threads.wait_all(5000) == []
