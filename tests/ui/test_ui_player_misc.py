"""Player page odds and ends + the Now Playing bar.

Covers PLY-09, 16 (DS-13), 17, 18, 19, 24 (DS-17), 25, 26, 27.
"""
from __future__ import annotations

import os

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from gui.themes.manager import get_manager
from gui.themes.tokens import THEMES
from gui.widgets.now_playing_bar import NowPlayingBar
from gui.workspaces import player as player_mod
from library_factory import TrackSpec, build_library
from player_helpers import FakeBackend, make_player, make_tracks, wait_scan

Act = QtWidgets.QAbstractSlider.SliderAction


def _logs(ws):
    out = []
    ws.log_message.connect(lambda m, lvl: out.append((lvl, m)))
    return out


# ── PLY-09: sliders reach the backend for every kind of input ──────────────

def test_page_volume_slider_keys_and_groove_reach_backend(qtbot):
    ws, fake = make_player(qtbot)
    seen = []
    ws.volume_changed.connect(seen.append)
    ws._vol_slider.triggerAction(Act.SliderSingleStepAdd)     # arrow key / wheel
    assert fake.volume == 81
    ws._vol_slider.triggerAction(Act.SliderPageStepSub)       # groove click / PageDown
    assert fake.volume == 71
    assert seen == [81, 71]
    ws.set_volume(30)                                         # public API
    assert fake.volume == 30 and ws._vol_slider.value() == 30


def test_page_seek_slider_keys_seek_and_ticks_do_not(qtbot, tmp_path):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(make_tracks(tmp_path, 1))
    fake._time = 60_000
    before = fake.count("set_time")
    ws._tick_position()                       # programmatic update: no seek
    assert fake.count("set_time") == before
    assert ws._seek_slider.value() == int(60_000 / fake.length * 1000)
    ws._seek_slider.triggerAction(Act.SliderPageStepAdd)
    assert fake.calls[-1][0] == "set_time"
    assert fake.calls[-1][1] == int(ws._seek_slider.value() / 1000 * fake.length)


def test_bar_sliders_emit_for_keys_and_groove_without_echo(qtbot):
    bar = NowPlayingBar()
    qtbot.addWidget(bar)
    vols, seeks = [], []
    bar.volume_changed.connect(vols.append)
    bar.seek_requested.connect(seeks.append)

    bar.set_volume(50)                        # sync from the page: silent
    assert vols == [] and bar._vol_slider.value() == 50
    bar._vol_slider.triggerAction(Act.SliderSingleStepAdd)
    assert vols == [51]

    bar.update_position(10_000, 100_000)      # position update: silent
    assert seeks == []
    bar._seek_slider.triggerAction(Act.SliderPageStepAdd)
    assert seeks == [int(bar._seek_slider.value() / 1000 * 100_000)]


def test_bar_and_page_volume_stay_in_sync_without_loops(qtbot):
    ws, fake = make_player(qtbot)
    bar = NowPlayingBar()
    qtbot.addWidget(bar)
    # main_window wiring (bar → page) plus the page → bar link it should add
    bar.volume_changed.connect(ws.set_volume)
    ws.volume_changed.connect(bar.set_volume)

    bar._vol_slider.triggerAction(Act.SliderPageStepSub)
    assert ws._vol_slider.value() == bar._vol_slider.value() == 70
    assert fake.volume == 70
    ws._vol_slider.triggerAction(Act.SliderSingleStepAdd)
    assert bar._vol_slider.value() == 71 and fake.volume == 71
    assert fake.count("volume") < 10


# ── PLY-24 / DS-17: keyboard-mode pulse ─────────────────────────────────────

@pytest.mark.motion(time_scale=1.0)
def test_kb_toggle_pulse_does_not_move_the_button(qtbot, motion):
    bar = NowPlayingBar()
    qtbot.addWidget(bar)
    bar.resize(1100, 60)
    bar.show()
    qtbot.waitExposed(bar)
    geo = bar._kb_btn.geometry()
    bar._kb_btn.toggle()
    qtbot.wait(30)
    bar._kb_btn.toggle()                       # second toggle mid-pulse
    assert motion.settle(2000) == []
    assert bar._kb_btn.geometry() == geo
    assert all(r.prop != "geometry" for r in motion.records)
    assert motion.orphans() == []
    assert bar._kb_fx.opacity() == pytest.approx(1.0)


# ── PLY-27 / PLY-14: the bar follows Stop / Clear ───────────────────────────

def test_clear_queue_resets_and_hides_the_bar(main_window, qtbot, tmp_path, monkeypatch, threads):
    ws = main_window._workspaces["player"]
    fake = FakeBackend()
    monkeypatch.setattr(ws, "_backend", fake)
    ws._init_backend()
    bar = main_window._now_playing_bar
    ws.load_tracks_and_play(make_tracks(tmp_path, 2), "Album")
    assert bar.isVisible() and bar._title_lbl.text() == "01 - Track 1"
    assert bar._play_btn.text() == "⏸"
    ws._on_stop()
    assert bar._play_btn.text() == "▶"
    ws._clear_queue()
    assert not bar.isVisible()
    assert bar._title_lbl.text() == "—"
    assert bar._art_lbl.pixmap().isNull()
    assert threads.wait_all(5000) == []


# ── PLY-16 / DS-13: light themes ────────────────────────────────────────────

def test_library_table_follows_a_light_theme(qtbot):
    ws, _ = make_player(qtbot)
    pearl = THEMES["pearl"]
    get_manager().apply("pearl")
    qss = ws._lib_table.styleSheet()
    assert player_mod._rgba(pearl.content_bg, 210) in qss
    assert "13,17,23" not in qss
    assert pearl.text_primary in qss
    assert pearl.card_bg in ws._hover_popup.styleSheet()
    img = ws._bg_art.grab().toImage()
    centre = QtGui.QColor(img.pixel(img.width() // 2, img.height() // 2))
    assert centre.lightness() > 180            # not a black slab


# ── PLY-17: default sort and numeric columns ───────────────────────────────

def test_library_sorted_by_artist_album_track_and_numbers_sort_numerically(
        qtbot, threads, tmp_path):
    lib = build_library(tmp_path / "lib", [
        TrackSpec("Beta", "B10", "Zed", 10, ext=".wav"),
        TrackSpec("Beta", "B2", "Zed", 2, ext=".wav"),
        TrackSpec("Alpha", "A3", "One", 3, ext=".wav"),
        TrackSpec("Alpha", "A1", "One", 1, ext=".wav"),
        TrackSpec(None, None, ext=".wav", relpath="untagged.wav", write_tags=False),
    ])
    ws, _ = make_player(qtbot, library_path=str(lib.root))
    ws._reload_library()
    wait_scan(ws, threads)
    t = ws._lib_table
    titles = [t.item(r, 1).text() for r in range(t.rowCount())]
    assert titles == ["A1", "A3", "B2", "B10", "untagged"]
    t.sortItems(0, QtCore.Qt.SortOrder.AscendingOrder)
    nums = [t.item(r, 0).text() for r in range(t.rowCount())]
    assert nums == ["", "1", "2", "3", "10"]


# ── PLY-18 / PLY-19: M3U ────────────────────────────────────────────────────

def test_load_cp1252_m3u(qtbot, tmp_path, dialogs):
    track = tmp_path / "Café.wav"
    track.write_bytes(b"\0" * 32)
    m3u = tmp_path / "list.m3u"
    m3u.write_bytes("#EXTM3U\nCafé.wav\n".encode("cp1252"))
    ws, fake = make_player(qtbot)
    dialogs.choose_open_file(m3u)
    ws._load_m3u()
    assert [r["path"] for r in ws._queue] == [str(track)]
    assert fake.plays() == [str(track)]


def test_cancelled_m3u_save_creates_no_folder(qtbot, tmp_path, dialogs):
    lib = tmp_path / "lib"
    lib.mkdir()
    ws, _ = make_player(qtbot)
    ws._library_path = str(lib)
    ws._add_paths_to_queue(make_tracks(tmp_path, 2))
    dialogs.answer("getSaveFileName", ("", ""))
    ws._save_m3u()
    assert not (lib / "Playlists").exists()

    dest = lib / "Playlists" / "mine.m3u"
    dialogs.choose_save_file(dest)
    ws._save_m3u()
    assert dest.read_text(encoding="utf-8").startswith("#EXTM3U\n")


# ── PLY-25 / PLY-26 ─────────────────────────────────────────────────────────

def test_replacing_a_queue_is_logged(qtbot, tmp_path):
    ws, _ = make_player(qtbot)
    logs = _logs(ws)
    paths = make_tracks(tmp_path, 3)
    ws._add_paths_to_queue(paths[:2])
    ws.load_tracks_and_play(paths[2:], "Graph selection")
    assert any("Replaced the queue (2 track(s))" in m and "Graph selection" in m
               for _, m in logs)


def test_landing_tile_on_artist_folder_plays_nested_albums(qtbot, tmp_path):
    lib = build_library(tmp_path / "lib", [
        TrackSpec("Art", "One", "First", 1, ext=".wav"),
        TrackSpec("Art", "Two", "Second", 1, ext=".wav"),
    ])
    (lib.root / "Art" / "folder.jpg").write_bytes(b"x")
    ws, fake = make_player(qtbot)
    ws.load_directory_and_play(str(lib.root / "Art"))
    assert len(ws._queue) == 2
    assert fake.plays() == [str(lib.root / "Art" / "First" / "01 - One.wav")]


def test_landing_tile_without_audio_says_so(qtbot, tmp_path):
    empty = tmp_path / "Empty"
    empty.mkdir()
    ws, fake = make_player(qtbot)
    logs = _logs(ws)
    ws.load_directory_and_play(str(empty))
    assert fake.plays() == []
    assert any("No audio files found in Empty" in m for _, m in logs)
    assert "No audio files found" in ws._status_lbl.text()
