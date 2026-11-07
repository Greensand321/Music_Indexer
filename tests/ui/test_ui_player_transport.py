"""Player transport + queue logic against a fake playback backend.

Covers PLY-01, 02, 03, 04, 05, 06, 07, 08, 20, 21 (docs/redesign/pages/07_player.md).
"""
from __future__ import annotations

import sys

import pytest

from gui import playback_backend as pb
from gui.playback_backend import PlaybackState
from gui.workspaces import player as player_mod
from player_helpers import FakeBackend, make_player, make_tracks


@pytest.fixture
def tracks(tmp_path):
    return make_tracks(tmp_path, 4)


def _logs(ws):
    out = []
    ws.log_message.connect(lambda m, lvl: out.append((lvl, m)))
    return out


# ── PLY-01 / PLY-02: no backend ─────────────────────────────────────────────

def test_window_builds_when_libvlc_is_missing(qtbot, monkeypatch, tmp_path, threads):
    """python-vlc present but libVLC missing: import raises NotImplementedError."""
    (tmp_path / "vlc.py").write_text("raise NotImplementedError('Cannot find libvlc lib')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "vlc", raising=False)
    from gui.main_window import AlphaDEXWindow

    window = AlphaDEXWindow()          # used to raise from PlayerWorkspace._init_vlc
    qtbot.addWidget(window)
    ws = window._workspaces["player"]
    assert not ws.backend.available
    assert "Cannot find libvlc lib" in ws.backend.unavailable_reason
    assert ws._vlc_status_lbl.text() == "Playback unavailable"
    window.close()
    monkeypatch.delitem(sys.modules, "vlc", raising=False)


def test_no_backend_never_shows_a_fake_playing_state(qtbot, tracks):
    ws, _ = make_player(qtbot, backend=pb.NullBackend("libVLC not found"))
    states = []
    ws.playback_state_changed.connect(states.append)
    now = []
    ws.now_playing_changed.connect(lambda *a: now.append(a))

    ws.load_tracks_and_play(tracks, "Album")
    ws._play_directly(tracks[1])
    ws._on_play_pause()

    assert True not in states
    assert now == []                              # the bar never appears
    assert ws._play_btn.text() == "▶"
    assert not ws._play_btn.isEnabled()
    assert not ws._pos_timer.isActive()
    assert "Playback unavailable" in ws._status_lbl.text()
    assert "libVLC not found" in ws._status_lbl.text()
    assert len(ws._queue) == 4                    # the queue itself still works


# ── PLY-03: build a queue, then Play ────────────────────────────────────────

def test_add_to_queue_then_play_plays_first_item(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws._add_paths_to_queue(tracks[:3])
    assert ws._play_btn.isEnabled()
    ws._on_play_pause()
    assert fake.plays() == [tracks[0]]
    assert ws._play_btn.text() == "⏸"
    assert ws._np_title.text() == "01 - Track 1"


def test_play_after_stop_plays_the_new_queue_not_old_media(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws._play_directly(tracks[3])
    ws._on_stop()
    ws._add_paths_to_queue(tracks[:2])
    ws._on_play_pause()
    assert fake.plays() == [tracks[3], tracks[0]]


def test_stop_then_play_replays_a_direct_track_when_queue_empty(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws._play_directly(tracks[2])
    ws._on_stop()
    ws._on_play_pause()
    assert fake.plays() == [tracks[2], tracks[2]]


def test_pause_resume_does_not_reload(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(tracks)
    ws._on_play_pause()
    assert fake.state() is PlaybackState.PAUSED
    ws._on_play_pause()
    assert fake.state() is PlaybackState.PLAYING
    assert fake.count("load") == 1


# ── PLY-04: queue item 0 is not skipped ─────────────────────────────────────

def test_direct_play_ending_starts_unstarted_queue_at_item_zero(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws._add_paths_to_queue(tracks[:3])
    ws._play_directly(tracks[3])
    fake.finish()
    ws._tick_position()
    assert fake.plays() == [tracks[3], tracks[0]]


def test_direct_play_mid_queue_resumes_after_the_interrupted_item(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(tracks[:3])       # plays 0
    ws.play_next()                            # plays 1
    ws._play_directly(tracks[3])
    fake.finish()
    ws._tick_position()
    assert fake.plays() == [tracks[0], tracks[1], tracks[3], tracks[2]]


def test_queue_end_stops_and_rewinds(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(tracks[:2])
    fake.finish(); ws._tick_position()
    fake.finish(); ws._tick_position()
    assert fake.plays() == [tracks[0], tracks[1]]
    assert ws._play_btn.text() == "▶"
    ws._on_play_pause()                       # starts over
    assert fake.plays()[-1] == tracks[0]


# ── PLY-05: removing the playing item ───────────────────────────────────────

def test_remove_playing_item_keeps_state_consistent(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(tracks)           # plays 0
    ws.play_next()                            # plays 1
    ws._remove_from_queue(1)
    # Still playing the removed track, now outside the queue; nothing in the
    # queue is highlighted as playing.
    assert ws._current_playing_row()["path"] == tracks[1]
    assert ws._is_currently_playing(tracks[1])
    bold = [ws._queue_list.item(i).font().bold() for i in range(ws._queue_list.count())]
    assert not any(bold)
    # When it ends, the track that slid into its slot plays.
    fake.finish()
    ws._tick_position()
    assert fake.plays()[-1] == tracks[2]


def test_remove_before_cursor_shifts_cursor_and_shuffle_history(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(tracks)
    ws._jump_queue(2)
    ws._shuffle_history = [0, 3, 2]
    ws._shuffle_next_idx = 3
    ws._remove_from_queue(0)
    assert ws._queue_index == 1
    assert ws._queue[ws._queue_index]["path"] == tracks[2]
    assert ws._shuffle_history == [2, 1]
    assert ws._shuffle_next_idx == 2


# ── PLY-06: the 30 s preview timer pauses with playback ─────────────────────

def test_preview_timer_pauses_and_resumes(qtbot, tracks, monkeypatch):
    monkeypatch.setattr(player_mod, "_PREVIEW_SEEK_DELAY_MS", 1)
    monkeypatch.setattr(player_mod, "_HIGHLIGHT_DURATION_MS", 300)
    ws, fake = make_player(qtbot)
    ws._highlight_chk.setChecked(True)
    ws.load_tracks_and_play(tracks[:2])
    qtbot.waitUntil(ws._highlight_timer.isActive, timeout=2000)
    assert ("set_time", player_mod._HIGHLIGHT_START_MS) in fake.calls

    ws._on_play_pause()                       # pause
    assert not ws._highlight_timer.isActive()
    qtbot.wait(450)
    assert fake.plays() == [tracks[0]]        # did not advance while paused

    ws._on_play_pause()                       # resume
    assert ws._highlight_timer.isActive()
    qtbot.waitUntil(lambda: fake.plays() == [tracks[0], tracks[1]], timeout=2000)


def test_stop_cancels_pending_preview(qtbot, tracks, monkeypatch):
    monkeypatch.setattr(player_mod, "_PREVIEW_SEEK_DELAY_MS", 1)
    monkeypatch.setattr(player_mod, "_HIGHLIGHT_DURATION_MS", 50)
    ws, fake = make_player(qtbot)
    ws._highlight_chk.setChecked(True)
    ws.load_tracks_and_play(tracks[:2])
    qtbot.waitUntil(ws._highlight_timer.isActive, timeout=2000)
    ws._on_stop()
    qtbot.wait(150)
    assert fake.plays() == [tracks[0]]


# ── PLY-07: keyboard ↓ library preview advances through the library ─────────

def test_library_preview_advances_to_next_library_row(qtbot, tracks, monkeypatch):
    monkeypatch.setattr(player_mod, "_PREVIEW_SEEK_DELAY_MS", 1)
    monkeypatch.setattr(player_mod, "_HIGHLIGHT_DURATION_MS", 30)
    ws, fake = make_player(qtbot)
    rows = [player_mod._load_row(p) for p in tracks]
    ws._on_tracks_ready(ws._scan_gen, rows)
    ws._on_scan_complete(ws._scan_gen, len(rows))
    ws._add_paths_to_queue([tracks[3]])      # a queue exists, must not be used

    ws.kb_lib_down()
    first = fake.plays()[-1]
    assert first == ws._get_lib_path_at(0)
    qtbot.waitUntil(lambda: len(fake.plays()) == 2, timeout=2000)
    assert fake.plays()[-1] == ws._get_lib_path_at(1)
    assert ws._direct_row is not None        # still library (direct) play


# ── PLY-08: undecodable files ───────────────────────────────────────────────

def test_error_state_is_reported_and_skipped(qtbot, tracks):
    ws, fake = make_player(qtbot)
    logs = _logs(ws)
    ws.load_tracks_and_play(tracks[:3])
    fake.fail()
    ws._tick_position()
    assert any(lvl == "error" and "Could not play" in m for lvl, m in logs)
    qtbot.waitUntil(lambda: fake.plays()[-1] == tracks[1], timeout=2000)


def test_a_queue_of_unplayable_files_stops_instead_of_looping(qtbot, tracks):
    ws, fake = make_player(qtbot)
    fake.fail_paths = set(tracks[:3])
    ws._cycle_repeat(); ws._cycle_repeat()   # Repeat: All
    states = []
    ws.playback_state_changed.connect(states.append)
    ws.load_tracks_and_play(tracks[:3])
    qtbot.wait(400)
    assert fake.count("load") == 3           # each tried once, then stop
    assert True not in states
    assert ws._play_btn.text() == "▶"


# ── PLY-20 / PLY-21 ─────────────────────────────────────────────────────────

def test_prev_with_empty_queue_restarts_direct_track(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws._play_directly(tracks[0])
    fake._time = 1500
    ws.play_prev()
    assert fake.calls[-1] == ("set_time", 0)


def test_prev_after_three_seconds_restarts(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(tracks)
    ws.play_next()
    fake._time = 10_000
    ws.play_prev()
    assert fake.calls[-1] == ("set_time", 0)
    assert fake.plays() == [tracks[0], tracks[1]]
    fake._time = 0
    ws.play_prev()
    assert fake.plays()[-1] == tracks[0]


def test_repeat_track_only_repeats_at_track_end(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(tracks)
    ws._cycle_repeat()                        # Repeat: Track
    ws.play_next()                            # explicit Next skips
    assert fake.plays() == [tracks[0], tracks[1]]
    fake.finish()
    ws._tick_position()                       # the end of a track repeats it
    assert fake.plays() == [tracks[0], tracks[1], tracks[1]]


def test_transport_buttons_reflect_backend(qtbot):
    ws, _ = make_player(qtbot)
    assert ws._next_btn.isEnabled() and not ws._play_btn.isEnabled()
    ws2, _ = make_player(qtbot, backend=pb.NullBackend("nope"))
    assert not ws2._next_btn.isEnabled()
    assert not ws2._prev_btn.isEnabled()
    assert "nope" in ws2._play_btn.toolTip()


def test_shutdown_releases_backend(qtbot, tracks):
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(tracks)
    ws.shutdown()
    assert fake.released
    assert not ws._pos_timer.isActive()
