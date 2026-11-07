"""Player background work: library scan, tag reads and cover art.

Covers PLY-11, 12, 13, 14, 15, 22, 23 (docs/redesign/pages/07_player.md).
"""
from __future__ import annotations

import gc
import os
import threading
import time

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

import utils.audio_metadata_reader as amr
from gui.workers import job_registry
from gui.workspaces import player as player_mod
from library_factory import TrackSpec, build_library
from player_helpers import FakeBackend, make_player, make_tracks, wait_scan


def _png_bytes(color: str) -> bytes:
    img = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(color))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


@pytest.fixture
def fake_covers(monkeypatch):
    """read_metadata → cover per path ({path: bytes | None}); optional delay."""
    covers: dict[str, bytes | None] = {}
    state = {"delay": 0.0, "gui_calls": 0}
    main = threading.main_thread()

    def read_metadata(path, include_cover=False):
        if threading.current_thread() is main:
            state["gui_calls"] += 1
        if state["delay"]:
            time.sleep(state["delay"])
        c = covers.get(path)
        return {}, ([c] if c else []), None, None

    monkeypatch.setattr(amr, "read_metadata", read_metadata)
    return covers, state


# ── PLY-11: stale scanner signals ───────────────────────────────────────────

def test_rescan_mid_scan_does_not_duplicate_rows(qtbot, threads, library, monkeypatch):
    real = player_mod._load_row

    def slow(path):
        time.sleep(0.01)
        return real(path)

    monkeypatch.setattr(player_mod, "_load_row", slow)
    ws, _ = make_player(qtbot, library_path=str(library.root))
    ws._reload_library()
    qtbot.wait(30)
    ws._reload_library()          # Reload while the first scan is running
    wait_scan(ws, threads)
    expected = len(library.audio_paths)
    assert ws._lib_table.rowCount() == expected
    assert len(ws._all_rows) == expected
    assert ws._status_lbl.text() == f"Loaded {expected} tracks."


def test_same_library_path_twice_scans_once(qtbot, threads, library):
    ws, _ = make_player(qtbot)
    ws.set_library_path(str(library.root))
    ws.set_library_path(str(library.root))    # the launch sequence does this
    wait_scan(ws, threads)
    assert threads.started.count("_LibraryScanner") == 1
    assert ws._lib_table.rowCount() == len(library.audio_paths)


def test_scanner_uses_the_job_registry_and_is_cancellable(qtbot, threads, library, monkeypatch):
    gate = threading.Event()
    real = player_mod._load_row

    def blocked(path):
        gate.wait(5)
        return real(path)

    monkeypatch.setattr(player_mod, "_load_row", blocked)
    ws, _ = make_player(qtbot, library_path=str(library.root))
    ws._reload_library()
    jobs = [j.name for j in job_registry().jobs()]
    assert "Player library scan" in jobs
    ws._lib_scanner = None
    gc.collect()                              # dropping the ref must not abort
    job_registry().cancel_all()
    gate.set()
    assert threads.wait_all(5000) == []


# ── PLY-22: reserved folders ────────────────────────────────────────────────

def test_scan_skips_non_library_folders_but_keeps_manual_review(qtbot, threads, tmp_path):
    lib = build_library(tmp_path / "lib", [
        TrackSpec("A", "Keep", "Al", 1, ext=".wav"),
        TrackSpec("A", "Dupe", "Al", 2, ext=".wav", relpath="Quarantine/dupe.wav"),
        TrackSpec("A", "Junk", "Al", 3, ext=".wav", relpath="Trash/junk.wav"),
        TrackSpec("A", "Pl", "Al", 4, ext=".wav", relpath="Playlists/x.wav"),
        TrackSpec("A", "Review", "Al", 5, ext=".wav", relpath="Manual Review/r.wav"),
        TrackSpec("A", "Unsorted", "Al", 6, ext=".wav", relpath="Not Sorted/n.wav"),
    ])
    ws, _ = make_player(qtbot, library_path=str(lib.root))
    ws._reload_library()
    wait_scan(ws, threads)
    # "/<relative path>" with "/" on every platform (on Windows the rows hold
    # native "\\" paths, and a "/"-only check would pass vacuously).
    names = sorted("/" + os.path.relpath(r["path"], lib.root).replace(os.sep, "/") for r in ws._all_rows)
    assert len(names) == 3
    assert not any(n.startswith(("/Quarantine", "/Trash", "/Playlists")) for n in names)
    assert any(n.startswith("/Manual Review") for n in names)
    assert any(n.startswith("/Not Sorted") for n in names)


# ── PLY-23: no tag reads on the GUI thread ──────────────────────────────────

def test_queue_adds_read_tags_off_the_gui_thread(qtbot, threads, tmp_path, monkeypatch):
    paths = make_tracks(tmp_path, 40)
    gui_reads = []
    main = threading.main_thread()
    real = player_mod._load_row

    def tracking(path):
        if threading.current_thread() is main:
            gui_reads.append(path)
        row = real(path)
        row["artist"] = "Tagged"
        return row

    monkeypatch.setattr(player_mod, "_load_row", tracking)
    ws, fake = make_player(qtbot)
    ws.load_tracks_and_play(paths, "Graph selection")
    ws._add_paths_to_queue(paths[:5])
    ws._play_directly(paths[7])
    assert gui_reads == []
    assert threads.wait_all(5000) == []
    qtbot.waitUntil(lambda: all(r["artist"] == "Tagged" for r in ws._queue), timeout=3000)
    assert "Tagged" in ws._queue_list.item(0).text()
    assert gui_reads == []


def test_scanned_rows_are_reused_for_the_queue(qtbot, threads, library, monkeypatch):
    ws, _ = make_player(qtbot, library_path=str(library.root))
    ws._reload_library()
    wait_scan(ws, threads)
    started = list(threads.started)
    ws._add_paths_to_queue([str(p) for p in library.audio_paths[:3]])
    assert threads.started == started          # no tag-read worker needed
    assert ws._queue[0]["title"] == ws._rows_by_path[ws._queue[0]["path"]]["title"]


# ── PLY-12 / PLY-14: cover art ──────────────────────────────────────────────

def test_rapid_track_changes_with_slow_art_never_apply_stale_art(
        qtbot, threads, tmp_path, fake_covers):
    covers, state = fake_covers
    paths = make_tracks(tmp_path, 3)
    covers[paths[0]] = _png_bytes("#ff0000")
    covers[paths[1]] = _png_bytes("#00ff00")
    state["delay"] = 0.2
    ws, _ = make_player(qtbot)
    ws.load_tracks_and_play(paths)
    ws.play_next()
    ws.play_next()                            # track 3 has no art
    ws._art_loader = ws._prev_art_loader = ws._next_art_loader = None
    gc.collect()
    assert threads.wait_all(5000) == []
    qtbot.wait(20)
    assert ws._covers._current_pm is None
    assert not ws._bg_art.has_art
    assert state["gui_calls"] == 0


def test_bar_drops_previous_cover_for_a_track_without_art(qtbot, threads, tmp_path, fake_covers):
    from gui.widgets.now_playing_bar import NowPlayingBar

    covers, _ = fake_covers
    paths = make_tracks(tmp_path, 2)
    covers[paths[0]] = _png_bytes("#ff0000")
    ws, _ = make_player(qtbot)
    bar = NowPlayingBar()
    qtbot.addWidget(bar)
    ws.now_playing_changed.connect(bar.update_now_playing)

    ws.load_tracks_and_play(paths)
    qtbot.waitUntil(lambda: not bar._art_lbl.pixmap().isNull(), timeout=3000)
    assert ws._bg_art.has_art
    ws.play_next()
    assert threads.wait_all(5000) == []
    qtbot.waitUntil(lambda: bar._art_lbl.pixmap().isNull(), timeout=3000)
    assert bar._art_lbl.text() == "♪"
    assert not ws._bg_art.has_art             # background fades to empty too


# ── PLY-13: closing during a scan ───────────────────────────────────────────

def test_closing_main_window_during_scan_is_clean(main_window, qtbot, threads, dialogs,
                                                  library, monkeypatch):
    monkeypatch.setattr("gui.playback_backend.create_backend", lambda: FakeBackend())
    real = player_mod._load_row

    def slow(path):
        time.sleep(0.05)
        return real(path)

    monkeypatch.setattr(player_mod, "_load_row", slow)
    ws = main_window._workspaces["player"]
    ws.set_library_path(str(library.root))
    assert ws._lib_scanner is not None and ws._lib_scanner.isRunning()
    dialogs.answer_yes()                      # "Background tasks running — stop and quit?"
    main_window.close()
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=8000)
    assert threads.wait_all(5000) == []


# ── PLY-15: hover popup ─────────────────────────────────────────────────────

def test_hover_popup_appears_without_a_button_and_does_no_gui_io(
        qtbot, threads, tmp_path, fake_covers, monkeypatch):
    covers, state = fake_covers
    paths = make_tracks(tmp_path, 3)
    covers[paths[0]] = _png_bytes("#0000ff")
    rows = [player_mod._load_row(p) for p in paths]
    for r in rows:
        r["title"] = "Hover " + r["title"]

    def no_gui_reads(path):
        raise AssertionError("tag read on the GUI thread")

    monkeypatch.setattr(player_mod, "_load_row", no_gui_reads)
    ws, _ = make_player(qtbot)
    ws._on_tracks_ready(ws._scan_gen, rows)
    ws._on_scan_complete(ws._scan_gen, len(rows))
    ws._hover_timer.setInterval(10)
    assert ws._lib_table.viewport().hasMouseTracking()

    vp = ws._lib_table.viewport()
    row0 = ws._lib_table.visualItemRect(ws._lib_table.item(0, 1)).center()
    QtTest.QTest.mouseMove(vp, row0)
    qtbot.waitUntil(ws._hover_popup.isVisible, timeout=2000)
    popup = ws._hover_popup
    assert popup.windowFlags() & QtCore.Qt.WindowType.ToolTip
    assert QtWidgets.QWidget.mouseGrabber() is None
    assert popup._title_lbl.text().startswith("Hover ")
    target = popup.path
    if covers.get(target):
        qtbot.waitUntil(lambda: not popup._art_lbl.pixmap().isNull(), timeout=3000)
    assert state["gui_calls"] == 0
    ws.hide()
    assert not popup.isVisible()
    assert threads.wait_all(5000) == []
