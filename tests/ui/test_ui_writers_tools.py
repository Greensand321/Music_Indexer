"""Tools / Playlists library-write regressions from the writers reviews.

6c  "Undo Last Cleanup" on the File Cleanup tile: preview, confirm, restore,
    and point the playlists back.
6d  Playlist writes (and the cleanup undo) are library writes: blocked while
    another writer runs, and started in the LIBRARY_WRITES group.
7a  A library change no longer cancels a confirmed File Cleanup part-way.
7b  An export that completes after a library change does not re-enable Open
    buttons that point into the old library.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest
from PySide6 import QtWidgets

from library_factory import ExtraFile, TrackSpec, build_library
from playlists_helpers import make_library
from ui_harness import process_events


def _cleanup_library(root: Path):
    return build_library(root, [
        TrackSpec("Band", "Song", "LP", 1, relpath="Band/LP/Song (1).mp3"),
        TrackSpec("Band", "Tune", "EP", 1, relpath="Band/EP/Tune - Copy.mp3"),
    ], [ExtraFile(
        "Playlists/mix.m3u",
        b"#EXTM3U\n#EXTINF:1,Tune\n../Band/EP/Tune - Copy.mp3\n#EXTINF:1,Song\n../Band/LP/Song (1).mp3\n",
    )])


def _idle(ws) -> bool:
    return (ws._cleanup_plan_worker is None and ws._cleanup_worker is None
            and ws._cleanup_run_btn.isEnabled())


def _run_cleanup(ws, dialogs, qtbot):
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Accepted)
    ws._cleanup_run_btn.click()
    qtbot.waitUntil(lambda: bool(dialogs.calls_of("exec")) and _idle(ws), timeout=15000)


# ── 6c: Undo Last Cleanup ─────────────────────────────────────────────────


def test_undo_last_cleanup_previews_restores_names_and_playlists(
    make_workspace, dialogs, threads, qtbot, tmp_path
):
    lib = _cleanup_library(tmp_path / "Music")
    playlist = lib.root / "Playlists" / "mix.m3u"
    original_playlist = playlist.read_bytes()
    ws = make_workspace("file_cleanup", library_path=lib.root)
    _run_cleanup(ws, dialogs, qtbot)
    assert (lib.root / "Band/LP/Song.mp3").exists()
    assert b"Song (1)" not in playlist.read_bytes()

    seen: dict = {}

    def review(dlg):
        seen["rows"] = dlg.rows()
        seen["title"] = dlg.windowTitle()
        seen["untouched"] = (lib.root / "Band/LP/Song.mp3").exists()
        return QtWidgets.QDialog.DialogCode.Accepted

    dialogs.on_exec(review)
    assert ws._cleanup_undo_btn.isEnabled()
    ws._cleanup_undo_btn.click()
    qtbot.waitUntil(lambda: "rows" in seen and _idle(ws), timeout=15000)

    assert seen["untouched"], "nothing may be renamed before the undo is confirmed"
    assert "Undo" in seen["title"]
    assert {(r[1], r[2], r[3]) for r in seen["rows"]} == {
        ("Song.mp3", "Song (1).mp3", "Restore"),
        ("Tune.mp3", "Tune - Copy.mp3", "Restore"),
    }
    assert (lib.root / "Band/LP/Song (1).mp3").exists()
    assert not (lib.root / "Band/LP/Song.mp3").exists()
    assert (lib.root / "Band/EP/Tune - Copy.mp3").exists()
    # The playlist edits made by the cleanup are undone too.
    assert playlist.read_bytes() == original_playlist
    assert "restored" in ws._cleanup_status.text()

    # Nothing left to undo now.
    ws._cleanup_undo_btn.click()
    info = dialogs.calls_of("information")
    assert info and "no file cleanup to undo" in info[-1].text


def test_undo_with_no_cleanup_log_says_so(make_workspace, dialogs, tmp_path):
    lib = tmp_path / "Empty"
    lib.mkdir()
    ws = make_workspace("file_cleanup", library_path=lib)
    ws._cleanup_undo_btn.click()
    assert not dialogs.calls_of("exec")
    assert "no file cleanup to undo" in dialogs.calls_of("information")[-1].text


def test_declined_undo_changes_nothing(make_workspace, dialogs, threads, qtbot, tmp_path):
    lib = _cleanup_library(tmp_path / "Music")
    ws = make_workspace("file_cleanup", library_path=lib.root)
    _run_cleanup(ws, dialogs, qtbot)
    after_cleanup = sorted(p.name for p in (lib.root / "Band/LP").iterdir())
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Rejected)
    ws._cleanup_undo_btn.click()
    process_events(50)
    assert sorted(p.name for p in (lib.root / "Band/LP").iterdir()) == after_cleanup
    assert ws._cleanup_worker is None


def test_undo_is_a_library_write(make_workspace, dialogs, threads, qtbot, tmp_path, monkeypatch):
    from controllers import tag_write_guard

    lib = _cleanup_library(tmp_path / "Music")
    ws = make_workspace("file_cleanup", library_path=lib.root)
    _run_cleanup(ws, dialogs, qtbot)
    exec_calls = len(dialogs.calls_of("exec"))

    assert tag_write_guard.acquire("Tag Fixer apply")
    try:
        ws._cleanup_undo_btn.click()
    finally:
        tag_write_guard.release("Tag Fixer apply")
    assert len(dialogs.calls_of("exec")) == exec_calls, "no preview while another writer runs"
    assert any(c.title == "Library Busy" for c in dialogs.calls_of("information"))
    assert (lib.root / "Band/LP/Song.mp3").exists()

    started = []
    real_start = ws._start_worker

    def record(worker, name="", **kw):
        started.append((name, kw))
        return real_start(worker, name, **kw)

    monkeypatch.setattr(ws, "_start_worker", record)
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Accepted)
    ws._cleanup_undo_btn.click()
    qtbot.waitUntil(lambda: _idle(ws), timeout=15000)
    assert started == [("Undo file cleanup", {"library_write": True})]


# ── 7a: a library change does not cancel a confirmed cleanup ─────────────


def test_library_change_lets_a_confirmed_cleanup_finish(
    make_workspace, dialogs, threads, qtbot, tmp_path, monkeypatch
):
    import file_cleanup

    gate = threading.Event()
    entered = threading.Event()
    real = file_cleanup.execute_cleanup_plan

    def slow_execute(plan, *, cancel_event=None, on_item=None):
        entered.set()
        gate.wait(10)
        return real(plan, cancel_event=cancel_event, on_item=on_item)

    monkeypatch.setattr(file_cleanup, "execute_cleanup_plan", slow_execute)
    lib = _cleanup_library(tmp_path / "Music")
    ws = make_workspace("file_cleanup", library_path=lib.root)
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Accepted)
    ws._cleanup_run_btn.click()
    qtbot.waitUntil(entered.is_set, timeout=15000)
    worker = ws._cleanup_worker
    assert worker is not None

    other = tmp_path / "Other"
    other.mkdir()
    ws.set_library_path(str(other))
    assert not worker._cancel.is_set(), "a confirmed library write was cancelled part-way"
    assert not ws._cleanup_run_btn.isEnabled()

    gate.set()
    qtbot.waitUntil(lambda: _idle(ws), timeout=15000)
    assert (lib.root / "Band/LP/Song.mp3").exists()
    assert (lib.root / "Band/EP/Tune.mp3").exists()
    assert "Done" in ws._cleanup_status.text()


def test_library_change_still_cancels_read_only_exports(
    make_workspace, library, qtbot, threads, monkeypatch, tmp_path
):
    import utils.audio_metadata_reader as reader

    gate = threading.Event()
    real = reader.read_tags

    def slow(path):
        gate.wait(5)
        return real(path)

    monkeypatch.setattr(reader, "read_tags", slow)
    ws = make_workspace("exports", library_path=library.root)
    ws._at_run_btn.click()
    worker = ws._at_worker
    other = tmp_path / "Other"
    other.mkdir()
    ws.set_library_path(str(other))
    assert worker._cancel.is_set()
    gate.set()
    qtbot.waitUntil(lambda: ws._at_worker is None, timeout=10000)
    assert threads.wait_all(5000) == []


# ── 7b: exports finishing after a library change ─────────────────────────


@pytest.mark.parametrize("kind", ["at", "codec"])
def test_export_finishing_after_library_change_keeps_open_disabled(
    make_workspace, qtbot, tmp_path, kind
):
    from gui.workspaces.tools import ArtistTitleWorker, CodecExportWorker

    old = tmp_path / "Old"
    old.mkdir()
    ws = make_workspace("exports", library_path=old)
    if kind == "at":
        worker = ArtistTitleWorker(str(old), False, False, parent=ws)
        worker.finished.connect(ws._on_at_finished)
        ws._at_worker = worker
        open_btn, folder_btn = ws._at_open, ws._at_folder
    else:
        worker = CodecExportWorker(str(old), {".mp3"}, False, parent=ws)
        worker.finished.connect(ws._on_codec_finished)
        ws._codec_worker = worker
        open_btn, folder_btn = ws._codec_open, ws._codec_folder

    new = tmp_path / "New"
    new.mkdir()
    ws.set_library_path(str(new))
    # The old library's export completes now (never started: emit by hand).
    out = str(old / "Docs" / "list.txt")
    if kind == "at":
        worker.finished.emit(out, 3, 0)
        assert ws._at_worker is None and ws._at_out_path is None
        assert ws._at_run_btn.isEnabled()
    else:
        worker.finished.emit(out, 3)
        assert ws._codec_worker is None and ws._codec_out_path is None
        assert ws._codec_run_btn.isEnabled()
    assert not open_btn.isEnabled() and not folder_btn.isEnabled()


def test_signal_from_an_untracked_export_is_ignored(make_workspace, qtbot, tmp_path):
    from gui.workspaces.tools import CodecExportWorker

    lib = tmp_path / "Lib"
    lib.mkdir()
    ws = make_workspace("exports", library_path=lib)
    stray = CodecExportWorker(str(lib), {".mp3"}, False, parent=ws)
    stray.finished.connect(ws._on_codec_finished)
    stray.finished.emit(str(lib / "Docs" / "x.txt"), 1)
    assert ws._codec_out_path is None and not ws._codec_open.isEnabled()


# ── 6d: playlist writes are library writes ───────────────────────────────


def _wait_idle(ws, qtbot, timeout=5000):
    qtbot.waitUntil(lambda: not ws.is_busy(), timeout=timeout)
    process_events(20)


def test_playlist_write_is_blocked_while_another_writer_runs(make_workspace, tmp_path, qtbot, dialogs):
    from controllers import tag_write_guard

    lib = make_library(tmp_path / "lib")
    ws = make_workspace("playlists", library_path=lib)
    assert tag_write_guard.acquire("Tag Fixer apply")
    try:
        ws._on_run("folder")
        _wait_idle(ws, qtbot)
    finally:
        tag_write_guard.release("Tag Fixer apply")
    assert not dialogs.calls_of("question"), "the write was offered while the library was busy"
    assert any(c.title == "Library Busy" for c in dialogs.calls_of("information"))
    assert not (lib / "Playlists").exists() or not list((lib / "Playlists").glob("*.m3u"))


def test_playlist_write_runs_in_the_library_writes_group(
    make_workspace, tmp_path, qtbot, dialogs, monkeypatch
):
    lib = make_library(tmp_path / "lib")
    ws = make_workspace("playlists", library_path=lib)
    started = []
    real_start = ws._start_worker

    def record(worker, name="", **kw):
        started.append((worker.mode, kw))
        return real_start(worker, name, **kw)

    monkeypatch.setattr(ws, "_start_worker", record)
    dialogs.answer_yes()
    ws._on_run("folder")
    _wait_idle(ws, qtbot)
    _wait_idle(ws, qtbot)
    assert started == [("folder", {"library_write": False}), ("write", {"library_write": True})]
    assert list((lib / "Playlists").glob("*.m3u"))


def test_library_change_does_not_abandon_a_confirmed_playlist_write(
    make_workspace, tmp_path, qtbot, dialogs, threads, monkeypatch
):
    import playlist_generator as pg

    gate = threading.Event()
    entered = threading.Event()
    real = pg.write_playlist_plan

    def slow_write(*a, **k):
        entered.set()
        gate.wait(10)
        return real(*a, **k)

    monkeypatch.setattr(pg, "write_playlist_plan", slow_write)
    lib = make_library(tmp_path / "lib")
    ws = make_workspace("playlists", library_path=lib)
    dialogs.answer_yes()
    ws._on_run("folder")
    qtbot.waitUntil(entered.is_set, timeout=10000)
    ws.set_library_path(str(make_library(tmp_path / "other")))
    assert ws.is_busy(), "the page must stay busy until the confirmed write ends"
    gate.set()
    _wait_idle(ws, qtbot)
    assert list((lib / "Playlists").glob("*.m3u"))
    assert threads.wait_all(5000) == []
