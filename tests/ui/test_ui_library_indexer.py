"""Indexer page, end to end against the synthetic library (area: library).

preview → confirm (queued via ``dialogs``) → execute, then the filesystem is
checked: nothing outside the plan moved, nothing was overwritten, reserved
folders are untouched and the execution log is in Docs/.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest
from PySide6 import QtCore

import indexer_control
from library_helpers import (
    add_reserved_content,
    content_multiset,
    reserved_tree,
    tree,
    wait_idle,
)


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url, *a, **k: opened.append(url) or True)
    yield opened
    assert not indexer_control.cancel_event.is_set(), "the Qt Indexer must not touch the global flag"


def _indexer(make_workspace, root):
    ws = make_workspace("indexer", library_path=root)
    statuses = []
    ws.status_changed.connect(lambda text, colour: statuses.append((text, colour)))
    ws._statuses = statuses
    return ws


def _run(qtbot, ws):
    assert ws._run_btn.isEnabled()
    qtbot.mouseClick(ws._run_btn, QtCore.Qt.MouseButton.LeftButton)
    wait_idle(qtbot, ws)


def _messy_library(library):
    root = library.root
    add_reserved_content(root, library.find("First Light"))
    # A second, different cover.jpg: the old sweep moved both flat into
    # Trash/cover.jpg, destroying one (and the one already in Trash/).
    (root / "Basalt" / "Granite" / "cover.jpg").write_bytes(b"basalt-cover")
    return root


def test_preview_then_execute_end_to_end(make_workspace, library, dialogs, qtbot, threads):
    root = _messy_library(library)
    before = tree(root)
    reserved_before = reserved_tree(root)
    ws = _indexer(make_workspace, root)
    assert ws.current_step() == 0

    # ── Preview: writes only the report + decision log ──────────────────
    _run(qtbot, ws)
    after_preview = tree(root)
    assert set(after_preview) - set(before) == {"Docs/MusicIndex.html", "Docs/indexer_log.txt"}
    assert all(after_preview[k] == v for k, v in before.items())
    plan = ws._plan
    assert plan is not None and ws.current_step() == 1
    assert ws._report_btn.isEnabled()
    counts = plan.counts()
    assert counts["manual_review"] == 1                       # untitled_track.mp3 (LIB-13)
    sources = {Path(s).relative_to(root).as_posix() for s in plan.moves}
    assert not any(s.split("/")[0] in ("Quarantine", "Trash", "Manual Review", "Not Sorted", "Docs")
                   for s in sources)                           # LIB-02
    assert "Delta Static/Coastal/01 - Low Tide.opus" in sources  # Opus is audio, not a leftover
    html = (root / "Docs" / "MusicIndex.html").read_text(encoding="utf-8")
    assert "Trash/Basalt/Granite/cover.jpg" in html            # the sweep is previewed (LIB-01)
    assert "Trash/Aurora Lines/Dawn Signals/cover.jpg" in html

    # ── Execute that preview ─────────────────────────────────────────────
    ws._dry_run_cb.setChecked(False)
    assert "Execute" in ws._run_btn.text() and ws._run_btn.isEnabled()   # LIB-16
    assert ws.current_step() == 2                                         # DS-22
    dialogs.answer_yes()
    _run(qtbot, ws)
    question = dialogs.calls_of("question")[-1]
    assert f"{counts['moves']} audio file(s)" in question.text
    assert f"{counts['leftovers']} non-audio file(s)" in question.text

    after = tree(root)
    # Exactly the planned moves happened, byte for byte.
    for src, dst in plan.file_moves.items():
        rel_src = Path(src).relative_to(root).as_posix()
        rel_dst = Path(dst).relative_to(root).as_posix()
        assert rel_src not in after
        assert after[rel_dst] == before[rel_src]
    for src, dst in plan.leftovers.items():
        assert after[Path(dst).relative_to(root).as_posix()] == before[Path(src).relative_to(root).as_posix()]
    # Nothing was lost or overwritten: every original file's bytes survive
    # (the only rewritten originals are playlists, whose paths are updated).
    originals = content_multiset({k: v for k, v in before.items() if not k.startswith("Playlists/")})
    assert not (originals - content_multiset(after)), "some file contents disappeared"
    # Reserved folders untouched (new Docs/ / Trash/ / Playlists/ entries are allowed).
    for rel, digest in reserved_before.items():
        if not rel.startswith("Playlists/"):
            assert after.get(rel) == digest, rel
    assert after["Manual Review/untitled_track.mp3"] == before["untitled_track.mp3"]
    assert after["Trash/cover.jpg"] == reserved_before["Trash/cover.jpg"]
    # Execution log in Docs/ lists every move (LIB-22).
    logs = sorted((root / "Docs").glob("indexer_executed_*.txt"))
    assert logs, "no execution log in Docs/"
    log_text = logs[-1].read_text(encoding="utf-8")
    assert log_text.count("[done] move:") == counts["moves"]
    assert log_text.count("[done] move_leftover:") == counts["leftovers"]
    # The plan was consumed; the next click previews again.
    assert ws._plan is None and ws._dry_run_cb.isChecked()
    assert ws._statuses[-1][0] == "Done"
    assert threads.wait_all(10_000) == []


def test_execute_needs_a_preview_of_this_library(make_workspace, library, dialogs, qtbot):
    ws = _indexer(make_workspace, library.root)
    ws._dry_run_cb.setChecked(False)
    assert not ws._run_btn.isEnabled()                         # LIB-24
    assert "preview first" in ws._run_btn.toolTip().lower()
    ws._dry_run_cb.setChecked(True)
    _run(qtbot, ws)
    assert ws._plan is not None
    other = library.root.parent / "Other"
    other.mkdir()
    ws.set_library_path(str(other))                            # plan belongs to the old library
    assert ws._plan is None
    ws._dry_run_cb.setChecked(False)
    assert not ws._run_btn.isEnabled()


def test_execute_refuses_a_stale_plan(make_workspace, library, dialogs, qtbot):
    root = library.root
    ws = _indexer(make_workspace, root)
    _run(qtbot, ws)
    shutil.copyfile(library.find("Magma"), root / "Basalt" / "Granite" / "late_arrival.flac")
    before = tree(root)
    ws._dry_run_cb.setChecked(False)
    dialogs.answer_yes()
    _run(qtbot, ws)
    assert tree(root) == before, "a stale plan must not move anything"
    warning = dialogs.calls_of("warning")[-1]
    assert "changed since the preview" in warning.text
    assert ws._plan is None


def test_cancel_clicked_right_after_run_is_honoured(make_workspace, library, dialogs, qtbot, monkeypatch):
    """LIB-09: the worker used to clear the global cancel flag at run() start."""
    import music_indexer_api  # the module the worker will import (not a stale copy)
    root = library.root
    ws = _indexer(make_workspace, root)
    _run(qtbot, ws)
    before = tree(root)

    real_check = music_indexer_api.check_plan_freshness

    def gated_check(plan):
        # Hold the execute until Cancel has been clicked, so the order
        # "cancel before the first move" is deterministic.
        deadline = time.monotonic() + 5
        while not ws_worker[0].cancel_requested and time.monotonic() < deadline:
            time.sleep(0.01)
        return real_check(plan)

    ws_worker = [None]
    monkeypatch.setattr(music_indexer_api, "check_plan_freshness", gated_check)
    ws._dry_run_cb.setChecked(False)
    dialogs.answer_yes()
    qtbot.mouseClick(ws._run_btn, QtCore.Qt.MouseButton.LeftButton)
    ws_worker[0] = ws._worker
    qtbot.mouseClick(ws._cancel_btn, QtCore.Qt.MouseButton.LeftButton)
    wait_idle(qtbot, ws)

    after = tree(root)
    assert {k: v for k, v in after.items() if not k.startswith("Docs/")} == \
        {k: v for k, v in before.items() if not k.startswith("Docs/")}
    text, colour = ws._statuses[-1]
    assert text == "Cancelled"                                  # LIB-21: not a red "Error"
    from gui.themes.manager import get_manager
    assert colour != get_manager().current.danger


def test_cancel_preview_then_rerun_immediately(make_workspace, library, qtbot, threads):
    """LIB-07: cancelling and re-running must never destroy a running QThread."""
    ws = _indexer(make_workspace, library.root)
    for _ in range(3):
        qtbot.mouseClick(ws._run_btn, QtCore.Qt.MouseButton.LeftButton)
        qtbot.mouseClick(ws._cancel_btn, QtCore.Qt.MouseButton.LeftButton)
        wait_idle(qtbot, ws)
        assert ws._plan is None
        assert ws._statuses[-1][0] == "Cancelled"
    _run(qtbot, ws)
    assert ws._plan is not None
    _run(qtbot, ws)                                            # immediate re-run
    assert ws._plan is not None
    assert threads.wait_all(10_000) == []


def test_library_change_cancels_a_running_preview(make_workspace, library, qtbot, monkeypatch, threads):
    import music_indexer_api  # the module the worker will import (not a stale copy)
    started = threading.Event()
    real = music_indexer_api.compute_moves_and_tag_index

    def slow(root_path, log_callback=None, progress_callback=None, **kw):
        started.set()
        for i in range(500):
            progress_callback(i, 0, "x", "A")   # raises once cancelled
            time.sleep(0.01)
        return real(root_path, log_callback, progress_callback, **kw)

    monkeypatch.setattr(music_indexer_api, "compute_moves_and_tag_index", slow)
    ws = _indexer(make_workspace, library.root)
    qtbot.mouseClick(ws._run_btn, QtCore.Qt.MouseButton.LeftButton)
    worker = ws._worker
    assert started.wait(5)
    other = library.root.parent / "Elsewhere"
    other.mkdir()
    ws.set_library_path(str(other))
    assert worker.cancel_requested
    assert ws._run_btn.isEnabled()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=10_000)
    assert ws._plan is None and ws._status_lbl.text() == "Ready to preview."
    assert threads.wait_all(10_000) == []


def test_nonexistent_library_is_not_created(make_workspace, tmp_path, dialogs, qtbot):
    """LIB-15: a typo'd path used to be created (Docs/, Not Sorted/, Manual Review/)."""
    missing = tmp_path / "no" / "such" / "library"
    ws = _indexer(make_workspace, missing)
    qtbot.mouseClick(ws._run_btn, QtCore.Qt.MouseButton.LeftButton)
    assert ws._worker is None
    assert not missing.exists() and not missing.parent.exists()
    assert "does not exist" in dialogs.calls_of("warning")[-1].text


def test_ignored_options_are_disabled_and_say_why(make_workspace, library):
    """LIB-12: Phase C / flush cache / max workers are ignored by the engine."""
    ws = _indexer(make_workspace, library.root)
    for widget in (ws._cross_album_cb, ws._flush_cache_cb, ws._max_workers_spin):
        assert not widget.isEnabled()
        assert "ignores" in widget.toolTip()
    assert ws._phase_bars[2][0].text() == "Phase C — Applying changes"


def test_quit_during_preview_cancels_and_waits(main_window, library, dialogs, qtbot, monkeypatch):
    """LIB-08: closing mid-run used to abort the process."""
    started = threading.Event()

    def slow(root_path, log_callback=None, progress_callback=None, **kw):
        started.set()
        while True:
            progress_callback(0, 0, "x", "A")   # raises IndexCancelled once cancelled
            time.sleep(0.01)

    monkeypatch.setattr("music_indexer_api.compute_moves_and_tag_index", slow)
    main_window.set_library(str(library.root))
    ws = main_window._workspaces["indexer"]
    qtbot.mouseClick(ws._run_btn, QtCore.Qt.MouseButton.LeftButton)
    worker = ws._worker
    assert started.wait(5)
    dialogs.answer_yes()
    main_window.close()
    assert worker.cancel_requested
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=10_000)
    assert not worker.isRunning()
