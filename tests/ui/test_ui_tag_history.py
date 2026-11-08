"""Metadata → Undo changes: list → preview (reads only) → confirm → restore.

Real files with real multi-value genre tags (``tests/_helpers/meta_tags.py``),
changed through the real ``apply_genre_changes`` with a real change log, then
undone through the panel. The read-back uses mutagen directly.
"""
from __future__ import annotations

import os

import pytest

from _helpers.meta_tags import build_genre_library, get_genres, needs_ffmpeg
from controllers import tag_write_guard
from controllers.normalize_controller import GenreChange, apply_genre_changes
from gui.compat import QtCore
from utils.tag_change_log import TagChangeLog

pytestmark = needs_ffmpeg


def _genre_apply(root, changes, tool="genre_normalize"):
    with TagChangeLog(str(root), tool) as log:
        result = apply_genre_changes(changes, change_log=log)
    assert not result.failed and not result.skipped
    return log.path


@pytest.fixture
def changed_library(tmp_path):
    files = build_genre_library(tmp_path, [
        ("multi.flac", "Artist", "Multi", ["Rock", "Pop", "Indie"]),
        ("edited.flac", "Artist", "Edited", ["Jazz"]),
        ("gone.flac", "Artist", "Gone", ["Funk"]),
    ])
    log_path = _genre_apply(tmp_path, [
        GenreChange(str(files["multi.flac"]), ["Rock", "Pop", "Indie"], ["Rock"]),
        GenreChange(str(files["edited.flac"]), ["Jazz"], ["Blues"]),
        GenreChange(str(files["gone.flac"]), ["Funk"], ["Soul"]),
    ])
    return tmp_path, files, log_path


def _select_first(ws):
    ws._log_list.topLevelItem(0).setSelected(True)


def _preview(ws, qtbot, threads):
    _select_first(ws)
    assert ws._preview_btn.isEnabled()
    qtbot.mouseClick(ws._preview_btn, QtCore.Qt.MouseButton.LeftButton)
    threads.wait_all()
    qtbot.waitUntil(lambda: ws._plan_worker is None, timeout=20000)


def _rows(ws):
    tree = ws._preview
    return [[tree.topLevelItem(i).text(c) for c in range(4)] for i in range(tree.topLevelItemCount())]


def test_lists_applies_newest_first_and_says_when_empty(make_workspace, tmp_path):
    ws = make_workspace("tag_history", library_path=tmp_path)
    assert ws._log_list.topLevelItemCount() == 0
    assert ws._logs_empty.isVisibleTo(ws) and not ws._preview_btn.isEnabled()


def test_preview_reads_only_then_undo_restores_exactly_the_preview(
        make_workspace, changed_library, dialogs, qtbot, threads):
    root, files, log_path = changed_library
    from _helpers.meta_tags import set_genres

    set_genres(files["edited.flac"], ["Edited By Hand"])   # changed since the apply
    os.remove(files["gone.flac"])                            # moved away since

    ws = make_workspace("tag_history", library_path=root)
    assert ws._log_list.topLevelItemCount() == 1
    first = ws._log_list.topLevelItem(0)
    assert first.text(1) == "Normalize genres" and first.text(3) == "3"
    assert first.isSelected() and ws._preview_btn.isEnabled()   # newest preselected

    _preview(ws, qtbot, threads)
    rows = _rows(ws)
    assert ["multi.flac", "genre", "Rock", "Rock; Pop; Indie"] in rows
    assert any(r[0] == "edited.flac" and r[3].startswith("Left alone: edited") for r in rows)
    assert any(r[0] == "gone.flac" and "not found" in r[3] for r in rows)
    assert "1 tag(s) in 1 file(s)" in ws._summary_lbl.text()
    # Previewing wrote nothing.
    assert get_genres(files["multi.flac"]) == ["Rock"]
    assert ws._undo_btn.isEnabled()

    dialogs.answer_yes()
    qtbot.mouseClick(ws._undo_btn, QtCore.Qt.MouseButton.LeftButton)
    threads.wait_all()
    qtbot.waitUntil(lambda: ws._apply_worker is None, timeout=20000)

    assert "Restore 1 tag(s) in 1 file(s)" in dialogs.calls_of("question")[-1].text
    assert get_genres(files["multi.flac"]) == ["Rock", "Pop", "Indie"]   # every value back
    assert get_genres(files["edited.flac"]) == ["Edited By Hand"]         # never overwritten
    assert tag_write_guard.current_owner() is None
    # The undo is listed (newest first) and the original is marked undone.
    assert ws._log_list.topLevelItemCount() == 2
    assert ws._log_list.topLevelItem(0).text(1) == "Undo"
    assert ws._log_list.topLevelItem(1).text(4).startswith("Undone")
    assert ws._plan is None and not ws._undo_btn.isEnabled()
    revert_logs = [n for n in os.listdir(root / "Docs") if n.startswith("tag_changes_revert_")]
    assert len(revert_logs) == 1


def test_declining_the_confirm_writes_nothing(make_workspace, changed_library, dialogs, qtbot, threads):
    root, files, _log = changed_library
    ws = make_workspace("tag_history", library_path=root)
    _preview(ws, qtbot, threads)
    dialogs.answer_no()
    qtbot.mouseClick(ws._undo_btn, QtCore.Qt.MouseButton.LeftButton)
    assert ws._apply_worker is None
    assert get_genres(files["multi.flac"]) == ["Rock"]
    assert not any(n.startswith("tag_changes_revert_") for n in os.listdir(root / "Docs"))


def test_another_tag_job_writing_blocks_the_undo(make_workspace, changed_library, dialogs, qtbot, threads):
    root, files, _log = changed_library
    ws = make_workspace("tag_history", library_path=root)
    _preview(ws, qtbot, threads)
    assert tag_write_guard.acquire("Tag Fixer apply")
    try:
        qtbot.mouseClick(ws._undo_btn, QtCore.Qt.MouseButton.LeftButton)
    finally:
        tag_write_guard.release("Tag Fixer apply")
    assert ws._apply_worker is None
    assert dialogs.calls_of("question") == []          # refused before asking
    assert "Tag Fixer apply" in dialogs.calls_of("information")[-1].text
    assert get_genres(files["multi.flac"]) == ["Rock"]


def test_library_change_drops_the_preview(make_workspace, changed_library, tmp_path_factory, qtbot, threads):
    root, _files, _log = changed_library
    ws = make_workspace("tag_history", library_path=root)
    _preview(ws, qtbot, threads)
    assert ws._plan is not None
    other = tmp_path_factory.mktemp("other_library")
    ws.set_library_path(str(other))
    assert ws._plan is None and ws._preview.topLevelItemCount() == 0
    assert ws._log_list.topLevelItemCount() == 0 and not ws._undo_btn.isEnabled()


def test_selecting_another_apply_clears_a_stale_preview(make_workspace, changed_library, qtbot, threads):
    root, files, _log = changed_library
    _genre_apply(root, [GenreChange(str(files["multi.flac"]), ["Rock"], ["Metal"])], tool="genre_fill")
    ws = make_workspace("tag_history", library_path=root)
    assert ws._log_list.topLevelItemCount() == 2
    assert {ws._log_list.topLevelItem(i).text(1) for i in range(2)} == {"Fill genres", "Normalize genres"}
    _preview(ws, qtbot, threads)
    assert ws._plan is not None
    ws._log_list.topLevelItem(0).setSelected(False)
    ws._log_list.topLevelItem(1).setSelected(True)
    assert ws._plan is None and not ws._undo_btn.isEnabled()


def test_undo_is_a_metadata_section(main_window):
    from gui import navigation as nav

    assert nav.resolve("undo_tags") == ("metadata", "undo_tags")
    assert nav.resolve("tag_history") == ("metadata", "undo_tags")
    assert "tag_history" in main_window._workspaces
