"""Tag Fixer page: preview-first, per-field, honest (META-01/06/07/09/10/11/12/17/18/20/22).

Real tagged files on disk (``tests/library_factory.py`` + multi-value genres
written with mutagen); the lookup source is a fake, so there is no network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from _helpers.meta_tags import build_genre_library, get_field, get_genres, needs_ffmpeg
from meta_helpers import install_source, set_row_checked, table_rows, wait_for

pytestmark = needs_ffmpeg


def _library(root: Path):
    return build_genre_library(root, [
        ("a.flac", "Art", "Song", ["Rock", "Pop"]),
        ("b.mp3", "Art", "Miss", ["Jazz"]),
        ("c.opus", "Art", "Other", []),
    ])


ANSWERS = {
    "Song": {"artist": "Fixed Artist", "genres": ["Electronic"], "score": 1.0},
    "Other": {"album": "Found Album", "score": 0.95},
}


def _scan(ws, qtbot):
    ws._scan_btn.click()
    wait_for(qtbot, lambda: ws._worker is None)


def _apply(ws, qtbot, dialogs, answer=True):
    ws._dry_run_cb.setChecked(False)
    (dialogs.answer_yes if answer else dialogs.answer_no)()
    ws._apply_btn.click()
    wait_for(qtbot, lambda: ws._apply_worker is None)


def test_preview_lists_exact_field_changes_and_separates_no_match(make_workspace, tmp_path, monkeypatch, qtbot):
    files = _library(tmp_path / "lib")
    install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    _scan(ws, qtbot)
    assert table_rows(ws) == sorted([
        ("a.flac", "Artist", "Art", "Fixed Artist", True),
        # the whole on-disk list, then the addition — what apply will write
        ("a.flac", "Genre", "Rock; Pop", "Rock; Pop; Electronic", True),
        ("c.opus", "Album", "Album", "Found Album", True),
    ])
    # META-06: the unmatched file is not a blank "X → ''" proposal...
    assert all(r[0] != "b.mp3" for r in table_rows(ws, visible_only=False))
    # ...it is listed separately, with the reason.
    assert ws._nomatch_list.count() == 1
    assert ws._nomatch_list.item(0).text().startswith("b.mp3")
    assert "1 file(s) with no match" in ws._summary_lbl.text()
    assert get_genres(files["a.flac"]) == ["Rock", "Pop"]  # a scan never writes


def test_apply_writes_exactly_the_ticked_rows(make_workspace, tmp_path, monkeypatch, qtbot, dialogs):
    """META-01 (Apply raised TypeError) and META-07 (unticked fields still written)."""
    files = _library(tmp_path / "lib")
    source = install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    _scan(ws, qtbot)
    calls_after_scan = len(source.calls)
    set_row_checked(ws, "a.flac", "Genre", False)
    set_row_checked(ws, "c.opus", "Album", False)

    _apply(ws, qtbot, dialogs)

    [question] = dialogs.calls_of("question")
    assert "1 ticked tag change(s) to 1 file(s)" in question.text
    assert get_field(files["a.flac"], "artist") == ["Fixed Artist"]
    assert get_genres(files["a.flac"]) == ["Rock", "Pop"]           # unticked: untouched
    assert get_field(files["c.opus"], "album") == ["Album"]          # unticked: untouched
    assert get_genres(files["b.mp3"]) == ["Jazz"]                    # unmatched: untouched
    # META-17: no automatic online rescan, and the apply log is kept.
    assert len(source.calls) == calls_after_scan
    assert "Updated tags for" in ws._log_area.toPlainText()
    # The written row is gone; the unticked ones remain, still unticked.
    assert table_rows(ws) == sorted([
        ("a.flac", "Genre", "Rock; Pop", "Rock; Pop; Electronic", False),
        ("c.opus", "Album", "Album", "Found Album", False),
    ])
    change_logs = list((tmp_path / "lib" / "Docs").glob("tag_changes_tagfixer_*.jsonl"))
    assert len(change_logs) == 1
    assert "Change log:" in ws._prog_status.text()


def test_field_filter_hides_rows_and_hidden_rows_are_never_written(make_workspace, tmp_path, monkeypatch, qtbot, dialogs):
    files = _library(tmp_path / "lib")
    install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    _scan(ws, qtbot)
    ws._field_cbs["Genre"].setChecked(False)
    assert sorted(r[1] for r in table_rows(ws)) == ["Album", "Artist"]
    _apply(ws, qtbot, dialogs)
    assert get_field(files["a.flac"], "artist") == ["Fixed Artist"]
    assert get_field(files["c.opus"], "album") == ["Found Album"]
    assert get_genres(files["a.flac"]) == ["Rock", "Pop"]


def test_sorting_then_toggling_rows_is_stable(make_workspace, tmp_path, monkeypatch, qtbot):
    from PySide6 import QtCore

    _library(tmp_path / "lib")
    install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    _scan(ws, qtbot)
    for column in (5, 1, 2):
        ws._table.sortItems(column, QtCore.Qt.SortOrder.AscendingOrder)
        ws._on_deselect_all()
        set_row_checked(ws, "a.flac", "Artist", True)
        set_row_checked(ws, "c.opus", "Album", True)
        selected = sorted((str(r.path).rsplit("/", 1)[-1], tuple(f)) for r, f in ws._selected_changes())
        assert selected == [("a.flac", ("artist",)), ("c.opus", ("album",))]


def test_declining_the_confirmation_writes_nothing(make_workspace, tmp_path, monkeypatch, qtbot, dialogs):
    files = _library(tmp_path / "lib")
    install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    _scan(ws, qtbot)
    _apply(ws, qtbot, dialogs, answer=False)
    assert get_field(files["a.flac"], "artist") == ["Art"]
    assert not list((tmp_path / "lib" / "Docs").glob("tag_changes_*.jsonl"))


def test_library_change_drops_proposals(make_workspace, tmp_path, monkeypatch, qtbot, dialogs):
    """META-09: old proposals stayed with Apply enabled and wrote the old library."""
    files = _library(tmp_path / "A")
    (tmp_path / "B").mkdir()
    install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "A")
    _scan(ws, qtbot)
    assert ws._apply_btn.isEnabled()
    ws.set_library_path(str(tmp_path / "B"))
    assert ws._table.topLevelItemCount() == 0
    assert not ws._apply_btn.isEnabled()
    ws._dry_run_cb.setChecked(False)
    ws._apply_btn.click()  # disabled: nothing happens, no dialog
    assert dialogs.calls_of("question") == []
    assert get_field(files["a.flac"], "artist") == ["Art"]


def test_library_change_mid_scan_discards_the_stale_result(make_workspace, tmp_path, monkeypatch, qtbot, threads):
    _library(tmp_path / "A")
    (tmp_path / "B").mkdir()
    source = install_source(monkeypatch, ANSWERS, delay=0.3)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "A")
    ws._scan_btn.click()
    wait_for(qtbot, lambda: len(source.calls) >= 1)
    ws.set_library_path(str(tmp_path / "B"))
    assert threads.wait_all(10_000) == []
    qtbot.wait(50)
    assert ws._table.topLevelItemCount() == 0
    assert not ws._apply_btn.isEnabled()
    assert ws._scan_btn.isEnabled()
    assert len(source.calls) < 3  # the old scan stopped early


def test_cancel_stops_the_scan_promptly(make_workspace, tmp_path, monkeypatch, qtbot):
    """META-10: Cancel used to be ignored until every file had been looked up."""
    build_genre_library(tmp_path / "lib", [(f"t{i}.flac", "Art", f"T{i}", []) for i in range(8)])
    source = install_source(monkeypatch, {}, delay=0.25)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    ws._scan_btn.click()
    wait_for(qtbot, lambda: len(source.calls) >= 1)
    ws._cancel_btn.click()
    wait_for(qtbot, lambda: ws._worker is None, timeout=3000)
    assert len(source.calls) <= 2
    assert ws._prog_status.text() == "Cancelled."
    assert ws._table.topLevelItemCount() == 0


def test_progress_is_a_percentage(make_workspace, tmp_path, monkeypatch, qtbot):
    """META-11: the bar was fed file indexes and pinned at 100 from file 100 on."""
    build_genre_library(tmp_path / "lib", [(f"t{i}.flac", "Art", f"T{i}", []) for i in range(5)])
    install_source(monkeypatch, {})
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    seen = []
    ws._scan_btn.click()
    ws._worker.progress.connect(seen.append)
    wait_for(qtbot, lambda: ws._worker is None)
    assert seen == [20, 40, 60, 80, 100]


def test_scan_explains_why_nothing_matched(make_workspace, tmp_path, monkeypatch, qtbot):
    """META-18: no fpcalc / no key / offline used to look like "no match"."""
    _library(tmp_path / "lib")
    source = install_source(monkeypatch, ANSWERS, name="AcoustID", active=False,
                            problem="fpcalc (Chromaprint) was not found")
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    _scan(ws, qtbot)
    assert source.calls == []
    assert "fpcalc" in ws._reason_lbl.text()
    assert ws._nomatch_list.count() == 3
    assert "no lookup source available" in ws._nomatch_list.item(0).text()


def test_apply_refused_while_another_tool_writes_tags(make_workspace, tmp_path, monkeypatch, qtbot, dialogs):
    """META-20: concurrent tag writers raced on the same files."""
    from controllers import tag_write_guard

    files = _library(tmp_path / "lib")
    install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    _scan(ws, qtbot)
    assert tag_write_guard.acquire("Genre normalize apply")
    try:
        ws._dry_run_cb.setChecked(False)
        dialogs.answer_yes()
        ws._apply_btn.click()
        assert ws._apply_worker is None
        [busy] = dialogs.calls_of("information")
        assert "Genre normalize apply is writing tags" in busy.text
    finally:
        tag_write_guard.release("Genre normalize apply")
    assert get_field(files["a.flac"], "artist") == ["Art"]


def test_labels_use_theme_roles_not_inline_colours(make_workspace):
    """META-22."""
    ws = make_workspace("tag_fixer")
    for label in (ws._prog_status, ws._summary_lbl, ws._reason_lbl):
        assert label.objectName() == "statusHint"
        assert label.styleSheet() == ""


def test_closing_the_app_mid_scan_stops_the_worker(main_window, tmp_path, monkeypatch, qtbot, dialogs, threads):
    """META-12: closing mid-run aborted the process (QThread destroyed while running)."""
    build_genre_library(tmp_path / "lib", [(f"t{i}.flac", "Art", f"T{i}", []) for i in range(6)])
    source = install_source(monkeypatch, {}, delay=0.3)
    main_window.set_library(str(tmp_path / "lib"))
    assert threads.wait_all(10_000) == []
    ws = main_window._workspaces["tag_fixer"]
    ws._scan_btn.click()
    wait_for(qtbot, lambda: len(source.calls) >= 1)
    dialogs.answer_yes()  # "Background tasks running — stop them and quit?"
    main_window.close()
    wait_for(qtbot, lambda: not main_window.isVisible(), timeout=5000)
    assert threads.wait_all(5000) == []
    assert len(source.calls) < 6
