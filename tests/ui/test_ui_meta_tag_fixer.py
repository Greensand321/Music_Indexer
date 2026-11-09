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
    # The counts are a StatRow (unmatched counted on their own, not as proposals).
    stats = {k: ws._stats.tile(k).value() for k in ws._stats.keys()}
    assert stats == {"proposed": "3", "files": "2", "ticked": "3", "nomatch": "1"}
    assert ws._nomatch_title.text() == "No match found (1)"
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
    # ...and the unmatched file is still listed as such.
    assert ws._nomatch_list.count() == 1
    assert ws._nomatch_list.item(0).text().startswith("b.mp3")
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
        selected = sorted((Path(r.path).name, tuple(f)) for r, f in ws._selected_changes())
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
        ws._apply_btn.click()
        assert ws._apply_worker is None
        # Refused up front (no "are you sure?" for something that can't run),
        # naming the job that is writing.
        assert dialogs.calls_of("question") == []
        [busy] = dialogs.calls_of("information")
        assert "Genre normalize apply" in busy.text
    finally:
        tag_write_guard.release("Genre normalize apply")
    assert get_field(files["a.flac"], "artist") == ["Art"]


def test_scan_of_the_standard_library(make_workspace, library, monkeypatch, qtbot):
    """Every container, an untagged file, and a Playlists/ folder (reserved, skipped)."""
    source = install_source(monkeypatch, {"Loose Wire": {"album": "Found", "score": 0.9}})
    ws = make_workspace("tag_fixer", library_path=library.root)
    _scan(ws, qtbot)
    assert len(source.calls) == len(library.audio_paths)
    assert table_rows(ws) == [("Echo Park - Loose Wire.mp3", "Album", "(empty)", "Found", True)]
    assert ws._nomatch_list.count() == len(library.audio_paths) - 1
    assert "Lookup sources: FakeID" in ws._reason_lbl.text()


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


# ── Liquid Glass layout (R2): empty state, review grid, scope, glyphs ───────


def test_empty_state_before_a_scan_then_the_review_grid(make_workspace, tmp_path, monkeypatch, qtbot):
    from gui.compat import QtCore
    from gui.widgets.glass_views import kind_of

    _library(tmp_path / "lib")
    install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    assert ws._empty.isVisibleTo(ws) and not ws._table.isVisibleTo(ws)
    assert not ws._stats.isVisibleTo(ws)
    _scan(ws, qtbot)
    assert ws._table.isVisibleTo(ws) and not ws._empty.isVisibleTo(ws)
    assert ws._stats.isVisibleTo(ws) and ws._nomatch_card.isVisibleTo(ws)
    # Changed values are painted on an accent chip from the theme and the
    # current value is muted: a painting kind, no per-item colour.
    item = ws._table.topLevelItem(0)
    assert (kind_of(item, 3), kind_of(item, 4)) == ("muted", "accent")
    assert item.data(4, QtCore.Qt.ItemDataRole.ForegroundRole) is None
    assert ws._apply_btn.text() == "Apply 3 changes"
    ws._on_deselect_all()
    assert ws._apply_btn.text() == "Apply changes"
    assert ws._stats.tile("ticked").value() == "0"


def test_nothing_matched_says_so(make_workspace, tmp_path, monkeypatch, qtbot):
    _library(tmp_path / "lib")
    install_source(monkeypatch, {})
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    _scan(ws, qtbot)
    assert ws._empty.isVisibleTo(ws)
    assert ws._empty.title() == "Nothing could be matched"
    assert ws._nomatch_list.count() == 3


def test_scope_can_be_one_folder_inside_the_library(make_workspace, tmp_path, monkeypatch, qtbot, dialogs):
    files = build_genre_library(tmp_path / "lib", [
        ("Inside/a.flac", "Art", "Song", []),
        ("Outside/c.opus", "Art", "Other", []),
        ("Quarantine/q.flac", "Art", "Song", []),
    ])
    source = install_source(monkeypatch, ANSWERS)
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    assert ws._scope_lib_pill.isVisibleTo(ws) and not ws._scope_reset_btn.isVisibleTo(ws)

    # A reserved folder and a folder outside the library are refused.
    for bad in (tmp_path / "lib" / "Quarantine", tmp_path):
        dialogs.choose_directory(bad)
        ws._scope_btn.click()
        assert ws._scope_folder == ""
    assert len(dialogs.calls_of("information")) == 2

    dialogs.choose_directory(tmp_path / "lib" / "Inside")
    ws._scope_btn.click()
    assert ws._scope_folder == str(tmp_path / "lib" / "Inside")
    assert ws._scope_dir_pill.isVisibleTo(ws) and ws._scope_reset_btn.isVisibleTo(ws)
    assert ws._scope_path.toolTip() == str(tmp_path / "lib" / "Inside")
    _scan(ws, qtbot)
    assert [Path(p).name for p in source.calls] == ["a.flac"]
    assert [r[0] for r in table_rows(ws)] == ["a.flac", "a.flac"]

    # Apply still writes (and logs) into the library's Docs/.
    ws._field_cbs["Genre"].setChecked(False)
    _apply(ws, qtbot, dialogs)
    assert get_field(files["Inside/a.flac"], "artist") == ["Fixed Artist"]
    assert list((tmp_path / "lib" / "Docs").glob("tag_changes_tagfixer_*.jsonl"))

    # The reset button means "whole library" again; a library change too.
    ws._scope_reset_btn.click()
    assert ws._scope_folder == "" and ws._scope_lib_pill.isVisibleTo(ws)
    dialogs.choose_directory(tmp_path / "lib" / "Inside")
    ws._scope_btn.click()
    (tmp_path / "B").mkdir()
    ws.set_library_path(str(tmp_path / "B"))
    assert ws._scope_folder == ""


def test_buttons_carry_glyphs_and_no_emoji(make_workspace):
    from gui.compat import QtWidgets

    ws = make_workspace("tag_fixer")
    glyphs = {
        ws._scan_btn: "search", ws._apply_btn: "check", ws._cancel_btn: "close",
        ws._scope_btn: "folder", ws._select_all_btn: "check", ws._deselect_btn: "minus",
    }
    for button, name in glyphs.items():
        assert button.property("glyphName") == name, button.text()
        assert not button.icon().isNull()
    for button in ws.findChildren(QtWidgets.QAbstractButton):
        assert all(ord(ch) < 0x2190 or ch in "…→" for ch in button.text()), button.text()


def test_sources_button_opens_the_settings_section(make_workspace):
    ws = make_workspace("tag_fixer")
    asked = []
    ws.navigate_requested.connect(asked.append)
    ws._sources_btn.click()
    assert asked == ["settings/services"]
