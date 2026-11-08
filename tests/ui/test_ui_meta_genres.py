"""Genre Normalizer page (META-02/03/08/13/20/24/25), on real multi-genre files.

MusicBrainz is never contacted: the Fill tab's lookup is a fake.
"""
from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import pytest

from _helpers.meta_tags import build_genre_library, get_genres, needs_ffmpeg
from meta_helpers import slow_reader, wait_for

pytestmark = needs_ffmpeg


@pytest.fixture
def fake_mb(monkeypatch):
    """Replace the MusicBrainz lookup with answers keyed by track title."""
    import update_genres
    from controllers import normalize_controller as nc
    from utils.audio_metadata_reader import read_tags

    answers: dict[str, list[str]] = {}
    calls: list[str] = []

    def lookup(path):
        calls.append(path)
        title = read_tags(path).get("title")
        if title in answers:
            return nc.FillLookup(genres=list(answers[title]), status="matched", detail=f"MusicBrainz: {title}")
        return nc.FillLookup(status="no_match", detail="no MusicBrainz match")

    monkeypatch.setattr(nc, "_default_lookup", lookup)
    monkeypatch.setattr(update_genres, "RATE_LIMIT_SECONDS", 0)
    return answers, calls


def _fill_library(root: Path):
    return build_genre_library(root, [
        ("multi.flac", "A", "Multi", ["Rock", "Pop"]),
        ("one.mp3", "A", "One", ["rock"]),
        ("none.opus", "A", "None", []),
    ])


# ── Fill from MusicBrainz: preview → confirm → write exactly the preview ──


def test_fill_previews_first_and_writes_only_after_confirming(make_workspace, tmp_path, fake_mb, qtbot, dialogs):
    """META-03: it wrote every matched file at once, with no preview or confirm."""
    answers, calls = fake_mb
    answers.update({"Multi": ["X"], "One": ["Electronic"], "None": ["Ambient", "Drone"]})
    files = _fill_library(tmp_path / "lib")
    ws = make_workspace("genres", library_path=tmp_path / "lib")

    ws._run_btn.click()
    wait_for(qtbot, lambda: ws._worker is None)
    # Looked up, nothing written, nothing asked yet.
    assert dialogs.calls_of("question") == []
    assert get_genres(files["one.mp3"]) == ["rock"]
    assert get_genres(files["none.opus"]) == []                # META-25: .opus is included
    assert str(files["multi.flac"]) not in calls               # META-04: 2+ genres skipped
    table = ws._fill_table
    rows = sorted(tuple(table.item(r, c).text() for c in range(3)) for r in range(table.rowCount()))
    assert rows == [("none.opus", "(none)", "Ambient; Drone"), ("one.mp3", "rock", "rock; Electronic")]
    assert ws._fill_apply_btn.isEnabled()

    dialogs.answer_no()
    ws._fill_apply_btn.click()
    assert get_genres(files["one.mp3"]) == ["rock"]

    dialogs.answer_yes()
    ws._fill_apply_btn.click()
    wait_for(qtbot, lambda: ws._fill_apply_worker is None)
    assert get_genres(files["one.mp3"]) == ["rock", "Electronic"]
    assert get_genres(files["none.opus"]) == ["Ambient", "Drone"]
    assert get_genres(files["multi.flac"]) == ["Rock", "Pop"]
    assert list((tmp_path / "lib" / "Docs").glob("tag_changes_genre_fill_*.jsonl"))
    assert not ws._fill_apply_btn.isEnabled()  # plan consumed; look up again


def test_fill_cancel_and_library_change_discard_the_lookup(make_workspace, tmp_path, fake_mb, qtbot, threads, monkeypatch):
    from controllers import normalize_controller as nc

    answers, calls = fake_mb
    answers.update({"One": ["Electronic"]})
    gate = threading.Event()
    real = nc._default_lookup

    def slow(path):
        gate.wait(0.3)
        return real(path)

    monkeypatch.setattr(nc, "_default_lookup", slow)
    _fill_library(tmp_path / "A")
    (tmp_path / "B").mkdir()
    ws = make_workspace("genres", library_path=tmp_path / "A")
    ws._run_btn.click()
    ws.set_library_path(str(tmp_path / "B"))
    gate.set()
    assert threads.wait_all(10_000) == []
    qtbot.wait(50)
    assert ws._fill_table.rowCount() == 0
    assert not ws._fill_apply_btn.isEnabled()
    assert ws._run_btn.isEnabled()


# ── Normalize with a mapping ──────────────────────────────────────────────


def _preview(ws, qtbot, mapping):
    ws._map_edit.setPlainText(json.dumps(mapping))
    ws._preview_btn.click()
    wait_for(qtbot, lambda: ws._plan_worker is None)


def test_preview_shows_and_apply_keeps_every_genre(make_workspace, tmp_path, qtbot, dialogs):
    """META-02: only the first genre was read, so apply deleted the rest."""
    files = build_genre_library(tmp_path / "lib", [("m.flac", "A", "T", ["rock", "Pop", "90s"])])
    ws = make_workspace("genres", library_path=tmp_path / "lib")
    ws._on_scan()
    wait_for(qtbot, lambda: ws._scan_worker is None)
    assert sorted(ws._raw_list.item(i).text() for i in range(ws._raw_list.count())) == ["90s", "Pop", "rock"]

    _preview(ws, qtbot, {"rock": ["Classic Rock"], "90s": ["invalid"]})
    t = ws._changes_table
    assert [t.item(0, c).text() for c in range(3)] == ["m.flac", "rock; Pop; 90s", "Classic Rock; Pop"]
    dialogs.answer_yes()
    ws._apply_btn.click()
    wait_for(qtbot, lambda: ws._apply_worker is None)
    assert get_genres(files["m.flac"]) == ["Classic Rock", "Pop"]
    assert list((tmp_path / "lib" / "Docs").glob("tag_changes_genre_normalize_*.jsonl"))


def test_preview_tells_about_files_it_cannot_write(make_workspace, tmp_path, qtbot):
    """META-13: unwritable files failed at apply and came back in every preview."""
    files = build_genre_library(tmp_path / "lib", [
        ("ok.flac", "A", "T1", ["rock"]), ("ro.flac", "A", "T2", ["rock"]),
    ])
    os.chmod(files["ro.flac"], stat.S_IRUSR | stat.S_IRGRP)
    try:
        ws = make_workspace("genres", library_path=tmp_path / "lib")
        _preview(ws, qtbot, {"rock": ["Rock"]})
        assert ws._changes_table.rowCount() == 1
        assert ws._changes_table.item(0, 0).text() == "ok.flac"
        assert "1 more would change but can't be written" in ws._preview_status.text()
    finally:
        os.chmod(files["ro.flac"], stat.S_IRUSR | stat.S_IWUSR)


def test_library_change_mid_preview_never_applies_the_old_plan(make_workspace, tmp_path, qtbot, threads, dialogs, monkeypatch):
    """META-08: the stale plan was adopted for the new library and Apply wrote library A."""
    from controllers import normalize_controller as nc

    files = build_genre_library(tmp_path / "A", [(f"t{i}.flac", "A", f"T{i}", ["rock"]) for i in range(4)])
    (tmp_path / "B").mkdir()
    monkeypatch.setattr(nc, "_default_read_tags", slow_reader(0.1))
    ws = make_workspace("genres", library_path=tmp_path / "A")
    ws._map_edit.setPlainText(json.dumps({"rock": ["Rock"]}))
    ws._preview_btn.click()
    ws.set_library_path(str(tmp_path / "B"))
    assert threads.wait_all(10_000) == []
    qtbot.wait(50)
    assert ws._plan is None
    assert ws._changes_table.rowCount() == 0
    assert not ws._apply_btn.isEnabled()
    ws._apply_btn.click()
    assert dialogs.calls_of("question") == []
    assert all(get_genres(p) == ["rock"] for p in files.values())


def test_no_preview_or_second_writer_while_applying(make_workspace, tmp_path, qtbot, dialogs, monkeypatch):
    """META-20: editing the mapping mid-apply re-enabled Preview; writers could overlap."""
    from controllers import normalize_controller as nc
    from controllers import tag_write_guard

    files = build_genre_library(tmp_path / "lib", [("a.flac", "A", "T", ["rock"])])
    gate = threading.Event()
    real_writer = nc._default_writer

    def gated_writer(path, genres):
        gate.wait(5)
        return real_writer(path, genres)

    monkeypatch.setattr(nc, "_default_writer", gated_writer)
    ws = make_workspace("genres", library_path=tmp_path / "lib")
    _preview(ws, qtbot, {"rock": ["Rock"]})
    dialogs.answer_yes()
    ws._apply_btn.click()
    try:
        assert ws._apply_worker is not None
        assert tag_write_guard.current_owner() == "Genre normalize apply"
        ws._map_edit.setPlainText(json.dumps({"rock": ["Hard Rock"]}))
        assert not ws._preview_btn.isEnabled()
        assert not ws._fill_apply_btn.isEnabled()
    finally:
        gate.set()
    wait_for(qtbot, lambda: ws._apply_worker is None)
    assert tag_write_guard.current_owner() is None
    assert get_genres(files["a.flac"]) == ["Rock"]
    assert ws._preview_btn.isEnabled()


def test_unsaved_mapping_survives_a_library_switch(make_workspace, tmp_path, qtbot, dialogs):
    """META-24: switching library (or Load saved mapping) wiped pasted text."""
    from controllers import normalize_controller as nc

    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    nc.save_mapping(str(tmp_path / "B"), {"b": ["B"]})
    ws = make_workspace("genres", library_path=tmp_path / "A")
    draft = '{"hip hop": ["Hip-Hop"]}'
    ws._map_edit.setPlainText(draft)
    ws.set_library_path(str(tmp_path / "B"))
    assert json.loads(ws._map_edit.toPlainText()) == {"b": ["B"]}   # B's saved mapping
    ws.set_library_path(str(tmp_path / "A"))
    assert ws._map_edit.toPlainText() == draft                      # A's draft is back

    nc.save_mapping(str(tmp_path / "A"), {"saved": ["Saved"]})
    dialogs.answer_no()
    ws._on_load_mapping()
    assert len(dialogs.calls_of("question")) == 1
    assert ws._map_edit.toPlainText() == draft                      # declined: kept
    dialogs.answer_yes()
    ws._on_load_mapping()
    assert json.loads(ws._map_edit.toPlainText()) == {"saved": ["Saved"]}


# ── Liquid Glass layout (R2) ──────────────────────────────────────────────


def _map_rows(ws):
    from gui.workspaces.metadata_views import PILL_ROLE, kind_of

    tree = ws._map_view
    out = {}
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        out[item.text(0)] = (item.text(1), kind_of(item, 1))
        assert item.data(1, PILL_ROLE) is None
    return out


def test_mapping_table_shows_drop_values_as_a_pill(make_workspace, tmp_path, qtbot):
    (tmp_path / "lib").mkdir()
    ws = make_workspace("genres", library_path=tmp_path / "lib")
    ws.set_section("normalize")
    ws._map_edit.setPlainText(json.dumps(
        {"90s": ["invalid"], "seen live": None, "hip hop": ["Hip-Hop"], "Rock": ["Rock"]}))
    wait_for(qtbot, lambda: ws._map_view.topLevelItemCount() == 4, timeout=3000)
    assert _map_rows(ws) == {
        "90s": ("Drop", "pill"),             # ["invalid"] means drop
        "seen live": ("Drop", "pill"),       # so does null
        "hip hop": ("Hip-Hop", "accent"),    # a real change
        "Rock": ("Rock", ""),                # maps to itself: nothing to tint
    }
    assert "2 marked to drop" in ws._map_status.text()
    ws._map_edit.setPlainText("{ not json")
    wait_for(qtbot, lambda: ws._map_view.topLevelItemCount() == 0, timeout=3000)


def test_preview_card_counts_and_marks_what_changes(make_workspace, tmp_path, qtbot):
    from gui.workspaces.metadata_views import kind_of

    build_genre_library(tmp_path / "lib", [
        ("m.flac", "A", "T1", ["rock", "90s"]),
        ("x.flac", "A", "T2", ["90s"]),
        ("ok.flac", "A", "T3", ["Rock"]),
        ("none.flac", "A", "T4", []),
    ])
    ws = make_workspace("genres", library_path=tmp_path / "lib")
    ws.set_section("normalize")
    assert ws._norm_empty.isVisibleTo(ws) and not ws._changes_table.isVisibleTo(ws)
    _preview(ws, qtbot, {"rock": ["Rock"], "90s": ["invalid"]})
    assert ws._changes_table.isVisibleTo(ws) and not ws._norm_empty.isVisibleTo(ws)
    stats = {k: ws._norm_stats.tile(k).value() for k in ws._norm_stats.keys()}
    assert stats == {"changes": "2", "canonical": "1", "untagged": "1", "blocked": "0"}
    t = ws._changes_table
    rows = {t.item(r, 0).text(): (t.item(r, 2).text(), kind_of(t.item(r, 2))) for r in range(t.rowCount())}
    assert rows == {
        "m.flac": ("Rock", "accent"),
        "x.flac": ("(none — all dropped)", "pill"),
    }
    assert ws._apply_btn.text() == "Apply 2 changes"
    ws._map_edit.setPlainText(json.dumps({"rock": ["Hard Rock"]}))
    assert ws._norm_empty.isVisibleTo(ws) and ws._apply_btn.text() == "Apply changes"


def test_fill_view_shows_an_empty_state_then_its_plan(make_workspace, tmp_path, fake_mb, qtbot):
    answers, _calls = fake_mb
    answers.update({"One": ["Electronic"]})
    _fill_library(tmp_path / "lib")
    ws = make_workspace("genres", library_path=tmp_path / "lib")
    assert ws._fill_empty.isVisibleTo(ws) and not ws._fill_table.isVisibleTo(ws)
    ws._run_btn.click()
    wait_for(qtbot, lambda: ws._worker is None)
    assert ws._fill_table.isVisibleTo(ws) and ws._fill_stats.isVisibleTo(ws)
    stats = {k: ws._fill_stats.tile(k).value() for k in ws._fill_stats.keys()}
    assert stats == {"changes": "1", "skipped": "1", "nomatch": "1", "problems": "0"}
    assert ws._fill_apply_btn.text() == "Apply 1 addition"


def test_each_view_is_sized_to_itself(make_workspace, tmp_path):
    """A stacked widget reserves its tallest page; the Fill view used to sit
    above a long empty scroll the height of the Normalize view."""
    (tmp_path / "lib").mkdir()
    ws = make_workspace("genres", library_path=tmp_path / "lib")
    ws.set_section_tabs_visible(False)
    ws.set_section("fill")
    fill = ws._tabs.heightForWidth(800)
    ws.set_section("normalize")
    normalize = ws._tabs.heightForWidth(800)
    assert fill < normalize
    assert fill == ws._tabs.widget(0).heightForWidth(800)


def test_switching_views_cross_fades(main_window, monkeypatch):
    from gui.themes import motion

    revealed = []
    real = motion.reveal
    monkeypatch.setattr(motion, "reveal", lambda w, **kw: (revealed.append(w), real(w, **kw))[1])
    genres = main_window._workspaces["genres"]
    main_window._on_nav_changed("metadata/fill_genres")
    revealed.clear()
    main_window._on_nav_changed("metadata/normalize_genres")
    assert revealed == [genres]           # same panel: the page itself never fades it
    main_window._on_nav_changed("metadata/fill_genres")
    assert revealed == [genres, genres]


def test_buttons_carry_glyphs_and_the_old_title_is_gone(make_workspace):
    from gui.compat import QtWidgets

    ws = make_workspace("genres")
    glyphs = {
        ws._run_btn: "search", ws._fill_apply_btn: "check", ws._cancel_btn: "close",
        ws._scan_btn: "search", ws._copy_raw_btn: "list", ws._copy_prompt_btn: "file",
        ws._load_map_btn: "folder", ws._save_map_btn: "save", ws._preview_btn: "list",
        ws._apply_btn: "check", ws._apply_cancel_btn: "close",
    }
    for button, name in glyphs.items():
        assert button.property("glyphName") == name, button.text()
    for button in ws.findChildren(QtWidgets.QAbstractButton):
        assert all(ord(ch) < 0x2190 or ch in "…→" for ch in button.text()), button.text()
    texts = [w.text() for w in ws.findChildren(QtWidgets.QLabel)]
    assert not any("Genre Normalizer" in t for t in texts)
    assert ws.findChildren(QtWidgets.QLabel, "sectionTitle") == []
