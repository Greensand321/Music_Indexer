"""Tests for the canonical genre-mapping backend (Qt-free).

Discovery, tag reading and tag writing are injected, so these run without
mutagen, musicbrainzngs, or audio files on disk.
"""
import json
import threading

import pytest

from controllers import normalize_controller as nc
from controllers.normalize_controller import (
    GenreChange,
    apply_genre_changes,
    clean_mapping,
    normalize_genres,
    plan_genre_normalization,
    scan_raw_genres,
)


# ── normalize_genres (pre-existing behaviour, kept) ───────────────────────


def test_normalize_genres_trims_and_dedupes_case_insensitive():
    mapping = {"rock": "Rock"}

    result = normalize_genres([" Rock ", "ROCK"], mapping)

    assert result == ["Rock"]


def test_normalize_genres_splits_combined_entries():
    genres = ["Hip-Hop/Rap; Indie "]

    result = normalize_genres(genres, {})

    assert result == ["Hip-Hop", "Rap", "Indie"]


def test_normalize_genres_handles_mapping_lists_and_invalid_entries():
    mapping = {
        "electronic": ["Electronic", " electronic ", None],
        "skip": None,
        5: "ignored key",
        "alt": ["Alt", "Alt/Rock"],
    }

    result = normalize_genres(["Electronic", "skip", None, "Alt"], mapping)

    assert result == ["Electronic", "Alt", "Rock"]


# ── clean_mapping ─────────────────────────────────────────────────────────


def test_clean_mapping_treats_invalid_marker_as_skip():
    """The prompt asks for ["invalid"]; that must mean drop, not write 'invalid'."""
    cleaned = clean_mapping({"90s": ["invalid"], "rock": ["Rock"]})
    assert cleaned == {"90s": None, "rock": ["Rock"]}


def test_clean_mapping_treats_null_as_skip():
    assert clean_mapping({"noise": None}) == {"noise": None}


def test_clean_mapping_splits_string_values_and_drops_empties():
    cleaned = clean_mapping({"x": "A / B", "y": "", "": ["Z"], 7: ["Q"]})
    assert cleaned == {"x": ["A", "B"], "y": None}


def test_normalize_genres_drops_invalid_marked_entries():
    result = normalize_genres(["90s", "Rock"], {"90s": ["invalid"]})
    assert result == ["Rock"]


# ── scan_raw_genres ───────────────────────────────────────────────────────


def test_scan_raw_genres_splits_and_sorts():
    tags = {
        "/m/a.mp3": {"genre": "Hip-Hop/Rap"},
        "/m/b.mp3": {"genre": ["Rock", "rock & roll"]},
        "/m/c.mp3": {"genre": None},
    }
    seen = []
    result = scan_raw_genres(
        "/m",
        progress_callback=lambda i, t: seen.append((i, t)),
        files=list(tags),
        read_tags=lambda p: tags[p],
    )
    assert result == ["Hip-Hop", "Rap", "Rock", "rock & roll"]
    assert seen[0] == (0, 3) and seen[-1] == (3, 3)


# ── plan_genre_normalization (dry run) ────────────────────────────────────


def _reader(tags):
    return lambda p: tags[p]


def test_plan_lists_only_files_that_would_change():
    tags = {
        "/m/a.mp3": {"genre": ["rock & roll"]},   # mapped -> changes
        "/m/b.mp3": {"genre": ["Rock"]},          # already canonical -> unchanged
        "/m/c.mp3": {"genre": None},              # no genre -> skipped
    }
    plan = plan_genre_normalization(
        "/m", {"rock & roll": ["Rock"]}, files=list(tags), read_tags=_reader(tags)
    )
    assert plan.scanned == 3
    assert plan.with_genres == 2
    assert plan.unchanged == 1
    assert [c.path for c in plan.changes] == ["/m/a.mp3"]
    assert plan.changes[0].before == ["rock & roll"]
    assert plan.changes[0].after == ["Rock"]


def test_plan_splits_combined_entries_even_without_mapping():
    tags = {"/m/a.mp3": {"genre": ["Hip-Hop/Rap"]}}
    plan = plan_genre_normalization("/m", {}, files=list(tags), read_tags=_reader(tags))
    assert plan.changes[0].after == ["Hip-Hop", "Rap"]


def test_plan_drops_invalid_and_can_empty_a_genre_list():
    tags = {"/m/a.mp3": {"genre": ["90s"]}}
    plan = plan_genre_normalization(
        "/m", {"90s": ["invalid"]}, files=list(tags), read_tags=_reader(tags)
    )
    assert plan.changes[0].after == []


def test_plan_never_calls_a_writer(monkeypatch):
    """A dry run must not touch files, whatever the default writer would do."""
    monkeypatch.setattr(nc, "_default_writer", lambda *_a, **_k: pytest.fail("wrote!"))
    tags = {"/m/a.mp3": {"genre": ["x"]}}
    plan_genre_normalization("/m", {"x": ["Y"]}, files=list(tags), read_tags=_reader(tags))


# ── apply_genre_changes ───────────────────────────────────────────────────


def test_apply_writes_exactly_the_planned_after_lists():
    written = {}
    changes = [
        GenreChange("/m/a.mp3", ["rock & roll"], ["Rock"]),
        GenreChange("/m/b.mp3", ["Hip-Hop/Rap"], ["Hip-Hop", "Rap"]),
    ]
    result = apply_genre_changes(
        changes, writer=lambda p, g: written.__setitem__(p, list(g)) or True
    )
    assert written == {"/m/a.mp3": ["Rock"], "/m/b.mp3": ["Hip-Hop", "Rap"]}
    assert result.applied == ["/m/a.mp3", "/m/b.mp3"]
    assert result.failed == [] and result.cancelled is False


def test_apply_reports_failures_and_continues():
    def writer(p, _g):
        if p.endswith("b.mp3"):
            raise OSError("locked")
        return True

    changes = [GenreChange(f"/m/{n}.mp3", ["x"], ["Y"]) for n in "abc"]
    result = apply_genre_changes(changes, writer=writer)
    assert result.applied == ["/m/a.mp3", "/m/c.mp3"]
    assert result.failed == ["/m/b.mp3"]


def test_apply_honours_cancellation_between_files():
    cancel = threading.Event()
    written = []

    def writer(p, _g):
        written.append(p)
        cancel.set()  # cancel after the first write
        return True

    changes = [GenreChange(f"/m/{n}.mp3", ["x"], ["Y"]) for n in "ab"]
    result = apply_genre_changes(changes, cancel_event=cancel, writer=writer)
    assert written == ["/m/a.mp3"]
    assert result.cancelled is True
    assert result.applied == ["/m/a.mp3"]


# ── mapping persistence ───────────────────────────────────────────────────


def test_save_and_load_mapping_round_trip_preserves_nulls(tmp_path):
    mapping = {"rock & roll": ["Rock"], "90s": None}
    path = nc.save_mapping(str(tmp_path), mapping)
    assert path.endswith(".genre_mapping.json")
    loaded, loaded_path = nc.load_mapping(str(tmp_path))
    assert loaded == mapping
    assert loaded_path == path
    assert json.loads(open(path, encoding="utf-8").read())["90s"] is None


def test_load_mapping_missing_or_corrupt_is_empty(tmp_path):
    assert nc.load_mapping(str(tmp_path))[0] == {}
    (tmp_path / "Docs").mkdir()
    (tmp_path / "Docs" / ".genre_mapping.json").write_text("{nope", encoding="utf-8")
    assert nc.load_mapping(str(tmp_path))[0] == {}
