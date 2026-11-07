"""Fill from MusicBrainz: preview-first and non-destructive (META-03/04/25).

MusicBrainz is never contacted: lookups are injected fakes, or the
``musicbrainzngs`` module the real lookup uses is monkeypatched.
"""
from __future__ import annotations

import io

import pytest

from _helpers.meta_tags import build_genre_library, get_genres, needs_ffmpeg

pytestmark = needs_ffmpeg


def _fake_lookup(genres_by_title, calls):
    from controllers.normalize_controller import FillLookup
    from utils.audio_metadata_reader import read_tags

    def lookup(path):
        calls.append(path)
        title = read_tags(path).get("title")
        if title in genres_by_title:
            return FillLookup(genres=list(genres_by_title[title]), status="matched",
                              detail=f"MusicBrainz: {title}")
        return FillLookup(status="no_match", detail="no MusicBrainz match")

    return lookup


def test_fill_plan_looks_up_but_never_writes_and_keeps_existing_genres(tmp_path):
    from controllers import normalize_controller as nc

    files = build_genre_library(tmp_path, [
        ("multi.flac", "A", "Multi", ["Rock", "Pop"]),   # 2+ genres: untouched
        ("one.mp3", "A", "One", ["rock & roll"]),        # 1 genre: MB tags added after it
        ("none.opus", "A", "None", []),                  # .opus is scanned too
        ("miss.m4a", "A", "Miss", ["Jazz"]),             # no MB match
    ])
    before = {k: get_genres(p) for k, p in files.items()}
    calls = []
    lookup = _fake_lookup({"Multi": ["X"], "One": ["Rock", "Rock & Roll", "Oldies"],
                           "None": ["Electronic", "House"]}, calls)
    plan = nc.plan_genre_fill(str(tmp_path), lookup=lookup, rate_limit=0)

    # Dry run: nothing on disk changed.
    assert {k: get_genres(p) for k, p in files.items()} == before
    # Files with 2+ genres are not even looked up.
    assert str(files["multi.flac"]) not in calls
    assert plan.skipped_multi == [str(files["multi.flac"])]
    assert [p for p, _why in plan.no_match] == [str(files["miss.m4a"])]
    changes = {c.path: (c.before, c.after) for c in plan.changes}
    assert changes == {
        # existing kept; "Rock & Roll" is already there (case-insensitively)
        str(files["one.mp3"]): (["rock & roll"], ["rock & roll", "Rock", "Oldies"]),
        str(files["none.opus"]): ([], ["Electronic", "House"]),
    }

    result = nc.apply_genre_changes(plan.changes)
    assert sorted(result.applied) == sorted(changes)
    assert get_genres(files["one.mp3"]) == ["rock & roll", "Rock", "Oldies"]
    assert get_genres(files["none.opus"]) == ["Electronic", "House"]
    assert get_genres(files["multi.flac"]) == ["Rock", "Pop"]
    assert get_genres(files["miss.m4a"]) == ["Jazz"]


def test_fill_plan_rate_limits_and_cancels(tmp_path):
    from controllers import normalize_controller as nc

    build_genre_library(tmp_path, [(f"t{i}.flac", "A", f"T{i}", []) for i in range(4)])
    calls, naps = [], []
    lookup = _fake_lookup({}, calls)
    nc.plan_genre_fill(str(tmp_path), lookup=lookup, rate_limit=0.25, sleep=naps.append)
    assert len(calls) == 4
    assert sum(naps) == pytest.approx(0.75, abs=0.11)  # a wait between consecutive lookups

    calls.clear()
    plan = nc.plan_genre_fill(str(tmp_path), lookup=lookup, rate_limit=0,
                              should_cancel=lambda: len(calls) >= 2)
    assert plan.cancelled and len(calls) == 2


def test_lookup_sets_user_agent_and_ranks_by_numeric_count(tmp_path, monkeypatch):
    """musicbrainzngs returns tag counts as *strings*; they were all treated as 0."""
    import update_genres

    files = build_genre_library(tmp_path, [("a.flac", "Art", "Song", [])])
    calls = []
    monkeypatch.setattr(update_genres.musicbrainzngs, "set_useragent",
                        lambda *a, **k: calls.append(("ua",) + a))

    def search_recordings(**kwargs):
        calls.append(("search", kwargs["artist"], kwargs["recording"]))
        return {"recording-list": [{"title": "Song", "tag-list": [
            {"name": "ambient", "count": "2"}, {"name": "techno", "count": "9"},
            {"name": "house", "count": "5"}, {"name": "acid", "count": "1"},
        ]}]}

    monkeypatch.setattr(update_genres.musicbrainzngs, "search_recordings", search_recordings)
    found = update_genres.lookup_musicbrainz_genres(str(files["a.flac"]))
    assert calls[0][0] == "ua" and calls[1] == ("search", "Art", "Song")
    assert found.status == "matched"
    assert found.genres == ["techno", "house", "ambient"]


def test_cli_process_file_respects_existing_genres(tmp_path, monkeypatch):
    """META-04: the 2+ genre guard never fired because only one value was read."""
    import update_genres

    files = build_genre_library(tmp_path, [
        ("multi.flac", "Art", "Multi", ["Rock", "Pop"]),
        ("one.flac", "Art", "One", ["Rock"]),
    ])
    monkeypatch.setattr(update_genres, "RATE_LIMIT_SECONDS", 0)
    monkeypatch.setattr(update_genres.musicbrainzngs, "set_useragent", lambda *a, **k: None)
    monkeypatch.setattr(
        update_genres.musicbrainzngs, "search_recordings",
        lambda **k: {"recording-list": [{"tag-list": [{"name": "electronic", "count": "5"}]}]},
    )
    log = io.StringIO()
    update_genres.process_file(str(files["multi.flac"]), log)
    update_genres.process_file(str(files["one.flac"]), log)
    assert get_genres(files["multi.flac"]) == ["Rock", "Pop"]
    assert get_genres(files["one.flac"]) == ["Rock", "electronic"]
    assert "skipped (already has genres" in log.getvalue()
