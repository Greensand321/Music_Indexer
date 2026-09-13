"""The library snapshot — above all, its folder policy.

The Indexer and Duplicate Finder skip Not Sorted/, Quarantine/ and Manual
Review/. For this feature those hold music the user already owns, so excluding
them would report owned tracks as missing and cause the exact duplicate download
the feature exists to prevent. The first test here guards that.
"""
from __future__ import annotations

import os

import pytest

from playlist_gap_snapshot import (
    DEFAULT_INCLUDE,
    build_snapshot,
    extract_video_id,
    iter_audio_files,
    normalize_filename,
    tokens_of,
)


def make_library(root, relpaths):
    for rel in relpaths:
        path = os.path.join(root, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"not audio")
    return root


# ── folder policy ────────────────────────────────────────────────────────────


def test_not_sorted_counts_as_owned(tmp_path):
    """The regression that protects the whole feature's correctness."""
    root = make_library(str(tmp_path), ["Not Sorted/Yotto - Radiate.opus"])
    found = [os.path.relpath(p, root) for p in iter_audio_files(root)]
    assert found == [os.path.join("Not Sorted", "Yotto - Radiate.opus")]


@pytest.mark.parametrize("folder", ["Not Sorted", "Quarantine", "Manual Review"])
def test_owned_reserved_folders_are_scanned(tmp_path, folder):
    root = make_library(str(tmp_path / folder.replace(" ", "_")), [f"{folder}/track.flac"])
    assert len(list(iter_audio_files(root))) == 1


@pytest.mark.parametrize("folder", ["Trash", "Docs", "Playlists"])
def test_non_music_reserved_folders_are_skipped(tmp_path, folder):
    root = make_library(str(tmp_path / folder), [f"{folder}/track.flac"])
    assert list(iter_audio_files(root)) == []


def test_nested_excluded_folder_is_skipped(tmp_path):
    root = make_library(str(tmp_path), ["By Artist/Trash/x.flac", "By Artist/keep.flac"])
    found = [os.path.basename(p) for p in iter_audio_files(root)]
    assert found == ["keep.flac"]


def test_counts_report_each_area(tmp_path):
    root = make_library(str(tmp_path), [
        "By Artist/a.flac", "Not Sorted/b.opus", "Quarantine/c.mp3",
    ])
    snap = build_snapshot(root, read_tags=lambda _p: {})
    assert snap.counts == {"Library": 1, "Not Sorted": 1, "Quarantine": 1}


def test_non_audio_files_are_ignored(tmp_path):
    root = make_library(str(tmp_path), ["By Artist/a.flac", "By Artist/cover.jpg"])
    assert len(list(iter_audio_files(root))) == 1


# ── identity recovery ────────────────────────────────────────────────────────


def test_video_id_from_filename():
    assert extract_video_id("/m/Cupid [Qc7_zRjH808].opus") == "Qc7_zRjH808"


@pytest.mark.parametrize("tag", ["purl", "comment", "website"])
def test_video_id_from_provenance_tag(tag):
    tags = {tag: "downloaded from https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    assert extract_video_id("/m/song.opus", tags) == "dQw4w9WgXcQ"


def test_video_id_from_short_url():
    assert extract_video_id("/m/s.opus", {"purl": "https://youtu.be/dQw4w9WgXcQ"}) == "dQw4w9WgXcQ"


def test_no_video_id_is_none():
    assert extract_video_id("/m/song.opus", {"comment": "ripped from CD"}) is None


def test_snapshot_indexes_by_video_id(tmp_path):
    root = make_library(str(tmp_path), ["Not Sorted/Cupid [Qc7_zRjH808].opus"])
    snap = build_snapshot(root, read_tags=lambda _p: {})
    assert "Qc7_zRjH808" in snap.by_video_id


# ── indexes ──────────────────────────────────────────────────────────────────


def test_filename_normalization_drops_the_id_suffix():
    assert normalize_filename("/m/Yotto - Radiate [abcdefghijk].opus") == "yotto radiate"


def test_tokens_drop_bare_numbers():
    assert "1979" not in tokens_of("Smashing Pumpkins 1979")


def test_artist_title_index_is_built_from_tags(tmp_path):
    root = make_library(str(tmp_path), ["By Artist/x.flac"])
    snap = build_snapshot(root, read_tags=lambda _p: {"artist": "HOME", "title": "Resonance"})
    assert ("home", "resonance") in snap.by_artist_title
    assert "home" in snap.artists


def test_cache_hit_avoids_reading_tags(tmp_path):
    root = make_library(str(tmp_path), ["By Artist/x.flac"])
    calls = []

    def reader(path):
        calls.append(path)
        return {}

    def cache(path, _db):
        return "FP", {"tags": {"artist": "HOME", "title": "Resonance"}, "duration": 212}

    snap = build_snapshot(root, cache_db="/tmp/x.db", read_tags=reader, read_cache=cache)
    assert calls == []
    assert snap.tracks[0].duration == 212
    assert snap.tracks[0].fingerprint == "FP"


def test_a_failing_tag_read_does_not_abort_the_scan(tmp_path):
    root = make_library(str(tmp_path), ["By Artist/a.flac", "By Artist/b.flac"])

    def reader(path):
        if path.endswith("a.flac"):
            raise OSError("boom")
        return {"title": "B"}

    snap = build_snapshot(root, read_tags=reader)
    assert len(snap) == 2
    assert snap.errors and "boom" in snap.errors[0]


def test_cancellation_stops_early(tmp_path):
    root = make_library(str(tmp_path), [f"By Artist/{i}.flac" for i in range(5)])
    seen = []

    def cancel():
        return len(seen) >= 2

    def reader(path):
        seen.append(path)
        return {}

    snap = build_snapshot(root, read_tags=reader, should_cancel=cancel)
    assert len(snap) < 5


def test_shortlist_by_tokens(tmp_path):
    root = make_library(str(tmp_path), ["By Artist/a.flac", "By Artist/b.flac"])
    tags = {"a.flac": {"title": "Radiate"}, "b.flac": {"title": "Resonance"}}
    snap = build_snapshot(root, read_tags=lambda p: tags[os.path.basename(p)])
    assert len(snap.shortlist_by_tokens(["radiate"])) == 1
    assert snap.shortlist_by_tokens(["nothing"]) == set()


def test_apostrophes_normalize_consistently(tmp_path):
    """A wanted "Don't Stop" must find a file tagged "Dont Stop"."""
    root = make_library(str(tmp_path), ["By Artist/x.flac"])
    snap = build_snapshot(root, read_tags=lambda _p: {"artist": "A", "title": "Dont Stop"})
    from playlist_gap_parse import compare_key
    assert ("a", compare_key("Don't Stop")) in snap.by_artist_title


def test_injected_readers_are_resolved_at_call_time(tmp_path, monkeypatch):
    """Regression: the readers were bound as default arguments, so patching the
    module attribute silently had no effect and the snapshot read no tags —
    a test could pass against a reader that was never called."""
    import playlist_gap_snapshot as module

    root = make_library(str(tmp_path), ["By Artist/x.flac"])
    monkeypatch.setattr(
        module, "_default_tag_reader", lambda _p: {"artist": "HOME", "title": "Resonance"}
    )
    snap = module.build_snapshot(root)
    assert ("home", "resonance") in snap.by_artist_title
