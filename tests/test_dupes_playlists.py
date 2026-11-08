"""Duplicate plans and executions must handle the playlists users really have.

Before:
* a cp1252 ``.m3u`` (what Windows tools write) raised ``UnicodeDecodeError``
  in the planner's playlist index — the whole duplicate scan crashed — and in
  the executor's step 1, so nothing could be executed;
* ``.m3u8`` playlists were never looked at, so they kept pointing at the
  quarantined loser;
* any playlist that already had one dead entry aborted every execution whose
  losers it referenced ("Playlist validation failed");
* with every track under one top-level folder the planner looked for
  ``<that folder>/Playlists`` and planned no playlist rewrites at all.
"""
from __future__ import annotations

import os
from pathlib import Path

from duplicate_consolidation import build_consolidation_plan
from duplicate_consolidation_executor import execute_consolidation_plan
from _helpers.dupes_plans import config, write

TAGS = {"artist": "Artist", "title": "Song", "album": "Album"}


def _pair(root: Path, sub: str = ""):
    base = root / sub if sub else root
    winner = write(base / "ArtistA" / "Album" / "Song.flac", b"winner")
    loser = write(base / "ArtistB" / "Other" / "Song.mp3", b"loser")
    tracks = [
        {"path": str(winner), "fingerprint": "1 2 3 4 5 6 7 8", "ext": ".flac", "tags": dict(TAGS),
         "discovery": {"scan_roots": [str(root)]}},
        {"path": str(loser), "fingerprint": "1 2 3 4 5 6 7 8", "ext": ".mp3", "tags": dict(TAGS),
         "discovery": {"scan_roots": [str(root)]}},
    ]
    return winner, loser, tracks


def _execute(root: Path, the_plan):
    return execute_consolidation_plan(the_plan, config(root, allow_review_required=True))


def test_cp1252_playlist_is_planned_and_rewritten_in_its_own_encoding(tmp_path):
    winner, loser, tracks = _pair(tmp_path)
    playlist = tmp_path / "Playlists" / "Mix.m3u"
    playlist.parent.mkdir()
    original = "#EXTM3U\r\nCafé del Mar.mp3\r\n../ArtistB/Other/Song.mp3\r\n".encode("cp1252")
    playlist.write_bytes(original)

    the_plan = build_consolidation_plan(tracks)  # crashed: UnicodeDecodeError
    (group,) = the_plan.groups
    assert group.playlist_rewrites == {str(loser): str(winner)}

    result = _execute(tmp_path, the_plan)

    assert result.success, [a.detail for a in result.actions if a.status in {"failed", "blocked"}]
    assert playlist.read_bytes() == "#EXTM3U\r\nCafé del Mar.mp3\r\n../ArtistA/Album/Song.flac\r\n".encode("cp1252")
    assert not loser.exists()


def test_m3u8_playlists_are_rewritten_too(tmp_path):
    winner, loser, tracks = _pair(tmp_path)
    playlist = tmp_path / "Playlists" / "Mix.m3u8"
    playlist.parent.mkdir()
    playlist.write_text(f"{loser}\n", encoding="utf-8")

    result = _execute(tmp_path, build_consolidation_plan(tracks))

    assert result.success, [a.detail for a in result.actions if a.status in {"failed", "blocked"}]
    assert playlist.read_text(encoding="utf-8") == f"{winner}\n"


def test_an_already_dead_entry_does_not_block_execution(tmp_path):
    winner, loser, tracks = _pair(tmp_path)
    playlist = tmp_path / "Playlists" / "Mix.m3u"
    playlist.parent.mkdir()
    playlist.write_text("../Gone/Deleted long ago.mp3\n../ArtistB/Other/Song.mp3\n", encoding="utf-8")

    result = _execute(tmp_path, build_consolidation_plan(tracks))

    assert result.success, [a.detail for a in result.actions if a.status in {"failed", "blocked"}]
    assert playlist.read_text(encoding="utf-8") == "../Gone/Deleted long ago.mp3\n../ArtistA/Album/Song.flac\n"
    assert not loser.exists()


def test_playlists_are_found_from_the_scan_root(tmp_path):
    winner, loser, tracks = _pair(tmp_path, sub="iTunes")  # every track under one folder
    playlist = tmp_path / "Playlists" / "Mix.m3u"
    playlist.parent.mkdir()
    playlist.write_text(os.path.relpath(loser, playlist.parent) + "\n", encoding="utf-8")

    the_plan = build_consolidation_plan(tracks)

    assert the_plan.groups[0].playlist_rewrites == {str(loser): str(winner)}
    assert the_plan.impacted_playlists == [str(playlist)]


def test_entry_that_cannot_be_encoded_stops_before_any_loser_moves(tmp_path):
    winner = write(tmp_path / "A" / "曲.flac", b"winner")  # a name cp1252 cannot spell
    loser = write(tmp_path / "B" / "Song.mp3", b"loser")
    tracks = [
        {"path": str(winner), "fingerprint": "1 2 3 4", "ext": ".flac", "tags": dict(TAGS),
         "discovery": {"scan_roots": [str(tmp_path)]}},
        {"path": str(loser), "fingerprint": "1 2 3 4", "ext": ".mp3", "tags": dict(TAGS),
         "discovery": {"scan_roots": [str(tmp_path)]}},
    ]
    playlist = tmp_path / "Playlists" / "Mix.m3u"
    playlist.parent.mkdir()
    original = "Café.mp3\n../B/Song.mp3\n".encode("cp1252")
    playlist.write_bytes(original)

    result = _execute(tmp_path, build_consolidation_plan(tracks))

    assert not result.success
    assert playlist.read_bytes() == original
    assert loser.exists()
