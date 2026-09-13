"""Reading a wanted list, and being honest about what it contains."""
from __future__ import annotations

import pytest

from playlist_gap_sources import (
    CsvSource,
    SourceSpec,
    detect_capabilities,
    extract_video_id,
    merge_wanted,
    parse_duration,
    propose_mapping,
    row_id_for,
)
from playlist_gap_types import WantedTrack

#: The real shape of a TuneMyMusic export of a YouTube Music playlist: the
#: columns are all there and almost none of the data is.
REAL_EXPORT = (
    "Track name,Artist name,Album,Playlist name,Type,ISRC\n"
    ",,,Liked videos,Favorite,\n"
    "Self Aware x Babydoll,,,Liked videos,Favorite,\n"
    "FIFTY FIFTY - Cupid (Twin Version),,,Liked videos,Favorite,\n"
    "Kate Bush - Running Up That Hill (A Deal With God),,,Liked videos,Favorite,\n"
)


def write_csv(tmp_path, text, name="export.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def read(tmp_path, text, **kwargs):
    spec = SourceSpec(name="Liked videos", kind="csv", location=write_csv(tmp_path, text), **kwargs)
    return CsvSource().read(spec)


# ── the real export ──────────────────────────────────────────────────────────


def test_real_export_maps_its_columns(tmp_path):
    result = read(tmp_path, REAL_EXPORT)
    assert result.mapping["display"] == "Track name"
    assert result.mapping["artist"] == "Artist name"
    assert result.mapping["isrc"] == "ISRC"


def test_an_empty_column_is_not_a_capability(tmp_path):
    """The ISRC column exists and is empty. Reporting it as usable would waste a
    ladder rung and mislead the UI about how accurate the run can be."""
    caps = read(tmp_path, REAL_EXPORT).capabilities
    assert caps.isrc is False
    assert caps.artist is False
    assert caps.album is False
    assert caps.duration is False
    assert caps.display_string is True


def test_blank_rows_are_kept_and_reported(tmp_path):
    """Silently dropping a row is the failure mode this feature exists to stop."""
    result = read(tmp_path, REAL_EXPORT)
    assert len(result.tracks) == 4
    assert result.blank_rows == 1
    assert any(not t.available for t in result.tracks)
    assert any("no title" in w for w in result.warnings)


def test_display_survives_verbatim(tmp_path):
    result = read(tmp_path, REAL_EXPORT)
    assert "FIFTY FIFTY - Cupid (Twin Version)" in [t.display for t in result.tracks]


# ── a richer export ──────────────────────────────────────────────────────────


RICH = (
    "Title,Artist,Album,Duration,ISRC,URL\n"
    "Cupid,FIFTY FIFTY,The Beginning,3:35,USUM72302108,https://youtu.be/Qc7_zRjH808\n"
    "Radiate,Yotto,Erased Dreams,6:12,GBUM71904561,https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
)


def test_rich_export_reports_full_capabilities(tmp_path):
    caps = read(tmp_path, RICH).capabilities
    assert caps.artist and caps.album and caps.duration and caps.isrc and caps.video_id


def test_rich_export_parses_fields(tmp_path):
    first = read(tmp_path, RICH).tracks[0]
    assert first.artist == "FIFTY FIFTY"
    assert first.title == "Cupid"
    assert first.duration == 215
    assert first.video_id == "Qc7_zRjH808"
    assert first.display == "FIFTY FIFTY - Cupid"


# ── mapping and detection ────────────────────────────────────────────────────


def test_mapping_is_by_synonym_not_position():
    mapping = propose_mapping(["ISRC", "Album", "Song Title", "Artist Name"])
    assert mapping["title"] == "Song Title"
    assert mapping["artist"] == "Artist Name"


def test_mapping_is_case_and_punctuation_insensitive():
    assert propose_mapping(["track_name"])["display"] == "track_name"


def test_explicit_mapping_overrides_the_proposal(tmp_path):
    text = "col_a,col_b\nCupid,FIFTY FIFTY\n"
    result = read(tmp_path, text, column_mapping={"display": "col_a", "artist": "col_b"})
    assert result.tracks[0].display == "Cupid"
    assert result.tracks[0].artist == "FIFTY FIFTY"


def test_semicolon_delimited_file(tmp_path):
    result = read(tmp_path, "Title;Artist\nCupid;FIFTY FIFTY\n")
    assert result.tracks[0].title == "Cupid"


def test_capabilities_of_an_empty_file(tmp_path):
    caps = read(tmp_path, "Title,Artist\n").capabilities
    assert caps.title is False


def test_sparse_column_is_not_a_capability():
    rows = [{"ISRC": ""} for _ in range(100)] + [{"ISRC": "X"}]
    caps = detect_capabilities(rows, {"isrc": "ISRC"})
    assert caps.isrc is False


# ── durations and ids ────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("3:35", 215), ("1:02:14", 3734), ("215", 215), (215, 215),
    ("215000", 215), (215000, 215), ("", None), (None, None), ("nonsense", None),
])
def test_parse_duration(raw, expected):
    assert parse_duration(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("https://youtu.be/Qc7_zRjH808", "Qc7_zRjH808"),
    ("https://www.youtube.com/watch?v=Qc7_zRjH808&t=1", "Qc7_zRjH808"),
    ("Qc7_zRjH808", "Qc7_zRjH808"),
    ("https://open.spotify.com/track/abc", None),
    ("", None), (None, None),
])
def test_extract_video_id(raw, expected):
    assert extract_video_id(raw) == expected


# ── identity ─────────────────────────────────────────────────────────────────


def test_row_id_prefers_a_real_identity():
    assert row_id_for(WantedTrack(row_id="", display="x", video_id="abc")).startswith("yt:")
    assert row_id_for(WantedTrack(row_id="", display="x", isrc="US-UM7-23-02108")).startswith("isrc:")
    assert row_id_for(WantedTrack(row_id="", display="x")).startswith("s:")


def test_row_id_survives_cosmetic_title_edits():
    a = WantedTrack(row_id="", display="FIFTY FIFTY - Cupid!", source_id="s")
    b = WantedTrack(row_id="", display="fifty  fifty - cupid", source_id="s")
    assert row_id_for(a) == row_id_for(b)


def test_row_id_is_the_same_song_across_sources():
    """One song wanted by three playlists is one song: answered once and
    downloaded once. Per-source bookkeeping is the ledger's job."""
    a = WantedTrack(row_id="", display="Cupid", source_id="one")
    b = WantedTrack(row_id="", display="Cupid", source_id="two")
    assert row_id_for(a) == row_id_for(b)


def test_isrc_identity_ignores_formatting():
    a = WantedTrack(row_id="", display="x", isrc="us-um7-23-02108")
    b = WantedTrack(row_id="", display="x", isrc="USUM72302108")
    assert row_id_for(a) == row_id_for(b)


# ── merging several sources ──────────────────────────────────────────────────


def test_merge_keeps_each_track_once_and_unions_playlists():
    a = WantedTrack(row_id="yt:1", display="Cupid", playlists=("Liked",))
    b = WantedTrack(row_id="yt:1", display="Cupid", playlists=("Deep house",))
    c = WantedTrack(row_id="yt:2", display="Radiate", playlists=("Liked",))
    merged = merge_wanted([[a, c], [b]])
    assert len(merged) == 2
    assert set(merged[0].playlists) == {"Liked", "Deep house"}


def test_merge_fills_gaps_from_the_richer_row():
    thin = WantedTrack(row_id="yt:1", display="Cupid")
    rich = WantedTrack(row_id="yt:1", display="Cupid", duration=215, album="The Beginning")
    merged = merge_wanted([[thin], [rich]])
    assert merged[0].duration == 215
    assert merged[0].album == "The Beginning"
