"""Phase 0 of the Playlist Gap spec — the two prerequisites.

1. The metadata readers must surface source-provenance tags (`comment`, `purl`,
   `website`). `yt-dlp --embed-metadata` writes the originating video URL into
   one of those, and without it the app cannot tell which upload a downloaded
   file came from — which is what the spec's rung 0b and the whole verification
   pass depend on.

2. `store_fingerprint` must populate the cache's `normalized_artist` /
   `normalized_title` / `normalized_album` columns. They existed but only the
   legacy Tkinter writer filled them, so in the Qt app they were always NULL.

See docs/playlist_gap_spec.md §1 and §4.2.
"""
from __future__ import annotations

import sqlite3

import pytest

import fingerprint_cache
from fingerprint_cache import normalized_key
from utils.audio_metadata_reader import TAG_KEYS, read_metadata_from_mutagen

VIDEO_URL = "https://www.youtube.com/watch?v=Qc7_zRjH808"


class DummyTags(dict):
    def getall(self, key):
        value = self.get(key)
        return value if isinstance(value, list) else ([] if value is None else [value])


class DummyAudio:
    def __init__(self, tags):
        self.tags = tags


# ── 1 · provenance tags reach read_metadata ──────────────────────────────────


def test_tag_keys_include_provenance_fields() -> None:
    for key in ("comment", "purl", "website"):
        assert key in TAG_KEYS


def test_vorbis_purl_is_read() -> None:
    """FLAC/Ogg: yt-dlp's URL lands in a plain PURL comment field."""
    audio = DummyAudio(DummyTags({"title": ["Cupid"], "PURL": [VIDEO_URL]}))
    tags, _covers, error, _hint = read_metadata_from_mutagen(audio, "song.flac")
    assert error is None
    assert tags["purl"] == VIDEO_URL


def test_id3_txxx_purl_is_read() -> None:
    """MP3: FFmpeg writes non-standard keys as TXXX frames."""
    audio = DummyAudio(DummyTags({"TIT2": ["Cupid"], "TXXX:purl": [VIDEO_URL]}))
    tags, _covers, _error, _hint = read_metadata_from_mutagen(audio, "song.mp3")
    assert tags["purl"] == VIDEO_URL


def test_id3_woas_is_read_as_purl() -> None:
    """MP3: some writers use the official source-webpage frame instead."""
    audio = DummyAudio(DummyTags({"TIT2": ["Cupid"], "WOAS": [VIDEO_URL]}))
    tags, _covers, _error, _hint = read_metadata_from_mutagen(audio, "song.mp3")
    assert tags["purl"] == VIDEO_URL


def test_id3_comment_is_read() -> None:
    audio = DummyAudio(DummyTags({"TIT2": ["Cupid"], "COMM": ["from " + VIDEO_URL]}))
    tags, _covers, _error, _hint = read_metadata_from_mutagen(audio, "song.mp3")
    assert tags["comment"] == "from " + VIDEO_URL


def test_mp4_freeform_purl_is_read() -> None:
    """M4A: FFmpeg writes unknown keys as iTunes freeform atoms."""
    audio = DummyAudio(
        DummyTags(
            {
                "\xa9nam": ["Cupid"],
                "\xa9cmt": ["downloaded"],
                "----:com.apple.iTunes:purl": [VIDEO_URL],
            }
        )
    )
    tags, _covers, _error, _hint = read_metadata_from_mutagen(audio, "song.m4a")
    assert tags["purl"] == VIDEO_URL
    assert tags["comment"] == "downloaded"


def test_opus_reader_exposes_provenance_keys() -> None:
    """Opus is yt-dlp's default YouTube container, and has its own reader."""
    from utils import opus_metadata_reader

    for key in ("comment", "purl", "website"):
        assert key in opus_metadata_reader.TAG_KEYS
        assert key in opus_metadata_reader._blank_tags()


def test_missing_provenance_tags_are_none_not_absent() -> None:
    audio = DummyAudio(DummyTags({"title": ["Cupid"]}))
    tags, _covers, _error, _hint = read_metadata_from_mutagen(audio, "song.flac")
    assert tags["purl"] is None
    assert tags["comment"] is None
    assert tags["website"] is None


# ── 2 · normalized_key ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("FIFTY FIFTY", "fifty fifty"),
        ("Cupid (Twin Version)", "cupid twin version"),
        ("Beyoncé", "beyonc"),  # matches the legacy algorithm, which drops non-ascii
        ("  spaced   out  ", "spaced out"),
        ("", None),
        ("!!!", None),
        (None, None),
        (123, None),
    ],
)
def test_normalized_key(raw, expected) -> None:
    assert normalized_key(raw) == expected


def test_normalized_key_matches_near_duplicate_detector() -> None:
    """A second spelling of this algorithm would silently split the index."""
    from near_duplicate_detector import _normalized

    for sample in ("The Beatles", "Cupid (Twin Version)", "AC/DC", "  x  "):
        assert (normalized_key(sample) or "") == _normalized(sample)


# ── 3 · store_fingerprint fills the normalized columns ───────────────────────


def _row(db_path: str, path: str) -> sqlite3.Row:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT normalized_artist, normalized_title, normalized_album "
            "FROM fingerprints WHERE path = ?",
            (path,),
        ).fetchone()
    finally:
        conn.close()


@pytest.fixture
def cache(tmp_path):
    db_path = str(tmp_path / "cache" / "fp.db")
    audio = tmp_path / "Cupid.flac"
    audio.write_bytes(b"not really audio")
    fingerprint_cache.ensure_fingerprint_cache(db_path)
    yield db_path, str(audio)
    fingerprint_cache.shutdown_fingerprint_writer()


def test_store_fingerprint_derives_normalized_columns(cache) -> None:
    """The Qt Duplicates workspace passes tags but not normalized values."""
    db_path, audio = cache
    assert fingerprint_cache.store_fingerprint(
        audio,
        db_path,
        215,
        "1,2,3",
        ext=".flac",
        tags={"artist": "FIFTY FIFTY", "title": "Cupid", "album": "The Beginning"},
        flush=True,
    )
    row = _row(db_path, audio)
    assert row["normalized_artist"] == "fifty fifty"
    assert row["normalized_title"] == "cupid"
    assert row["normalized_album"] == "the beginning"


def test_store_fingerprint_falls_back_to_albumartist(cache) -> None:
    db_path, audio = cache
    fingerprint_cache.store_fingerprint(
        audio, db_path, 215, "1,2,3",
        tags={"albumartist": "Various Artists", "title": "Cupid"},
        flush=True,
    )
    assert _row(db_path, audio)["normalized_artist"] == "various artists"


def test_explicit_normalized_values_win(cache) -> None:
    """The legacy Tkinter writer passes its own; derivation must not override."""
    db_path, audio = cache
    fingerprint_cache.store_fingerprint(
        audio, db_path, 215, "1,2,3",
        tags={"artist": "Ignored", "title": "Ignored"},
        normalized_artist="explicit artist",
        normalized_title="explicit title",
        normalized_album="explicit album",
        flush=True,
    )
    row = _row(db_path, audio)
    assert row["normalized_artist"] == "explicit artist"
    assert row["normalized_title"] == "explicit title"


def test_store_fingerprint_without_tags_leaves_columns_null(cache) -> None:
    db_path, audio = cache
    fingerprint_cache.store_fingerprint(audio, db_path, 215, "1,2,3", flush=True)
    row = _row(db_path, audio)
    assert row["normalized_artist"] is None
    assert row["normalized_title"] is None


# ── 4 · the public entry point the snapshot actually calls ───────────────────


class DummyFrame:
    """Mutagen frames expose their value via ``.text``, not by being a str."""

    def __init__(self, text):
        self.text = text


def test_read_tags_returns_provenance_as_plain_strings(monkeypatch) -> None:
    """`read_tags` is what the library snapshot calls; it must coerce frames."""
    import utils.audio_metadata_reader as reader

    audio = DummyAudio(
        DummyTags({"title": [DummyFrame("Cupid")], "PURL": [DummyFrame(VIDEO_URL)]})
    )
    monkeypatch.setattr(reader, "MutagenFile", lambda _path: audio)

    tags = reader.read_tags("song.flac")
    assert tags["purl"] == VIDEO_URL
    assert isinstance(tags["purl"], str)


def test_normalize_text_tags_covers_provenance_keys() -> None:
    """By this point `_first_value` has already unwrapped mutagen frames, so the
    realistic inputs are str, bytes or None."""
    from utils.audio_metadata_reader import _normalize_text_tags

    out = _normalize_text_tags(
        {"purl": b"https://youtu.be/abc", "comment": "  note  ", "website": None}
    )
    assert out["purl"] == "https://youtu.be/abc"
    assert out["comment"] == "note"
    assert out["website"] is None
