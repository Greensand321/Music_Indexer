"""Metadata readers and year normaliser (listening backends audit).

* raw-byte tag text was decoded as UTF-16 whenever UTF-8 failed, which
  succeeds on almost any even-length input (``b"Caf\\xe9"`` -> two CJK
  characters);
* M4A ``trkn``/``disk`` lost their totals ("3/12" read as the int 3);
* the Opus reader ignored "ALBUM ARTIST"/"ALBUM_ARTIST";
* an Ogg Vorbis (or FLAC) file named ``.opus`` read as blank;
* ``normalize_year`` raised on NaN/inf;
* readers never raise on a bad file, whatever its format.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from utils import audio_metadata_reader as amr
from utils import opus_metadata_reader as omr
from utils.text_decoding import decode_tag_bytes
from utils.year_normalizer import normalize_year

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _encode(path: Path, codec_args: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
         *codec_args, str(path)],
        check=True,
    )
    return path


# ── bytes ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw, text", [
    (b"Caf\xe9", "Café"),                          # Windows-1252 / Latin-1
    ("Café".encode("utf-8"), "Café"),
    (b"\xef\xbb\xbfCaf\xc3\xa9", "Café"),          # UTF-8 with BOM
    ("Café".encode("utf-16"), "Café"),             # UTF-16 with BOM
    (b"Ab", "Ab"),                                 # even length, not UTF-16
    (b"Title\x00", "Title"),                       # C-string terminator
    (b"\x81\x8d\x8f", "\x81\x8d\x8f"),             # undefined in cp1252: Latin-1
])
def test_tag_bytes_are_not_misread_as_utf16(raw, text):
    assert decode_tag_bytes(raw) == text
    assert amr._normalize_text_value(raw, "title") == text
    assert amr._first_value([raw]) == text
    assert omr._first_value([raw]) == text


# ── M4A ───────────────────────────────────────────────────────────────────


@needs_ffmpeg
def test_m4a_track_and_disc_keep_their_totals(tmp_path):
    from mutagen.mp4 import MP4

    path = _encode(tmp_path / "t.m4a", ["-c:a", "aac"])
    audio = MP4(str(path))
    audio["\xa9nam"] = ["Song"]
    audio["trkn"] = [(3, 12)]
    audio["disk"] = [(1, 2)]
    audio.save()

    tags, _covers, error, _hint = amr.read_metadata(str(path))

    assert error is None
    assert (tags["track"], tags["tracknumber"]) == (3, "3/12")
    assert (tags["disc"], tags["discnumber"]) == (1, "1/2")


# ── Opus ──────────────────────────────────────────────────────────────────


@needs_ffmpeg
@pytest.mark.parametrize("key", ["ALBUMARTIST", "ALBUM ARTIST", "album_artist", "AlbumArtist"])
def test_opus_album_artist_spellings(tmp_path, key):
    from mutagen.oggopus import OggOpus

    path = _encode(tmp_path / "t.opus", ["-c:a", "libopus"])
    audio = OggOpus(str(path))
    audio[key] = "Various Artists"
    audio.save()

    tags, _covers, error = omr.read_opus_metadata(str(path))

    assert error is None and tags["albumartist"] == "Various Artists"


@needs_ffmpeg
def test_ogg_vorbis_named_opus_is_read_as_what_it_is(tmp_path):
    from mutagen.oggvorbis import OggVorbis

    path = _encode(tmp_path / "real.ogg", ["-c:a", "libvorbis"])
    audio = OggVorbis(str(path))
    audio["title"] = "Misnamed"
    audio["date"] = "2001-05-04"
    audio.save()
    misnamed = path.with_name("misnamed.opus")
    path.rename(misnamed)

    tags, _covers, error, hint = amr.read_metadata(str(misnamed))

    assert error is None, error
    assert tags["title"] == "Misnamed" and tags["year"] == "2001"
    assert "not an Opus stream" in hint
    assert amr.read_tags(str(misnamed))["title"] == "Misnamed"


def test_unreadable_opus_has_the_same_shape_as_other_unreadable_files(tmp_path):
    bad = tmp_path / "bad.opus"
    bad.write_bytes(b"OggS\x00garbage")
    other = tmp_path / "bad.mp3"
    other.write_bytes(b"garbage")

    opus_tags, _c, opus_error, _h = amr.read_metadata(str(bad))
    mp3_tags, _c, mp3_error, _h = amr.read_metadata(str(other))

    assert opus_error.startswith("read failed") and mp3_error.startswith("read failed")
    assert opus_tags == mp3_tags  # blank, year "Unknown"


# ── never raise ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name, data", [
    ("empty.mp3", b""),
    ("trunc.mp3", b"ID3\x04\x00\x00\x00\x00\x00\x10" + b"\x00" * 5),
    ("trunc.flac", b"fLaC\x00\x00"),
    ("garbage.m4a", b"\x00\x00\x00\x18ftypM4A "),
    ("garbage.ogg", b"OggS" + b"\xff" * 40),
    ("garbage.opus", b"OggS\x00garbage"),
    ("garbage.wav", b"RIFF\x10\x00\x00\x00WAVE"),
    ("text.aac", b"hello world"),
])
def test_readers_never_raise_on_a_broken_file(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    tags, covers, error, _hint = amr.read_metadata(str(path))
    assert isinstance(tags, dict) and set(amr.TAG_KEYS) <= set(tags)
    assert covers == [] or all(isinstance(c, bytes) for c in covers)
    assert error is None or isinstance(error, str)
    assert isinstance(amr.read_tags(str(path)), dict)
    assert isinstance(amr.read_genres(str(path)), list)


def test_missing_file_does_not_raise(tmp_path):
    tags, error = amr.read_tags_checked(str(tmp_path / "gone.flac"))
    assert error.startswith("read failed") and tags["title"] is None


# ── years ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value, year", [
    (float("nan"), "Unknown"),
    (float("inf"), "Unknown"),
    (float("-inf"), "Unknown"),
    (True, "Unknown"),
    (1999.0, "1999"),
    ("2001-05-04", "2001"),
    ("2001/05", "2001"),
    ("c. 1999", "1999"),
    ("05/04/2001", "2001"),
    ("20010504", "2001"),
    ("(P) 1987 Sony", "1987"),
    ("", "Unknown"),
    (None, "Unknown"),
])
def test_normalize_year_forms(value, year):
    assert normalize_year(value) == year
