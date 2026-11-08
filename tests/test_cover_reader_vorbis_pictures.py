"""OGG/Opus embedded covers are read, not reported missing.

Ogg Vorbis and Opus store embedded artwork as a ``METADATA_BLOCK_PICTURE``
Vorbis comment (a base64 FLAC picture block). The general reader,
``utils.audio_metadata_reader._extract_cover_payloads``, only knew FLAC
picture blocks, ID3 ``APIC``, MP4 ``covr`` and ASF ``WM/Picture``, so every
OGG/Opus cover looked missing — including the ones the duplicate executor
embeds into a winner (``duplicate_consolidation_executor`` reads artwork back
through the same function). The files here are real, made with ffmpeg.
"""
from __future__ import annotations

import base64
import shutil

import pytest

from library_factory import TrackSpec, _real_mutagen, encode_clip, ffmpeg_available, write_tags
from utils import audio_metadata_reader as reader

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="needs ffmpeg to make real audio files")

JPEG = b"\xff\xd8\xff\xe0" + b"embedded-front-cover" * 8


def _picture(data: bytes):
    pic = _real_mutagen()["flac"].Picture()
    pic.type = 3
    pic.mime = "image/jpeg"
    pic.desc = "Front"
    pic.data = data
    return pic


def _track(tmp_path, ext: str, *, cover: bytes | None, legacy_coverart: bool = False):
    path = tmp_path / "album" / f"song{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(encode_clip(ext, 2.0, 440.0), path)
    write_tags(path, TrackSpec("Artist", "Song", "Album", 1, "2020", "Rock", ext=ext))
    m = _real_mutagen()
    if cover is not None:
        if ext == ".flac":
            audio = m["flac"].FLAC(str(path))
            audio.add_picture(_picture(cover))
        else:
            cls = m["oggvorbis"].OggVorbis if ext == ".ogg" else m["oggopus"].OggOpus
            audio = cls(str(path))
            if legacy_coverart:
                audio["coverart"] = [base64.b64encode(cover).decode("ascii")]
            else:
                audio["metadata_block_picture"] = [base64.b64encode(_picture(cover).write()).decode("ascii")]
        audio.save()
    return path


def _mutagen_file(path):
    from mutagen import File

    return File(str(path))


@pytest.mark.parametrize("ext", [".ogg", ".opus"])
def test_vorbis_comment_picture_is_read(tmp_path, ext):
    path = _track(tmp_path, ext, cover=JPEG)
    assert reader._extract_cover_payloads(_mutagen_file(path)) == [JPEG]


@pytest.mark.parametrize("ext", [".ogg", ".opus"])
def test_legacy_coverart_comment_is_read(tmp_path, ext):
    path = _track(tmp_path, ext, cover=JPEG, legacy_coverart=True)
    assert reader._extract_cover_payloads(_mutagen_file(path)) == [JPEG]


def test_read_metadata_finds_an_ogg_cover(tmp_path):
    """``read_metadata`` sends .ogg through the general reader (no sidecar here)."""
    path = _track(tmp_path, ".ogg", cover=JPEG)
    _tags, covers, error, _hint = reader.read_metadata(str(path))
    assert error is None
    assert covers == [JPEG]


@pytest.mark.parametrize("ext", [".ogg", ".opus"])
def test_no_cover_still_reads_as_none(tmp_path, ext):
    path = _track(tmp_path, ext, cover=None)
    assert reader._extract_cover_payloads(_mutagen_file(path)) == []


def test_flac_picture_is_listed_once(tmp_path):
    path = _track(tmp_path, ".flac", cover=JPEG)
    assert reader._extract_cover_payloads(_mutagen_file(path)) == [JPEG]


@pytest.mark.parametrize("ext", [".ogg", ".opus"])
def test_duplicate_executor_sees_the_embedded_cover(tmp_path, ext):
    """The executor reads a loser's/winner's artwork through this reader."""
    import duplicate_consolidation_executor as executor

    path = _track(tmp_path, ext, cover=JPEG)
    assert executor._extract_artwork_bytes(str(path)) == JPEG
