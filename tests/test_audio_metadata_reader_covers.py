"""Embedded cover art must be found for every container the app reads."""
from __future__ import annotations

from library_factory import TrackSpec, build_library

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def test_m4a_embedded_cover_is_returned(tmp_path):
    import mutagen.mp4 as mp4
    from utils.audio_metadata_reader import read_metadata

    lib = build_library(tmp_path / "lib", [TrackSpec("A", "T", "Al", 1, ext=".m4a")])
    path = lib.audio_paths[0]
    audio = mp4.MP4(str(path))
    audio["covr"] = [mp4.MP4Cover(PNG, imageformat=mp4.MP4Cover.FORMAT_PNG)]
    audio.save()

    tags, covers, error, _hint = read_metadata(str(path), include_cover=True)
    assert error is None
    assert tags.get("title") == "T"
    assert covers and covers[0].startswith(b"\x89PNG")


def test_mp3_embedded_cover_is_returned(tmp_path):
    from mutagen.id3 import APIC, ID3
    from utils.audio_metadata_reader import read_metadata

    lib = build_library(tmp_path / "lib", [TrackSpec("A", "T", "Al", 1, ext=".mp3")])
    path = lib.audio_paths[0]
    id3 = ID3(str(path))
    id3.add(APIC(encoding=3, mime="image/png", type=3, desc="cover", data=PNG))
    id3.save()

    _tags, covers, _error, _hint = read_metadata(str(path), include_cover=True)
    assert covers and covers[0].startswith(b"\x89PNG")
