"""Real multi-value genre files for the metadata-repair (Tag Fixer / Genres) tests.

``tests/library_factory.py`` writes one value per tag. The genre bugs these
tests guard against only show up with *several* genre values in one file, so
this helper writes (and independently reads back) multi-value genres with
mutagen directly — never through the app's own reader or writer, so a bug
there can't hide itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pytest

from library_factory import TrackSpec, build_library, ffmpeg_available, _real_mutagen

ALL_FORMATS = (".flac", ".mp3", ".opus", ".m4a", ".ogg", ".wav")

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg is needed to build non-WAV files")


def set_genres(path: Path | str, genres: Sequence[str]) -> None:
    """Write ``genres`` as a real multi-value genre tag (replacing any)."""
    m = _real_mutagen()
    path = Path(path)
    ext = path.suffix.lower()
    genres = list(genres)
    if ext == ".mp3":
        try:
            audio = m["easyid3"].EasyID3(str(path))
        except m["id3"].ID3NoHeaderError:
            audio = m["easyid3"].EasyID3()
        audio["genre"] = genres
        audio.save(str(path))
    elif ext == ".m4a":
        audio = m["mp4"].MP4(str(path))
        audio["\xa9gen"] = genres
        audio.save()
    elif ext == ".wav":
        audio = m["wave"].WAVE(str(path))
        if audio.tags is None:
            audio.add_tags()
        audio.tags.setall("TCON", [m["id3"].TCON(encoding=3, text=genres)])
        audio.save()
    else:
        cls = {".flac": m["flac"].FLAC, ".ogg": m["oggvorbis"].OggVorbis,
               ".opus": m["oggopus"].OggOpus}[ext]
        audio = cls(str(path))
        audio["genre"] = genres
        audio.save()


def get_genres(path: Path | str) -> list[str]:
    """Every genre value actually on disk, read with mutagen directly."""
    m = _real_mutagen()
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".mp3":
        try:
            return list(m["easyid3"].EasyID3(str(path)).get("genre", []))
        except m["id3"].ID3NoHeaderError:
            return []
    if ext == ".m4a":
        return list((m["mp4"].MP4(str(path)).tags or {}).get("\xa9gen", []))
    if ext == ".wav":
        tags = m["wave"].WAVE(str(path)).tags
        if tags is None:
            return []
        out: list[str] = []
        for frame in tags.getall("TCON"):
            out.extend(frame.genres)
        return out
    cls = {".flac": m["flac"].FLAC, ".ogg": m["oggvorbis"].OggVorbis,
           ".opus": m["oggopus"].OggOpus}[ext]
    return list((cls(str(path)).tags or {}).get("genre", []))


def get_field(path: Path | str, field: str) -> list[str]:
    """Raw values of an easy-interface field (artist/title/album) on disk."""
    m = _real_mutagen()
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".wav":
        frame_id = {"artist": "TPE1", "title": "TIT2", "album": "TALB"}[field]
        tags = m["wave"].WAVE(str(path)).tags
        return [str(t) for f in (tags.getall(frame_id) if tags else []) for t in f.text]
    if ext == ".mp3":
        return list(m["easyid3"].EasyID3(str(path)).get(field, []))
    if ext == ".m4a":
        key = {"artist": "\xa9ART", "title": "\xa9nam", "album": "\xa9alb"}[field]
        return list((m["mp4"].MP4(str(path)).tags or {}).get(key, []))
    cls = {".flac": m["flac"].FLAC, ".ogg": m["oggvorbis"].OggVorbis,
           ".opus": m["oggopus"].OggOpus}[ext]
    return list((cls(str(path)).tags or {}).get(field, []))


def build_genre_library(root: Path | str, files: Iterable[tuple]) -> dict[str, Path]:
    """Build real files and give each a multi-value genre tag.

    ``files`` is ``(relpath, artist, title, genres)`` tuples (``genres`` may be
    empty). The extension comes from ``relpath``. Returns ``{relpath: Path}``.
    """
    files = list(files)
    specs = []
    for i, (relpath, artist, title, _genres) in enumerate(files):
        specs.append(TrackSpec(
            artist, title, "Album", ext=Path(relpath).suffix.lower(),
            freq=300 + 17 * i, relpath=relpath,
        ))
    build_library(root, specs)
    out: dict[str, Path] = {}
    for relpath, _artist, _title, genres in files:
        path = Path(root) / relpath
        if genres:
            set_genres(path, genres)
        out[relpath] = path
    return out
