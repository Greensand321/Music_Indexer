"""Library statistics (tracks, size, artists, albums) — Qt-free and cheap to import.

Used by the main window's page header and by the landing's folder row, both
from read-only worker threads. Lives on its own so the landing can count a
library without importing the main window (and every workspace with it).

``library_stats`` keeps its three-number result (callers unpack it), and
remembers the full count, albums included, per library (``last_counts``).
Home reads that after the main window's stats scan reports, so it can show
an album count without walking the library a second time.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import NamedTuple

AUDIO_EXTS = frozenset({".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"})


class LibraryCounts(NamedTuple):
    tracks: int
    size_bytes: int
    artists: int
    albums: int


# The newest full count per library (written by worker threads, read on the
# GUI thread; guarded so a reader never sees a half-updated dict).
_LAST: dict[str, LibraryCounts] = {}
_LAST_LOCK = threading.Lock()
_LAST_MAX = 8


def _key(library_path: str) -> str:
    return os.path.normcase(os.path.abspath(str(library_path)))


def last_counts(library_path: str) -> LibraryCounts | None:
    """The newest finished count of ``library_path`` (None if never counted)."""
    if not library_path:
        return None
    with _LAST_LOCK:
        return _LAST.get(_key(library_path))


def _remember(library_path: str, counts: LibraryCounts) -> None:
    with _LAST_LOCK:
        if len(_LAST) >= _LAST_MAX:
            _LAST.clear()
        _LAST[_key(library_path)] = counts


def library_counts(library_path: str, should_stop=lambda: False) -> LibraryCounts | None:
    """Tracks, size, artists and albums; None if stopped early.

    Skips the reserved folders (Quarantine/, Trash/, Docs/, ...), never
    follows directory symlinks, and skips files it cannot stat (broken
    symlinks, permission errors) instead of failing the whole count.
    "Artists" is the folder two levels above a track (``Artist/Album/track``),
    or the folder above a track that sits one level deep (``Artist/track``);
    tracks at the library root count no artist. "Albums" is a folder holding
    tracks at least two levels deep (``Artist/Album/``), the same folder
    convention: loose tracks under an artist or at the root count no album.
    """
    from utils.reserved_folders import walk_library

    tracks = 0
    size_bytes = 0
    artists: set[str] = set()
    albums: set[tuple[str, ...]] = set()
    for dirpath, _dirnames, filenames in walk_library(library_path):
        if should_stop():
            return None
        rel = os.path.relpath(dirpath, library_path)
        parts = () if rel in (".", "") else Path(rel).parts
        artist = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
        album_key = tuple(p.casefold() for p in parts) if len(parts) >= 2 else None
        for name in filenames:
            if os.path.splitext(name)[1].lower() not in AUDIO_EXTS:
                continue
            try:
                size = os.stat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
            tracks += 1
            size_bytes += size
            if artist:
                artists.add(artist.casefold())
            if album_key is not None:
                albums.add(album_key)
    return LibraryCounts(tracks, size_bytes, len(artists), len(albums))


def library_stats(library_path: str, should_stop=lambda: False) -> tuple[int, int, int] | None:
    """(tracks, size_bytes, artists); None if stopped early.

    The full count (albums too) is remembered for ``last_counts``; see
    ``library_counts`` for what each number means."""
    counts = library_counts(library_path, should_stop)
    if counts is None:
        return None
    _remember(library_path, counts)
    return counts.tracks, counts.size_bytes, counts.artists


def format_stats(tracks: int, size_gb: float | None, artists: int) -> str:
    """``"4,812 tracks · 162.4 GB · 391 artists"`` (size left out when None)."""
    parts = [f"{tracks:,} {'track' if tracks == 1 else 'tracks'}"]
    if size_gb is not None:
        parts.append(f"{size_gb:.1f} GB")
    parts.append(f"{artists:,} {'artist' if artists == 1 else 'artists'}")
    return " · ".join(parts)


def format_size(size_gb: float) -> str:
    """A compact size for a stat tile: ``"812 MB"``, ``"162.4 GB"``, ``"1.62 TB"``."""
    if size_gb < 1.0:
        return f"{size_gb * 1024:.0f} MB"
    if size_gb >= 1000.0:
        return f"{size_gb / 1024:.2f} TB"
    return f"{size_gb:.1f} GB"
