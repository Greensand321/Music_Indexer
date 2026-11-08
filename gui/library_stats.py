"""Library statistics (tracks, size, artists) — Qt-free and cheap to import.

Used by the main window's page header and by the landing's folder row, both
from read-only worker threads. Lives on its own so the landing can count a
library without importing the main window (and every workspace with it).
"""
from __future__ import annotations

import os
from pathlib import Path

AUDIO_EXTS = frozenset({".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"})


def library_stats(library_path: str, should_stop=lambda: False) -> tuple[int, int, int] | None:
    """(tracks, size_bytes, artists); None if stopped early.

    Skips the reserved folders (Quarantine/, Trash/, Docs/, ...), never
    follows directory symlinks, and skips files it cannot stat (broken
    symlinks, permission errors) instead of failing the whole count.
    "Artists" is the folder two levels above a track (``Artist/Album/track``),
    or the folder above a track that sits one level deep (``Artist/track``);
    tracks at the library root count no artist.
    """
    from utils.reserved_folders import walk_library

    tracks = 0
    size_bytes = 0
    artists: set[str] = set()
    for dirpath, _dirnames, filenames in walk_library(library_path):
        if should_stop():
            return None
        rel = os.path.relpath(dirpath, library_path)
        parts = () if rel in (".", "") else Path(rel).parts
        artist = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
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
    return tracks, size_bytes, len(artists)


def format_stats(tracks: int, size_gb: float | None, artists: int) -> str:
    """``"4,812 tracks · 162.4 GB · 391 artists"`` (size left out when None)."""
    parts = [f"{tracks:,} {'track' if tracks == 1 else 'tracks'}"]
    if size_gb is not None:
        parts.append(f"{size_gb:.1f} GB")
    parts.append(f"{artists:,} {'artist' if artists == 1 else 'artists'}")
    return " · ".join(parts)
