"""Playlist export helpers."""

from __future__ import annotations

import os
from typing import Iterable

from playlist_generator import write_playlist, update_playlists


def save_playlist(tracks: Iterable[str], outfile: str) -> str:
    """Write ``tracks`` to ``outfile`` and refresh library playlists."""

    write_playlist(list(tracks), outfile)

    abs_tracks = [os.path.abspath(p) for p in tracks]
    update_playlists(abs_tracks)
    return outfile
