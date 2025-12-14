"""Helpers for generating playlists grouped by genre."""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, Iterable, List, Mapping

from mutagen import File as MutagenFile

from controllers.normalize_controller import normalize_genres
from playlist_generator import write_playlist

GenreGroups = Dict[str, List[str]]

_SPLIT_RE = re.compile(r"[;,/|]")


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "_", text).strip(" _")
    return cleaned or "Unknown"


def read_genres(path: str, split_multi: bool = True) -> list[str]:
    """Return list of genres found in ``path``.

    Parameters
    ----------
    path:
        Audio file path.
    split_multi:
        When True, split combined genre strings on common separators.
    """

    try:
        audio = MutagenFile(path, easy=True)
    except Exception:
        return []

    if not audio or not audio.tags:
        return []

    genres = audio.tags.get("genre", []) or []
    results: list[str] = []
    for raw in genres:
        parts = _SPLIT_RE.split(raw) if split_multi else [raw]
        for part in parts:
            part = part.strip()
            if part:
                results.append(part)
    return results


def group_tracks_by_genre(
    tracks: Iterable[str],
    mapping: Mapping[str, str] | None = None,
    include_unknown: bool = False,
    split_multi: bool = True,
    log_callback: Callable[[str], None] | None = None,
) -> GenreGroups:
    """Group ``tracks`` into playlists keyed by genre."""

    mapping = mapping or {}
    log = log_callback or (lambda _m: None)
    grouped: GenreGroups = {}

    for track in tracks:
        genres = read_genres(track, split_multi=split_multi)
        if mapping:
            genres = normalize_genres(genres, mapping)

        if not genres:
            if include_unknown:
                grouped.setdefault("Unknown", []).append(track)
            log(f"! No genre tag for {track}")
            continue

        for genre in genres:
            grouped.setdefault(genre, []).append(track)
            log(f"• {os.path.basename(track)} → {genre}")

    return grouped


def write_genre_playlists(
    grouped: Mapping[str, List[str]],
    playlists_dir: str,
    log_callback: Callable[[str], None] | None = None,
) -> Dict[str, str]:
    """Write one playlist per genre and return mapping of genre->path."""

    os.makedirs(playlists_dir, exist_ok=True)
    log = log_callback or (lambda _m: None)

    used_names: set[str] = set()
    out_paths: Dict[str, str] = {}
    for genre in sorted(grouped.keys(), key=str.lower):
        base = _safe_name(genre)
        name = base
        suffix = 2
        while name in used_names:
            name = f"{base}_{suffix}"
            suffix += 1
        used_names.add(name)

        outfile = os.path.join(playlists_dir, f"{name}.m3u")
        write_playlist(grouped[genre], outfile)
        out_paths[genre] = outfile
        log(f"→ Wrote {outfile}")

    return out_paths
