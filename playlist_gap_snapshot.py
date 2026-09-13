"""Build the "what I already own" side of a Playlist Gap comparison.

Reads the library once into an in-memory index. No fingerprinting happens here —
pass 1 never needs audio, and the fingerprint cache already holds everything this
needs for any file the Duplicate Finder has seen.

**The folder policy is the single most consequential thing in this module.** The
Indexer and Duplicate Finder deliberately skip ``Not Sorted/``, ``Quarantine/``
and ``Manual Review/``. For *this* feature all three are music the user already
has, so excluding them would report owned tracks as missing and cause exactly the
duplicate re-download the feature exists to prevent. Reusing their skip lists
would be a quiet, damaging bug — hence a policy defined locally and a regression
test that guards it.

Qt-free. The tag reader and cache reader are injectable.

See ``docs/playlist_gap_spec.md`` §4.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set

from playlist_gap_parse import compare_key
from playlist_gap_types import LibraryTrack

AUDIO_EXTS: FrozenSet[str] = frozenset(
    {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus", ".alac", ".aiff", ".aif"}
)

#: Reserved folders that hold music the user owns. Counted, not skipped.
OWNED_RESERVED: FrozenSet[str] = frozenset({"not sorted", "quarantine", "manual review"})

#: Reserved folders with no owned audio in them.
SKIP_RESERVED: FrozenSet[str] = frozenset({"trash", "docs", "playlists"})

#: An 11-character YouTube id in square brackets — yt-dlp's default output
#: template is ``%(title)s [%(id)s].%(ext)s``, so a downloaded file usually
#: still carries the identity of the upload it came from.
_VIDEO_ID_IN_NAME = re.compile(r"\[([A-Za-z0-9_-]{11})\]")
_VIDEO_ID_IN_URL = re.compile(
    r"(?:youtube\.com/watch\?(?:.*&)?v=|youtu\.be/|music\.youtube\.com/watch\?(?:.*&)?v=)"
    r"([A-Za-z0-9_-]{11})"
)

#: Tags a downloader may have written the source URL into.
PROVENANCE_TAGS = ("purl", "comment", "website")

#: Tokens appearing in more than this share of the library select nothing useful.
_TOKEN_MAX_SHARE = 0.05


@dataclass(frozen=True)
class FolderPolicy:
    """Which reserved folders count as "I already have this"."""

    include_reserved: FrozenSet[str] = OWNED_RESERVED
    exclude_reserved: FrozenSet[str] = SKIP_RESERVED

    def skips(self, folder_name: str) -> bool:
        return folder_name.strip().lower() in self.exclude_reserved


DEFAULT_INCLUDE = FolderPolicy()


@dataclass
class LibrarySnapshot:
    """An indexed view of the library, built once per run."""

    tracks: List[LibraryTrack] = field(default_factory=list)
    by_video_id: Dict[str, List[int]] = field(default_factory=dict)
    by_artist: Dict[str, List[int]] = field(default_factory=dict)
    by_title_token: Dict[str, Set[int]] = field(default_factory=dict)
    by_filename_token: Dict[str, Set[int]] = field(default_factory=dict)
    by_artist_title: Dict[tuple, List[int]] = field(default_factory=dict)
    read_at: float = 0.0
    counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracks)

    @property
    def artists(self) -> FrozenSet[str]:
        return frozenset(self.by_artist)

    def shortlist_by_tokens(self, tokens: Iterable[str], *, filenames: bool = False) -> Set[int]:
        """Candidate row indexes for a set of title tokens.

        This is what keeps the ladder fast when the source supplies no artist to
        block on: rather than comparing against every track, gather only rows
        sharing a rare token.
        """
        index = self.by_filename_token if filenames else self.by_title_token
        out: Set[int] = set()
        for token in tokens:
            hits = index.get(token)
            if hits:
                out |= hits
        return out


def extract_video_id(path: str, tags: Optional[Dict[str, object]] = None) -> Optional[str]:
    """Recover the source video id from a filename or a provenance tag."""
    name_match = _VIDEO_ID_IN_NAME.search(os.path.basename(path))
    if name_match:
        return name_match.group(1)
    for key in PROVENANCE_TAGS:
        value = (tags or {}).get(key)
        if isinstance(value, str):
            url_match = _VIDEO_ID_IN_URL.search(value)
            if url_match:
                return url_match.group(1)
    return None


def normalize_filename(path: str) -> str:
    """Comparison key for a file's basename, minus extension and id suffix."""
    base = os.path.splitext(os.path.basename(path))[0]
    base = _VIDEO_ID_IN_NAME.sub(" ", base)
    return compare_key(base.replace("_", " "))


def tokens_of(text: str) -> List[str]:
    """Comparison tokens, dropping bare numbers (they match everything)."""
    return [tok for tok in compare_key(text).split() if tok and not tok.isdigit()]


def iter_audio_files(
    library_root: str, policy: FolderPolicy = DEFAULT_INCLUDE
) -> Iterable[str]:
    """Walk ``library_root`` under this feature's own inclusion policy."""
    for dirpath, dirnames, filenames in os.walk(library_root):
        dirnames[:] = sorted(d for d in dirnames if not policy.skips(d))
        rel = os.path.relpath(dirpath, library_root)
        parts = {p.strip().lower() for p in rel.split(os.sep)} if rel != "." else set()
        if parts & policy.exclude_reserved:
            dirnames[:] = []
            continue
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                yield os.path.join(dirpath, name)


def _bucket_for(library_root: str, path: str) -> str:
    """Which top-level area a file sits in, for the UI's honesty counts."""
    rel = os.path.relpath(path, library_root)
    head = rel.split(os.sep)[0] if os.sep in rel else ""
    lowered = head.strip().lower()
    if lowered in OWNED_RESERVED:
        return head
    return "Library"


def _default_cache_reader(path: str, db_path: str):
    from fingerprint_cache import get_cached_fingerprint_metadata

    return get_cached_fingerprint_metadata(path, db_path)


def _default_tag_reader(path: str) -> Dict[str, object]:
    from utils.audio_metadata_reader import read_tags

    return read_tags(path) or {}


def _resolve(override, fallback):
    """Pick an injected collaborator, else the module default.

    Resolved *inside* the call rather than bound as a default argument: a default
    argument captures the function object at import time, so monkeypatching the
    module attribute would silently have no effect — which is exactly how a test
    ends up passing against a reader that was never used.
    """
    return override if override is not None else fallback


def build_snapshot(
    library_root: str,
    *,
    policy: FolderPolicy = DEFAULT_INCLUDE,
    cache_db: Optional[str] = None,
    read_tags: Optional[Callable[[str], Dict[str, object]]] = None,
    read_cache: Optional[Callable[[str, str], tuple]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    paths: Optional[Sequence[str]] = None,
) -> LibrarySnapshot:
    """Index the library. Cheapest data source first, no audio decoded."""
    read_tags = _resolve(read_tags, _default_tag_reader)
    read_cache = _resolve(read_cache, _default_cache_reader)
    snapshot = LibrarySnapshot(read_at=time.time())
    all_paths = list(paths) if paths is not None else list(iter_audio_files(library_root, policy))
    total = len(all_paths)

    for position, path in enumerate(all_paths):
        if should_cancel is not None and should_cancel():
            break
        if progress is not None:
            progress(position + 1, total, path)

        tags: Dict[str, object] = {}
        fingerprint = None
        duration = bitrate = None
        norm_artist = norm_title = norm_album = None

        if cache_db:
            try:
                fingerprint, meta = read_cache(path, cache_db)
            except Exception as exc:  # a cache miss must never abort the scan
                snapshot.errors.append(f"{path}: cache read failed: {exc}")
                meta = None
            if meta:
                tags = dict(meta.get("tags") or {})
                duration = meta.get("duration")
                bitrate = meta.get("bitrate")
                norm_artist = meta.get("normalized_artist")
                norm_title = meta.get("normalized_title")
                norm_album = meta.get("normalized_album")

        if not tags:
            try:
                tags = dict(read_tags(path) or {})
            except Exception as exc:
                snapshot.errors.append(f"{path}: tag read failed: {exc}")
                tags = {}

        # Derive from tags with `compare_key` in preference to the cache's own
        # normalized_* columns. Those use `fingerprint_cache.normalized_key`,
        # which splits on apostrophes ("don t stop"); mixing the two spellings
        # across the two sides of a comparison would silently split the index.
        # The cache values are kept only as a fallback when a tag is absent.
        def _norm(field: str, cached):
            value = tags.get(field)
            if isinstance(value, str) and value.strip():
                return compare_key(value)
            return cached or None

        artist_value = tags.get("artist") or tags.get("albumartist")
        norm_artist = (
            compare_key(artist_value) if isinstance(artist_value, str) and artist_value.strip()
            else (norm_artist or None)
        )
        norm_title = _norm("title", norm_title)
        norm_album = _norm("album", norm_album)

        track = LibraryTrack(
            path=path,
            ext=os.path.splitext(path)[1].lower(),
            duration=int(duration) if isinstance(duration, (int, float)) and duration else None,
            bitrate=int(bitrate) if isinstance(bitrate, (int, float)) and bitrate else None,
            fingerprint=fingerprint,
            tags=tags,
            norm_artist=norm_artist or None,
            norm_title=norm_title or None,
            norm_album=norm_album or None,
            video_id=extract_video_id(path, tags),
            filename_norm=normalize_filename(path),
        )
        index = len(snapshot.tracks)
        snapshot.tracks.append(track)

        bucket = _bucket_for(library_root, path)
        snapshot.counts[bucket] = snapshot.counts.get(bucket, 0) + 1

        if track.video_id:
            snapshot.by_video_id.setdefault(track.video_id, []).append(index)
        if track.norm_artist:
            snapshot.by_artist.setdefault(track.norm_artist, []).append(index)
        if track.norm_artist and track.norm_title:
            snapshot.by_artist_title.setdefault(
                (track.norm_artist, track.norm_title), []
            ).append(index)
        for token in tokens_of(track.norm_title or ""):
            snapshot.by_title_token.setdefault(token, set()).add(index)
        for token in tokens_of(track.filename_norm):
            snapshot.by_filename_token.setdefault(token, set()).add(index)

    _drop_common_tokens(snapshot)
    return snapshot


def _drop_common_tokens(snapshot: LibrarySnapshot) -> None:
    """Remove tokens so common they shortlist most of the library."""
    if len(snapshot.tracks) < 50:
        return
    ceiling = max(2, int(len(snapshot.tracks) * _TOKEN_MAX_SHARE))
    for index in (snapshot.by_title_token, snapshot.by_filename_token):
        for token in [t for t, rows in index.items() if len(rows) > ceiling]:
            del index[token]
