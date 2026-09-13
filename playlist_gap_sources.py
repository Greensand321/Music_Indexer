"""Where the list of "songs I want" comes from.

A source is anything that can produce ``WantedTrack`` rows. It declares which
fields it can actually supply, and the match ladder skips the rungs those fields
would have fed. That declaration is the whole design: it lets a bare CSV of
YouTube video titles and a fully-populated ytmusicapi feed drive the same engine,
and it means adding a richer source later is a strict upgrade rather than a
rewrite.

Phase 1 ships the CSV source only. ``ytmusic`` and ``ytdlp`` implement the same
protocol and slot into ``SOURCE_REGISTRY`` without touching the matcher.

Qt-free, no network. See ``docs/playlist_gap_spec.md`` §3.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from playlist_gap_parse import compare_key, fold_text
from playlist_gap_types import SourceCapabilities, WantedTrack

#: Header synonyms, per logical field. Matched case- and punctuation-insensitively.
COLUMN_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "display": ("track name", "title", "song", "song title", "name", "track"),
    "title": ("track name", "title", "song", "song title", "track title"),
    "artist": ("artist name", "artist", "artists", "album artist", "performer", "channel"),
    "album": ("album", "album name", "release"),
    "duration": ("duration", "length", "time", "duration ms", "duration (ms)", "track duration"),
    "isrc": ("isrc", "isrc code"),
    "video_id": ("video id", "videoid", "track id", "id", "spotify track id", "uri", "url"),
    "playlist": ("playlist name", "playlist"),
}

#: A field is treated as available only if this share of sampled rows carry it.
#: Without this, a column that exists but is empty in every row — the `ISRC`
#: column in a YouTube Music export is exactly this — would be reported as a
#: usable signal and the ladder would waste a rung on it.
MIN_FILL_RATE = 0.05
SAMPLE_ROWS = 200

_DURATION_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.\d+)?\s*$")
_VIDEO_ID_RE = re.compile(r"([A-Za-z0-9_-]{11})")


@dataclass
class SourceSpec:
    """A saved source: everything one run needs, stored and re-runnable."""

    source_id: str = ""
    name: str = ""
    kind: str = "csv"
    location: str = ""
    options: Dict[str, object] = field(default_factory=dict)
    column_mapping: Optional[Dict[str, str]] = None

    def ensure_id(self) -> str:
        if not self.source_id:
            seed = f"{self.kind}\x00{self.location}\x00{self.name}"
            self.source_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        return self.source_id


def parse_duration(value: object) -> Optional[int]:
    """Accept "3:35", "1:02:14", "215", or milliseconds. Return seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return int(round(number / 1000)) if number > 10000 else int(round(number))
    text = str(value).strip()
    if not text:
        return None
    clock = _DURATION_RE.match(text)
    if clock:
        hours, minutes, seconds = clock.groups()
        return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    try:
        number = float(text)
    except ValueError:
        return None
    return int(round(number / 1000)) if number > 10000 else int(round(number))


def extract_video_id(value: object) -> Optional[str]:
    """Pull an 11-character YouTube id out of a url or bare id cell."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for marker in ("v=", "youtu.be/", "/watch/"):
        if marker in text:
            tail = text.split(marker, 1)[1]
            found = _VIDEO_ID_RE.match(tail)
            if found:
                return found.group(1)
    return text if _VIDEO_ID_RE.fullmatch(text) else None


def row_id_for(wanted: WantedTrack) -> str:
    """A stable identity for a wanted row, across runs and title edits.

    Preference order matters. A source id survives an upstream title edit; an
    ISRC identifies a recording; the fallback hashes an aggressively normalized
    display string so a cosmetic edit does not orphan a stored decision — but a
    substantial rewrite will, which is the practical argument for a source that
    supplies real ids.
    """
    if wanted.video_id:
        return f"yt:{wanted.video_id}"
    if wanted.isrc:
        return f"isrc:{wanted.isrc.replace('-', '').upper()}"
    # Deliberately NOT scoped to the source: one song wanted by three
    # playlists is one song. It should be answered once and downloaded once.
    # Per-source bookkeeping is the ledger's (row_id, source_id) table.
    return "s:" + hashlib.sha1(
        compare_key(wanted.display).encode("utf-8")
    ).hexdigest()[:20]


# ── column mapping ───────────────────────────────────────────────────────────


def _header_key(header: str) -> str:
    return compare_key(header)


def propose_mapping(headers: Sequence[str]) -> Dict[str, str]:
    """Guess which column feeds which field.

    Never positional and never hard-coded: TuneMyMusic's headers differ by the
    service the playlist came from and change over time, so the UI shows this
    proposal and lets the user correct it.
    """
    normalized = {_header_key(h): h for h in headers if h}
    mapping: Dict[str, str] = {}
    for field_name, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            key = compare_key(synonym)
            if key in normalized:
                mapping[field_name] = normalized[key]
                break
    if "display" not in mapping and "title" in mapping:
        mapping["display"] = mapping["title"]
    return mapping


def detect_capabilities(
    rows: Sequence[Dict[str, str]], mapping: Dict[str, str]
) -> SourceCapabilities:
    """Decide what this file *actually* supplies, by sampling its rows."""
    sample = rows[:SAMPLE_ROWS]
    if not sample:
        return SourceCapabilities()

    def filled(field_name: str) -> bool:
        column = mapping.get(field_name)
        if not column:
            return False
        hits = sum(1 for row in sample if str(row.get(column) or "").strip())
        return (hits / len(sample)) >= MIN_FILL_RATE

    return SourceCapabilities(
        display_string=True,
        title=filled("title"),
        artist=filled("artist"),
        album=filled("album"),
        duration=filled("duration"),
        isrc=filled("isrc"),
        video_id=filled("video_id"),
        availability=True,   # a blank row is reported as unavailable
        artwork_url=False,
    )


# ── CSV source ───────────────────────────────────────────────────────────────


@dataclass
class CsvReadResult:
    tracks: List[WantedTrack]
    capabilities: SourceCapabilities
    mapping: Dict[str, str]
    headers: List[str]
    blank_rows: int = 0
    warnings: List[str] = field(default_factory=list)


class CsvSource:
    """Read a wanted list from any CSV-shaped export."""

    key = "csv"

    def __init__(self, mapping: Optional[Dict[str, str]] = None) -> None:
        self._mapping = mapping

    def capabilities(self) -> SourceCapabilities:
        # Real capabilities are computed per file in `read`; this is the floor.
        return SourceCapabilities()

    def read(
        self,
        spec: SourceSpec,
        *,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> CsvReadResult:
        path = spec.location
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            headers = [h for h in (reader.fieldnames or []) if h]
            rows = [dict(row) for row in reader]

        mapping = dict(spec.column_mapping or self._mapping or propose_mapping(headers))
        caps = detect_capabilities(rows, mapping)
        source_id = spec.ensure_id()

        tracks: List[WantedTrack] = []
        blank = 0
        total = len(rows)
        for position, row in enumerate(rows):
            if progress is not None and (position % 100 == 0 or position + 1 == total):
                progress(position + 1, total, spec.name or os.path.basename(path))

            def cell(field_name: str) -> Optional[str]:
                column = mapping.get(field_name)
                if not column:
                    return None
                value = row.get(column)
                text = fold_text(str(value)) if value is not None else ""
                return text or None

            artist = cell("artist")
            title = cell("title")
            raw_display = cell("display")
            # The display string is the full human label the parser and the
            # filename rung work on. When the file gives artist and title in
            # separate columns, a bare title would throw the artist away — so
            # compose. When it gives only one blob (a YouTube export), use it.
            if artist and title:
                display = f"{artist} - {title}"
            else:
                display = raw_display or title or ""

            wanted = WantedTrack(
                row_id="",
                display=display,
                title=title,
                artist=artist,
                album=cell("album"),
                duration=parse_duration(cell("duration")),
                isrc=cell("isrc"),
                video_id=extract_video_id(cell("video_id")),
                available=bool(display),
                source_id=source_id,
                playlists=(cell("playlist") or spec.name or "",),
                position=position,
            )
            if not display:
                # Never dropped: a blank row means a deleted or private track, and
                # silently losing it is the failure mode this feature exists to
                # prevent. It is carried through and reported as unavailable.
                blank += 1
            wanted.row_id = row_id_for(wanted)
            tracks.append(wanted)

        warnings: List[str] = []
        if "display" not in mapping and "title" not in mapping:
            warnings.append(
                "No title-like column was recognised — pick one in the column mapping."
            )
        if blank:
            warnings.append(f"{blank} row(s) have no title; reported as unavailable.")
        return CsvReadResult(
            tracks=tracks,
            capabilities=caps,
            mapping=mapping,
            headers=headers,
            blank_rows=blank,
            warnings=warnings,
        )


SOURCE_REGISTRY: Dict[str, Callable[[], object]] = {"csv": CsvSource}


def merge_wanted(groups: Sequence[Sequence[WantedTrack]]) -> List[WantedTrack]:
    """Merge several sources' rows, keeping each track once.

    A track wanted by three playlists should be downloaded once, so rows are
    merged by ``row_id`` and their playlist names unioned.
    """
    merged: Dict[str, WantedTrack] = {}
    for group in groups:
        for row in group:
            existing = merged.get(row.row_id)
            if existing is None:
                merged[row.row_id] = row
                continue
            names = [p for p in (*existing.playlists, *row.playlists) if p]
            existing.playlists = tuple(dict.fromkeys(names))
            for attr in ("title", "artist", "album", "duration", "isrc", "video_id"):
                if getattr(existing, attr) is None:
                    setattr(existing, attr, getattr(row, attr))
    return list(merged.values())
