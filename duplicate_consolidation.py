"""Duplicate consolidation planning (no file mutations).

This module performs a dry-run duplicate scan that groups musically identical
tracks using audio fingerprints, selects a deterministic "winner" based on
quality, and produces a consolidation plan describing artwork/metadata actions.
No files are moved or written—callers can safely surface the plan for review
before executing any changes.
"""
from __future__ import annotations

import base64
import datetime
import html
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence
from collections import defaultdict

from utils.path_helpers import ensure_long_path
try:
    _MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
except ValueError:  # pragma: no cover - defensive: broken mutagen installs
    _MUTAGEN_AVAILABLE = False

if _MUTAGEN_AVAILABLE:
    from mutagen import File as MutagenFile  # type: ignore
else:  # pragma: no cover - optional dependency
    MutagenFile = None  # type: ignore

_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
if _PIL_AVAILABLE:
    from PIL import Image  # type: ignore
else:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

from music_indexer_api import extract_primary_and_collabs
from near_duplicate_detector import fingerprint_distance, _parse_fp as _parse_fingerprint

LOSSLESS_EXTS = {".flac", ".wav", ".alac", ".ape", ".aiff", ".aif"}
EXACT_DUPLICATE_THRESHOLD = 0.02
NEAR_DUPLICATE_THRESHOLD = 0.10
ARTWORK_HASH_SIZE = 8
ARTWORK_SIMILARITY_THRESHOLD = 10
DEFAULT_MAX_CANDIDATES = 5000
DEFAULT_MAX_COMPARISONS = 50_000
DEFAULT_TIMEOUT_SEC = 15.0
REVIEW_KEYWORDS = (
    "remix",
    "edit",
    "version",
    "sped up",
    "slowed",
    "nightcore",
    "speed up",
    "speed-up",
)
PLACEHOLDER_TOKENS = {"demo", "tbd", "placeholder", "sample", "example", "unknown"}
TAG_KEYS = [
    "artist",
    "albumartist",
    "title",
    "album",
    "album_type",
    "track",
    "tracknumber",
    "disc",
    "discnumber",
    "date",
    "year",
    "genre",
    "compilation",
]
TITLE_JUNK_TOKENS = {
    "remaster",
    "remastered",
    "explicit",
    "clean",
    "mono",
    "stereo",
    "version",
    "deluxe",
    "bonus",
}
COARSE_FP_BAND_SIZE = 8
COARSE_FP_BANDS = 6
COARSE_FP_QUANTIZATION = 4


def _now() -> float:
    return time.perf_counter()


def _default_progress(_current: int, _total: int, _msg: str) -> None:
    return None


def _blank_tags() -> Dict[str, object]:
    return {key: None for key in TAG_KEYS}


def _first_value(value: object) -> object:
    if isinstance(value, list):
        return _first_value(value[0]) if value else None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return _first_value(value[0]) if value else None
    if hasattr(value, "text"):
        try:
            text_value = value.text
            if isinstance(text_value, (list, tuple)):
                return _first_value(text_value[0]) if text_value else None
            return str(text_value) if text_value is not None else None
        except Exception:
            pass
    try:
        return str(value)
    except Exception:
        return None


def _parse_int(value: object) -> int | None:
    try:
        ivalue = int(str(value).split("/")[0])
        return ivalue
    except Exception:
        return None


def _parse_track_total(value: object) -> int | None:
    if isinstance(value, list):
        return _parse_track_total(value[0]) if value else None
    if isinstance(value, tuple):
        return _parse_track_total(value[0]) if value else None
    if value is None:
        return None
    text = str(value)
    if "/" in text:
        parts = text.split("/", 1)
        if len(parts) == 2 and parts[1].strip():
            return _parse_int(parts[1])
    return None


def _release_track_total(tags: Mapping[str, object]) -> int | None:
    for key in ("track", "tracknumber"):
        total = _parse_track_total(tags.get(key))
        if total is not None:
            return total
    for key in ("tracktotal", "totaltracks", "track_total", "total_tracks"):
        total = _parse_int(tags.get(key))
        if total is not None:
            return total
    return None


def _release_size_context(tags: Mapping[str, object]) -> tuple[str, str]:
    album_type = str(tags.get("album_type") or tags.get("release_type") or "").lower()
    track_total = _release_track_total(tags)
    if track_total == 1:
        return "single", "Track total indicates 1"
    if track_total and track_total >= 2:
        return "multi", f"Track total indicates {track_total}"
    if album_type == "single":
        return "single", "Album type tag indicates single"
    if album_type in {"album", "lp", "ep"}:
        return "multi", f"Album type tag indicates {album_type}"
    return "unknown", "No track-total or album-type signal"


def _normalized_tokens(text: str) -> List[str]:
    lowered = (text or "").lower()
    cleaned = re.sub(r"[\\[\\]{}()]+", " ", lowered)
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", cleaned) if tok]
    return tokens


def _normalize_primary_artist(tags: Mapping[str, object], path: str) -> str:
    raw_artist = str(tags.get("artist") or tags.get("albumartist") or "").strip()
    lowered = raw_artist.lower()
    placeholders = {"various", "various artists", "va", "unknown"}
    if lowered in placeholders:
        raw_artist = ""
    if not raw_artist:
        stem = os.path.splitext(os.path.basename(path))[0]
        if " - " in stem:
            candidate = stem.split(" - ", 1)[0].strip()
            if candidate.lower() not in placeholders:
                raw_artist = candidate
    primary, _ = extract_primary_and_collabs(raw_artist) if raw_artist else ("", [])
    tokens = _normalized_tokens(primary)
    return " ".join(tokens or ["unknown"])


def _normalize_title(tags: Mapping[str, object], path: str) -> str:
    raw_title = str(tags.get("title") or "").strip()
    if not raw_title:
        raw_title = os.path.splitext(os.path.basename(path))[0]
    tokens = _normalized_tokens(raw_title)
    filtered = [tok for tok in tokens if tok not in TITLE_JUNK_TOKENS]
    filtered = [tok for tok in filtered if not (tok.isdigit() and len(tok) == 4)]
    return " ".join(filtered or tokens or ["unknown"])


def _normalize_album_key(tags: Mapping[str, object], path: str) -> tuple[str, str] | None:
    raw_album = str(tags.get("album") or "").strip()
    if not raw_album:
        return None
    album_tokens = _normalized_tokens(raw_album)
    if not album_tokens:
        return None
    album_title = " ".join(album_tokens)
    album_artist = _normalize_primary_artist(tags, path)
    if album_artist == "unknown":
        album_artist = ""
    return album_artist, album_title


def _album_display_label(tags: Mapping[str, object]) -> str:
    album = str(tags.get("album") or "").strip()
    artist = str(tags.get("albumartist") or tags.get("artist") or "").strip()
    if album and artist:
        return f"{artist} — {album}"
    return album or artist or "Unknown album"


def _metadata_bucket_key(track: DuplicateTrack) -> tuple[str, str]:
    tags = track.current_tags if isinstance(track.current_tags, Mapping) else track.tags
    tags = tags or {}
    return _normalize_primary_artist(tags, track.path), _normalize_title(tags, track.path)


def _build_metadata_buckets(tracks: Sequence[DuplicateTrack]) -> Dict[tuple[str, str], List[DuplicateTrack]]:
    title_to_artists: Dict[str, set[str]] = defaultdict(set)
    normalized: List[tuple[DuplicateTrack, tuple[str, str]]] = []
    for track in tracks:
        key = _metadata_bucket_key(track)
        normalized.append((track, key))
        artist, title = key
        if artist and artist != "unknown":
            title_to_artists[title].add(artist)
    buckets: Dict[tuple[str, str], List[DuplicateTrack]] = defaultdict(list)
    for track, (artist, title) in normalized:
        artist_key = artist
        known_artists = title_to_artists.get(title, set())
        if artist_key == "unknown" and len(known_artists) == 1:
            artist_key = next(iter(known_artists))
        buckets[(artist_key, title)].append(track)
    return buckets


def _fingerprint_to_ints(fp: str | None, *, limit: int = COARSE_FP_BAND_SIZE * COARSE_FP_BANDS) -> List[int]:
    if not fp:
        return []
    parsed = _parse_fingerprint(fp) if "_parse_fingerprint" in globals() else None
    ints: List[int] = []
    if parsed:
        kind, payload = parsed
        if kind == "ints":
            ints = list(payload)
        elif kind == "bytes":
            ints = list(payload)
    if not ints:
        try:
            ints = [int(tok) for tok in fp.replace(",", " ").split() if tok]
        except Exception:
            ints = [ord(ch) for ch in fp]
    return ints[:limit]


def _coarse_fingerprint_keys(fp: str | None) -> List[str]:
    if not fp:
        return []
    ints = _fingerprint_to_ints(fp)
    if not ints:
        return []
    quantized = [val // COARSE_FP_QUANTIZATION for val in ints]
    keys: List[str] = []
    for idx in range(0, len(quantized), COARSE_FP_BAND_SIZE):
        band = quantized[idx : idx + COARSE_FP_BAND_SIZE]
        if not band:
            break
        digest = hashlib.blake2s(",".join(map(str, band)).encode("utf-8"), digest_size=4).hexdigest()
        keys.append(f"{idx // COARSE_FP_BAND_SIZE}:{digest}")
    if not keys:
        keys.append(f"band0:{hashlib.blake2s(fp.encode('utf-8'), digest_size=4).hexdigest()}")
    return keys


def _extract_image_dimensions(payload: bytes) -> tuple[Optional[int], Optional[int]]:
    if not payload or not Image:
        return None, None
    try:
        with Image.open(io.BytesIO(payload)) as img:
            return img.width, img.height
    except Exception:
        return None, None


def _image_resample_filter():
    if not Image:
        return None
    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS


def _artwork_perceptual_hash(payload: bytes) -> int | None:
    if not payload or not Image:
        return None
    try:
        with Image.open(io.BytesIO(payload)) as img:
            resized = img.convert("L").resize(
                (ARTWORK_HASH_SIZE, ARTWORK_HASH_SIZE),
                resample=_image_resample_filter(),
            )
            pixels = list(resized.getdata())
    except Exception:
        return None
    if not pixels:
        return None
    avg = sum(pixels) / len(pixels)
    value = 0
    for px in pixels:
        value = (value << 1) | (1 if px >= avg else 0)
    return value


def _hamming_distance(left: int, right: int) -> int:
    xor = left ^ right
    return xor.bit_count() if hasattr(xor, "bit_count") else bin(xor).count("1")


def _artwork_hash_for_track(track: "DuplicateTrack") -> int | None:
    blob = _best_artwork_blob(track.artwork)
    if not blob or not blob.bytes:
        return None
    return _artwork_perceptual_hash(blob.bytes)


def _artwork_tracks_similar(
    left: "DuplicateTrack",
    right: "DuplicateTrack",
    hashes: Mapping[str, int | None],
) -> bool:
    left_hash = hashes.get(left.path)
    right_hash = hashes.get(right.path)
    if left_hash is None or right_hash is None:
        return False
    return _hamming_distance(left_hash, right_hash) <= ARTWORK_SIMILARITY_THRESHOLD


def _split_by_artwork_similarity(
    tracks: Sequence["DuplicateTrack"],
) -> tuple[List[List["DuplicateTrack"]], Dict[str, int | None]]:
    if len(tracks) <= 1:
        return [list(tracks)], {}
    hashes: Dict[str, int | None] = {t.path: _artwork_hash_for_track(t) for t in tracks}
    pending = sorted(tracks, key=lambda t: t.path)
    groups: List[List[DuplicateTrack]] = []
    while pending:
        anchor = pending.pop(0)
        group = [anchor]
        remaining: List[DuplicateTrack] = []
        for candidate in pending:
            if any(_artwork_tracks_similar(candidate, member, hashes) for member in group):
                group.append(candidate)
            else:
                remaining.append(candidate)
        pending = remaining
        groups.append(group)
    return groups, hashes


def _read_audio_file(path: str):
    if MutagenFile is None:
        return None, "mutagen unavailable"
    try:
        return MutagenFile(ensure_long_path(path)), None
    except Exception as exc:  # pragma: no cover - depends on local files
        return None, f"read failed: {exc}"


def _normalize_tags_from_audio(audio) -> Dict[str, object]:
    if audio is None or not getattr(audio, "tags", None):
        return _blank_tags()

    tags = audio.tags
    values: Dict[str, object] = _blank_tags()

    def pick(keys: Sequence[str]) -> object:
        for key in keys:
            if key in tags:
                return _first_value(tags.get(key))
            if hasattr(tags, "getall"):
                found = tags.getall(key)
                if found:
                    return _first_value(found)
        return None

    values["artist"] = pick(["artist", "TPE1"])
    values["albumartist"] = pick(["albumartist", "album artist", "TPE2"])
    values["title"] = pick(["title", "TIT2"])
    values["album"] = pick(["album", "TALB"])
    values["album_type"] = pick(["albumtype", "album_type", "releasetype", "release_type"])

    track_no = pick(["tracknumber", "track", "TRCK"])
    disc_no = pick(["discnumber", "disc", "TPOS"])
    values["track"] = _parse_int(track_no) or track_no or None
    values["tracknumber"] = values["track"]
    values["disc"] = _parse_int(disc_no) or disc_no or None
    values["discnumber"] = values["disc"]

    date_val = pick(["date", "year", "TDRC"])
    values["date"] = str(date_val) if date_val is not None else None
    values["year"] = _parse_int(date_val) or None

    values["genre"] = pick(["genre", "TCON"])
    values["compilation"] = pick(["compilation", "TCMP"])
    return values


def _extract_artwork_from_audio(audio, path: str) -> tuple[List[ArtworkCandidate], Optional[str]]:
    if audio is None:
        return [], "audio not readable"

    candidates: List[ArtworkCandidate] = []

    def _record(payload: bytes, status: str = "ok") -> None:
        if not payload:
            return
        width, height = _extract_image_dimensions(payload)
        candidates.append(
            ArtworkCandidate(
                path=path,
                hash=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
                width=width,
                height=height,
                status=status,
                bytes=payload,
            )
        )

    if hasattr(audio, "pictures"):
        for pic in getattr(audio, "pictures"):
            payload = getattr(pic, "data", None)
            if payload:
                width = getattr(pic, "width", None)
                height = getattr(pic, "height", None)
                _record(payload, status="ok")
                # Override dimensions if provided by mutagen
                if candidates[-1].width is None and width:
                    candidates[-1].width = width
                if candidates[-1].height is None and height:
                    candidates[-1].height = height
    if getattr(audio, "tags", None):
        try:
            apics = audio.tags.getall("APIC")
            for apic in apics:
                payload = getattr(apic, "data", None)
                if payload:
                    _record(payload, status="ok")
        except Exception:
            pass
        if "covr" in audio.tags:
            for cover in audio.tags.get("covr", []):
                payload = bytes(cover) if cover else None
                if payload:
                    _record(payload, status="ok")
        if "WM/Picture" in audio.tags:
            for pic in audio.tags.get("WM/Picture", []):
                payload = getattr(pic, "value", None)
                if payload:
                    _record(payload, status="ok")

    if not candidates and getattr(audio, "tags", None):
        return [], "no artwork tags"
    if not candidates:
        return [], "no artwork found"
    return candidates, None


def _coerce_int(value: object) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _extract_audio_properties(audio, path: str, provided_tags: Mapping[str, object], ext: str) -> Dict[str, object]:
    info = getattr(audio, "info", None)

    def pick_int(*candidates: object) -> int:
        for candidate in candidates:
            if candidate in (None, "", []):
                continue
            coerced = _coerce_int(candidate)
            if coerced is not None:
                return coerced
        return 0

    container = (
        str(provided_tags.get("container") or provided_tags.get("format") or ext or "")
        .replace(".", "")
        .upper()
    )
    if not container and audio is not None:
        container = audio.__class__.__name__.upper()

    codec = str(
        provided_tags.get("codec")
        or provided_tags.get("codec_name")
        or getattr(info, "codec_name", "")
        or getattr(info, "codec", "")
    )
    if not codec and hasattr(info, "pprint"):
        codec = str(info.pprint()).split(",")[0]
    if not codec:
        codec = container or os.path.splitext(path)[1].replace(".", "").upper()

    bitrate = pick_int(
        provided_tags.get("bitrate"),
        getattr(info, "bitrate", None),
        getattr(info, "bit_rate", None),
    )
    sample_rate = pick_int(
        provided_tags.get("sample_rate"),
        provided_tags.get("samplerate"),
        getattr(info, "sample_rate", None),
        getattr(info, "samplerate", None),
    )
    bit_depth = pick_int(
        provided_tags.get("bit_depth"),
        provided_tags.get("bitdepth"),
        getattr(info, "bits_per_sample", None),
        getattr(info, "bitdepth", None),
    )
    channels = pick_int(
        provided_tags.get("channels"),
        getattr(info, "channels", None),
        getattr(info, "channel", None),
    )

    return {
        "container": container,
        "codec": codec,
        "bitrate": bitrate,
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "channels": channels,
    }


def _read_tags_and_artwork(path: str, provided_tags: Mapping[str, object] | None) -> tuple[Dict[str, object], List[ArtworkCandidate], Optional[str], Optional[str], Dict[str, object]]:
    audio, error = _read_audio_file(path)
    file_tags = _normalize_tags_from_audio(audio)
    base = dict(file_tags)
    fallback = provided_tags or {}
    for key, val in (fallback or {}).items():
        if key in base and base[key] not in (None, "", []):
            continue
        base[key] = val
    artwork, art_error = _extract_artwork_from_audio(audio, path)
    if artwork:
        base["cover_hash"] = artwork[0].hash
        base["artwork_hash"] = artwork[0].hash
    audio_props = _extract_audio_properties(audio, path, provided_tags or {}, os.path.splitext(path)[1])
    return base, artwork, error, art_error, audio_props


def _placeholder_present(tags: Mapping[str, object]) -> bool:
    for key, value in tags.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            vals = value
        else:
            vals = [value]
        for val in vals:
            text = str(val).strip().lower()
            if any(token in text for token in PLACEHOLDER_TOKENS):
                return True
    return False


def _capture_library_state(path: str) -> Dict[str, object]:
    """Capture stable file attributes to detect changes between preview and execution."""

    state: Dict[str, object] = {"exists": os.path.exists(path)}
    if not state["exists"]:
        return state
    try:
        stat = os.stat(path)
        state["size"] = stat.st_size
        state["mtime"] = int(stat.st_mtime)
    except Exception as exc:  # pragma: no cover - depends on filesystem permissions
        state["stat_error"] = str(exc)
        return state

    hasher = hashlib.sha256()
    try:
        with open(ensure_long_path(path), "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        state["sha256"] = hasher.hexdigest()
    except Exception as exc:  # pragma: no cover - depends on filesystem permissions
        state["hash_error"] = str(exc)
    return state


@dataclass
class ArtworkCandidate:
    """Discovered artwork and its metadata."""

    path: str
    hash: str
    size: int
    width: Optional[int]
    height: Optional[int]
    status: str
    bytes: bytes | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "hash": self.hash,
            "size": self.size,
            "width": self.width,
            "height": self.height,
            "status": self.status,
        }


@dataclass
class DuplicateTrack:
    """Normalized representation of a track for consolidation planning."""

    path: str
    fingerprint: str | None
    ext: str
    bitrate: int = 0
    sample_rate: int = 0
    bit_depth: int = 0
    channels: int = 0
    codec: str = ""
    container: str = ""
    tags: Dict[str, object] = field(default_factory=dict)
    current_tags: Dict[str, object] = field(default_factory=dict)
    artwork: List[ArtworkCandidate] = field(default_factory=list)
    context_evidence: List[str] = field(default_factory=list)
    metadata_error: Optional[str] = None
    artwork_error: Optional[str] = None
    library_state: Dict[str, object] = field(default_factory=dict)

    @property
    def cover_hash(self) -> str | None:
        value = None
        tag_source = self.current_tags if isinstance(self.current_tags, Mapping) else self.tags
        value = tag_source.get("cover_hash") if isinstance(tag_source, Mapping) else None
        if value:
            return str(value)
        art = tag_source.get("artwork_hash") if isinstance(tag_source, Mapping) else None
        return str(art) if art else None

    @property
    def is_lossless(self) -> bool:
        ext = self.ext.lower()
        if not ext or ext == ".":
            ext = os.path.splitext(self.path)[1].lower()
        elif not ext.startswith("."):
            ext = f".{ext}"
        return ext in LOSSLESS_EXTS

    @property
    def metadata_count(self) -> int:
        tags = self.current_tags if isinstance(self.current_tags, Mapping) else self.tags
        if not isinstance(tags, Mapping):
            return 0
        keys = [
            "artist",
            "albumartist",
            "title",
            "album",
            "track",
            "tracknumber",
            "disc",
            "discnumber",
            "date",
            "year",
        ]
        return sum(1 for k in keys if tags.get(k))


@dataclass
class ArtworkDirective:
    """Instruction to copy artwork from a source to a target."""

    source: str
    target: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"source": self.source, "target": self.target, "reason": self.reason}


@dataclass
class PlaylistImpact:
    """Summary of playlist rewrites driven by a duplicate group."""

    playlists: int = 0
    entries: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {"playlists": self.playlists, "entries": self.entries}


@dataclass
class GroupingDecision:
    """Evidence for why a candidate joined a duplicate group."""

    anchor_path: str
    candidate_path: str
    metadata_key: tuple[str, str]
    coarse_keys_anchor: List[str]
    coarse_keys_candidate: List[str]
    shared_coarse_keys: List[str]
    distance_to_anchor: float
    max_group_distance: float
    threshold: float
    match_type: str
    coarse_gate: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "anchor_path": self.anchor_path,
            "candidate_path": self.candidate_path,
            "metadata_key": list(self.metadata_key),
            "coarse_keys_anchor": list(self.coarse_keys_anchor),
            "coarse_keys_candidate": list(self.coarse_keys_candidate),
            "shared_coarse_keys": list(self.shared_coarse_keys),
            "distance_to_anchor": self.distance_to_anchor,
            "max_group_distance": self.max_group_distance,
            "threshold": self.threshold,
            "match_type": self.match_type,
            "coarse_gate": self.coarse_gate,
        }


@dataclass
class ClusterResult:
    """Grouping outcome with evidence for why tracks clustered."""

    tracks: List[DuplicateTrack]
    metadata_key: tuple[str, str]
    decisions: List[GroupingDecision]
    exact_threshold: float
    near_threshold: float

@dataclass
class GroupPlan:
    """Planned actions for a duplicate group."""

    group_id: str
    winner_path: str
    losers: List[str]
    planned_winner_tags: Dict[str, object]
    winner_current_tags: Dict[str, object]
    current_tags: Dict[str, Dict[str, object]]
    metadata_changes: Dict[str, Dict[str, object]]
    winner_quality: Dict[str, object]
    artwork: List[ArtworkDirective]
    artwork_candidates: List[ArtworkCandidate]
    chosen_artwork_source: Dict[str, object]
    artwork_status: str
    loser_disposition: Dict[str, str]
    playlist_rewrites: Dict[str, str]
    playlist_impact: PlaylistImpact
    review_flags: List[str]
    context_summary: Dict[str, List[str]]
    context_evidence: Dict[str, List[str]]
    tag_source: Optional[str]
    placeholders_present: bool
    tag_source_reason: str
    tag_source_evidence: List[str]
    track_quality: Dict[str, Dict[str, object]]
    group_confidence: str
    group_match_type: str
    grouping_metadata_key: tuple[str, str]
    grouping_thresholds: Dict[str, float]
    grouping_decisions: List[GroupingDecision]
    artwork_evidence: List[str]
    fingerprint_distances: Dict[str, Dict[str, float]] = field(default_factory=dict)
    library_state: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "group_id": self.group_id,
            "winner_path": self.winner_path,
            "losers": list(self.losers),
            "planned_winner_tags": dict(self.planned_winner_tags),
            "winner_current_tags": dict(self.winner_current_tags),
            "current_tags": {k: dict(v) for k, v in self.current_tags.items()},
            "metadata_changes": {k: dict(v) for k, v in self.metadata_changes.items()},
            "winner_quality": dict(self.winner_quality),
            "artwork": [a.to_dict() for a in self.artwork],
            "artwork_candidates": [a.to_dict() for a in self.artwork_candidates],
            "chosen_artwork_source": dict(self.chosen_artwork_source),
            "artwork_status": self.artwork_status,
            "loser_disposition": dict(self.loser_disposition),
            "playlist_rewrites": dict(self.playlist_rewrites),
            "playlist_impact": self.playlist_impact.to_dict(),
            "review_flags": list(self.review_flags),
            "context_summary": {k: list(v) for k, v in self.context_summary.items()},
            "context_evidence": {k: list(v) for k, v in self.context_evidence.items()},
            "tag_source": self.tag_source,
            "placeholders_present": self.placeholders_present,
            "tag_source_reason": self.tag_source_reason,
            "tag_source_evidence": list(self.tag_source_evidence),
            "track_quality": {k: dict(v) for k, v in self.track_quality.items()},
            "group_confidence": self.group_confidence,
            "group_match_type": self.group_match_type,
            "grouping_metadata_key": list(self.grouping_metadata_key),
            "grouping_thresholds": dict(self.grouping_thresholds),
            "grouping_decisions": [d.to_dict() for d in self.grouping_decisions],
            "artwork_evidence": list(self.artwork_evidence),
            "fingerprint_distances": {k: dict(v) for k, v in self.fingerprint_distances.items()},
            "library_state": {k: dict(v) for k, v in self.library_state.items()},
        }


@dataclass
class ConsolidationPlan:
    """Aggregate plan across all duplicate groups."""

    groups: List[GroupPlan] = field(default_factory=list)
    review_flags: List[str] = field(default_factory=list)
    generated_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    placeholders_present: bool = False
    source_snapshot: Dict[str, Dict[str, object]] = field(default_factory=dict)
    fingerprint_settings: Dict[str, object] = field(default_factory=dict)
    threshold_settings: Dict[str, float] = field(default_factory=dict)
    plan_signature: str | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "review_flags": list(self.review_flags),
            "generated_at": self.generated_at.isoformat(),
            "placeholders_present": self.placeholders_present,
            "source_snapshot": {k: dict(v) for k, v in self.source_snapshot.items()},
            "fingerprint_settings": dict(self.fingerprint_settings),
            "threshold_settings": dict(self.threshold_settings),
            "plan_signature": self.plan_signature,
        }

    @property
    def review_required_groups(self) -> List[GroupPlan]:
        """Return groups that need manual review."""
        return [g for g in self.groups if g.review_flags]

    @property
    def review_required_count(self) -> int:
        """Return total number of review-required groups."""
        return len(self.review_required_groups) + (1 if self.review_flags else 0)

    @property
    def requires_review(self) -> bool:
        """Return whether any review flags (global or per-group) are present."""
        return bool(self.review_flags or self.review_required_groups)

    def __post_init__(self) -> None:
        if not self.source_snapshot:
            snapshot: Dict[str, Dict[str, object]] = {}
            for group in self.groups:
                for path, state in group.library_state.items():
                    snapshot[path] = dict(state)
            self.source_snapshot = snapshot
        if self.plan_signature is None:
            self.refresh_plan_signature()

    def refresh_plan_signature(self) -> str:
        canonical = json.dumps(
            {
                "generated_at": self.generated_at.isoformat(),
                "groups": [g.to_dict() for g in self.groups],
                "snapshot": {k: dict(v) for k, v in self.source_snapshot.items()},
                "fingerprint_settings": dict(self.fingerprint_settings),
                "threshold_settings": dict(self.threshold_settings),
            },
            sort_keys=True,
        )
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.plan_signature = signature
        return signature


@dataclass
class PairInspectionStep:
    name: str
    status: str
    detail: str


@dataclass
class DuplicatePairReport:
    generated_at: datetime.datetime
    track_a: DuplicateTrack
    track_b: DuplicateTrack
    anchor_path: str
    candidate_path: str
    metadata_key_a: tuple[str, str]
    metadata_key_b: tuple[str, str]
    bucket_key_a: tuple[str, str] | None
    bucket_key_b: tuple[str, str] | None
    metadata_bucket_match: bool
    coarse_keys_a: List[str]
    coarse_keys_b: List[str]
    shared_coarse_keys: List[str]
    coarse_gate: str
    fingerprint_distance: float | None
    exact_threshold: float
    near_threshold: float
    mixed_codec_boost: float
    mixed_codec: bool
    effective_threshold: float
    match_type: str
    verdict: str
    steps: List[PairInspectionStep]
    fingerprint_settings: Mapping[str, object] | None = None
    threshold_settings: Mapping[str, float] | None = None


def _expect_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping.")
    return value


def _expect_list(value: object, field: str) -> List[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")
    return value


def _expect_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string.")
    return value


def _expect_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


def _expect_optional_str(value: object, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null.")
    return value


def _expect_datetime(value: object, field: str) -> datetime.datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 datetime string.")
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime string.") from exc


def _expect_str_list(value: object, field: str) -> List[str]:
    items = _expect_list(value, field)
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"{field} must contain strings.")
        out.append(item)
    return out


def _expect_str_mapping(value: object, field: str) -> Dict[str, object]:
    mapping = _expect_mapping(value, field)
    return {str(k): v for k, v in mapping.items()}


def build_duplicate_pair_report(
    track_a: Mapping[str, object],
    track_b: Mapping[str, object],
    *,
    exact_duplicate_threshold: float = EXACT_DUPLICATE_THRESHOLD,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    mixed_codec_threshold_boost: float = 0.0,
    fingerprint_settings: Mapping[str, object] | None = None,
    threshold_settings: Mapping[str, float] | None = None,
) -> DuplicatePairReport:
    normalized_a = _normalize_track(track_a)
    normalized_b = _normalize_track(track_b)

    metadata_key_a = _metadata_bucket_key(normalized_a)
    metadata_key_b = _metadata_bucket_key(normalized_b)
    buckets = _build_metadata_buckets([normalized_a, normalized_b])
    bucket_key_a = next((key for key, items in buckets.items() if normalized_a in items), None)
    bucket_key_b = next((key for key, items in buckets.items() if normalized_b in items), None)
    metadata_bucket_match = bucket_key_a is not None and bucket_key_a == bucket_key_b

    coarse_keys_a = _coarse_fingerprint_keys(normalized_a.fingerprint)
    coarse_keys_b = _coarse_fingerprint_keys(normalized_b.fingerprint)
    shared_coarse_keys = sorted(set(coarse_keys_a) & set(coarse_keys_b))
    if shared_coarse_keys:
        coarse_gate = "match"
    elif not coarse_keys_a or not coarse_keys_b:
        coarse_gate = "missing coarse keys"
    else:
        coarse_gate = "none"

    exact_threshold = float(exact_duplicate_threshold)
    near_threshold = max(float(near_duplicate_threshold), exact_threshold)
    mixed_boost = float(mixed_codec_threshold_boost)
    mixed_codec = normalized_a.is_lossless != normalized_b.is_lossless
    effective_threshold = near_threshold + mixed_boost if mixed_codec else near_threshold

    fingerprint_distance_value: float | None = None
    if normalized_a.fingerprint and normalized_b.fingerprint:
        fingerprint_distance_value = fingerprint_distance(
            normalized_a.fingerprint, normalized_b.fingerprint
        )

    steps: List[PairInspectionStep] = []
    if metadata_bucket_match:
        steps.append(
            PairInspectionStep(
                name="Metadata gate",
                status="pass",
                detail=f"Bucket key {bucket_key_a!s} matched for both tracks.",
            )
        )
        metadata_ok = True
    else:
        steps.append(
            PairInspectionStep(
                name="Metadata gate",
                status="fail",
                detail=f"Bucket mismatch: {bucket_key_a!s} vs {bucket_key_b!s}.",
            )
        )
        metadata_ok = False

    if not metadata_ok:
        steps.append(
            PairInspectionStep(
                name="Coarse fingerprint gate",
                status="blocked",
                detail="Skipped because metadata gate failed.",
            )
        )
        coarse_ok = False
    elif shared_coarse_keys:
        steps.append(
            PairInspectionStep(
                name="Coarse fingerprint gate",
                status="pass",
                detail=f"Shared coarse keys: {', '.join(shared_coarse_keys)}.",
            )
        )
        coarse_ok = True
    else:
        steps.append(
            PairInspectionStep(
                name="Coarse fingerprint gate",
                status="fail",
                detail="No shared coarse keys to justify deeper comparison.",
            )
        )
        coarse_ok = False

    if not metadata_ok or not coarse_ok:
        steps.append(
            PairInspectionStep(
                name="Fingerprint distance gate",
                status="blocked",
                detail="Skipped because an earlier gate failed.",
            )
        )
        match_type = "blocked"
        verdict = "Not a match (blocked by earlier gate)"
    elif fingerprint_distance_value is None:
        steps.append(
            PairInspectionStep(
                name="Fingerprint distance gate",
                status="fail",
                detail="Fingerprint unavailable for one or both tracks.",
            )
        )
        match_type = "missing"
        verdict = "Not a match (fingerprint unavailable)"
    else:
        if fingerprint_distance_value <= exact_threshold:
            match_type = "exact"
            verdict = "Exact duplicate"
            steps.append(
                PairInspectionStep(
                    name="Fingerprint distance gate",
                    status="pass",
                    detail=(
                        f"Distance {fingerprint_distance_value:.4f} ≤ exact threshold "
                        f"{exact_threshold:.4f}."
                    ),
                )
            )
        elif fingerprint_distance_value <= effective_threshold:
            match_type = "near"
            verdict = "Near duplicate"
            steps.append(
                PairInspectionStep(
                    name="Fingerprint distance gate",
                    status="pass",
                    detail=(
                        f"Distance {fingerprint_distance_value:.4f} ≤ effective threshold "
                        f"{effective_threshold:.4f}."
                    ),
                )
            )
        else:
            match_type = "none"
            verdict = "Not a match"
            steps.append(
                PairInspectionStep(
                    name="Fingerprint distance gate",
                    status="fail",
                    detail=(
                        f"Distance {fingerprint_distance_value:.4f} exceeded effective "
                        f"threshold {effective_threshold:.4f}."
                    ),
                )
            )

    return DuplicatePairReport(
        generated_at=datetime.datetime.now(datetime.timezone.utc),
        track_a=normalized_a,
        track_b=normalized_b,
        anchor_path=normalized_a.path,
        candidate_path=normalized_b.path,
        metadata_key_a=metadata_key_a,
        metadata_key_b=metadata_key_b,
        bucket_key_a=bucket_key_a,
        bucket_key_b=bucket_key_b,
        metadata_bucket_match=metadata_bucket_match,
        coarse_keys_a=coarse_keys_a,
        coarse_keys_b=coarse_keys_b,
        shared_coarse_keys=shared_coarse_keys,
        coarse_gate=coarse_gate,
        fingerprint_distance=fingerprint_distance_value,
        exact_threshold=exact_threshold,
        near_threshold=near_threshold,
        mixed_codec_boost=mixed_boost,
        mixed_codec=mixed_codec,
        effective_threshold=effective_threshold,
        match_type=match_type,
        verdict=verdict,
        steps=steps,
        fingerprint_settings=fingerprint_settings,
        threshold_settings=threshold_settings,
    )


def _artwork_candidate_from_dict(raw: Mapping[str, object]) -> ArtworkCandidate:
    return ArtworkCandidate(
        path=_expect_str(raw.get("path"), "artwork_candidates.path"),
        hash=_expect_str(raw.get("hash"), "artwork_candidates.hash"),
        size=int(raw.get("size") or 0),
        width=raw.get("width") if raw.get("width") is None else int(raw.get("width") or 0),
        height=raw.get("height") if raw.get("height") is None else int(raw.get("height") or 0),
        status=_expect_str(raw.get("status"), "artwork_candidates.status"),
    )


def _artwork_directive_from_dict(raw: Mapping[str, object]) -> ArtworkDirective:
    return ArtworkDirective(
        source=_expect_str(raw.get("source"), "artwork.source"),
        target=_expect_str(raw.get("target"), "artwork.target"),
        reason=_expect_str(raw.get("reason"), "artwork.reason"),
    )


def _playlist_impact_from_dict(raw: Mapping[str, object]) -> PlaylistImpact:
    return PlaylistImpact(
        playlists=int(raw.get("playlists") or 0),
        entries=int(raw.get("entries") or 0),
    )


def _grouping_decision_from_dict(raw: Mapping[str, object]) -> GroupingDecision:
    metadata_key = raw.get("metadata_key")
    if not isinstance(metadata_key, (list, tuple)) or len(metadata_key) != 2:
        raise ValueError("grouping_decisions.metadata_key must be a 2-item list.")
    return GroupingDecision(
        anchor_path=_expect_str(raw.get("anchor_path"), "grouping_decisions.anchor_path"),
        candidate_path=_expect_str(raw.get("candidate_path"), "grouping_decisions.candidate_path"),
        metadata_key=(str(metadata_key[0]), str(metadata_key[1])),
        coarse_keys_anchor=_expect_str_list(raw.get("coarse_keys_anchor"), "grouping_decisions.coarse_keys_anchor"),
        coarse_keys_candidate=_expect_str_list(raw.get("coarse_keys_candidate"), "grouping_decisions.coarse_keys_candidate"),
        shared_coarse_keys=_expect_str_list(raw.get("shared_coarse_keys"), "grouping_decisions.shared_coarse_keys"),
        distance_to_anchor=float(raw.get("distance_to_anchor") or 0.0),
        max_group_distance=float(raw.get("max_group_distance") or 0.0),
        threshold=float(raw.get("threshold") or 0.0),
        match_type=_expect_str(raw.get("match_type"), "grouping_decisions.match_type"),
        coarse_gate=_expect_str(raw.get("coarse_gate"), "grouping_decisions.coarse_gate"),
    )


def _group_plan_from_dict(raw: Mapping[str, object]) -> GroupPlan:
    metadata_key = raw.get("grouping_metadata_key")
    if not isinstance(metadata_key, (list, tuple)) or len(metadata_key) != 2:
        raise ValueError("grouping_metadata_key must be a 2-item list.")

    artwork_raw = _expect_list(raw.get("artwork"), "artwork")
    artwork_candidates_raw = _expect_list(raw.get("artwork_candidates"), "artwork_candidates")
    decisions_raw = _expect_list(raw.get("grouping_decisions"), "grouping_decisions")
    playlist_impact_raw = _expect_mapping(raw.get("playlist_impact"), "playlist_impact")

    return GroupPlan(
        group_id=_expect_str(raw.get("group_id"), "group_id"),
        winner_path=_expect_str(raw.get("winner_path"), "winner_path"),
        losers=_expect_str_list(raw.get("losers"), "losers"),
        planned_winner_tags=_expect_str_mapping(raw.get("planned_winner_tags"), "planned_winner_tags"),
        winner_current_tags=_expect_str_mapping(raw.get("winner_current_tags"), "winner_current_tags"),
        current_tags=_expect_mapping(raw.get("current_tags"), "current_tags"),
        metadata_changes=_expect_mapping(raw.get("metadata_changes"), "metadata_changes"),
        winner_quality=_expect_mapping(raw.get("winner_quality"), "winner_quality"),
        artwork=[_artwork_directive_from_dict(_expect_mapping(item, "artwork[]")) for item in artwork_raw],
        artwork_candidates=[
            _artwork_candidate_from_dict(_expect_mapping(item, "artwork_candidates[]")) for item in artwork_candidates_raw
        ],
        chosen_artwork_source=_expect_mapping(raw.get("chosen_artwork_source"), "chosen_artwork_source"),
        artwork_status=_expect_str(raw.get("artwork_status"), "artwork_status"),
        loser_disposition=_expect_str_mapping(raw.get("loser_disposition"), "loser_disposition"),
        playlist_rewrites=_expect_str_mapping(raw.get("playlist_rewrites"), "playlist_rewrites"),
        playlist_impact=_playlist_impact_from_dict(playlist_impact_raw),
        review_flags=_expect_str_list(raw.get("review_flags"), "review_flags"),
        context_summary=_expect_mapping(raw.get("context_summary"), "context_summary"),
        context_evidence=_expect_mapping(raw.get("context_evidence"), "context_evidence"),
        tag_source=_expect_optional_str(raw.get("tag_source"), "tag_source"),
        placeholders_present=_expect_bool(raw.get("placeholders_present"), "placeholders_present"),
        tag_source_reason=_expect_str(raw.get("tag_source_reason"), "tag_source_reason"),
        tag_source_evidence=_expect_str_list(raw.get("tag_source_evidence"), "tag_source_evidence"),
        track_quality=_expect_mapping(raw.get("track_quality"), "track_quality"),
        group_confidence=_expect_str(raw.get("group_confidence"), "group_confidence"),
        group_match_type=_expect_str(raw.get("group_match_type"), "group_match_type"),
        grouping_metadata_key=(str(metadata_key[0]), str(metadata_key[1])),
        grouping_thresholds=_expect_mapping(raw.get("grouping_thresholds"), "grouping_thresholds"),
        grouping_decisions=[
            _grouping_decision_from_dict(_expect_mapping(item, "grouping_decisions[]")) for item in decisions_raw
        ],
        artwork_evidence=_expect_str_list(raw.get("artwork_evidence"), "artwork_evidence"),
        fingerprint_distances=_expect_mapping(raw.get("fingerprint_distances"), "fingerprint_distances"),
        library_state=_expect_mapping(raw.get("library_state"), "library_state"),
    )


def consolidation_plan_from_dict(raw: Mapping[str, object]) -> ConsolidationPlan:
    """Convert a serialized plan mapping into a ConsolidationPlan."""
    groups_raw = _expect_list(raw.get("groups"), "groups")
    review_flags = _expect_str_list(raw.get("review_flags"), "review_flags")
    generated_at = _expect_datetime(raw.get("generated_at"), "generated_at")
    placeholders_present = _expect_bool(raw.get("placeholders_present"), "placeholders_present")
    source_snapshot = _expect_mapping(raw.get("source_snapshot"), "source_snapshot")
    fingerprint_settings = raw.get("fingerprint_settings") if isinstance(raw, Mapping) else None
    threshold_settings = raw.get("threshold_settings") if isinstance(raw, Mapping) else None
    plan_signature = raw.get("plan_signature")
    if plan_signature is not None and not isinstance(plan_signature, str):
        raise ValueError("plan_signature must be a string or null.")

    groups = [_group_plan_from_dict(_expect_mapping(item, "groups[]")) for item in groups_raw]
    return ConsolidationPlan(
        groups=groups,
        review_flags=review_flags,
        generated_at=generated_at,
        placeholders_present=placeholders_present,
        source_snapshot=source_snapshot,
        fingerprint_settings=dict(fingerprint_settings or {}),
        threshold_settings={str(k): float(v) for k, v in dict(threshold_settings or {}).items()},
        plan_signature=plan_signature,
    )


def _normalize_track(raw: Mapping[str, object]) -> DuplicateTrack:
    path = str(raw.get("path"))
    ext = os.path.splitext(path)[1].lower() or str(raw.get("ext", "")).lower()
    provided_tags = raw.get("tags") if isinstance(raw.get("tags"), Mapping) else {}
    current_tags, artwork, meta_err, art_err, audio_props = _read_tags_and_artwork(path, provided_tags)
    library_state = _capture_library_state(path)
    provided_artwork: List[ArtworkCandidate] = []
    for art in raw.get("artwork", []) if isinstance(raw.get("artwork"), list) else []:
        try:
            provided_artwork.append(
                ArtworkCandidate(
                    path=path,
                    hash=str(art.get("hash")),
                    size=int(art.get("size") or 0),
                    width=art.get("width"),
                    height=art.get("height"),
                    status=art.get("status", "ok"),
                    bytes=art.get("bytes"),
                )
            )
        except Exception:
            continue
    if provided_artwork and not artwork:
        artwork = provided_artwork
        if "cover_hash" not in current_tags:
            current_tags["cover_hash"] = provided_artwork[0].hash
            current_tags["artwork_hash"] = provided_artwork[0].hash
    tags = dict(current_tags)
    return DuplicateTrack(
        path=path,
        fingerprint=raw.get("fingerprint"),
        ext=ext,
        bitrate=int(audio_props.get("bitrate") or raw.get("bitrate") or 0),
        sample_rate=int(audio_props.get("sample_rate") or raw.get("sample_rate") or raw.get("samplerate") or 0),
        bit_depth=int(audio_props.get("bit_depth") or raw.get("bit_depth") or raw.get("bitdepth") or 0),
        channels=int(audio_props.get("channels") or raw.get("channels") or 0),
        codec=str(audio_props.get("codec") or raw.get("codec") or ""),
        container=str(audio_props.get("container") or raw.get("container") or ext.replace(".", "")),
        tags=dict(tags),
        current_tags=dict(current_tags),
        artwork=artwork,
        metadata_error=meta_err,
        artwork_error=art_err,
        library_state=library_state,
    )


def _classify_context(track: DuplicateTrack) -> tuple[str, List[str]]:
    tags = track.current_tags or track.tags or {}
    album = str(tags.get("album") or "").strip()
    album_type = str(tags.get("album_type") or tags.get("release_type") or "").lower()
    track_no = tags.get("track") or tags.get("tracknumber")
    disc_no = tags.get("disc") or tags.get("discnumber")

    evidence: List[str] = []

    if album_type == "single":
        evidence.append("Album type tag indicates single")
        return "single", evidence
    if album_type in {"album", "lp"}:
        evidence.append("Album type tag indicates album/LP")
        return "album", evidence
    if album.lower().endswith(" - single") or "(single" in album.lower():
        evidence.append("Album title formatted like a single")
        return "single", evidence
    if album and (track_no or disc_no):
        evidence.append("Album tag present with track/disc number")
        return "album", evidence
    if not album:
        evidence.append("No album tag present")
        return "single", evidence
    if track.cover_hash:
        evidence.append("Embedded artwork present suggesting album packaging")
        return "album", evidence
    evidence.append("Insufficient context to classify")
    return "unknown", evidence


def _quality_tuple(track: DuplicateTrack, context: str) -> tuple:
    return (
        1 if track.is_lossless else 0,
        track.bitrate,
        track.sample_rate,
        track.bit_depth,
        track.channels,
        track.path.lower(),
    )


def _stable_group_id(paths: Sequence[str]) -> str:
    canonical = "|".join(sorted(paths))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _has_review_keyword(track: DuplicateTrack) -> bool:
    title = ""
    tags = track.tags or {}
    if isinstance(tags, Mapping):
        title = str(tags.get("title") or tags.get("name") or "").lower()
    if not title:
        title = os.path.splitext(os.path.basename(track.path))[0].lower()
    return any(keyword in title for keyword in REVIEW_KEYWORDS)


def _select_metadata_source(candidates: Sequence[DuplicateTrack], contexts: Mapping[str, str]) -> DuplicateTrack | None:
    album_candidates = [c for c in candidates if contexts.get(c.path) == "album"]
    sorted_cands = sorted(
        album_candidates or list(candidates),
        key=lambda c: (c.metadata_count, c.bitrate, c.sample_rate, c.bit_depth, c.path.lower()),
        reverse=True,
    )
    return sorted_cands[0] if sorted_cands else None


def _merge_tags(existing: MutableMapping[str, object], source: Mapping[str, object]) -> Dict[str, object]:
    merged = _blank_tags()
    for key, value in existing.items():
        merged[key] = value
    for key, value in source.items():
        if key in merged and merged[key]:
            continue
        merged[key] = value
    return merged


def _build_planned_tags(
    winner: DuplicateTrack,
    metadata_source: DuplicateTrack | None,
    contexts: Mapping[str, str],
) -> tuple[Dict[str, object], Optional[str], str, List[str]]:
    planned_tags: Dict[str, object] = _merge_tags(_blank_tags(), winner.current_tags)
    tag_source: Optional[str] = None
    tag_source_reason = "Winner tags retained (no metadata source found)"
    tag_source_evidence: List[str] = []

    if metadata_source:
        tag_source = metadata_source.path
        src_context = contexts.get(metadata_source.path, "unknown")
        tag_source_reason = f"Selected {src_context} metadata source with {metadata_source.metadata_count} populated fields"
        src_tags = metadata_source.current_tags if isinstance(metadata_source.current_tags, Mapping) else {}
        normalized_keys = [
            "album",
            "albumartist",
            "artist",
            "album_type",
            "track",
            "tracknumber",
            "disc",
            "discnumber",
            "date",
            "year",
        ]
        for key in normalized_keys:
            if src_tags.get(key) in (None, "", []):
                continue
            planned_tags[key] = src_tags.get(key)
        if src_context == "album":
            planned_tags["album_type"] = src_tags.get("album_type") or "album"
        elif src_context == "single":
            planned_tags["album_type"] = src_tags.get("album_type") or "single"
        tag_source_evidence.append(tag_source_reason)
    else:
        tag_source_evidence.append("No album-context source available; keeping winner metadata")

    return planned_tags, tag_source, tag_source_reason, tag_source_evidence


def _metadata_changes(winner: DuplicateTrack, planned_tags: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    keys = TAG_KEYS
    changes: Dict[str, Dict[str, object]] = {}
    for key in keys:
        current = winner.current_tags.get(key) if isinstance(winner.current_tags, Mapping) else None
        planned = planned_tags.get(key)
        if (current or "") == (planned or ""):
            continue
        changes[key] = {"from": current, "to": planned}
    return changes


def _quality_rationale(winner: DuplicateTrack, runner_up: DuplicateTrack | None, context: str) -> Dict[str, object]:
    reasons: List[str] = []

    def fmt_codec(t: DuplicateTrack) -> str:
        container = t.container or t.ext.replace(".", "").upper()
        codec = t.codec or container
        return f"{container} / {codec}"

    reasons.append(f"Container/codec: {fmt_codec(winner)}")
    if winner.is_lossless:
        reasons.append("Lossless encoding")
    if runner_up and winner.is_lossless and not runner_up.is_lossless:
        reasons.append(f"Lossless beats lossy ({fmt_codec(winner)} > {fmt_codec(runner_up)})")
    if runner_up and winner.bitrate != runner_up.bitrate:
        if winner.bitrate > runner_up.bitrate:
            reasons.append(f"Higher bitrate: {winner.bitrate} vs {runner_up.bitrate} kbps")
        else:
            reasons.append(f"Bitrate tie broken by other properties ({winner.bitrate} vs {runner_up.bitrate} kbps)")
    if runner_up and winner.sample_rate != runner_up.sample_rate:
        if winner.sample_rate > runner_up.sample_rate:
            reasons.append(f"Higher sample rate: {winner.sample_rate} Hz vs {runner_up.sample_rate} Hz")
        else:
            reasons.append("Sample rate tie handled by deterministic ordering")
    if runner_up and winner.bit_depth != runner_up.bit_depth:
        if winner.bit_depth > runner_up.bit_depth:
            reasons.append(f"Higher bit depth: {winner.bit_depth}-bit vs {runner_up.bit_depth}-bit")
        else:
            reasons.append("Bit depth tie handled by deterministic ordering")
    if runner_up and winner.channels != runner_up.channels:
        if winner.channels > runner_up.channels:
            reasons.append(f"More channels: {winner.channels} vs {runner_up.channels}")
        else:
            reasons.append("Channel count tie handled by deterministic ordering")

    return {
        "context": context,
        "is_lossless": winner.is_lossless,
        "bitrate": winner.bitrate,
        "sample_rate": winner.sample_rate,
        "bit_depth": winner.bit_depth,
        "channels": winner.channels,
        "codec": winner.codec or "",
        "container": winner.container or winner.ext.replace(".", "").upper(),
        "metadata_count": winner.metadata_count,
        "reasons": reasons or ["Deterministic ordering by quality"],
        "runner_up": {
            "bitrate": runner_up.bitrate if runner_up else None,
            "sample_rate": runner_up.sample_rate if runner_up else None,
            "bit_depth": runner_up.bit_depth if runner_up else None,
            "channels": runner_up.channels if runner_up else None,
            "codec": runner_up.codec if runner_up else "",
            "container": runner_up.container if runner_up else "",
            "path": runner_up.path if runner_up else None,
        },
    }


def _artwork_blob_score(blob: ArtworkCandidate) -> tuple:
    resolution = 0
    if blob.width and blob.height:
        resolution = blob.width * blob.height
    has_dims = 1 if blob.width and blob.height else 0
    return has_dims, resolution, blob.size


def _best_artwork_blob(candidates: Sequence[ArtworkCandidate]) -> ArtworkCandidate | None:
    usable = [c for c in candidates if c.status == "ok"] or list(candidates)
    if not usable:
        return None
    return sorted(usable, key=lambda c: (_artwork_blob_score(c), c.hash), reverse=True)[0]


def _select_single_release_artwork_candidate(
    candidates: Sequence[DuplicateTrack], release_sizes: Mapping[str, str]
) -> tuple[DuplicateTrack | None, ArtworkCandidate | None, bool]:
    best_track: DuplicateTrack | None = None
    best_blob: ArtworkCandidate | None = None
    best_score: tuple | None = None
    ambiguous = False
    for track in candidates:
        if release_sizes.get(track.path) != "single" or not track.artwork:
            continue
        blob = _best_artwork_blob(track.artwork)
        if not blob:
            continue
        score = _artwork_blob_score(blob)
        if best_score is None or score > best_score:
            best_score = score
            best_track = track
            best_blob = blob
            ambiguous = False
        elif score == best_score:
            ambiguous = True
    return best_track, best_blob, ambiguous


def _select_overall_artwork_candidate(candidates: Sequence[DuplicateTrack]) -> tuple[DuplicateTrack | None, ArtworkCandidate | None, bool]:
    best_track: DuplicateTrack | None = None
    best_blob: ArtworkCandidate | None = None
    best_score: tuple | None = None
    ambiguous = False
    for track in candidates:
        if not track.artwork:
            continue
        blob = _best_artwork_blob(track.artwork)
        if not blob:
            continue
        score = _artwork_blob_score(blob)
        if best_score is None or score > best_score:
            best_score = score
            best_track = track
            best_blob = blob
            ambiguous = False
        elif score == best_score:
            ambiguous = True
    return best_track, best_blob, ambiguous


def _cluster_duplicates(
    tracks: Sequence[DuplicateTrack],
    *,
    exact_duplicate_threshold: float,
    near_duplicate_threshold: float,
    mixed_codec_threshold_boost: float,
    cancel_event: threading.Event,
    max_comparisons: int,
    timeout_sec: float,
    progress_callback: Callable[[int, int, str], None],
    start_time: float,
    review_flags: List[str],
) -> List[ClusterResult]:
    near_duplicate_threshold = max(near_duplicate_threshold, exact_duplicate_threshold)
    buckets = _build_metadata_buckets(tracks)
    bucket_items = sorted(buckets.items(), key=lambda x: x[0])
    clusters: List[ClusterResult] = []
    comparisons = 0
    processed = 0
    for bucket_key, bucket_tracks in bucket_items:
        if cancel_event.is_set() or (timeout_sec and (_now() - start_time) > timeout_sec):
            review_flags.append("Consolidation planning cancelled or timed out during grouping.")
            break
        bucket_tracks = [t for t in bucket_tracks if t.fingerprint]
        if not bucket_tracks:
            continue
        bucket_tracks = sorted(bucket_tracks, key=lambda t: t.path.lower())
        path_to_track = {t.path: t for t in bucket_tracks}
        path_order = {t.path: idx for idx, t in enumerate(bucket_tracks)}

        track_keys: Dict[str, List[str]] = {t.path: _coarse_fingerprint_keys(t.fingerprint) for t in bucket_tracks}
        coarse_index: Dict[str, set[str]] = defaultdict(set)
        for path, keys in track_keys.items():
            for key in keys:
                coarse_index[key].add(path)

        used: set[str] = set()
        for idx, track in enumerate(bucket_tracks):
            if (
                cancel_event.is_set()
                or comparisons >= max_comparisons
                or (timeout_sec and (_now() - start_time) > timeout_sec)
            ):
                if cancel_event.is_set() and "Consolidation planning cancelled or timed out during grouping." not in review_flags:
                    review_flags.append("Consolidation planning cancelled or timed out during grouping.")
                elif comparisons >= max_comparisons and "Comparison budget reached; grouping may be incomplete." not in review_flags:
                    review_flags.append("Comparison budget reached; grouping may be incomplete.")
                elif timeout_sec and (_now() - start_time) > timeout_sec and "Consolidation planning timed out while grouping." not in review_flags:
                    review_flags.append("Consolidation planning timed out while grouping.")
                break
            if track.path in used:
                continue
            group = [track]
            decisions: List[GroupingDecision] = []
            used.add(track.path)
            candidate_paths: set[str] = set()
            for key in track_keys.get(track.path, []):
                candidate_paths.update(coarse_index.get(key, set()))
            if not candidate_paths:
                fallback_candidates = True
                candidate_paths.update(t.path for t in bucket_tracks[idx + 1 :])
            else:
                fallback_candidates = False
            for other_path in sorted(candidate_paths, key=lambda p: path_order.get(p, len(bucket_tracks))):
                if cancel_event.is_set() or comparisons >= max_comparisons:
                    review_flags.append("Comparison budget reached; grouping may be incomplete.")
                    break
                if timeout_sec and (_now() - start_time) > timeout_sec:
                    review_flags.append("Consolidation planning timed out while grouping.")
                    break
                if other_path in used or path_order.get(other_path, 0) <= idx:
                    continue
                comparisons += 1
                other = path_to_track[other_path]
                dist = fingerprint_distance(track.fingerprint, other.fingerprint)
                pair_threshold = near_duplicate_threshold
                if track.is_lossless != other.is_lossless:
                    pair_threshold += mixed_codec_threshold_boost
                if dist > pair_threshold:
                    continue
                compatible = True
                max_candidate_distance = dist
                for member in group:
                    if member.path == other.path:
                        continue
                    cmp_dist = fingerprint_distance(member.fingerprint, other.fingerprint)
                    max_candidate_distance = max(max_candidate_distance, cmp_dist)
                    member_threshold = near_duplicate_threshold
                    if member.is_lossless != other.is_lossless:
                        member_threshold += mixed_codec_threshold_boost
                    if cmp_dist > member_threshold:
                        compatible = False
                        break
                if compatible:
                    anchor_keys = track_keys.get(track.path, [])
                    candidate_keys = track_keys.get(other.path, [])
                    shared_keys = sorted(set(anchor_keys) & set(candidate_keys))
                    coarse_gate = "match" if shared_keys else "fallback (no shared coarse keys)" if fallback_candidates else "none"
                    match_type = "exact" if max_candidate_distance <= exact_duplicate_threshold else "near"
                    decisions.append(
                        GroupingDecision(
                            anchor_path=track.path,
                            candidate_path=other.path,
                            metadata_key=bucket_key,
                            coarse_keys_anchor=anchor_keys,
                            coarse_keys_candidate=candidate_keys,
                            shared_coarse_keys=shared_keys,
                            distance_to_anchor=dist,
                            max_group_distance=max_candidate_distance,
                            threshold=pair_threshold,
                            match_type=match_type,
                            coarse_gate=coarse_gate,
                        )
                    )
                    group.append(other)
                    used.add(other.path)
            if len(group) > 1:
                clusters.append(
                    ClusterResult(
                        tracks=group,
                        metadata_key=bucket_key,
                        decisions=decisions,
                        exact_threshold=exact_duplicate_threshold,
                        near_threshold=near_duplicate_threshold,
                    )
                )
            processed += 1
            progress_callback(processed, len(bucket_tracks), track.path)
        if cancel_event.is_set() or comparisons >= max_comparisons or (timeout_sec and (_now() - start_time) > timeout_sec):
            break
    return clusters


def build_consolidation_plan(
    tracks: Iterable[Mapping[str, object]],
    *,
    exact_duplicate_threshold: float = EXACT_DUPLICATE_THRESHOLD,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
    mixed_codec_threshold_boost: float = 0.0,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_comparisons: int = DEFAULT_MAX_COMPARISONS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    fingerprint_settings: Mapping[str, object] | None = None,
    threshold_settings: Mapping[str, float] | None = None,
) -> ConsolidationPlan:
    """Generate a deterministic consolidation plan without modifying files."""

    cancel_event = cancel_event or threading.Event()
    progress_callback = progress_callback or _default_progress
    review_flags: List[str] = []
    start_time = _now()

    normalized: List[DuplicateTrack] = []
    for raw in tracks:
        if cancel_event.is_set():
            review_flags.append("Cancelled before normalization.")
            break
        normalized.append(_normalize_track(raw))
        if len(normalized) >= max_candidates:
            review_flags.append(f"Truncated candidate set to {max_candidates} items to protect runtime.")
            break

    clusters = _cluster_duplicates(
        normalized,
        exact_duplicate_threshold=exact_duplicate_threshold,
        near_duplicate_threshold=near_duplicate_threshold,
        mixed_codec_threshold_boost=mixed_codec_threshold_boost,
        cancel_event=cancel_event,
        max_comparisons=max_comparisons,
        timeout_sec=timeout_sec,
        progress_callback=progress_callback,
        start_time=start_time,
        review_flags=review_flags,
    )

    plans: List[GroupPlan] = []
    plan_placeholders = False
    for cluster in sorted(clusters, key=lambda c: _stable_group_id([t.path for t in c.tracks])):
        cluster_tracks = cluster.tracks
        cluster_has_missing_artwork = any(not t.artwork for t in cluster_tracks)
        cluster_has_unreadable_artwork = any(t.artwork_error for t in cluster_tracks)
        artwork_groups, artwork_hashes = _split_by_artwork_similarity(cluster_tracks)
        artwork_split = len(artwork_groups) > 1
        for group_tracks in sorted(artwork_groups, key=lambda g: _stable_group_id([t.path for t in g])):
            contexts: Dict[str, str] = {}
            context_evidence: Dict[str, List[str]] = {}
            release_sizes: Dict[str, str] = {}
            release_size_evidence: Dict[str, str] = {}
            for t in group_tracks:
                ctx, evidence = _classify_context(t)
                contexts[t.path] = ctx
                t.context_evidence = evidence
                context_evidence[t.path] = evidence
                tags = t.current_tags if isinstance(t.current_tags, Mapping) else t.tags
                tags = tags or {}
                release_size, release_reason = _release_size_context(tags)
                release_sizes[t.path] = release_size
                release_size_evidence[t.path] = release_reason

            album_label_by_key: Dict[tuple[str, str], str] = {}
            album_keys: set[tuple[str, str]] = set()
            for t in group_tracks:
                tags = t.current_tags if isinstance(t.current_tags, Mapping) else t.tags
                tags = tags or {}
                album_key = _normalize_album_key(tags, t.path)
                if album_key:
                    album_keys.add(album_key)
                    if album_key not in album_label_by_key:
                        album_label_by_key[album_key] = _album_display_label(tags)

            suppress_review_flags = artwork_split and len(group_tracks) == 1
            release_notes: List[str] = []
            if len(album_keys) > 1 and not suppress_review_flags:
                album_labels = [album_label_by_key.get(key, "Unknown album") for key in sorted(album_keys)]
                album_summary = ", ".join(sorted({label for label in album_labels if label}))
                cross_album_note = "Identical recording across albums; consolidated into one canonical track."
                if album_summary:
                    cross_album_note = f"{cross_album_note} Albums: {album_summary}."
                release_notes.append(cross_album_note)
            if len({contexts[path] for path in contexts}) > 1 and not suppress_review_flags:
                release_notes.append("Cluster spans multiple release contexts; using one canonical winner across variants.")

            group_paths = {t.path for t in group_tracks}
            group_contexts = {path: contexts[path] for path in group_paths}
            group_context_evidence = {path: context_evidence[path] for path in group_paths}
            group_decisions = [
                d for d in cluster.decisions if d.anchor_path in group_paths and d.candidate_path in group_paths
            ]

            quality_sorted = sorted(group_tracks, key=lambda t: _quality_tuple(t, group_contexts[t.path]), reverse=True)
            winner = quality_sorted[0]
            losers = [t.path for t in quality_sorted[1:]]
            runner_up = quality_sorted[1] if len(quality_sorted) > 1 else None

            metadata_source = _select_metadata_source(group_tracks, group_contexts)
            planned_tags, tag_source, tag_source_reason, tag_source_evidence = _build_planned_tags(
                winner, metadata_source, group_contexts
            )
            group_review: List[str] = list(release_notes)
            if not metadata_source and not suppress_review_flags:
                review_flags.append(f"Missing metadata source for group containing {winner.path}.")
                group_review.append("Metadata source missing; tags may be incomplete.")

            meta_changes = _metadata_changes(winner, planned_tags)
            winner_context = group_contexts.get(winner.path, "unknown")
            winner_quality = _quality_rationale(winner, runner_up, winner_context)

            distance_samples: List[float] = []
            pair_distances: Dict[str, Dict[str, float]] = {t.path: {} for t in group_tracks}
            missing_fingerprints = any(not t.fingerprint for t in group_tracks)
            for idx, t in enumerate(group_tracks):
                for other in group_tracks[idx + 1 :]:
                    dist = fingerprint_distance(t.fingerprint, other.fingerprint)
                    distance_samples.append(dist)
                    pair_distances[t.path][other.path] = dist
                    pair_distances[other.path][t.path] = dist
            max_distance = max(distance_samples) if distance_samples else (1.0 if missing_fingerprints else 0.0)

            track_quality: Dict[str, Dict[str, object]] = {
                t.path: {
                    "container": t.container or t.ext.replace(".", "").upper(),
                    "codec": t.codec,
                    "bitrate": t.bitrate,
                    "sample_rate": t.sample_rate,
                    "bit_depth": t.bit_depth,
                    "channels": t.channels,
                    "is_lossless": t.is_lossless,
                }
                for t in quality_sorted
            }

            single_art_track, single_art_blob, ambiguous_single_art = _select_single_release_artwork_candidate(
                group_tracks, release_sizes
            )
            overall_art_track, overall_art_blob, ambiguous_overall_art = _select_overall_artwork_candidate(
                group_tracks
            )
            artwork_actions: List[ArtworkDirective] = []
            chosen_artwork_source: Dict[str, object] = {"path": None, "reason": ""}
            artwork_status = "unchanged"
            artwork_evidence: List[str] = []
            winner_release = release_sizes.get(winner.path, "unknown")
            winner_multi_release = winner_release == "multi"
            if winner_release != "unknown":
                artwork_evidence.append(
                    f"Winner release size classified as {winner_release} ({release_size_evidence.get(winner.path)})."
                )
            if artwork_split:
                if len(group_tracks) == 1:
                    group_track = group_tracks[0]
                    if not group_track.artwork or group_track.artwork_error:
                        artwork_evidence.append(
                            "Missing or unreadable artwork treated as non-matching; preserving this track as an intentional variant."
                        )
                    elif cluster_has_missing_artwork or cluster_has_unreadable_artwork:
                        artwork_evidence.append(
                            "Another duplicate lacks readable artwork; missing artwork treated as non-matching and this track is preserved."
                        )
                    else:
                        artwork_evidence.append(
                            "Artwork differs from other audio duplicates; preserving this track as an intentional variant."
                        )
                else:
                    artwork_evidence.append(
                        "Artwork similarity gate grouped only tracks with matching covers for consolidation."
                    )

            if single_art_blob:
                chosen_artwork_source = {
                    "path": single_art_track.path if single_art_track else None,
                    "hash": single_art_blob.hash,
                    "size": single_art_blob.size,
                    "width": single_art_blob.width,
                    "height": single_art_blob.height,
                    "context": "single-release",
                    "reason": "Best single-release artwork by resolution/size",
                }
                artwork_evidence.append(
                    f"Single-release artwork selected from {single_art_track.path if single_art_track else 'unknown'} "
                    f"({single_art_blob.width}x{single_art_blob.height}, {single_art_blob.size} bytes)"
                )
                if single_art_track:
                    artwork_evidence.append(
                        f"Single-release classification: {release_size_evidence.get(single_art_track.path)}."
                    )
            elif overall_art_blob:
                chosen_artwork_source = {
                    "path": overall_art_track.path if overall_art_track else None,
                    "hash": overall_art_blob.hash,
                    "size": overall_art_blob.size,
                    "width": overall_art_blob.width,
                    "height": overall_art_blob.height,
                    "context": "fallback",
                    "reason": "No single artwork found; best available artwork",
                }
                artwork_evidence.append(
                    f"No single artwork found; using best available artwork from {overall_art_track.path if overall_art_track else 'unknown'} "
                    f"({overall_art_blob.width}x{overall_art_blob.height}, {overall_art_blob.size} bytes)"
                )
            else:
                chosen_artwork_source["reason"] = "No artwork candidates available"
                artwork_status = "none found"
                artwork_evidence.append("No readable artwork candidates were discovered.")

            if winner_multi_release and single_art_blob and single_art_track:
                if not winner.cover_hash or winner.cover_hash != single_art_blob.hash:
                    artwork_status = "apply"
                    artwork_actions.append(
                        ArtworkDirective(
                            source=single_art_track.path,
                            target=winner.path,
                            reason="Migrate single-release artwork onto multi-track album winner",
                        )
                    )
                    artwork_evidence.append(
                        "Multi-track album winner will receive single-release artwork to preserve unique single cover."
                    )
                else:
                    artwork_status = "unchanged"
                    chosen_artwork_source["reason"] = "Winner already has selected single-release artwork"
                    artwork_evidence.append("Winner already matches selected single-release artwork hash; no copy needed.")
            elif winner_multi_release and not single_art_blob and overall_art_blob and overall_art_track:
                source_release = release_sizes.get(overall_art_track.path, "unknown")
                if source_release == "multi" and overall_art_track.path != winner.path:
                    artwork_status = "unchanged"
                    chosen_artwork_source["reason"] = "Skipped multi-track artwork migration to avoid cross-album contamination"
                    artwork_evidence.append(
                        "Skipping artwork migration between multi-track releases to avoid cross-album contamination."
                    )
                elif not winner.cover_hash or winner.cover_hash != overall_art_blob.hash:
                    artwork_status = "apply"
                    artwork_actions.append(
                        ArtworkDirective(
                            source=overall_art_track.path,
                            target=winner.path,
                            reason="Fill missing artwork from best available source",
                        )
                    )
                    artwork_evidence.append(
                        "No single-release artwork available; applying best available artwork to album winner."
                    )
                else:
                    artwork_evidence.append("Album winner artwork already matches best available source.")
            else:
                if winner_context == "single":
                    artwork_evidence.append("Single-context winner retains its own artwork; album art will not be pulled.")
                    chosen_artwork_source["reason"] = (
                        chosen_artwork_source.get("reason") or "Single winner keeps artwork"
                    )
                elif not single_art_blob and not overall_art_blob and not suppress_review_flags:
                    group_review.append("No artwork available to apply.")

            cover_hashes = {t.cover_hash for t in group_tracks if t.cover_hash}
            ambiguous_art = ambiguous_single_art or ambiguous_overall_art
            if len(cover_hashes) > 1:
                ambiguous_art = True
                artwork_evidence.append("Conflicting embedded artwork hashes between candidates.")
            if ambiguous_art and not suppress_review_flags:
                review_flags.append(
                    f"Artwork selection ambiguous for group {_stable_group_id([t.path for t in group_tracks])}."
                )
            if not (single_art_blob or overall_art_blob) and not suppress_review_flags:
                missing_reason = "No artwork candidates available"
                if any(t.artwork_error for t in group_tracks):
                    missing_reason = "Artwork present but unreadable"
                review_flags.append(f"{missing_reason} for group {_stable_group_id([t.path for t in group_tracks])}.")
                chosen_artwork_source["reason"] = missing_reason
                artwork_status = "none found"
                group_review.append(missing_reason)

            required_tag_gaps = [key for key in ("artist", "title") if not planned_tags.get(key)]
            if required_tag_gaps and not suppress_review_flags:
                group_review.append(f"Missing critical tags: {', '.join(sorted(required_tag_gaps))}.")

            group_confidence = "High (identical fingerprint cluster)"
            if missing_fingerprints and not suppress_review_flags:
                group_confidence = "Low (missing fingerprints in cluster)"
                group_review.append("One or more tracks missing fingerprints; requires review.")
            near_distances = [d for d in distance_samples if d > exact_duplicate_threshold]
            if near_distances and not suppress_review_flags:
                group_confidence = f"Medium (near-duplicate fingerprint distance up to {max_distance:.3f})"
                group_review.append(
                    f"Audio match requires review (max distance {max_distance:.3f}; near-duplicate threshold {near_duplicate_threshold:.3f})."
                )
            if max_distance > near_duplicate_threshold and not suppress_review_flags:
                group_confidence = f"Low (max fingerprint distance {max_distance:.3f})"
                if not missing_fingerprints:
                    group_review.append(
                        f"Fingerprint distances exceed near-duplicate threshold ({near_duplicate_threshold:.3f}); grouping may be unsafe."
                    )

            dispositions = {loser: "quarantine" for loser in losers}
            playlist_map = {loser: winner.path for loser in losers}

            group_id = _stable_group_id([t.path for t in group_tracks])
            if ambiguous_art and not suppress_review_flags:
                group_review.append("Artwork selection requires review.")
            if not losers and not suppress_review_flags:
                group_review.append("No losers to consolidate.")
            if any(_has_review_keyword(t) for t in group_tracks) and not suppress_review_flags:
                group_review.append("Contains remix/sped-up variant indicators; review recommended.")
            placeholders = _placeholder_present(planned_tags) or any(
                _placeholder_present(t.current_tags) for t in group_tracks
            )
            if placeholders and not suppress_review_flags:
                group_review.append("Placeholder metadata detected; requires review.")
                plan_placeholders = True
            if winner.metadata_error and not suppress_review_flags:
                group_review.append(f"Metadata read issue for winner: {winner.metadata_error}")
            if any(t.artwork_error for t in group_tracks) and not suppress_review_flags:
                group_review.append("Artwork extraction failed for at least one track.")

            if suppress_review_flags:
                group_review = []
                placeholders = False

            playlist_impact = PlaylistImpact(playlists=len(losers), entries=len(losers))
            track_states = {t.path: dict(t.library_state) for t in group_tracks}

            artwork_candidates: List[ArtworkCandidate] = []
            for t in group_tracks:
                artwork_candidates.extend(t.artwork)

            current_tags = {t.path: _merge_tags(_blank_tags(), t.current_tags) for t in group_tracks}

            plans.append(
                GroupPlan(
                    group_id=group_id,
                    winner_path=winner.path,
                    losers=losers,
                    planned_winner_tags=planned_tags,
                    winner_current_tags=_merge_tags(_blank_tags(), winner.current_tags),
                    current_tags=current_tags,
                    metadata_changes=meta_changes,
                    winner_quality=winner_quality,
                    artwork=artwork_actions,
                    artwork_candidates=artwork_candidates,
                    chosen_artwork_source=chosen_artwork_source,
                    artwork_status=artwork_status,
                    loser_disposition=dispositions,
                    playlist_rewrites=playlist_map,
                    playlist_impact=playlist_impact,
                    review_flags=group_review,
                    context_summary={
                        "album": sorted([p for p in group_paths if group_contexts.get(p) == "album"]),
                        "single": sorted([p for p in group_paths if group_contexts.get(p) == "single"]),
                        "unknown": sorted([p for p in group_paths if group_contexts.get(p) == "unknown"]),
                    },
                    context_evidence=group_context_evidence,
                    tag_source=tag_source,
                    placeholders_present=placeholders,
                    tag_source_reason=tag_source_reason,
                    tag_source_evidence=tag_source_evidence,
                    track_quality=track_quality,
                    group_confidence=group_confidence,
                    group_match_type="Exact" if max_distance <= exact_duplicate_threshold else "Near-duplicate",
                    grouping_metadata_key=cluster.metadata_key,
                    grouping_thresholds={
                        "exact": cluster.exact_threshold,
                        "near": cluster.near_threshold,
                        "mixed_codec_boost": mixed_codec_threshold_boost,
                    },
                    grouping_decisions=group_decisions,
                    artwork_evidence=artwork_evidence,
                    fingerprint_distances=pair_distances,
                    library_state=track_states,
                )
            )

    if plan_placeholders and "Placeholder metadata detected; review required." not in review_flags:
        review_flags.append("Placeholder metadata detected; review required.")

    return ConsolidationPlan(
        groups=plans,
        review_flags=review_flags,
        placeholders_present=plan_placeholders,
        fingerprint_settings=dict(fingerprint_settings or {}),
        threshold_settings=dict(threshold_settings or {}),
    )


def export_consolidation_preview(plan: ConsolidationPlan, output_json_path: str) -> str:
    """Write a JSON audit of the consolidation plan."""

    plan.refresh_plan_signature()
    summary = {
        "groups": len(plan.groups),
        "review_required": plan.review_required_count,
        "review_flags": plan.review_flags,
    }
    payload = {
        "generated_at": plan.generated_at.isoformat(),
        "summary": summary,
        "plan": plan.to_dict(),
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_json_path


def export_duplicate_pair_report_html(report: DuplicatePairReport, output_html_path: str) -> str:
    """Write an HTML report for a duplicate-finder pair inspection."""

    def esc(value: object) -> str:
        return html.escape(str(value))

    def _format_setting(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _track_row(track: DuplicateTrack, label: str) -> List[str]:
        return [
            "<tr>",
            f"<th>{esc(label)}</th>",
            "<td>",
            f"<div class='path'>{esc(track.path)}</div>",
            "<div class='meta'>"
            f"Codec: {esc(track.codec or track.container or track.ext)} | "
            f"Bitrate: {esc(track.bitrate or 'n/a')} | "
            f"Sample rate: {esc(track.sample_rate or 'n/a')} | "
            f"Channels: {esc(track.channels or 'n/a')} | "
            f"Bit depth: {esc(track.bit_depth or 'n/a')}",
            "</div>",
            "</td>",
            "</tr>",
        ]

    fingerprint_settings = report.fingerprint_settings or {}
    threshold_settings = report.threshold_settings or {}

    html_lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8' />",
        "<title>Duplicate Finder Pair Report</title>",
        "<style>",
        "body{font-family:Arial, sans-serif; margin:24px; color:#222;}",
        "h1{font-size:20px; margin-bottom:6px;}",
        "h2{font-size:16px; margin-top:24px;}",
        "table{border-collapse:collapse; width:100%; margin-top:8px;}",
        "th,td{border:1px solid #ddd; padding:8px; text-align:left; vertical-align:top;}",
        "th{background:#f4f4f4; width:180px;}",
        ".badge{display:inline-block; padding:2px 8px; border-radius:12px; font-size:12px; margin-left:6px;}",
        ".ok{background:#d4f4dd; color:#116329;}",
        ".fail{background:#ffe0e0; color:#8a1f1f;}",
        ".blocked{background:#fff1c1; color:#7a5c00;}",
        ".path{font-family:monospace; word-break:break-all;}",
        ".meta{color:#555; font-size:12px; margin-top:4px;}",
        ".muted{color:#666; font-size:12px;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Duplicate Finder Pair Report</h1>",
        f"<div class='muted'>Generated: {esc(report.generated_at.isoformat())}</div>",
        "<h2>Verdict</h2>",
        "<table>",
        "<tr><th>Verdict</th><td>",
        esc(report.verdict),
        "</td></tr>",
        "<tr><th>Match type</th><td>",
        esc(report.match_type),
        "</td></tr>",
        "<tr><th>Distance</th><td>",
        esc(f\"{report.fingerprint_distance:.4f}\" if report.fingerprint_distance is not None else \"n/a\"),
        "</td></tr>",
        "<tr><th>Effective threshold</th><td>",
        esc(f\"{report.effective_threshold:.4f}\"),
        "</td></tr>",
        "</table>",
        "<h2>Tracks</h2>",
        "<table>",
        *_track_row(report.track_a, "Song A"),
        *_track_row(report.track_b, "Song B"),
        "</table>",
        "<h2>Gate Checks</h2>",
        "<table>",
        "<tr><th>Step</th><th>Status</th><th>Detail</th></tr>",
    ]

    status_class = {
        "pass": "ok",
        "fail": "fail",
        "blocked": "blocked",
    }
    for step in report.steps:
        css = status_class.get(step.status, "blocked")
        html_lines.extend(
            [
                "<tr>",
                f"<td>{esc(step.name)}</td>",
                f\"<td><span class='badge {css}'>{esc(step.status)}</span></td>\",
                f"<td>{esc(step.detail)}</td>",
                "</tr>",
            ]
        )

    html_lines.extend(
        [
            "</table>",
            "<h2>Thresholds</h2>",
            "<table>",
            f\"<tr><th>Exact threshold</th><td>{esc(f'{report.exact_threshold:.4f}')}</td></tr>\",
            f\"<tr><th>Near threshold</th><td>{esc(f'{report.near_threshold:.4f}')}</td></tr>\",
            f\"<tr><th>Mixed-codec boost</th><td>{esc(f'{report.mixed_codec_boost:.4f}')}</td></tr>\",
            f\"<tr><th>Mixed-codec applied</th><td>{esc('Yes' if report.mixed_codec else 'No')}</td></tr>\",
            "</table>",
        ]
    )

    if threshold_settings:
        html_lines.extend(
            [
                "<h2>Threshold Settings Snapshot</h2>",
                "<table>",
                "<tr><th>Setting</th><th>Value</th></tr>",
            ]
        )
        for key, value in sorted(threshold_settings.items(), key=lambda item: str(item[0])):
            html_lines.append(f"<tr><td>{esc(key)}</td><td>{esc(_format_setting(value))}</td></tr>")
        html_lines.append("</table>")

    if fingerprint_settings:
        html_lines.extend(
            [
                "<h2>Fingerprint Settings Snapshot</h2>",
                "<table>",
                "<tr><th>Setting</th><th>Value</th></tr>",
            ]
        )
        for key, value in sorted(fingerprint_settings.items(), key=lambda item: str(item[0])):
            html_lines.append(f"<tr><td>{esc(key)}</td><td>{esc(_format_setting(value))}</td></tr>")
        html_lines.append("</table>")

    html_lines.extend(["</body>", "</html>"])

    with open(output_html_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(html_lines))
    return output_html_path


def export_consolidation_preview_html(plan: ConsolidationPlan, output_html_path: str) -> str:
    """Write an HTML preview of the consolidation plan."""

    plan.refresh_plan_signature()
    def esc(value: object) -> str:
        return html.escape(str(value))

    def _basename(value: str) -> str:
        return os.path.basename(value) or value

    def _status_badge_class(status: str) -> str:
        if status in {"success", "ready"}:
            return "ok"
        if status in {"failed", "blocked", "cancelled"}:
            return "fail"
        return "warn"

    def _format_metadata_notes(meta: Dict[str, object]) -> str:
        notes = []
        for key, value in meta.items():
            if key in {"group_id", "group_ids"}:
                continue
            if isinstance(value, (list, tuple, set)):
                rendered = ", ".join(str(v) for v in value)
            else:
                rendered = str(value)
            notes.append(f"{key}: {rendered}")
        return " | ".join(notes)

    def _format_notes(detail: str, meta_notes: str) -> str:
        detail_html = html.escape(detail)
        if meta_notes:
            meta_html = html.escape(meta_notes)
            return f"{detail_html}<div class='tiny muted'>{meta_html}</div>"
        return detail_html

    def _planned_actions(group: GroupPlan) -> List[Dict[str, object]]:
        actions: List[Dict[str, object]] = []
        for path, changes in group.metadata_changes.items():
            fields = sorted(changes.keys())
            actions.append(
                {
                    "step": "metadata",
                    "target": path,
                    "status": "review",
                    "detail": "Planned metadata updates.",
                    "metadata": {"fields": fields},
                }
            )
        for directive in group.artwork:
            actions.append(
                {
                    "step": "artwork",
                    "target": directive.target,
                    "status": "review",
                    "detail": f"Migrate artwork from {directive.source} to {directive.target}.",
                    "metadata": {"source": directive.source, "reason": directive.reason},
                }
            )
        for loser in group.losers:
            disposition = group.loser_disposition.get(loser, "quarantine")
            actions.append(
                {
                    "step": "loser_cleanup",
                    "target": loser,
                    "status": "review",
                    "detail": f"Loser marked for {disposition}.",
                    "metadata": {"disposition": disposition},
                }
            )
        for playlist, destination in group.playlist_rewrites.items():
            actions.append(
                {
                    "step": "playlist",
                    "target": playlist,
                    "status": "review",
                    "detail": "Planned playlist rewrite.",
                    "metadata": {"destination": destination},
                }
            )
        return actions

    def _missing_artwork_gate(group: GroupPlan) -> bool:
        for note in group.artwork_evidence:
            lowered = note.lower()
            if "non-matching" in lowered and "artwork" in lowered and "missing" in lowered:
                return True
        return False

    def _action_badges(group: GroupPlan, actions: List[Dict[str, object]]) -> List[str]:
        badges = []
        meta_count = sum(1 for act in actions if act["step"] == "metadata")
        art_count = sum(1 for act in actions if act["step"] == "artwork")
        playlist_count = sum(1 for act in actions if act["step"] == "playlist")
        loser_count = len(group.losers)
        if meta_count:
            badges.append(f"metadata updates ({meta_count})")
        if art_count:
            badges.append(f"artwork copies ({art_count})")
        if playlist_count:
            badges.append(f"playlist rewrites ({playlist_count})")
        if loser_count:
            badges.append(f"losers ({loser_count})")
        return badges

    def _group_type_hint(actions: List[Dict[str, object]]) -> str:
        if not actions:
            return "unknown"
        if all(act["step"] == "metadata" for act in actions):
            return "metadata-only"
        return "mixed"

    def _group_disposition_count(group: GroupPlan) -> int:
        return sum(
            1
            for loser in group.losers
            if group.loser_disposition.get(loser, "quarantine") != "retain"
        )

    def _album_art_src(
        group: GroupPlan,
        track_path: str,
        include_group_chosen: bool = True,
    ) -> str | None:
        chosen_hash = ""
        if include_group_chosen and isinstance(group.chosen_artwork_source, Mapping):
            raw_hash = group.chosen_artwork_source.get("hash")
            if raw_hash:
                chosen_hash = str(raw_hash)
        fallback: ArtworkCandidate | None = None
        path_fallback: ArtworkCandidate | None = None
        for candidate in group.artwork_candidates:
            if not candidate.bytes:
                continue
            if chosen_hash and candidate.hash == chosen_hash:
                return _image_data_uri(candidate.bytes)
            if track_path and candidate.path == track_path and path_fallback is None:
                path_fallback = candidate
            if fallback is None:
                fallback = candidate
        if path_fallback and path_fallback.bytes:
            return _image_data_uri(path_fallback.bytes)
        if fallback and fallback.bytes:
            return _image_data_uri(fallback.bytes)
        return None

    def _image_data_uri(payload: bytes) -> str:
        mime = _image_mime(payload)
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _image_mime(payload: bytes) -> str:
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if payload.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    generated_at_iso = plan.generated_at.isoformat()
    host_name = platform.node() or "unknown-host"
    reports_dir = os.path.dirname(output_html_path)
    plan_signature = plan.plan_signature or "not-generated"
    global_notes = "Groups are collapsed by default. Full paths and verbose messages are in “Context”."
    if plan.review_flags:
        joined_flags = "; ".join(plan.review_flags)
        global_notes = f"{global_notes} Review flags: {joined_flags}"

    settings_entries: List[tuple[str, object]] = []
    if plan.threshold_settings:
        settings_entries.extend(sorted(plan.threshold_settings.items(), key=lambda item: str(item[0])))
    if plan.fingerprint_settings:
        settings_entries.extend(sorted(plan.fingerprint_settings.items(), key=lambda item: str(item[0])))

    def _format_setting(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    html_lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8' />",
        "<meta name='viewport' content='width=device-width,initial-scale=1' />",
        "<title>Execution Report Preview</title>",
        "<style>",
        ":root{",
        "--bg:#ffffff;",
        "--text:#111;",
        "--muted:#666;",
        "--border:#e6e6e6;",
        "--card:#fafafa;",
        "--good:#1b7a1b;",
        "--bad:#a30000;",
        "--warn:#8a5b00;",
        "--chip:#f2f2f2;",
        "--mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace;",
        "--sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, \"Apple Color Emoji\",\"Segoe UI Emoji\";",
        "}",
        "* { box-sizing: border-box; }",
        "body{",
        "margin: 18px;",
        "font-family: var(--sans);",
        "background: var(--bg);",
        "color: var(--text);",
        "line-height: 1.35;",
        "}",
        "header{",
        "display:flex;",
        "flex-wrap:wrap;",
        "gap:12px 16px;",
        "align-items:baseline;",
        "justify-content:space-between;",
        "margin-bottom: 12px;",
        "}",
        "h1{",
        "font-size: 20px;",
        "margin: 0;",
        "letter-spacing: .2px;",
        "}",
        ".row{",
        "display:flex;",
        "flex-wrap:wrap;",
        "gap:10px;",
        "align-items:center;",
        "}",
        ".card{",
        "border:1px solid var(--border);",
        "border-radius: 10px;",
        "padding: 12px;",
        "background: var(--card);",
        "}",
        ".context-card{",
        "display:flex;",
        "gap:12px;",
        "align-items:flex-start;",
        "}",
        ".context-card-art{",
        "flex: 0 0 auto;",
        "}",
        ".context-card-details{",
        "flex: 1;",
        "}",
        ".card-stack{",
        "display:grid;",
        "gap:10px;",
        "}",
        ".summary{",
        "display:grid;",
        "grid-template-columns: 1fr;",
        "gap: 10px;",
        "margin-bottom: 14px;",
        "}",
        "@media (min-width: 900px){",
        ".summary{ grid-template-columns: 2fr 1fr; }",
        "}",
        ".kv{",
        "display:grid;",
        "grid-template-columns: 160px 1fr;",
        "gap:6px 10px;",
        "align-items:start;",
        "}",
        ".k{ color: var(--muted); font-size: 12px; }",
        ".v{ font-size: 13px; }",
        ".mono{ font-family: var(--mono); font-size: 12px; }",
        ".path{",
        "font-family: var(--mono);",
        "font-size: 12px;",
        "word-break: break-all;",
        "}",
        ".status{",
        "font-weight: 600;",
        "display:inline-flex;",
        "align-items:center;",
        "gap:6px;",
        "}",
        ".ok{ color: var(--good); }",
        ".fail{ color: var(--bad); }",
        ".warn{ color: var(--warn); }",
        ".pill{",
        "display:inline-flex;",
        "align-items:center;",
        "gap:8px;",
        "padding: 6px 10px;",
        "border-radius: 999px;",
        "background: var(--chip);",
        "border: 1px solid var(--border);",
        "font-size: 12px;",
        "white-space: nowrap;",
        "}",
        ".pill strong{ font-weight: 700; }",
        ".controls{",
        "display:flex;",
        "flex-wrap:wrap;",
        "gap:10px;",
        "align-items:center;",
        "justify-content:space-between;",
        "margin: 14px 0 10px;",
        "}",
        ".controls .left, .controls .right{",
        "display:flex; flex-wrap:wrap; gap:10px; align-items:center;",
        "}",
        "input[type='search']{",
        "padding: 9px 10px;",
        "border:1px solid var(--border);",
        "border-radius: 10px;",
        "width: min(520px, 92vw);",
        "font-size: 13px;",
        "background: #fff;",
        "}",
        "select{",
        "padding: 9px 10px;",
        "border:1px solid var(--border);",
        "border-radius: 10px;",
        "font-size: 13px;",
        "background: #fff;",
        "}",
        "button.copy{",
        "border:1px solid var(--border);",
        "background:#fff;",
        "border-radius: 10px;",
        "padding: 8px 10px;",
        "font-size: 12px;",
        "cursor:pointer;",
        "}",
        "button.copy:active{ transform: translateY(1px); }",
        ".settings-panel{",
        "margin-top: 10px;",
        "}",
        "details.group{",
        "border:1px solid var(--border);",
        "border-radius: 10px;",
        "background: #fff;",
        "margin-bottom: 10px;",
        "overflow:hidden;",
        "}",
        "details.group[open]{ box-shadow: 0 1px 0 rgba(0,0,0,.03); }",
        "summary.group-summary{",
        "list-style:none;",
        "cursor:pointer;",
        "padding: 12px 12px;",
        "display:grid;",
        "grid-template-columns: auto 1fr;",
        "grid-template-rows: auto auto;",
        "column-gap: 12px;",
        "row-gap: 6px;",
        "background: #fff;",
        "}",
        "summary.group-summary::-webkit-details-marker{ display:none; }",
        ".album-art{",
        "width: 42px;",
        "height: 42px;",
        "display: inline-block;",
        "grid-column: 1;",
        "grid-row: 1 / span 2;",
        "align-self: center;",
        "border-radius: 8px;",
        "border: 1px solid var(--border);",
        "position: relative;",
        "overflow: hidden;",
        "background: #f7f7f7;",
        "}",
        ".album-art.album-art-thumb{",
        "width: 64px;",
        "height: 64px;",
        "border-radius: 6px;",
        "}",
        ".album-art::after{",
        "content: \"\";",
        "position: absolute;",
        "inset: 0;",
        "pointer-events: none;",
        "box-shadow: inset 0 0 0 1px rgba(0,0,0,.06), inset 0 0 10px rgba(0,0,0,.20);",
        "}",
        ".album-art img{",
        "width: 100%;",
        "height: 100%;",
        "object-fit: cover;",
        "display: block;",
        "}",
        ".group-top{",
        "grid-column: 2;",
        "grid-row: 1;",
        "display:flex;",
        "flex-wrap:wrap;",
        "gap:10px 12px;",
        "align-items:center;",
        "justify-content:space-between;",
        "}",
        "summary.group-summary .tiny{",
        "grid-column: 2;",
        "grid-row: 2;",
        "}",
        ".group-title{",
        "display:flex;",
        "align-items:baseline;",
        "gap:10px;",
        "min-width: 240px;",
        "}",
        ".gid{",
        "font-family: var(--mono);",
        "font-size: 12px;",
        "color: var(--muted);",
        "}",
        ".winner-short{",
        "font-weight: 650;",
        "font-size: 13px;",
        "max-width: 74ch;",
        "overflow:hidden;",
        "text-overflow: ellipsis;",
        "white-space: nowrap;",
        "}",
        ".chips{",
        "display:flex;",
        "flex-wrap:wrap;",
        "gap:8px;",
        "justify-content:flex-end;",
        "}",
        ".group-body{",
        "border-top: 1px solid var(--border);",
        "background: var(--card);",
        "padding: 12px;",
        "display:grid;",
        "grid-template-columns: 1fr;",
        "gap: 10px;",
        "}",
        "@media (min-width: 980px){",
        ".group-body{ grid-template-columns: 1fr 1.2fr; }",
        "}",
        ".section h3{",
        "font-size: 12px;",
        "margin: 0 0 8px;",
        "color: var(--muted);",
        "text-transform: uppercase;",
        "letter-spacing: .08em;",
        "}",
        "table.ops{",
        "width:100%;",
        "border-collapse: collapse;",
        "background:#fff;",
        "border:1px solid var(--border);",
        "border-radius: 10px;",
        "overflow:hidden;",
        "}",
        "table.ops th, table.ops td{",
        "border-bottom:1px solid var(--border);",
        "padding: 9px 10px;",
        "font-size: 12px;",
        "vertical-align: top;",
        "}",
        "table.ops th{",
        "text-align:left;",
        "color: var(--muted);",
        "background: #fcfcfc;",
        "font-weight: 650;",
        "}",
        "table.ops tr:last-child td{ border-bottom: none; }",
        ".op-key{",
        "font-family: var(--mono);",
        "font-size: 12px;",
        "color:#222;",
        "white-space:nowrap;",
        "}",
        ".note{",
        "color: var(--muted);",
        "}",
        ".badge{",
        "display:inline-flex;",
        "align-items:center;",
        "padding: 3px 8px;",
        "border-radius: 999px;",
        "border:1px solid var(--border);",
        "font-size: 12px;",
        "font-weight: 650;",
        "background: #fff;",
        "white-space: nowrap;",
        "}",
        ".badge.ok{ border-color: rgba(27,122,27,.25); }",
        ".badge.fail{ border-color: rgba(163,0,0,.25); }",
        ".badge.warn{ border-color: rgba(138,91,0,.35); }",
        "details.quarantine{",
        "border:1px solid var(--border);",
        "border-radius: 10px;",
        "background:#fff;",
        "margin-top: 14px;",
        "overflow:hidden;",
        "}",
        "summary.quarantine-summary{",
        "cursor:pointer;",
        "padding: 12px;",
        "list-style:none;",
        "}",
        "summary.quarantine-summary::-webkit-details-marker{ display:none; }",
        "ul.q-list{",
        "margin:0;",
        "padding: 0 12px 12px 28px;",
        "color:#222;",
        "}",
        "ul.q-list li{",
        "font-family: var(--mono);",
        "font-size: 12px;",
        "word-break: break-all;",
        "margin: 6px 0;",
        "}",
        ".muted{ color: var(--muted); }",
        ".tiny{ font-size: 12px; }",
        ".hidden{ display:none !important; }",
        "</style>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>Execution Report Preview</h1>",
        "<div class='status warn' id='runStatus' data-status='review'>📝 review</div>",
        "</header>",
        "<section class='summary'>",
        "<div class='card'>",
        "<div class='row' style='justify-content:space-between;'>",
        "<div class='row' style='gap:8px;'>",
        "<span class='pill'><strong>Groups</strong> <span id='statGroups'>0</span></span>",
        "<span class='pill'><strong>Winners</strong> <span id='statWinners'>0</span></span>",
        "<span class='pill'><strong>Quarantined</strong> <span id='statQuarantined'>0</span></span>",
        "<span class='pill'><strong>Ops</strong> <span id='statOpsOk'>0</span>/<span id='statOpsTotal'>0</span> ok</span>",
        "</div>",
        "<div class='row'>",
        "<button class='copy' data-copy='#planSignature'>Copy plan signature</button>",
        "<button class='copy' data-copy='#reportsDir'>Copy reports dir</button>",
        "</div>",
        "</div>",
        "<div style='height:10px'></div>",
        "<div class='kv'>",
        "<div class='k'>Plan signature</div>",
        f"<div class='v mono' id='planSignature'>{esc(plan_signature)}</div>",
        "<div class='k'>Reports directory</div>",
        f"<div class='v path' id='reportsDir'>{esc(reports_dir)}</div>",
        "</div>",
        "</div>",
        "<div class='card'>",
        "<div class='kv'>",
        "<div class='k'>Generated</div>",
        f"<div class='v tiny'>{esc(generated_at_iso)}</div>",
        "<div class='k'>Host</div>",
        f"<div class='v tiny'>{esc(host_name)}</div>",
        "<div class='k'>Notes</div>",
        f"<div class='v tiny muted'>{esc(global_notes)}</div>",
        "</div>",
        "</div>",
    ]

    if settings_entries:
        insert_at = html_lines.index(
            "<button class='copy' data-copy='#reportsDir'>Copy reports dir</button>"
        ) + 1
        html_lines.insert(insert_at, "<button class='copy' id='toggleSettings'>⚙️ Thresholds</button>")
        html_lines.extend(
            [
                "<div class='card hidden settings-panel' id='settingsPanel'>",
                "<div class='kv'>",
                "<div class='k'>Threshold settings</div>",
                "<div class='v tiny muted'>Thresholds + normalization inputs used for this plan.</div>",
            ]
        )
        for key, value in settings_entries:
            html_lines.append(f"<div class='k'>{esc(str(key))}</div>")
            html_lines.append(f"<div class='v tiny'>{esc(_format_setting(value))}</div>")
        html_lines.extend(["</div>", "</div>"])

    html_lines.extend(
        [
            "</section>",
            "<section class='controls'>",
            "<div class='left'>",
            "<input id='search' type='search' placeholder='Search groups, winner filename, paths, notes…' />",
            "<select id='filter'>",
            "<option value='all'>All groups</option>",
            "<option value='has-quarantine'>Has quarantine</option>",
            "<option value='metadata-only'>Metadata only</option>",
            "<option value='failed'>Any failures</option>",
            "</select>",
            "</div>",
            "<div class='right'>",
            "<span class='tiny muted' id='visibleCount'>0 visible</span>",
            "<button class='copy' id='expandAll'>Expand all</button>",
            "<button class='copy' id='collapseAll'>Collapse all</button>",
            "</div>",
            "</section>",
            "<main id='groups'>",
        ]
    )

    all_actions: List[Dict[str, object]] = []
    for group in plan.groups:
        actions = _planned_actions(group)
        all_actions.extend(actions)
        state = "review" if group.review_flags else "ready"
        badges = _action_badges(group, actions)
        group_disposition_count = _group_disposition_count(group)
        group_type_hint = _group_type_hint(actions)
        search_tokens = [group.group_id, group.winner_path, _basename(group.winner_path), state]
        search_tokens.extend(badges)
        for act in actions:
            search_tokens.extend([act["step"], act["target"], act["status"], act["detail"]])
            meta_notes = _format_metadata_notes(act["metadata"])
            if meta_notes:
                search_tokens.append(meta_notes)
        search_text = " ".join(str(token) for token in search_tokens if token)
        summary_line = (
            f"Status: {state}. " + "; ".join(badges) if badges else f"Status: {state}."
        )
        html_lines.append(
            "<details class='group' "
            f"data-group-id='{esc(group.group_id)}' "
            f"data-has-quarantine='{str(group_disposition_count > 0).lower()}' "
            "data-has-failure='false' "
            f"data-type='{esc(group_type_hint)}' "
            f"data-search='{esc(search_text.lower())}' "
            f"data-quarantine-count='{group_disposition_count}'>"
        )
        album_art_src = _album_art_src(group, group.winner_path)
        html_lines.append("<summary class='group-summary'>")
        html_lines.append("<span class='album-art' title='Album Art'>")
        if album_art_src:
            html_lines.append(f"<img src='{album_art_src}' alt='' />")
        html_lines.append("</span>")
        html_lines.append("<div class='group-top'>")
        html_lines.append("<div class='group-title'>")
        html_lines.append(f"<span class='gid'>Group {esc(group.group_id)}</span>")
        html_lines.append(
            "<span class='winner-short' "
            f"title='{esc(group.winner_path)}'>{esc(_basename(group.winner_path))}</span>"
        )
        html_lines.append("</div>")
        html_lines.append("<div class='chips'>")
        html_lines.append(
            f"<span class='badge {_status_badge_class(state)}' data-status='{esc(state)}'>"
            f"{esc(state)}</span>"
        )
        for badge in badges:
            html_lines.append(f"<span class='pill'>{esc(badge)}</span>")
        html_lines.append("</div>")
        html_lines.append("</div>")
        html_lines.append(f"<div class='tiny muted'>{esc(summary_line)}</div>")
        html_lines.append("</summary>")
        html_lines.append("<div class='group-body'>")
        html_lines.append("<div class='section'>")
        html_lines.append("<h3>Context</h3>")
        loser_reason_map = {
            act["target"]: str(act["detail"])
            for act in actions
            if act["step"] == "loser_cleanup"
        }
        loser_paths = group.losers or [""]
        winner_art_src = _album_art_src(group, group.winner_path)
        html_lines.append("<div class='card-stack'>")
        html_lines.append("<div class='card context-card' style='background:#fff;'>")
        html_lines.append("<div class='context-card-art'>")
        html_lines.append("<span class='album-art album-art-thumb' title='Album Art'>")
        if winner_art_src:
            html_lines.append(f"<img src='{winner_art_src}' alt='' />")
        html_lines.append("</span>")
        html_lines.append("</div>")
        html_lines.append("<div class='kv context-card-details'>")
        html_lines.append("<div class='k'>Kept file (winner)</div>")
        html_lines.append(f"<div class='v path'>{esc(group.winner_path)}</div>")
        html_lines.append("<div class='k'>Group ID</div>")
        html_lines.append(f"<div class='v mono'>{esc(group.group_id)}</div>")
        html_lines.append("<div class='k'>Debug</div>")
        html_lines.append(
            "<div class='v tiny muted'>Shown for traceability; safe to ignore for normal review.</div>"
        )
        html_lines.append("</div>")
        html_lines.append("</div>")
        for loser_path in loser_paths:
            loser_disposition = (
                group.loser_disposition.get(loser_path, "quarantine") if loser_path else ""
            )
            loser_reason = loser_reason_map.get(loser_path, "")
            loser_art_src = (
                _album_art_src(group, loser_path, include_group_chosen=False)
                if loser_path
                else None
            )
            html_lines.append("<div class='card context-card' style='background:#fff;'>")
            html_lines.append("<div class='context-card-art'>")
            html_lines.append("<span class='album-art album-art-thumb' title='Album Art'>")
            if loser_art_src:
                html_lines.append(f"<img src='{loser_art_src}' alt='' />")
            html_lines.append("</span>")
            html_lines.append("</div>")
            html_lines.append("<div class='kv context-card-details'>")
            html_lines.append("<div class='k'>Loser file</div>")
            if loser_path:
                html_lines.append(f"<div class='v path'>{esc(loser_path)}</div>")
            else:
                html_lines.append("<div class='v path'>—</div>")
            html_lines.append("<div class='k'>Disposition</div>")
            if loser_disposition:
                html_lines.append(f"<div class='v'>{esc(loser_disposition)}</div>")
            else:
                html_lines.append("<div class='v'>—</div>")
            if loser_reason:
                html_lines.append("<div class='k'>Reason</div>")
                html_lines.append(f"<div class='v tiny muted'>{esc(loser_reason)}</div>")
            html_lines.append("</div>")
            html_lines.append("</div>")
        html_lines.append("</div>")
        html_lines.append("</div>")
        html_lines.append("<div class='section'>")
        html_lines.append("<h3>Operations</h3>")
        html_lines.append("<table class='ops'>")
        html_lines.append("<thead>")
        html_lines.append("<tr>")
        html_lines.append("<th style='width: 160px;'>Operation</th>")
        html_lines.append("<th>Target</th>")
        html_lines.append("<th style='width: 110px;'>Status</th>")
        html_lines.append("<th>Notes</th>")
        html_lines.append("</tr>")
        html_lines.append("</thead>")
        html_lines.append("<tbody>")
        if actions:
            for act in actions:
                meta_notes = _format_metadata_notes(act["metadata"])
                notes = _format_notes(str(act["detail"]), meta_notes)
                status_class = _status_badge_class(str(act["status"]))
                html_lines.append(
                    "<tr>"
                    f"<td class='op-key'>{esc(act['step'])}</td>"
                    f"<td class='path' title='{esc(act['target'])}'>"
                    f"{esc(_basename(str(act['target'])))}</td>"
                    f"<td><span class='badge {status_class}'>{esc(act['status'])}</span></td>"
                    f"<td class='note'>{notes}</td>"
                    "</tr>"
                )
        else:
            no_action_note = "No planned operations for this group."
            no_action_status = "review"
            if _missing_artwork_gate(group):
                no_action_note = (
                    "No planned operations; missing or unreadable artwork was treated as non-matching."
                )
                no_action_status = "ready"
            status_class = _status_badge_class(no_action_status)
            html_lines.append(
                "<tr><td class='op-key'>none</td><td class='path'>—</td>"
                f"<td><span class='badge {status_class}'>{esc(no_action_status)}</span></td>"
                f"<td class='note'>{esc(no_action_note)}</td></tr>"
            )
        html_lines.append("</tbody>")
        html_lines.append("</table>")
        html_lines.append("</div>")
        html_lines.append("</div>")
        html_lines.append("</details>")

    html_lines.append("</main>")
    html_lines.append("<details class='quarantine' id='quarantineIndex'>")
    html_lines.append("<summary class='quarantine-summary'>")
    html_lines.append("<div class='row' style='justify-content:space-between;'>")
    html_lines.append(
        "<div><strong>Quarantined Files</strong> <span class='muted'>(<span id='qCount'>0</span>)</span></div>"
    )
    html_lines.append("<div class='muted tiny'>Collapsed by default</div>")
    html_lines.append("</div>")
    html_lines.append("</summary>")
    html_lines.append("<ul class='q-list' id='qList'>")
    for group in plan.groups:
        for loser in group.losers:
            disposition = group.loser_disposition.get(loser, "quarantine")
            if disposition == "retain":
                continue
            html_lines.append(f"<li>{esc(loser)} → {esc(disposition)}</li>")
    html_lines.append("</ul>")
    html_lines.append("</details>")

    missing_sections: List[str] = []
    metadata_planned = sum(
        len(group.metadata_changes) for group in plan.groups if group.metadata_changes
    )
    missing_sections.append(f"Metadata operations planned: {metadata_planned}")
    planned_quarantine = sum(
        1
        for group in plan.groups
        for loser in group.losers
        if group.loser_disposition.get(loser, "quarantine") != "retain"
    )
    missing_sections.append(f"Files quarantined: {planned_quarantine}")

    playlist_planned = sum(len(group.playlist_rewrites) for group in plan.groups)
    if playlist_planned:
        missing_sections.append("")
        missing_sections.append("Playlist Changes")
        missing_sections.append("<table class='ops'>")
        missing_sections.append(
            "<thead><tr><th>Playlist</th><th>Status</th><th>Details</th></tr></thead>"
        )
        missing_sections.append("<tbody>")
        for group in plan.groups:
            for playlist, destination in group.playlist_rewrites.items():
                missing_sections.append(
                    "<tr>"
                    f"<td class='path'>{esc(playlist)}</td>"
                    "<td><span class='badge warn'>review</span></td>"
                    f"<td class='note'>planned rewrite → {esc(destination)}</td>"
                    "</tr>"
                )
        missing_sections.append("</tbody></table>")

    if missing_sections:
        html_lines.append(
            "<div style='margin-top:14px;'>"
            "<strong>NEED TO INCORPORATE AT LATER DATE</strong>"
            "</div>"
        )
        for section in missing_sections:
            if not section:
                html_lines.append("<div style='height:10px'></div>")
                continue
            html_lines.append(f"<div class='tiny'>{section}</div>")
    html_lines.append(
        "<script>"
        "(function(){"
        "const $ = (s, r=document) => r.querySelector(s);"
        "const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));"
        "const groupsEl = $('#groups');"
        "const groups = () => $$('#groups details.group');"
        "const searchEl = $('#search');"
        "const filterEl = $('#filter');"
        "const visibleCountEl = $('#visibleCount');"
        "const expandAllBtn = $('#expandAll');"
        "const collapseAllBtn = $('#collapseAll');"
        "const settingsBtn = $('#toggleSettings');"
        "const settingsPanel = $('#settingsPanel');"
        "function computeStats(){"
        "const g = groups();"
        "$('#statGroups').textContent = g.length.toString();"
        "$('#statWinners').textContent = g.length.toString();"
        "let qTotal = 0;"
        "for (const d of g){"
        "const chip = d.querySelector('.pill');"
        "const attr = d.getAttribute('data-quarantine-count');"
        "if (attr) qTotal += Number(attr) || 0;"
        "}"
        "$('#statQuarantined').textContent = qTotal.toString();"
        "let opsTotal = 0, opsOk = 0;"
        "for (const d of g){"
        "const rows = $$('table.ops tbody tr', d);"
        "opsTotal += rows.length;"
        "for (const r of rows){"
        "const badge = r.querySelector('.badge');"
        "if (badge){"
        "const status = badge.textContent.trim().toLowerCase();"
        "if (status === 'success' || status === 'review' || status === 'ready') opsOk += 1;"
        "}"
        "}"
        "}"
        "$('#statOpsTotal').textContent = opsTotal.toString();"
        "$('#statOpsOk').textContent = opsOk.toString();"
        "const qItems = $$('#qList li').filter(li => li.textContent.trim().length > 0);"
        "$('#qCount').textContent = qItems.length.toString();"
        "}"
        "function applyFilters(){"
        "const term = (searchEl.value || '').trim().toLowerCase();"
        "const mode = filterEl.value;"
        "let visible = 0;"
        "for (const d of groups()){"
        "const hay = (d.getAttribute('data-search') || '').toLowerCase();"
        "const matchesSearch = !term || hay.includes(term);"
        "const hasQuarantine = (d.getAttribute('data-has-quarantine') || 'false') === 'true';"
        "const hasFailure = (d.getAttribute('data-has-failure') || 'false') === 'true';"
        "const typeHint = (d.getAttribute('data-type') || '').toLowerCase();"
        "let matchesFilter = true;"
        "if (mode === 'has-quarantine') matchesFilter = hasQuarantine;"
        "if (mode === 'metadata-only') matchesFilter = typeHint === 'metadata-only';"
        "if (mode === 'failed') matchesFilter = hasFailure;"
        "const show = matchesSearch && matchesFilter;"
        "d.classList.toggle('hidden', !show);"
        "if (show) visible += 1;"
        "}"
        "visibleCountEl.textContent = `${visible} visible`;"
        "}"
        "for (const btn of $$('button.copy[data-copy]')){"
        "btn.addEventListener('click', async () => {"
        "const sel = btn.getAttribute('data-copy');"
        "const el = sel ? $(sel) : null;"
        "const text = el ? el.textContent.trim() : '';"
        "if (!text) return;"
        "try{"
        "await navigator.clipboard.writeText(text);"
        "const old = btn.textContent;"
        "btn.textContent = 'Copied';"
        "setTimeout(()=>btn.textContent=old, 900);"
        "}catch(e){"
        "const ta = document.createElement('textarea');"
        "ta.value = text;"
        "document.body.appendChild(ta);"
        "ta.select();"
        "document.execCommand('copy');"
        "ta.remove();"
        "}"
        "});"
        "}"
        "expandAllBtn?.addEventListener('click', () => groups().forEach(d => d.open = true));"
        "collapseAllBtn?.addEventListener('click', () => groups().forEach(d => d.open = false));"
        "if (settingsBtn && settingsPanel){"
        "settingsBtn.addEventListener('click', () => {"
        "const hidden = settingsPanel.classList.toggle('hidden');"
        "settingsBtn.textContent = hidden ? '⚙️ Thresholds' : 'Hide thresholds';"
        "});"
        "}"
        "searchEl?.addEventListener('input', applyFilters);"
        "filterEl?.addEventListener('change', applyFilters);"
        "computeStats();"
        "applyFilters();"
        "})();"
        "</script>"
    )
    html_lines.append("</body></html>")

    with open(ensure_long_path(output_html_path), "w", encoding="utf-8") as handle:
        handle.write("\n".join(html_lines))
    return output_html_path
