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
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence
from collections import defaultdict

from config import PREVIEW_ARTWORK_MAX_DIM, PREVIEW_ARTWORK_QUALITY, load_config
from utils.path_helpers import ensure_long_path
from utils.year_normalizer import normalize_year
from utils.audio_metadata_reader import read_metadata, read_sidecar_artwork_bytes
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

logger = logging.getLogger(__name__)

LOSSLESS_EXTS = {".flac", ".wav", ".alac", ".ape", ".aiff", ".aif"}
EXACT_DUPLICATE_THRESHOLD = 0.02
NEAR_DUPLICATE_THRESHOLD = 0.10
ARTWORK_HASH_SIZE = 8
ARTWORK_SIMILARITY_THRESHOLD = 5
ARTWORK_VASTLY_DIFFERENT_THRESHOLD = 24
DEFAULT_MAX_CANDIDATES = 20_000
DEFAULT_MAX_COMPARISONS = 500_000
DEFAULT_TIMEOUT_SEC = 120.0
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
PRELOADED_TAG_FIELDS = ("title", "artist", "albumartist")
PRELOADED_AUDIO_FIELDS = ("bitrate", "sample_rate", "channels")
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
METADATA_CACHE_VERSION = 1
METADATA_CACHE_FILENAME = ".duplicate_metadata_cache.json"


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


def _normalize_provided_tags(provided_tags: Mapping[str, object]) -> Dict[str, object]:
    normalized = {key: _first_value(val) for key, val in dict(provided_tags or {}).items()}

    track_raw = _first_value(normalized.get("tracknumber") or normalized.get("track"))
    track_int = _parse_int(track_raw) if track_raw not in (None, "") else None
    if track_raw in (None, ""):
        normalized["track"] = None
        normalized["tracknumber"] = None
    else:
        normalized["track"] = track_int if track_int is not None else track_raw
        normalized["tracknumber"] = track_raw

    disc_raw = _first_value(normalized.get("discnumber") or normalized.get("disc"))
    disc_int = _parse_int(disc_raw) if disc_raw not in (None, "") else None
    if disc_raw in (None, ""):
        normalized["disc"] = None
        normalized["discnumber"] = None
    else:
        normalized["disc"] = disc_int if disc_int is not None else disc_raw
        normalized["discnumber"] = disc_raw

    date_val = _first_value(normalized.get("date") or normalized.get("year"))
    normalized["date"] = str(date_val).strip() if date_val not in (None, "") else None
    normalized["year"] = normalize_year(normalized.get("year") or normalized.get("date"))
    return normalized


def _provided_tags_complete(tags: Mapping[str, object]) -> bool:
    return bool(tags.get("title") and (tags.get("artist") or tags.get("albumartist")))


def _provided_audio_props_complete(raw: Mapping[str, object]) -> bool:
    for key in PRELOADED_AUDIO_FIELDS:
        value = raw.get(key)
        coerced = _coerce_int(value)
        if coerced is None or coerced <= 0:
            return False
    return True


def _provided_artwork_complete(raw_artwork: Sequence[object]) -> bool:
    for art in raw_artwork:
        if not isinstance(art, Mapping):
            continue
        payload = art.get("bytes")
        art_hash = art.get("hash")
        size = art.get("size")
        if isinstance(payload, (bytes, bytearray)) and payload:
            return True
        if art_hash and size:
            return True
    return False


def _metadata_payload_complete(
    tags: Mapping[str, object],
    raw: Mapping[str, object],
    raw_artwork: Sequence[object],
) -> tuple[bool, Dict[str, bool]]:
    tags_complete = _provided_tags_complete(tags)
    audio_complete = _provided_audio_props_complete(raw)
    artwork_complete = _provided_artwork_complete(raw_artwork) if raw_artwork else False
    can_skip = tags_complete and audio_complete and (not raw_artwork or artwork_complete)
    return (
        can_skip,
        {
            "tags_complete": tags_complete,
            "audio_props_complete": audio_complete,
            "artwork_complete": artwork_complete,
        },
    )


def _metadata_cache_key(path: str, state: Mapping[str, object]) -> str | None:
    if not state.get("exists"):
        return None
    size = state.get("size")
    mtime = state.get("mtime")
    if size is None or mtime is None:
        return None
    return f"{path}|{int(size)}|{int(mtime)}"


def _metadata_cache_path_for_tracks(tracks: Sequence[Mapping[str, object]]) -> str | None:
    scan_roots: List[str] = []
    for raw in tracks:
        discovery = raw.get("discovery") if isinstance(raw.get("discovery"), Mapping) else {}
        roots = discovery.get("scan_roots") if isinstance(discovery.get("scan_roots"), list) else []
        for root in roots:
            if isinstance(root, str) and root:
                scan_roots.append(root)

    for root in scan_roots:
        if os.path.isdir(root):
            docs_dir = os.path.join(root, "Docs")
            os.makedirs(docs_dir, exist_ok=True)
            return os.path.join(docs_dir, METADATA_CACHE_FILENAME)

    paths = [str(raw.get("path")) for raw in tracks if raw.get("path")]
    if not paths:
        return None
    try:
        common = os.path.commonpath(paths)
    except ValueError:
        return None
    root = common if os.path.isdir(common) else os.path.dirname(common)
    if not root:
        return None
    docs_dir = os.path.join(root, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    return os.path.join(docs_dir, METADATA_CACHE_FILENAME)


def _load_metadata_cache(cache_path: str) -> Dict[str, object]:
    if not cache_path or not os.path.exists(cache_path):
        return {"version": METADATA_CACHE_VERSION, "entries": {}}
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"version": METADATA_CACHE_VERSION, "entries": {}}
    if not isinstance(payload, Mapping):
        return {"version": METADATA_CACHE_VERSION, "entries": {}}
    if payload.get("version") != METADATA_CACHE_VERSION:
        return {"version": METADATA_CACHE_VERSION, "entries": {}}
    entries = payload.get("entries") if isinstance(payload.get("entries"), Mapping) else {}
    return {"version": METADATA_CACHE_VERSION, "entries": dict(entries)}


def _save_metadata_cache(cache_path: str, cache: Mapping[str, object]) -> None:
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)
    except OSError:
        return


def _preload_cached_metadata(
    raw: Mapping[str, object],
    cached_entry: Mapping[str, object],
) -> Dict[str, object]:
    payload = dict(raw)

    cached_tags = cached_entry.get("tags") if isinstance(cached_entry.get("tags"), Mapping) else None
    raw_tags = payload.get("tags") if isinstance(payload.get("tags"), Mapping) else {}
    normalized_raw = _normalize_provided_tags(raw_tags)
    if cached_tags and not _provided_tags_complete(normalized_raw):
        payload["tags"] = dict(cached_tags)

    cached_audio = (
        cached_entry.get("audio_props")
        if isinstance(cached_entry.get("audio_props"), Mapping)
        else None
    )
    if cached_audio and not _provided_audio_props_complete(payload):
        for key in ("bitrate", "sample_rate", "bit_depth", "channels"):
            if _coerce_int(payload.get(key)) is None or int(payload.get(key) or 0) <= 0:
                payload[key] = cached_audio.get(key)
        for key in ("codec", "container"):
            if not payload.get(key) and cached_audio.get(key):
                payload[key] = cached_audio.get(key)

    if not payload.get("ext") and cached_entry.get("ext"):
        payload["ext"] = cached_entry.get("ext")

    raw_artwork = payload.get("artwork") if isinstance(payload.get("artwork"), list) else []
    cached_artwork = (
        cached_entry.get("artwork") if isinstance(cached_entry.get("artwork"), list) else None
    )
    if cached_artwork and not raw_artwork:
        payload["artwork"] = list(cached_artwork)

    return payload


def _metadata_cache_entry(track: "DuplicateTrack") -> Dict[str, object] | None:
    key = _metadata_cache_key(track.path, track.library_state)
    if not key:
        return None
    tags_source = track.current_tags if isinstance(track.current_tags, Mapping) else track.tags
    normalized_tags = _normalize_provided_tags(tags_source or {})
    normalized_tags = _merge_tags(_blank_tags(), normalized_tags)
    audio_props = {
        "bitrate": int(track.bitrate or 0),
        "sample_rate": int(track.sample_rate or 0),
        "bit_depth": int(track.bit_depth or 0),
        "channels": int(track.channels or 0),
        "codec": str(track.codec or ""),
        "container": str(track.container or ""),
    }
    artwork_payloads = [art.to_dict() for art in track.artwork] if track.artwork else []
    return {
        "path": track.path,
        "size": track.library_state.get("size"),
        "mtime": track.library_state.get("mtime"),
        "ext": track.ext,
        "tags": normalized_tags,
        "audio_props": audio_props,
        "artwork": artwork_payloads,
        "key": key,
    }


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


def _track_tags(track: DuplicateTrack) -> Mapping[str, object]:
    tags = track.current_tags if isinstance(track.current_tags, Mapping) else track.tags
    return tags if isinstance(tags, Mapping) else {}


def _has_metadata_seed(track: DuplicateTrack) -> bool:
    tags = _track_tags(track)
    title = tags.get("title")
    artist = tags.get("albumartist") or tags.get("artist")
    return bool(title and artist)


def _build_metadata_buckets(tracks: Sequence[DuplicateTrack]) -> List["BucketingBucket"]:
    buckets: List[BucketingBucket] = []
    by_key: Dict[tuple[str, str], List[int]] = defaultdict(list)

    for track in sorted(tracks, key=lambda t: t.path.lower()):
        key = _metadata_bucket_key(track)
        seeded = _has_metadata_seed(track)
        target_id = by_key.get(key, [None])[0]
        if target_id is None:
            target_id = len(buckets)
            buckets.append(BucketingBucket(id=target_id, metadata_seeded=seeded))
            by_key[key].append(target_id)
        bucket = buckets[target_id]
        bucket.tracks.append(track)
        bucket.metadata_keys.add(key)
        bucket.metadata_seeded = bucket.metadata_seeded or seeded
        album_key = _normalize_album_key(_track_tags(track), track.path)
        if album_key:
            bucket.album_keys.add(album_key)
        else:
            bucket.missing_album = True
        bucket.sources[track.path] = "metadata" if seeded else "fallback"

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


def _image_mime(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _image_data_uri(payload: bytes) -> str:
    mime = _image_mime(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _preview_artwork_settings() -> tuple[int, int]:
    cfg = load_config()
    max_dim = int(cfg.get("preview_artwork_max_dim", PREVIEW_ARTWORK_MAX_DIM))
    quality = int(cfg.get("preview_artwork_quality", PREVIEW_ARTWORK_QUALITY))
    max_dim = max(16, min(max_dim, 1024))
    quality = max(1, min(quality, 95))
    return max_dim, quality


def _compress_preview_artwork(payload: bytes) -> tuple[bytes, str, Optional[int], Optional[int]]:
    if not payload:
        mime = _image_mime(payload)
        logger.info(
            "Preview artwork compress: size=0 bytes mime=%s pil=%s fallback=raw",
            mime,
            bool(Image),
        )
        return payload, mime, None, None
    if not Image:
        width, height = _extract_image_dimensions(payload)
        mime = _image_mime(payload)
        logger.info(
            "Preview artwork compress: size=%d bytes mime=%s pil=%s fallback=raw",
            len(payload),
            mime,
            bool(Image),
        )
        return payload, mime, width, height
    max_dim, quality = _preview_artwork_settings()
    original_size = len(payload)
    start = time.perf_counter()
    try:
        with Image.open(io.BytesIO(payload)) as img:
            img = img.convert("RGB")
            img.thumbnail((max_dim, max_dim), resample=_image_resample_filter())
            width, height = img.size
            buffer = io.BytesIO()
            try:
                img.save(buffer, format="WEBP", quality=quality)
                mime = "image/webp"
            except Exception:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality, optimize=True)
                mime = "image/jpeg"
        compressed = buffer.getvalue()
        if not compressed:
            raw_mime = _image_mime(payload)
            logger.info(
                "Preview artwork compress: size=%d bytes mime=%s pil=%s fallback=raw",
                original_size,
                raw_mime,
                bool(Image),
            )
            return payload, raw_mime, width, height
        logger.info(
            "Preview artwork compress: size=%d->%d bytes mime=%s pil=%s fallback=none elapsed=%.2fs",
            original_size,
            len(compressed),
            mime,
            bool(Image),
            time.perf_counter() - start,
        )
        return compressed, mime, width, height
    except Exception as exc:
        width, height = _extract_image_dimensions(payload)
        mime = _image_mime(payload)
        logger.warning(
            "Preview artwork compress: size=%d bytes mime=%s pil=%s fallback=raw elapsed=%.2fs",
            len(payload),
            mime,
            bool(Image),
            time.perf_counter() - start,
        )
        logger.exception(
            "Preview artwork compress failed: size=%d bytes error=%s",
            len(payload),
            exc,
        )
        return payload, mime, width, height


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
    hashes: Mapping[str, int | None] | None = None,
) -> tuple[List[List["DuplicateTrack"]], Dict[str, int | None]]:
    if len(tracks) <= 1:
        return [list(tracks)], {t.path: hashes.get(t.path) if hashes else _artwork_hash_for_track(t) for t in tracks}
    hashes = dict(hashes or {t.path: _artwork_hash_for_track(t) for t in tracks})
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


def _artwork_status(track: "DuplicateTrack", art_hash: int | None) -> Dict[str, str]:
    if art_hash is not None:
        return {"status": "known", "reason": "embedded artwork"}
    if track.artwork_error == "deferred":
        return {"status": "unknown", "reason": "artwork deferred"}
    reason = track.artwork_error or "no artwork found"
    ext = track.ext.lower()
    if ext in {".m4a", ".mp4"} and reason == "no artwork found":
        return {"status": "unknown", "reason": "art missing/unreadable (m4a cover not extracted)"}
    if track.artwork_error:
        return {"status": "unknown", "reason": f"art missing/unreadable ({track.artwork_error})"}
    return {"status": "unknown", "reason": "art missing/unreadable (no artwork found)"}


def _has_artwork_error(track: "DuplicateTrack") -> bool:
    return bool(track.artwork_error and track.artwork_error != "deferred")


def _load_artwork_for_track(track: "DuplicateTrack") -> None:
    if track.artwork:
        if track.artwork_error == "deferred":
            track.artwork_error = None
        return
    _tags, cover_payloads, read_error, reader_hint = read_metadata(track.path, include_cover=True)
    sidecar_used = False
    if not cover_payloads:
        sidecar_payload = read_sidecar_artwork_bytes(track.path)
        if sidecar_payload:
            cover_payloads = [sidecar_payload]
            sidecar_used = True

    artwork: List[ArtworkCandidate] = []
    if cover_payloads:
        for payload in cover_payloads:
            width, height = _extract_image_dimensions(payload)
            artwork.append(
                ArtworkCandidate(
                    path=track.path,
                    hash=hashlib.sha256(payload).hexdigest(),
                    size=len(payload),
                    width=width,
                    height=height,
                    status="ok",
                    bytes=payload,
                )
            )
        if not track.cover_hash:
            track.current_tags["cover_hash"] = artwork[0].hash
            track.current_tags["artwork_hash"] = artwork[0].hash
        track.artwork_error = None
    else:
        track.artwork_error = read_error or "no artwork found"

    track.artwork = artwork
    ext = track.ext.lower()
    album_art_trace = track.trace.get("album_art") if isinstance(track.trace, Mapping) else {}
    album_art_trace = dict(album_art_trace or {})
    album_art_trace.update(
        {
            "success": track.artwork_error is None,
            "error": "" if track.artwork_error is None else track.artwork_error,
            "cover_count": len(cover_payloads),
            "mp4_covr_missing": ext in {".m4a", ".mp4"} and not cover_payloads,
            "deferred": False,
            "reader_hint": reader_hint,
            "sidecar_used": sidecar_used,
        }
    )
    track.trace["album_art"] = album_art_trace


def _compress_artwork_for_preview(track: "DuplicateTrack") -> None:
    if not track.artwork:
        return
    for candidate in track.artwork:
        raw_payload = candidate.bytes or b""
        logger.info(
            "Preview artwork candidate: track=%s size=%d",
            track.path,
            len(raw_payload),
        )
        if not candidate.bytes:
            continue
        if not candidate.hash:
            candidate.hash = hashlib.sha256(raw_payload).hexdigest()
        compressed_payload, _mime, width, height = _compress_preview_artwork(raw_payload)
        candidate.bytes = compressed_payload
        candidate.size = len(compressed_payload)
        if width is not None:
            candidate.width = width
        if height is not None:
            candidate.height = height
        logger.info(
            "Preview artwork candidate compressed: track=%s size=%d->%d dims=%sx%s",
            track.path,
            len(raw_payload),
            candidate.size,
            width,
            height,
        )


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


def _read_tags_and_artwork(
    path: str, provided_tags: Mapping[str, object] | None
) -> tuple[Dict[str, object], List[ArtworkCandidate], Optional[str], Optional[str], Dict[str, object], Dict[str, object]]:
    audio, error = _read_audio_file(path)
    file_tags, cover_payloads, read_error, _reader = read_metadata(
        path, include_cover=False, audio=audio
    )
    base = _merge_tags(_blank_tags(), file_tags)
    fallback = provided_tags or {}
    for key, val in (fallback or {}).items():
        if key in base and base[key] not in (None, "", []):
            continue
        base[key] = val
    if read_error and error is None:
        error = read_error
    artwork: List[ArtworkCandidate] = []
    art_error: Optional[str] = "deferred"
    audio_props = _extract_audio_properties(audio, path, provided_tags or {}, os.path.splitext(path)[1])
    ext = os.path.splitext(path)[1].lower()
    metadata_trace = {
        "reader_hint": _reader,
        "cover_count": None,
        "mp4_covr_missing": ext in {".m4a", ".mp4"} and not cover_payloads,
        "sidecar_used": False,
        "cover_deferred": True,
        "source": "file",
    }
    return base, artwork, error, art_error, audio_props, metadata_trace


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


def _capture_library_state(path: str, *, quick: bool = False) -> Dict[str, object]:
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

    if not quick:
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
    trace: Dict[str, object] = field(default_factory=dict)

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
class BucketingBucket:
    """Bucket of candidate tracks for duplicate matching."""

    id: int
    tracks: List[DuplicateTrack] = field(default_factory=list)
    metadata_seeded: bool = False
    album_keys: set[tuple[str, str]] = field(default_factory=set)
    missing_album: bool = False
    sources: Dict[str, str] = field(default_factory=dict)
    metadata_keys: set[tuple[str, str]] = field(default_factory=set)
    artwork_merges: List[Dict[str, object]] = field(default_factory=list)


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
    bucket_diagnostics: Dict[str, object] = field(default_factory=dict)

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
    artwork_variant_id: int
    artwork_variant_total: int
    artwork_variant_label: str
    artwork_unknown_tracks: List[str]
    artwork_unknown_reasons: Dict[str, str]
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
    bucket_diagnostics: Dict[str, object] = field(default_factory=dict)
    artwork_hashes: Dict[str, int | None] = field(default_factory=dict)
    fingerprint_distances: Dict[str, Dict[str, float]] = field(default_factory=dict)
    library_state: Dict[str, Dict[str, object]] = field(default_factory=dict)
    pipeline_trace: Dict[str, object] = field(default_factory=dict)

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
            "artwork_variant_id": self.artwork_variant_id,
            "artwork_variant_total": self.artwork_variant_total,
            "artwork_variant_label": self.artwork_variant_label,
            "artwork_unknown_tracks": list(self.artwork_unknown_tracks),
            "artwork_unknown_reasons": dict(self.artwork_unknown_reasons),
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
            "bucket_diagnostics": dict(self.bucket_diagnostics),
            "artwork_hashes": {
                k: int(v) if isinstance(v, int) else None for k, v in self.artwork_hashes.items()
            },
            "fingerprint_distances": {k: dict(v) for k, v in self.fingerprint_distances.items()},
            "library_state": {k: dict(v) for k, v in self.library_state.items()},
            "pipeline_trace": dict(self.pipeline_trace),
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


def _planned_actions(group: GroupPlan) -> List[Dict[str, object]]:
    actions: List[Dict[str, object]] = []
    for path, changes in group.metadata_changes.items():
        fields = sorted(changes.keys()) if isinstance(changes, Mapping) else list(changes or [])
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


def _is_noop_group(actions: List[Dict[str, object]]) -> bool:
    has_non_loser_action = any(
        act["step"] in {"metadata", "artwork", "playlist"} for act in actions
    )
    has_non_retain_loser = any(
        act["step"] == "loser_cleanup" and act.get("metadata", {}).get("disposition") != "retain"
        for act in actions
    )
    return not has_non_loser_action and not has_non_retain_loser


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


def _iter_playlists(playlists_dir: str) -> Iterable[str]:
    if not os.path.isdir(playlists_dir):
        return []
    for dirpath, _dirs, files in os.walk(playlists_dir):
        for fname in files:
            if fname.lower().endswith(".m3u"):
                yield os.path.join(dirpath, fname)


def _normalize_playlist_entry(entry: str, playlist_dir: str) -> str:
    if not entry or entry.startswith("#"):
        return entry
    if os.path.isabs(entry):
        return os.path.normpath(entry)
    return os.path.normpath(os.path.join(playlist_dir, entry))


def _playlist_rewrite_losers(playlists_dir: str | None, losers: Iterable[str]) -> tuple[set[str], int]:
    if not playlists_dir or not os.path.isdir(playlists_dir):
        return set(), 0
    loser_set = {os.path.normpath(path) for path in losers if path}
    if not loser_set:
        return set(), 0
    matches: set[str] = set()
    playlists_with_hits = 0
    for playlist in _iter_playlists(playlists_dir):
        try:
            with open(playlist, "r", encoding="utf-8") as handle:
                lines = [ln.rstrip("\n") for ln in handle]
        except OSError:
            continue
        playlist_dir = os.path.dirname(playlist)
        normalized_lines = {_normalize_playlist_entry(ln, playlist_dir) for ln in lines if ln}
        hit = loser_set & normalized_lines
        if hit:
            playlists_with_hits += 1
            matches.update(hit)
            if len(matches) == len(loser_set):
                return matches, playlists_with_hits
    return matches, playlists_with_hits


def _infer_playlists_dir(paths: Sequence[str]) -> str | None:
    if not paths:
        return None
    try:
        common = os.path.commonpath(paths)
    except ValueError:
        return None
    candidate = common if os.path.isdir(common) else os.path.dirname(common)
    playlists_dir = os.path.join(candidate, "Playlists")
    if os.path.isdir(playlists_dir):
        return playlists_dir
    return None


def _build_playlist_index(playlists_dir: str | None) -> Dict[str, set[str]]:
    if not playlists_dir or not os.path.isdir(playlists_dir):
        return {}
    index: Dict[str, set[str]] = {}
    for playlist in _iter_playlists(playlists_dir):
        try:
            with open(playlist, "r", encoding="utf-8") as handle:
                lines = [ln.rstrip("\n") for ln in handle]
        except OSError:
            continue
        playlist_dir = os.path.dirname(playlist)
        for line in lines:
            normalized = _normalize_playlist_entry(line, playlist_dir)
            if not normalized or normalized.startswith("#"):
                continue
            normalized = os.path.normpath(normalized)
            index.setdefault(normalized, set()).add(playlist)
    return index


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
    else:
        coarse_gate = "fallback (no shared coarse keys)"

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
                status="pass",
                detail="No shared coarse keys; duplicate finder would fall back to direct comparison.",
            )
        )
        coarse_ok = True

    if not metadata_ok or not coarse_ok:
        steps.append(
            PairInspectionStep(
                name="Fingerprint availability",
                status="blocked",
                detail="Skipped because an earlier gate failed.",
            )
        )
        steps.append(
            PairInspectionStep(
                name="Fingerprint distance gate",
                status="blocked",
                detail="Skipped because an earlier gate failed.",
            )
        )
        match_type = "blocked"
        verdict = "Not a match (blocked by earlier gate)"
    else:
        if fingerprint_distance_value is None:
            steps.append(
                PairInspectionStep(
                    name="Fingerprint availability",
                    status="fail",
                    detail="Fingerprint unavailable for one or both tracks.",
                )
            )
            steps.append(
                PairInspectionStep(
                    name="Fingerprint distance gate",
                    status="blocked",
                    detail="Skipped because fingerprint data was missing.",
                )
            )
            match_type = "missing"
            verdict = "Not a match (fingerprint unavailable)"
        elif fingerprint_distance_value <= exact_threshold:
            match_type = "exact"
            verdict = "Exact duplicate"
            steps.append(
                PairInspectionStep(
                    name="Fingerprint availability",
                    status="pass",
                    detail="Fingerprints loaded for both tracks.",
                )
            )
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
                    name="Fingerprint availability",
                    status="pass",
                    detail="Fingerprints loaded for both tracks.",
                )
            )
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
                    name="Fingerprint availability",
                    status="pass",
                    detail="Fingerprints loaded for both tracks.",
                )
            )
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
    bucket_diagnostics_raw = raw.get("bucket_diagnostics")
    if bucket_diagnostics_raw is None:
        bucket_diagnostics = {}
    elif isinstance(bucket_diagnostics_raw, Mapping):
        bucket_diagnostics = dict(bucket_diagnostics_raw)
    else:
        raise ValueError("bucket_diagnostics must be a mapping.")
    artwork_hashes_raw = raw.get("artwork_hashes")
    if artwork_hashes_raw is None:
        artwork_hashes: Dict[str, int | None] = {}
    elif isinstance(artwork_hashes_raw, Mapping):
        artwork_hashes = {}
        for key, value in artwork_hashes_raw.items():
            if value is None:
                artwork_hashes[str(key)] = None
            else:
                artwork_hashes[str(key)] = int(value)
    else:
        raise ValueError("artwork_hashes must be a mapping.")
    pipeline_trace_raw = raw.get("pipeline_trace")
    if pipeline_trace_raw is None:
        pipeline_trace: Dict[str, object] = {}
    elif isinstance(pipeline_trace_raw, Mapping):
        pipeline_trace = dict(pipeline_trace_raw)
    else:
        raise ValueError("pipeline_trace must be a mapping.")

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
        artwork_variant_id=int(raw.get("artwork_variant_id") or 1),
        artwork_variant_total=int(raw.get("artwork_variant_total") or 1),
        artwork_variant_label=_expect_str(
            raw.get("artwork_variant_label") or "Single artwork variant",
            "artwork_variant_label",
        ),
        artwork_unknown_tracks=_expect_str_list(raw.get("artwork_unknown_tracks") or [], "artwork_unknown_tracks"),
        artwork_unknown_reasons=_expect_str_mapping(
            raw.get("artwork_unknown_reasons") or {},
            "artwork_unknown_reasons",
        ),
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
        bucket_diagnostics=bucket_diagnostics,
        artwork_hashes=artwork_hashes,
        fingerprint_distances=_expect_mapping(raw.get("fingerprint_distances"), "fingerprint_distances"),
        library_state=_expect_mapping(raw.get("library_state"), "library_state"),
        pipeline_trace=pipeline_trace,
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


def _normalize_track(raw: Mapping[str, object], *, quick_state: bool = False) -> DuplicateTrack:
    path = str(raw.get("path"))
    ext = os.path.splitext(path)[1].lower() or str(raw.get("ext", "")).lower()
    raw_tags = raw.get("tags") if isinstance(raw.get("tags"), Mapping) else {}
    provided_tags = _normalize_provided_tags(raw_tags)
    raw_artwork = raw.get("artwork") if isinstance(raw.get("artwork"), list) else []
    should_skip, completeness = _metadata_payload_complete(provided_tags, raw, raw_artwork)
    if should_skip:
        current_tags = _merge_tags(_blank_tags(), provided_tags)
        artwork = []
        meta_err = None
        art_err = "deferred"
        audio_payload = {
            "bitrate": raw.get("bitrate"),
            "sample_rate": raw.get("sample_rate"),
            "samplerate": raw.get("samplerate"),
            "bit_depth": raw.get("bit_depth"),
            "bitdepth": raw.get("bitdepth"),
            "channels": raw.get("channels"),
            "codec": raw.get("codec"),
            "codec_name": raw.get("codec_name"),
            "container": raw.get("container"),
            "format": raw.get("format"),
        }
        audio_props = _extract_audio_properties(None, path, audio_payload, ext)
        meta_trace = {
            "reader_hint": "provided metadata",
            "cover_count": None,
            "mp4_covr_missing": ext in {".m4a", ".mp4"} and not raw_artwork,
            "sidecar_used": False,
            "cover_deferred": True,
            "source": "provided",
            **completeness,
        }
    else:
        current_tags, artwork, meta_err, art_err, audio_props, meta_trace = _read_tags_and_artwork(
            path, provided_tags
        )
    library_state = _capture_library_state(path, quick=quick_state)
    provided_artwork: List[ArtworkCandidate] = []
    for art in raw_artwork:
        try:
            payload = art.get("bytes")
            width = art.get("width")
            height = art.get("height")
            size = art.get("size")
            art_hash = art.get("hash")
            if isinstance(payload, (bytes, bytearray)) and payload:
                payload = bytes(payload)
                if art_hash is None:
                    art_hash = hashlib.sha256(payload).hexdigest()
                if size is None:
                    size = len(payload)
                if width is None or height is None:
                    raw_width, raw_height = _extract_image_dimensions(payload)
                    if width is None:
                        width = raw_width
                    if height is None:
                        height = raw_height
            provided_artwork.append(
                ArtworkCandidate(
                    path=path,
                    hash=str(art_hash or ""),
                    size=int(size or 0),
                    width=width,
                    height=height,
                    status=art.get("status", "ok"),
                    bytes=payload if isinstance(payload, (bytes, bytearray)) else None,
                )
            )
        except Exception:
            continue
    if provided_artwork and not artwork:
        artwork = provided_artwork
        if "cover_hash" not in current_tags:
            current_tags["cover_hash"] = provided_artwork[0].hash
            current_tags["artwork_hash"] = provided_artwork[0].hash
        art_err = None
        meta_trace = dict(meta_trace)
        meta_trace["cover_count"] = len(provided_artwork)
        meta_trace["cover_deferred"] = False
        meta_trace["artwork_source"] = "provided"
    tags = dict(current_tags)
    discovery = raw.get("discovery") if isinstance(raw.get("discovery"), Mapping) else {}
    fingerprint_trace = (
        raw.get("fingerprint_trace") if isinstance(raw.get("fingerprint_trace"), Mapping) else {}
    )
    normalized_fields = {
        "artist": tags.get("artist"),
        "albumartist": tags.get("albumartist"),
        "title": tags.get("title"),
        "album": tags.get("album"),
        "track": tags.get("track") or tags.get("tracknumber"),
        "disc": tags.get("disc") or tags.get("discnumber"),
        "date": tags.get("date"),
        "year": tags.get("year"),
    }
    trace = {
        "discovery": dict(discovery),
        "metadata_read": {
            "success": meta_err is None,
            "error": meta_err or "",
            "reader_hint": meta_trace.get("reader_hint"),
            "normalized_fields": normalized_fields,
            "source": meta_trace.get("source"),
            "tags_complete": meta_trace.get("tags_complete"),
            "audio_props_complete": meta_trace.get("audio_props_complete"),
        },
        "album_art": {
            "success": art_err is None,
            "error": "" if art_err in (None, "deferred") else art_err,
            "cover_count": meta_trace.get("cover_count"),
            "mp4_covr_missing": meta_trace.get("mp4_covr_missing"),
            "deferred": art_err == "deferred",
            "source": meta_trace.get("artwork_source"),
        },
        "fingerprint": {
            "success": bool(raw.get("fingerprint")),
            "source": fingerprint_trace.get("source"),
            "error": fingerprint_trace.get("error"),
        },
    }
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
        trace=trace,
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


def _artwork_tiebreak_key(track: DuplicateTrack, blob: ArtworkCandidate) -> tuple[str, str]:
    return (track.path or "", blob.hash or "")


def _select_single_release_artwork_candidate(
    candidates: Sequence[DuplicateTrack], release_sizes: Mapping[str, str]
) -> tuple[DuplicateTrack | None, ArtworkCandidate | None, bool]:
    best_track: DuplicateTrack | None = None
    best_blob: ArtworkCandidate | None = None
    best_score: tuple | None = None
    best_tiebreak: tuple[str, str] | None = None
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
            best_tiebreak = _artwork_tiebreak_key(track, blob)
            ambiguous = False
        elif score == best_score:
            tiebreak = _artwork_tiebreak_key(track, blob)
            if best_tiebreak is None or tiebreak < best_tiebreak:
                best_track = track
                best_blob = blob
                best_tiebreak = tiebreak
    return best_track, best_blob, ambiguous


def _select_overall_artwork_candidate(candidates: Sequence[DuplicateTrack]) -> tuple[DuplicateTrack | None, ArtworkCandidate | None, bool]:
    best_track: DuplicateTrack | None = None
    best_blob: ArtworkCandidate | None = None
    best_score: tuple | None = None
    best_tiebreak: tuple[str, str] | None = None
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
            best_tiebreak = _artwork_tiebreak_key(track, blob)
            ambiguous = False
        elif score == best_score:
            tiebreak = _artwork_tiebreak_key(track, blob)
            if best_tiebreak is None or tiebreak < best_tiebreak:
                best_track = track
                best_blob = blob
                best_tiebreak = tiebreak
    return best_track, best_blob, ambiguous


def _bucket_primary_key(bucket: BucketingBucket) -> tuple[str, str]:
    if bucket.metadata_keys:
        return sorted(bucket.metadata_keys)[0]
    return ("unknown", "unknown")


def _bucket_diagnostics(bucket: BucketingBucket) -> Dict[str, object]:
    return {
        "bucket_id": bucket.id,
        "metadata_seeded": bucket.metadata_seeded,
        "metadata_keys": [list(key) for key in sorted(bucket.metadata_keys)],
        "album_keys": [list(key) for key in sorted(bucket.album_keys)],
        "missing_album": bucket.missing_album,
        "sources": dict(bucket.sources),
        "artwork_merges": list(bucket.artwork_merges),
    }


def _merge_buckets_by_artwork(
    buckets: List[BucketingBucket],
    similarity_threshold: int,
) -> List[BucketingBucket]:
    if not buckets:
        return []

    parent = list(range(len(buckets)))
    root_seeded = [bucket.metadata_seeded for bucket in buckets]

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a
            root_seeded[root_a] = root_seeded[root_a] or root_seeded[root_b]

    entries: List[tuple[int, str, int]] = []
    path_to_bucket: Dict[str, int] = {}
    for bucket in buckets:
        for track in bucket.tracks:
            path_to_bucket[track.path] = bucket.id
            art_hash = _artwork_hash_for_track(track)
            if art_hash is not None:
                entries.append((bucket.id, track.path, art_hash))

    merge_events: List[Dict[str, object]] = []
    for i, (bucket_id, left_path, left_hash) in enumerate(entries):
        for right_bucket_id, right_path, right_hash in entries[i + 1 :]:
            if bucket_id == right_bucket_id:
                continue
            left_root = find(bucket_id)
            right_root = find(right_bucket_id)
            if not (root_seeded[left_root] or root_seeded[right_root]):
                continue
            distance = _hamming_distance(left_hash, right_hash)
            if distance <= similarity_threshold:
                union(bucket_id, right_bucket_id)
                merge_events.append(
                    {"left": left_path, "right": right_path, "distance": distance}
                )

    root_metadata_seeded: Dict[int, bool] = {}
    for bucket in buckets:
        root = find(bucket.id)
        root_metadata_seeded[root] = root_metadata_seeded.get(root, False) or bucket.metadata_seeded

    merged: Dict[int, BucketingBucket] = {}
    for bucket in buckets:
        root = find(bucket.id)
        if root not in merged:
            merged[root] = BucketingBucket(id=root)
        target = merged[root]
        target.metadata_seeded = root_metadata_seeded[root]
        target.album_keys.update(bucket.album_keys)
        target.missing_album = target.missing_album or bucket.missing_album
        target.metadata_keys.update(bucket.metadata_keys)
        for track in bucket.tracks:
            target.tracks.append(track)
            source = bucket.sources.get(track.path, "fallback")
            if source != "metadata" and not bucket.metadata_seeded and root_metadata_seeded[root]:
                source = "artwork"
            target.sources[track.path] = source

    for event in merge_events:
        bucket_id = path_to_bucket.get(str(event.get("left")))
        if bucket_id is None:
            continue
        root = find(bucket_id)
        merged[root].artwork_merges.append(event)

    merged_buckets = sorted(merged.values(), key=lambda b: b.id)
    for new_id, bucket in enumerate(merged_buckets):
        bucket.id = new_id
        bucket.tracks = sorted({t.path: t for t in bucket.tracks}.values(), key=lambda t: t.path.lower())
    return merged_buckets


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
    log_callback: Callable[[str], None] | None = None,
) -> List[ClusterResult]:
    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    near_duplicate_threshold = max(near_duplicate_threshold, exact_duplicate_threshold)
    log("Grouping phase: building metadata buckets...")
    buckets = _build_metadata_buckets(tracks)
    total_tracks = sum(len(bucket.tracks) for bucket in buckets)
    log(
        "Grouping phase: metadata buckets ready "
        f"({len(buckets)} buckets, {total_tracks} tracks)."
    )
    for bucket in buckets:
        bucket_key = _bucket_primary_key(bucket)
        for track in bucket.tracks:
            track.trace.setdefault("bucketing", {})
            track.trace["bucketing"] = {
                "bucket_id": bucket.id,
                "bucket_primary_key": list(bucket_key),
                "metadata_key": list(_metadata_bucket_key(track)),
                "metadata_seeded": bucket.metadata_seeded,
                "bucket_source": bucket.sources.get(track.path, "fallback"),
                "album_keys": [list(key) for key in sorted(bucket.album_keys)],
                "missing_album": bucket.missing_album,
            }
    clusters: List[ClusterResult] = []
    comparisons = 0
    processed = 0
    stop_requested = False
    log("Grouping phase: comparing fingerprint distances...")
    for bucket in buckets:
        if cancel_event.is_set() or (timeout_sec and (_now() - start_time) > timeout_sec):
            review_flags.append("Consolidation planning cancelled or timed out during grouping.")
            break
        bucket_tracks = [t for t in bucket.tracks if t.fingerprint]
        if len(bucket_tracks) < 2:
            processed += len(bucket_tracks)
            continue
        bucket_tracks = sorted(bucket_tracks, key=lambda t: t.path.lower())
        path_to_track = {t.path: t for t in bucket_tracks}
        pair_info: Dict[tuple[str, str], tuple[float, float, str, List[str]]] = {}
        coarse_keys = {t.path: _coarse_fingerprint_keys(t.fingerprint) for t in bucket_tracks}

        for idx, left in enumerate(bucket_tracks):
            for right in bucket_tracks[idx + 1 :]:
                if cancel_event.is_set() or comparisons >= max_comparisons:
                    review_flags.append("Comparison budget reached; grouping may be incomplete.")
                    stop_requested = True
                    break
                if timeout_sec and (_now() - start_time) > timeout_sec:
                    review_flags.append("Consolidation planning timed out while grouping.")
                    stop_requested = True
                    break
                comparisons += 1
                dist = fingerprint_distance(left.fingerprint, right.fingerprint)
                pair_threshold = near_duplicate_threshold
                if left.is_lossless != right.is_lossless:
                    pair_threshold += mixed_codec_threshold_boost
                if dist <= exact_duplicate_threshold:
                    verdict = "exact"
                elif dist <= pair_threshold:
                    verdict = "near"
                else:
                    verdict = "no match"
                shared_keys = sorted(set(coarse_keys.get(left.path, [])) & set(coarse_keys.get(right.path, [])))
                pair_info[(left.path, right.path)] = (dist, pair_threshold, verdict, shared_keys)
            if stop_requested:
                break

        if stop_requested:
            break

        used: set[str] = set()
        bucket_key = _bucket_primary_key(bucket)
        bucket_diag = _bucket_diagnostics(bucket)
        for idx, track in enumerate(bucket_tracks):
            if cancel_event.is_set():
                review_flags.append("Consolidation planning cancelled or timed out during grouping.")
                stop_requested = True
                break
            if track.path in used:
                continue
            group = [track]
            decisions: List[GroupingDecision] = []
            used.add(track.path)
            for other in bucket_tracks[idx + 1 :]:
                if other.path in used:
                    continue
                left_key = (track.path, other.path)
                right_key = (other.path, track.path)
                pair = pair_info.get(left_key) or pair_info.get(right_key)
                if not pair:
                    continue
                dist, pair_threshold, verdict, shared_keys = pair
                if verdict not in {"exact", "near"}:
                    continue
                compatible = True
                max_candidate_distance = dist
                for member in group:
                    if member.path == other.path:
                        continue
                    cmp_pair = pair_info.get((member.path, other.path)) or pair_info.get(
                        (other.path, member.path)
                    )
                    if not cmp_pair:
                        compatible = False
                        break
                    cmp_dist, member_threshold, cmp_verdict, _ = cmp_pair
                    if cmp_verdict not in {"exact", "near"} or cmp_dist > member_threshold:
                        compatible = False
                        break
                    max_candidate_distance = max(max_candidate_distance, cmp_dist)
                if compatible:
                    match_type = "exact" if max_candidate_distance <= exact_duplicate_threshold else "near"
                    decisions.append(
                        GroupingDecision(
                            anchor_path=track.path,
                            candidate_path=other.path,
                            metadata_key=bucket_key,
                            coarse_keys_anchor=sorted(coarse_keys.get(track.path, [])),
                            coarse_keys_candidate=sorted(coarse_keys.get(other.path, [])),
                            shared_coarse_keys=shared_keys,
                            distance_to_anchor=dist,
                            max_group_distance=max_candidate_distance,
                            threshold=pair_threshold,
                            match_type=match_type,
                            coarse_gate="bucket",
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
                        bucket_diagnostics=bucket_diag,
                    )
                )
            processed += 1
            progress_callback(processed, total_tracks, track.path)
        if stop_requested:
            break
    log(
        "Grouping phase: comparison pass complete "
        f"({comparisons} comparisons, {len(clusters)} clusters)."
    )
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
    log_callback: Callable[[str], None] | None = None,
    fingerprint_settings: Mapping[str, object] | None = None,
    threshold_settings: Mapping[str, float] | None = None,
) -> ConsolidationPlan:
    """Generate a deterministic consolidation plan without modifying files."""

    cancel_event = cancel_event or threading.Event()
    progress_callback = progress_callback or _default_progress
    review_flags: List[str] = []
    start_time = _now()
    threshold_settings = dict(threshold_settings or {})

    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    raw_tracks = [dict(raw) for raw in tracks]
    cache_path = _metadata_cache_path_for_tracks(raw_tracks)
    metadata_cache = _load_metadata_cache(cache_path) if cache_path else {"entries": {}}
    cache_entries = (
        metadata_cache.get("entries")
        if isinstance(metadata_cache.get("entries"), Mapping)
        else {}
    )
    if cache_entries:
        for idx, raw in enumerate(raw_tracks):
            path = str(raw.get("path") or "")
            if not path:
                continue
            state = _capture_library_state(path, quick=True)
            cache_key = _metadata_cache_key(path, state)
            cached_entry = cache_entries.get(cache_key) if cache_key else None
            if cached_entry and isinstance(cached_entry, Mapping):
                raw_tracks[idx] = _preload_cached_metadata(raw, cached_entry)

    normalized: List[DuplicateTrack] = []
    log("Grouping phase: normalizing tracks...")
    for raw in raw_tracks:
        if cancel_event.is_set():
            review_flags.append("Cancelled before normalization.")
            break
        normalized.append(_normalize_track(raw, quick_state=True))
        if len(normalized) >= max_candidates:
            review_flags.append(f"Truncated candidate set to {max_candidates} items to protect runtime.")
            break
    log(f"Grouping phase: normalized {len(normalized)} tracks.")

    log("Grouping phase: running duplicate clustering...")
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
        log_callback=log_callback,
    )
    log(f"Grouping phase: clustering complete ({len(clusters)} clusters).")

    cluster_track_map: Dict[str, DuplicateTrack] = {}
    for cluster in clusters:
        for track in cluster.tracks:
            cluster_track_map[track.path] = track
    if cluster_track_map:
        log("Grouping phase: loading artwork for clustered tracks...")
        for track in sorted(cluster_track_map.values(), key=lambda t: t.path):
            if track.artwork_error == "deferred" and not track.artwork:
                _load_artwork_for_track(track)
            _compress_artwork_for_preview(track)
        log("Grouping phase: artwork loading complete.")

    if cache_path:
        updated_entries: Dict[str, object] = (
            dict(cache_entries) if isinstance(cache_entries, Mapping) else {}
        )
        for track in normalized:
            entry = _metadata_cache_entry(track)
            if entry and entry.get("key"):
                updated_entries[str(entry["key"])] = entry
        _save_metadata_cache(
            cache_path,
            {"version": METADATA_CACHE_VERSION, "entries": updated_entries},
        )

    log("Grouping phase: building consolidation plan...")
    plans: List[GroupPlan] = []
    plan_placeholders = False
    playlists_dir = _infer_playlists_dir([t.path for t in normalized])
    playlist_index: Dict[str, set[str]] = {}
    if playlists_dir:
        log("Grouping phase: indexing playlists...")
        playlist_index = _build_playlist_index(playlists_dir)
    for cluster in sorted(clusters, key=lambda c: _stable_group_id([t.path for t in c.tracks])):
        cluster_tracks = cluster.tracks
        artwork_hashes = {t.path: _artwork_hash_for_track(t) for t in cluster_tracks}
        artwork_statuses = {t.path: _artwork_status(t, artwork_hashes.get(t.path)) for t in cluster_tracks}
        known_art_tracks = [t for t in cluster_tracks if artwork_hashes.get(t.path) is not None]
        unknown_art_tracks = [t for t in cluster_tracks if artwork_hashes.get(t.path) is None]
        unknown_art_reasons = {
            t.path: artwork_statuses[t.path]["reason"]
            for t in unknown_art_tracks
            if t.path in artwork_statuses
        }
        if known_art_tracks:
            artwork_groups, _ = _split_by_artwork_similarity(
                known_art_tracks,
                hashes=artwork_hashes,
            )
            artwork_groups = sorted(artwork_groups, key=lambda g: _stable_group_id([t.path for t in g]))
            best_known = sorted(
                known_art_tracks,
                key=lambda t: _quality_tuple(t, "unknown"),
                reverse=True,
            )[0]
            primary_variant_index = 0
            for idx, group in enumerate(artwork_groups):
                if best_known in group:
                    primary_variant_index = idx
                    break
            groups_to_plan: List[tuple[List[DuplicateTrack], int, bool]] = []
            for idx, group in enumerate(artwork_groups):
                group_tracks = list(group)
                if idx == primary_variant_index and unknown_art_tracks:
                    existing_paths = {t.path for t in group_tracks}
                    for t in unknown_art_tracks:
                        if t.path not in existing_paths:
                            group_tracks.append(t)
                groups_to_plan.append((group_tracks, idx, idx == primary_variant_index))
            artwork_variant_total = len(artwork_groups)
        else:
            groups_to_plan = [(list(cluster_tracks), 0, True)]
            artwork_variant_total = 1
        artwork_split = artwork_variant_total > 1
        for group_tracks, artwork_variant_id, is_primary_variant in groups_to_plan:
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
            if artwork_split and not suppress_review_flags:
                release_notes.append(
                    "Audio-identical cluster contains multiple artwork variants; consolidation limited to each variant."
                )

            group_paths = {t.path for t in group_tracks}
            group_contexts = {path: contexts[path] for path in group_paths}
            group_context_evidence = {path: context_evidence[path] for path in group_paths}
            group_unknown_reasons = {
                path: reason
                for path, reason in unknown_art_reasons.items()
                if path in group_paths
            }
            group_decisions = [
                d for d in cluster.decisions if d.anchor_path in group_paths and d.candidate_path in group_paths
            ]
            group_artwork_hashes = {path: artwork_hashes.get(path) for path in group_paths}

            known_group_paths = {p for p in group_paths if artwork_hashes.get(p) is not None}
            quality_sorted = sorted(
                group_tracks,
                key=lambda t: (
                    1 if t.path in known_group_paths else 0,
                    _quality_tuple(t, group_contexts[t.path]),
                ),
                reverse=True,
            )
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
                    status = artwork_statuses.get(group_track.path, {})
                    if status.get("status") == "unknown":
                        artwork_evidence.append(
                            "Artwork missing/unreadable; track kept in its artwork variant but missing art does not block consolidation."
                        )
                    else:
                        artwork_evidence.append(
                            "Artwork differs from other audio duplicates; preserving this track as an intentional variant."
                        )
                else:
                    artwork_evidence.append(
                        "Artwork similarity gate grouped only tracks with matching covers for consolidation."
                    )
            if group_unknown_reasons:
                missing_paths = ", ".join(sorted(os.path.basename(p) or p for p in group_unknown_reasons))
                missing_notes = ", ".join(sorted(set(group_unknown_reasons.values())))
                artwork_evidence.append(
                    f"Artwork missing/unreadable for: {missing_paths or 'unknown'} ({missing_notes})."
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
                if any(_has_artwork_error(t) for t in group_tracks):
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
            if playlist_index:
                normalized_losers = {os.path.normpath(loser): loser for loser in losers}
                hit_norms = {
                    norm for norm in normalized_losers.keys() if norm in playlist_index
                }
                playlist_hits = {normalized_losers[norm] for norm in hit_norms}
                playlist_count = len(
                    {
                        playlist
                        for norm in hit_norms
                        for playlist in playlist_index.get(norm, set())
                    }
                )
            else:
                playlist_hits, playlist_count = _playlist_rewrite_losers(playlists_dir, losers)
            playlist_map = {loser: winner.path for loser in losers if loser in playlist_hits}

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
            if any(_has_artwork_error(t) for t in group_tracks) and not suppress_review_flags:
                group_review.append("Artwork extraction failed for at least one track.")

            if suppress_review_flags:
                group_review = []
                placeholders = False

            playlist_impact = PlaylistImpact(playlists=playlist_count, entries=len(playlist_hits))
            track_states = {t.path: dict(t.library_state) for t in group_tracks}

            artwork_candidates: List[ArtworkCandidate] = []
            for t in group_tracks:
                artwork_candidates.extend(t.artwork)

            current_tags = {t.path: _merge_tags(_blank_tags(), t.current_tags) for t in group_tracks}

            bucket_diag = cluster.bucket_diagnostics or {}
            bucket_sources = (
                bucket_diag.get("sources")
                if isinstance(bucket_diag.get("sources"), Mapping)
                else {}
            )
            has_metadata = bool(bucket_diag.get("metadata_seeded")) or (
                isinstance(bucket_sources, Mapping)
                and any(source == "metadata" for source in bucket_sources.values())
            )
            if has_metadata:
                formation = "metadata-seeded"
            else:
                formation = "fallback"
            art_threshold = float(
                threshold_settings.get("artwork_vastly_different_threshold", ARTWORK_VASTLY_DIFFERENT_THRESHOLD)
            )
            comparisons: List[Dict[str, object]] = []
            seen_pairs: set[tuple[str, str]] = set()
            for left, neighbors in pair_distances.items():
                for right, dist in neighbors.items():
                    pair = tuple(sorted((left, right)))
                    if left == right or pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    left_lossless = bool(track_quality.get(left, {}).get("is_lossless"))
                    right_lossless = bool(track_quality.get(right, {}).get("is_lossless"))
                    effective_near = near_duplicate_threshold
                    mixed_codec = left_lossless != right_lossless
                    if mixed_codec:
                        effective_near += mixed_codec_threshold_boost
                    verdict = "no match"
                    if dist <= exact_duplicate_threshold:
                        verdict = "exact"
                    elif dist <= effective_near:
                        verdict = "near"
                    comparisons.append(
                        {
                            "left": left,
                            "right": right,
                            "distance": dist,
                            "threshold": effective_near,
                            "verdict": verdict,
                            "mixed_codec": mixed_codec,
                        }
                    )
            preserve_overrides: Dict[str, Dict[str, object]] = {}
            winner_hash = group_artwork_hashes.get(winner.path)
            for loser in losers:
                loser_hash = group_artwork_hashes.get(loser)
                if winner_hash is None or loser_hash is None:
                    continue
                distance = _hamming_distance(winner_hash, loser_hash)
                if distance > art_threshold:
                    preserve_overrides[loser] = {
                        "distance": distance,
                        "threshold": art_threshold,
                    }
            track_traces: Dict[str, Dict[str, object]] = {}
            for t in group_tracks:
                track_trace = dict(t.trace)
                track_trace["artwork_hashing"] = {
                    "hash": group_artwork_hashes.get(t.path),
                    "hash_present": group_artwork_hashes.get(t.path) is not None,
                }
                track_trace["artwork_status"] = artwork_statuses.get(t.path, {})
                if winner_hash is not None and t.path != winner.path:
                    candidate_hash = group_artwork_hashes.get(t.path)
                    if candidate_hash is not None:
                        distance = _hamming_distance(winner_hash, candidate_hash)
                        track_trace["artwork_comparison"] = {
                            "winner_distance": distance,
                            "threshold": art_threshold,
                            "vastly_different": distance > art_threshold,
                        }
                track_traces[t.path] = track_trace
            pipeline_trace = {
                "summary": {
                    "group_id": group_id,
                    "bucket_id": bucket_diag.get("bucket_id", "n/a"),
                    "formation": formation,
                    "metadata_key": list(cluster.metadata_key),
                    "bucket_sources": dict(bucket_sources),
                    "album_keys": bucket_diag.get("album_keys", []),
                    "missing_album": bucket_diag.get("missing_album"),
                    "thresholds": {
                        "exact": exact_duplicate_threshold,
                        "near": near_duplicate_threshold,
                        "mixed_codec_boost": mixed_codec_threshold_boost,
                        "artwork_vastly_different_threshold": art_threshold,
                    },
                    "comparisons": comparisons,
                    "outcome": {
                        "winner": winner.path,
                        "losers": list(losers),
                        "dispositions": dict(dispositions),
                        "preserve_different_art": preserve_overrides,
                    },
                    "artwork_variants": {
                        "variant_id": artwork_variant_id + 1,
                        "variant_total": artwork_variant_total,
                        "primary_variant": is_primary_variant,
                    },
                    "artwork_unknown": dict(group_unknown_reasons),
                },
                "tracks": track_traces,
            }

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
                    bucket_diagnostics=cluster.bucket_diagnostics,
                    artwork_hashes=group_artwork_hashes,
                    fingerprint_distances=pair_distances,
                    library_state=track_states,
                    pipeline_trace=pipeline_trace,
                    artwork_variant_id=artwork_variant_id + 1,
                    artwork_variant_total=artwork_variant_total,
                    artwork_variant_label=(
                        f"Variant {artwork_variant_id + 1} of {artwork_variant_total}"
                        if artwork_variant_total > 1
                        else "Single artwork variant"
                    ),
                    artwork_unknown_tracks=sorted(group_unknown_reasons.keys()),
                    artwork_unknown_reasons=dict(group_unknown_reasons),
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
    start_time = time.monotonic()
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
    try:
        output_size = os.path.getsize(output_json_path)
    except OSError:
        output_size = None
    elapsed = time.monotonic() - start_time
    if output_size is not None:
        logger.info(
            "Preview JSON output: size=%d bytes groups=%d elapsed=%.2fs",
            output_size,
            len(plan.groups),
            elapsed,
        )
    else:
        logger.info(
            "Preview JSON output: groups=%d elapsed=%.2fs",
            len(plan.groups),
            elapsed,
        )
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
        esc(f"{report.fingerprint_distance:.4f}" if report.fingerprint_distance is not None else "n/a"),
        "</td></tr>",
        "<tr><th>Effective threshold</th><td>",
        esc(f"{report.effective_threshold:.4f}"),
        "</td></tr>",
        "</table>",
        "<h2>Gate Inputs</h2>",
        "<table>",
        f"<tr><th>Metadata key (Song A)</th><td>{esc(report.metadata_key_a)}</td></tr>",
        f"<tr><th>Metadata key (Song B)</th><td>{esc(report.metadata_key_b)}</td></tr>",
        f"<tr><th>Bucket match</th><td>{esc('Yes' if report.metadata_bucket_match else 'No')}</td></tr>",
        f"<tr><th>Bucket key (Song A)</th><td>{esc(report.bucket_key_a)}</td></tr>",
        f"<tr><th>Bucket key (Song B)</th><td>{esc(report.bucket_key_b)}</td></tr>",
        f"<tr><th>Coarse keys (Song A)</th><td>{esc(', '.join(report.coarse_keys_a) or 'n/a')}</td></tr>",
        f"<tr><th>Coarse keys (Song B)</th><td>{esc(', '.join(report.coarse_keys_b) or 'n/a')}</td></tr>",
        f"<tr><th>Shared coarse keys</th><td>{esc(', '.join(report.shared_coarse_keys) or 'none')}</td></tr>",
        f"<tr><th>Coarse gate behavior</th><td>{esc(report.coarse_gate)}</td></tr>",
        "</table>",
        "<h2>Tracks</h2>",
        "<table>",
        *_track_row(report.track_a, "Song A"),
        *_track_row(report.track_b, "Song B"),
        "</table>",
        "<h2>Pipeline Steps</h2>",
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
                f"<td><span class='badge {css}'>{esc(step.status)}</span></td>",
                f"<td>{esc(step.detail)}</td>",
                "</tr>",
            ]
        )

    html_lines.extend(
        [
            "</table>",
            "<h2>Thresholds</h2>",
            "<table>",
            f"<tr><th>Exact threshold</th><td>{esc(f'{report.exact_threshold:.4f}')}</td></tr>",
            f"<tr><th>Near threshold</th><td>{esc(f'{report.near_threshold:.4f}')}</td></tr>",
            f"<tr><th>Mixed-codec boost</th><td>{esc(f'{report.mixed_codec_boost:.4f}')}</td></tr>",
            f"<tr><th>Mixed-codec applied</th><td>{esc('Yes' if report.mixed_codec else 'No')}</td></tr>",
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


def export_consolidation_preview_html(
    plan: ConsolidationPlan,
    output_html_path: str,
    *,
    show_artwork_variants: bool = True,
) -> str:
    """Write an HTML preview of the consolidation plan."""

    plan.refresh_plan_signature()
    embedded_artwork_count = 0
    data_uri_cache: dict[bytes, str] = {}
    start_time = time.monotonic()

    def esc(value: object) -> str:
        return html.escape(str(value))

    def _data_uri(payload: bytes) -> str:
        nonlocal embedded_artwork_count
        cached = data_uri_cache.get(payload)
        if cached is not None:
            return cached
        embedded_artwork_count += 1
        uri = _image_data_uri(payload)
        data_uri_cache[payload] = uri
        return uri

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
        loser_count = sum(1 for act in actions if act["step"] == "loser_cleanup")
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

    def _is_artwork_variant(group: GroupPlan, loser_path: str, threshold: float) -> bool:
        winner_hash = group.artwork_hashes.get(group.winner_path)
        loser_hash = group.artwork_hashes.get(loser_path)
        if winner_hash is None or loser_hash is None:
            return False
        return _hamming_distance(winner_hash, loser_hash) > threshold

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
                return _data_uri(candidate.bytes)
            if track_path and candidate.path == track_path and path_fallback is None:
                path_fallback = candidate
            if fallback is None:
                fallback = candidate
        if path_fallback and path_fallback.bytes:
            return _data_uri(path_fallback.bytes)
        if fallback and fallback.bytes:
            return _data_uri(fallback.bytes)
        return None

    path_counter = 0

    def _path_row(path_value: str) -> str:
        nonlocal path_counter
        path_counter += 1
        el_id = f"path-{path_counter}"
        safe = esc(path_value)
        return (
            "<div class='path-row'>"
            f"<span class='path path-truncate' id='{el_id}'>{safe}</span>"
            f"<button class='copy tiny' data-copy-text='{safe}'>Copy</button>"
            f"<button class='copy tiny' data-expand='{el_id}'>Expand</button>"
            "</div>"
        )

    def _pairwise_stats(group: GroupPlan) -> Dict[str, object]:
        thresholds = group.grouping_thresholds or {}
        exact = float(thresholds.get("exact", 0.0))
        near = float(thresholds.get("near", exact))
        boost = float(thresholds.get("mixed_codec_boost", 0.0))
        distances = group.fingerprint_distances or {}
        track_quality = group.track_quality or {}
        exact_count = 0
        near_count = 0
        no_match_count = 0
        best_distance = None
        edges: List[Dict[str, object]] = []
        no_match_edges: List[Dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for left, neighbors in distances.items():
            for right, dist in neighbors.items():
                key = tuple(sorted((left, right)))
                if left == right or key in seen:
                    continue
                seen.add(key)
                left_lossless = bool(track_quality.get(left, {}).get("is_lossless"))
                right_lossless = bool(track_quality.get(right, {}).get("is_lossless"))
                effective_near = near + boost if left_lossless != right_lossless else near
                verdict = "no match"
                if dist <= exact:
                    verdict = "exact"
                    exact_count += 1
                elif dist <= effective_near:
                    verdict = "near"
                    near_count += 1
                else:
                    no_match_count += 1
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                payload = {
                    "left": left,
                    "right": right,
                    "distance": dist,
                    "threshold": effective_near,
                    "verdict": verdict,
                }
                if verdict == "no match":
                    no_match_edges.append(payload)
                else:
                    edges.append(payload)
        return {
            "exact": exact_count,
            "near": near_count,
            "no_match": no_match_count,
            "best": best_distance,
            "edges": edges,
            "no_match_edges": no_match_edges,
            "exact_threshold": exact,
            "near_threshold": near,
            "mixed_boost": boost,
        }

    def _format_trace_value(value: object) -> str:
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value) if value else "—"
        if isinstance(value, dict):
            return json.dumps(value, sort_keys=True)
        if value in (None, ""):
            return "—"
        return str(value)

    def _trace_panel(group: GroupPlan) -> List[str]:
        trace = group.pipeline_trace if isinstance(group.pipeline_trace, Mapping) else {}
        summary = trace.get("summary") if isinstance(trace.get("summary"), Mapping) else {}
        tracks = trace.get("tracks") if isinstance(trace.get("tracks"), Mapping) else {}
        trace_id = f"trace-{group.group_id}"
        lines: List[str] = []
        lines.append(f"<details class='trace' id='{esc(trace_id)}'>")
        lines.append("<summary class='trace-summary'>Pipeline Trace</summary>")
        lines.append("<div class='trace-grid'>")
        lines.append("<div class='k'>Bucket ID</div>")
        lines.append(f"<div class='v mono'>{esc(_format_trace_value(summary.get('bucket_id')))}</div>")
        lines.append("<div class='k'>Formation</div>")
        lines.append(f"<div class='v'>{esc(_format_trace_value(summary.get('formation')))}</div>")
        lines.append("<div class='k'>Metadata key</div>")
        lines.append(f"<div class='v'>{esc(_format_trace_value(summary.get('metadata_key')))}</div>")
        lines.append("<div class='k'>Bucket sources</div>")
        lines.append(f"<div class='v tiny'>{esc(_format_trace_value(summary.get('bucket_sources')))}</div>")
        lines.append("<div class='k'>Album keys</div>")
        lines.append(f"<div class='v'>{esc(_format_trace_value(summary.get('album_keys')))}</div>")
        lines.append("<div class='k'>Missing album</div>")
        lines.append(f"<div class='v'>{esc(_format_trace_value(summary.get('missing_album')))}</div>")
        lines.append("<div class='k'>Artwork variants</div>")
        lines.append(f"<div class='v tiny'>{esc(_format_trace_value(summary.get('artwork_variants')))}</div>")
        lines.append("<div class='k'>Artwork unknown</div>")
        lines.append(f"<div class='v tiny'>{esc(_format_trace_value(summary.get('artwork_unknown')))}</div>")
        lines.append("<div class='k'>Thresholds</div>")
        lines.append(f"<div class='v tiny'>{esc(_format_trace_value(summary.get('thresholds')))}</div>")
        lines.append("</div>")

        comparisons = summary.get("comparisons") if isinstance(summary.get("comparisons"), list) else []
        lines.append("<div class='trace-block'>")
        lines.append(f"<div class='tiny muted'>Comparisons attempted ({len(comparisons)})</div>")
        if comparisons:
            lines.append("<ul class='edge-list'>")
            for item in sorted(comparisons, key=lambda e: float(e.get("distance", 0.0))):
                left = str(item.get("left"))
                right = str(item.get("right"))
                distance = float(item.get("distance", 0.0))
                threshold = float(item.get("threshold", 0.0))
                verdict = str(item.get("verdict", ""))
                mixed_codec = " mixed-codec" if item.get("mixed_codec") else ""
                lines.append(
                    "<li>"
                    f"{esc(_basename(left))} ↔ {esc(_basename(right))} "
                    f"({distance:.4f} ≤ {threshold:.4f}, {esc(verdict)}{mixed_codec})"
                    "</li>"
                )
            lines.append("</ul>")
        else:
            lines.append("<div class='tiny muted'>No comparisons recorded.</div>")
        lines.append("</div>")

        outcome = summary.get("outcome") if isinstance(summary.get("outcome"), Mapping) else {}
        preserve = outcome.get("preserve_different_art") if isinstance(outcome.get("preserve_different_art"), Mapping) else {}
        lines.append("<div class='trace-block'>")
        lines.append("<div class='tiny muted'>Final outcome</div>")
        lines.append("<div class='trace-grid'>")
        lines.append("<div class='k'>Winner</div>")
        lines.append(f"<div class='v'>{esc(_format_trace_value(outcome.get('winner')))}</div>")
        lines.append("<div class='k'>Losers</div>")
        lines.append(f"<div class='v'>{esc(_format_trace_value(outcome.get('losers')))}</div>")
        lines.append("<div class='k'>Dispositions</div>")
        lines.append(f"<div class='v tiny'>{esc(_format_trace_value(outcome.get('dispositions')))}</div>")
        lines.append("<div class='k'>Preserve (different art)</div>")
        lines.append(f"<div class='v tiny'>{esc(_format_trace_value(preserve))}</div>")
        lines.append("</div>")
        lines.append("</div>")

        lines.append("<div class='trace-block'>")
        lines.append("<div class='tiny muted'>Per-file trace</div>")
        for path, payload in sorted(tracks.items(), key=lambda item: str(item[0]).lower()):
            trace_payload = payload if isinstance(payload, Mapping) else {}
            lines.append("<details class='trace-track'>")
            lines.append(
                "<summary class='trace-track-summary'>"
                f"{esc(_basename(str(path)))}"
                "</summary>"
            )
            lines.append("<div class='trace-grid'>")
            discovery = trace_payload.get("discovery") if isinstance(trace_payload.get("discovery"), Mapping) else {}
            lines.append("<div class='k'>Discovery</div>")
            lines.append(f"<div class='v tiny'>{esc(_format_trace_value(discovery))}</div>")
            metadata_read = trace_payload.get("metadata_read") if isinstance(trace_payload.get("metadata_read"), Mapping) else {}
            lines.append("<div class='k'>Metadata read</div>")
            lines.append(f"<div class='v tiny'>{esc(_format_trace_value(metadata_read))}</div>")
            album_art = trace_payload.get("album_art") if isinstance(trace_payload.get("album_art"), Mapping) else {}
            lines.append("<div class='k'>Album art</div>")
            lines.append(f"<div class='v tiny'>{esc(_format_trace_value(album_art))}</div>")
            artwork_hashing = trace_payload.get("artwork_hashing") if isinstance(trace_payload.get("artwork_hashing"), Mapping) else {}
            lines.append("<div class='k'>Artwork hashing</div>")
            lines.append(f"<div class='v tiny'>{esc(_format_trace_value(artwork_hashing))}</div>")
            artwork_status = trace_payload.get("artwork_status") if isinstance(trace_payload.get("artwork_status"), Mapping) else {}
            lines.append("<div class='k'>Artwork status</div>")
            lines.append(f"<div class='v tiny'>{esc(_format_trace_value(artwork_status))}</div>")
            artwork_comp = trace_payload.get("artwork_comparison") if isinstance(trace_payload.get("artwork_comparison"), Mapping) else {}
            if artwork_comp:
                lines.append("<div class='k'>Artwork compare</div>")
                lines.append(f"<div class='v tiny'>{esc(_format_trace_value(artwork_comp))}</div>")
            fingerprint = trace_payload.get("fingerprint") if isinstance(trace_payload.get("fingerprint"), Mapping) else {}
            lines.append("<div class='k'>Fingerprint</div>")
            lines.append(f"<div class='v tiny'>{esc(_format_trace_value(fingerprint))}</div>")
            bucketing = trace_payload.get("bucketing") if isinstance(trace_payload.get("bucketing"), Mapping) else {}
            lines.append("<div class='k'>Bucketing</div>")
            lines.append(f"<div class='v tiny'>{esc(_format_trace_value(bucketing))}</div>")
            lines.append("</div>")
            lines.append("</details>")
        if not tracks:
            lines.append("<div class='tiny muted'>No per-file trace data recorded.</div>")
        lines.append("</div>")
        lines.append("</details>")
        return lines

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

    group_pair_stats = {group.group_id: _pairwise_stats(group) for group in plan.groups}
    total_exact = sum(stats["exact"] for stats in group_pair_stats.values())
    total_near = sum(stats["near"] for stats in group_pair_stats.values())
    art_threshold = float(
        plan.threshold_settings.get("artwork_vastly_different_threshold", ARTWORK_VASTLY_DIFFERENT_THRESHOLD)
    )

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
        ".path-row{",
        "display:flex;",
        "flex-wrap:wrap;",
        "gap:8px;",
        "align-items:center;",
        "}",
        ".path-truncate{",
        "overflow:hidden;",
        "text-overflow: ellipsis;",
        "white-space: nowrap;",
        "max-width: 60ch;",
        "}",
        ".path-expanded{",
        "white-space: normal;",
        "word-break: break-all;",
        "max-width: none;",
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
        "button.copy.tiny{",
        "padding: 3px 7px;",
        "font-size: 11px;",
        "border-radius: 8px;",
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
        "button.trace-btn{",
        "border:1px solid var(--border);",
        "background:#fff;",
        "border-radius: 999px;",
        "padding: 3px 10px;",
        "font-size: 12px;",
        "font-weight: 650;",
        "cursor:pointer;",
        "white-space: nowrap;",
        "}",
        "button.trace-btn:hover{ background:#f7f7f7; }",
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
        ".badge.keep{ border-color: rgba(27,122,27,.35); background:#f2fff2; }",
        "details.bucket{",
        "border:1px dashed var(--border);",
        "border-radius: 10px;",
        "background:#fff;",
        "padding: 10px;",
        "}",
        "details.trace{",
        "border:1px dashed var(--border);",
        "border-radius: 10px;",
        "background:#fff;",
        "padding: 10px;",
        "margin-top: 8px;",
        "}",
        "summary.trace-summary{",
        "cursor:pointer;",
        "list-style:none;",
        "font-weight: 650;",
        "font-size: 12px;",
        "}",
        "summary.trace-summary::-webkit-details-marker{ display:none; }",
        ".trace-grid{",
        "display:grid;",
        "grid-template-columns: 180px 1fr;",
        "gap:6px 10px;",
        "font-size: 12px;",
        "margin-top: 8px;",
        "}",
        ".trace-block{ margin-top: 8px; }",
        "details.trace-track{",
        "border:1px solid var(--border);",
        "border-radius: 8px;",
        "padding: 8px;",
        "background:#fafafa;",
        "margin-top: 6px;",
        "}",
        "summary.trace-track-summary{",
        "cursor:pointer;",
        "list-style:none;",
        "font-weight: 600;",
        "font-size: 12px;",
        "}",
        "summary.trace-track-summary::-webkit-details-marker{ display:none; }",
        "summary.bucket-summary{",
        "cursor:pointer;",
        "list-style:none;",
        "font-weight: 650;",
        "font-size: 12px;",
        "}",
        "summary.bucket-summary::-webkit-details-marker{ display:none; }",
        ".bucket-grid{",
        "display:grid;",
        "grid-template-columns: 150px 1fr;",
        "gap:6px 10px;",
        "font-size: 12px;",
        "margin-top: 8px;",
        "}",
        ".edge-list{",
        "margin: 6px 0 0 18px;",
        "padding: 0;",
        "font-size: 12px;",
        "}",
        ".edge-list li{ margin: 4px 0; }",
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
        ".toggle{ display:flex; align-items:center; gap:6px; }",
        ".empty{ margin-top:14px; padding:18px; border:1px dashed var(--border); border-radius:8px; color:var(--muted); }",
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
        "<div class='row' style='gap:8px; margin-top:6px;'>",
        f"<span class='pill'><strong>Exact</strong> <span id='statExact'>{total_exact}</span></span>",
        f"<span class='pill'><strong>Near</strong> <span id='statNear'>{total_near}</span></span>",
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
        html_lines.insert(insert_at, "<button class='copy' id='toggleSettings'>Show settings</button>")
        html_lines.extend(
            [
                "<div class='card hidden settings-panel' id='settingsPanel'>",
                "<div class='kv'>",
                "<div class='k'>Active settings</div>",
                "<div class='v tiny muted'>Thresholds + fingerprint inputs used for this plan.</div>",
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
            "<label class='toggle tiny' title='When unchecked, hides groups marked ready.'>"
            "<input type='checkbox' id='showReady' /> Show ready groups</label>",
            "<label class='toggle tiny' title='When unchecked, hides groups with no planned operations.'>"
            "<input type='checkbox' id='showNoOps' /> Show no-op groups</label>",
            "<span class='tiny muted'>Hides groups with no planned operations when off.</span>",
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
    actionable_groups = 0
    for group in plan.groups:
        actions = _planned_actions(group)
        visible_losers = list(group.losers)
        hidden_losers: set[str] = set()
        if not show_artwork_variants:
            hidden_losers = {
                loser for loser in group.losers if _is_artwork_variant(group, loser, art_threshold)
            }
            if hidden_losers:
                visible_losers = [loser for loser in group.losers if loser not in hidden_losers]
                actions = [
                    act
                    for act in actions
                    if not (act["step"] == "loser_cleanup" and act["target"] in hidden_losers)
                ]
        if not show_artwork_variants and not actions and not visible_losers:
            continue
        is_noop = _is_noop_group(actions)
        all_actions.extend(actions)
        if not is_noop:
            actionable_groups += 1
        state = "review" if group.review_flags else "ready"
        badges = _action_badges(group, actions)
        group_disposition_count = sum(
            1
            for loser in visible_losers
            if group.loser_disposition.get(loser, "quarantine") != "retain"
        )
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
        group_paths = {group.winner_path, *group.losers}
        bucket_stats = group_pair_stats.get(group.group_id, {})
        best_distance = bucket_stats.get("best")
        best_distance_label = f"{best_distance:.4f}" if isinstance(best_distance, float) else "n/a"
        exact_count = int(bucket_stats.get("exact", 0) or 0)
        near_count = int(bucket_stats.get("near", 0) or 0)
        no_match_count = int(bucket_stats.get("no_match", 0) or 0)
        html_lines.append(
            "<details class='group' "
            f"data-group-id='{esc(group.group_id)}' "
            f"data-has-quarantine='{str(group_disposition_count > 0).lower()}' "
            "data-has-failure='false' "
            f"data-type='{esc(group_type_hint)}' "
            f"data-search='{esc(search_text.lower())}' "
            f"data-no-op='{str(is_noop).lower()}' "
            f"data-status='{esc(state)}' "
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
        trace_target = f"trace-{group.group_id}"
        html_lines.append(
            f"<button class='trace-btn' type='button' data-trace-target='{esc(trace_target)}'>Trace</button>"
        )
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
        loser_paths = visible_losers
        winner_art_src = _album_art_src(group, group.winner_path)
        winner_art_hash = group.artwork_hashes.get(group.winner_path)
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
        html_lines.append(f"<div class='v'>{_path_row(group.winner_path)}</div>")
        html_lines.append("<div class='k'>Group ID</div>")
        html_lines.append(f"<div class='v mono'>{esc(group.group_id)}</div>")
        html_lines.append("<div class='k'>Metadata bucket key</div>")
        html_lines.append(f"<div class='v mono'>{esc(group.grouping_metadata_key)}</div>")
        html_lines.append("<div class='k'>Audio identity</div>")
        html_lines.append(
            f"<div class='v tiny'>"
            f"{esc(group.group_match_type)} · best {esc(best_distance_label)} "
            f"(exact {exact_count} · near {near_count} · no-match {no_match_count})"
            "</div>"
        )
        html_lines.append("<div class='k'>Artwork variant</div>")
        html_lines.append(
            f"<div class='v tiny'>{esc(getattr(group, 'artwork_variant_label', ''))}</div>"
        )
        if getattr(group, "artwork_variant_total", 1) > 1:
            html_lines.append("<div class='k'>Artwork variants</div>")
            html_lines.append(
                "<div class='v tiny muted'>Audio-identical but different artwork variants preserved.</div>"
            )
        if getattr(group, "artwork_unknown_reasons", {}):
            missing_notes = ", ".join(
                f"{_basename(path)} ({reason})"
                for path, reason in sorted(
                    getattr(group, "artwork_unknown_reasons", {}).items(),
                    key=lambda item: item[0].lower(),
                )
            )
            html_lines.append("<div class='k'>Artwork missing/unreadable</div>")
            html_lines.append(f"<div class='v tiny'>{esc(missing_notes)}</div>")
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
            art_distance = None
            is_art_variant = False
            if loser_path:
                loser_hash = group.artwork_hashes.get(loser_path)
                if winner_art_hash is not None and loser_hash is not None:
                    art_distance = _hamming_distance(winner_art_hash, loser_hash)
                    is_art_variant = art_distance > art_threshold
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
                html_lines.append(f"<div class='v'>{_path_row(loser_path)}</div>")
            else:
                html_lines.append("<div class='v path'>—</div>")
            html_lines.append("<div class='k'>Disposition</div>")
            if loser_disposition:
                if is_art_variant:
                    html_lines.append(
                        "<div class='v'>Preserve (different art) "
                        "<span class='badge keep'>different art → keep</span></div>"
                    )
                else:
                    html_lines.append(f"<div class='v'>{esc(loser_disposition)}</div>")
            else:
                html_lines.append("<div class='v'>—</div>")
            if loser_reason:
                html_lines.append("<div class='k'>Reason</div>")
                html_lines.append(f"<div class='v tiny muted'>{esc(loser_reason)}</div>")
            if art_distance is not None:
                html_lines.append("<div class='k'>Artwork distance</div>")
                html_lines.append(
                    "<div class='v tiny'>"
                    "<details class='tiny'><summary>details</summary>"
                    f"<div class='muted'>Winner ↔ loser distance: {art_distance} "
                    f"(threshold {art_threshold:.0f})</div>"
                    "</details></div>"
                )
            html_lines.append("</div>")
            html_lines.append("</div>")
        html_lines.append("</div>")
        bucket_diag = group.bucket_diagnostics or {}
        bucket_sources = bucket_diag.get("sources") if isinstance(bucket_diag.get("sources"), Mapping) else {}
        bucket_size = len(bucket_sources) if bucket_sources else len(group_paths)
        bucket_id = bucket_diag.get("bucket_id", "n/a")
        has_metadata = bool(bucket_diag.get("metadata_seeded")) or (
            isinstance(bucket_sources, Mapping)
            and any(source == "metadata" for source in bucket_sources.values())
        )
        if has_metadata:
            formation = "metadata-seeded"
        else:
            formation = "fallback"
        edges = bucket_stats.get("edges", []) if isinstance(bucket_stats.get("edges"), list) else []
        no_match_edges = (
            bucket_stats.get("no_match_edges", [])
            if isinstance(bucket_stats.get("no_match_edges"), list)
            else []
        )
        html_lines.append("<details class='bucket'>")
        html_lines.append("<summary class='bucket-summary'>Bucket diagnostics</summary>")
        html_lines.append("<div class='bucket-grid'>")
        html_lines.append("<div class='k'>Bucket ID</div>")
        html_lines.append(f"<div class='v mono'>{esc(bucket_id)}</div>")
        html_lines.append("<div class='k'>Bucket size</div>")
        html_lines.append(f"<div class='v'>{bucket_size}</div>")
        html_lines.append("<div class='k'>Formation</div>")
        html_lines.append(f"<div class='v'>{esc(formation)}</div>")
        html_lines.append("<div class='k'>Exact / near / no-match</div>")
        html_lines.append(f"<div class='v'>{exact_count} / {near_count} / {no_match_count}</div>")
        html_lines.append("<div class='k'>Best distance</div>")
        html_lines.append(f"<div class='v'>{best_distance_label}</div>")
        html_lines.append("<div class='k'>Thresholds</div>")
        html_lines.append(
            f"<div class='v tiny'>exact {bucket_stats.get('exact_threshold', 0.0):.4f} · "
            f"near {bucket_stats.get('near_threshold', 0.0):.4f} · "
            f"mixed boost +{bucket_stats.get('mixed_boost', 0.0):.4f}</div>"
        )
        html_lines.append("</div>")
        html_lines.append("<div class='tiny muted' style='margin-top:6px;'>Match edges</div>")
        if edges:
            html_lines.append("<ul class='edge-list'>")
            for edge in sorted(edges, key=lambda e: float(e.get("distance", 0.0))):
                left = str(edge.get("left"))
                right = str(edge.get("right"))
                distance = float(edge.get("distance", 0.0))
                verdict = str(edge.get("verdict"))
                html_lines.append(
                    "<li>"
                    f"{esc(_basename(left))} ↔ {esc(_basename(right))} "
                    f"({distance:.4f}, {esc(verdict)})"
                    "</li>"
                )
            html_lines.append("</ul>")
        else:
            html_lines.append("<div class='tiny muted'>No exact/near edges in this group.</div>")
        html_lines.append("<details class='tiny' style='margin-top:6px;'>")
        html_lines.append(f"<summary>No-match edges ({len(no_match_edges)})</summary>")
        if no_match_edges:
            html_lines.append("<ul class='edge-list'>")
            for edge in sorted(no_match_edges, key=lambda e: float(e.get("distance", 0.0))):
                left = str(edge.get("left"))
                right = str(edge.get("right"))
                distance = float(edge.get("distance", 0.0))
                html_lines.append(
                    "<li>"
                    f"{esc(_basename(left))} ↔ {esc(_basename(right))} "
                    f"({distance:.4f})"
                    "</li>"
                )
            html_lines.append("</ul>")
        else:
            html_lines.append("<div class='tiny muted'>No no-match edges.</div>")
        html_lines.append("</details>")
        html_lines.append("</details>")
        html_lines.extend(_trace_panel(group))
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

    if actionable_groups == 0:
        html_lines.append("<div class='empty'>No changes needed.</div>")
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
        "const showReadyEl = $('#showReady');"
        "const showNoOpsEl = $('#showNoOps');"
        "const visibleCountEl = $('#visibleCount');"
        "const expandAllBtn = $('#expandAll');"
        "const collapseAllBtn = $('#collapseAll');"
        "const settingsBtn = $('#toggleSettings');"
        "const settingsPanel = $('#settingsPanel');"
        "function computeStats(){"
        "const showReady = showReadyEl ? showReadyEl.checked : false;"
        "const showNoOps = showNoOpsEl ? showNoOpsEl.checked : false;"
        "const g = groups().filter(d => {"
        "const isNoOp = (d.getAttribute('data-no-op') || 'false') === 'true';"
        "const status = (d.getAttribute('data-status') || '').toLowerCase();"
        "const isReady = status === 'ready';"
        "return (showNoOps || !isNoOp) && (showReady || !isReady);"
        "});"
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
        "const showReady = showReadyEl ? showReadyEl.checked : false;"
        "const showNoOps = showNoOpsEl ? showNoOpsEl.checked : false;"
        "let visible = 0;"
        "for (const d of groups()){"
        "const hay = (d.getAttribute('data-search') || '').toLowerCase();"
        "const matchesSearch = !term || hay.includes(term);"
        "const hasQuarantine = (d.getAttribute('data-has-quarantine') || 'false') === 'true';"
        "const hasFailure = (d.getAttribute('data-has-failure') || 'false') === 'true';"
        "const typeHint = (d.getAttribute('data-type') || '').toLowerCase();"
        "const isNoOp = (d.getAttribute('data-no-op') || 'false') === 'true';"
        "const status = (d.getAttribute('data-status') || '').toLowerCase();"
        "const isReady = status === 'ready';"
        "let matchesFilter = true;"
        "if (mode === 'has-quarantine') matchesFilter = hasQuarantine;"
        "if (mode === 'metadata-only') matchesFilter = typeHint === 'metadata-only';"
        "if (mode === 'failed') matchesFilter = hasFailure;"
        "const show = matchesSearch && matchesFilter && (showNoOps || !isNoOp) && (showReady || !isReady);"
        "d.classList.toggle('hidden', !show);"
        "if (show) visible += 1;"
        "}"
        "visibleCountEl.textContent = `${visible} visible`;"
        "computeStats();"
        "}"
        "for (const btn of $$('button.copy[data-copy], button.copy[data-copy-text]')){"
        "btn.addEventListener('click', async () => {"
        "const explicit = btn.getAttribute('data-copy-text');"
        "const sel = btn.getAttribute('data-copy');"
        "const el = sel ? $(sel) : null;"
        "const text = explicit || (el ? el.textContent.trim() : '');"
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
        "for (const btn of $$('button.copy[data-expand]')){"
        "btn.addEventListener('click', () => {"
        "const id = btn.getAttribute('data-expand');"
        "const el = id ? document.getElementById(id) : null;"
        "if (!el) return;"
        "const expanded = el.classList.toggle('path-expanded');"
        "btn.textContent = expanded ? 'Collapse' : 'Expand';"
        "});"
        "}"
        "for (const btn of $$('button.trace-btn[data-trace-target]')){"
        "btn.addEventListener('click', (event) => {"
        "event.preventDefault();"
        "const id = btn.getAttribute('data-trace-target');"
        "const panel = id ? document.getElementById(id) : null;"
        "if (!panel) return;"
        "panel.open = !panel.open;"
        "panel.scrollIntoView({behavior: 'smooth', block: 'start'});"
        "});"
        "}"
        "expandAllBtn?.addEventListener('click', () => groups().forEach(d => d.open = true));"
        "collapseAllBtn?.addEventListener('click', () => groups().forEach(d => d.open = false));"
        "if (settingsBtn && settingsPanel){"
        "settingsBtn.addEventListener('click', () => {"
        "const hidden = settingsPanel.classList.toggle('hidden');"
        "settingsBtn.textContent = hidden ? 'Show settings' : 'Hide settings';"
        "});"
        "}"
        "searchEl?.addEventListener('input', applyFilters);"
        "filterEl?.addEventListener('change', applyFilters);"
        "showReadyEl?.addEventListener('change', applyFilters);"
        "showNoOpsEl?.addEventListener('change', applyFilters);"
        "computeStats();"
        "applyFilters();"
        "})();"
        "</script>"
    )
    html_lines.append("</body></html>")

    with open(ensure_long_path(output_html_path), "w", encoding="utf-8") as handle:
        handle.write("\n".join(html_lines))
    try:
        output_size = os.path.getsize(output_html_path)
    except OSError:
        output_size = None
    elapsed = time.monotonic() - start_time
    if output_size is not None:
        logger.info(
            "Preview HTML output: size=%d bytes groups=%d embedded_artwork=%d elapsed=%.2fs",
            output_size,
            len(plan.groups),
            embedded_artwork_count,
            elapsed,
        )
    else:
        logger.info(
            "Preview HTML output: groups=%d embedded_artwork=%d elapsed=%.2fs",
            len(plan.groups),
            embedded_artwork_count,
            elapsed,
        )
    return output_html_path
