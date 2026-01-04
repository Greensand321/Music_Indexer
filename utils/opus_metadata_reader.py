"""Helpers for reading Opus metadata and embedded cover art."""
from __future__ import annotations

import base64
import importlib.util
import logging
import os
from typing import Dict, List, Optional, Tuple

from utils.path_helpers import ensure_long_path

logger = logging.getLogger(__name__)

_MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
if _MUTAGEN_AVAILABLE:
    from mutagen.flac import Picture  # type: ignore
    from mutagen.oggopus import OggOpus  # type: ignore
else:  # pragma: no cover - optional dependency
    Picture = None  # type: ignore
    OggOpus = None  # type: ignore


TAG_KEYS = (
    "artist",
    "albumartist",
    "title",
    "album",
    "date",
    "year",
    "track",
    "tracknumber",
    "disc",
    "discnumber",
    "genre",
    "compilation",
)

SIDECAR_ARTWORK_SUFFIXES = (".artwork", ".cover", ".jpg")


def _blank_tags() -> Dict[str, object]:
    return {key: None for key in TAG_KEYS}


def _first_value(value: object) -> object:
    if isinstance(value, list):
        return _first_value(value[0]) if value else None
    if isinstance(value, tuple):
        return _first_value(value[0]) if value else None
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="replace")
    return value


def _parse_int(value: object) -> int | None:
    try:
        return int(str(value).split("/")[0])
    except Exception:
        return None


def _parse_year(value: object) -> int | None:
    if value in (None, ""):
        return None
    text = str(value)
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return None


def _read_sidecar_artwork_bytes(path: str) -> bytes | None:
    for suffix in SIDECAR_ARTWORK_SUFFIXES:
        candidate = f"{path}{suffix}"
        if os.path.exists(candidate):
            try:
                with open(candidate, "rb") as handle:
                    return handle.read()
            except OSError:
                continue
    return None


def _get_tag(tags: Dict[str, object], key: str) -> object | None:
    lowered = key.lower()
    for tag_key, tag_value in tags.items():
        if tag_key.lower() == lowered:
            return tag_value
    return None


def _decode_picture_payload(payload: str) -> bytes | None:
    if Picture is None:
        return None
    try:
        raw = base64.b64decode(payload)
        picture = Picture(raw)
        return picture.data or None
    except Exception as exc:
        logger.debug("Failed to decode Opus picture payload: %s", exc)
        return None


def _decode_coverart_payload(payload: str) -> bytes | None:
    try:
        return base64.b64decode(payload)
    except Exception as exc:
        logger.debug("Failed to decode Opus coverart payload: %s", exc)
        return None


def _extract_cover_payloads(tags: Dict[str, object]) -> List[bytes]:
    payloads: List[bytes] = []

    block_pictures = _get_tag(tags, "metadata_block_picture")
    if block_pictures:
        for item in block_pictures if isinstance(block_pictures, list) else [block_pictures]:
            data = _decode_picture_payload(str(item))
            if data:
                payloads.append(data)

    coverart = _get_tag(tags, "coverart")
    if coverart:
        for item in coverart if isinstance(coverart, list) else [coverart]:
            data = _decode_coverart_payload(str(item))
            if data:
                payloads.append(data)

    return payloads


def read_opus_metadata(
    path: str,
) -> Tuple[Dict[str, object], List[bytes], Optional[str]]:
    """Return (tags, cover_payloads, error) for an Opus file."""
    tags = _blank_tags()
    cover_payloads: List[bytes] = []
    error: Optional[str] = None

    if OggOpus is None:
        return tags, cover_payloads, "mutagen unavailable"

    try:
        audio = OggOpus(ensure_long_path(path))
    except Exception as exc:  # pragma: no cover - depends on local files
        return tags, cover_payloads, f"read failed: {exc}"

    raw_tags = audio.tags or {}

    tags["title"] = _first_value(_get_tag(raw_tags, "title"))
    tags["artist"] = _first_value(_get_tag(raw_tags, "artist"))
    tags["album"] = _first_value(_get_tag(raw_tags, "album"))
    tags["albumartist"] = _first_value(_get_tag(raw_tags, "albumartist"))
    tags["date"] = _first_value(_get_tag(raw_tags, "date") or _get_tag(raw_tags, "year"))
    tags["genre"] = _first_value(_get_tag(raw_tags, "genre"))
    tags["compilation"] = _first_value(_get_tag(raw_tags, "compilation"))

    track_raw = _first_value(_get_tag(raw_tags, "tracknumber") or _get_tag(raw_tags, "track"))
    disc_raw = _first_value(_get_tag(raw_tags, "discnumber") or _get_tag(raw_tags, "disc"))
    tags["tracknumber"] = track_raw
    tags["discnumber"] = disc_raw
    tags["track"] = _parse_int(track_raw) if track_raw not in (None, "") else None
    tags["disc"] = _parse_int(disc_raw) if disc_raw not in (None, "") else None

    tags["year"] = _parse_year(tags.get("date"))

    cover_payloads = _extract_cover_payloads(raw_tags)
    if not cover_payloads:
        sidecar = _read_sidecar_artwork_bytes(path)
        if sidecar:
            cover_payloads = [sidecar]

    return tags, cover_payloads, error
