"""Shared helpers for reading audio metadata and embedded cover art."""
from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Mapping
from typing import Dict, Iterable, List, Optional, Tuple

from utils.path_helpers import ensure_long_path
from utils.year_normalizer import normalize_year
from utils.opus_metadata_reader import read_opus_metadata

logger = logging.getLogger(__name__)

_MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
if _MUTAGEN_AVAILABLE:
    from mutagen import File as MutagenFile  # type: ignore
    from mutagen.mp4 import MP4  # type: ignore
else:  # pragma: no cover - optional dependency
    MutagenFile = None  # type: ignore
    MP4 = None  # type: ignore

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


def _normalize_text_value(value: object, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return value.decode(encoding)
            except UnicodeDecodeError:
                continue
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if not isinstance(value, (int, float)):
        logger.warning("Unexpected tag type for %s: %s", key, type(value).__name__)
    try:
        return str(value).strip() or None
    except Exception:
        logger.warning("Unsupported tag type for %s: %s", key, type(value).__name__)
        return None


def _normalize_text_tags(tags: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(tags)
    for key in ("artist", "albumartist", "title", "album", "genre", "year", "date"):
        normalized[key] = _normalize_text_value(tags.get(key), key)
    return normalized


def read_sidecar_artwork_bytes(path: str) -> bytes | None:
    for suffix in SIDECAR_ARTWORK_SUFFIXES:
        candidate = f"{path}{suffix}"
        if os.path.exists(candidate):
            try:
                with open(candidate, "rb") as handle:
                    return handle.read()
            except OSError:
                continue
    return None


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
    if hasattr(value, "text"):
        try:
            text_value = value.text
            if isinstance(text_value, (list, tuple)):
                return _first_value(text_value[0]) if text_value else None
            return str(text_value) if text_value is not None else None
        except Exception:
            return None
    return value


def _parse_int(value: object) -> int | None:
    try:
        return int(str(value).split("/")[0])
    except Exception:
        return None


def _tags_from_easy(tags: Dict[str, object]) -> Dict[str, object]:
    values = _blank_tags()
    values["artist"] = _first_value(tags.get("artist"))
    values["albumartist"] = _first_value(
        tags.get("albumartist") or tags.get("album artist")
    )
    values["title"] = _first_value(tags.get("title"))
    values["album"] = _first_value(tags.get("album"))
    values["date"] = _first_value(tags.get("date") or tags.get("year"))
    values["genre"] = _first_value(tags.get("genre"))
    values["compilation"] = _first_value(tags.get("compilation"))

    track_raw = _first_value(tags.get("tracknumber") or tags.get("track"))
    disc_raw = _first_value(tags.get("discnumber") or tags.get("disc"))
    values["tracknumber"] = track_raw
    values["discnumber"] = disc_raw
    values["track"] = _parse_int(track_raw) if track_raw not in (None, "") else None
    values["disc"] = _parse_int(disc_raw) if disc_raw not in (None, "") else None

    return values


def _format_part_of_set(value: object) -> Tuple[object, object]:
    if isinstance(value, tuple):
        number = value[0] if len(value) > 0 else None
        total = value[1] if len(value) > 1 else None
        raw = str(number) if number is not None else None
        if total:
            raw = f"{raw}/{total}" if raw else str(total)
        return raw, number
    return value, _parse_int(value)


def _tags_from_mp4_mapping(tags: Mapping[str, object]) -> Dict[str, object]:
    values = _blank_tags()

    def atom(key: str) -> object:
        return _first_value(tags.get(key))

    values["artist"] = atom("\xa9ART")
    values["title"] = atom("\xa9nam")
    values["album"] = atom("\xa9alb")
    values["albumartist"] = atom("aART")
    values["date"] = atom("\xa9day")
    values["genre"] = atom("\xa9gen")
    values["compilation"] = atom("cpil")

    track_raw, track_num = _format_part_of_set(atom("trkn"))
    disc_raw, disc_num = _format_part_of_set(atom("disk"))

    values["tracknumber"] = track_raw
    values["track"] = track_num
    values["discnumber"] = disc_raw
    values["disc"] = disc_num

    return values


def _tags_from_mp4_atoms(path: str) -> Dict[str, object]:
    if MP4 is None:
        return {}
    values = _blank_tags()
    mp4 = MP4(ensure_long_path(path))
    tags = mp4.tags or {}
    return _tags_from_mp4_mapping(tags)


def _tags_from_mutagen_tags(tags: Mapping[str, object]) -> Dict[str, object]:
    values = _blank_tags()

    def pick(keys: Iterable[str]) -> object:
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
    values["date"] = pick(["date", "year", "TDRC"])
    values["genre"] = pick(["genre", "TCON"])
    values["compilation"] = pick(["compilation", "TCMP"])

    track_raw = pick(["tracknumber", "track", "TRCK"])
    disc_raw = pick(["discnumber", "disc", "TPOS"])
    values["tracknumber"] = track_raw
    values["discnumber"] = disc_raw
    values["track"] = _parse_int(track_raw) if track_raw not in (None, "") else None
    values["disc"] = _parse_int(disc_raw) if disc_raw not in (None, "") else None

    return values


def _merge_missing(base: Dict[str, object], fallback: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base)
    for key, value in fallback.items():
        if merged.get(key) in (None, "", []):
            merged[key] = value
    return merged


def _extract_cover_payloads(audio) -> List[bytes]:
    payloads: List[bytes] = []
    if audio is None:
        return payloads

    if hasattr(audio, "pictures"):
        for pic in getattr(audio, "pictures") or []:
            data = getattr(pic, "data", None)
            if data:
                payloads.append(data)

    tags = getattr(audio, "tags", None)
    if tags:
        if hasattr(tags, "getall"):
            try:
                apics = tags.getall("APIC")
            except Exception:
                apics = []
            for apic in apics:
                data = getattr(apic, "data", None)
                if data:
                    payloads.append(data)
        else:
            for key in tags.keys():
                if str(key).startswith("APIC"):
                    data = getattr(tags.get(key), "data", None)
                    if data:
                        payloads.append(data)
                        break

        covr = tags.get("covr") if isinstance(tags, Mapping) else None
        if covr:
            for cover in covr:
                if cover:
                    payloads.append(bytes(cover))

        wmp = tags.get("WM/Picture") if isinstance(tags, Mapping) else None
        if wmp:
            for pic in wmp:
                data = getattr(pic, "value", None)
                if data:
                    payloads.append(data)

    return payloads


def read_metadata_from_mutagen(
    audio,
    path: str,
    *,
    include_cover: bool = True,
    reader_hint_suffix: str | None = None,
) -> Tuple[Dict[str, object], List[bytes], Optional[str], Optional[str]]:
    """Return (tags, cover_payloads, error, reader_hint) for a loaded mutagen object."""
    tags = _blank_tags()
    cover_payloads: List[bytes] = []
    error: Optional[str] = None
    reader_hint: Optional[str] = None
    ext = os.path.splitext(path)[1].lower()
    suffix = f" ({reader_hint_suffix})" if reader_hint_suffix else ""

    if audio and getattr(audio, "tags", None):
        if ext in {".m4a", ".mp4"}:
            mp4_tags = _tags_from_mp4_mapping(audio.tags)
            fallback_tags = _tags_from_mutagen_tags(audio.tags)
            if mp4_tags:
                tags = _merge_missing(mp4_tags, fallback_tags)
                reader_hint = f"mp4 atoms{suffix}"
            else:
                tags = fallback_tags
                reader_hint = f"mutagen tags{suffix}"
        else:
            tags = _tags_from_mutagen_tags(audio.tags)
            reader_hint = f"mutagen tags{suffix}"
    elif audio:
        reader_hint = f"mutagen tags{suffix}"

    tags["year"] = normalize_year(tags.get("year") or tags.get("date"))
    if not tags.get("track") and tags.get("tracknumber"):
        tags["track"] = _parse_int(tags.get("tracknumber"))
    if not tags.get("disc") and tags.get("discnumber"):
        tags["disc"] = _parse_int(tags.get("discnumber"))

    if include_cover:
        cover_payloads = _extract_cover_payloads(audio)
        if not cover_payloads:
            sidecar = read_sidecar_artwork_bytes(path)
            if sidecar:
                cover_payloads = [sidecar]

    return tags, cover_payloads, error, reader_hint


def read_metadata(
    path: str,
    *,
    include_cover: bool = True,
    audio=None,
) -> Tuple[Dict[str, object], List[bytes], Optional[str], Optional[str]]:
    """Return (tags, cover_payloads, error, reader_hint) for ``path``."""
    tags = _blank_tags()
    cover_payloads: List[bytes] = []
    error: Optional[str] = None
    reader_hint: Optional[str] = None

    ext = os.path.splitext(path)[1].lower()
    if ext == ".opus":
        tags, cover_payloads, error = read_opus_metadata(path)
        if not include_cover:
            cover_payloads = []
        reader_hint = "opus metadata"
        return tags, cover_payloads, error, reader_hint

    preloaded = audio is not None
    if audio is None:
        if MutagenFile is None:
            return tags, cover_payloads, "mutagen unavailable", None
        try:
            audio = MutagenFile(ensure_long_path(path))
        except Exception as exc:  # pragma: no cover - depends on local files
            audio = None
            error = f"read failed: {exc}"

    tags, cover_payloads, extra_error, reader_hint = read_metadata_from_mutagen(
        audio,
        path,
        include_cover=include_cover,
        reader_hint_suffix="preloaded" if preloaded else "file",
    )
    if error is None:
        error = extra_error

    if ext in {".m4a", ".mp4"}:
        logger.debug("M4A metadata reader used: %s for %s", reader_hint, path)

    return tags, cover_payloads, error, reader_hint


def read_tags(path: str) -> Dict[str, object]:
    """Return just the tag dictionary for ``path``."""
    tags, _covers, _error, _reader = read_metadata(path, include_cover=False)
    return _normalize_text_tags(tags)
