"""Shared helpers for reading audio metadata and embedded cover art."""
from __future__ import annotations

import importlib.util
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple

from utils.path_helpers import ensure_long_path

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


def _tags_from_mp4_atoms(path: str) -> Dict[str, object]:
    if MP4 is None:
        return {}
    values = _blank_tags()
    mp4 = MP4(ensure_long_path(path))
    tags = mp4.tags or {}

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

        covr = tags.get("covr") if isinstance(tags, dict) else None
        if covr:
            for cover in covr:
                if cover:
                    payloads.append(bytes(cover))

        wmp = tags.get("WM/Picture") if isinstance(tags, dict) else None
        if wmp:
            for pic in wmp:
                data = getattr(pic, "value", None)
                if data:
                    payloads.append(data)

    return payloads


def read_metadata(
    path: str,
    *,
    include_cover: bool = True,
) -> Tuple[Dict[str, object], List[bytes], Optional[str], Optional[str]]:
    """Return (tags, cover_payloads, error, reader_hint) for ``path``."""
    tags = _blank_tags()
    cover_payloads: List[bytes] = []
    error: Optional[str] = None
    reader_hint: Optional[str] = None

    if MutagenFile is None:
        return tags, cover_payloads, "mutagen unavailable", None

    ext = os.path.splitext(path)[1].lower()

    try:
        audio_easy = MutagenFile(ensure_long_path(path), easy=True)
    except Exception as exc:  # pragma: no cover - depends on local files
        audio_easy = None
        error = f"read failed: {exc}"

    if audio_easy and getattr(audio_easy, "tags", None):
        tags = _merge_missing(tags, _tags_from_easy(audio_easy.tags))

    if ext in {".m4a", ".mp4"}:
        mp4_tags = {}
        try:
            mp4_tags = _tags_from_mp4_atoms(path)
        except Exception as exc:  # pragma: no cover - depends on local files
            if error is None:
                error = f"mp4 read failed: {exc}"
            mp4_tags = {}
        needs_atoms = any(
            not tags.get(key) and mp4_tags.get(key)
            for key in ("artist", "title", "album", "albumartist", "date", "tracknumber", "discnumber", "genre")
        )
        if needs_atoms:
            tags = _merge_missing(tags, mp4_tags)
            reader_hint = "mp4 atoms"
        else:
            reader_hint = "easy tags"
        logger.debug("M4A metadata reader used: %s for %s", reader_hint, path)

    if not tags.get("year"):
        tags["year"] = _parse_year(tags.get("date"))
    if not tags.get("track") and tags.get("tracknumber"):
        tags["track"] = _parse_int(tags.get("tracknumber"))
    if not tags.get("disc") and tags.get("discnumber"):
        tags["disc"] = _parse_int(tags.get("discnumber"))

    if include_cover:
        try:
            audio_full = MutagenFile(ensure_long_path(path))
        except Exception as exc:  # pragma: no cover - depends on local files
            audio_full = None
            if error is None:
                error = f"read failed: {exc}"
        cover_payloads = _extract_cover_payloads(audio_full)

    return tags, cover_payloads, error, reader_hint


def read_tags(path: str) -> Dict[str, object]:
    """Return just the tag dictionary for ``path``."""
    tags, _covers, _error, _reader = read_metadata(path, include_cover=False)
    return tags
