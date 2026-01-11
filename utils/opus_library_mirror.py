"""Helpers for mirroring a music library into Opus files."""
from __future__ import annotations

import base64
import importlib.util
import os
import shutil
import subprocess
from typing import Callable, Iterable, Mapping

from utils.path_helpers import ensure_long_path, strip_ext_prefix

_MUTAGEN_AVAILABLE = importlib.util.find_spec("mutagen") is not None
if _MUTAGEN_AVAILABLE:
    from mutagen.flac import FLAC, Picture  # type: ignore
    from mutagen.oggopus import OggOpus  # type: ignore
else:  # pragma: no cover - optional dependency
    FLAC = None  # type: ignore
    Picture = None  # type: ignore
    OggOpus = None  # type: ignore


ProgressCallback = Callable[[int, int, int, int, int], None]
LogCallback = Callable[[str], None]


def mirror_library(
    source: str,
    destination: str,
    overwrite: bool,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> dict[str, int]:
    total_files = 0
    converted = 0
    copied = 0
    skipped = 0
    errors = 0

    for root, _dirs, files in os.walk(source):
        rel = os.path.relpath(root, source)
        dest_root = destination if rel == "." else os.path.join(destination, rel)
        os.makedirs(dest_root, exist_ok=True)

        for filename in files:
            total_files += 1
            src_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".flac":
                dest_name = f"{os.path.splitext(filename)[0]}.opus"
                dest_path = os.path.join(dest_root, dest_name)
                if os.path.exists(dest_path) and not overwrite:
                    skipped += 1
                    continue
                result = convert_flac_to_opus(
                    src_path, dest_path, overwrite, log_callback=log_callback
                )
                if result:
                    converted += 1
                else:
                    errors += 1
            else:
                dest_path = os.path.join(dest_root, filename)
                if os.path.exists(dest_path) and not overwrite:
                    skipped += 1
                    continue
                try:
                    shutil.copy2(src_path, dest_path)
                    copied += 1
                except OSError:
                    errors += 1

            if progress_callback and total_files % 50 == 0:
                progress_callback(total_files, converted, copied, skipped, errors)

    return {
        "total": total_files,
        "converted": converted,
        "copied": copied,
        "skipped": skipped,
        "errors": errors,
    }


def convert_flac_to_opus(
    source_path: str,
    dest_path: str,
    overwrite: bool,
    log_callback: LogCallback | None = None,
) -> bool:
    tags, pictures = _load_flac_metadata(source_path, log_callback=log_callback)
    if tags is None:
        return False
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        ensure_long_path(source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "libopus",
        "-b:a",
        "96k",
    ]
    cmd.append("-y" if overwrite else "-n")
    cmd.append(ensure_long_path(dest_path))

    try:
        result = subprocess.run(
            cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except OSError:
        return False
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")[-500:]
        _log_message(
            log_callback,
            "Opus Library Mirror: conversion failed for "
            f"{strip_ext_prefix(source_path)}: {error}",
        )
        return False

    _write_opus_metadata(dest_path, tags, pictures, log_callback=log_callback)

    return True


def _load_flac_metadata(
    source_path: str, log_callback: LogCallback | None = None
) -> tuple[Mapping[str, list[str]] | None, list["Picture"]]:
    if FLAC is None:
        _log_message(
            log_callback,
            "Opus Library Mirror: mutagen unavailable for metadata transfer.",
        )
        return None, []
    try:
        flac = FLAC(ensure_long_path(source_path))
        tags = flac.tags or {}
        pictures = list(flac.pictures)
        return tags, pictures
    except Exception:
        _log_message(
            log_callback,
            "Opus Library Mirror: failed to read metadata from "
            f"{strip_ext_prefix(source_path)}.",
        )
        return None, []


def _write_opus_metadata(
    dest_path: str,
    tags: Mapping[str, list[str]],
    pictures: Iterable["Picture"],
    log_callback: LogCallback | None = None,
) -> None:
    if OggOpus is None:
        _log_message(
            log_callback,
            "Opus Library Mirror: mutagen unavailable for metadata transfer.",
        )
        return

    try:
        audio = OggOpus(ensure_long_path(dest_path))
        if audio.tags is None:
            audio.add_tags()
        audio.tags.clear()
        normalized = _normalize_tags(tags, log_callback=log_callback)
        for key, value in normalized.items():
            audio.tags[key] = value
        encoded = _encode_picture(pictures, log_callback=log_callback)
        if encoded:
            audio.tags["METADATA_BLOCK_PICTURE"] = [encoded]
        audio.save()
    except Exception as exc:
        _log_message(
            log_callback,
            f"Opus Library Mirror: failed to write metadata to "
            f"{strip_ext_prefix(dest_path)}: {exc}",
        )


def _normalize_tags(
    tags: Mapping[str, object], log_callback: LogCallback | None = None
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for key, value in tags.items():
        if value is None:
            _log_message(
                log_callback,
                f"Opus Library Mirror: skipped empty tag {key}.",
            )
            continue
        if isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
                normalized[key] = [value.decode("utf-8", errors="ignore")]
            else:
                normalized[key] = [value]
            _log_message(
                log_callback,
                f"Opus Library Mirror: normalized tag {key}.",
            )
            continue
        if isinstance(value, Mapping):
            normalized[key] = [str(value)]
            _log_message(
                log_callback,
                f"Opus Library Mirror: normalized tag {key}.",
            )
            continue
        if isinstance(value, Iterable):
            converted: list[str] = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, bytes):
                    converted.append(item.decode("utf-8", errors="ignore"))
                else:
                    converted.append(str(item))
            if converted:
                normalized[key] = converted
                _log_message(
                    log_callback,
                    f"Opus Library Mirror: normalized tag {key}.",
                )
            else:
                _log_message(
                    log_callback,
                    f"Opus Library Mirror: skipped empty tag {key}.",
                )
            continue
        normalized[key] = [str(value)]
        _log_message(
            log_callback,
            f"Opus Library Mirror: normalized tag {key}.",
        )
    return normalized


def _encode_picture(
    pictures: Iterable["Picture"], log_callback: LogCallback | None = None
) -> str | None:
    selected = None
    for picture in pictures:
        if picture.type == 3:
            selected = picture
            break
        if selected is None:
            selected = picture
    if selected is None:
        _log_message(
            log_callback,
            "Opus Library Mirror: no artwork found; skipping embed.",
        )
        return None
    try:
        encoded = base64.b64encode(selected.write()).decode("ascii")
        _log_message(
            log_callback,
            "Opus Library Mirror: embedded artwork.",
        )
        return encoded
    except Exception:
        _log_message(
            log_callback,
            "Opus Library Mirror: failed to encode artwork; skipping embed.",
        )
        return None


def _log_message(log_callback: LogCallback | None, message: str) -> None:
    if log_callback:
        log_callback(message)
