"""Helpers for mirroring a music library into Opus files."""
from __future__ import annotations

import base64
import importlib.util
import os
import shutil
import subprocess
from typing import Callable, Iterable

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
    pictures = _load_flac_pictures(source_path)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        ensure_long_path(source_path),
        "-vn",
        "-c:a",
        "libopus",
        "-b:a",
        "96k",
        "-map_metadata",
        "0",
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

    if pictures:
        _write_opus_pictures(dest_path, pictures, log_callback=log_callback)

    return True


def _load_flac_pictures(source_path: str) -> list["Picture"]:
    if FLAC is None:
        return []
    try:
        return list(FLAC(ensure_long_path(source_path)).pictures)
    except Exception:
        return []


def _write_opus_pictures(
    dest_path: str,
    pictures: Iterable["Picture"],
    log_callback: LogCallback | None = None,
) -> None:
    if OggOpus is None:
        _log_message(
            log_callback,
            "Opus Library Mirror: mutagen unavailable for embedding artwork.",
        )
        return

    encoded = []
    for picture in pictures:
        try:
            encoded.append(base64.b64encode(picture.write()).decode("ascii"))
        except Exception:
            continue

    if not encoded:
        return

    try:
        audio = OggOpus(ensure_long_path(dest_path))
        if audio.tags is None:
            audio.add_tags()
        audio.tags["metadata_block_picture"] = encoded
        audio.save()
    except Exception as exc:
        _log_message(
            log_callback,
            f"Opus Library Mirror: failed to embed artwork in "
            f"{strip_ext_prefix(dest_path)}: {exc}",
        )


def _log_message(log_callback: LogCallback | None, message: str) -> None:
    if log_callback:
        log_callback(message)
