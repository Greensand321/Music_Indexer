"""Helpers for mirroring a music library into Opus files."""
from __future__ import annotations

import base64
import html
import importlib.util
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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


ProgressCallback = Callable[[int, int, int, int, int, int], None]
LogCallback = Callable[[str], None]

AUDIO_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}


def mirror_library(
    source: str,
    destination: str,
    overwrite: bool,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
) -> dict[str, object]:
    total_files = 0
    counters = {"total": 0, "converted": 0, "copied": 0, "skipped": 0, "errors": 0}
    skipped_files: list[tuple[str, str]] = []
    error_files: list[tuple[str, str]] = []
    lock = threading.Lock()
    tasks: list[tuple[str, str, str]] = []

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
                    counters["skipped"] += 1
                    skipped_files.append((src_path, "Destination already exists."))
                    continue
                tasks.append(("convert", src_path, dest_path))
            else:
                dest_path = os.path.join(dest_root, filename)
                if os.path.exists(dest_path) and not overwrite:
                    counters["skipped"] += 1
                    skipped_files.append((src_path, "Destination already exists."))
                    continue
                tasks.append(("copy", src_path, dest_path))

    total_tasks = len(tasks)
    if progress_callback:
        progress_callback(
            total_tasks,
            counters["total"],
            counters["converted"],
            counters["copied"],
            counters["skipped"],
            counters["errors"],
        )

    def _process_task(task: tuple[str, str, str]) -> None:
        action, src_path, dest_path = task
        try:
            if action == "convert":
                result, error = convert_flac_to_opus(
                    src_path, dest_path, overwrite, log_callback=log_callback
                )
                with lock:
                    counters["total"] += 1
                    if result:
                        counters["converted"] += 1
                    else:
                        counters["errors"] += 1
                        error_files.append(
                            (src_path, error or "Conversion failed.")
                        )
            else:
                try:
                    shutil.copy2(src_path, dest_path)
                    with lock:
                        counters["total"] += 1
                        counters["copied"] += 1
                except OSError as exc:
                    with lock:
                        counters["total"] += 1
                        counters["errors"] += 1
                        error_files.append((src_path, f"Copy failed: {exc}"))
        except Exception as exc:  # pragma: no cover - safety net
            with lock:
                counters["total"] += 1
                counters["errors"] += 1
                error_files.append((src_path, f"Task failed: {exc}"))
        if progress_callback:
            with lock:
                progress_callback(
                    total_tasks,
                    counters["total"],
                    counters["converted"],
                    counters["copied"],
                    counters["skipped"],
                    counters["errors"],
                )

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        futures = [executor.submit(_process_task, task) for task in tasks]
        for future in futures:
            future.result()

    return {
        "total": total_files,
        "converted": counters["converted"],
        "copied": counters["copied"],
        "skipped": counters["skipped"],
        "errors": counters["errors"],
        "skipped_files": skipped_files,
        "error_files": error_files,
    }


def convert_flac_to_opus(
    source_path: str,
    dest_path: str,
    overwrite: bool,
    log_callback: LogCallback | None = None,
) -> tuple[bool, str | None]:
    tags, pictures, metadata_error = _load_flac_metadata(
        source_path, log_callback=log_callback
    )
    if tags is None:
        return False, metadata_error or "Metadata unavailable."
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
    except OSError as exc:
        return False, f"FFmpeg failed to start: {exc}"
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="ignore")[-500:]
        _log_message(
            log_callback,
            "Opus Library Mirror: conversion failed for "
            f"{strip_ext_prefix(source_path)}: {error}",
        )
        return False, f"FFmpeg error: {error}"

    _write_opus_metadata(dest_path, tags, pictures, log_callback=log_callback)

    return True, None


def _load_flac_metadata(
    source_path: str, log_callback: LogCallback | None = None
) -> tuple[Mapping[str, list[str]] | None, list["Picture"], str | None]:
    if FLAC is None:
        _log_message(
            log_callback,
            "Opus Library Mirror: mutagen unavailable for metadata transfer.",
        )
        return None, [], "Mutagen not available."
    try:
        flac = FLAC(ensure_long_path(source_path))
        tags = flac.tags or {}
        pictures = list(flac.pictures)
        return tags, pictures, None
    except Exception:
        _log_message(
            log_callback,
            "Opus Library Mirror: failed to read metadata from "
            f"{strip_ext_prefix(source_path)}.",
        )
        return None, [], "Failed to read FLAC metadata."


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


def write_mirror_report(report_path: str, summary: Mapping[str, object]) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    skipped_files = summary.get("skipped_files", [])
    error_files = summary.get("error_files", [])

    def format_rows(rows: Iterable[tuple[str, str]]) -> str:
        entries = []
        for path, reason in rows:
            display_path = strip_ext_prefix(path)
            entries.append(
                "<tr>"
                f"<td>{html.escape(display_path)}</td>"
                f"<td>{html.escape(reason)}</td>"
                "</tr>"
            )
        return "\n".join(entries)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Opus Library Mirror Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f1f1f; }}
    h1 {{ margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f2f2f2; }}
    .empty {{ color: #666; font-style: italic; }}
  </style>
</head>
<body>
  <h1>Opus Library Mirror Report</h1>
  <p>Generated: {html.escape(now)}</p>
  <h2>Summary</h2>
  <ul>
    <li>Total files processed: {summary.get("total", 0)}</li>
    <li>FLAC converted: {summary.get("converted", 0)}</li>
    <li>Other files copied: {summary.get("copied", 0)}</li>
    <li>Skipped audio files: {summary.get("skipped", 0)}</li>
    <li>Errors: {summary.get("errors", 0)}</li>
  </ul>
  <h2>Errors</h2>
  {"<table><thead><tr><th>File</th><th>Reason</th></tr></thead><tbody>"
   + format_rows(error_files) + "</tbody></table>" if error_files else "<p class=\"empty\">No errors.</p>"}
  <h2>Skipped Audio Files</h2>
  {"<table><thead><tr><th>File</th><th>Reason</th></tr></thead><tbody>"
   + format_rows(skipped_files) + "</tbody></table>" if skipped_files else "<p class=\"empty\">No skipped audio files.</p>"}
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html_content)
