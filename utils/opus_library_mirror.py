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
from utils.reserved_folders import DOCS, QUARANTINE, TRASH, prune_reserved_dirs

# See utils/audio_metadata_reader.py for why this probes by importing rather
# than via importlib.util.find_spec (which raises on a spec-less stub module).
try:
    from mutagen.flac import FLAC, Picture  # type: ignore
    from mutagen.oggopus import OggOpus  # type: ignore
    _MUTAGEN_AVAILABLE = True
except Exception:  # noqa: BLE001 - ImportError, or ValueError from a spec-less stub
    _MUTAGEN_AVAILABLE = False
    FLAC = None  # type: ignore
    Picture = None  # type: ignore
    OggOpus = None  # type: ignore


ProgressCallback = Callable[[int, int, int, int, int, int], None]
LogCallback = Callable[[str], None]

AUDIO_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}

# Reserved top-level folders that are not part of a portable copy of the
# music: duplicate losers, non-audio leftovers and AlphaDEX's own reports.
# (Playlists/, Not Sorted/ and Manual Review/ hold the user's own material and
# are mirrored.) Pass ``exclude_folders=()`` to mirror everything.
MIRROR_EXCLUDED_FOLDERS: tuple[str, ...] = (DOCS, TRASH, QUARANTINE)

_PARTIAL_MARKER = ".alphadex-partial"

# How often a running ffmpeg is checked for cancellation (seconds).
_POLL_INTERVAL = 0.2


class MirrorCancelled(Exception):
    """Raised inside a task when the mirror run was cancelled."""


def _partial_path(dest_path: str) -> str:
    """Temporary sibling of ``dest_path`` (same extension, so ffmpeg can infer
    the container). Output only gets its final name once it is complete, so an
    interrupted run never leaves a half-written or untagged file that a later
    run would skip as "already exists"."""
    folder, name = os.path.split(dest_path)
    stem, ext = os.path.splitext(name)
    return os.path.join(
        folder, f".{stem}{_PARTIAL_MARKER}-{os.getpid()}-{threading.get_ident()}{ext}"
    )


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def mirror_library(
    source: str,
    destination: str,
    overwrite: bool,
    progress_callback: ProgressCallback | None = None,
    log_callback: LogCallback | None = None,
    bitrate_kbps: int = 96,
    *,
    cancel_event: threading.Event | None = None,
    error_callback: LogCallback | None = None,
    exclude_folders: Iterable[str] = MIRROR_EXCLUDED_FOLDERS,
    max_workers: int | None = None,
) -> dict[str, object]:
    """Mirror ``source`` into ``destination``: FLAC → Opus, everything else copied.

    ``cancel_event``: when set, running ffmpeg processes are terminated, their
    partial output is removed, and queued files are not started; the summary
    then has ``cancelled=True``. ``error_callback`` receives per-file failures
    (defaults to ``log_callback``). Top-level ``exclude_folders`` of the source
    are not mirrored (see ``MIRROR_EXCLUDED_FOLDERS``).
    """
    report_error = error_callback or log_callback
    total_files = 0
    counters = {"total": 0, "converted": 0, "copied": 0, "skipped": 0, "errors": 0}
    skipped_files: list[tuple[str, str]] = []
    error_files: list[tuple[str, str]] = []
    excluded_folders: list[str] = []
    lock = threading.Lock()
    tasks: list[tuple[str, str, str, int]] = []
    exclude = tuple(exclude_folders)
    not_started = 0

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    for root, dirs, files in os.walk(source):
        if _cancelled():
            break
        if exclude:
            before = set(dirs)
            prune_reserved_dirs(root, dirs, source, exclude)
            excluded_folders.extend(sorted(before - set(dirs)))
        rel = os.path.relpath(root, source)
        dest_root = destination if rel == "." else os.path.join(destination, rel)
        os.makedirs(dest_root, exist_ok=True)

        for filename in files:
            if _PARTIAL_MARKER in filename:
                continue
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
                tasks.append(("convert", src_path, dest_path, bitrate_kbps))
            else:
                dest_path = os.path.join(dest_root, filename)
                if os.path.exists(dest_path) and not overwrite:
                    counters["skipped"] += 1
                    skipped_files.append((src_path, "Destination already exists."))
                    continue
                tasks.append(("copy", src_path, dest_path, 0))

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

    def _process_task(task: tuple[str, str, str, int]) -> None:
        nonlocal not_started
        action, src_path, dest_path, task_bitrate = task
        if _cancelled():
            with lock:
                not_started += 1
            return
        try:
            if action == "convert":
                result, error = convert_flac_to_opus(
                    src_path, dest_path, overwrite, log_callback=log_callback,
                    bitrate_kbps=task_bitrate, cancel_event=cancel_event,
                    error_callback=report_error,
                )
                with lock:
                    if not result and _cancelled():
                        not_started += 1
                        return
                    counters["total"] += 1
                    if result:
                        counters["converted"] += 1
                    else:
                        counters["errors"] += 1
                        error_files.append((src_path, error or "Conversion failed."))
            else:
                partial = _partial_path(dest_path)
                try:
                    shutil.copy2(ensure_long_path(src_path), ensure_long_path(partial))
                    os.replace(ensure_long_path(partial), ensure_long_path(dest_path))
                    with lock:
                        counters["total"] += 1
                        counters["copied"] += 1
                except OSError as exc:
                    _remove_quietly(ensure_long_path(partial))
                    _log_message(
                        report_error,
                        "Opus Library Mirror: copy failed for "
                        f"{strip_ext_prefix(src_path)}: {exc}",
                    )
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

    workers = max_workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
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
        "cancelled": _cancelled(),
        "not_started": not_started,
        "excluded_folders": excluded_folders,
    }


def _run_ffmpeg(cmd: list[str], cancel_event: threading.Event | None) -> tuple[int, bytes]:
    """Run ffmpeg, polling ``cancel_event``; on cancel terminate it and raise."""
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    while True:
        try:
            _out, err = proc.communicate(timeout=_POLL_INTERVAL)
            return proc.returncode, err or b""
        except subprocess.TimeoutExpired:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                raise MirrorCancelled() from None


def convert_flac_to_opus(
    source_path: str,
    dest_path: str,
    overwrite: bool,
    log_callback: LogCallback | None = None,
    bitrate_kbps: int = 96,
    *,
    cancel_event: threading.Event | None = None,
    error_callback: LogCallback | None = None,
) -> tuple[bool, str | None]:
    """Convert one FLAC to Opus, copying its tags and front cover.

    The output is encoded and tagged under a temporary name and renamed to
    ``dest_path`` only once complete. On cancel the ffmpeg process is
    terminated and the partial file removed; returns ``(False, "Cancelled.")``.
    """
    report_error = error_callback or log_callback
    if cancel_event is not None and cancel_event.is_set():
        return False, "Cancelled."
    if os.path.exists(dest_path) and not overwrite:
        return False, "Destination already exists."
    tags, pictures, metadata_error = _load_flac_metadata(
        source_path, log_callback=report_error
    )
    if tags is None:
        return False, metadata_error or "Metadata unavailable."
    partial = _partial_path(dest_path)
    cmd = [
        "ffmpeg",
        "-nostdin",
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
        f"{bitrate_kbps}k",
        # The temporary name is ours to replace; whether dest_path may be
        # overwritten was decided above.
        "-y",
        ensure_long_path(partial),
    ]

    try:
        returncode, stderr = _run_ffmpeg(cmd, cancel_event)
    except MirrorCancelled:
        _remove_quietly(ensure_long_path(partial))
        return False, "Cancelled."
    except OSError as exc:
        _remove_quietly(ensure_long_path(partial))
        _log_message(report_error, f"Opus Library Mirror: FFmpeg failed to start: {exc}")
        return False, f"FFmpeg failed to start: {exc}"
    if returncode != 0:
        _remove_quietly(ensure_long_path(partial))
        error = stderr.decode("utf-8", errors="ignore")[-500:]
        _log_message(
            report_error,
            "Opus Library Mirror: conversion failed for "
            f"{strip_ext_prefix(source_path)}: {error}",
        )
        return False, f"FFmpeg error: {error}"

    meta_error = _write_opus_metadata(partial, tags, pictures, log_callback=report_error)
    if meta_error:
        _remove_quietly(ensure_long_path(partial))
        return False, meta_error
    try:
        os.replace(ensure_long_path(partial), ensure_long_path(dest_path))
    except OSError as exc:
        _remove_quietly(ensure_long_path(partial))
        _log_message(
            report_error,
            f"Opus Library Mirror: could not finalise {strip_ext_prefix(dest_path)}: {exc}",
        )
        return False, f"Could not finalise output: {exc}"

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
) -> str | None:
    """Write tags + cover to ``dest_path``; return an error message or None."""
    if OggOpus is None:
        _log_message(
            log_callback,
            "Opus Library Mirror: mutagen unavailable for metadata transfer.",
        )
        return "Mutagen not available."

    try:
        audio = OggOpus(ensure_long_path(dest_path))
        if audio.tags is None:
            audio.add_tags()
        audio.tags.clear()
        normalized = _normalize_tags(tags)
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
        return f"Failed to write Opus metadata: {exc}"
    return None


def _normalize_tags(
    tags: Mapping[str, object], log_callback: LogCallback | None = None
) -> dict[str, list[str]]:
    """Convert a FLAC tag mapping into Vorbis-comment lists of str.

    Empty values are dropped. (This used to log a line for *every* tag of
    every file, which buried real failures in the activity log; ``log_callback``
    is accepted for backward compatibility and no longer used.)
    """
    normalized: dict[str, list[str]] = {}
    for key, value in tags.items():
        if value is None:
            continue
        if isinstance(value, bytes):
            normalized[key] = [value.decode("utf-8", errors="ignore")]
            continue
        if isinstance(value, str):
            normalized[key] = [value]
            continue
        if isinstance(value, Mapping):
            normalized[key] = [str(value)]
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
            continue
        normalized[key] = [str(value)]
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
        return None
    try:
        return base64.b64encode(selected.write()).decode("ascii")
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
    _table_head = "<table><thead><tr><th>File</th><th>Reason</th></tr></thead><tbody>"
    _table_tail = "</tbody></table>"
    _empty_p = '<p class="empty">'
    errors_html = (
        _table_head + format_rows(error_files) + _table_tail
        if error_files
        else _empty_p + "No errors.</p>"
    )
    skipped_html = (
        _table_head + format_rows(skipped_files) + _table_tail
        if skipped_files
        else _empty_p + "No skipped audio files.</p>"
    )
    status_items = ""
    if summary.get("cancelled"):
        status_items += (
            "<li><strong>Cancelled</strong> — "
            f"{summary.get('not_started', 0)} file(s) were not processed.</li>"
        )
    excluded = summary.get("excluded_folders") or []
    if excluded:
        status_items += (
            "<li>Reserved folders not mirrored: "
            f"{html.escape(', '.join(str(e) for e in excluded))}</li>"
        )
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
    {status_items}
  </ul>
  <h2>Errors</h2>
  {errors_html}
  <h2>Skipped Audio Files</h2>
  {skipped_html}
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html_content)
