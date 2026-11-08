"""Helpers for mirroring a music library into Opus files.

:func:`mirror_library` writes only inside the destination ("the mirror") and
never touches the source library:

* the destination may not be the source, lie inside it, or contain it, and a
  destination folder that resolves (through a symlink or junction) outside
  the mirror or into the source is refused, so ``overwrite`` can never
  replace a file elsewhere;
* output gets its final name only once complete (see :func:`_partial_path`);
* ``.m3u``/``.m3u8`` playlists are copied with their relative entries for
  converted FLAC files pointed at the ``.opus`` (everything else in them,
  and every other file, is copied unchanged);
* an existing mirror file whose source changed since it was mirrored is
  *stale*: it is reported, and replaced only with ``update_changed=True``;
* files in the mirror that no source file maps to any more (the source was
  deleted or renamed) are *orphans*: they are listed in the summary, never
  deleted by a run. :func:`remove_mirror_orphans` deletes a reviewed list of
  them, and only inside the mirror.
"""
from __future__ import annotations

import base64
import hashlib
import html
import importlib.util
import os
import re
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

#: Playlists are copied with their entries for converted FLAC files pointed
#: at the ``.opus`` (a mirrored playlist otherwise lists files the mirror
#: does not have).
PLAYLIST_EXTS = {".m3u", ".m3u8"}

# Reserved top-level folders that are not part of a portable copy of the
# music: duplicate losers, non-audio leftovers and AlphaDEX's own reports.
# (Playlists/, Not Sorted/ and Manual Review/ hold the user's own material and
# are mirrored.) Pass ``exclude_folders=()`` to mirror everything.
MIRROR_EXCLUDED_FOLDERS: tuple[str, ...] = (DOCS, TRASH, QUARANTINE)

#: Where the GUI writes run reports, relative to the destination. Never an
#: orphan, whatever ``exclude_folders`` is.
MIRROR_REPORTS_DIR = os.path.join(DOCS, "opus_library_mirror_reports")

_PARTIAL_MARKER = ".alphadex-partial"
_PARTIAL_RE = re.compile(re.escape(_PARTIAL_MARKER) + r"-(\d+)-(\d+)(\.[^.]*)?$")

# Keep temporary names well inside the 255-byte (Linux) / 255-unit (Windows)
# file-name limit: a long track name plus the marker used to exceed it, so
# such tracks could never be mirrored.
_MAX_PARTIAL_NAME_BYTES = 240

# FAT/exFAT store modification times in 2-second steps.
_MTIME_SLACK_S = 2.0

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
    tag = f"{_PARTIAL_MARKER}-{os.getpid()}-{threading.get_ident()}{ext}"
    candidate = f".{stem}{tag}"
    if len(os.fsencode(candidate)) > _MAX_PARTIAL_NAME_BYTES:
        digest = hashlib.sha1(os.fsencode(stem)).hexdigest()[:16]
        candidate = f".{digest}{tag}"
    return os.path.join(folder, candidate)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _real(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_within(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or inside it (both already ``_real``)."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives
        return False


def check_mirror_folders(source: str, destination: str) -> None:
    """Raise ``ValueError`` unless ``destination`` is a safe mirror of ``source``.

    The two must be different folders and neither may contain the other: a
    mirror inside the library would be mirrored into itself on the next run
    (and scanned as music), and a library inside the mirror would look like
    a mirror full of orphans.
    """
    if not os.path.isdir(source):
        raise ValueError(f"Source folder not found: {source}")
    src_real, dst_real = _real(source), _real(destination)
    if src_real == dst_real:
        raise ValueError("The destination must be a different folder from the source library.")
    if _is_within(dst_real, src_real):
        raise ValueError("The destination cannot be inside the source library.")
    if _is_within(src_real, dst_real):
        raise ValueError(
            "The source library is inside the destination folder; choose a destination "
            "outside the library."
        )


def _claim_key(rel_path: str) -> str:
    """One key per destination file on any file system: case-folded, ``/``-separated."""
    return os.path.normpath(rel_path).replace(os.sep, "/").casefold()


def _is_stale(src_path: str, dest_path: str, action: str) -> bool:
    """Whether the mirrored ``dest_path`` predates a change to ``src_path``.

    Conversions are written after their source, so a source modified later is
    newer than its Opus. Copies keep the source's modification time
    (``shutil.copy2``), so a copy is stale when the source is newer or its
    size differs.
    """
    try:
        src = os.stat(ensure_long_path(src_path))
        dst = os.stat(ensure_long_path(dest_path))
    except OSError:
        return False
    if src.st_mtime > dst.st_mtime + _MTIME_SLACK_S:
        return True
    return action == "copy" and src.st_size != dst.st_size


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
    update_changed: bool = False,
) -> dict[str, object]:
    """Mirror ``source`` into ``destination``: FLAC → Opus, everything else copied
    (playlists with their converted FLAC entries pointed at the ``.opus``).

    ``overwrite`` replaces every existing destination file. Without it an
    existing file is kept; one whose source changed since it was mirrored is
    counted as ``stale`` and reported, and is replaced only when
    ``update_changed`` is true.

    ``cancel_event``: when set, running ffmpeg processes are terminated, their
    partial output is removed, and queued files are not started; the summary
    then has ``cancelled=True``. ``error_callback`` receives per-file failures
    (defaults to ``log_callback``). Top-level ``exclude_folders`` of the source
    are not mirrored (see ``MIRROR_EXCLUDED_FOLDERS``).

    After a complete run the mirror is checked for ``orphan_files``: files no
    source file maps to any more. They are listed, never deleted (see
    :func:`remove_mirror_orphans`). Temporary files left by an interrupted
    earlier run (ours only, by name) are removed.

    Raises ``ValueError`` for an unsafe pair of folders
    (:func:`check_mirror_folders`).
    """
    check_mirror_folders(source, destination)
    src_real, dst_real = _real(source), _real(destination)
    report_error = error_callback or log_callback
    total_files = 0
    counters = {
        "total": 0, "converted": 0, "copied": 0, "skipped": 0, "errors": 0,
        "stale": 0, "updated": 0,
    }
    skipped_files: list[tuple[str, str]] = []
    error_files: list[tuple[str, str]] = []
    excluded_folders: list[str] = []
    walk_errors: list[str] = []
    lock = threading.Lock()
    # (action, src, dest, bitrate, replace, is_update)
    tasks: list[tuple[str, str, str, int, bool, bool]] = []
    exclude = tuple(exclude_folders)
    not_started = 0
    # Every destination file a source maps to (claim key -> source rel path),
    # across the whole tree: names that differ only in case in *different*
    # folders ("Artist/" and "artist/") also land on one file on a
    # case-insensitive card.
    claimed: dict[str, str] = {}

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _walk_error(exc: OSError) -> None:
        walk_errors.append(getattr(exc, "filename", None) or str(exc))
        error_files.append((
            str(getattr(exc, "filename", "") or source),
            f"Could not read this folder: {exc}",
        ))
        counters["errors"] += 1

    def _folder_error(root: str, files: list[str], reason: str) -> None:
        for filename in files:
            if _PARTIAL_MARKER in filename:
                continue
            counters["errors"] += 1
            error_files.append((os.path.join(root, filename), reason))
        _log_message(report_error, f"Opus Library Mirror: {reason}")

    for root, dirs, files in os.walk(source, onerror=_walk_error):
        if _cancelled():
            break
        if exclude:
            before = set(dirs)
            prune_reserved_dirs(root, dirs, source, exclude)
            excluded_folders.extend(sorted(before - set(dirs)))
        dirs.sort()  # a deterministic walk, so clash decisions are too
        rel = os.path.relpath(root, source)
        dest_root = destination if rel == "." else os.path.join(destination, rel)
        real_dest_root = _real(dest_root)
        if not _is_within(real_dest_root, dst_real) or _is_within(real_dest_root, src_real):
            total_files += sum(1 for f in files if _PARTIAL_MARKER not in f)
            _folder_error(
                root, files,
                f"Destination folder {dest_root} leads outside the mirror (a link?); "
                "nothing was written there.",
            )
            continue
        try:
            os.makedirs(ensure_long_path(dest_root), exist_ok=True)
        except OSError as exc:
            # A name the destination drive cannot store (":" or "?" on
            # FAT/exFAT/NTFS), a file in the way, a full or read-only drive:
            # this folder fails, the rest of the mirror still runs.
            total_files += sum(1 for f in files if _PARTIAL_MARKER not in f)
            _folder_error(root, files, f"Could not create destination folder {dest_root}: {exc}")
            continue

        entries: list[tuple[str, str, str]] = []  # (filename, dest_name, action)
        for filename in files:
            if _PARTIAL_MARKER in filename:
                continue
            total_files += 1
            stem, ext = os.path.splitext(filename)
            if ext.lower() == ".flac":
                entries.append((filename, f"{stem}.opus", "convert"))
            elif ext.lower() in PLAYLIST_EXTS:
                entries.append((filename, filename, "playlist"))
            else:
                entries.append((filename, filename, "copy"))

        # Two source files can target one destination name: "Song.flac"
        # (converted to Song.opus) beside an existing "Song.opus", or names
        # that differ only in letter case (the mirror often lands on a
        # case-insensitive card or drive). Decide deterministically, before
        # any worker runs: a file copied as-is (the existing .opus) wins over
        # a conversion, then the first name in sorted order (and the first
        # folder in sorted walk order). The others are skipped and reported —
        # never left to whichever worker finishes last.
        for filename, dest_name, action in sorted(
            entries, key=lambda e: (e[2] == "convert", e[0])
        ):
            src_path = os.path.join(root, filename)
            src_rel = filename if rel == "." else os.path.join(rel, filename)
            key = _claim_key(dest_name if rel == "." else os.path.join(rel, dest_name))
            if key in claimed:
                counters["skipped"] += 1
                skipped_files.append((
                    src_path,
                    f"Same destination name ({dest_name}) as {claimed[key]}; "
                    f"{claimed[key]} was mirrored instead.",
                ))
                continue
            claimed[key] = src_rel
            dest_path = os.path.join(dest_root, dest_name)
            replace = overwrite
            is_update = False
            if os.path.exists(ensure_long_path(dest_path)) and not overwrite:
                if _is_stale(src_path, dest_path, action):
                    counters["stale"] += 1
                    if update_changed:
                        replace = is_update = True
                    else:
                        counters["skipped"] += 1
                        skipped_files.append((
                            src_path,
                            "Destination already exists but is older than the source "
                            "(the source changed after it was mirrored); not replaced.",
                        ))
                        continue
                else:
                    counters["skipped"] += 1
                    skipped_files.append((src_path, "Destination already exists."))
                    continue
            tasks.append((
                action, src_path, dest_path,
                bitrate_kbps if action == "convert" else 0, replace, is_update,
            ))

    walk_cancelled = _cancelled()

    # FFmpeg missing: say so once instead of failing to start it per file.
    if any(t[0] == "convert" for t in tasks) and shutil.which("ffmpeg") is None:
        message = "FFmpeg was not found on PATH; FLAC files cannot be converted."
        _log_message(report_error, f"Opus Library Mirror: {message}")
        kept = []
        for task in tasks:
            if task[0] == "convert":
                counters["errors"] += 1
                error_files.append((task[1], message))
            else:
                kept.append(task)
        tasks = kept

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

    def _process_task(task: tuple[str, str, str, int, bool, bool]) -> None:
        nonlocal not_started
        action, src_path, dest_path, task_bitrate, replace, is_update = task
        if _cancelled():
            with lock:
                not_started += 1
            return
        try:
            if action == "convert":
                result, error = convert_flac_to_opus(
                    src_path, dest_path, replace, log_callback=log_callback,
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
                        counters["updated"] += int(is_update)
                    else:
                        counters["errors"] += 1
                        error_files.append((src_path, error or "Conversion failed."))
            else:
                partial = _partial_path(dest_path)
                try:
                    if action == "playlist":
                        _write_mirrored_playlist(src_path, partial, source, claimed)
                    else:
                        shutil.copy2(ensure_long_path(src_path), ensure_long_path(partial))
                    os.replace(ensure_long_path(partial), ensure_long_path(dest_path))
                    with lock:
                        counters["total"] += 1
                        counters["copied"] += 1
                        counters["updated"] += int(is_update)
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

    orphan_files: list[str] = []
    partials_removed = 0
    if walk_cancelled or _cancelled():
        orphan_scan = "skipped: the run was cancelled"
    elif walk_errors:
        # A folder we could not read would make all of its mirror files
        # look orphaned.
        orphan_scan = "skipped: some source folders could not be read"
    else:
        orphan_files, partials_removed = _scan_mirror(destination, claimed, exclude)
        orphan_scan = "done"

    return {
        "total": total_files,
        "converted": counters["converted"],
        "copied": counters["copied"],
        "skipped": counters["skipped"],
        "errors": counters["errors"],
        "stale": counters["stale"],
        "updated": counters["updated"],
        "skipped_files": skipped_files,
        "error_files": error_files,
        "cancelled": _cancelled(),
        "not_started": not_started,
        "excluded_folders": excluded_folders,
        "orphan_files": orphan_files,
        "orphan_scan": orphan_scan,
        "partials_removed": partials_removed,
    }


def _resolve_playlist_entry(entry: bytes, playlist_dir: str, source: str) -> str | None:
    """The source path a relative playlist entry points at, or ``None``.

    Absolute entries, URLs and entries leading out of the source are left
    alone (``None``): where the mirror will be mounted is unknown.
    """
    text = None
    for encoding in ("utf-8", "cp1252"):
        try:
            text = entry.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None or text.lower().startswith("file:") or "://" in text:
        return None
    if os.sep == "/":
        text = text.replace("\\", "/")  # a playlist written on Windows
    if os.path.isabs(text) or os.path.splitdrive(text)[0]:
        return None
    target = os.path.normpath(os.path.join(playlist_dir, text))
    root = os.path.normcase(os.path.abspath(source))
    if not _is_within(os.path.normcase(os.path.abspath(target)), root):
        return None
    return target


def _write_mirrored_playlist(
    src_path: str, out_path: str, source: str, claimed: Mapping[str, str]
) -> int:
    """Copy playlist ``src_path`` to ``out_path``, pointing the entries of
    FLAC files the mirror converts at their ``.opus``.

    Works on bytes, so the file's encoding, line endings, comments and every
    other entry are kept exactly; only ``.flac`` -> ``.opus`` changes. The
    copy keeps the source's modification time (as ``shutil.copy2`` does), so
    change detection compares it like any copied file. Returns the number of
    entries rewritten.
    """
    with open(ensure_long_path(src_path), "rb") as handle:
        data = handle.read()
    playlist_dir = os.path.dirname(src_path)
    out: list[bytes] = []
    rewritten = 0
    for line in data.splitlines(keepends=True):
        body = line.rstrip(b"\r\n")
        entry = body.strip()
        if not entry or entry.startswith(b"#") or not entry.lower().endswith(b".flac"):
            out.append(line)
            continue
        target = _resolve_playlist_entry(entry, playlist_dir, source)
        if target is not None:
            rel = os.path.relpath(target, source)
            key = _claim_key(os.path.splitext(rel)[0] + ".opus")
            # Only when the mirror's file of that name *is* this FLAC's
            # conversion (not an .opus that won the name, say).
            if _claim_key(claimed.get(key, "")) == _claim_key(rel):
                cut = body.lower().rfind(b".flac")
                out.append(body[:cut] + b".opus" + body[cut + 5:] + line[len(body):])
                rewritten += 1
                continue
        out.append(line)
    with open(ensure_long_path(out_path), "wb") as handle:
        handle.write(b"".join(out))
    shutil.copystat(ensure_long_path(src_path), ensure_long_path(out_path))
    return rewritten


def _is_stale_partial(name: str) -> bool:
    """A temporary file of ours (by exact name pattern) from another process."""
    match = _PARTIAL_RE.search(name)
    return bool(match) and name.startswith(".") and int(match.group(1)) != os.getpid()


def _scan_mirror(
    destination: str, claimed: Mapping[str, str], exclude: Iterable[str]
) -> tuple[list[str], int]:
    """Return ``(orphan paths, temporary files removed)`` for the mirror.

    Read-only apart from removing our own temporary files left by an
    interrupted run. Never descends a link, never leaves the mirror, skips
    the excluded top-level folders and the report folder.
    """
    dst_real = _real(destination)
    reports_key = _claim_key(MIRROR_REPORTS_DIR)
    exclude = tuple(exclude)
    orphans: list[str] = []
    removed = 0
    for root, dirs, files in os.walk(destination):
        if not _is_within(_real(root), dst_real):
            dirs[:] = []
            continue
        rel_root = os.path.relpath(root, destination)
        if rel_root != "." and _claim_key(rel_root) == reports_key:
            dirs[:] = []
            continue
        if rel_root == "." and exclude:
            prune_reserved_dirs(root, dirs, destination, exclude)
        dirs[:] = sorted(
            d for d in dirs
            if not os.path.islink(os.path.join(root, d))
            and not _is_junction(os.path.join(root, d))
        )
        for name in sorted(files):
            path = os.path.join(root, name)
            if _PARTIAL_MARKER in name:
                if _is_stale_partial(name) and not os.path.islink(path):
                    try:
                        os.remove(ensure_long_path(path))
                        removed += 1
                    except OSError:
                        pass
                continue
            if os.path.islink(path):
                continue
            rel = name if rel_root == "." else os.path.join(rel_root, name)
            if _claim_key(rel) not in claimed:
                orphans.append(path)
    return orphans, removed


def _is_junction(path: str) -> bool:
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    try:
        return bool(isjunction(path)) if isjunction else False
    except OSError:
        return False


def remove_mirror_orphans(
    source: str,
    destination: str,
    orphan_paths: Iterable[str],
    *,
    log_path: str | None = None,
) -> dict[str, object]:
    """Delete reviewed orphan files from the mirror — and nothing else.

    ``orphan_paths`` is the (reviewed) ``orphan_files`` list of a
    :func:`mirror_library` summary. Each path is deleted only if it is still a
    regular file (not a link or folder) inside ``destination``, outside the
    source, outside the report folder, and no source file maps to it again.
    Emptied folders inside the mirror are removed too (never the mirror
    root). Every outcome is written to ``log_path`` (default: a timestamped
    file in ``<destination>/Docs/opus_library_mirror_reports/``).

    Returns ``{"removed": [...], "refused": [(path, reason)], "failed":
    [(path, error)], "log_path": ...}``.
    """
    check_mirror_folders(source, destination)
    dst_real, src_real = _real(destination), _real(source)
    reports_real = _real(os.path.join(destination, MIRROR_REPORTS_DIR))
    removed: list[str] = []
    refused: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    emptied: set[str] = set()

    for path in orphan_paths:
        path = os.path.abspath(path)
        if not os.path.lexists(path):
            refused.append((path, "already gone"))
            continue
        if os.path.islink(path) or not os.path.isfile(path):
            refused.append((path, "not a regular file"))
            continue
        parent_real = _real(os.path.dirname(path))
        target = os.path.join(parent_real, os.path.normcase(os.path.basename(path)))
        if not _is_within(parent_real, dst_real) or target == dst_real:
            refused.append((path, "outside the mirror"))
            continue
        if _is_within(target, src_real):
            refused.append((path, "inside the source library"))
            continue
        if _is_within(target, reports_real):
            refused.append((path, "a mirror report"))
            continue
        rel = os.path.relpath(target, dst_real)
        if _source_counterpart_exists(source, rel):
            refused.append((path, "its source file exists again"))
            continue
        try:
            os.remove(ensure_long_path(path))
        except OSError as exc:
            failed.append((path, str(exc)))
            continue
        removed.append(path)
        emptied.add(os.path.dirname(path))

    # Remove folders the deletions emptied, deepest first, inside the mirror.
    for folder in sorted(emptied, key=len, reverse=True):
        current = folder
        while _is_within(_real(current), dst_real) and _real(current) != dst_real:
            try:
                os.rmdir(ensure_long_path(current))
            except OSError:
                break
            current = os.path.dirname(current)

    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(
            destination, MIRROR_REPORTS_DIR, f"opus_library_mirror_orphans_removed_{stamp}.txt"
        )
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"# Opus mirror orphan removal {datetime.now().isoformat(timespec='seconds')}\n")
            handle.write(f"# source: {source}\n# mirror: {destination}\n")
            for path in removed:
                handle.write(f"removed\t{path}\n")
            for path, reason in refused:
                handle.write(f"kept\t{path}\t{reason}\n")
            for path, error in failed:
                handle.write(f"failed\t{path}\t{error}\n")
    except OSError:
        log_path = None
    return {"removed": removed, "refused": refused, "failed": failed, "log_path": log_path}


def _source_counterpart_exists(source: str, rel_dest: str) -> bool:
    """Whether a source file now maps to the mirror file ``rel_dest`` again."""
    rel_dir, name = os.path.split(rel_dest)
    folder = os.path.join(source, rel_dir)
    try:
        names = {n.casefold() for n in os.listdir(folder)}
    except OSError:
        return False
    stem, ext = os.path.splitext(name)
    wanted = {name.casefold()}
    if ext.lower() == ".opus":
        wanted.add(f"{stem}.flac".casefold())
    return bool(wanted & names)


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
    if summary.get("stale"):
        status_items += (
            f"<li>Mirror files older than their changed source: {summary.get('stale', 0)} "
            f"({summary.get('updated', 0)} updated)</li>"
        )
    if summary.get("partials_removed"):
        status_items += (
            "<li>Unfinished temporary files from an interrupted run removed: "
            f"{summary.get('partials_removed', 0)}</li>"
        )
    orphan_files = list(summary.get("orphan_files") or [])
    orphan_scan = str(summary.get("orphan_scan") or "")
    if orphan_files:
        orphans_html = (
            "<p>These files are in the mirror but no file in the source maps to them any "
            "more (the source was deleted, renamed or moved). They were not deleted.</p>"
            "<table><thead><tr><th>File</th></tr></thead><tbody>"
            + "\n".join(
                f"<tr><td>{html.escape(strip_ext_prefix(str(p)))}</td></tr>" for p in orphan_files
            )
            + "</tbody></table>"
        )
    elif orphan_scan.startswith("skipped"):
        orphans_html = _empty_p + f"Not checked ({html.escape(orphan_scan[len('skipped: '):])}).</p>"
    else:
        orphans_html = _empty_p + "No orphaned files.</p>"
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
    <li>Orphaned files in the mirror: {len(orphan_files)}</li>
    {status_items}
  </ul>
  <h2>Errors</h2>
  {errors_html}
  <h2>Skipped Audio Files</h2>
  {skipped_html}
  <h2>Orphaned Files in the Mirror</h2>
  {orphans_html}
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html_content)
