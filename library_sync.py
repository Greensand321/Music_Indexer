"""Helpers for comparing an existing library to an incoming folder."""
from __future__ import annotations

import os
import logging
from typing import Dict, List, Tuple, Callable, Optional

import music_indexer_api as idx
from fingerprint_cache import get_fingerprint
from near_duplicate_detector import fingerprint_distance
from config import NEAR_DUPLICATE_THRESHOLD, FORMAT_PRIORITY


debug: bool = False
_logger = logging.getLogger(__name__)
_logger.propagate = False
_logger.addHandler(logging.NullHandler())
_file_handler: Optional[logging.Handler] = None


def set_debug(enabled: bool, log_root: str | None = None) -> None:
    """Enable or disable verbose debug logging."""
    global debug, _file_handler
    debug = enabled
    if not enabled:
        if _file_handler:
            _logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
        return

    _logger.setLevel(logging.DEBUG)
    if log_root:
        os.makedirs(log_root, exist_ok=True)
        log_path = os.path.join(log_root, "library_sync_debug.log")
        if _file_handler:
            _logger.removeHandler(_file_handler)
            _file_handler.close()
        _file_handler = logging.FileHandler(log_path, encoding="utf-8")
        _file_handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(_file_handler)


def _dlog(msg: str, log_callback: Callable[[str], None] | None = None) -> None:
    if not debug:
        return
    if log_callback:
        log_callback(msg)
    _logger.debug(msg)


import chromaprint_utils


def _compute_fp(path: str) -> tuple[int | None, str | None]:
    try:
        fp = chromaprint_utils.fingerprint_fpcalc(path)
        return 0, fp
    except Exception:
        return None, None


def _scan_folder(
    folder: str,
    db_path: str,
    log_callback: Callable[[str], None] | None = None,
) -> Dict[str, Dict[str, object]]:
    _dlog(f"DEBUG: Scanning folder {folder}", log_callback)
    infos: Dict[str, Dict[str, object]] = {}
    for dirpath, _dirs, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in idx.SUPPORTED_EXTS:
                continue
            path = os.path.join(dirpath, fname)
            _dlog(f"DEBUG: Processing {path}", log_callback)
            tags = idx.get_tags(path)
            fp = get_fingerprint(path, db_path, _compute_fp)
            bitrate = 0
            try:
                audio = idx.MutagenFile(path)
                if audio and getattr(audio, "info", None):
                    bitrate = getattr(audio.info, "bitrate", 0) or getattr(audio.info, "sample_rate", 0) or 0
            except Exception:
                pass
            if fp:
                _dlog(f"DEBUG: Fingerprint prefix={fp[:16]!r}", log_callback)
            else:
                _dlog("DEBUG: No fingerprint", log_callback)
            _dlog(
                f"DEBUG: Tags artist={tags.get('artist')!r} title={tags.get('title')!r} album={tags.get('album')!r} bitrate={bitrate}",
                log_callback,
            )
            infos[path] = {
                **tags,
                "fp": fp,
                "ext": ext,
                "bitrate": bitrate,
            }
    return infos


def compute_quality_score(info: Dict[str, object], fmt_priority: Dict[str, int]) -> int:
    pr = fmt_priority.get(info.get("ext"), 0)
    br = int(info.get("bitrate") or 0)
    score = pr * br
    _dlog(
        f"DEBUG: Quality score ext={info.get('ext')} priority={pr} bitrate={br} score={score}",
        None,
    )
    return score


def compare_libraries(
    library_folder: str,
    incoming_folder: str,
    db_path: str,
    threshold: float | None = None,
    fmt_priority: Dict[str, int] | None = None,
    thresholds: Dict[str, float] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    """Return classification of incoming files vs. an existing library.

    Parameters
    ----------
    thresholds:
        Optional mapping of file extensions to fingerprint distance thresholds.
        The ``default`` key is used when an extension is not present.
    """
    if thresholds is None:
        if threshold is not None:
            thresholds = {"default": threshold}
        else:
            from config import DEFAULT_FP_THRESHOLDS

            thresholds = DEFAULT_FP_THRESHOLDS
    fmt_priority = fmt_priority or FORMAT_PRIORITY
    lib_infos = _scan_folder(library_folder, db_path, log_callback)
    inc_infos = _scan_folder(incoming_folder, db_path, log_callback)

    new: List[str] = []
    existing: List[Tuple[str, str]] = []
    improved: List[Tuple[str, str]] = []

    for inc_path, inc_info in inc_infos.items():
        _dlog(f"DEBUG: Comparing incoming {inc_path}", log_callback)
        best_match = None
        best_dist = 1.0
        for lib_path, lib_info in lib_infos.items():
            dist = fingerprint_distance(inc_info.get("fp"), lib_info.get("fp"))
            _dlog(
                f"DEBUG: Distance {inc_path} -> {lib_path}: {dist:.4f}",
                log_callback,
            )
            if dist == 0:
                best_match = lib_path
                best_dist = 0
                break
            if dist < best_dist:
                best_match = lib_path
                best_dist = dist
        if best_match is None:
            _dlog("DEBUG: No match found", log_callback)
            new.append(inc_path)
            continue
        best_ext = lib_infos[best_match]["ext"]
        thr = thresholds.get(best_ext, thresholds.get("default", 0.3))
        _dlog(
            f"DEBUG: Best distance {best_dist:.4f} threshold {thr:.4f}",
            log_callback,
        )
        if best_dist >= thr:
            _dlog("DEBUG: Above threshold, marked as new", log_callback)
            new.append(inc_path)
            continue
        inc_score = compute_quality_score(inc_info, fmt_priority)
        lib_score = compute_quality_score(lib_infos[best_match], fmt_priority)
        _dlog(
            f"DEBUG: Quality inc={inc_score} lib={lib_score}",
            log_callback,
        )
        if inc_score > lib_score:
            _dlog("DEBUG: Incoming is higher quality", log_callback)
            improved.append((inc_path, best_match))
        else:
            _dlog("DEBUG: Existing is higher quality", log_callback)
            existing.append((inc_path, best_match))

    _dlog(
        f"DEBUG: Compare complete new={len(new)} existing={len(existing)} improved={len(improved)}",
        log_callback,
    )
    return {"new": new, "existing": existing, "improved": improved}

def copy_new_tracks(
    new_paths: List[str],
    incoming_root: str,
    library_root: str,
    log_callback: Callable[[str], None] | None = None,
) -> List[str]:
    """Copy each path in ``new_paths`` from ``incoming_root`` into ``library_root``.

    Returns the list of destination paths created.
    """
    import shutil

    _dlog("DEBUG: Copying new tracks", log_callback)
    dest_paths = []
    for src in new_paths:
        rel = os.path.relpath(src, incoming_root)
        dest = os.path.join(library_root, rel)
        base, ext = os.path.splitext(dest)
        idx = 1
        final = dest
        while os.path.exists(final):
            final = f"{base} ({idx}){ext}"
            idx += 1
        os.makedirs(os.path.dirname(final), exist_ok=True)
        _dlog(f"DEBUG: Copy {src} -> {final}", log_callback)
        shutil.copy2(src, final)
        dest_paths.append(final)
    return dest_paths


def replace_tracks(
    pairs: List[Tuple[str, str]],
    backup_dirname: str = "__backup__",
    log_callback: Callable[[str], None] | None = None,
) -> List[str]:
    """Replace library files with higher quality versions.

    Each pair is ``(incoming, existing)``. The original file is moved into a
    ``backup_dirname`` folder alongside the existing file before the incoming
    file is moved into place.

    Returns the list of replaced library paths.
    """
    import shutil

    _dlog("DEBUG: Replacing tracks", log_callback)
    replaced = []
    for inc, lib in pairs:
        backup = os.path.join(os.path.dirname(lib), backup_dirname, os.path.basename(lib))
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        _dlog(f"DEBUG: Backup {lib} -> {backup}", log_callback)
        shutil.move(lib, backup)
        _dlog(f"DEBUG: Move {inc} -> {lib}", log_callback)
        shutil.move(inc, lib)
        replaced.append(lib)
    return replaced
