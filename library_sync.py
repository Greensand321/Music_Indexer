"""Helpers for comparing an existing library to an incoming folder."""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import music_indexer_api as idx
from fingerprint_cache import get_fingerprint
from near_duplicate_detector import fingerprint_distance
from config import NEAR_DUPLICATE_THRESHOLD, FORMAT_PRIORITY


def _compute_fp(path: str) -> tuple[int | None, str | None]:
    try:
        import acoustid
        return acoustid.fingerprint_file(path)
    except Exception:
        return None, None


def _scan_folder(folder: str, db_path: str) -> Dict[str, Dict[str, object]]:
    infos: Dict[str, Dict[str, object]] = {}
    for dirpath, _dirs, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in idx.SUPPORTED_EXTS:
                continue
            path = os.path.join(dirpath, fname)
            tags = idx.get_tags(path)
            fp = get_fingerprint(path, db_path, _compute_fp)
            bitrate = 0
            try:
                audio = idx.MutagenFile(path)
                if audio and getattr(audio, "info", None):
                    bitrate = getattr(audio.info, "bitrate", 0) or getattr(audio.info, "sample_rate", 0) or 0
            except Exception:
                pass
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
    return pr * br


def compare_libraries(
    library_folder: str,
    incoming_folder: str,
    db_path: str,
    threshold: float | None = None,
    fmt_priority: Dict[str, int] | None = None,
) -> Dict[str, object]:
    """Return classification of incoming files vs. an existing library."""
    threshold = threshold if threshold is not None else NEAR_DUPLICATE_THRESHOLD
    fmt_priority = fmt_priority or FORMAT_PRIORITY
    lib_infos = _scan_folder(library_folder, db_path)
    inc_infos = _scan_folder(incoming_folder, db_path)

    new: List[str] = []
    existing: List[Tuple[str, str]] = []
    improved: List[Tuple[str, str]] = []

    for inc_path, inc_info in inc_infos.items():
        best_match = None
        best_dist = 1.0
        for lib_path, lib_info in lib_infos.items():
            dist = fingerprint_distance(inc_info.get("fp"), lib_info.get("fp"))
            if dist == 0:
                best_match = lib_path
                best_dist = 0
                break
            if dist <= threshold and dist < best_dist:
                best_match = lib_path
                best_dist = dist
        if best_match is None:
            new.append(inc_path)
            continue
        inc_score = compute_quality_score(inc_info, fmt_priority)
        lib_score = compute_quality_score(lib_infos[best_match], fmt_priority)
        if inc_score > lib_score:
            improved.append((inc_path, best_match))
        else:
            existing.append((inc_path, best_match))

    return {"new": new, "existing": existing, "improved": improved}

def copy_new_tracks(new_paths: List[str], incoming_root: str, library_root: str) -> List[str]:
    """Copy each path in ``new_paths`` from ``incoming_root`` into ``library_root``.

    Returns the list of destination paths created.
    """
    import shutil

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
        shutil.copy2(src, final)
        dest_paths.append(final)
    return dest_paths


def replace_tracks(pairs: List[Tuple[str, str]], backup_dirname: str = "__backup__") -> List[str]:
    """Replace library files with higher quality versions.

    Each pair is ``(incoming, existing)``. The original file is moved into a
    ``backup_dirname`` folder alongside the existing file before the incoming
    file is moved into place.

    Returns the list of replaced library paths.
    """
    import shutil

    replaced = []
    for inc, lib in pairs:
        backup = os.path.join(os.path.dirname(lib), backup_dirname, os.path.basename(lib))
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        shutil.move(lib, backup)
        shutil.move(inc, lib)
        replaced.append(lib)
    return replaced
