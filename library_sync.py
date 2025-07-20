"""Compare incoming tracks with existing library to suggest additions or upgrades."""

from __future__ import annotations
import os
import shutil
from typing import Dict, Tuple, Iterable

from music_indexer_api import SUPPORTED_EXTS, get_tags, MutagenFile
from fingerprint_cache import get_fingerprint
from near_duplicate_detector import fingerprint_distance
from playlist_generator import generate_playlists
from config import get_library_sync_config


def _quality_score(path: str, format_priority: Dict[str, int]) -> int:
    ext = os.path.splitext(path)[1].lower()
    pri = format_priority.get(ext, 0)
    try:
        audio = MutagenFile(path)
        rate = getattr(getattr(audio, "info", None), "bitrate", None)
        if not rate:
            rate = getattr(getattr(audio, "info", None), "sample_rate", None)
        rate = rate or 0
    except Exception:
        rate = 0
    return int(pri * rate)


def _scan_folder(folder: str, db_path: str, compute) -> Dict[str, dict]:
    infos: Dict[str, dict] = {}
    for dirpath, _, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            path = os.path.join(dirpath, fname)
            tags = get_tags(path)
            fp = get_fingerprint(path, db_path, compute)
            infos[path] = {
                **tags,
                "fp": fp,
                "ext": ext,
            }
    return infos


def compare_folders(library: str, incoming: str, db_path: str) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, str]]:
    """Return new tracks, existing mapping, and improvement mapping."""
    cfg = get_library_sync_config()
    thresh = cfg["near_duplicate_threshold"]
    fmt_pri = cfg["format_priority"]

    def _compute(path: str) -> Tuple[int | None, str | None]:
        try:
            import acoustid
            return acoustid.fingerprint_file(path)
        except Exception:
            return None, None

    lib_infos = _scan_folder(library, db_path, _compute)
    inc_infos = _scan_folder(incoming, db_path, _compute)

    for d in (lib_infos, inc_infos):
        for path, info in d.items():
            info["score"] = _quality_score(path, fmt_pri)

    by_fp = {info["fp"]: path for path, info in lib_infos.items() if info.get("fp")}

    new: Dict[str, dict] = {}
    existing: Dict[str, str] = {}
    improvements: Dict[str, str] = {}

    for ipath, info in inc_infos.items():
        fp = info.get("fp")
        if fp and fp in by_fp:
            lib_path = by_fp[fp]
            if info["score"] > lib_infos[lib_path]["score"]:
                improvements[ipath] = lib_path
            else:
                existing[ipath] = lib_path
            continue
        best = None
        best_path = None
        for lpath, linfo in lib_infos.items():
            if not fp or not linfo.get("fp"):
                continue
            dist = fingerprint_distance(fp, linfo["fp"])
            if dist <= thresh and (best is None or dist < best):
                best = dist
                best_path = lpath
        if best_path:
            if info["score"] > lib_infos[best_path]["score"]:
                improvements[ipath] = best_path
            else:
                existing[ipath] = best_path
        else:
            new[ipath] = info
    return new, existing, improvements


def copy_new_tracks(files: Iterable[str], library_root: str, incoming_root: str) -> Dict[str, str]:
    """Copy selected new files into the library, preserving relative paths."""
    moves: Dict[str, str] = {}
    for src in files:
        rel = os.path.relpath(src, incoming_root)
        dest = os.path.join(library_root, rel)
        base, ext = os.path.splitext(os.path.basename(dest))
        parent = os.path.dirname(dest)
        os.makedirs(parent, exist_ok=True)
        final = dest
        idx = 1
        while os.path.exists(final):
            final = os.path.join(parent, f"{base} ({idx}){ext}")
            idx += 1
        shutil.copy2(src, final)
        moves[src] = final
    if moves:
        generate_playlists(moves, library_root, overwrite=False, log_callback=lambda m: None)
    return moves


def replace_tracks(mapping: Dict[str, str], library_root: str) -> Dict[str, str]:
    """Replace existing files with higher quality versions."""
    moves: Dict[str, str] = {}
    for src, existing in mapping.items():
        backup_dir = os.path.join(os.path.dirname(existing), "__backup__")
        os.makedirs(backup_dir, exist_ok=True)
        backup = os.path.join(backup_dir, os.path.basename(existing))
        shutil.move(existing, backup)
        shutil.copy2(src, existing)
        moves[src] = existing
    if moves:
        generate_playlists(moves, library_root, overwrite=False, log_callback=lambda m: None)
    return moves
