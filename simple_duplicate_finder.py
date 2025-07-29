import os
import time
from typing import List, Tuple, Dict, Optional

from fingerprint_cache import get_fingerprint
from near_duplicate_detector import fingerprint_distance
import chromaprint_utils

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
EXT_PRIORITY = {".flac": 0, ".m4a": 1, ".aac": 1, ".mp3": 2, ".wav": 3, ".ogg": 4}
FP_PREFIX_LEN = 16

# enable verbose debug logging
verbose: bool = True


def _dlog(label: str, msg: str) -> None:
    """Print debug log message if ``verbose`` is True."""
    if not verbose:
        return
    ts = time.strftime("%H:%M:%S")
    _log(f"{ts} [{label}] {msg}")

# log callback used by _compute_fp; set inside find_duplicates
_log = print


def _compute_fp(path: str) -> Tuple[Optional[int], Optional[str]]:
    """Compute fingerprint for ``path`` using Chromaprint."""
    try:
        fp = chromaprint_utils.fingerprint_fpcalc(
            path,
            trim=True,
            start_sec=5.0,
            duration_sec=60.0,
        )
        _dlog("FP", f"computed fingerprint for {path}: {fp}")
        _dlog("FP", f"prefix={fp[:FP_PREFIX_LEN]}")
        return 0, fp
    except chromaprint_utils.FingerprintError as e:
        _log(f"! Fingerprint failed for {path}: {e}")
        return None, None
    except Exception as e:
        _log(f"! Unexpected fingerprint error for {path}: {e}")
        return None, None


def _keep_score(path: str, ext_priority: Dict[str, int]) -> float:
    ext = os.path.splitext(path)[1].lower()
    pri = ext_priority.get(ext, 99)
    ext_score = 1000.0 / (pri + 1)
    fname_score = len(os.path.splitext(os.path.basename(path))[0])
    return ext_score + fname_score


def _walk_audio_files(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        _dlog("WALK", f"enter {dirpath}")
        rel = os.path.relpath(dirpath, root)
        parts = {p.lower() for p in rel.split(os.sep)}
        if {"not sorted", "playlists"} & parts:
            _dlog("WALK", f"skip dir {dirpath}")
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                path = os.path.join(dirpath, fname)
                _dlog("WALK", f"match {path}")
                paths.append(path)
            else:
                _dlog("WALK", f"skip file {fname}")
    return paths


def find_duplicates(
    root: str,
    threshold: float = 0.03,
    db_path: Optional[str] = None,
    log_callback: Optional[callable] = None,
) -> Tuple[List[Tuple[str, str]], int, float, float]:
    """Return (duplicates, missing_count, min_dist, max_dist) for audio files in ``root``."""
    if db_path is None:
        db_path = os.path.join(root, "Docs", ".simple_fps.db")

    if log_callback is None:
        log_callback = print

    global _log
    _log = log_callback

    missing_fp = 0

    def compute(path: str) -> Tuple[Optional[int], Optional[str]]:
        nonlocal missing_fp
        duration, fp = _compute_fp(path)
        if fp is None:
            missing_fp += 1
        return duration, fp

    audio_paths = _walk_audio_files(root)
    file_data: List[Tuple[str, str]] = []
    for p in audio_paths:
        fp = get_fingerprint(p, db_path, compute, log_callback=log_callback)
        if fp:
            log_callback(f"\u2713 Fingerprinted {p}")
            _dlog("FP", f"prefix={fp[:FP_PREFIX_LEN]} value={fp}")
            file_data.append((p, fp))
            _dlog("GROUP", f"added file_data {p}")
        else:
            log_callback(f"\u2717 No fingerprint for {p}")

    groups: List[Dict[str, object]] = []
    prefix_map: Dict[str, List[Dict[str, object]]] = {}
    min_dist = float("inf")
    max_dist = 0.0
    for path, fp in file_data:
        prefix = fp[:FP_PREFIX_LEN]
        log_callback(f"[GROUP] path={path}, prefix={prefix}")
        cand_groups = prefix_map.get(prefix, [])
        _dlog("GROUP", f"file {path} -> prefix {prefix}")
        _dlog("GROUP", f"{len(cand_groups)} groups for prefix")
        placed = False
        for g in cand_groups:
            dist = fingerprint_distance(fp, g["fp"])
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
            _dlog("DIST", f"{path} vs {g['paths'][0]} dist={dist:.3f} threshold={threshold}")
            log_callback(
                f"[DIST] {path} \u2194 {g['paths'][0]} distance={dist:.4f} (thr={threshold:.4f})"
            )
            if dist <= threshold:
                g["paths"].append(path)
                _dlog("GROUP", f"added to existing group with {g['paths'][0]}")
                placed = True
                break
        if not placed:
            g = {"fp": fp, "paths": [path]}
            cand_groups.append(g)
            prefix_map[prefix] = cand_groups
            groups.append(g)
            _dlog("GROUP", f"new group for prefix {prefix}")

    duplicates: List[Tuple[str, str]] = []
    for g in groups:
        paths = g["paths"]
        if len(paths) <= 1:
            continue
        scored = sorted(paths, key=lambda p: _keep_score(p, EXT_PRIORITY), reverse=True)
        keep = scored[0]
        for dup in scored[1:]:
            log_callback(f"Duplicate: keep {keep} -> drop {dup}")
            _dlog(
                "RESULT",
                f"keep={keep} score={_keep_score(keep, EXT_PRIORITY):.2f} dup={dup}",
            )
            duplicates.append((keep, dup))
    if min_dist == float("inf"):
        min_dist = 0.0
    return duplicates, missing_fp, min_dist, max_dist

