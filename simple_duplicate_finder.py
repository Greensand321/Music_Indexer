import os
from typing import List, Tuple, Dict, Optional

from fingerprint_cache import get_fingerprint
from near_duplicate_detector import fingerprint_distance

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
EXT_PRIORITY = {".flac": 0, ".m4a": 1, ".aac": 1, ".mp3": 2, ".wav": 3, ".ogg": 4}
FP_PREFIX_LEN = 16


def _compute_fp(path: str) -> Tuple[Optional[int], Optional[str]]:
    try:
        import acoustid
        return acoustid.fingerprint_file(path)
    except Exception:
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
        rel = os.path.relpath(dirpath, root)
        parts = {p.lower() for p in rel.split(os.sep)}
        if {"not sorted", "playlists"} & parts:
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                paths.append(os.path.join(dirpath, fname))
    return paths


def find_duplicates(
    root: str,
    threshold: float = 0.03,
    db_path: Optional[str] = None,
    log_callback: Optional[callable] = None,
) -> List[Tuple[str, str]]:
    """Return list of duplicate pairs detected in ``root``."""
    if db_path is None:
        db_path = os.path.join(root, "Docs", ".simple_fps.db")

    if log_callback is None:
        def log_callback(msg: str) -> None:
            pass

    audio_paths = _walk_audio_files(root)
    file_data: List[Tuple[str, str]] = []
    for p in audio_paths:
        fp = get_fingerprint(p, db_path, _compute_fp)
        if fp:
            file_data.append((p, fp))
        else:
            log_callback(f"No fingerprint for {p}")

    groups: List[Dict[str, object]] = []
    prefix_map: Dict[str, List[Dict[str, object]]] = {}
    for path, fp in file_data:
        prefix = fp[:FP_PREFIX_LEN]
        cand_groups = prefix_map.get(prefix, [])
        placed = False
        for g in cand_groups:
            dist = fingerprint_distance(fp, g["fp"])
            if dist <= threshold:
                g["paths"].append(path)
                placed = True
                break
        if not placed:
            g = {"fp": fp, "paths": [path]}
            cand_groups.append(g)
            prefix_map[prefix] = cand_groups
            groups.append(g)

    duplicates: List[Tuple[str, str]] = []
    for g in groups:
        paths = g["paths"]
        if len(paths) <= 1:
            continue
        scored = sorted(paths, key=lambda p: _keep_score(p, EXT_PRIORITY), reverse=True)
        keep = scored[0]
        for dup in scored[1:]:
            log_callback(f"Duplicate: keep {keep} -> drop {dup}")
            duplicates.append((keep, dup))
    return duplicates

