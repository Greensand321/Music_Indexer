import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Tuple, Dict, Optional, Iterator

from fingerprint_cache import get_fingerprint
from near_duplicate_detector import fingerprint_distance, _coarse_fingerprint_keys
import chromaprint_utils
from config import load_config, FP_DURATION_MS, FP_OFFSET_MS, FP_SILENCE_MIN_LEN_MS, FP_SILENCE_THRESHOLD_DB

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}
EXT_PRIORITY = {".flac": 0, ".m4a": 1, ".aac": 1, ".mp3": 2, ".wav": 3, ".ogg": 4}
FP_PREFIX_LEN = 16
PREFIX_THRESHOLD = 2


def prefix_distance(p1: str, p2: str) -> int:
    """Return Hamming distance between two prefix strings of equal length."""
    return sum(c1 != c2 for c1, c2 in zip(p1, p2))

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
        cfg = load_config()
        start_sec = float(cfg.get("fingerprint_offset_ms", FP_OFFSET_MS)) / 1000.0
        duration_sec = float(cfg.get("fingerprint_duration_ms", FP_DURATION_MS)) / 1000.0
        duration_sec = duration_sec if duration_sec > 0 else 120.0
        trim_silence = bool(cfg.get("trim_silence", False))
        silence_threshold_db = float(
            cfg.get("fingerprint_silence_threshold_db", FP_SILENCE_THRESHOLD_DB)
        )
        silence_min_len_ms = float(cfg.get("fingerprint_silence_min_len_ms", FP_SILENCE_MIN_LEN_MS))
        fp = chromaprint_utils.fingerprint_fpcalc(
            path,
            trim=trim_silence,
            start_sec=start_sec,
            duration_sec=duration_sec,
            threshold_db=silence_threshold_db,
            min_silence_duration=silence_min_len_ms / 1000.0,
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


def _walk_audio_files(
    root: str,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Iterator[str]:
    found = 0
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
                found += 1
                if progress_callback:
                    progress_callback("walk", found, 0, path)
                yield path
                if cancel_event and cancel_event.is_set():
                    return
            else:
                _dlog("WALK", f"skip file {fname}")
            if cancel_event and cancel_event.is_set():
                return
        if cancel_event and cancel_event.is_set():
            return


def find_duplicates(
    root: str,
    threshold: float = 0.03,
    prefix_len: int | None = FP_PREFIX_LEN,
    db_path: Optional[str] = None,
    log_callback: Optional[callable] = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    max_workers: int | None = None,
) -> Tuple[List[Tuple[str, str]], int]:
    """Return (duplicates, missing_count) for audio files in ``root``.

    ``prefix_len`` controls how many characters of the fingerprint are used to
    pre-group files before performing expensive distance comparisons. A value of
    ``0`` or ``None`` disables prefix grouping entirely, comparing every file
    against all existing groups.
    """
    if db_path is None:
        db_path = os.path.join(root, "Docs", ".simple_fps.db")

    if log_callback is None:
        log_callback = print

    global _log
    _log = log_callback

    missing_fp = 0
    missing_lock = threading.Lock()

    def compute(path: str) -> Tuple[Optional[int], Optional[str]]:
        nonlocal missing_fp
        duration, fp = _compute_fp(path)
        if fp is None:
            with missing_lock:
                missing_fp += 1
        return duration, fp

    total = 0  # Unknown upfront; progress callbacks receive 0 for total.
    in_flight: Dict[object, Tuple[int, str]] = {}
    cancelled = False

    groups: List[Dict[str, object]] = []
    prefix_map: Dict[str, List[Dict[str, object]]] = {}
    coarse_index: Dict[str, List[Dict[str, object]]] = {}
    ungated_groups: List[Dict[str, object]] = []
    use_prefix = prefix_len is not None and prefix_len > 0

    def handle_fingerprint(path: str, fp: str) -> None:
        prefix = fp[:prefix_len] if use_prefix else ""
        log_callback(f"[GROUP] path={path}, prefix={prefix}")
        coarse_keys = _coarse_fingerprint_keys(fp)
        candidate_groups: Dict[int, Dict[str, object]] = {}
        if coarse_keys:
            for key in coarse_keys:
                for g in coarse_index.get(key, []):
                    candidate_groups[id(g)] = g
        for g in ungated_groups:
            candidate_groups[id(g)] = g

        coarse_candidate_count = len(candidate_groups)
        cand_groups: List[Tuple[Dict[str, object], str]] = []
        if use_prefix:
            for g in candidate_groups.values():
                group_prefix = g.get("prefix", "")
                if prefix_distance(prefix, group_prefix) <= PREFIX_THRESHOLD:
                    cand_groups.append((g, group_prefix))
        else:
            for g in candidate_groups.values():
                cand_groups.append((g, g.get("prefix", "")))
        log_callback(
            f"[COARSE] {path} coarse candidates {coarse_candidate_count} -> "
            f"prefix candidates {len(cand_groups)}"
        )
        _dlog("GROUP", f"file {path} -> prefix {prefix}")
        _dlog("GROUP", f"{len(cand_groups)} groups for prefix")
        placed = False
        for g, key in cand_groups:
            dist = fingerprint_distance(fp, g["fp"])
            _dlog("DIST", f"{path} vs {g['paths'][0]} dist={dist:.3f} threshold={threshold}")
            log_callback(
                f"[DIST] {path} \u2194 {g['paths'][0]} distance={dist:.4f} (thr={threshold:.4f})"
            )
            if dist <= threshold:
                if key != prefix:
                    log_callback(
                        f"[FUZZY] {path} matched prefix {key} (dist={prefix_distance(prefix, key)})"
                    )
                g["paths"].append(path)
                _dlog("GROUP", f"added to existing group with {g['paths'][0]}")
                placed = True
                break
        if not placed:
            g = {"fp": fp, "paths": [path], "prefix": prefix}
            prefix_map.setdefault(prefix, []).append(g)
            groups.append(g)
            if coarse_keys:
                for key in coarse_keys:
                    coarse_index.setdefault(key, []).append(g)
            else:
                ungated_groups.append(g)
            _dlog("GROUP", f"new group for prefix {prefix}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        effective_workers = max_workers if max_workers is not None else executor._max_workers
        max_in_flight = max(1, effective_workers * 4)

        def submit_task(idx: int, path: str) -> None:
            if progress_callback:
                progress_callback("fp_start", idx, total, path)
            fut = executor.submit(
                get_fingerprint, path, db_path, compute, log_callback=log_callback
            )
            in_flight[fut] = (idx, path)

        for idx, p in enumerate(_walk_audio_files(root, progress_callback, cancel_event), 1):
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break
            submit_task(idx, p)
            while len(in_flight) >= max_in_flight:
                fut = next(as_completed(in_flight))
                idx, p = in_flight.pop(fut)
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break
                try:
                    fp = fut.result()
                except Exception as exc:
                    fp = None
                    log_callback(f"! Fingerprint task failed for {p}: {exc}")
                if progress_callback:
                    progress_callback("fp_end", idx, total, p)
                if fp:
                    log_callback(f"\u2713 Fingerprinted {p}")
                    show_pref = fp[:prefix_len] if (prefix_len and prefix_len > 0) else ""
                    _dlog("FP", f"prefix={show_pref} value={fp}")
                    handle_fingerprint(p, fp)
                else:
                    log_callback(f"\u2717 No fingerprint for {p}")
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break

        if not cancelled:
            for fut in as_completed(in_flight):
                idx, p = in_flight[fut]
                if cancel_event and cancel_event.is_set():
                    cancelled = True
                    break
                try:
                    fp = fut.result()
                except Exception as exc:
                    fp = None
                    log_callback(f"! Fingerprint task failed for {p}: {exc}")
                if progress_callback:
                    progress_callback("fp_end", idx, total, p)
                if fp:
                    log_callback(f"\u2713 Fingerprinted {p}")
                    show_pref = fp[:prefix_len] if (prefix_len and prefix_len > 0) else ""
                    _dlog("FP", f"prefix={show_pref} value={fp}")
                    handle_fingerprint(p, fp)
                else:
                    log_callback(f"\u2717 No fingerprint for {p}")

    if cancel_event and cancel_event.is_set() or cancelled:
        for fut in in_flight:
            fut.cancel()

    duplicates: List[Tuple[str, str]] = []
    for g in groups:
        if cancel_event and cancel_event.is_set():
            break
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
    if progress_callback:
        progress_callback("complete", len(duplicates), 0, "")
    return duplicates, missing_fp
