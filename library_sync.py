"""Helpers for comparing an existing library to an incoming folder."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import music_indexer_api as idx
import fingerprint_generator
from fingerprint_cache import _ensure_db
from near_duplicate_detector import fingerprint_distance
from config import FORMAT_PRIORITY, DEFAULT_FP_THRESHOLDS
from crash_watcher import record_event
from crash_logger import watcher

debug: bool = False
_logger = logging.getLogger(__name__)
_logger.propagate = False
_logger.addHandler(logging.NullHandler())
_file_handler: Optional[logging.Handler] = None

NEAR_MISS_MARGIN_RATIO = 0.25
NEAR_MISS_MARGIN_FLOOR = 0.02
SIGNATURE_TOKEN_COUNT = 8
SHORTLIST_FALLBACK = 50


def set_debug(enabled: bool, log_root: str | None = None) -> None:
    """Enable or disable verbose debug logging.

    If ``log_root`` is ``None`` the log will be written to the ``docs``
    directory next to this module. Existing logs are overwritten on each run.
    """
    global debug, _file_handler
    debug = enabled
    if not enabled:
        if _file_handler:
            _logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
        return

    _logger.setLevel(logging.DEBUG)
    if log_root is None:
        log_root = os.path.join(os.path.dirname(__file__), "docs")
    os.makedirs(log_root, exist_ok=True)
    log_path = os.path.join(log_root, "library_sync_debug.log")
    if _file_handler:
        _logger.removeHandler(_file_handler)
        _file_handler.close()
    _file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_file_handler)


def _dlog(msg: str, log_callback: Callable[[str], None] | None = None) -> None:
    if not debug:
        return
    if log_callback:
        log_callback(msg)
    _logger.debug(msg)


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _stable_track_id(path: str) -> str:
    norm = _normalize_path(path)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


class MatchStatus(str, Enum):
    NEW = "new"
    COLLISION = "collision"
    EXACT_MATCH = "exact_match"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class TrackRecord:
    """Normalized representation of a scanned track."""

    track_id: str
    path: str
    normalized_path: str
    ext: str
    bitrate: int | None
    size: int
    mtime: float
    fingerprint: str | None
    tags: Dict[str, object]
    duration: int | None = None

    def signature(self) -> str | None:
        return _fingerprint_signature(self.fingerprint)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "path": self.path,
            "normalized_path": self.normalized_path,
            "ext": self.ext,
            "bitrate": self.bitrate,
            "size": self.size,
            "mtime": self.mtime,
            "fingerprint": self.fingerprint,
            "duration": self.duration,
            "tags": dict(self.tags),
        }


@dataclass(order=True)
class CandidateDistance:
    """Distance info for a shortlist candidate."""

    distance: float
    track: TrackRecord = field(compare=False)

    def to_dict(self) -> Dict[str, object]:
        return {"track_id": self.track.track_id, "path": self.track.path, "distance": self.distance}


@dataclass
class MatchResult:
    """Matching outcome for a single incoming track."""

    incoming: TrackRecord
    existing: TrackRecord | None
    status: MatchStatus
    distance: float | None
    threshold_used: float
    near_miss_margin: float
    confidence: float
    candidates: List[CandidateDistance]
    quality_label: str | None
    incoming_score: int
    existing_score: int | None

    def to_dict(self) -> Dict[str, object]:
        return {
            "incoming": self.incoming.to_dict(),
            "existing": self.existing.to_dict() if self.existing else None,
            "status": self.status.value,
            "distance": self.distance,
            "threshold_used": self.threshold_used,
            "near_miss_margin": self.near_miss_margin,
            "confidence": self.confidence,
            "candidates": [c.to_dict() for c in self.candidates],
            "quality_label": self.quality_label,
            "incoming_score": self.incoming_score,
            "existing_score": self.existing_score,
        }


@dataclass
class PerformanceProfile:
    """Capture lightweight performance metrics for profiling runs."""

    cache_hits: int = 0
    fingerprints_computed: int = 0
    shortlist_sizes: List[int] = field(default_factory=list)
    signature_shortlists: int = 0
    extension_shortlists: int = 0
    fallback_shortlists: int = 0
    progress_updates: int = 0
    library_scan_duration: float = 0.0
    incoming_scan_duration: float = 0.0
    match_duration: float = 0.0

    def record_shortlist(self, size: int, source: str) -> None:
        self.shortlist_sizes.append(size)
        if source == "signature":
            self.signature_shortlists += 1
        elif source == "extension":
            self.extension_shortlists += 1
        else:
            self.fallback_shortlists += 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "cache_hits": self.cache_hits,
            "fingerprints_computed": self.fingerprints_computed,
            "shortlist_sizes": list(self.shortlist_sizes),
            "signature_shortlists": self.signature_shortlists,
            "extension_shortlists": self.extension_shortlists,
            "fallback_shortlists": self.fallback_shortlists,
            "progress_updates": self.progress_updates,
            "library_scan_duration": self.library_scan_duration,
            "incoming_scan_duration": self.incoming_scan_duration,
            "match_duration": self.match_duration,
        }


def _fingerprint_signature(fp: str | None) -> str | None:
    if not fp:
        return None
    tokens = fp.replace(",", " ").split()
    if not tokens:
        return None
    return " ".join(tokens[:SIGNATURE_TOKEN_COUNT])


def _select_threshold(ext: str, thresholds: Dict[str, float]) -> float:
    return float(thresholds.get(ext, thresholds.get("default", 0.3)))


def compute_quality_score(info: Dict[str, object], fmt_priority: Dict[str, int]) -> int:
    pr = fmt_priority.get(info.get("ext"), 0)
    bitrate_val = info.get("bitrate")
    br = int(bitrate_val) if bitrate_val not in (None, "", False) else 0
    score = pr * br if br else pr
    _dlog(
        f"DEBUG: Quality score ext={info.get('ext')} priority={pr} bitrate={br} score={score}",
        None,
    )
    return score


def _compute_confidence(distance: float | None, threshold: float, margin: float) -> float:
    if distance is None:
        return 0.0
    if distance <= 0:
        return 1.0
    limit = threshold + margin
    if limit <= 0:
        return 0.0
    score = max(0.0, min(1.0, 1 - (distance / limit)))
    return score


def _scan_folder(
    folder: str,
    db_path: str,
    on_record: Callable[[TrackRecord], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    profiler: PerformanceProfile | None = None,
) -> List[TrackRecord]:
    """Enumerate audio files, normalize paths, and attach cached fingerprints."""
    cancel_event = cancel_event or threading.Event()
    record_event(f"library_sync: scanning folder {folder}")
    _dlog(f"DEBUG: Scanning folder {folder}", log_callback)
    entries: List[Dict[str, object]] = []
    total_files = 0
    for dirpath, _dirs, files in os.walk(folder):
        if cancel_event.is_set():
            if log_callback:
                log_callback(json.dumps({"event": "scan_cancelled", "folder": folder, "stage": "walk"}))
            break
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in idx.SUPPORTED_EXTS:
                continue
            path = os.path.join(dirpath, fname)
            norm = _normalize_path(path)
            try:
                stat = os.stat(path)
                mtime = stat.st_mtime
                size = stat.st_size
            except OSError as e:
                _dlog(f"DEBUG: Skipping unreadable file {path}: {e}", log_callback)
                if log_callback:
                    log_callback(json.dumps({"event": "skip", "path": path, "reason": str(e)}))
                continue
            tags = idx.get_tags(path)
            bitrate = 0
            try:
                audio = idx.MutagenFile(path)
                if audio and getattr(audio, "info", None):
                    bitrate = getattr(audio.info, "bitrate", 0) or getattr(audio.info, "sample_rate", 0) or 0
            except Exception:
                bitrate = 0
            entries.append(
                {
                    "path": path,
                    "normalized_path": norm,
                    "ext": ext,
                    "bitrate": bitrate,
                    "tags": tags,
                    "mtime": mtime,
                    "size": size,
                }
            )
            total_files += 1
            if progress_callback:
                progress_callback(total_files, 0, path, "scan")
            if profiler:
                profiler.progress_updates += 1

    if not entries:
        return []

    conn = _ensure_db(db_path)
    to_compute: List[str] = []
    entry_map = {e["path"]: e for e in entries}

    for entry in entries:
        row = conn.execute(
            "SELECT mtime, size, duration, fingerprint FROM fingerprints WHERE path=?",
            (entry["path"],),
        ).fetchone()
        cached_mtime = row[0] if row else None
        cached_size = row[1] if row else None
        if row and abs(cached_mtime - entry["mtime"]) < 1e-6 and int(cached_size or 0) == int(entry["size"]):
            entry["duration"] = row[2]
            entry["fingerprint"] = row[3]
            _dlog(f"DEBUG: cache hit {entry['path']}", log_callback)
            if log_callback:
                log_callback(json.dumps({"event": "cache_hit", "path": entry["path"]}))
            if profiler:
                profiler.cache_hits += 1
        else:
            if row:
                msg = (
                    "DEBUG: cache invalidation "
                    f"path={entry['path']} stored_mtime={cached_mtime} stored_size={cached_size} "
                    f"new_mtime={entry['mtime']} new_size={entry['size']}"
                )
                if log_callback:
                    log_callback(msg)
                    log_callback(json.dumps({"event": "cache_miss", "path": entry["path"], "reason": "stale"}))
                _dlog(msg, log_callback)
            to_compute.append(entry["path"])

    def fp_progress(current: int, total: int, path: str, phase: str) -> None:
        if progress_callback:
            progress_callback(current, total, path, phase)
        if log_callback and phase.startswith("fp") and current and total:
            log_callback(
                json.dumps(
                    {
                        "event": "fingerprint_progress",
                        "path": path,
                        "current": current,
                        "total": total,
                        "phase": phase,
                    }
                )
            )

    if to_compute:
        if log_callback:
            log_callback(json.dumps({"event": "fingerprint_start", "pending": len(to_compute), "folder": folder}))
        for result in fingerprint_generator.compute_fingerprints_parallel(
            to_compute, db_path, log_callback, fp_progress, cancel_event=cancel_event
        ):
            path = result[0]
            duration = result[1] if len(result) > 2 else None
            fp = result[-1]
            if profiler:
                profiler.fingerprints_computed += 1
            entry = entry_map.get(path)
            if entry is None:
                continue
            entry["fingerprint"] = fp
            entry["duration"] = duration
            conn.execute(
                "INSERT OR REPLACE INTO fingerprints (path, mtime, size, duration, fingerprint) VALUES (?, ?, ?, ?, ?)",
                (path, entry["mtime"], entry["size"], duration, fp),
            )
            if cancel_event.is_set():
                _dlog(f"DEBUG: cancellation after fingerprinting {path}", log_callback)
                break
        conn.commit()
    conn.close()

    records: List[TrackRecord] = []
    for entry in entries:
        rec = TrackRecord(
            track_id=_stable_track_id(entry["normalized_path"]),
            path=entry["path"],
            normalized_path=entry["normalized_path"],
            ext=entry["ext"],
            bitrate=int(entry.get("bitrate") or 0),
            size=int(entry.get("size") or 0),
            mtime=float(entry.get("mtime") or 0.0),
            fingerprint=entry.get("fingerprint"),
            tags=entry.get("tags", {}),
            duration=entry.get("duration"),
        )
        records.append(rec)
        if on_record:
            on_record(rec)

    record_event(f"library_sync: finished scanning {folder}")
    return records


def _build_candidate_index(records: Iterable[TrackRecord]) -> Tuple[Dict[str, List[TrackRecord]], Dict[str, List[TrackRecord]]]:
    sig_index: Dict[str, List[TrackRecord]] = {}
    ext_index: Dict[str, List[TrackRecord]] = {}
    for rec in records:
        sig = rec.signature()
        if sig:
            sig_index.setdefault(sig, []).append(rec)
        ext_index.setdefault(rec.ext, []).append(rec)
    return sig_index, ext_index


def _shortlist_candidates(
    incoming: TrackRecord,
    sig_index: Dict[str, List[TrackRecord]],
    ext_index: Dict[str, List[TrackRecord]],
    fallback_pool: List[TrackRecord],
    log_callback: Callable[[str], None] | None = None,
    profiler: PerformanceProfile | None = None,
) -> List[TrackRecord]:
    sig = incoming.signature()
    candidates: List[TrackRecord] = []
    shortlist_source = "fallback"
    if sig and sig in sig_index:
        candidates.extend(sig_index[sig])
        shortlist_source = "signature"
    if not candidates:
        ext_candidates = ext_index.get(incoming.ext, [])[:SHORTLIST_FALLBACK]
        candidates.extend(ext_candidates)
        shortlist_source = "extension" if ext_candidates else shortlist_source
    if not candidates and fallback_pool:
        candidates.extend(fallback_pool[:SHORTLIST_FALLBACK])
    if profiler:
        profiler.record_shortlist(len(candidates), shortlist_source)
    _dlog(
        f"DEBUG: shortlist for {incoming.path} -> {len(candidates)} candidates",
        log_callback,
    )
    return candidates


def _match_tracks(
    incoming: List[TrackRecord],
    library: List[TrackRecord],
    thresholds: Dict[str, float],
    fmt_priority: Dict[str, int],
    log_callback: Callable[[str], None] | None = None,
    profiler: PerformanceProfile | None = None,
) -> List[MatchResult]:
    sig_index, ext_index = _build_candidate_index(library)
    fallback_pool = sorted([r for r in library if r.fingerprint], key=lambda r: r.normalized_path)
    results: List[MatchResult] = []
    start_time = time.perf_counter()
    for inc in incoming:
        threshold_used = _select_threshold(inc.ext, thresholds)
        near_miss_margin = max(NEAR_MISS_MARGIN_FLOOR, threshold_used * NEAR_MISS_MARGIN_RATIO)
        candidates = _shortlist_candidates(
            inc, sig_index, ext_index, fallback_pool, log_callback=log_callback, profiler=profiler
        )
        cand_dist: List[CandidateDistance] = []
        for cand in candidates:
            dist = fingerprint_distance(inc.fingerprint, cand.fingerprint)
            cand_dist.append(CandidateDistance(distance=dist, track=cand))
        cand_dist.sort()
        best = cand_dist[0] if cand_dist else None
        best_track = best.track if best else None
        best_distance = best.distance if best else None

        if best_distance is None or best_track is None:
            status = MatchStatus.NEW
        elif best_distance == 0:
            status = MatchStatus.EXACT_MATCH
        elif best_distance <= threshold_used:
            status = MatchStatus.COLLISION
        elif best_distance <= threshold_used + near_miss_margin:
            status = MatchStatus.LOW_CONFIDENCE
        else:
            status = MatchStatus.NEW

        inc_score = compute_quality_score(inc.__dict__, fmt_priority)
        existing_score = compute_quality_score(best_track.__dict__, fmt_priority) if best_track else None
        if best_track is None:
            quality_label = None
        elif inc_score > (existing_score or 0):
            quality_label = "Potential Upgrade"
        else:
            quality_label = "Keep Existing"

        confidence = _compute_confidence(best_distance, threshold_used, near_miss_margin)
        _dlog(
            f"DEBUG: match incoming={inc.path} best={best_track.path if best_track else 'none'} "
            f"dist={best_distance} thr={threshold_used} margin={near_miss_margin} status={status} confidence={confidence}",
            log_callback,
        )
        results.append(
            MatchResult(
                incoming=inc,
                existing=best_track,
                status=status,
                distance=best_distance,
                threshold_used=threshold_used,
                near_miss_margin=near_miss_margin,
                confidence=confidence,
                candidates=cand_dist,
                quality_label=quality_label,
                incoming_score=inc_score,
                existing_score=existing_score,
            )
        )
    if profiler:
        profiler.match_duration = time.perf_counter() - start_time
    return results


@watcher.traced
def compare_libraries(
    library_folder: str,
    incoming_folder: str,
    db_path: str,
    threshold: float | None = None,
    fmt_priority: Dict[str, int] | None = None,
    thresholds: Dict[str, float] | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    profiler: PerformanceProfile | None = None,
    include_profile: bool = False,
) -> Dict[str, object]:
    """Return classification of incoming files vs. an existing library.

    Scans are performed independently so the UI can remain responsive, and
    matching uses indexed shortlist candidates to avoid pairwise explosion.

    Parameters
    ----------
    thresholds:
        Optional mapping of file extensions to fingerprint distance thresholds.
        The ``default`` key is used when an extension is not present.
    """
    record_event(
        f"library_sync: comparing {library_folder} to {incoming_folder}"
    )
    cancel_event = cancel_event or threading.Event()
    if profiler is None and include_profile:
        profiler = PerformanceProfile()
    if thresholds is None:
        if threshold is not None:
            thresholds = {"default": threshold}
        else:
            thresholds = DEFAULT_FP_THRESHOLDS
    fmt_priority = fmt_priority or FORMAT_PRIORITY

    start = time.perf_counter()
    lib_records = _scan_folder(
        library_folder,
        db_path,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        profiler=profiler,
    )
    if profiler:
        profiler.library_scan_duration = time.perf_counter() - start
    start = time.perf_counter()
    inc_records = _scan_folder(
        incoming_folder,
        db_path,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        profiler=profiler,
    )
    if profiler:
        profiler.incoming_scan_duration = time.perf_counter() - start

    match_results = _match_tracks(
        inc_records, lib_records, thresholds, fmt_priority, log_callback=log_callback, profiler=profiler
    )

    new_paths: List[str] = []
    existing_matches: List[Tuple[str, str]] = []
    improved: List[Tuple[str, str]] = []

    for res in match_results:
        if res.status == MatchStatus.NEW:
            new_paths.append(res.incoming.path)
            continue
        if res.existing is None:
            new_paths.append(res.incoming.path)
            continue
        if res.quality_label == "Potential Upgrade":
            improved.append((res.incoming.path, res.existing.path))
        else:
            existing_matches.append((res.incoming.path, res.existing.path))

    result = {
        "existing": [r.path for r in lib_records],
        "new_tracks": [r.path for r in inc_records],
        "new": new_paths,
        "existing_matches": existing_matches,
        "improved": improved,
        "library_records": [r.to_dict() for r in lib_records],
        "incoming_records": [r.to_dict() for r in inc_records],
        "matches": [m.to_dict() for m in match_results],
        "partial": cancel_event.is_set(),
    }
    if include_profile and profiler:
        result["profile"] = profiler.to_dict()
    record_event(
        "library_sync: comparison complete "
        f"library_existing={len(lib_records)} incoming_tracks={len(inc_records)} "
        f"new={len(new_paths)} existing_matches={len(existing_matches)} improved={len(improved)}"
    )
    return result


def profile_compare_libraries(
    library_folder: str,
    incoming_folder: str,
    db_path: str,
    threshold: float | None = None,
    fmt_priority: Dict[str, int] | None = None,
    thresholds: Dict[str, float] | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Tuple[Dict[str, object], PerformanceProfile]:
    """Run a profiled comparison and return both results and metrics."""
    profile = PerformanceProfile()
    result = compare_libraries(
        library_folder,
        incoming_folder,
        db_path,
        threshold=threshold,
        fmt_priority=fmt_priority,
        thresholds=thresholds,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        profiler=profile,
        include_profile=True,
    )
    return result, profile


def copy_new_tracks(*_args, **_kwargs):
    """Deprecated: blocked per review-first redesign."""
    raise RuntimeError("File operations are disabled in the Library Sync review tool.")


def replace_tracks(*_args, **_kwargs):
    """Deprecated: blocked per review-first redesign."""
    raise RuntimeError("File operations are disabled in the Library Sync review tool.")
