"""Helpers for comparing an existing library to an incoming folder."""
from __future__ import annotations

import filecmp
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from library_sync_indexer_engine import (
    config as idx_config,
    fingerprint_cache as idx_fingerprint_cache,
    fingerprint_generator as idx_fingerprint_generator,
    music_indexer_api as idx,
    near_duplicate_detector as idx_near_duplicate_detector,
)
from playlist_generator import write_playlist

fingerprint_generator = idx_fingerprint_generator
fingerprint_distance = idx_near_duplicate_detector.fingerprint_distance
_ensure_db = idx_fingerprint_cache._ensure_db
FORMAT_PRIORITY = idx_config.FORMAT_PRIORITY
DEFAULT_FP_THRESHOLDS = idx_config.DEFAULT_FP_THRESHOLDS

from crash_watcher import record_event
from crash_logger import watcher
from indexer_control import IndexCancelled

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


class ExecutionDecision(str, Enum):
    """Execution outcome for a planned incoming track."""

    COPY = "COPY"
    REPLACE = "REPLACE"
    SKIP_DUPLICATE = "SKIP_DUPLICATE"
    SKIP_KEEP_EXISTING = "SKIP_KEEP_EXISTING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


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


@dataclass
class PlannedItem:
    """Planned movement decision for a single incoming track."""

    source: str
    destination: str
    decision: ExecutionDecision
    reason: str | None = None
    action: str = "move"

    def to_dict(self) -> Dict[str, str]:
        return {
            "source": self.source,
            "destination": self.destination,
            "decision": self.decision.value,
            "reason": self.reason or "",
            "action": self.action,
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


@dataclass
class LibrarySyncPlan:
    """Move/route plan computed by the indexer engine for Library Sync."""

    library_root: str
    incoming_root: str
    destination_root: str
    moves: Dict[str, str]
    tag_index: Dict[str, object]
    decision_log: List[str]
    copy_only: set[str] = field(default_factory=set)
    allowed_replacements: set[str] = field(default_factory=set)
    items: List[PlannedItem] = field(default_factory=list)
    transfer_mode: str = "move"

    def planned_moves(self) -> Dict[str, str]:
        """Return a copy of the planned move mapping for execution."""
        if self.items:
            return {item.source: item.destination for item in self.items}
        return dict(self.moves)

    def render_preview(self, output_html_path: str) -> str:
        """Generate a dry-run HTML preview from this plan."""
        heading = "Library Sync (Dry Run Preview)"
        preview_items = [item.to_dict() for item in self.items] if self.items else None
        idx.render_dry_run_html_from_plan(
            self.destination_root,
            output_html_path,
            self.moves,
            self.tag_index,
            plan_items=preview_items,
            heading_text=heading,
            title_prefix=heading,
        )
        return output_html_path


def _files_match(path_a: str, path_b: str) -> bool:
    try:
        return filecmp.cmp(path_a, path_b, shallow=False)
    except Exception:
        return False


def _compute_plan_items(
    moves: Dict[str, str],
    *,
    destination_root: str,
    copy_only: Iterable[str] | None = None,
    allow_all_replacements: bool = False,
    allowed_replacements: Iterable[str] | None = None,
    transfer_mode: str = "move",
) -> List[PlannedItem]:
    """Attach deterministic execution decisions to each planned move."""

    copy_only_set = set(copy_only or [])
    allowed_replacements_set = set(allowed_replacements or [])
    normalized_transfer_mode = "copy" if str(transfer_mode).lower() == "copy" else "move"
    items: List[PlannedItem] = []

    for src, dst in sorted(moves.items(), key=lambda item: item[1]):
        action = "copy" if src in copy_only_set else normalized_transfer_mode
        dest_exists = os.path.exists(dst)
        replacement_allowed = allow_all_replacements or dst in allowed_replacements_set

        if dest_exists and not replacement_allowed:
            if _files_match(src, dst):
                decision = ExecutionDecision.SKIP_DUPLICATE
                reason = "Exact duplicate already in library"
            else:
                decision = ExecutionDecision.SKIP_KEEP_EXISTING
                reason = "Destination exists; keeping library copy"
        elif dest_exists and replacement_allowed:
            decision = ExecutionDecision.REPLACE
            reason = "Replacing existing file"
        else:
            decision = ExecutionDecision.COPY
            reason = "Plan will transfer file"

        items.append(
            PlannedItem(
                source=src,
                destination=dst,
                decision=decision,
                reason=reason,
                action=action,
            )
        )

    return items


def compute_library_sync_plan(
    library_root: str,
    incoming_folder: str,
    *,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    flush_cache: bool = False,
    max_workers: int | None = None,
    cancel_event: threading.Event | None = None,
    transfer_mode: str = "move",
) -> LibrarySyncPlan:
    """Compute a deterministic move/route plan for Library Sync."""
    cancel_event = cancel_event or threading.Event()
    if log_callback is None:
        def log_callback(msg: str) -> None:
            _dlog(msg)
    if progress_callback is None:
        def progress_callback(_c: int, _t: int, _m: str, _p: str) -> None:
            pass

    def _cancellable_progress(current: int, total: int, message: str, phase: str) -> None:
        if cancel_event.is_set():
            raise IndexCancelled()
        progress_callback(current, total, message, phase)

    def _cancellable_log(message: str) -> None:
        if cancel_event.is_set():
            raise IndexCancelled()
        log_callback(message)

    if cancel_event.is_set():
        raise IndexCancelled()

    moves, tag_index, decision_log = idx.compute_moves_and_tag_index(
        incoming_folder,
        _cancellable_log,
        _cancellable_progress,
        dry_run=True,
        enable_phase_c=False,
        flush_cache=flush_cache,
        max_workers=max_workers,
        coord=None,
    )

    destination_root = os.path.join(library_root, "Music")
    if not os.path.isdir(destination_root):
        destination_root = library_root

    def _remap_path(path: str) -> str:
        rel = os.path.relpath(path, incoming_folder)
        return _normalize_path(os.path.join(destination_root, rel))

    remapped_moves = {_normalize_path(src): _remap_path(dst) for src, dst in moves.items()}
    remapped_tag_index = {_remap_path(dst): data for dst, data in tag_index.items()}

    plan_items = _compute_plan_items(
        remapped_moves,
        destination_root=_normalize_path(destination_root),
        copy_only=None,
        allow_all_replacements=False,
        allowed_replacements=None,
        transfer_mode=transfer_mode,
    )

    return LibrarySyncPlan(
        library_root=_normalize_path(library_root),
        incoming_root=_normalize_path(incoming_folder),
        destination_root=_normalize_path(destination_root),
        moves=remapped_moves,
        tag_index=remapped_tag_index,
        decision_log=decision_log,
        items=plan_items,
        transfer_mode="copy" if str(transfer_mode).lower() == "copy" else "move",
    )


def build_library_sync_preview(
    library_root: str,
    incoming_folder: str,
    output_html_path: str,
    *,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    flush_cache: bool = False,
    max_workers: int | None = None,
    cancel_event: threading.Event | None = None,
    transfer_mode: str = "move",
) -> LibrarySyncPlan:
    """Compute a Library Sync plan and write the dry-run preview."""
    plan = compute_library_sync_plan(
        library_root,
        incoming_folder,
        log_callback=log_callback,
        progress_callback=progress_callback,
        flush_cache=flush_cache,
        max_workers=max_workers,
        cancel_event=cancel_event,
        transfer_mode=transfer_mode,
    )
    plan.render_preview(output_html_path)
    return plan


def _ensure_docs_dir(root: str) -> str:
    docs_dir = os.path.join(root, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    return docs_dir


def _render_execution_report(
    plan: LibrarySyncPlan,
    executed_moves: Dict[str, str],
    skipped: List[Dict[str, str]],
    failed: List[Dict[str, str]],
    review_required: List[Dict[str, str]],
    output_html_path: str,
    heading: str,
    dry_run: bool,
) -> str:
    """Write an executed-plan HTML using the indexer preview style."""
    idx.render_dry_run_html_from_plan(
        plan.destination_root,
        output_html_path,
        executed_moves,
        plan.tag_index,
        heading_text=heading,
        title_prefix=heading,
    )

    sections: List[str] = []
    if dry_run:
        sections.append("<p><strong>Dry run:</strong> no files were modified.</p>")

    def _format_section(title: str, items: List[Dict[str, str]]) -> None:
        if not items:
            return
        sections.append(f"<h3>{idx.sanitize(title)}</h3>")
        sections.append("<ul>")
        for item in items:
            src = idx.sanitize(os.path.basename(item.get("source", "")))
            dst = idx.sanitize(os.path.basename(item.get("destination", "")))
            reason = idx.sanitize(item.get("reason", ""))
            line = f"<li><code>{src}</code> → <code>{dst}</code>"
            if reason:
                line += f" — {reason}"
            line += "</li>"
            sections.append(line)
        sections.append("</ul>")

    _format_section("Review Required", review_required)
    _format_section("Skipped", skipped)
    _format_section("Failed", failed)

    if sections:
        extras = "\n".join(sections)
        try:
            with open(output_html_path, "r+", encoding="utf-8") as f:
                content = f.read()
                marker = "</body>"
                if marker in content:
                    content = content.replace(marker, extras + "\n" + marker, 1)
                else:
                    content += extras
                f.seek(0)
                f.write(content)
                f.truncate()
        except Exception:
            pass
    return output_html_path


def execute_plan(
    plan: LibrarySyncPlan,
    dry_run: bool = False,
    *,
    allow_replacements: bool | Iterable[str] | None = None,
    audit_path: str | None = None,
    executed_report_path: str | None = None,
    playlist_path: str | None = None,
    create_playlist: bool = True,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Dict[str, object]:
    """Execute a Library Sync plan with auditing and HTML reporting.

    The execution routine performs only the operations expressed by the plan,
    capturing a JSON audit trail and an executed HTML report. Existing files
    are only replaced when explicitly allowed (via ``allow_replacements`` or
    ``plan.allowed_replacements``). Backups are written for any destructive
    action before the destination is overwritten.
    """

    log_callback = log_callback or (lambda _msg: None)
    progress_callback = progress_callback or (lambda _c, _t, _p, _ph: None)
    cancel_event = cancel_event or threading.Event()

    docs_dir = _ensure_docs_dir(plan.library_root)
    audit_path = audit_path or os.path.join(docs_dir, "LibrarySyncAudit.json")
    executed_report_path = executed_report_path or os.path.join(docs_dir, "LibrarySyncExecuted.html")
    playlist_root = os.path.join(plan.library_root, "Playlists")
    os.makedirs(playlist_root, exist_ok=True)
    if playlist_path is None:
        playlist_stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        playlist_path = os.path.join(playlist_root, f"LibrarySync_Added_{playlist_stamp}.m3u8")
    backup_root = os.path.join(docs_dir, "Backups", "LibrarySync")
    os.makedirs(backup_root, exist_ok=True)

    allow_all_replacements = allow_replacements is True
    allowed_replacements = set(getattr(plan, "allowed_replacements", set()) or [])
    if allow_replacements not in (None, False, True):
        allowed_replacements.update(allow_replacements)  # type: ignore[arg-type]

    copy_only = set(getattr(plan, "copy_only", set()) or [])

    plan_items = _compute_plan_items(
        plan.planned_moves(),
        destination_root=plan.destination_root,
        copy_only=copy_only,
        allow_all_replacements=allow_all_replacements,
        allowed_replacements=allowed_replacements,
        transfer_mode=getattr(plan, "transfer_mode", "move"),
    )
    plan.items = plan_items

    total = len(plan_items)
    timestamp = datetime.now(timezone.utc).isoformat()

    summary: Dict[str, object] = {
        "moved": 0,
        "copied": 0,
        "transferred": 0,
        "errors": [],
        "skipped": [],
        "review_required": [],
        "backups": [],
        "cancelled": False,
        "dry_run": dry_run,
    }
    audit_items: List[Dict[str, object]] = []
    executed_moves: Dict[str, str] = {}

    if cancel_event.is_set():
        log_callback("✘ Execution cancelled before start.")
        summary["cancelled"] = True
    else:
        log_callback(f"Executing plan with {total} operations…")

    for idx, item in enumerate(plan_items, start=1):
        if cancel_event.is_set():
            summary["cancelled"] = True
            log_callback("✘ Execution cancelled.")
            break

        src = item.source
        dst = item.destination
        progress_callback(idx, total, src, "execute")
        action = item.action or ("copy" if src in copy_only else "move")
        outcome: Dict[str, object] = {
            "source": src,
            "destination": dst,
            "action": action,
            "timestamp": timestamp,
        }

        if item.decision in (
            ExecutionDecision.SKIP_DUPLICATE,
            ExecutionDecision.SKIP_KEEP_EXISTING,
            ExecutionDecision.REVIEW_REQUIRED,
        ):
            reason = item.reason or "Skipped by plan decision"
            skip_entry = {"source": src, "destination": dst, "reason": reason}
            summary["skipped"].append(skip_entry)
            if item.decision in (
                ExecutionDecision.SKIP_DUPLICATE,
                ExecutionDecision.SKIP_KEEP_EXISTING,
                ExecutionDecision.REVIEW_REQUIRED,
            ):
                summary["review_required"].append(skip_entry)
            outcome["status"] = "skipped"
            outcome["reason"] = reason
            audit_items.append(outcome)
            continue

        if not os.path.exists(src):
            reason = "Source missing"
            summary["errors"].append(f"{reason}: {src}")  # type: ignore[arg-type]
            outcome["status"] = "failed"
            outcome["reason"] = reason
            audit_items.append(outcome)
            continue

        dest_exists = os.path.exists(dst)
        replacement_allowed = item.decision == ExecutionDecision.REPLACE

        if dest_exists and not replacement_allowed:
            reason = "Destination exists; replacement not allowed"
            skip_entry = {"source": src, "destination": dst, "reason": reason}
            summary["review_required"].append(skip_entry)
            summary["skipped"].append(skip_entry)
            outcome["status"] = "skipped"
            outcome["reason"] = reason
            audit_items.append(outcome)
            continue

        backup_path = None
        if dest_exists and replacement_allowed:
            rel = os.path.relpath(dst, plan.destination_root if os.path.isdir(plan.destination_root) else plan.library_root)
            backup_path = os.path.join(backup_root, rel)
            if not dry_run:
                try:
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    shutil.copy2(dst, backup_path)
                    summary["backups"].append(backup_path)
                except Exception as exc:  # pragma: no cover - OS interaction
                    summary["errors"].append(f"Backup failed for {dst}: {exc}")  # type: ignore[arg-type]
            outcome["backup_path"] = backup_path

        if dry_run:
            outcome["status"] = "dry_run"
            outcome["reason"] = outcome.get("reason", "No changes written (dry run)")
            executed_moves[src] = dst
            audit_items.append(outcome)
            continue

        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if dest_exists and replacement_allowed:
                try:
                    os.remove(dst)
                except FileNotFoundError:
                    pass
            if action == "copy":
                shutil.copy2(src, dst)
                summary["copied"] = int(summary["copied"]) + 1
            else:
                shutil.move(src, dst)
                summary["moved"] = int(summary["moved"]) + 1
            summary["transferred"] = int(summary["transferred"]) + 1
            executed_moves[src] = dst
            outcome["status"] = "success"
            if backup_path:
                outcome["backup_path"] = backup_path
        except Exception as exc:  # pragma: no cover - OS interaction
            msg = f"Failed to {action} {src} → {dst}: {exc}"
            summary["errors"].append(msg)  # type: ignore[arg-type]
            outcome["status"] = "failed"
            outcome["reason"] = str(exc)
            log_callback(msg)

        audit_items.append(outcome)

    summary["transferred"] = int(summary.get("moved", 0)) + int(summary.get("copied", 0))

    report_heading = "Library Sync (Executed)" if not dry_run else "Library Sync (Dry Run Execution)"

    report_skipped = [item for item in audit_items if item.get("status") == "skipped"]
    report_failed = [item for item in audit_items if item.get("status") == "failed"]
    report_review = [dict(rr) for rr in summary["review_required"]]  # type: ignore[arg-type]
    _render_execution_report(
        plan,
        executed_moves,
        report_skipped,  # type: ignore[arg-type]
        report_failed,  # type: ignore[arg-type]
        report_review,
        executed_report_path,
        report_heading,
        dry_run,
    )
    summary["report_path"] = executed_report_path

    audit_payload = {
        "executed_at": timestamp,
        "dry_run": dry_run,
        "plan": {
            "library_root": plan.library_root,
            "incoming_root": plan.incoming_root,
            "destination_root": plan.destination_root,
            "total_operations": total,
        },
        "summary": summary,
        "items": audit_items,
    }
    try:
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2)
    except Exception as exc:  # pragma: no cover - OS interaction
        log_callback(f"✘ Failed to write audit log: {exc}")
    summary["audit_path"] = audit_path

    playlist_created = None
    if create_playlist and executed_moves and not dry_run:
        try:
            write_playlist(sorted(executed_moves.values()), playlist_path)
            playlist_created = playlist_path
        except Exception as exc:  # pragma: no cover - OS interaction
            log_callback(f"✘ Failed to write playlist: {exc}")
    summary["playlist_path"] = playlist_created

    if not summary.get("cancelled"):
        log_callback("✓ Execution complete.")
    return summary


def execute_library_sync_plan(
    plan: LibrarySyncPlan,
    *,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    dry_run: bool = False,
    create_playlist: bool = True,
) -> Dict[str, object]:
    """Compatibility wrapper around :func:`execute_plan`."""

    return execute_plan(
        plan,
        dry_run=dry_run,
        allow_replacements=False,
        create_playlist=create_playlist,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )
