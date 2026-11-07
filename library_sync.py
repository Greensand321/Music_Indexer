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
    dry_run_coordinator as idx_dry_run_coordinator,
    fingerprint_cache as idx_fingerprint_cache,
    fingerprint_generator as idx_fingerprint_generator,
    music_indexer_api as idx,
    near_duplicate_detector as idx_near_duplicate_detector,
)
from playlist_generator import update_playlists, write_playlist

fingerprint_generator = idx_fingerprint_generator
fingerprint_distance = idx_near_duplicate_detector.fingerprint_distance
_ensure_db = idx_fingerprint_cache._ensure_db
FORMAT_PRIORITY = idx_config.FORMAT_PRIORITY
DEFAULT_FP_THRESHOLDS = idx_config.DEFAULT_FP_THRESHOLDS
MIXED_CODEC_THRESHOLD_BOOST = idx_config.MIXED_CODEC_THRESHOLD_BOOST

from crash_watcher import record_event
from crash_logger import watcher
from indexer_control import IndexCancelled
from library_sync_types import MatchResult, MatchStatus, TrackRecord, CandidateDistance
from library_sync_review_state import ReviewStateStore
from utils.reserved_folders import prune_reserved_dirs

debug: bool = False
_logger = logging.getLogger(__name__)
_logger.propagate = False
_logger.addHandler(logging.NullHandler())
_file_handler: Optional[logging.Handler] = None

NEAR_MISS_MARGIN_RATIO = 0.25
NEAR_MISS_MARGIN_FLOOR = 0.02
SHORTLIST_FALLBACK = 50
LOSSLESS_EXTS = {".flac", ".wav", ".alac", ".ape", ".aiff", ".aif"}


def set_debug(enabled: bool, log_root: str | None = None) -> None:
    """Enable or disable verbose debug logging.

    If ``log_root`` is ``None`` the log is written to ``~/.soundvault_logs``.
    Existing logs are overwritten on each run.
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
        log_root = os.path.join(os.path.expanduser("~"), ".soundvault_logs")
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


class ExecutionDecision(str, Enum):
    """Execution outcome for a planned incoming track."""

    COPY = "COPY"
    REPLACE = "REPLACE"
    SKIP_DUPLICATE = "SKIP_DUPLICATE"
    SKIP_KEEP_EXISTING = "SKIP_KEEP_EXISTING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ReviewAction(str, Enum):
    """Review-first replacement actions for Library Sync."""

    REPLACE_ALL = "replace_all"
    REPLACE_WITH_BETTER = "replace_with_better"
    CANCEL = "cancel"


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


def _select_threshold(ext: str, thresholds: Dict[str, float]) -> float:
    return float(thresholds.get(ext, thresholds.get("default", 0.3)))


def _select_pair_threshold(
    ext_a: str,
    ext_b: str,
    thresholds: Dict[str, float],
    mixed_codec_boost: float,
) -> float:
    base = max(_select_threshold(ext_a, thresholds), _select_threshold(ext_b, thresholds))
    if ext_a and ext_b and ext_a != ext_b:
        lossless_a = ext_a in LOSSLESS_EXTS
        lossless_b = ext_b in LOSSLESS_EXTS
        if lossless_a != lossless_b:
            base += mixed_codec_boost
    return base


def _fingerprint_settings_from_config(cfg: Dict[str, object]) -> Dict[str, object]:
    return {
        "trim_silence": bool(cfg.get("trim_silence", idx_config.FP_TRIM_SILENCE)),
        "fingerprint_offset_ms": int(cfg.get("fingerprint_offset_ms", idx_config.FP_OFFSET_MS)),
        "fingerprint_duration_ms": int(cfg.get("fingerprint_duration_ms", idx_config.FP_DURATION_MS)),
        "allow_mismatched_edits": bool(cfg.get("allow_mismatched_edits", idx_config.ALLOW_MISMATCHED_EDITS)),
        "silence_threshold_db": float(
            cfg.get("fingerprint_silence_threshold_db", idx_config.FP_SILENCE_THRESHOLD_DB)
        ),
        "silence_min_len_ms": int(cfg.get("fingerprint_silence_min_len_ms", idx_config.FP_SILENCE_MIN_LEN_MS)),
        "trim_lead_max_ms": int(cfg.get("fingerprint_trim_lead_max_ms", idx_config.FP_TRIM_LEAD_MAX_MS)),
        "trim_trail_max_ms": int(cfg.get("fingerprint_trim_trail_max_ms", idx_config.FP_TRIM_TRAIL_MAX_MS)),
        "trim_padding_ms": int(cfg.get("fingerprint_trim_padding_ms", idx_config.FP_TRIM_PADDING_MS)),
    }


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
    fingerprint_settings: Dict[str, object] | None = None,
    skip_reserved: bool = False,
) -> List[TrackRecord]:
    """Enumerate audio files, normalize paths, and attach cached fingerprints.

    With ``skip_reserved`` the folder's reserved top-level folders
    (Quarantine/, Trash/, Docs/, Manual Review/, Playlists/, Not Sorted/ — see
    ``utils.reserved_folders``) are not scanned, so quarantined duplicate
    losers and Library Sync's own backups are never "existing" tracks.
    """
    cancel_event = cancel_event or threading.Event()
    record_event(f"library_sync: scanning folder {folder}")
    _dlog(f"DEBUG: Scanning folder {folder}", log_callback)
    entries: List[Dict[str, object]] = []
    total_files = 0
    for dirpath, _dirs, files in os.walk(folder):
        if skip_reserved:
            prune_reserved_dirs(dirpath, _dirs, folder)
        _dirs.sort()
        if cancel_event.is_set():
            if log_callback:
                log_callback(json.dumps({"event": "scan_cancelled", "folder": folder, "stage": "walk"}))
            break
        for fname in sorted(files):
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
        if fingerprint_settings is None:
            cfg = idx_config.load_config()
            fingerprint_settings = _fingerprint_settings_from_config(cfg)
        if log_callback:
            log_callback(
                json.dumps({"event": "fingerprint_settings", "settings": dict(fingerprint_settings)})
            )
        if log_callback:
            log_callback(json.dumps({"event": "fingerprint_start", "pending": len(to_compute), "folder": folder}))
        for result in fingerprint_generator.compute_fingerprints_parallel(
            to_compute,
            db_path,
            log_callback,
            fp_progress,
            cancel_event=cancel_event,
            trim_silence=bool(fingerprint_settings.get("trim_silence", False)),
            fingerprint_offset_ms=int(fingerprint_settings.get("fingerprint_offset_ms", 0)),
            fingerprint_duration_ms=int(fingerprint_settings.get("fingerprint_duration_ms", 0)),
            allow_mismatched_edits=bool(fingerprint_settings.get("allow_mismatched_edits", True)),
            silence_threshold_db=float(
                fingerprint_settings.get("silence_threshold_db", idx_config.FP_SILENCE_THRESHOLD_DB)
            ),
            silence_min_len_ms=int(fingerprint_settings.get("silence_min_len_ms", 500)),
            trim_lead_max_ms=int(fingerprint_settings.get("trim_lead_max_ms", 500)),
            trim_trail_max_ms=int(fingerprint_settings.get("trim_trail_max_ms", 500)),
            trim_padding_ms=int(fingerprint_settings.get("trim_padding_ms", 100)),
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
    mixed_codec_threshold_boost: float = MIXED_CODEC_THRESHOLD_BOOST,
    log_callback: Callable[[str], None] | None = None,
    profiler: PerformanceProfile | None = None,
) -> List[MatchResult]:
    sig_index, ext_index = _build_candidate_index(library)
    fallback_pool = sorted([r for r in library if r.fingerprint], key=lambda r: r.normalized_path)
    results: List[MatchResult] = []
    start_time = time.perf_counter()
    for inc in incoming:
        candidates = _shortlist_candidates(
            inc, sig_index, ext_index, fallback_pool, log_callback=log_callback, profiler=profiler
        )
        cand_dist: List[CandidateDistance] = []
        candidate_meta: List[Tuple[TrackRecord, float, float, float, MatchStatus]] = []
        for cand in candidates:
            dist = fingerprint_distance(inc.fingerprint, cand.fingerprint)
            threshold_for_pair = _select_pair_threshold(
                inc.ext, cand.ext, thresholds, mixed_codec_threshold_boost
            )
            near_miss_margin = max(NEAR_MISS_MARGIN_FLOOR, threshold_for_pair * NEAR_MISS_MARGIN_RATIO)
            if dist == 0:
                status = MatchStatus.EXACT_MATCH
            elif dist <= threshold_for_pair:
                status = MatchStatus.COLLISION
            elif dist <= threshold_for_pair + near_miss_margin:
                status = MatchStatus.LOW_CONFIDENCE
            else:
                status = MatchStatus.NEW
            candidate_meta.append((cand, dist, threshold_for_pair, near_miss_margin, status))
            cand_dist.append(CandidateDistance(distance=dist, track=cand))
        cand_dist.sort()
        best_track = None
        best_distance = None
        threshold_used = _select_threshold(inc.ext, thresholds)
        near_miss_margin = max(NEAR_MISS_MARGIN_FLOOR, threshold_used * NEAR_MISS_MARGIN_RATIO)
        status = MatchStatus.NEW
        best_rank = -1
        for cand, dist, thr, margin, cand_status in candidate_meta:
            rank = {
                MatchStatus.EXACT_MATCH: 3,
                MatchStatus.COLLISION: 2,
                MatchStatus.LOW_CONFIDENCE: 1,
                MatchStatus.NEW: 0,
            }[cand_status]
            if best_track is None or rank > best_rank or (rank == best_rank and dist < (best_distance or 1)):
                best_track = cand
                best_distance = dist
                threshold_used = thr
                near_miss_margin = margin
                status = cand_status
                best_rank = rank

        if best_distance is None or best_track is None:
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
            f"dist={best_distance} thr={threshold_used} margin={near_miss_margin} "
            f"mixed_boost={mixed_codec_threshold_boost} status={status} confidence={confidence}",
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
    mixed_codec_threshold_boost: float | None = None,
    fingerprint_settings: Dict[str, object] | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    profiler: PerformanceProfile | None = None,
    include_profile: bool = False,
    include_match_objects: bool = False,
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
    validate_sync_folders(library_folder, incoming_folder)
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
    if mixed_codec_threshold_boost is None:
        cfg = idx_config.load_config()
        mixed_codec_threshold_boost = float(
            cfg.get("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST)
        )

    start = time.perf_counter()
    lib_records = _scan_folder(
        library_folder,
        db_path,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        profiler=profiler,
        fingerprint_settings=fingerprint_settings,
        skip_reserved=True,
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
        fingerprint_settings=fingerprint_settings,
        skip_reserved=True,
    )
    if profiler:
        profiler.incoming_scan_duration = time.perf_counter() - start

    match_results = _match_tracks(
        inc_records,
        lib_records,
        thresholds,
        fmt_priority,
        mixed_codec_threshold_boost=mixed_codec_threshold_boost,
        log_callback=log_callback,
        profiler=profiler,
    )

    result = _comparison_result(
        lib_records, inc_records, match_results, partial=cancel_event.is_set(),
        include_match_objects=include_match_objects,
    )
    if include_profile and profiler:
        result["profile"] = profiler.to_dict()
    return result


def _comparison_result(
    lib_records: List[TrackRecord],
    inc_records: List[TrackRecord],
    match_results: List[MatchResult],
    *,
    partial: bool,
    include_match_objects: bool,
) -> Dict[str, object]:
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
        "partial": partial,
    }
    if include_match_objects:
        result["match_objects"] = match_results
        result["library_record_objects"] = lib_records
        result["incoming_record_objects"] = inc_records
    record_event(
        "library_sync: comparison complete "
        f"library_existing={len(lib_records)} incoming_tracks={len(inc_records)} "
        f"new={len(new_paths)} existing_matches={len(existing_matches)} improved={len(improved)}"
    )
    return result


def validate_sync_folders(library_folder: str, incoming_folder: str) -> None:
    """Raise ``ValueError`` unless both folders exist and are distinct, non-nested
    directories. (A typo'd library path used to be silently created and every
    incoming track then looked "new".)"""
    for label, folder in (("Existing library", library_folder), ("Incoming folder", incoming_folder)):
        if not folder or not os.path.isdir(folder):
            raise ValueError(f"{label} not found: {folder or '(empty)'}")
    lib = _normalize_path(library_folder)
    inc = _normalize_path(incoming_folder)
    if lib == inc:
        raise ValueError("The incoming folder is the existing library itself.")
    if inc.startswith(lib + os.sep):
        raise ValueError("The incoming folder is inside the existing library; choose a folder outside it.")
    if lib.startswith(inc + os.sep):
        raise ValueError("The existing library is inside the incoming folder; choose separate folders.")


def summarize_matches(match_results: Iterable[MatchResult]) -> Dict[str, int]:
    """Counts by outcome, for user-facing summaries."""
    counts = {"incoming": 0, "new": 0, "matched": 0, "upgrades": 0, "low_confidence": 0}
    for res in match_results:
        counts["incoming"] += 1
        if res.status == MatchStatus.NEW or res.existing is None:
            counts["new"] += 1
        elif res.status == MatchStatus.LOW_CONFIDENCE:
            counts["low_confidence"] += 1
        else:
            counts["matched"] += 1
            if res.quality_label == "Potential Upgrade":
                counts["upgrades"] += 1
    return counts


def _record_from_dict(data: Dict[str, object]) -> TrackRecord:
    return TrackRecord(
        track_id=str(data["track_id"]),
        path=str(data["path"]),
        normalized_path=str(data.get("normalized_path") or _normalize_path(str(data["path"]))),
        ext=str(data.get("ext") or ""),
        bitrate=int(data.get("bitrate") or 0),
        size=int(data.get("size") or 0),
        mtime=float(data.get("mtime") or 0.0),
        fingerprint=data.get("fingerprint"),  # type: ignore[arg-type]
        tags=dict(data.get("tags") or {}),  # type: ignore[arg-type]
        duration=data.get("duration"),  # type: ignore[arg-type]
    )


def rematch_scan_result(
    scan_result: Dict[str, object],
    *,
    thresholds: Dict[str, float] | None = None,
    fmt_priority: Dict[str, int] | None = None,
    mixed_codec_threshold_boost: float | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    """Re-run matching on an earlier scan's records with new thresholds.

    No file is read or fingerprinted again: the records (with their
    fingerprints) come from ``scan_result`` (a ``compare_libraries`` result).
    Returns a result of the same shape, with match objects.
    """
    lib_records = scan_result.get("library_record_objects") or [
        _record_from_dict(d) for d in scan_result.get("library_records", [])  # type: ignore[union-attr]
    ]
    inc_records = scan_result.get("incoming_record_objects") or [
        _record_from_dict(d) for d in scan_result.get("incoming_records", [])  # type: ignore[union-attr]
    ]
    if mixed_codec_threshold_boost is None:
        cfg = idx_config.load_config()
        mixed_codec_threshold_boost = float(
            cfg.get("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST)
        )
    match_results = _match_tracks(
        list(inc_records),  # type: ignore[arg-type]
        list(lib_records),  # type: ignore[arg-type]
        thresholds or DEFAULT_FP_THRESHOLDS,
        fmt_priority or FORMAT_PRIORITY,
        mixed_codec_threshold_boost=mixed_codec_threshold_boost,
        log_callback=log_callback,
    )
    return _comparison_result(
        list(lib_records), list(inc_records), match_results,  # type: ignore[arg-type]
        partial=bool(scan_result.get("partial")), include_match_objects=True,
    )


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
    """Move/route plan computed by the indexer engine for Library Sync.

    ``items`` is the reviewed, executable plan: ``execute_plan`` performs
    exactly these decisions (it only recomputes them for legacy plans built
    without items). ``copy_only`` / ``allowed_replacements`` /
    ``replacement_targets`` record the user's flags that produced them.
    """

    library_root: str
    incoming_root: str
    destination_root: str
    moves: Dict[str, str]
    tag_index: Dict[str, object]
    decision_log: List[str]
    copy_only: set[str] = field(default_factory=set)
    allowed_replacements: set[str] = field(default_factory=set)
    replacement_targets: Dict[str, str] = field(default_factory=dict)
    items: List[PlannedItem] = field(default_factory=list)
    transfer_mode: str = "move"
    match_based: bool = False   # items decided from the scan's match results

    def planned_moves(self) -> Dict[str, str]:
        """Return a copy of the planned move mapping for execution."""
        if self.items:
            return {item.source: item.destination for item in self.items}
        return dict(self.moves)

    def counts(self) -> Dict[str, int]:
        """How many items transfer (copy/move), replace, skip or need review."""
        counts = {"total": len(self.items), "copy": 0, "move": 0, "replace": 0, "skip": 0, "review": 0}
        for item in self.items:
            if item.decision == ExecutionDecision.COPY:
                counts["copy" if item.action == "copy" else "move"] += 1
            elif item.decision == ExecutionDecision.REPLACE:
                counts["replace"] += 1
            elif item.decision == ExecutionDecision.REVIEW_REQUIRED:
                counts["review"] += 1
            else:
                counts["skip"] += 1
        return counts

    def render_preview(self, output_html_path: str) -> str:
        """Generate a dry-run HTML preview from this plan."""
        heading = "Library Sync (Dry Run Preview)"
        preview_items = [item.to_dict() for item in self.items] if self.items else None
        coord = idx_dry_run_coordinator.DryRunCoordinator()
        if self.items:
            coord.set_html_section("B", _render_plan_summary_html(self))
        idx.render_dry_run_html_from_plan(
            self.destination_root,
            output_html_path,
            self.moves,
            self.tag_index,
            plan_items=preview_items,
            heading_text=heading,
            title_prefix=heading,
            coord=coord,
        )
        return output_html_path


def _rel_to(path: str, root: str) -> str:
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path
    return path if rel.startswith("..") else rel


def _render_plan_summary_html(plan: LibrarySyncPlan) -> str:
    """Counts, mode and per-item reasons — what the folder tree cannot show."""
    import html as _html

    esc = _html.escape
    c = plan.counts()
    mode_text = (
        "copy — originals stay in the incoming folder"
        if plan.transfer_mode == "copy" else
        "move — originals are moved out of the incoming folder"
    )
    lines = [
        "<h2>What Execute will do</h2>",
        f"<p>Transfer mode: <strong>{esc(mode_text)}</strong></p>",
        "<ul>",
        f"<li>{c['copy'] + c['move']} track(s) added ({c['copy']} copied, {c['move']} moved)</li>",
        f"<li>{c['replace']} existing track(s) replaced — each replaced file is backed up to "
        "Docs/Backups/LibrarySync/ first</li>",
        f"<li>{c['skip']} skipped (already in the library)</li>",
        f"<li>{c['review']} need review — skipped unless you flag them Copy or Replace</li>",
        "</ul>",
    ]
    others = [i for i in plan.items if i.decision != ExecutionDecision.COPY or i.reason]
    if others:
        lines.append("<h3>Decisions</h3>")
        lines.append("<table border='1' cellpadding='4' cellspacing='0'>")
        lines.append("<tr><th>Incoming track</th><th>Decision</th><th>Destination</th><th>Why</th></tr>")
        for item in others:
            lines.append(
                "<tr>"
                f"<td>{esc(_rel_to(item.source, plan.incoming_root))}</td>"
                f"<td>{esc(item.decision.value)} ({esc(item.action)})</td>"
                f"<td>{esc(_rel_to(item.destination, plan.library_root))}</td>"
                f"<td>{esc(item.reason or '')}</td>"
                "</tr>"
            )
        lines.append("</table>")
    return "\n".join(lines)


def _files_match(path_a: str, path_b: str) -> bool:
    try:
        return filecmp.cmp(path_a, path_b, shallow=False)
    except Exception:
        return False


def _unique_destination(path: str, taken: set[str]) -> str:
    if path not in taken and not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        candidate = _normalize_path(f"{base} ({n}){ext}")
        if candidate not in taken and not os.path.exists(candidate):
            return candidate
        n += 1


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


def _decide_items_from_matches(
    moves: Dict[str, str],
    match_results: Iterable[MatchResult],
    review_flags: ReviewStateStore | None,
    *,
    library_root: str,
    transfer_mode: str,
) -> Tuple[List[PlannedItem], set[str], set[str], Dict[str, str]]:
    """Decide every incoming track from its match result and the user's flags.

    * New track → transferred to its routed destination.
    * Exact match / collision → skipped (already in the library); a "Potential
      Upgrade" needs review instead.
    * Low-confidence match → needs review.
    * Flag **Copy** → added regardless (as a copy; the original stays), under
      a ``(n)`` name if its destination is taken.
    * Flag **Replace** → takes the matched track's place (same folder and
      name, incoming extension); the old file is backed up and retired.
    * Files the reviewed scan never saw → need review (scan again).

    Returns ``(items, copy_only, allowed_replacements, replacement_targets)``.
    """
    mode = "copy" if str(transfer_mode).lower() == "copy" else "move"
    by_source = {_normalize_path(m.incoming.path): m for m in match_results}
    copy_ids = set(review_flags.flags.copy) if review_flags else set()
    replace_ids = dict(review_flags.flags.replace) if review_flags else {}
    items: List[PlannedItem] = []
    copy_only: set[str] = set()
    allowed: set[str] = set()
    targets: Dict[str, str] = {}
    taken: set[str] = set()

    def add(src: str, dst: str, decision: ExecutionDecision, reason: str, action: str = mode) -> None:
        items.append(PlannedItem(source=src, destination=dst, decision=decision, reason=reason, action=action))

    for src, dst in sorted(moves.items(), key=lambda kv: kv[1]):
        match = by_source.get(src)
        if match is None:
            add(src, dst, ExecutionDecision.REVIEW_REQUIRED,
                "Not part of the reviewed scan — scan again to include it")
            continue
        track_id = match.incoming.track_id
        existing = match.existing
        existing_path = _normalize_path(existing.path) if existing is not None else ""
        existing_rel = _rel_to(existing_path, library_root) if existing_path else ""

        if track_id in replace_ids and existing is not None:
            stem = os.path.splitext(os.path.basename(existing_path))[0]
            new_dst = _normalize_path(os.path.join(os.path.dirname(existing_path), stem + match.incoming.ext))
            if new_dst != existing_path and (os.path.exists(new_dst) or new_dst in taken):
                add(src, new_dst, ExecutionDecision.REVIEW_REQUIRED,
                    f"Flagged Replace, but {_rel_to(new_dst, library_root)} already exists "
                    "and is not the matched track")
                continue
            taken.add(new_dst)
            allowed.add(new_dst)
            targets[src] = existing_path
            add(src, new_dst, ExecutionDecision.REPLACE,
                f"Flagged Replace: takes the place of {existing_rel} (backed up first)")
            continue

        if track_id in copy_ids:
            final = _unique_destination(dst, taken)
            taken.add(final)
            copy_only.add(src)
            reason = "Flagged Copy"
            if existing is not None and match.status != MatchStatus.NEW:
                reason += f": added even though it matches {existing_rel}"
            add(src, final, ExecutionDecision.COPY, reason, action="copy")
            continue

        if match.status == MatchStatus.NEW or existing is None:
            if os.path.exists(dst) or dst in taken:
                if os.path.exists(dst) and _files_match(src, dst):
                    add(src, dst, ExecutionDecision.SKIP_DUPLICATE, "Exact duplicate already at its destination")
                else:
                    add(src, dst, ExecutionDecision.SKIP_KEEP_EXISTING,
                        "Destination already exists; keeping the library copy")
            else:
                taken.add(dst)
                add(src, dst, ExecutionDecision.COPY, "New track")
            continue

        if match.status == MatchStatus.LOW_CONFIDENCE:
            add(src, existing_path, ExecutionDecision.REVIEW_REQUIRED,
                f"Low-confidence match with {existing_rel} — flag Copy to add it or Replace to swap it in")
            continue

        if match.quality_label == "Potential Upgrade":
            add(src, existing_path, ExecutionDecision.REVIEW_REQUIRED,
                f"Possible upgrade of {existing_rel} — flag Replace to use it")
        else:
            add(src, existing_path, ExecutionDecision.SKIP_DUPLICATE,
                f"Already in the library as {existing_rel}")

    return items, copy_only, allowed, targets


def resolve_review_flags_to_paths(
    review_flags: ReviewStateStore | None,
    match_results: Iterable[MatchResult] | None = None,
) -> Tuple[List[str], List[str]]:
    """Convert ReviewStateStore flags to source and destination path lists.

    Legacy helper: the Qt workspace now passes the flags and match results
    straight to :func:`compute_library_sync_plan`, which also handles Replace
    (whose "allowed" path here is the matched *existing* file, which a
    destination-based plan never compares equal to).

    Returns ``(copy_only_paths, allowed_replacement_paths)``.
    """
    if not review_flags or not match_results:
        return [], []

    copy_only_paths: List[str] = []
    allowed_replacement_paths: List[str] = []

    # Build a map of track_id -> MatchResult for quick lookup
    track_map = {res.incoming.track_id: res for res in match_results}

    # Process copy flags
    for track_id in review_flags.flags.copy:
        if track_id in track_map:
            copy_only_paths.append(_normalize_path(track_map[track_id].incoming.path))

    # Process replace flags
    for incoming_id, existing_id in review_flags.flags.replace.items():
        if incoming_id in track_map:
            # For replace, we allow the destination to be replaced
            existing_path = track_map[incoming_id].existing.path if track_map[incoming_id].existing else None
            if existing_path:
                allowed_replacement_paths.append(_normalize_path(existing_path))

    return copy_only_paths, allowed_replacement_paths


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
    copy_only_paths: Iterable[str] | None = None,
    allowed_replacement_paths: Iterable[str] | None = None,
    match_results: Iterable[MatchResult] | None = None,
    review_flags: ReviewStateStore | None = None,
) -> LibrarySyncPlan:
    """Compute a deterministic move/route plan for Library Sync.

    With ``match_results`` (from ``compare_libraries``) the plan follows the
    reviewed matches and ``review_flags`` (see
    :func:`_decide_items_from_matches`): duplicates are not copied in and
    Copy/Replace flags win. Without them, the legacy destination-exists rules
    apply. Either way the plan stores the decisions in ``items`` and the flags
    in ``copy_only`` / ``allowed_replacements`` / ``replacement_targets``, and
    ``execute_plan`` runs exactly those items.
    """
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
    normalized_mode = "copy" if str(transfer_mode).lower() == "copy" else "move"

    if match_results is not None:
        plan_items, copy_only, allowed, targets = _decide_items_from_matches(
            remapped_moves,
            list(match_results),
            review_flags,
            library_root=_normalize_path(library_root),
            transfer_mode=normalized_mode,
        )
        for item in plan_items:
            original = remapped_moves.get(item.source)
            if original and original != item.destination and original in remapped_tag_index:
                remapped_tag_index.setdefault(item.destination, remapped_tag_index[original])
        decision_log = list(decision_log) + [
            f"{os.path.basename(i.source)}: {i.decision.value} — {i.reason}" for i in plan_items
        ]
    else:
        copy_only = {_normalize_path(p) for p in (copy_only_paths or [])}
        allowed = {_normalize_path(p) for p in (allowed_replacement_paths or [])}
        targets = {}
        plan_items = _compute_plan_items(
            remapped_moves,
            destination_root=_normalize_path(destination_root),
            copy_only=copy_only,
            allow_all_replacements=False,
            allowed_replacements=allowed,
            transfer_mode=transfer_mode,
        )

    return LibrarySyncPlan(
        library_root=_normalize_path(library_root),
        incoming_root=_normalize_path(incoming_folder),
        destination_root=_normalize_path(destination_root),
        moves=remapped_moves,
        tag_index=remapped_tag_index,
        decision_log=decision_log,
        copy_only=set(copy_only),
        allowed_replacements=set(allowed),
        replacement_targets=dict(targets),
        items=plan_items,
        transfer_mode=normalized_mode,
        match_based=match_results is not None,
    )


def _select_replacement_sources(
    results: Iterable[MatchResult],
    action: ReviewAction,
) -> Dict[str, MatchResult]:
    if action == ReviewAction.CANCEL:
        return {}

    selected: Dict[str, MatchResult] = {}
    for res in results:
        if res.existing is None:
            continue
        if action == ReviewAction.REPLACE_WITH_BETTER:
            if res.existing_score is None:
                continue
            if res.incoming_score <= (res.existing_score or 0):
                continue
        selected[_normalize_path(res.incoming.path)] = res
    return selected


def build_review_replacement_plan(
    library_root: str,
    incoming_folder: str,
    results: Iterable[MatchResult],
    action: ReviewAction,
    *,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    flush_cache: bool = False,
    max_workers: int | None = None,
    cancel_event: threading.Event | None = None,
    transfer_mode: str = "copy",
) -> LibrarySyncPlan:
    """Build a Library Sync plan that replaces matched tracks per review action."""
    if action == ReviewAction.CANCEL:
        raise ValueError("Cannot build a replacement plan for a cancelled action.")

    selected = _select_replacement_sources(results, action)
    base_plan = compute_library_sync_plan(
        library_root,
        incoming_folder,
        log_callback=log_callback,
        progress_callback=progress_callback,
        flush_cache=flush_cache,
        max_workers=max_workers,
        cancel_event=cancel_event,
        transfer_mode=transfer_mode,
    )
    if not selected:
        base_plan.moves = {}
        base_plan.tag_index = {}
        base_plan.allowed_replacements = set()
        base_plan.replacement_targets = {}
        base_plan.items = []
        base_plan.decision_log.append("No replacement candidates selected.")
        return base_plan

    filtered_moves = {
        src: dst for src, dst in base_plan.moves.items() if src in selected
    }
    filtered_tag_index = {
        dst: data for dst, data in base_plan.tag_index.items() if dst in filtered_moves.values()
    }
    replacement_targets = {
        src: _normalize_path(selected[src].existing.path)  # type: ignore[union-attr]
        for src in filtered_moves
        if selected[src].existing is not None
    }
    base_plan.moves = filtered_moves
    base_plan.tag_index = filtered_tag_index
    base_plan.allowed_replacements = set(filtered_moves.values())
    base_plan.replacement_targets = replacement_targets
    base_plan.items = _compute_plan_items(
        base_plan.moves,
        destination_root=base_plan.destination_root,
        copy_only=base_plan.copy_only,
        allow_all_replacements=False,
        allowed_replacements=base_plan.allowed_replacements,
        transfer_mode=base_plan.transfer_mode,
    )
    base_plan.decision_log.append(
        f"Review action {action.value} selected {len(filtered_moves)} replacement(s)."
    )
    return base_plan


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
    copy_only_paths: Iterable[str] | None = None,
    allowed_replacement_paths: Iterable[str] | None = None,
    match_results: Iterable[MatchResult] | None = None,
    review_flags: ReviewStateStore | None = None,
) -> LibrarySyncPlan:
    """Compute a Library Sync plan and write the dry-run preview.

    Parameters
    ----------
    copy_only_paths : Iterable[str] | None
        Source paths that should always be copied regardless of auto-decision.
    allowed_replacement_paths : Iterable[str] | None
        Destination paths where replacement is allowed regardless of auto-decision.
    match_results, review_flags
        The reviewed scan and the user's flags; when given, the plan follows
        them (see :func:`compute_library_sync_plan`).
    """
    plan = compute_library_sync_plan(
        library_root,
        incoming_folder,
        log_callback=log_callback,
        progress_callback=progress_callback,
        flush_cache=flush_cache,
        max_workers=max_workers,
        cancel_event=cancel_event,
        transfer_mode=transfer_mode,
        copy_only_paths=copy_only_paths,
        allowed_replacement_paths=allowed_replacement_paths,
        match_results=match_results,
        review_flags=review_flags,
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


def _unique_file_path(path: str) -> str:
    if not os.path.lexists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while os.path.lexists(f"{base} ({n}){ext}"):
        n += 1
    return f"{base} ({n}){ext}"


def _transfer(src: str, dst: str, action: str, *, replace: bool) -> None:
    """Copy or move ``src`` to ``dst``. With ``replace`` the existing ``dst`` is
    swapped out atomically (never left missing if the transfer fails)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not replace:
        if os.path.lexists(dst):
            raise FileExistsError(f"destination exists: {dst}")
        if action == "copy":
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        return
    tmp = _unique_file_path(dst + ".alphadex-incoming")
    if action == "copy":
        shutil.copy2(src, tmp)
    else:
        shutil.move(src, tmp)
    os.replace(tmp, dst)


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

    A plan with ``items`` (every plan built by :func:`compute_library_sync_plan`)
    is executed **exactly as previewed**: its decisions, copy/move actions and
    replacement targets are used verbatim (``allow_replacements`` cannot widen
    them). Only a legacy plan without items has its decisions computed here,
    from ``plan.moves`` plus ``plan.copy_only`` / ``plan.allowed_replacements``
    / ``allow_replacements``.

    Safety checks still apply per item at execution time: a missing source
    fails, an existing destination is never overwritten unless the item is a
    REPLACE, and every replaced file is backed up (into a per-run folder under
    ``Docs/Backups/LibrarySync/``) before it is swapped out. Everything is
    recorded in a JSON audit and an executed-report HTML.
    """

    log_callback = log_callback or (lambda _msg: None)
    progress_callback = progress_callback or (lambda _c, _t, _p, _ph: None)
    cancel_event = cancel_event or threading.Event()

    docs_dir = _ensure_docs_dir(plan.library_root)
    audit_path = audit_path or os.path.join(docs_dir, "LibrarySyncAudit.json")
    executed_report_path = executed_report_path or os.path.join(docs_dir, "LibrarySyncExecuted.html")
    playlist_root = os.path.join(plan.library_root, "Playlists")
    if playlist_path is None:
        playlist_stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        playlist_path = os.path.join(playlist_root, f"LibrarySync_Added_{playlist_stamp}.m3u8")
    # One backup folder per run: an earlier run's backup is never overwritten.
    backup_root = os.path.join(
        docs_dir, "Backups", "LibrarySync", datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    copy_only = set(getattr(plan, "copy_only", set()) or [])
    if plan.items:
        plan_items = list(plan.items)
        if allow_replacements not in (None, False):
            log_callback("! allow_replacements ignored: a previewed plan runs exactly as previewed.")
    else:
        allow_all_replacements = allow_replacements is True
        allowed_replacements = set(getattr(plan, "allowed_replacements", set()) or [])
        if allow_replacements not in (None, False, True):
            allowed_replacements.update(allow_replacements)  # type: ignore[arg-type]
        plan_items = _compute_plan_items(
            plan.planned_moves(),
            destination_root=plan.destination_root,
            copy_only=copy_only,
            allow_all_replacements=allow_all_replacements,
            allowed_replacements=allowed_replacements,
            transfer_mode=getattr(plan, "transfer_mode", "move"),
        )

    total = len(plan_items)
    timestamp = datetime.now(timezone.utc).isoformat()

    summary: Dict[str, object] = {
        "moved": 0,
        "copied": 0,
        "replaced": 0,
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
    playlist_updates: Dict[str, str] = {}
    replacement_targets = dict(getattr(plan, "replacement_targets", {}) or {})
    backup_base = plan.destination_root if os.path.isdir(plan.destination_root) else plan.library_root

    def _backup_path_for(path: str) -> str:
        rel = os.path.relpath(path, backup_base)
        if rel.startswith(".."):
            rel = os.path.relpath(path, plan.library_root)
        if rel.startswith(".."):
            rel = os.path.basename(path)
        return _unique_file_path(os.path.join(backup_root, rel))

    if cancel_event.is_set():
        log_callback("✘ Execution cancelled before start.")
        summary["cancelled"] = True
    else:
        log_callback(f"Executing plan with {total} operations…")

    for idx_num, item in enumerate(plan_items, start=1):
        if cancel_event.is_set():
            summary["cancelled"] = True
            log_callback("✘ Execution cancelled.")
            break

        src = item.source
        dst = item.destination
        progress_callback(idx_num, total, src, "execute")
        action = item.action or ("copy" if src in copy_only else "move")
        outcome: Dict[str, object] = {
            "source": src,
            "destination": dst,
            "action": action,
            "decision": item.decision.value,
            "timestamp": timestamp,
        }
        replacement_target = replacement_targets.get(src)

        if item.decision in (
            ExecutionDecision.SKIP_DUPLICATE,
            ExecutionDecision.SKIP_KEEP_EXISTING,
            ExecutionDecision.REVIEW_REQUIRED,
        ):
            reason = item.reason or "Skipped by plan decision"
            skip_entry = {"source": src, "destination": dst, "reason": reason}
            summary["skipped"].append(skip_entry)
            if item.decision != ExecutionDecision.SKIP_DUPLICATE:
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

        if dry_run:
            outcome["status"] = "dry_run"
            outcome["reason"] = "No changes written (dry run)"
            if replacement_target and replacement_target != dst:
                outcome["replaced_path"] = replacement_target
            executed_moves[src] = dst
            audit_items.append(outcome)
            continue

        if dest_exists and replacement_allowed:
            # Never replace a file without a backup of it.
            backup_path = _backup_path_for(dst)
            try:
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy2(dst, backup_path)
            except Exception as exc:  # pragma: no cover - OS interaction
                msg = f"Backup failed for {dst}: {exc}; nothing replaced"
                summary["errors"].append(msg)  # type: ignore[arg-type]
                outcome["status"] = "failed"
                outcome["reason"] = msg
                audit_items.append(outcome)
                log_callback(msg)
                continue
            summary["backups"].append(backup_path)
            outcome["backup_path"] = backup_path

        try:
            _transfer(src, dst, action, replace=dest_exists and replacement_allowed)
        except Exception as exc:  # pragma: no cover - OS interaction
            msg = f"Failed to {action} {src} → {dst}: {exc}"
            summary["errors"].append(msg)  # type: ignore[arg-type]
            outcome["status"] = "failed"
            outcome["reason"] = str(exc)
            log_callback(msg)
            audit_items.append(outcome)
            continue

        if action == "copy":
            summary["copied"] = int(summary["copied"]) + 1
        else:
            summary["moved"] = int(summary["moved"]) + 1
        if replacement_allowed:
            summary["replaced"] = int(summary["replaced"]) + 1
        executed_moves[src] = dst
        outcome["status"] = "success"

        # Retire the file this one replaces (e.g. an MP3 upgraded to FLAC):
        # moved into the run's backup folder, never deleted outright.
        if replacement_target and replacement_target != dst and os.path.exists(replacement_target):
            old_backup = _backup_path_for(replacement_target)
            try:
                os.makedirs(os.path.dirname(old_backup), exist_ok=True)
                shutil.move(replacement_target, old_backup)
                summary["backups"].append(old_backup)
                outcome["replaced_path"] = replacement_target
                outcome["replaced_backup_path"] = old_backup
                playlist_updates[replacement_target] = dst
            except Exception as exc:  # pragma: no cover - OS interaction
                msg = f"Could not retire replaced file {replacement_target}: {exc} (left in place)"
                summary["errors"].append(msg)  # type: ignore[arg-type]
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
            "transfer_mode": getattr(plan, "transfer_mode", "move"),
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

    if playlist_updates and not dry_run:
        try:
            update_playlists(playlist_updates)
        except Exception as exc:  # pragma: no cover - OS interaction
            log_callback(f"✘ Failed to update playlists: {exc}")

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
