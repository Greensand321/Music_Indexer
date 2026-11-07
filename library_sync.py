"""Helpers for comparing an existing library to an incoming folder."""
from __future__ import annotations

import errno
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
    """Comparison key for ``path`` (lower-cased on Windows).

    Only ever use it to *compare* or *look up* paths: handing it to the file
    system would create lower-cased folders and files on Windows. Real paths
    go through :func:`_abs_path`.
    """
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _abs_path(path: str) -> str:
    """Absolute, normalised ``path`` with its original letter case kept."""
    return os.path.normpath(os.path.abspath(path))


def _rel_inside(path: str, root: str) -> str | None:
    """``path`` relative to ``root`` when it lies inside ``root``, else None."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:      # another drive (Windows)
        return None
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return None
    return rel


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
    # What the plan assumed about the disk when it was built: path ->
    # (size, mtime_ns) for every incoming file to transfer and every library
    # file a Replace displaces, or None for a destination that must not exist
    # yet. ``execute_plan`` skips (and reports) every item whose assumption no
    # longer holds. Empty for hand-built legacy plans (nothing is checked).
    snapshot: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)

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


def _unique_destination(path: str, taken) -> str:
    """``path``, or ``name (n).ext`` when it exists on disk or its comparison
    key is in ``taken`` (a set or dict of :func:`_normalize_path` keys).
    The returned path keeps its original letter case."""
    if _normalize_path(path) not in taken and not os.path.lexists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        candidate = f"{base} ({n}){ext}"
        if _normalize_path(candidate) not in taken and not os.path.lexists(candidate):
            return candidate
        n += 1


def _file_sig(path: str) -> Optional[Tuple[int, int]]:
    """``(size, mtime_ns)`` of ``path``, or None when it does not exist."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (int(st.st_size), int(st.st_mtime_ns))


def _plan_snapshot(items: Iterable[PlannedItem], replacement_targets: Dict[str, str]) -> Dict[str, Optional[Tuple[int, int]]]:
    """What execute must find on disk for ``items`` to still mean what the
    preview said (see :attr:`LibrarySyncPlan.snapshot`)."""
    snap: Dict[str, Optional[Tuple[int, int]]] = {}
    for item in items:
        if item.decision not in (ExecutionDecision.COPY, ExecutionDecision.REPLACE):
            continue
        snap[item.source] = _file_sig(item.source)
        if item.decision == ExecutionDecision.REPLACE:
            snap[item.destination] = _file_sig(item.destination)
            target = replacement_targets.get(item.source)
            if target:
                snap[target] = _file_sig(target)
        else:
            snap[item.destination] = None
    return snap


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

    # Compared by key (case-insensitively on Windows), whatever form the
    # caller's paths are in.
    copy_only_set = {_normalize_path(p) for p in (copy_only or [])}
    allowed_replacements_set = {_normalize_path(p) for p in (allowed_replacements or [])}
    normalized_transfer_mode = "copy" if str(transfer_mode).lower() == "copy" else "move"
    items: List[PlannedItem] = []

    for src, dst in sorted(moves.items(), key=lambda item: item[1]):
        action = "copy" if _normalize_path(src) in copy_only_set else normalized_transfer_mode
        dest_exists = os.path.exists(dst)
        replacement_allowed = allow_all_replacements or _normalize_path(dst) in allowed_replacements_set

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

    * New track → transferred to its routed destination; when a *different*
      file already has that name it is added under a ``(n)`` name (and the
      reason says so). Only a byte-identical file there (or an identical
      incoming file already being added) makes it a skip.
    * Exact match / collision → skipped (already in the library); a "Potential
      Upgrade" needs review instead.
    * Low-confidence match → needs review.
    * Flag **Copy** → added regardless (as a copy; the original stays), under
      a ``(n)`` name if its destination is taken.
    * Flag **Replace** → takes the matched track's place (same folder and
      name, incoming extension); the old file is backed up and retired.
    * Files the reviewed scan never saw → need review (scan again).

    Paths in the result keep their letter case; matching uses
    :func:`_normalize_path` keys (case-insensitive on Windows).

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
    taken: Dict[str, str] = {}      # destination key -> incoming source taking it

    def add(src: str, dst: str, decision: ExecutionDecision, reason: str, action: str = mode) -> None:
        items.append(PlannedItem(source=src, destination=dst, decision=decision, reason=reason, action=action))

    for src, dst in sorted(moves.items(), key=lambda kv: kv[1]):
        match = by_source.get(_normalize_path(src))
        if match is None:
            add(src, dst, ExecutionDecision.REVIEW_REQUIRED,
                "Not part of the reviewed scan — scan again to include it")
            continue
        track_id = match.incoming.track_id
        existing = match.existing
        existing_path = _abs_path(existing.path) if existing is not None else ""
        existing_rel = _rel_to(existing_path, library_root) if existing_path else ""

        if track_id in replace_ids and existing is not None:
            stem = os.path.splitext(os.path.basename(existing_path))[0]
            new_dst = os.path.join(os.path.dirname(existing_path), stem + match.incoming.ext)
            new_key = _normalize_path(new_dst)
            if new_key != _normalize_path(existing_path) and (os.path.lexists(new_dst) or new_key in taken):
                add(src, new_dst, ExecutionDecision.REVIEW_REQUIRED,
                    f"Flagged Replace, but {_rel_to(new_dst, library_root)} already exists "
                    "and is not the matched track")
                continue
            taken[new_key] = src
            allowed.add(new_dst)
            targets[src] = existing_path
            add(src, new_dst, ExecutionDecision.REPLACE,
                f"Flagged Replace: takes the place of {existing_rel} (backed up first)")
            continue

        if track_id in copy_ids:
            final = _unique_destination(dst, taken)
            taken[_normalize_path(final)] = src
            copy_only.add(src)
            reason = "Flagged Copy"
            if existing is not None and match.status != MatchStatus.NEW:
                reason += f": added even though it matches {existing_rel}"
            add(src, final, ExecutionDecision.COPY, reason, action="copy")
            continue

        if match.status == MatchStatus.NEW or existing is None:
            other = taken.get(_normalize_path(dst))
            if other is not None and _files_match(src, other):
                add(src, dst, ExecutionDecision.SKIP_DUPLICATE,
                    f"Identical to {os.path.basename(other)}, which this plan already adds")
                continue
            if other is None and os.path.lexists(dst) and _files_match(src, dst):
                add(src, dst, ExecutionDecision.SKIP_DUPLICATE, "Exact duplicate already at its destination")
                continue
            final = _unique_destination(dst, taken)
            taken[_normalize_path(final)] = src
            if final == dst:
                add(src, dst, ExecutionDecision.COPY, "New track")
            else:
                # A *different* song already has the routed name: the track is
                # still new, so it is added under a "(n)" name, not skipped.
                add(src, final, ExecutionDecision.COPY,
                    f"New track: {_rel_to(dst, library_root)} is already taken by a different "
                    f"file, so it is added as {os.path.basename(final)}")
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
            copy_only_paths.append(_abs_path(track_map[track_id].incoming.path))

    # Process replace flags
    for incoming_id, existing_id in review_flags.flags.replace.items():
        if incoming_id in track_map:
            # For replace, we allow the destination to be replaced
            existing_path = track_map[incoming_id].existing.path if track_map[incoming_id].existing else None
            if existing_path:
                allowed_replacement_paths.append(_abs_path(existing_path))

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

    # Real paths keep their letter case (normcase is for comparisons only:
    # on Windows it lower-cases, and every folder and file Library Sync
    # created used to come out lower-cased).
    library_abs = _abs_path(library_root)
    incoming_abs = _abs_path(incoming_folder)
    destination_root = os.path.join(library_abs, "Music")
    if not os.path.isdir(destination_root):
        destination_root = library_abs
    # The engine routes an incoming folder's tracks under *its* music root
    # (``incoming/Music`` when that exists), so routes are re-based from
    # there: an incoming ``Music/`` folder used to give ``Music/Music/…``.
    incoming_music_root = os.path.join(incoming_abs, "Music")
    if not os.path.isdir(incoming_music_root):
        incoming_music_root = incoming_abs

    def _remap_path(path: str) -> str:
        path = _abs_path(path)
        rel = _rel_inside(path, incoming_music_root)
        if rel is not None:
            return _abs_path(os.path.join(destination_root, rel))
        rel = _rel_inside(path, incoming_abs)
        if rel is not None:      # e.g. a top-level folder beside incoming/Music
            return _abs_path(os.path.join(library_abs, rel))
        return path

    remapped_moves = {_abs_path(src): _remap_path(dst) for src, dst in moves.items()}
    remapped_tag_index = {_remap_path(dst): data for dst, data in tag_index.items()}
    normalized_mode = "copy" if str(transfer_mode).lower() == "copy" else "move"

    if match_results is not None:
        plan_items, copy_only, allowed, targets = _decide_items_from_matches(
            remapped_moves,
            list(match_results),
            review_flags,
            library_root=library_abs,
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
        copy_only = {_abs_path(p) for p in (copy_only_paths or [])}
        allowed = {_abs_path(p) for p in (allowed_replacement_paths or [])}
        targets = {}
        plan_items = _compute_plan_items(
            remapped_moves,
            destination_root=destination_root,
            copy_only=copy_only,
            allow_all_replacements=False,
            allowed_replacements=allowed,
            transfer_mode=transfer_mode,
        )

    return LibrarySyncPlan(
        library_root=library_abs,
        incoming_root=incoming_abs,
        destination_root=destination_root,
        moves=remapped_moves,
        tag_index=remapped_tag_index,
        decision_log=decision_log,
        copy_only=set(copy_only),
        allowed_replacements=set(allowed),
        replacement_targets=dict(targets),
        items=plan_items,
        transfer_mode=normalized_mode,
        match_based=match_results is not None,
        snapshot=_plan_snapshot(plan_items, targets),
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
        base_plan.snapshot = {}
        base_plan.decision_log.append("No replacement candidates selected.")
        return base_plan

    filtered_moves = {
        src: dst for src, dst in base_plan.moves.items() if _normalize_path(src) in selected
    }
    filtered_tag_index = {
        dst: data for dst, data in base_plan.tag_index.items() if dst in filtered_moves.values()
    }
    replacement_targets = {
        src: _abs_path(selected[_normalize_path(src)].existing.path)  # type: ignore[union-attr]
        for src in filtered_moves
        if selected[_normalize_path(src)].existing is not None
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
    base_plan.snapshot = _plan_snapshot(base_plan.items, base_plan.replacement_targets)
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
    notes: Iterable[str] = (),
) -> str:
    """Write an executed-plan HTML using the indexer preview style."""
    import html as _html

    esc = _html.escape
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
    for note in notes:
        sections.append(f"<p>{esc(note)}</p>")

    def _format_section(title: str, items: List[Dict[str, str]]) -> None:
        if not items:
            return
        sections.append(f"<h3>{esc(title)}</h3>")
        sections.append("<ul>")
        for item in items:
            # Escaped, not "sanitized": reasons carry paths (e.g. a leftover
            # temporary file) that must stay readable.
            src = esc(os.path.basename(str(item.get("source", ""))))
            dst = esc(os.path.basename(str(item.get("destination", ""))))
            reason = esc(str(item.get("reason", "") or ""))
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


# ── Reversible file transfers ────────────────────────────────────────────────
#
# Every step that changes the library can be undone if a later one fails:
# content is written under a temporary name in the destination folder and
# renamed to its final name only once it is complete (a failed or interrupted
# copy never leaves a truncated file under a real track name); a move to
# another drive copies first and deletes the original only after everything
# else succeeded; a Replace backs the existing file up before displacing it.
# When an undo itself fails the leftover path is reported, never abandoned.

_STAGE_SUFFIX = ".alphadex-incoming"
_PARTIAL_SUFFIX = ".alphadex-partial"
_COPY_CHUNK = 1024 * 1024


class TransferError(Exception):
    """A transfer step failed.

    ``undone`` describes the state it was left in (what was put back, or
    where a file that could not be put back now is); ``leftover`` is that
    file's path (None when everything was undone).
    """

    def __init__(self, message: str, *, undone: str = "", leftover: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.undone = undone
        self.leftover = leftover

    def describe(self) -> str:
        parts = [self.message.rstrip(".") + "."]
        if self.undone:
            parts.append(self.undone)
        return " ".join(parts)


def _discard(path: str) -> str | None:
    """Delete a file this run created; returns ``path`` if that failed."""
    try:
        if os.path.lexists(path):
            os.unlink(path)
        return None
    except OSError:
        return path


def _make_parents(path: str) -> List[str]:
    """Create the missing parent folders of ``path``; return them deepest first."""
    created: List[str] = []
    parent = os.path.dirname(path)
    while parent and not os.path.isdir(parent):
        created.append(parent)
        nxt = os.path.dirname(parent)
        if nxt == parent:
            break
        parent = nxt
    if created:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return created


def _remove_if_empty(folders: Iterable[str]) -> None:
    for folder in folders:          # deepest first
        try:
            os.rmdir(folder)
        except OSError:
            break


def _is_cross_device(exc: OSError) -> bool:
    return exc.errno == errno.EXDEV or getattr(exc, "winerror", None) == 17


def _same_device(path: str, folder: str) -> bool:
    try:
        return os.stat(path).st_dev == os.stat(folder).st_dev
    except OSError:
        return False


def _rename_no_clobber(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, "destination exists; not overwritten", dst)
    os.rename(src, dst)


def _copy_new(src: str, tmp: str) -> None:
    """Copy ``src`` into the *new* file ``tmp`` (never an existing one), flush
    it to disk and copy the timestamps. On any failure the partial ``tmp`` is
    deleted and :class:`TransferError` raised."""
    created = False
    try:
        with open(src, "rb") as fsrc:
            with open(tmp, "xb") as fdst:
                created = True
                shutil.copyfileobj(fsrc, fdst, _COPY_CHUNK)
                fdst.flush()
                try:
                    os.fsync(fdst.fileno())
                except OSError:
                    pass
        try:
            shutil.copystat(src, tmp)
        except OSError:
            pass
    except Exception as exc:
        leftover = _discard(tmp) if created else None
        if leftover:
            undone = f"The partial copy could not be removed and was left at {leftover}."
        else:
            undone = "The partial copy was removed." if created else ""
        raise TransferError(f"copying failed ({exc})", undone=undone, leftover=leftover) from exc


def _copy_file(src: str, dst: str) -> None:
    """Copy ``src`` to the free name ``dst`` through a temporary file, so
    ``dst`` only ever exists complete."""
    _make_parents(dst)
    tmp = _unique_file_path(dst + _PARTIAL_SUFFIX)
    _copy_new(src, tmp)
    try:
        _rename_no_clobber(tmp, dst)
    except OSError:
        _discard(tmp)
        raise


@dataclass
class _Staged:
    """An incoming file's content placed in the library (under a temporary
    or its final name), and how to take it back out."""

    source: str
    path: str
    renamed: bool                     # the source itself was renamed here (one drive, move)
    created_dirs: List[str] = field(default_factory=list)

    def undo(self) -> Tuple[str, str | None]:
        """Take the file back out of the library — rename it back to its
        source, or delete this run's copy. Returns ``(what was done, leftover)``."""
        if self.renamed:
            try:
                _rename_no_clobber(self.path, self.source)
            except OSError as exc:
                return (f"The incoming file could not be put back ({exc}); it is now at "
                        f"{self.path} — move it back to {self.source}."), self.path
            undone = f"The incoming file was put back at {self.source}."
        else:
            leftover = _discard(self.path)
            if leftover:
                return (f"This run's copy could not be removed and was left at {leftover}; "
                        "the incoming file is untouched."), leftover
            undone = "This run's copy was removed; the incoming file is untouched."
        _remove_if_empty(self.created_dirs)
        return undone, None

    def finish(self, action: str) -> str | None:
        """Complete a move that had to copy (another drive): delete the
        original. Returns a problem message, or None."""
        if action != "move" or self.renamed:
            return None
        try:
            os.unlink(self.source)
        except OSError as exc:
            return ("it was copied in, but the original could not be removed from the "
                    f"incoming folder ({exc})")
        return None


def _stage(src: str, final: str, action: str) -> _Staged:
    """Put ``src``'s content next to ``final`` under a temporary name: a
    rename for a move on one drive, otherwise a complete, flushed copy."""
    tmp = _unique_file_path(final + _STAGE_SUFFIX)
    if action == "move" and _same_device(src, os.path.dirname(final)):
        try:
            os.rename(src, tmp)
            return _Staged(src, tmp, renamed=True)
        except OSError as exc:
            if not _is_cross_device(exc):
                raise TransferError(f"the incoming file could not be moved ({exc})") from exc
    _copy_new(src, tmp)
    return _Staged(src, tmp, renamed=False)


def _place_new(src: str, dst: str, action: str) -> _Staged:
    """Transfer ``src`` to the free name ``dst``: staged next to it, then
    renamed into place. On failure the library is as it was."""
    if os.path.lexists(dst):
        raise TransferError("the destination exists; not overwritten")
    try:
        created = _make_parents(dst)
    except OSError as exc:
        raise TransferError(f"its folder could not be created ({exc})") from exc
    try:
        staged = _stage(src, dst, action)
    except TransferError:
        _remove_if_empty(created)
        raise
    staged.created_dirs = created
    try:
        _rename_no_clobber(staged.path, dst)
    except OSError as exc:
        undone, leftover = staged.undo()
        raise TransferError(f"it could not be renamed into place ({exc})",
                            undone=undone, leftover=leftover) from exc
    staged.path = dst
    return staged


def _replace_in_place(src: str, dst: str, action: str, backup_path: str) -> _Staged:
    """Swap ``src`` in for the existing ``dst`` (already backed up at
    ``backup_path``) with one atomic replace. On failure the library file is
    untouched — or restored from the backup if it went missing."""
    staged = _stage(src, dst, action)
    try:
        os.replace(staged.path, dst)
    except OSError as exc:
        undone, leftover = staged.undo()
        notes = [undone] if undone else []
        library_intact = os.path.lexists(dst)
        if library_intact:
            notes.insert(0, "The library file was not changed.")
        else:
            try:
                _copy_file(backup_path, dst)
                notes.append("The library file was restored from its backup.")
            except Exception as restore_exc:  # noqa: BLE001
                notes.append(f"The library file is missing and could not be restored "
                             f"from its backup at {backup_path} ({restore_exc}).")
        error = TransferError(f"the library file could not be replaced ({exc})",
                              undone=" ".join(notes), leftover=leftover)
        error.backup_needed = not library_intact
        raise error from exc
    staged.path = dst
    return staged


def _retire(path: str, backup_path: str) -> None:
    """Move the replaced library file ``path`` into the run's backups."""
    _make_parents(backup_path)
    if _same_device(path, os.path.dirname(backup_path)):
        try:
            _rename_no_clobber(path, backup_path)
            return
        except OSError as exc:
            if not _is_cross_device(exc):
                raise
    _copy_file(path, backup_path)
    try:
        os.unlink(path)
    except OSError:
        _discard(backup_path)
        raise


_UNSET = object()


def _stale_reason(
    item: PlannedItem,
    snapshot: Dict[str, Optional[Tuple[int, int]]],
    replacement_target: str | None,
    library_root: str,
) -> str | None:
    """Why ``item`` no longer means what the preview said, or None."""
    if not snapshot:
        return None
    expected = snapshot.get(item.source, _UNSET)
    if expected is not _UNSET and expected is not None and _file_sig(item.source) != expected:
        return f"the incoming file {os.path.basename(item.source)} changed"
    if item.decision != ExecutionDecision.REPLACE:
        return None
    for path in (item.destination, replacement_target):
        if not path:
            continue
        expected = snapshot.get(path, _UNSET)
        if expected is _UNSET:
            continue
        now = _file_sig(path)
        label = _rel_to(path, library_root)
        if expected is None and now is not None:
            return f"{label} appeared"
        if expected is not None and now is None:
            return f"{label}, the file it replaces, is gone"
        if expected is not None and now != expected:
            return f"{label}, the file it replaces, changed"
    return None


LATEST_AUDIT_NAME = "LibrarySyncAudit.json"
LATEST_REPORT_NAME = "LibrarySyncExecuted.html"


def _new_run_paths(docs_dir: str) -> Tuple[str, str, str, str]:
    """``(run_id, backup folder, audit path, report path)`` for a new run —
    none of them used by an earlier run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = 0
    while True:
        run_id = stamp if n == 0 else f"{stamp}_{n}"
        paths = (
            os.path.join(docs_dir, "Backups", "LibrarySync", run_id),
            os.path.join(docs_dir, f"LibrarySyncAudit_{run_id}.json"),
            os.path.join(docs_dir, f"LibrarySyncExecuted_{run_id}.html"),
        )
        if not any(os.path.lexists(p) for p in paths):
            return (run_id, *paths)
        n += 1


def _write_alias(src: str, alias: str, log_callback: Callable[[str], None]) -> str | None:
    """Make ``alias`` a copy of ``src`` (the stable "latest run" name)."""
    tmp = alias + ".tmp"
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, alias)
        return alias
    except OSError as exc:  # pragma: no cover - OS interaction
        _discard(tmp)
        log_callback(f"! Could not update {os.path.basename(alias)}: {exc}")
        return None


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

    Safety checks still apply per item at execution time:

    * an item whose files changed since the plan was built (``plan.snapshot``:
      the incoming file, or the library file a Replace displaces) is skipped
      and reported as stale — a Replace never silently becomes a plain copy;
    * a missing source fails, and an existing destination is never
      overwritten unless the item is a REPLACE;
    * every transfer is reversible: files are copied under a temporary name
      and renamed into place when complete, a move to another drive deletes
      the original only after success, a replaced file is backed up (into a
      per-run folder under ``Docs/Backups/LibrarySync/``) before it is
      displaced, and on any failure the step is undone (what was undone, and
      any leftover file, is in the audit).

    ``replaced`` counts only files that were actually displaced. Each run
    writes its own ``Docs/LibrarySyncAudit_<run>.json`` and
    ``Docs/LibrarySyncExecuted_<run>.html`` (``run_id`` matches the backup
    folder); ``Docs/LibrarySyncAudit.json`` / ``LibrarySyncExecuted.html`` are
    copies of the latest run.
    """

    log_callback = log_callback or (lambda _msg: None)
    progress_callback = progress_callback or (lambda _c, _t, _p, _ph: None)
    cancel_event = cancel_event or threading.Event()

    docs_dir = _ensure_docs_dir(plan.library_root)
    run_id, backup_root, run_audit_path, run_report_path = _new_run_paths(docs_dir)
    latest_audit_path = os.path.join(docs_dir, LATEST_AUDIT_NAME) if audit_path is None else None
    latest_report_path = (
        os.path.join(docs_dir, LATEST_REPORT_NAME) if executed_report_path is None else None
    )
    audit_path = audit_path or run_audit_path
    executed_report_path = executed_report_path or run_report_path
    playlist_root = os.path.join(plan.library_root, "Playlists")
    if playlist_path is None:
        playlist_stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        playlist_path = _unique_file_path(
            os.path.join(playlist_root, f"LibrarySync_Added_{playlist_stamp}.m3u8")
        )

    copy_only_keys = {_normalize_path(p) for p in (getattr(plan, "copy_only", set()) or [])}
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
            copy_only=getattr(plan, "copy_only", set()) or set(),
            allow_all_replacements=allow_all_replacements,
            allowed_replacements=allowed_replacements,
            transfer_mode=getattr(plan, "transfer_mode", "move"),
        )
    snapshot = dict(getattr(plan, "snapshot", {}) or {})

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
        "stale": [],
        "backups": [],
        "leftovers": [],
        "cancelled": False,
        "dry_run": dry_run,
        "run_id": run_id,
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

    def _skip(outcome: Dict[str, object], reason: str, *, review: bool, stale: bool = False) -> None:
        entry = {"source": outcome["source"], "destination": outcome["destination"], "reason": reason}
        summary["skipped"].append(entry)  # type: ignore[union-attr]
        if review:
            summary["review_required"].append(entry)  # type: ignore[union-attr]
        if stale:
            summary["stale"].append(entry)  # type: ignore[union-attr]
            msg = f"{os.path.basename(str(outcome['source']))}: {reason}"
            summary["errors"].append(msg)  # type: ignore[union-attr]
            log_callback(f"! {msg}")
            outcome["stale"] = True
        outcome["status"] = "skipped"
        outcome["reason"] = reason
        audit_items.append(outcome)

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
        action = item.action or ("copy" if _normalize_path(src) in copy_only_keys else "move")
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
            _skip(outcome, item.reason or "Skipped by plan decision",
                  review=item.decision != ExecutionDecision.SKIP_DUPLICATE)
            continue

        if not os.path.exists(src):
            reason = "Source missing"
            summary["errors"].append(f"{reason}: {src}")  # type: ignore[arg-type]
            outcome["status"] = "failed"
            outcome["reason"] = reason
            audit_items.append(outcome)
            continue

        stale = _stale_reason(item, snapshot, replacement_target, plan.library_root)
        if stale:
            _skip(outcome, f"Not done: the library changed since the plan was built — {stale}. "
                           "Scan again and rebuild the plan.", review=True, stale=True)
            continue

        is_replace = item.decision == ExecutionDecision.REPLACE
        dest_exists = os.path.lexists(dst)
        if dest_exists and not is_replace:
            _skip(outcome, "Destination exists; replacement not allowed (the file appeared after "
                           "the plan was built and is never overwritten)", review=True)
            continue
        in_place = is_replace and dest_exists
        retire_target = (
            replacement_target
            if is_replace and replacement_target
            and _normalize_path(replacement_target) != _normalize_path(dst)
            and os.path.lexists(replacement_target)
            else None
        )

        if dry_run:
            outcome["status"] = "dry_run"
            outcome["reason"] = "No changes written (dry run)"
            if retire_target:
                outcome["replaced_path"] = retire_target
            elif in_place:
                outcome["replaced_path"] = dst
            executed_moves[src] = dst
            audit_items.append(outcome)
            continue

        backup_path: str | None = None
        old_backup: str | None = None
        try:
            if in_place:
                # Backed up before it is displaced; the swap itself is atomic.
                backup_path = _backup_path_for(dst)
                before = _file_sig(dst)
                try:
                    _copy_file(dst, backup_path)
                except Exception as exc:  # noqa: BLE001
                    detail = exc.describe() if isinstance(exc, TransferError) else str(exc)
                    backup_path = None
                    raise TransferError(f"backing up the library file failed ({detail}); "
                                        "nothing was replaced") from exc
                try:
                    staged = _replace_in_place(src, dst, action, backup_path)
                except TransferError as exc:
                    if (not getattr(exc, "backup_needed", False) and _file_sig(dst) == before
                            and _discard(backup_path) is None):
                        backup_path = None      # unused: the library file is unchanged
                    raise
            else:
                staged = _place_new(src, dst, action)
                if retire_target:
                    old_backup = _backup_path_for(retire_target)
                    try:
                        _retire(retire_target, old_backup)
                    except Exception as exc:  # noqa: BLE001
                        old_backup = None
                        undone, leftover = staged.undo()
                        raise TransferError(
                            f"the file it replaces ({_rel_to(retire_target, plan.library_root)}) could "
                            f"not be moved to the backups ({exc}), so the replace was undone",
                            undone=undone, leftover=leftover,
                        ) from exc
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            detail = exc.describe() if isinstance(exc, TransferError) else f"{exc}."
            leftover = exc.leftover if isinstance(exc, TransferError) else None
            msg = f"Failed to {action} {src} → {dst}: {detail}"
            summary["errors"].append(msg)  # type: ignore[union-attr]
            outcome["status"] = "failed"
            outcome["reason"] = detail
            if leftover:
                outcome["leftover_path"] = leftover
                summary["leftovers"].append(leftover)  # type: ignore[union-attr]
            if backup_path:     # kept: the library file was restored from it
                outcome["backup_path"] = backup_path
                summary["backups"].append(backup_path)  # type: ignore[union-attr]
            log_callback(msg)
            audit_items.append(outcome)
            continue

        problem = staged.finish(action)
        done = "copy" if action == "copy" or problem else "move"
        counter = "copied" if done == "copy" else "moved"
        summary[counter] = int(summary[counter]) + 1
        outcome["done"] = done
        if problem:
            msg = f"{os.path.basename(src)}: {problem}"
            summary["errors"].append(msg)  # type: ignore[union-attr]
            outcome["note"] = problem
            log_callback(f"! {msg}")
        if in_place:
            summary["replaced"] = int(summary["replaced"]) + 1
            summary["backups"].append(backup_path)  # type: ignore[union-attr]
            outcome["backup_path"] = backup_path
            outcome["replaced_path"] = dst
        elif retire_target:
            # The file it replaces (e.g. an MP3 upgraded to FLAC) is in the
            # run's backup folder — moved there, never deleted.
            summary["replaced"] = int(summary["replaced"]) + 1
            summary["backups"].append(old_backup)  # type: ignore[union-attr]
            outcome["replaced_path"] = retire_target
            outcome["replaced_backup_path"] = old_backup
            playlist_updates[retire_target] = dst
        elif is_replace:
            outcome["replace_note"] = "nothing was replaced: the file it was to replace is gone"
        executed_moves[src] = dst
        outcome["status"] = "success"
        audit_items.append(outcome)

    summary["transferred"] = int(summary.get("moved", 0)) + int(summary.get("copied", 0))
    used_backups = bool(summary["backups"])
    summary["backup_folder"] = backup_root if used_backups else None

    report_heading = "Library Sync (Executed)" if not dry_run else "Library Sync (Dry Run Execution)"

    report_skipped = [item for item in audit_items if item.get("status") == "skipped"]
    report_failed = [item for item in audit_items if item.get("status") == "failed"]
    report_review = [dict(rr) for rr in summary["review_required"]]  # type: ignore[arg-type]
    notes = [f"Run {run_id} — audit: {os.path.basename(audit_path)}."]
    if used_backups:
        notes.append(f"Files this run replaced are backed up in {backup_root}.")
    if summary["leftovers"]:
        notes.append("Temporary files that could not be cleaned up: "
                     + ", ".join(summary["leftovers"]))  # type: ignore[arg-type]
    _render_execution_report(
        plan,
        executed_moves,
        report_skipped,  # type: ignore[arg-type]
        report_failed,  # type: ignore[arg-type]
        report_review,
        executed_report_path,
        report_heading,
        dry_run,
        notes=notes,
    )
    summary["report_path"] = executed_report_path
    summary["audit_path"] = audit_path

    audit_payload = {
        "run_id": run_id,
        "executed_at": timestamp,
        "dry_run": dry_run,
        "backup_folder": summary["backup_folder"],
        "report_path": executed_report_path,
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
    summary["latest_audit_path"] = (
        _write_alias(audit_path, latest_audit_path, log_callback)
        if latest_audit_path and os.path.exists(audit_path) else None
    )
    summary["latest_report_path"] = (
        _write_alias(executed_report_path, latest_report_path, log_callback)
        if latest_report_path and os.path.exists(executed_report_path) else None
    )

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
