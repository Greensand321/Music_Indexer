import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, List, Tuple, Dict, Optional

import chromaprint_utils
from config import (
    EXACT_DUPLICATE_THRESHOLD,
    FP_DURATION_MS,
    FP_OFFSET_MS,
    FP_SILENCE_MIN_LEN_MS,
    FP_SILENCE_THRESHOLD_DB,
    MIXED_CODEC_THRESHOLD_BOOST,
    NEAR_DUPLICATE_THRESHOLD,
    load_config,
)
from duplicate_consolidation import (
    ARTWORK_SIMILARITY_THRESHOLD,
    _artwork_perceptual_hash,
    _hamming_distance,
)
from fingerprint_cache import get_fingerprint
from near_duplicate_detector import LOSSLESS_EXTS, fingerprint_distance
from utils.audio_metadata_reader import read_metadata
from utils.path_helpers import ensure_long_path

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


@dataclass
class TrackInfo:
    path: str
    title_norm: str
    artist_norm: str
    album_norm: str
    artwork_hash: int | None
    size_bytes: int
    codec: str

    @property
    def has_metadata(self) -> bool:
        return bool(self.title_norm or self.artist_norm)


@dataclass
class ArtworkMerge:
    left: str
    right: str
    distance: int


@dataclass
class Bucket:
    id: int
    tracks: List[str] = field(default_factory=list)
    metadata_seeded: bool = False
    albums: set[str] = field(default_factory=set)
    missing_album: bool = False
    sources: Dict[str, str] = field(default_factory=dict)
    artwork_merges: List[ArtworkMerge] = field(default_factory=list)
    metadata_keys: set[tuple[str, str]] = field(default_factory=set)


def _normalize(text: object) -> str:
    if not text:
        return ""
    return " ".join("".join(ch if ch.isalnum() else " " for ch in str(text).lower()).split())


def _extract_track_info(
    paths: List[str],
    log_callback: Callable[[str], None],
) -> Dict[str, TrackInfo]:
    info: Dict[str, TrackInfo] = {}
    for path in paths:
        tags, cover_payloads, error, _reader = read_metadata(path, include_cover=True)
        if error:
            log_callback(f"Metadata read failed for {path}: {error}")
        title_raw = tags.get("title")
        artist_raw = tags.get("albumartist") or tags.get("artist")
        album_raw = tags.get("album")
        title_norm = _normalize(title_raw)
        artist_norm = _normalize(artist_raw)
        album_norm = _normalize(album_raw)
        artwork_hash = None
        if cover_payloads:
            artwork_hash = _artwork_perceptual_hash(cover_payloads[0])
        try:
            size_bytes = os.path.getsize(path)
        except OSError:
            size_bytes = 0
        codec = os.path.splitext(path)[1].lstrip(".").upper() or "UNKNOWN"
        info[path] = TrackInfo(
            path=path,
            title_norm=title_norm,
            artist_norm=artist_norm,
            album_norm=album_norm,
            artwork_hash=artwork_hash,
            size_bytes=size_bytes,
            codec=codec,
        )
    return info


def _album_compatible(bucket: Bucket, album_norm: str) -> bool:
    if not album_norm:
        return True
    if bucket.missing_album:
        return True
    if not bucket.albums:
        return True
    return album_norm in bucket.albums


def _build_metadata_buckets(track_infos: Dict[str, TrackInfo]) -> List[Bucket]:
    buckets: List[Bucket] = []
    by_key: Dict[tuple[str, str], List[int]] = {}
    unassigned: List[str] = []

    for path, info in sorted(track_infos.items(), key=lambda item: item[0]):
        if not info.has_metadata:
            unassigned.append(path)
            continue
        key = (info.title_norm, info.artist_norm)
        target_id = None
        for bucket_id in by_key.get(key, []):
            bucket = buckets[bucket_id]
            if _album_compatible(bucket, info.album_norm):
                target_id = bucket_id
                break
        if target_id is None:
            target_id = len(buckets)
            buckets.append(Bucket(id=target_id, metadata_seeded=True))
            by_key.setdefault(key, []).append(target_id)
        bucket = buckets[target_id]
        bucket.tracks.append(path)
        bucket.sources[path] = "metadata"
        bucket.metadata_keys.add(key)
        if info.album_norm:
            bucket.albums.add(info.album_norm)
        else:
            bucket.missing_album = True

    if unassigned:
        bucket_id = len(buckets)
        bucket = Bucket(id=bucket_id, metadata_seeded=False)
        for path in unassigned:
            bucket.tracks.append(path)
            bucket.sources[path] = "solo"
        buckets.append(bucket)

    return buckets


def _merge_buckets_by_artwork(
    buckets: List[Bucket],
    track_infos: Dict[str, TrackInfo],
    similarity_threshold: int,
) -> List[Bucket]:
    if not buckets:
        return []

    parent = list(range(len(buckets)))
    root_seeded = [bucket.metadata_seeded for bucket in buckets]

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a
            root_seeded[root_a] = root_seeded[root_a] or root_seeded[root_b]

    entries: List[tuple[int, str, int]] = []
    for bucket in buckets:
        for path in bucket.tracks:
            art_hash = track_infos[path].artwork_hash
            if art_hash is not None:
                entries.append((bucket.id, path, art_hash))

    merge_events: List[ArtworkMerge] = []
    for i, (bucket_id, left_path, left_hash) in enumerate(entries):
        for right_bucket_id, right_path, right_hash in entries[i + 1 :]:
            if bucket_id == right_bucket_id:
                continue
            left_root = find(bucket_id)
            right_root = find(right_bucket_id)
            if not (root_seeded[left_root] or root_seeded[right_root]):
                continue
            distance = _hamming_distance(left_hash, right_hash)
            if distance <= similarity_threshold:
                union(bucket_id, right_bucket_id)
                merge_events.append(
                    ArtworkMerge(left=left_path, right=right_path, distance=distance)
                )

    root_metadata_seeded: Dict[int, bool] = {}
    for bucket in buckets:
        root = find(bucket.id)
        root_metadata_seeded[root] = root_metadata_seeded.get(root, False) or bucket.metadata_seeded

    merged: Dict[int, Bucket] = {}
    for bucket in buckets:
        root = find(bucket.id)
        if root not in merged:
            merged[root] = Bucket(id=root)
        target = merged[root]
        target.metadata_seeded = root_metadata_seeded[root]
        target.albums.update(bucket.albums)
        target.missing_album = target.missing_album or bucket.missing_album
        target.metadata_keys.update(bucket.metadata_keys)
        for path in bucket.tracks:
            target.tracks.append(path)
            source = bucket.sources.get(path, "solo")
            if source == "solo" and not bucket.metadata_seeded and root_metadata_seeded[root]:
                source = "artwork"
            target.sources[path] = source

    for event in merge_events:
        left_bucket = None
        for bucket in buckets:
            if event.left in bucket.tracks:
                left_bucket = bucket
                break
        if left_bucket is None:
            continue
        root = find(left_bucket.id)
        merged[root].artwork_merges.append(event)

    for bucket in merged.values():
        bucket.tracks = sorted(set(bucket.tracks))
    return sorted(merged.values(), key=lambda b: b.id)


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
            ensure_long_path(path),
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
) -> List[str]:
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
                if progress_callback:
                    progress_callback("walk", len(paths), 0, path)
                if cancel_event and cancel_event.is_set():
                    return paths
            else:
                _dlog("WALK", f"skip file {fname}")
            if cancel_event and cancel_event.is_set():
                return paths
        if cancel_event and cancel_event.is_set():
            return paths
    return paths


def _compute_fingerprints(
    paths: List[str],
    db_path: str,
    log_callback: Callable[[str], None],
    max_workers: int | None = None,
) -> Dict[str, str | None]:
    def compute(path: str) -> tuple[Optional[int], Optional[str]]:
        return _compute_fp(path)

    results: Dict[str, str | None] = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path in paths:
            futures[executor.submit(get_fingerprint, path, db_path, compute, log_callback=log_callback)] = path
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                results[path] = fut.result()
            except Exception as exc:  # pragma: no cover - defensive
                log_callback(f"! Fingerprint task failed for {path}: {exc}")
                results[path] = None
    return results


def _is_lossless(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in LOSSLESS_EXTS


def find_duplicates(
    root: str,
    threshold: float = 0.03,
    prefix_len: int | None = FP_PREFIX_LEN,
    db_path: Optional[str] = None,
    log_callback: Optional[callable] = None,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    max_workers: int | None = None,
    exact_threshold: float | None = None,
    near_threshold: float | None = None,
    mixed_codec_boost: float | None = None,
) -> Tuple[List[Tuple[str, str]], int]:
    """Return (duplicates, missing_count) for audio files in ``root``.

    ``prefix_len`` is retained for compatibility but no longer drives grouping
    for metadata-seeded buckets.
    """
    if db_path is None:
        db_path = os.path.join(root, "Docs", ".simple_fps.db")

    if log_callback is None:
        log_callback = print

    global _log
    _log = log_callback

    audio_paths = _walk_audio_files(root, progress_callback, cancel_event)
    if cancel_event and cancel_event.is_set():
        return [], 0

    track_infos = _extract_track_info(audio_paths, log_callback)
    buckets = _build_metadata_buckets(track_infos)
    buckets = _merge_buckets_by_artwork(buckets, track_infos, ARTWORK_SIMILARITY_THRESHOLD)

    paths_for_fp = [p for bucket in buckets if len(bucket.tracks) > 1 for p in bucket.tracks]
    total = len(paths_for_fp)
    fingerprints: Dict[str, str | None] = {}

    if paths_for_fp:
        for idx, p in enumerate(paths_for_fp, 1):
            if cancel_event and cancel_event.is_set():
                break
            if progress_callback:
                progress_callback("fp_start", idx, total, p)
        fingerprints = _compute_fingerprints(paths_for_fp, db_path, log_callback, max_workers=max_workers)
        for idx, p in enumerate(paths_for_fp, 1):
            if progress_callback:
                progress_callback("fp_end", idx, total, p)
            if fingerprints.get(p):
                log_callback(f"\u2713 Fingerprinted {p}")
            else:
                log_callback(f"\u2717 No fingerprint for {p}")

    missing_fp = sum(1 for p in paths_for_fp if not fingerprints.get(p))

    cfg = load_config()
    exact = float(exact_threshold) if exact_threshold is not None else float(
        cfg.get("exact_duplicate_threshold", EXACT_DUPLICATE_THRESHOLD)
    )
    near = float(near_threshold) if near_threshold is not None else float(
        cfg.get("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD)
    )
    if threshold is not None:
        exact = float(threshold)
        near = max(near, exact)
    mixed_boost = float(mixed_codec_boost) if mixed_codec_boost is not None else float(
        cfg.get("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST)
    )

    duplicates: List[Tuple[str, str]] = []
    for bucket in buckets:
        if cancel_event and cancel_event.is_set():
            break
        paths = bucket.tracks
        if len(paths) <= 1:
            continue
        comparisons: List[tuple[str, str, float, str, float]] = []
        for left, right in combinations(paths, 2):
            fp_left = fingerprints.get(left)
            fp_right = fingerprints.get(right)
            if not fp_left or not fp_right:
                log_callback(f"[DIST] {left} \u2194 {right} missing fingerprint")
                continue
            distance = fingerprint_distance(fp_left, fp_right)
            mixed_codec = _is_lossless(left) != _is_lossless(right)
            effective_near = near + (mixed_boost if mixed_codec else 0.0)
            if distance <= exact:
                verdict = "exact"
            elif distance <= effective_near:
                verdict = "near"
            else:
                verdict = "no match"
            comparisons.append((left, right, distance, verdict, effective_near))
            log_callback(
                f"[DIST] {left} \u2194 {right} distance={distance:.4f} "
                f"({verdict}, exact={exact:.4f}, near={effective_near:.4f})"
            )

        adjacency: Dict[str, set[str]] = {}
        for left, right, _distance, verdict, _effective in comparisons:
            if verdict not in {"exact", "near"}:
                continue
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)

        seen: set[str] = set()
        for node in adjacency:
            if node in seen:
                continue
            stack = [node]
            component: List[str] = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                component.append(current)
                stack.extend(adjacency.get(current, set()) - seen)
            if len(component) <= 1:
                continue
            scored = sorted(component, key=lambda p: _keep_score(p, EXT_PRIORITY), reverse=True)
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
