from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from controllers.tagfix_controller import discover_files
from duplicate_consolidation import (
    ARTWORK_SIMILARITY_THRESHOLD,
    _artwork_perceptual_hash,
    _hamming_distance,
)
from fingerprint_cache import get_fingerprint
from near_duplicate_detector import LOSSLESS_EXTS, fingerprint_distance
from utils.audio_metadata_reader import read_metadata
from utils.path_helpers import ensure_long_path


@dataclass
class TrackInfo:
    path: str
    title_norm: str
    artist_norm: str
    album_norm: str
    artwork_hash: int | None

    @property
    def has_metadata(self) -> bool:
        return bool(self.title_norm and self.artist_norm)


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


def _normalize(text: object) -> str:
    if not text:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _extract_track_info(paths: Iterable[str], log_callback: Callable[[str], None]) -> Dict[str, TrackInfo]:
    info: Dict[str, TrackInfo] = {}
    for path in paths:
        tags, cover_payloads, error, _reader = read_metadata(path, include_cover=True)
        if error:
            log_callback(f"Metadata read failed for {path}: {error}")
        title_norm = _normalize(tags.get("title"))
        artist_norm = _normalize(tags.get("albumartist") or tags.get("artist"))
        album_norm = _normalize(tags.get("album"))
        artwork_hash = None
        if cover_payloads:
            artwork_hash = _artwork_perceptual_hash(cover_payloads[0])
        info[path] = TrackInfo(
            path=path,
            title_norm=title_norm,
            artist_norm=artist_norm,
            album_norm=album_norm,
            artwork_hash=artwork_hash,
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
        if info.album_norm:
            bucket.albums.add(info.album_norm)
        else:
            bucket.missing_album = True

    for path in unassigned:
        bucket_id = len(buckets)
        bucket = Bucket(id=bucket_id, metadata_seeded=False)
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


def _compute_fingerprints(
    paths: Iterable[str],
    db_path: str,
    log_callback: Callable[[str], None],
    max_workers: int | None = None,
) -> Dict[str, str | None]:
    cfg = load_config()
    start_sec = float(cfg.get("fingerprint_offset_ms", FP_OFFSET_MS)) / 1000.0
    duration_sec = float(cfg.get("fingerprint_duration_ms", FP_DURATION_MS)) / 1000.0
    duration_sec = duration_sec if duration_sec > 0 else 120.0
    trim_silence = bool(cfg.get("trim_silence", False))
    silence_threshold_db = float(cfg.get("fingerprint_silence_threshold_db", FP_SILENCE_THRESHOLD_DB))
    silence_min_len_ms = float(cfg.get("fingerprint_silence_min_len_ms", FP_SILENCE_MIN_LEN_MS))

    def compute(path: str) -> tuple[Optional[int], Optional[str]]:
        try:
            fp = chromaprint_utils.fingerprint_fpcalc(
                ensure_long_path(path),
                trim=trim_silence,
                start_sec=start_sec,
                duration_sec=duration_sec,
                threshold_db=silence_threshold_db,
                min_silence_duration=silence_min_len_ms / 1000.0,
            )
            return 0, fp
        except chromaprint_utils.FingerprintError as exc:
            log_callback(f"Fingerprint failed for {path}: {exc}")
            return None, None
        except Exception as exc:  # pragma: no cover - guard against unexpected failures
            log_callback(f"Unexpected fingerprint error for {path}: {exc}")
            return None, None

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
                log_callback(f"Fingerprint task failed for {path}: {exc}")
                results[path] = None
    return results


def _is_lossless(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in LOSSLESS_EXTS


def _bucket_report(
    bucket: Bucket,
    fingerprints: Dict[str, str | None],
    exact_threshold: float,
    near_threshold: float,
    mixed_boost: float,
) -> tuple[List[str], List[str]]:
    track_lines = []
    for path in bucket.tracks:
        reason = bucket.sources.get(path, "unknown")
        track_lines.append(f"<li><code>{html.escape(path)}</code> <em>({html.escape(reason)})</em></li>")

    match_lines: List[str] = []
    for left, right in combinations(bucket.tracks, 2):
        fp_left = fingerprints.get(left)
        fp_right = fingerprints.get(right)
        if not fp_left or not fp_right:
            match_lines.append(
                "<li>"
                f"<code>{html.escape(left)}</code> vs <code>{html.escape(right)}</code>: "
                "missing fingerprint</li>"
            )
            continue
        distance = fingerprint_distance(fp_left, fp_right)
        mixed_codec = _is_lossless(left) != _is_lossless(right)
        effective_near = near_threshold + (mixed_boost if mixed_codec else 0.0)
        if distance <= exact_threshold:
            verdict = "exact"
        elif distance <= effective_near:
            verdict = "near"
        else:
            verdict = "no match"
        match_lines.append(
            "<li>"
            f"<code>{html.escape(left)}</code> vs <code>{html.escape(right)}</code>: "
            f"distance={distance:.4f}, verdict={verdict}, "
            f"near_threshold={effective_near:.4f}"
            "</li>"
        )
    return track_lines, match_lines


def run_duplicate_bucketing_poc(
    root: str,
    *,
    log_callback: Callable[[str], None] | None = None,
) -> str:
    if log_callback is None:
        def log_callback(msg: str) -> None:
            print(msg)

    start = time.time()
    log_callback(f"Duplicate Bucketing POC: scanning {root}")
    paths = discover_files(root)
    track_infos = _extract_track_info(paths, log_callback)

    buckets = _build_metadata_buckets(track_infos)
    buckets = _merge_buckets_by_artwork(buckets, track_infos, ARTWORK_SIMILARITY_THRESHOLD)

    docs_dir = os.path.join(root, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    db_path = os.path.join(docs_dir, ".duplicate_bucketing_poc_fps.db")

    paths_for_fp = [p for bucket in buckets if len(bucket.tracks) > 1 for p in bucket.tracks]
    fingerprints = _compute_fingerprints(paths_for_fp, db_path, log_callback)

    cfg = load_config()
    exact_threshold = float(cfg.get("exact_duplicate_threshold", EXACT_DUPLICATE_THRESHOLD))
    near_threshold = float(cfg.get("near_duplicate_threshold", NEAR_DUPLICATE_THRESHOLD))
    mixed_boost = float(cfg.get("mixed_codec_threshold_boost", MIXED_CODEC_THRESHOLD_BOOST))

    bucket_sizes = [len(bucket.tracks) for bucket in buckets]
    bucket_sizes_sorted = sorted(bucket_sizes, reverse=True)
    total_tracks = sum(bucket_sizes)
    total_buckets = len(buckets)
    total_artwork_merges = sum(len(bucket.artwork_merges) for bucket in buckets)

    def esc(value: object) -> str:
        return html.escape(str(value))

    html_lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8' />",
        "<title>Duplicate Bucketing POC Report</title>",
        "<style>",
        "body{font-family:Arial, sans-serif; margin:24px; color:#222;}",
        "h1{font-size:20px; margin-bottom:6px;}",
        "h2{font-size:16px; margin-top:24px;}",
        "table{border-collapse:collapse; width:100%; margin-top:8px;}",
        "th,td{border:1px solid #ddd; padding:8px; text-align:left; vertical-align:top;}",
        "th{background:#f4f4f4; width:200px;}",
        "code{font-size:12px;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Duplicate Bucketing POC Report</h1>",
        f"<div>Target: <code>{esc(root)}</code></div>",
        f"<div>Generated: {esc(time.strftime('%Y-%m-%d %H:%M:%S'))}</div>",
        "<h2>Summary</h2>",
        "<table>",
        f"<tr><th>Total tracks scanned</th><td>{total_tracks}</td></tr>",
        f"<tr><th>Total buckets</th><td>{total_buckets}</td></tr>",
        f"<tr><th>Bucket sizes</th><td>{esc(bucket_sizes_sorted)}</td></tr>",
        f"<tr><th>Artwork merge events</th><td>{total_artwork_merges}</td></tr>",
        f"<tr><th>Exact threshold</th><td>{exact_threshold:.4f}</td></tr>",
        f"<tr><th>Near threshold</th><td>{near_threshold:.4f}</td></tr>",
        f"<tr><th>Mixed-codec boost</th><td>{mixed_boost:.4f}</td></tr>",
        "</table>",
    ]

    for idx, bucket in enumerate(sorted(buckets, key=lambda b: (-len(b.tracks), b.id)), start=1):
        metadata_tracks = [p for p in bucket.tracks if bucket.sources.get(p) == "metadata"]
        artwork_tracks = [p for p in bucket.tracks if bucket.sources.get(p) == "artwork"]
        solo_tracks = [p for p in bucket.tracks if bucket.sources.get(p) == "solo"]
        html_lines.append(f"<h2>Bucket {idx} (size {len(bucket.tracks)})</h2>")
        html_lines.append("<table>")
        html_lines.append(f"<tr><th>Metadata-grouped tracks</th><td>{len(metadata_tracks)}</td></tr>")
        html_lines.append(f"<tr><th>Artwork-merged tracks</th><td>{len(artwork_tracks)}</td></tr>")
        html_lines.append(f"<tr><th>Solo/ungrouped tracks</th><td>{len(solo_tracks)}</td></tr>")
        html_lines.append("</table>")
        if bucket.artwork_merges:
            html_lines.append("<h3>Artwork merges</h3><ul>")
            for event in bucket.artwork_merges:
                html_lines.append(
                    "<li>"
                    f"<code>{esc(event.left)}</code> ↔ <code>{esc(event.right)}</code> "
                    f"(distance {event.distance})"
                    "</li>"
                )
            html_lines.append("</ul>")
        track_lines, match_lines = _bucket_report(
            bucket, fingerprints, exact_threshold, near_threshold, mixed_boost
        )
        html_lines.append("<h3>Tracks</h3><ul>")
        html_lines.extend(track_lines)
        html_lines.append("</ul>")
        html_lines.append("<h3>Fingerprint comparisons</h3><ul>")
        html_lines.extend(match_lines or ["<li>No comparisons (bucket size < 2).</li>"])
        html_lines.append("</ul>")

    elapsed = time.time() - start
    html_lines.append(f"<div>Elapsed: {elapsed:.2f}s</div>")
    html_lines.extend(["</body>", "</html>"])

    report_path = os.path.join(docs_dir, "duplicate_bucketing_poc_report.html")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(html_lines))

    log_callback(f"Duplicate Bucketing POC report saved to {report_path}")
    return report_path
