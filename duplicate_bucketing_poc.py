from __future__ import annotations

import base64
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
    artwork_data_uri: str | None
    size_bytes: int
    codec: str

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
    metadata_keys: set[tuple[str, str]] = field(default_factory=set)


def _normalize(text: object) -> str:
    if not text:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def _cover_mime(payload: bytes) -> str:
    if payload.startswith(b"\x89PNG"):
        return "image/png"
    if payload.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "image/jpeg"


def _cover_data_uri(payload: bytes, max_bytes: int = 500_000) -> str | None:
    if not payload or len(payload) > max_bytes:
        return None
    mime = _cover_mime(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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
        artwork_data_uri = None
        if cover_payloads:
            for payload in cover_payloads:
                artwork_data_uri = _cover_data_uri(payload)
                if artwork_data_uri:
                    break
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
            artwork_data_uri=artwork_data_uri,
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


def _truncate_path(path: str, max_len: int = 60) -> str:
    if len(path) <= max_len:
        return path
    tail = path[-(max_len - 1) :]
    return f"…{tail}"


def _format_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024 or unit == "GB":
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"


def _bucket_comparisons(
    bucket: Bucket,
    fingerprints: Dict[str, str | None],
    exact_threshold: float,
    near_threshold: float,
    mixed_boost: float,
) -> List[dict]:
    comparisons: List[dict] = []
    for left, right in combinations(bucket.tracks, 2):
        fp_left = fingerprints.get(left)
        fp_right = fingerprints.get(right)
        if not fp_left or not fp_right:
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "distance": None,
                    "verdict": "missing",
                    "effective_near": None,
                    "mixed_codec": None,
                }
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
        comparisons.append(
            {
                "left": left,
                "right": right,
                "distance": distance,
                "verdict": verdict,
                "effective_near": effective_near,
                "mixed_codec": mixed_codec,
            }
        )
    return comparisons


def _expand_bidirectional(comparisons: Iterable[dict]) -> List[dict]:
    expanded: List[dict] = []
    for comp in comparisons:
        expanded.append(comp)
        if comp.get("left") != comp.get("right"):
            mirrored = dict(comp)
            mirrored["left"], mirrored["right"] = comp.get("right"), comp.get("left")
            expanded.append(mirrored)
    return expanded


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

    def bucket_actionability(exact_count: int, near_count: int) -> int:
        return exact_count * 3 + near_count * 2

    bucket_payloads: List[dict] = []
    for bucket in buckets:
        comparisons = _bucket_comparisons(bucket, fingerprints, exact_threshold, near_threshold, mixed_boost)
        expanded_comparisons = _expand_bidirectional(comparisons)
        exact_count = sum(1 for comp in comparisons if comp["verdict"] == "exact")
        near_count = sum(1 for comp in comparisons if comp["verdict"] == "near")
        no_match_count = sum(1 for comp in comparisons if comp["verdict"] == "no match")
        missing_count = sum(1 for comp in comparisons if comp["verdict"] == "missing")
        best_distance = min(
            (comp["distance"] for comp in comparisons if comp["distance"] is not None),
            default=None,
        )
        metadata_grouped = sum(1 for path in bucket.tracks if bucket.sources.get(path) == "metadata")
        artwork_merged = sum(1 for path in bucket.tracks if bucket.sources.get(path) == "artwork")
        bucket_payloads.append(
            {
                "bucket": bucket,
                "comparisons": comparisons,
                "expanded_comparisons": expanded_comparisons,
                "exact_count": exact_count,
                "near_count": near_count,
                "no_match_count": no_match_count,
                "missing_count": missing_count,
                "best_distance": best_distance,
                "metadata_grouped": metadata_grouped,
                "artwork_merged": artwork_merged,
                "actionability": bucket_actionability(exact_count, near_count),
            }
        )

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
        "h3{font-size:14px; margin-top:18px;}",
        "table{border-collapse:collapse; width:100%; margin-top:8px;}",
        "th,td{border:1px solid #ddd; padding:6px 8px; text-align:left; vertical-align:top;}",
        "th{background:#f4f4f4; font-weight:600;}",
        "code{font-size:12px;}",
        ".muted{color:#666; font-size:12px;}",
        ".dashboard-controls{display:flex; gap:12px; align-items:center; margin:12px 0;}",
        ".dashboard-table th{cursor:pointer; white-space:nowrap;}",
        ".dashboard-table td{font-size:12px;}",
        ".bucket-card{border:1px solid #ddd; border-radius:6px; padding:12px; margin:12px 0; background:#fafafa;}",
        ".compare-grid{display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:10px;}",
        ".track-card{display:flex; gap:10px; border:1px solid #ddd; background:#fff; padding:8px; border-radius:6px;}",
        ".thumb{width:48px; height:48px; flex-shrink:0; border-radius:4px; background:#e6e6e6; object-fit:cover;}",
        ".thumb.placeholder{display:flex; align-items:center; justify-content:center; font-size:10px; color:#555;}",
        ".track-meta{font-size:12px;}",
        ".badge{display:inline-block; padding:2px 6px; border-radius:12px; font-size:11px; background:#e9eef5;}",
        ".verdict-exact{background:#dff3e7;}",
        ".verdict-near{background:#fff3cd;}",
        "details summary{cursor:pointer; font-weight:600;}",
        ".subsection{margin-top:10px;}",
        ".compare-header{display:flex; align-items:center; justify-content:space-between; gap:12px;}",
        ".comparison-row{border-bottom:1px solid #eee; padding:6px 0; font-size:12px;}",
        ".comparison-row:last-child{border-bottom:none;}",
        "</style>",
        "<script>",
        "function sortTable(tableId, colIndex, numeric){",
        "  const table = document.getElementById(tableId);",
        "  const tbody = table.tBodies[0];",
        "  const rows = Array.from(tbody.rows);",
        "  const dir = table.getAttribute('data-sort-dir') === 'asc' ? 'desc' : 'asc';",
        "  rows.sort((a,b)=>{",
        "    const av = a.cells[colIndex].getAttribute('data-sort') || a.cells[colIndex].innerText;",
        "    const bv = b.cells[colIndex].getAttribute('data-sort') || b.cells[colIndex].innerText;",
        "    const left = numeric ? parseFloat(av) || 0 : av.toString();",
        "    const right = numeric ? parseFloat(bv) || 0 : bv.toString();",
        "    if(left < right) return dir === 'asc' ? -1 : 1;",
        "    if(left > right) return dir === 'asc' ? 1 : -1;",
        "    return 0;",
        "  });",
        "  rows.forEach(row => tbody.appendChild(row));",
        "  table.setAttribute('data-sort-dir', dir);",
        "}",
        "function applyDashboardFilter(){",
        "  const toggle = document.getElementById('show-all-buckets');",
        "  const showAll = toggle && toggle.checked;",
        "  document.querySelectorAll('[data-has-match]').forEach(row => {",
        "    const hasMatch = row.getAttribute('data-has-match') === '1';",
        "    row.style.display = showAll || hasMatch ? '' : 'none';",
        "  });",
        "  document.querySelectorAll('[data-bucket-details]').forEach(panel => {",
        "    const hasMatch = panel.getAttribute('data-has-match') === '1';",
        "    panel.style.display = showAll || hasMatch ? '' : 'none';",
        "  });",
        "}",
        "window.addEventListener('load', applyDashboardFilter);",
        "</script>",
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
        "<h2>Dashboard</h2>",
        "<div class='dashboard-controls'>",
        "<label><input type='checkbox' id='show-all-buckets' onchange='applyDashboardFilter()' /> Show buckets with no matches</label>",
        "<span class='muted'>Default view shows buckets with at least one exact or near match.</span>",
        "</div>",
        "<table class='dashboard-table' id='bucket-dashboard' data-sort-dir='desc'>",
        "<thead>",
        "<tr>",
        "<th onclick=\"sortTable('bucket-dashboard',0,false)\">Bucket</th>",
        "<th onclick=\"sortTable('bucket-dashboard',1,true)\">Size</th>",
        "<th onclick=\"sortTable('bucket-dashboard',2,true)\">Exact</th>",
        "<th onclick=\"sortTable('bucket-dashboard',3,true)\">Near</th>",
        "<th onclick=\"sortTable('bucket-dashboard',4,true)\">No match</th>",
        "<th onclick=\"sortTable('bucket-dashboard',5,true)\">Metadata grouped</th>",
        "<th onclick=\"sortTable('bucket-dashboard',6,true)\">Artwork merged</th>",
        "<th onclick=\"sortTable('bucket-dashboard',7,true)\">Best distance</th>",
        "</tr>",
        "</thead>",
        "<tbody>",
    ]

    for payload in sorted(
        bucket_payloads,
        key=lambda p: (-p["actionability"], -len(p["bucket"].tracks), p["bucket"].id),
    ):
        bucket = payload["bucket"]
        has_matches = payload["exact_count"] + payload["near_count"] > 0
        best_distance = payload["best_distance"]
        html_lines.append(
            "<tr data-has-match='{}'>".format("1" if has_matches else "0")
            + f"<td data-sort='{bucket.id}'><a href='#bucket-{bucket.id}'>Bucket {bucket.id}</a></td>"
            + f"<td data-sort='{len(bucket.tracks)}'>{len(bucket.tracks)}</td>"
            + f"<td data-sort='{payload['exact_count']}'>{payload['exact_count']}</td>"
            + f"<td data-sort='{payload['near_count']}'>{payload['near_count']}</td>"
            + f"<td data-sort='{payload['no_match_count']}'>{payload['no_match_count']}</td>"
            + f"<td data-sort='{payload['metadata_grouped']}'>{payload['metadata_grouped']}</td>"
            + f"<td data-sort='{payload['artwork_merged']}'>{payload['artwork_merged']}</td>"
            + (
                f"<td data-sort='{best_distance:.4f}'>{best_distance:.4f}</td>"
                if best_distance is not None
                else "<td data-sort=''>-</td>"
            )
            + "</tr>"
        )

    html_lines.extend(["</tbody>", "</table>", "<h2>Drilldown</h2>"])

    for payload in sorted(
        bucket_payloads,
        key=lambda p: (-p["actionability"], -len(p["bucket"].tracks), p["bucket"].id),
    ):
        bucket = payload["bucket"]
        metadata_tracks = [p for p in bucket.tracks if bucket.sources.get(p) == "metadata"]
        artwork_tracks = [p for p in bucket.tracks if bucket.sources.get(p) == "artwork"]
        solo_tracks = [p for p in bucket.tracks if bucket.sources.get(p) == "solo"]
        has_matches = payload["exact_count"] + payload["near_count"] > 0
        html_lines.append(
            f"<details id='bucket-{bucket.id}' class='bucket-card' data-bucket-details data-has-match='{'1' if has_matches else '0'}'>"
        )
        html_lines.append(
            "<summary>"
            f"Bucket {bucket.id} • size {len(bucket.tracks)} • "
            f"{payload['exact_count']} exact / {payload['near_count']} near"
            "</summary>"
        )
        html_lines.append("<div class='subsection'>")
        html_lines.append(
            "<div class='compare-header'>"
            "<h3>Matches</h3>"
            f"<span class='muted'>Showing {sum(1 for comp in payload['expanded_comparisons'] if comp['verdict'] in {'exact', 'near'})} near/exact pairs (bidirectional)</span>"
            "</div>"
        )
        match_pairs = [
            comp for comp in payload["expanded_comparisons"] if comp["verdict"] in {"exact", "near"}
        ]
        if match_pairs:
            for comp in match_pairs:
                left_info = track_infos.get(comp["left"])
                right_info = track_infos.get(comp["right"])
                verdict_class = "verdict-exact" if comp["verdict"] == "exact" else "verdict-near"
                html_lines.append("<div class='compare-grid'>")
                for label, info in (("Track A", left_info), ("Track B", right_info)):
                    if info and info.artwork_data_uri:
                        thumb = f"<img class='thumb' src='{info.artwork_data_uri}' alt='artwork' />"
                    else:
                        thumb = "<div class='thumb placeholder'>no art</div>"
                    path_display = _truncate_path(info.path) if info else "unknown"
                    full_path = info.path if info else ""
                    html_lines.append(
                        "<div class='track-card'>"
                        f"{thumb}"
                        "<div class='track-meta'>"
                        f"<div><strong>{label}</strong></div>"
                        f"<div>{esc(os.path.basename(info.path) if info else 'unknown')}</div>"
                        f"<div class='muted' title='{esc(full_path)}'>{esc(path_display)}</div>"
                        f"<div class='muted'>{esc(info.codec if info else 'UNKNOWN')} • "
                        f"{esc(_format_size(info.size_bytes) if info else 'unknown')}</div>"
                        "</div>"
                        "</div>"
                    )
                distance = comp["distance"]
                effective = comp["effective_near"]
                html_lines.append(
                    "<div class='track-card'>"
                    "<div class='track-meta'>"
                    f"<div class='badge {verdict_class}'>{comp['verdict'].upper()}</div>"
                    f"<div class='muted'>distance {distance:.4f}</div>"
                    f"<div class='muted'>effective near {effective:.4f}</div>"
                    "</div>"
                    "</div>"
                )
                html_lines.append("</div>")
        else:
            html_lines.append("<div class='muted'>No near/exact matches in this bucket.</div>")
        html_lines.append("</div>")

        html_lines.append("<details class='subsection'>")
        html_lines.append("<summary>All comparisons</summary>")
        html_lines.append(
            "<div class='muted'>Includes no-match and missing fingerprint rows (bidirectional).</div>"
        )
        if payload["expanded_comparisons"]:
            for comp in payload["expanded_comparisons"]:
                left = esc(_truncate_path(comp["left"]))
                right = esc(_truncate_path(comp["right"]))
                if comp["distance"] is None:
                    detail = "missing fingerprint"
                else:
                    detail = (
                        f"distance {comp['distance']:.4f} • "
                        f"effective near {comp['effective_near']:.4f}"
                    )
                html_lines.append(
                    "<div class='comparison-row'>"
                    f"<strong>{esc(comp['verdict'])}</strong>: "
                    f"{left} vs {right} • {detail}"
                    "</div>"
                )
        else:
            html_lines.append("<div class='comparison-row'>No comparisons (bucket size &lt; 2).</div>")
        html_lines.append("</details>")

        html_lines.append("<details class='subsection'>")
        html_lines.append("<summary>Why bucketed</summary>")
        if bucket.metadata_keys:
            html_lines.append("<div><strong>Metadata keys:</strong></div><ul>")
            for title_norm, artist_norm in sorted(bucket.metadata_keys):
                html_lines.append(
                    f"<li><code>{esc(title_norm)}</code> • <code>{esc(artist_norm)}</code></li>"
                )
            html_lines.append("</ul>")
        else:
            html_lines.append("<div class='muted'>No metadata keys (solo bucket).</div>")
        html_lines.append(
            "<div class='muted'>"
            f"Metadata-grouped tracks: {len(metadata_tracks)} • "
            f"Artwork-merged tracks: {len(artwork_tracks)} • "
            f"Solo/ungrouped tracks: {len(solo_tracks)}"
            "</div>"
        )
        if bucket.artwork_merges:
            html_lines.append("<div><strong>Artwork merge events:</strong></div><ul>")
            for event in bucket.artwork_merges:
                html_lines.append(
                    "<li>"
                    f"<code>{esc(event.left)}</code> ↔ <code>{esc(event.right)}</code> "
                    f"(distance {event.distance})"
                    "</li>"
                )
            html_lines.append("</ul>")
        else:
            html_lines.append("<div class='muted'>No artwork merge events.</div>")
        html_lines.append("</details>")
        html_lines.append("</details>")

    elapsed = time.time() - start
    html_lines.append(f"<div>Elapsed: {elapsed:.2f}s</div>")
    html_lines.extend(["</body>", "</html>"])

    report_path = os.path.join(docs_dir, "duplicate_bucketing_poc_report.html")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(html_lines))

    log_callback(f"Duplicate Bucketing POC report saved to {report_path}")
    return report_path
