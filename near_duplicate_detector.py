# near_duplicate_detector.py
"""Helpers for fuzzy fingerprint grouping."""
from __future__ import annotations

import os
import re
import hashlib
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Set, List, Iterable, Tuple, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import base64
from itertools import combinations

logger = logging.getLogger(__name__)

EXCLUSION_KEYWORDS = ['remix', 'remastered', 'edit', 'version']
COARSE_FP_BAND_SIZE = 8
COARSE_FP_BANDS = 6
COARSE_FP_QUANTIZATION = 4
LOSSLESS_EXTS = {".flac", ".wav", ".alac", ".ape", ".aiff", ".aif"}


def _keep_score(path: str, info: Mapping[str, object], ext_priority: Mapping[str, int]) -> float:
    """Compute keep score for a file based on extension, metadata and filename."""
    ext = os.path.splitext(path)[1].lower()
    pri = ext_priority.get(ext, 99)
    ext_score = 1000.0 / (pri + 1)
    meta_score = int(info.get("meta_count", 0) or 0) * 10
    fname_score = len(os.path.splitext(os.path.basename(path))[0])
    return ext_score + meta_score + fname_score


@dataclass
class NearDuplicateGroup:
    winner: str
    losers: List[str]
    max_distance: float
    reason: str


@dataclass
class NearDuplicateResult:
    auto_deletes: Dict[str, str]
    review_groups: List[NearDuplicateGroup]

    @property
    def review_required(self) -> bool:
        return bool(self.review_groups)

    def items(self):
        return self.auto_deletes.items()


def _parse_fp(fp: str) -> tuple[str, List[int] | bytes] | None:
    """Parse fingerprint string as integer list or base64 bytes."""
    text = fp.strip()
    if not text:
        return None
    try:
        arr = [int(x) for x in text.replace(',', ' ').split()]
        if arr:
            return ("ints", arr)
    except Exception:
        pass
    try:
        data = base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))
        if data:
            return ("bytes", data)
    except Exception:
        pass
    return None


def _hamming_ints(a: List[int], b: List[int]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = sum(x != y for x, y in zip(a[:n], b[:n]))
    return diff / n


def _hamming_bytes(a: bytes, b: bytes) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    diff = 0
    for i in range(n):
        diff += (a[i] ^ b[i]).bit_count()
    return diff / (n * 8)


def fingerprint_distance(fp1: str | None, fp2: str | None) -> float:
    """Return normalized Hamming distance between two fingerprint strings."""
    if not fp1 or not fp2:
        return 1.0
    if fp1 == fp2:
        return 0.0
    p1 = _parse_fp(fp1)
    p2 = _parse_fp(fp2)
    if not p1 or not p2 or p1[0] != p2[0]:
        return 1.0
    kind, a = p1
    _, b = p2
    if kind == "ints":
        return _hamming_ints(a, b)  # type: ignore[arg-type]
    else:
        return _hamming_bytes(a, b)  # type: ignore[arg-type]


def _has_exclusion(info: Dict[str, str | None]) -> bool:
    text = f"{info.get('title','')} {info.get('album','')}".lower()
    return any(k in text for k in EXCLUSION_KEYWORDS)


def _normalized(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _metadata_key(info: Mapping[str, str | None]) -> tuple[str, str, str]:
    return (
        _normalized(info.get("title")),
        _normalized(info.get("primary") or info.get("artist")),
        _normalized(info.get("album")),
    )


def _metadata_gate(info_a: Mapping[str, str | None], info_b: Mapping[str, str | None]) -> bool:
    title_a, primary_a, album_a = _metadata_key(info_a)
    title_b, primary_b, album_b = _metadata_key(info_b)
    if not title_a or not title_b or title_a != title_b:
        return False
    if primary_a and primary_b and primary_a == primary_b:
        return True
    if album_a and album_b and album_a == album_b:
        return True
    return False


def _path_ext(path: str, info: Mapping[str, str | None]) -> str:
    ext = info.get("ext") if isinstance(info, Mapping) else None
    if ext:
        return str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
    return os.path.splitext(path)[1].lower()


def _is_lossless(ext: str) -> bool:
    return ext in LOSSLESS_EXTS


def _pair_threshold(
    path_a: str,
    path_b: str,
    info_a: Mapping[str, str | None],
    info_b: Mapping[str, str | None],
    base_threshold: float,
    mixed_codec_boost: float,
) -> float:
    ext_a = _path_ext(path_a, info_a)
    ext_b = _path_ext(path_b, info_b)
    if not ext_a or not ext_b:
        return base_threshold
    if _is_lossless(ext_a) != _is_lossless(ext_b):
        return base_threshold + mixed_codec_boost
    return base_threshold


def _coarse_fingerprint_keys(fp: str | None) -> List[str]:
    parsed = _parse_fp(fp) if fp else None
    values: List[int] = []
    if parsed:
        kind, payload = parsed
        if kind == "ints":
            values = list(payload)
        elif kind == "bytes":
            values = list(payload)
    if not values and fp:
        values = [ord(ch) for ch in fp]
    if not values:
        return []
    quantized = [val // COARSE_FP_QUANTIZATION for val in values]
    keys: List[str] = []
    limit = COARSE_FP_BAND_SIZE * COARSE_FP_BANDS
    for idx in range(0, min(len(quantized), limit), COARSE_FP_BAND_SIZE):
        band = quantized[idx : idx + COARSE_FP_BAND_SIZE]
        if not band:
            break
        digest = hashlib.blake2s(",".join(map(str, band)).encode("utf-8"), digest_size=4).hexdigest()
        keys.append(f"{idx // COARSE_FP_BAND_SIZE}:{digest}")
    if not keys and fp:
        keys.append(f"band0:{hashlib.blake2s(fp.encode('utf-8'), digest_size=4).hexdigest()}")
    return keys


def _coarse_gate(keys_a: List[str], keys_b: List[str]) -> bool:
    if not keys_a or not keys_b:
        return False
    return bool(set(keys_a) & set(keys_b))


def _cluster_max_distance(cluster: Set[str], distances: Dict[frozenset[str], float]) -> float:
    pairs = [
        distances[frozenset({a, b})]
        for a, b in combinations(sorted(cluster), 2)
        if frozenset({a, b}) in distances
    ]
    return max(pairs) if pairs else 0.0


def _scan_paths(
    paths: Iterable[str],
    file_infos: Dict[str, Dict[str, str | None]],
    threshold: float,
    mixed_codec_boost: float,
    log_callback,
) -> Tuple[List[Tuple[Set[str], float]], int, int, int]:
    """Scan a list of paths for near duplicate clusters."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    comparisons = 0
    metadata_blocked = 0
    coarse_blocked = 0
    coarse_keys = {p: _coarse_fingerprint_keys(file_infos[p].get("fp")) for p in paths}
    pair_distances: Dict[frozenset[str], float] = {}
    path_list = list(paths)
    for i, p in enumerate(path_list):
        for q in path_list[i + 1 :]:
            if not _metadata_gate(file_infos[p], file_infos[q]):
                metadata_blocked += 1
                continue
            if not _coarse_gate(coarse_keys.get(p, []), coarse_keys.get(q, [])):
                coarse_blocked += 1
                continue
            dist = fingerprint_distance(file_infos[p]["fp"], file_infos[q]["fp"])
            pair_threshold = _pair_threshold(
                p,
                q,
                file_infos[p],
                file_infos[q],
                threshold,
                mixed_codec_boost,
            )
            if dist <= pair_threshold:
                adj[p].add(q)
                adj[q].add(p)
                pair_distances[frozenset({p, q})] = dist
                log_callback(
                    f"   → near-dup {os.path.basename(p)} vs {os.path.basename(q)} "
                    f"dist={dist:.3f} thr={pair_threshold:.3f}"
                )
            comparisons += 1

    visited: Set[str] = set()
    clusters: List[Set[str]] = []
    for p in path_list:
        if p in visited:
            continue
        stack = [p]
        comp = {p}
        visited.add(p)
        while stack:
            cur = stack.pop()
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    comp.add(nb)
                    stack.append(nb)
        if len(comp) > 1:
            max_distance = _cluster_max_distance(comp, pair_distances)
            clusters.append((comp, max_distance))
    return clusters, comparisons, metadata_blocked, coarse_blocked




def _scan_album(
    album: str,
    paths: Iterable[str],
    file_infos: Dict[str, Dict[str, str | None]],
    threshold: float,
    mixed_codec_boost: float,
    log_callback,
) -> tuple[str, List[Tuple[Set[str], float]], int, int, int]:
    """Worker helper: scan a single album for near duplicates."""
    clusters, comparisons, meta_blocked, coarse_blocked = _scan_paths(
        paths,
        file_infos,
        threshold,
        mixed_codec_boost,
        log_callback,
    )
    return album, clusters, comparisons, meta_blocked, coarse_blocked


def find_near_duplicates(
    file_infos: Dict[str, Dict[str, str | None]],
    ext_priority: Dict[str, int],
    threshold: float,
    log_callback=None,
    enable_cross_album: bool = False,
    coord=None,
    max_workers: int | None = None,
    mixed_codec_boost: float = 0.0,
) -> NearDuplicateResult:
    """Return reviewable near-duplicate groups (auto-merging is gated by review)."""
    if log_callback is None:
        def log_callback(msg: str) -> None:
            pass

    paths = [p for p, info in file_infos.items() if info.get("fp") and not _has_exclusion(info)]

    by_album: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        album = file_infos[p].get("album") or ""
        by_album[album].append(p)

    album_items = sorted(by_album.items(), key=lambda x: x[0])
    total_albums = len(album_items)
    total_comparisons = 0
    metadata_blocked = 0
    coarse_blocked = 0
    clusters: List[Tuple[Set[str], float]] = []
    html_lines = ["<h2>Phase B – Album Near-Duplicates</h2>", "<pre>"]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for idx, (album, album_paths) in enumerate(album_items, start=1):
            if len(album_paths) < 2:
                continue
            logger.debug(
                f"Phase B: [{idx}/{total_albums}] Scanning album '{album}' ({len(album_paths)} tracks)"
            )
            html_lines.append(f"Album: {album or '<no album>'} ({len(album_paths)} tracks)")
            futures[ex.submit(_scan_album, album, album_paths, file_infos, threshold, mixed_codec_boost, log_callback)] = album

        for fut in as_completed(futures):
            album = futures[fut]
            try:
                _alb, c, comps, meta_block, coarse_block = fut.result()
                clusters.extend(c)
                total_comparisons += comps
                metadata_blocked += meta_block
                coarse_blocked += coarse_block
                if coord is not None and c:
                    coord.add_near_dupe_clusters([list(s[0]) for s in c])
            except Exception as e:
                logger.error(f"Phase B album '{album}' failed: {e}")
                html_lines.append(f"Error scanning album '{album}'")

    html_lines.append("</pre>")
    html_str = "\n".join(html_lines)
    if coord is not None:
        coord.set_html_section("B", html_str)

    cross_clusters: List[Tuple[Set[str], float]] = []
    if enable_cross_album:
        cross_lines = ["<h2>Phase C – Cross-Album Near-Duplicates</h2>", "<pre>"]
        by_song: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for p in paths:
            key = (
                file_infos[p].get("primary") or "",
                file_infos[p].get("title") or "",
            )
            by_song[key].append(p)

        song_items = sorted(by_song.items(), key=lambda x: (x[0][0], x[0][1]))
        total_songs = len(song_items)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for idx, ((prim, title), song_paths) in enumerate(song_items, start=1):
                if len(song_paths) < 2:
                    continue
                logger.debug(
                    f"Phase C: [{idx}/{total_songs}] Scanning '{title}' ({len(song_paths)} tracks)"
                )
                cross_lines.append(f"{title or '<no title>'} ({len(song_paths)} tracks)")
                futures[ex.submit(_scan_paths, song_paths, file_infos, threshold, mixed_codec_boost, log_callback)] = (prim, title)

            for fut in as_completed(futures):
                _key = futures[fut]
                try:
                    c, comps, meta_block, coarse_block = fut.result()
                    cross_clusters.extend(c)
                    if coord is not None and c:
                        coord.add_near_dupe_clusters([list(s[0]) for s in c])
                    total_comparisons += comps
                    metadata_blocked += meta_block
                    coarse_blocked += coarse_block
                except Exception as e:
                    logger.error(f"Phase C group '{_key}' failed: {e}")
                    cross_lines.append(f"Error scanning group '{_key}'")

        cross_lines.append("</pre>")
        cross_html = "\n".join(cross_lines)
        if coord is not None:
            coord.set_html_section("C", cross_html)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Phase B summary: {total_albums} albums scanned, {total_comparisons} comparisons, {len(clusters)} clusters"
        )

    log_callback(
        "   • Near-duplicate gating summary: "
        f"{metadata_blocked} metadata comparisons blocked, {coarse_blocked} coarse fingerprint comparisons blocked "
        "(duration/tempo normalization deferred)."
    )

    all_clusters = clusters + cross_clusters

    result = NearDuplicateResult(auto_deletes={}, review_groups=[])
    for cluster, max_distance in all_clusters:
        by_alb: Dict[str, List[str]] = defaultdict(list)
        for path in cluster:
            album = file_infos[path].get("album") or ""
            by_alb[album].append(path)
        for album_paths in by_alb.values():
            if len(album_paths) <= 1:
                continue
            scored = sorted(
                album_paths,
                key=lambda p: _keep_score(p, file_infos[p], ext_priority),
                reverse=True,
            )
            keep = scored[0]
            losers = scored[1:]
            if not losers:
                continue
            reason = (
                f"Near-duplicate of {os.path.basename(keep)} "
                f"(max fingerprint distance {max_distance:.3f}); review required"
            )
            result.review_groups.append(
                NearDuplicateGroup(
                    winner=keep,
                    losers=losers,
                    max_distance=max_distance,
                    reason=reason,
                )
            )
    return result
