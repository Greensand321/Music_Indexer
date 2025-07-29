# near_duplicate_detector.py
"""Helpers for fuzzy fingerprint grouping."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Set, List, Iterable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import base64

logger = logging.getLogger(__name__)

from music_indexer_api import _keep_score

EXCLUSION_KEYWORDS = ['remix', 'remastered', 'edit', 'version']


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


def _scan_paths(
    paths: Iterable[str],
    file_infos: Dict[str, Dict[str, str | None]],
    threshold: float,
    log_callback,
) -> Tuple[List[Set[str]], int]:
    """Scan a list of paths for near duplicate clusters."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    comparisons = 0
    path_list = list(paths)
    for i, p in enumerate(path_list):
        for q in path_list[i + 1 :]:
            dist = fingerprint_distance(file_infos[p]["fp"], file_infos[q]["fp"])
            if dist <= threshold:
                adj[p].add(q)
                adj[q].add(p)
                log_callback(
                    f"   → near-dup {os.path.basename(p)} vs {os.path.basename(q)} dist={dist:.3f}"
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
            clusters.append(comp)
    return clusters, comparisons




def _scan_album(
    album: str,
    paths: Iterable[str],
    file_infos: Dict[str, Dict[str, str | None]],
    threshold: float,
    log_callback,
) -> tuple[str, List[Set[str]], int]:
    """Worker helper: scan a single album for near duplicates."""
    clusters, comparisons = _scan_paths(paths, file_infos, threshold, log_callback)
    return album, clusters, comparisons


def find_near_duplicates(
    file_infos: Dict[str, Dict[str, str | None]],
    ext_priority: Dict[str, int],
    threshold: float,
    log_callback=None,
    enable_cross_album: bool = False,
    coord=None,
    max_workers: int | None = None,
) -> Dict[str, str]:
    """Return mapping of files to delete as near duplicates."""
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
    clusters: List[Set[str]] = []
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
            futures[ex.submit(_scan_album, album, album_paths, file_infos, threshold, log_callback)] = album

        for fut in as_completed(futures):
            album = futures[fut]
            try:
                _alb, c, comps = fut.result()
                clusters.extend(c)
                total_comparisons += comps
                if coord is not None and c:
                    coord.add_near_dupe_clusters([list(s) for s in c])
            except Exception as e:
                logger.error(f"Phase B album '{album}' failed: {e}")
                html_lines.append(f"Error scanning album '{album}'")

    html_lines.append("</pre>")
    html_str = "\n".join(html_lines)
    if coord is not None:
        coord.set_html_section("B", html_str)

    cross_clusters: List[Set[str]] = []
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
                futures[ex.submit(_scan_paths, song_paths, file_infos, threshold, log_callback)] = (prim, title)

            for fut in as_completed(futures):
                _key = futures[fut]
                try:
                    c, comps = fut.result()
                    cross_clusters.extend(c)
                    if coord is not None and c:
                        coord.add_near_dupe_clusters([list(s) for s in c])
                    total_comparisons += comps
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

    all_clusters = clusters + cross_clusters

    to_delete: Dict[str, str] = {}
    for cluster in all_clusters:
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
            for loser in scored[1:]:
                if loser not in to_delete:
                    to_delete[loser] = f"Near-duplicate of {os.path.basename(keep)}"
    return to_delete
