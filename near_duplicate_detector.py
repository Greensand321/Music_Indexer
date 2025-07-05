# near_duplicate_detector.py
"""Helpers for fuzzy fingerprint grouping."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Set, List
import logging

logger = logging.getLogger(__name__)

from music_indexer_api import _keep_score

EXCLUSION_KEYWORDS = ['remix', 'remastered', 'edit', 'version']


def fingerprint_distance(fp1: str | None, fp2: str | None) -> float:
    """Return normalized Hamming distance between two fingerprint strings."""
    if not fp1 or not fp2:
        return 1.0
    try:
        arr1 = [int(x) for x in fp1.split()]
        arr2 = [int(x) for x in fp2.split()]
    except Exception:
        return 1.0
    n = min(len(arr1), len(arr2))
    if n == 0:
        return 1.0
    diff = sum(a != b for a, b in zip(arr1[:n], arr2[:n]))
    return diff / n


def _has_exclusion(info: Dict[str, str | None]) -> bool:
    text = f"{info.get('title','')} {info.get('album','')}".lower()
    return any(k in text for k in EXCLUSION_KEYWORDS)


def find_near_duplicates(
    file_infos: Dict[str, Dict[str, str | None]],
    ext_priority: Dict[str, int],
    threshold: float,
    log_callback=None,
    enable_cross_album: bool = False,
) -> Dict[str, str]:
    """Return mapping of files to delete as near duplicates."""
    if log_callback is None:
        def log_callback(msg: str) -> None:
            pass

    # Filter out tracks with exclusion keywords or missing fingerprints
    paths = [p for p, info in file_infos.items() if info.get('fp') and not _has_exclusion(info)]

    # Build similarity graph in album partitions
    adj: Dict[str, Set[str]] = defaultdict(set)
    by_album: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        album = file_infos[p].get('album') or ''
        by_album[album].append(p)

    album_items = list(by_album.items())
    for album, album_paths in album_items:
        if len(album_paths) < 2:
            continue
        log_callback(f"   • Phase B album '{album or '<no album>'}' ({len(album_paths)} tracks)")
        for i, p in enumerate(album_paths):
            for q in album_paths[i + 1:]:
                dist = fingerprint_distance(file_infos[p]['fp'], file_infos[q]['fp'])
                if dist <= threshold:
                    adj[p].add(q)
                    adj[q].add(p)
                    log_callback(
                        f"   → near-dup {os.path.basename(p)} vs {os.path.basename(q)} dist={dist:.3f}")

    if enable_cross_album:
        all_paths = list(paths)
        log_callback("   • Phase C cross-album scan…")
        for i, p in enumerate(all_paths):
            for q in all_paths[i + 1:]:
                if file_infos[p].get('album') == file_infos[q].get('album'):
                    continue
                dist = fingerprint_distance(file_infos[p]['fp'], file_infos[q]['fp'])
                if dist <= threshold:
                    adj[p].add(q)
                    adj[q].add(p)
                    log_callback(
                        f"   → near-dup {os.path.basename(p)} vs {os.path.basename(q)} dist={dist:.3f}")

    # Compute connected components
    visited: Set[str] = set()
    clusters: List[Set[str]] = []
    for p in paths:
        if p in visited:
            continue
        stack = [p]
        comp = set([p])
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

    to_delete: Dict[str, str] = {}
    for cluster in clusters:
        # group by album title
        by_album: Dict[str, List[str]] = defaultdict(list)
        for path in cluster:
            album = file_infos[path].get('album') or ''
            by_album[album].append(path)
        for album_paths in by_album.values():
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
