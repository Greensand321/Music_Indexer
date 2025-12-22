"""Duplicate consolidation planning (no file mutations).

This module performs a dry-run duplicate scan that groups musically identical
tracks using audio fingerprints, selects a deterministic "winner" based on
quality, and produces a consolidation plan describing artwork/metadata actions.
No files are moved or written—callers can safely surface the plan for review
before executing any changes.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from near_duplicate_detector import fingerprint_distance

LOSSLESS_EXTS = {".flac", ".wav", ".alac", ".ape", ".aiff", ".aif"}
DEFAULT_DISTANCE_THRESHOLD = 0.0
DEFAULT_MAX_CANDIDATES = 5000
DEFAULT_MAX_COMPARISONS = 50_000
DEFAULT_TIMEOUT_SEC = 15.0


def _now() -> float:
    return time.perf_counter()


def _default_progress(_current: int, _total: int, _msg: str) -> None:
    return None


@dataclass
class DuplicateTrack:
    """Normalized representation of a track for consolidation planning."""

    path: str
    fingerprint: str | None
    ext: str
    bitrate: int = 0
    sample_rate: int = 0
    bit_depth: int = 0
    tags: Dict[str, object] = field(default_factory=dict)

    @property
    def cover_hash(self) -> str | None:
        value = self.tags.get("cover_hash") if isinstance(self.tags, Mapping) else None
        if value:
            return str(value)
        art = self.tags.get("artwork_hash") if isinstance(self.tags, Mapping) else None
        return str(art) if art else None

    @property
    def is_lossless(self) -> bool:
        return os.path.splitext(self.ext)[1].lower() in LOSSLESS_EXTS

    @property
    def metadata_count(self) -> int:
        if not isinstance(self.tags, Mapping):
            return 0
        keys = [
            "artist",
            "albumartist",
            "title",
            "album",
            "track",
            "tracknumber",
            "disc",
            "discnumber",
            "date",
            "year",
        ]
        return sum(1 for k in keys if self.tags.get(k))


@dataclass
class ArtworkDirective:
    """Instruction to copy artwork from a source to a target."""

    source: str
    target: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"source": self.source, "target": self.target, "reason": self.reason}


@dataclass
class GroupPlan:
    """Planned actions for a duplicate group."""

    group_id: str
    winner_path: str
    losers: List[str]
    planned_winner_tags: Dict[str, object]
    artwork: List[ArtworkDirective]
    loser_disposition: Dict[str, str]
    playlist_rewrites: Dict[str, str]
    review_flags: List[str]
    context_summary: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "group_id": self.group_id,
            "winner_path": self.winner_path,
            "losers": list(self.losers),
            "planned_winner_tags": dict(self.planned_winner_tags),
            "artwork": [a.to_dict() for a in self.artwork],
            "loser_disposition": dict(self.loser_disposition),
            "playlist_rewrites": dict(self.playlist_rewrites),
            "review_flags": list(self.review_flags),
            "context_summary": {k: list(v) for k, v in self.context_summary.items()},
        }


@dataclass
class ConsolidationPlan:
    """Aggregate plan across all duplicate groups."""

    groups: List[GroupPlan] = field(default_factory=list)
    review_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "review_flags": list(self.review_flags),
        }


def _normalize_track(raw: Mapping[str, object]) -> DuplicateTrack:
    path = str(raw.get("path"))
    ext = os.path.splitext(path)[1].lower() or str(raw.get("ext", "")).lower()
    tags = raw.get("tags") if isinstance(raw.get("tags"), Mapping) else {}
    return DuplicateTrack(
        path=path,
        fingerprint=raw.get("fingerprint"),
        ext=ext,
        bitrate=int(raw.get("bitrate") or 0),
        sample_rate=int(raw.get("sample_rate") or raw.get("samplerate") or 0),
        bit_depth=int(raw.get("bit_depth") or raw.get("bitdepth") or 0),
        tags=dict(tags),
    )


def _classify_context(track: DuplicateTrack) -> str:
    tags = track.tags or {}
    album = str(tags.get("album") or "").strip()
    album_type = str(tags.get("album_type") or tags.get("release_type") or "").lower()
    track_no = tags.get("track") or tags.get("tracknumber")
    disc_no = tags.get("disc") or tags.get("discnumber")

    if album_type == "single":
        return "single"
    if album_type in {"album", "lp"}:
        return "album"
    if album.lower().endswith(" - single") or "(single" in album.lower():
        return "single"
    if album and (track_no or disc_no):
        return "album"
    if not album:
        return "single"
    if track.cover_hash:
        return "album"
    return "unknown"


def _quality_tuple(track: DuplicateTrack, context: str) -> tuple:
    return (
        1 if track.is_lossless else 0,
        track.bitrate,
        track.sample_rate,
        track.bit_depth,
        1 if context == "single" else 0,
        track.path.lower(),
    )


def _stable_group_id(paths: Sequence[str]) -> str:
    canonical = "|".join(sorted(paths))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _select_metadata_source(candidates: Sequence[DuplicateTrack], contexts: Mapping[str, str]) -> DuplicateTrack | None:
    album_candidates = [c for c in candidates if contexts.get(c.path) == "album"]
    sorted_cands = sorted(
        album_candidates or list(candidates),
        key=lambda c: (c.metadata_count, c.bitrate, c.sample_rate, c.bit_depth, c.path.lower()),
        reverse=True,
    )
    return sorted_cands[0] if sorted_cands else None


def _merge_tags(existing: MutableMapping[str, object], source: Mapping[str, object]) -> Dict[str, object]:
    merged = dict(existing)
    for key, value in source.items():
        if key in merged and merged[key]:
            continue
        merged[key] = value
    return merged


def _artwork_score(track: DuplicateTrack) -> tuple:
    tags = track.tags or {}
    size_hint = 0
    art_bytes = tags.get("artwork_bytes") if isinstance(tags, Mapping) else None
    if isinstance(art_bytes, (bytes, bytearray)):
        size_hint = len(art_bytes)
    elif isinstance(art_bytes, int):
        size_hint = art_bytes
    return (
        1 if track.cover_hash else 0,
        size_hint,
        track.bitrate,
        track.sample_rate,
        track.path.lower(),
    )


def _select_artwork_candidate(candidates: Sequence[DuplicateTrack]) -> tuple[DuplicateTrack | None, bool]:
    ranked = sorted(candidates, key=_artwork_score, reverse=True)
    if not ranked:
        return None, False
    if len(ranked) == 1:
        return ranked[0], False
    top_score = _artwork_score(ranked[0])
    ambiguous = sum(1 for c in ranked if _artwork_score(c) == top_score) > 1
    return ranked[0], ambiguous


def _cluster_duplicates(
    tracks: Sequence[DuplicateTrack],
    *,
    threshold: float,
    cancel_event: threading.Event,
    max_comparisons: int,
    timeout_sec: float,
    progress_callback: Callable[[int, int, str], None],
    start_time: float,
    review_flags: List[str],
) -> List[List[DuplicateTrack]]:
    buckets: Dict[str, List[DuplicateTrack]] = {}
    for track in tracks:
        if not track.fingerprint:
            continue
        key = str(track.fingerprint) if threshold <= 0 else str(track.fingerprint)[:20]
        buckets.setdefault(key, []).append(track)

    clusters: List[List[DuplicateTrack]] = []
    comparisons = 0
    processed = 0
    bucket_items = sorted(buckets.items(), key=lambda x: x[0])
    for _, bucket_tracks in bucket_items:
        if cancel_event.is_set() or (timeout_sec and (_now() - start_time) > timeout_sec):
            review_flags.append("Consolidation planning cancelled or timed out during grouping.")
            break
        bucket_tracks = sorted(bucket_tracks, key=lambda t: t.path.lower())
        used: set[str] = set()
        for idx, track in enumerate(bucket_tracks):
            if (
                cancel_event.is_set()
                or comparisons >= max_comparisons
                or (timeout_sec and (_now() - start_time) > timeout_sec)
            ):
                break
            if track.path in used:
                continue
            group = [track]
            used.add(track.path)
            for other in bucket_tracks[idx + 1 :]:
                if cancel_event.is_set() or comparisons >= max_comparisons:
                    review_flags.append("Comparison budget reached; grouping may be incomplete.")
                    break
                if timeout_sec and (_now() - start_time) > timeout_sec:
                    review_flags.append("Consolidation planning timed out while grouping.")
                    break
                comparisons += 1
                dist = fingerprint_distance(track.fingerprint, other.fingerprint)
                if dist <= threshold:
                    group.append(other)
                    used.add(other.path)
            if len(group) > 1:
                clusters.append(group)
            processed += 1
            progress_callback(processed, len(bucket_tracks), track.path)
        if cancel_event.is_set() or comparisons >= max_comparisons or (timeout_sec and (_now() - start_time) > timeout_sec):
            break
    return clusters


def build_consolidation_plan(
    tracks: Iterable[Mapping[str, object]],
    *,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_comparisons: int = DEFAULT_MAX_COMPARISONS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ConsolidationPlan:
    """Generate a deterministic consolidation plan without modifying files."""

    cancel_event = cancel_event or threading.Event()
    progress_callback = progress_callback or _default_progress
    review_flags: List[str] = []
    start_time = _now()

    normalized: List[DuplicateTrack] = []
    for raw in tracks:
        if cancel_event.is_set():
            review_flags.append("Cancelled before normalization.")
            break
        normalized.append(_normalize_track(raw))
        if len(normalized) >= max_candidates:
            review_flags.append(f"Truncated candidate set to {max_candidates} items to protect runtime.")
            break

    clusters = _cluster_duplicates(
        normalized,
        threshold=distance_threshold,
        cancel_event=cancel_event,
        max_comparisons=max_comparisons,
        timeout_sec=timeout_sec,
        progress_callback=progress_callback,
        start_time=start_time,
        review_flags=review_flags,
    )

    plans: List[GroupPlan] = []
    for cluster in sorted(clusters, key=lambda c: _stable_group_id([t.path for t in c])):
        contexts = {t.path: _classify_context(t) for t in cluster}
        quality_sorted = sorted(cluster, key=lambda t: _quality_tuple(t, contexts[t.path]), reverse=True)
        winner = quality_sorted[0]
        losers = [t.path for t in quality_sorted[1:]]

        metadata_source = _select_metadata_source(cluster, contexts)
        planned_tags: Dict[str, object] = dict(winner.tags)
        if metadata_source:
            planned_tags = _merge_tags(planned_tags, metadata_source.tags)
            if contexts.get(metadata_source.path) == "album":
                planned_tags.setdefault("album_type", "album")
            elif contexts.get(metadata_source.path) == "single":
                planned_tags.setdefault("album_type", "single")
        else:
            review_flags.append(f"Missing metadata source for group containing {winner.path}.")

        single_candidates = [t for t in cluster if contexts.get(t.path) == "single"]
        best_art, ambiguous_art = _select_artwork_candidate(single_candidates)
        artwork_actions: List[ArtworkDirective] = []
        if best_art and (not winner.cover_hash or winner.cover_hash != best_art.cover_hash):
            reason = "Copy single artwork to preserve release look"
            artwork_actions.append(ArtworkDirective(source=best_art.path, target=winner.path, reason=reason))
            if ambiguous_art:
                review_flags.append(
                    f"Artwork selection ambiguous for group {_stable_group_id([t.path for t in cluster])}."
                )

        dispositions = {loser: "quarantine" for loser in losers}
        playlist_map = {loser: winner.path for loser in losers}

        group_id = _stable_group_id([t.path for t in cluster])
        group_review: List[str] = []
        if ambiguous_art:
            group_review.append("Artwork selection requires review.")
        if not losers:
            group_review.append("No losers to consolidate.")

        plans.append(
            GroupPlan(
                group_id=group_id,
                winner_path=winner.path,
                losers=losers,
                planned_winner_tags=planned_tags,
                artwork=artwork_actions,
                loser_disposition=dispositions,
                playlist_rewrites=playlist_map,
                review_flags=group_review,
                context_summary={
                    "album": sorted([t.path for t in cluster if contexts.get(t.path) == "album"]),
                    "single": sorted([t.path for t in cluster if contexts.get(t.path) == "single"]),
                    "unknown": sorted([t.path for t in cluster if contexts.get(t.path) == "unknown"]),
                },
            )
        )

    return ConsolidationPlan(groups=plans, review_flags=review_flags)
