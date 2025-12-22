"""Duplicate consolidation planning (no file mutations).

This module performs a dry-run duplicate scan that groups musically identical
tracks using audio fingerprints, selects a deterministic "winner" based on
quality, and produces a consolidation plan describing artwork/metadata actions.
No files are moved or written—callers can safely surface the plan for review
before executing any changes.
"""
from __future__ import annotations

import datetime
import hashlib
import html
import json
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
        ext = self.ext.lower()
        if not ext or ext == ".":
            ext = os.path.splitext(self.path)[1].lower()
        elif not ext.startswith("."):
            ext = f".{ext}"
        return ext in LOSSLESS_EXTS

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
class PlaylistImpact:
    """Summary of playlist rewrites driven by a duplicate group."""

    playlists: int = 0
    entries: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {"playlists": self.playlists, "entries": self.entries}


@dataclass
class GroupPlan:
    """Planned actions for a duplicate group."""

    group_id: str
    winner_path: str
    losers: List[str]
    planned_winner_tags: Dict[str, object]
    metadata_changes: Dict[str, Dict[str, object]]
    winner_quality: Dict[str, object]
    artwork: List[ArtworkDirective]
    loser_disposition: Dict[str, str]
    playlist_rewrites: Dict[str, str]
    playlist_impact: PlaylistImpact
    review_flags: List[str]
    context_summary: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "group_id": self.group_id,
            "winner_path": self.winner_path,
            "losers": list(self.losers),
            "planned_winner_tags": dict(self.planned_winner_tags),
            "metadata_changes": {k: dict(v) for k, v in self.metadata_changes.items()},
            "winner_quality": dict(self.winner_quality),
            "artwork": [a.to_dict() for a in self.artwork],
            "loser_disposition": dict(self.loser_disposition),
            "playlist_rewrites": dict(self.playlist_rewrites),
            "playlist_impact": self.playlist_impact.to_dict(),
            "review_flags": list(self.review_flags),
            "context_summary": {k: list(v) for k, v in self.context_summary.items()},
        }


@dataclass
class ConsolidationPlan:
    """Aggregate plan across all duplicate groups."""

    groups: List[GroupPlan] = field(default_factory=list)
    review_flags: List[str] = field(default_factory=list)
    generated_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

    def to_dict(self) -> Dict[str, object]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "review_flags": list(self.review_flags),
            "generated_at": self.generated_at.isoformat(),
        }

    @property
    def review_required_groups(self) -> List[GroupPlan]:
        """Return groups that need manual review."""
        return [g for g in self.groups if g.review_flags]

    @property
    def review_required_count(self) -> int:
        """Return total number of review-required groups."""
        return len(self.review_required_groups)


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


def _metadata_changes(winner: DuplicateTrack, planned_tags: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    keys = [
        "artist",
        "albumartist",
        "title",
        "album",
        "album_type",
        "track",
        "tracknumber",
        "disc",
        "discnumber",
        "date",
        "year",
        "genre",
    ]
    changes: Dict[str, Dict[str, object]] = {}
    for key in keys:
        current = winner.tags.get(key) if isinstance(winner.tags, Mapping) else None
        planned = planned_tags.get(key)
        if (current or "") == (planned or ""):
            continue
        changes[key] = {"from": current, "to": planned}
    return changes


def _quality_rationale(winner: DuplicateTrack, runner_up: DuplicateTrack | None, context: str) -> Dict[str, object]:
    reasons: List[str] = []
    if winner.is_lossless:
        reasons.append("Lossless format")
    if runner_up and not runner_up.is_lossless and winner.is_lossless:
        reasons.append("Higher-quality format than the next candidate")
    if runner_up and winner.bitrate > runner_up.bitrate:
        reasons.append("Higher bitrate than the next candidate")
    if runner_up and winner.sample_rate > runner_up.sample_rate:
        reasons.append("Higher sample rate than the next candidate")
    if runner_up and winner.bit_depth > runner_up.bit_depth:
        reasons.append("Higher bit depth than the next candidate")
    if context == "album":
        reasons.append("Album context favored for metadata consistency")
    elif context == "single":
        reasons.append("Single context favored for artwork alignment")

    return {
        "context": context,
        "is_lossless": winner.is_lossless,
        "bitrate": winner.bitrate,
        "sample_rate": winner.sample_rate,
        "bit_depth": winner.bit_depth,
        "metadata_count": winner.metadata_count,
        "reasons": reasons or ["Deterministic ordering by quality"],
    }


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
        runner_up = quality_sorted[1] if len(quality_sorted) > 1 else None

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

        meta_changes = _metadata_changes(winner, planned_tags)
        winner_context = contexts.get(winner.path, "unknown")
        winner_quality = _quality_rationale(winner, runner_up, winner_context)

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

        playlist_impact = PlaylistImpact(playlists=len(losers), entries=len(losers))

        plans.append(
            GroupPlan(
                group_id=group_id,
                winner_path=winner.path,
                losers=losers,
                planned_winner_tags=planned_tags,
                metadata_changes=meta_changes,
                winner_quality=winner_quality,
                artwork=artwork_actions,
                loser_disposition=dispositions,
                playlist_rewrites=playlist_map,
                playlist_impact=playlist_impact,
                review_flags=group_review,
                context_summary={
                    "album": sorted([t.path for t in cluster if contexts.get(t.path) == "album"]),
                    "single": sorted([t.path for t in cluster if contexts.get(t.path) == "single"]),
                    "unknown": sorted([t.path for t in cluster if contexts.get(t.path) == "unknown"]),
                },
            )
        )

    return ConsolidationPlan(groups=plans, review_flags=review_flags)


def _html_escape(text: object) -> str:
    return html.escape("" if text is None else str(text))


def render_consolidation_preview(plan: ConsolidationPlan, output_html_path: str) -> str:
    """Render an HTML preview summarizing the consolidation plan."""

    def _render_metadata(changes: Dict[str, Dict[str, object]]) -> str:
        if not changes:
            return "<em>No tag changes planned.</em>"
        rows = []
        for key in sorted(changes.keys()):
            diff = changes[key]
            rows.append(
                f"<tr><td>{_html_escape(key)}</td><td>{_html_escape(diff.get('from') or '—')}</td>"
                f"<td>{_html_escape(diff.get('to') or '—')}</td></tr>"
            )
        return (
            "<table class='metadata'>"
            "<tr><th>Tag</th><th>Current</th><th>Planned</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    def _render_artwork(actions: List[ArtworkDirective]) -> str:
        if not actions:
            return "<em>Winner artwork unchanged.</em>"
        rows = [
            f"<li>Copy artwork from {_html_escape(act.source)} → {_html_escape(act.target)} "
            f"({_html_escape(act.reason)})</li>"
            for act in actions
        ]
        return "<ul>" + "".join(rows) + "</ul>"

    review_required = plan.review_required_groups
    lines = [
        "<html><head><style>",
        "body{font-family:Arial, sans-serif; margin:20px;}",
        ".group{border:1px solid #ddd; padding:12px; margin-bottom:10px; border-radius:6px;}",
        ".review{background:#fff3cd;}",
        ".header{display:flex;justify-content:space-between;align-items:center;}",
        ".muted{color:#666; font-size:0.9em;}",
        ".metadata{border-collapse:collapse;width:100%;}",
        ".metadata th,.metadata td{border:1px solid #ddd;padding:4px 6px;text-align:left;}",
        ".pill{display:inline-block;padding:4px 8px;border-radius:12px;font-size:12px;}",
        ".ok{background:#e7f5e8;color:#266534;}",
        ".warn{background:#ffe8cc;color:#b05e00;}",
        "</style></head><body>",
        "<h1>Duplicate Consolidation Preview</h1>",
        f"<p class='muted'>Generated at {plan.generated_at.isoformat()}</p>",
        f"<p>Total groups: {len(plan.groups)} | Review required: {len(review_required)} | "
        f"Global flags: {len(plan.review_flags)}</p>",
    ]

    if plan.review_flags:
        lines.append("<div class='group review'><strong>Plan Warnings</strong><ul>")
        for flag in plan.review_flags:
            lines.append(f"<li>{_html_escape(flag)}</li>")
        lines.append("</ul></div>")

    for group in plan.groups:
        review_badge = (
            "<span class='pill warn'>Review required</span>" if group.review_flags else "<span class='pill ok'>Ready</span>"
        )
        lines.append("<div class='group {cls}'>".format(cls="review" if group.review_flags else ""))
        lines.append(
            f"<div class='header'><div><strong>Group {group.group_id}</strong></div>{review_badge}</div>"
        )
        lines.append(f"<p class='muted'>Winner: {_html_escape(group.winner_path)}</p>")
        lines.append("<ul>")
        for loser in group.losers:
            disposition = group.loser_disposition.get(loser, "quarantine")
            lines.append(
                f"<li>Loser: {_html_escape(loser)} → {_html_escape(disposition)} | "
                f"Playlist rewrite → {_html_escape(group.playlist_rewrites.get(loser, 'n/a'))}</li>"
            )
        lines.append("</ul>")

        lines.append("<p><strong>Quality rationale</strong></p>")
        reasons = group.winner_quality.get("reasons") or []
        lines.append(
            "<ul>"
            + "".join(f"<li>{_html_escape(r)}</li>" for r in reasons)
            + "</ul>"
        )
        lines.append("<p><strong>Metadata normalization</strong></p>")
        lines.append(_render_metadata(group.metadata_changes))

        lines.append("<p><strong>Artwork</strong></p>")
        lines.append(_render_artwork(group.artwork))

        lines.append(
            f"<p><strong>Playlist impact</strong>: {group.playlist_impact.playlists} playlists, "
            f"{group.playlist_impact.entries} entries</p>"
        )

        if group.review_flags:
            lines.append("<p><strong>Review flags:</strong></p><ul>")
            for flag in group.review_flags:
                lines.append(f"<li>{_html_escape(flag)}</li>")
            lines.append("</ul>")

        lines.append(
            "<details><summary>Context</summary><ul>"
            + f"<li>Album tracks: {len(group.context_summary.get('album', []))}</li>"
            + f"<li>Singles: {len(group.context_summary.get('single', []))}</li>"
            + f"<li>Unknown: {len(group.context_summary.get('unknown', []))}</li>"
            + "</ul></details>"
        )
        lines.append("</div>")

    lines.append("</body></html>")

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_html_path


def export_consolidation_preview(plan: ConsolidationPlan, output_json_path: str) -> str:
    """Write a JSON audit of the consolidation plan."""

    summary = {
        "groups": len(plan.groups),
        "review_required": plan.review_required_count,
        "review_flags": plan.review_flags,
    }
    payload = {
        "generated_at": plan.generated_at.isoformat(),
        "summary": summary,
        "plan": plan.to_dict(),
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_json_path
