"""Shared type definitions for library_sync and library_sync_review_state.

This module contains only types (enums, dataclasses) with no business logic.
It exists to break the circular import between library_sync and library_sync_review_state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

# Signature token count constant
SIGNATURE_TOKEN_COUNT = 8


def _fingerprint_signature(fp: str | None) -> str | None:
    """Extract signature from fingerprint string."""
    if not fp:
        return None
    tokens = fp.replace(",", " ").split()
    if not tokens:
        return None
    return " ".join(tokens[:SIGNATURE_TOKEN_COUNT])


class MatchStatus(str, Enum):
    """Status of a match between incoming and existing tracks."""

    NEW = "new"
    COLLISION = "collision"
    EXACT_MATCH = "exact_match"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class TrackRecord:
    """Normalized representation of a scanned track."""

    track_id: str
    path: str
    normalized_path: str
    ext: str
    bitrate: int | None
    size: int
    mtime: float
    fingerprint: str | None
    tags: Dict[str, object]
    duration: int | None = None

    def signature(self) -> str | None:
        return _fingerprint_signature(self.fingerprint)

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "path": self.path,
            "normalized_path": self.normalized_path,
            "ext": self.ext,
            "bitrate": self.bitrate,
            "size": self.size,
            "mtime": self.mtime,
            "fingerprint": self.fingerprint,
            "duration": self.duration,
            "tags": dict(self.tags),
        }


@dataclass(order=True)
class CandidateDistance:
    """Distance info for a shortlist candidate."""

    distance: float
    track: TrackRecord = field(compare=False)

    def to_dict(self) -> Dict[str, object]:
        return {"track_id": self.track.track_id, "path": self.track.path, "distance": self.distance}


@dataclass
class MatchResult:
    """Matching outcome for a single incoming track."""

    incoming: TrackRecord
    existing: TrackRecord | None
    status: MatchStatus
    distance: float | None
    threshold_used: float
    near_miss_margin: float
    confidence: float
    candidates: List[CandidateDistance]
    quality_label: str | None
    incoming_score: int
    existing_score: int | None

    def to_dict(self) -> Dict[str, object]:
        return {
            "incoming": self.incoming.to_dict(),
            "existing": self.existing.to_dict() if self.existing else None,
            "status": self.status.value,
            "distance": self.distance,
            "threshold_used": self.threshold_used,
            "near_miss_margin": self.near_miss_margin,
            "confidence": self.confidence,
            "candidates": [c.to_dict() for c in self.candidates],
            "quality_label": self.quality_label,
            "incoming_score": self.incoming_score,
            "existing_score": self.existing_score,
        }
