"""Review state storage and pure helpers for the Library Sync redesign."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from library_sync import MatchResult, MatchStatus


@dataclass
class ReviewFlags:
    """User selections captured during review."""

    copy: set[str] = field(default_factory=set)
    replace: Dict[str, str] = field(default_factory=dict)


class ReviewStateStore:
    """In-memory store for review flags that survives view transforms."""

    def __init__(self) -> None:
        self.flags = ReviewFlags()

    def flag_for_copy(self, incoming_track_id: str) -> None:
        """Mark an incoming row for copy."""
        self.flags.copy.add(incoming_track_id)

    def unflag_copy(self, incoming_track_id: str) -> None:
        """Remove a copy flag from an incoming row."""
        self.flags.copy.discard(incoming_track_id)

    def is_copy_flagged(self, incoming_track_id: str) -> bool:
        """Return whether the incoming row is flagged for copy."""
        return incoming_track_id in self.flags.copy

    def flag_for_replace(self, match: MatchResult) -> None:
        """Flag the best-match pair for replacement."""
        if match.existing is None:
            raise ValueError("Cannot flag for replace without a best match.")
        self.flags.replace[match.incoming.track_id] = match.existing.track_id

    def replace_target(self, incoming_track_id: str) -> str | None:
        """Return the flagged replacement target, if any."""
        return self.flags.replace.get(incoming_track_id)

    def clear_all(self) -> None:
        """Remove all flags across the review session."""
        self.flags.copy.clear()
        self.flags.replace.clear()

    def reconcile_best_matches(self, results: Iterable[MatchResult]) -> List[str]:
        """Rebind replacement flags when the best match changes.

        Returns a list of warning messages describing any rebinding or clears.
        """
        best_map = {
            res.incoming.track_id: res.existing.track_id if res.existing else None for res in results
        }
        warnings: List[str] = []
        for incoming_id, existing_id in list(self.flags.replace.items()):
            best_existing = best_map.get(incoming_id)
            if best_existing is None:
                warnings.append(
                    f"Cleared replace flag for {incoming_id} because it no longer has a best match."
                )
                del self.flags.replace[incoming_id]
                continue
            if best_existing != existing_id:
                warnings.append(
                    f"Rebound replace flag for {incoming_id} from {existing_id} to {best_existing} "
                    "after best match changed."
                )
                self.flags.replace[incoming_id] = best_existing
        return warnings


def filter_collisions_only(results: Iterable[MatchResult]) -> List[MatchResult]:
    """Return only matched rows (collisions, exact matches, and near-misses)."""
    return [res for res in results if res.existing is not None and res.status != MatchStatus.NEW]


def quality_delta(result: MatchResult) -> int | None:
    """Compute quality difference between incoming and existing tracks."""
    if result.existing_score is None:
        return None
    return result.incoming_score - (result.existing_score or 0)


def sort_by_quality_delta(results: Iterable[MatchResult], descending: bool = True) -> List[MatchResult]:
    """Sort incoming tracks by quality delta while keeping unmatched rows last."""
    indexed = list(enumerate(results))

    def _key(item: tuple[int, MatchResult]) -> tuple[bool, float, int]:
        idx, res = item
        delta = quality_delta(res)
        if delta is None:
            return True, float("inf"), idx
        if descending:
            return False, -float(delta), idx
        return False, float(delta), idx

    return [res for _idx, res in sorted(indexed, key=_key)]
