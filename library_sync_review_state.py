"""Review state storage and pure helpers for the Library Sync redesign."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from library_sync_types import MatchResult, MatchStatus


@dataclass
class ReviewFlags:
    """User selections captured during review."""

    copy: set[str] = field(default_factory=set)
    replace: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


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

    def set_note(self, incoming_track_id: str, note: str | None) -> None:
        """Attach or clear a freeform note for an incoming row."""
        if note is None or not str(note).strip():
            self.flags.notes.pop(incoming_track_id, None)
            return
        self.flags.notes[incoming_track_id] = str(note)

    def note_for(self, incoming_track_id: str) -> str | None:
        """Return a stored note for the incoming row, if present."""
        return self.flags.notes.get(incoming_track_id)

    def replace_target(self, incoming_track_id: str) -> str | None:
        """Return the flagged replacement target, if any."""
        return self.flags.replace.get(incoming_track_id)

    def clear_all(self) -> None:
        """Remove all flags across the review session."""
        self.flags.copy.clear()
        self.flags.replace.clear()
        self.flags.notes.clear()

    def clear_flags(self, incoming_track_id: str) -> None:
        """Remove the Copy and Replace flags of one row (its note stays)."""
        self.flags.copy.discard(incoming_track_id)
        self.flags.replace.pop(incoming_track_id, None)

    def has_flags(self) -> bool:
        return bool(self.flags.copy or self.flags.replace or self.flags.notes)

    def decision_signature(self) -> tuple:
        """Hashable snapshot of the flags that change a plan (notes do not)."""
        return (frozenset(self.flags.copy), frozenset(self.flags.replace.items()))

    def carry_over(self, results: Iterable[MatchResult]) -> List[str]:
        """Keep the flags that still make sense after a re-scan; drop the rest.

        * Copy flags and notes survive for incoming tracks that are still there.
        * A Replace flag survives only while the track's best match is the same
          existing track — it is *not* silently re-pointed at a different file.

        Returns one warning per flag or note that could not be carried over.
        """
        results = list(results)
        present = {res.incoming.track_id: res for res in results}
        names = {tid: res.incoming.path for tid, res in present.items()}
        warnings: List[str] = []
        for track_id in sorted(self.flags.copy):
            if track_id not in present:
                self.flags.copy.discard(track_id)
                warnings.append(f"Dropped a Copy flag: track {track_id[:8]}… is no longer in the incoming folder.")
        for track_id, existing_id in sorted(self.flags.replace.items()):
            res = present.get(track_id)
            if res is None:
                del self.flags.replace[track_id]
                warnings.append(f"Dropped a Replace flag: track {track_id[:8]}… is no longer in the incoming folder.")
            elif res.existing is None or res.existing.track_id != existing_id:
                del self.flags.replace[track_id]
                warnings.append(
                    f"Cleared the Replace flag on {names[track_id]}: its best match changed. "
                    "Flag it again if you still want it to replace a library track."
                )
        for track_id in sorted(self.flags.notes):
            if track_id not in present:
                self.flags.notes.pop(track_id, None)
                warnings.append(f"Dropped a note: track {track_id[:8]}… is no longer in the incoming folder.")
        return warnings

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
