"""The match ladder — decide, for each wanted track, whether the library has it.

Not one similarity score but a cascade. Each rung is cheaper and more certain
than the one below it, and a row leaves as soon as a rung resolves it. Rungs
whose inputs the source cannot supply are skipped, which is what lets a bare CSV
of video titles and a fully-populated feed share one engine.

Two rules govern everything here:

1. **Uncertainty never resolves toward OWNED.** A wrong "you already have this"
   silently drops a song the user wanted and is never mentioned again; a wrong
   "you're missing this" only costs a duplicate the Duplicate Finder can clean
   up. ``Verdict`` precedence enforces this, and a test asserts no fuzzy score
   can override it.
2. **Every split candidate is probed.** The parser does not commit to one
   reading, so the matcher tries them all. Exactly one hitting is the match *and*
   the proof the parse was right; two hitting different tracks is an honest
   "unsure".

Phase 1 implements rungs 0, 0b, 2, 5b and 6. Rung 3's modifier adjudication and
rungs 4/5's fuzzy-by-artist land in Phase 2; until then a modifier difference
resolves to UNSURE rather than guessing.

Qt-free. See ``docs/playlist_gap_spec.md`` §7.
"""
from __future__ import annotations

import difflib
import os
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from playlist_gap_lexicon import Lexicon, modifier_delta
from playlist_gap_parse import candidate_splits, compare_key, looks_like_mashup
from playlist_gap_snapshot import LibrarySnapshot, tokens_of
from playlist_gap_types import (
    DEFAULT_THRESHOLDS,
    Candidate,
    GapResult,
    LibraryTrack,
    ReasonCode,
    Rung,
    SourceCapabilities,
    SplitCandidate,
    Thresholds,
    Verdict,
    WantedTrack,
    stronger,
)


class Scorer:
    """String similarity. Swappable so rapidfuzz can replace difflib later."""

    def ratio(self, a: str, b: str) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class DifflibScorer(Scorer):
    """stdlib similarity — already this project's tool of choice."""

    def ratio(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0
        return difflib.SequenceMatcher(None, a, b).ratio()


DEFAULT_SCORER = DifflibScorer()


def duration_delta(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None:
        return None
    return abs(int(a) - int(b))


def _too_far_apart(delta: Optional[int], thresholds: Thresholds) -> bool:
    """True when two runtimes are so different they cannot be one recording.

    Brought forward from Phase 2 deliberately: it is only ever used to *demote* a
    would-be OWNED, never to promote anything, so it can only make the result
    safer. It is what catches ``Cupid`` against ``Cupid (Twin Version)`` — same
    title, same artist, 41 seconds apart.
    """
    return delta is not None and delta > thresholds.differ_seconds


def _id_rung(track: LibraryTrack) -> Rung:
    """Rung 0 when the id came from a tag, 0b when it came from the filename."""
    if track.video_id and f"[{track.video_id}]" in os.path.basename(track.path):
        return Rung.FILE_IDENTITY
    return Rung.IDENTITY


def _result(
    wanted: WantedTrack,
    verdict: Verdict,
    rung: Rung,
    reason: ReasonCode,
    text: str,
    candidates: Optional[Sequence[Candidate]] = None,
) -> GapResult:
    return GapResult(
        wanted=wanted,
        verdict=verdict,
        rung=rung,
        candidates=list(candidates or []),
        reason_code=reason,
        reason_text=text,
    )


# ── individual rungs ─────────────────────────────────────────────────────────


def _rung_identity(
    wanted: WantedTrack, snapshot: LibrarySnapshot, caps: SourceCapabilities
) -> Optional[GapResult]:
    """Rung 0 / 0b — an exact source identity on both sides. Definitive."""
    if not caps.video_id or not wanted.video_id:
        return None
    rows = snapshot.by_video_id.get(wanted.video_id)
    if not rows:
        return None
    cands = [
        Candidate(
            track=snapshot.tracks[i],
            score=1.0,
            rung=_id_rung(snapshot.tracks[i]),
            duration_delta=duration_delta(wanted.duration, snapshot.tracks[i].duration),
        )
        for i in rows
    ]
    rung = cands[0].rung
    reason = ReasonCode.FILE_ID_MATCH if rung is Rung.FILE_IDENTITY else ReasonCode.ID_MATCH
    where = "the filename" if rung is Rung.FILE_IDENTITY else "a source tag"
    return _result(
        wanted, Verdict.OWNED, rung, reason,
        f"Same source id ({wanted.video_id}) found in {where} — this is the same upload.",
        cands,
    )


def _rung_core_artist(
    wanted: WantedTrack,
    splits: Sequence[SplitCandidate],
    snapshot: LibrarySnapshot,
    thresholds: Thresholds,
) -> Optional[GapResult]:
    """Rung 2 — core title and primary artist agree exactly.

    Probes every split. If two different splits land on different tracks the
    parse could not be disambiguated, and the row is unsure by construction.
    """
    hits: List[Tuple[SplitCandidate, int]] = []
    for split in splits:
        if not split.artist:
            continue
        key = (compare_key(split.artist), compare_key(split.title))
        for row in snapshot.by_artist_title.get(key, []):
            hits.append((split, row))

    if not hits:
        return None

    distinct_tracks = {row for _split, row in hits}
    cands: List[Candidate] = []
    for split, row in hits:
        track = snapshot.tracks[row]
        wanted_only, cand_only = modifier_delta(
            split.modifiers, _library_modifiers(track)
        )
        cands.append(
            Candidate(
                track=track,
                score=1.0,
                rung=Rung.CORE_ARTIST,
                duration_delta=duration_delta(wanted.duration, track.duration),
                modifier_delta=tuple(wanted_only) + tuple(cand_only),
                split=split,
            )
        )
    cands.sort(key=lambda c: (len(c.modifier_delta), c.duration_delta or 0))
    best = cands[0]

    if len({(s.artist, s.title) for s, _ in hits}) > 1 and len(distinct_tracks) > 1:
        return _result(
            wanted, Verdict.UNSURE, Rung.CORE_ARTIST, ReasonCode.SPLIT_AMBIGUOUS,
            "More than one way of reading this title matched a different track — "
            "the library can't tell which reading is right.",
            cands,
        )

    # The duration gate outranks the modifier check: a runtime gap this large
    # means a different recording however the titles read, and MISSING is the
    # safer of the two answers (spec §7.2).
    if _too_far_apart(best.duration_delta, thresholds):
        return _result(
            wanted, Verdict.MISSING, Rung.CORE_ARTIST, ReasonCode.DURATION_GAP,
            f"Title and artist match, but the runtimes are {best.duration_delta}s apart "
            f"(over the {thresholds.differ_seconds}s limit) — a different recording.",
            cands,
        )

    if best.modifier_delta:
        names = ", ".join(sorted({m.raw for m in best.modifier_delta}))
        return _result(
            wanted, Verdict.UNSURE, Rung.MODIFIER, ReasonCode.MOD_AMBIGUOUS,
            f"Same title and artist, but the versions differ ({names}).",
            cands,
        )

    return _result(
        wanted, Verdict.OWNED, Rung.CORE_ARTIST, ReasonCode.EXACT_MATCH,
        "Same core title and artist, and no version difference.",
        cands,
    )


def _rung_filename(
    wanted: WantedTrack,
    snapshot: LibrarySnapshot,
    thresholds: Thresholds,
    scorer: Scorer,
) -> Optional[GapResult]:
    """Rung 5b — fuzzy whole display string against library filenames.

    The primary path for a library sourced from YouTube: those files usually keep
    the video title as their filename while their tags are empty or junk, so the
    filename is closer to exact string matching than the tags are.
    """
    target = compare_key(wanted.display)
    if not target:
        return None
    shortlist = snapshot.shortlist_by_tokens(tokens_of(wanted.display), filenames=True)
    if not shortlist:
        return None

    scored: List[Candidate] = []
    for row in shortlist:
        track = snapshot.tracks[row]
        score = scorer.ratio(target, track.filename_norm)
        if score < thresholds.fuzzy_floor:
            continue
        scored.append(
            Candidate(
                track=track,
                score=score,
                rung=Rung.FUZZY_FILENAME,
                duration_delta=duration_delta(wanted.duration, track.duration),
            )
        )
    if not scored:
        return None
    scored.sort(key=lambda c: -c.score)
    best = scored[0]

    if _too_far_apart(best.duration_delta, thresholds):
        return _result(
            wanted, Verdict.MISSING, Rung.FUZZY_FILENAME, ReasonCode.DURATION_GAP,
            f"A file with a very similar name exists but runs {best.duration_delta}s "
            "longer or shorter — a different recording.",
            scored,
        )
    if best.score >= thresholds.fuzzy_confident:
        return _result(
            wanted, Verdict.OWNED, Rung.FUZZY_FILENAME, ReasonCode.FUZZY_CONFIDENT,
            f"A library filename matches this almost exactly ({best.score:.0%}).",
            scored,
        )
    return _result(
        wanted, Verdict.UNSURE, Rung.FUZZY_FILENAME, ReasonCode.FUZZY_WEAK,
        f"A library filename looks similar ({best.score:.0%}) but not similar enough to be sure.",
        scored,
    )


def _library_modifiers(track: LibraryTrack) -> Tuple:
    """Modifiers already parsed onto a library track, if any. Phase 2 fills this."""
    return getattr(track, "_modifiers", ()) or ()


# ── the ladder ───────────────────────────────────────────────────────────────


def match_one(
    wanted: WantedTrack,
    snapshot: LibrarySnapshot,
    *,
    caps: SourceCapabilities,
    lexicon: Lexicon,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    scorer: Scorer = DEFAULT_SCORER,
) -> GapResult:
    """Run one wanted track down the ladder."""
    if caps.availability and not wanted.available:
        return _result(
            wanted, Verdict.UNAVAILABLE, Rung.NONE, ReasonCode.SOURCE_UNAVAILABLE,
            "The source says this row is unavailable (deleted or private).",
        )
    if not (wanted.display or wanted.title):
        return _result(
            wanted, Verdict.UNAVAILABLE, Rung.NONE, ReasonCode.SOURCE_UNAVAILABLE,
            "This row has no title — it is reported rather than dropped.",
        )

    identity = _rung_identity(wanted, snapshot, caps)
    if identity is not None:
        return identity

    splits = candidate_splits(
        wanted.display,
        lexicon=lexicon,
        provided_artist=wanted.artist if caps.artist else None,
        provided_title=wanted.title if caps.title else None,
        known_artists=snapshot.artists,
    )

    exact = _rung_core_artist(wanted, splits, snapshot, thresholds)
    if exact is not None:
        return exact

    by_name = _rung_filename(wanted, snapshot, thresholds, scorer)
    if by_name is not None:
        return by_name

    # Checked across every split, not just the first: a source that supplies a
    # title column contributes a higher-prior "provided" candidate that would
    # otherwise hide the mashup reading behind it.
    if looks_like_mashup(wanted.display) and all(s.artist is None for s in splits):
        return _result(
            wanted, Verdict.UNSURE, Rung.NONE, ReasonCode.MASHUP_UNPARSED,
            "This looks like a mashup with no artist to search on — needs a look.",
        )
    return _result(
        wanted, Verdict.MISSING, Rung.NONE, ReasonCode.NO_CANDIDATE,
        "Nothing in the library plausibly matches this.",
    )


def match_all(
    wanted: Sequence[WantedTrack],
    snapshot: LibrarySnapshot,
    *,
    caps: SourceCapabilities,
    lexicon: Lexicon,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    scorer: Scorer = DEFAULT_SCORER,
    decisions: Optional[Mapping[str, Tuple[Verdict, str]]] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[GapResult]:
    """Run every wanted track down the ladder, applying stored decisions.

    ``decisions`` maps ``row_id`` to a remembered (verdict, note) the user made
    previously. A user decision always beats a fresh automatic one — that is what
    stops the same question being asked every month.
    """
    results: List[GapResult] = []
    total = len(wanted)
    for position, row in enumerate(wanted):
        if should_cancel is not None and should_cancel():
            break
        if progress is not None:
            progress(position + 1, total, row.label())

        result = match_one(
            row, snapshot, caps=caps, lexicon=lexicon,
            thresholds=thresholds, scorer=scorer,
        )

        stored = (decisions or {}).get(row.row_id)
        if stored is not None:
            verdict, note = stored
            result.verdict = verdict
            result.auto = False
            result.reason_code = ReasonCode.USER_DECISION
            result.reason_text = note or f"You decided this one before: {verdict.value}."
        results.append(result)
    return results


def summarize(results: Iterable[GapResult]) -> Dict[str, int]:
    """Bucket counts, for the tile strip and the run report."""
    counts: Dict[str, int] = {v.value: 0 for v in Verdict}
    for result in results:
        counts[result.verdict.value] = counts.get(result.verdict.value, 0) + 1
    counts["total"] = sum(counts[v.value] for v in Verdict)
    return counts


def group_by_reason(results: Iterable[GapResult]) -> Dict[str, List[GapResult]]:
    """Group rows by reason code — the keys for bulk triage."""
    grouped: Dict[str, List[GapResult]] = {}
    for result in results:
        grouped.setdefault(result.reason_code.value, []).append(result)
    return grouped
