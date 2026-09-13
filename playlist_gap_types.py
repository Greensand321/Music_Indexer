"""Shared type definitions for the Playlist Gap feature.

Types only — enums and dataclasses, no business logic and no imports beyond the
standard library. This module exists for the same reason ``library_sync_types``
does: the matcher, the ledger, the sources and the reporters all need these
names, and without a leaf module they would import each other in a cycle.

See ``docs/playlist_gap_spec.md`` §2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Verdict(str, Enum):
    """Which bucket a wanted track lands in."""

    OWNED = "owned"
    UNSURE = "unsure"
    MISSING = "missing"
    IGNORED = "ignored"
    UNAVAILABLE = "unavailable"


#: Verdict precedence, lowest first. Used to resolve disagreeing signals.
#:
#: The ordering encodes the feature's governing rule: uncertainty must never
#: resolve toward OWNED, because a wrong "you already have this" silently drops a
#: song the user wanted, while a wrong "you're missing this" only costs a
#: duplicate the Duplicate Finder can clean up.
_PRECEDENCE: Tuple[Verdict, ...] = (
    Verdict.OWNED,
    Verdict.UNSURE,
    Verdict.MISSING,
    Verdict.IGNORED,
    Verdict.UNAVAILABLE,
)


def stronger(a: Verdict, b: Verdict) -> Verdict:
    """Return whichever verdict wins when two signals disagree."""
    return a if _PRECEDENCE.index(a) >= _PRECEDENCE.index(b) else b


class Rung(str, Enum):
    """Which step of the match ladder resolved a row."""

    IDENTITY = "0"
    FILE_IDENTITY = "0b"
    EXACT_TAG = "1"
    CORE_ARTIST = "2"
    MODIFIER = "3"
    FUZZY_IN_ARTIST = "4"
    FUZZY_GLOBAL = "5"
    FUZZY_FILENAME = "5b"
    NONE = "6"


class ModifierClass(str, Enum):
    """What a title modifier means for identity."""

    COSMETIC = "cosmetic"        # same recording, labelled differently
    SUBSTANTIVE = "substantive"  # a genuinely different recording
    AMBIGUOUS = "ambiguous"      # library-dependent; never guessed
    TITLE_PART = "title_part"    # not a modifier — part of the real title


class ReasonCode(str, Enum):
    """Stable machine reason for a verdict.

    These double as the grouping keys for bulk triage, so they are deliberately
    coarse: a user should be able to resolve a whole class at once.
    """

    ID_MATCH = "id_match"
    FILE_ID_MATCH = "file_id_match"
    EXACT_MATCH = "exact_match"
    MOD_COSMETIC_ONLY = "mod_cosmetic_only"
    MOD_WANTED_SUBSTANTIVE = "mod_wanted_substantive"
    MOD_CANDIDATE_SUBSTANTIVE = "mod_candidate_substantive"
    MOD_AMBIGUOUS = "mod_ambiguous"
    MOD_DIFFERENT_ACTOR = "mod_different_actor"
    DURATION_GAP = "duration_gap"
    FUZZY_CONFIDENT = "fuzzy_confident"
    FUZZY_WEAK = "fuzzy_weak"
    SPLIT_AMBIGUOUS = "split_ambiguous"
    MASHUP_UNPARSED = "mashup_unparsed"
    NO_CANDIDATE = "no_candidate"
    SOURCE_UNAVAILABLE = "source_unavailable"
    USER_DECISION = "user_decision"


@dataclass(frozen=True)
class SourceCapabilities:
    """What a wanted-list source can actually provide.

    The match ladder skips rungs whose inputs are unavailable, which is what lets
    a bare CSV of video titles and a fully-populated ytmusicapi feed share one
    engine. ``display_string`` is the floor and is always true.
    """

    display_string: bool = True
    title: bool = False
    artist: bool = False
    album: bool = False
    duration: bool = False
    isrc: bool = False
    video_id: bool = False
    availability: bool = False
    artwork_url: bool = False

    def describe(self) -> str:
        """Human-readable list of the fields this source supplies."""
        names = [
            name
            for name in (
                "title", "artist", "album", "duration", "isrc", "video_id",
                "availability", "artwork_url",
            )
            if getattr(self, name)
        ]
        return " · ".join(names) if names else "display string only"


@dataclass
class WantedTrack:
    """One row of "songs I want", normalized across sources."""

    row_id: str
    display: str
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    duration: Optional[int] = None          # seconds
    isrc: Optional[str] = None
    video_id: Optional[str] = None
    artwork_url: Optional[str] = None
    available: bool = True
    source_id: str = ""
    playlists: Tuple[str, ...] = ()
    position: Optional[int] = None

    def label(self) -> str:
        """Best human label: "Artist - Title" when known, else the raw display."""
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        return self.title or self.display


@dataclass
class LibraryTrack:
    """One audio file in the library snapshot."""

    path: str
    ext: str = ""
    duration: Optional[int] = None
    bitrate: Optional[int] = None
    fingerprint: Optional[str] = None
    tags: Dict[str, object] = field(default_factory=dict)
    norm_artist: Optional[str] = None
    norm_title: Optional[str] = None
    norm_album: Optional[str] = None
    video_id: Optional[str] = None
    filename_norm: str = ""


@dataclass(frozen=True)
class Modifier:
    """A classified title modifier, e.g. "(Twin Version)" or "feat. X"."""

    raw: str
    cls: ModifierClass
    key: str
    actor: Optional[str] = None


@dataclass(frozen=True)
class SplitCandidate:
    """One possible reading of a display string.

    The parser never commits to a single reading; it emits several of these and
    lets the library decide which one is right.
    """

    artist: Optional[str]
    title: str
    modifiers: Tuple[Modifier, ...] = ()
    prior: float = 0.5
    origin: str = "whole"

    def modifier_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(m.key for m in self.modifiers))


@dataclass
class Candidate:
    """A library track that might be the wanted recording."""

    track: LibraryTrack
    score: float
    rung: Rung
    duration_delta: Optional[int] = None
    modifier_delta: Tuple[Modifier, ...] = ()
    split: Optional[SplitCandidate] = None


@dataclass
class GapResult:
    """The outcome for one wanted track."""

    wanted: WantedTrack
    verdict: Verdict
    rung: Rung = Rung.NONE
    candidates: List[Candidate] = field(default_factory=list)
    reason_code: ReasonCode = ReasonCode.NO_CANDIDATE
    reason_text: str = ""
    auto: bool = True

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True)
class Thresholds:
    """Tunable comparison limits. Starting values; see spec §15."""

    same_seconds: int = 3
    differ_seconds: int = 15
    fuzzy_floor: float = 0.82
    fuzzy_confident: float = 0.94


DEFAULT_THRESHOLDS = Thresholds()
