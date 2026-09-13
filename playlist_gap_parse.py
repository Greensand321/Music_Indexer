"""Stage 2.5 — read a free-text display string into candidate artist/title splits.

The problem this solves: a YouTube Music export gives one free-text blob per row
and nothing else. ``FIFTY FIFTY - Cupid (Twin Version)`` is easy;
``French 79 . New Constellations - Colors Collide`` has two different delimiters
and no way to tell which marks the artist boundary; ``Self Aware x Babydoll`` is
a mashup with no artist anywhere.

**The governing rule is: never commit to a parse.** Emitting one "best guess"
split and matching on it turns an unlucky guess into a wrong verdict. Instead
this module emits *several* candidate readings, ranked by a prior, and the
matcher probes the library with all of them. The library is the oracle — exactly
one candidate hitting is both the match and the proof that the parse was right;
two candidates hitting different tracks is an honest "unsure".

Qt-free, no I/O, no third-party imports. See ``docs/playlist_gap_spec.md`` §5.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

from playlist_gap_lexicon import Lexicon
from playlist_gap_types import Modifier, ModifierClass, SplitCandidate

#: Delimiters that separate an artist from a title, in priority order.
DELIMITERS: Tuple[str, ...] = (" - ", " – ", " — ", " · ", " | ", " _ ", ": ")

#: Markers meaning "these two things were mashed together", never "featuring".
#:
#: This is why ``music_indexer_api.extract_primary_and_collabs()`` must not be
#: reused on display strings: its separator list contains " x ", so it reads
#: "Self Aware x Babydoll" as the artist *Self Aware* featuring *Babydoll*.
#:
#: Case matters for ``x`` and only for ``x``: an upper-case " X " is far more
#: often part of a name (Malcolm X, Gen X, X Ambassadors) than a mashup join.
#: Missing a mashup only sends the row to "missing" instead of "unsure", which
#: is the safe direction; misreading a name would mangle the artist.
MASHUP_MARKERS_CASED: Tuple[str, ...] = (" x ",)
MASHUP_MARKERS_ANY_CASE: Tuple[str, ...] = (" vs ", " vs. ", " v/s ", " mashup ")
MASHUP_MARKERS: Tuple[str, ...] = MASHUP_MARKERS_CASED + MASHUP_MARKERS_ANY_CASE

_BRACKETS = {"(": ")", "[": "]", "{": "}"}
_BRACKET_RE = re.compile(r"[\(\[\{]([^\)\]\}]*)[\)\]\}]")
_LEADING_BRACKET_RE = re.compile(r"^\s*[\(\[\{]([^\)\]\}]*)[\)\]\}]\s*")
_WORD_RE = re.compile(r"[a-z0-9]+")
_SMART = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    " ": " ", "​": "", "﻿": "",
}


def fold_text(text: str) -> str:
    """Structural normalization applied before parsing.

    NFKC, straighten quotes, unify dashes, collapse whitespace. Accents are
    *kept* here — this output is still shown to the user; accent folding belongs
    to comparison keys (``compare_key``), not to display.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    for bad, good in _SMART.items():
        out = out.replace(bad, good)
    return re.sub(r"\s+", " ", out).strip()


def strip_accents(text: str) -> str:
    """Drop combining marks: Beyonce <- Beyoncé. No dependency needed."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def compare_key(text: Optional[str], *, fold_accents: bool = True) -> str:
    """Normalized key for comparing two strings.

    Deliberately lossy: lowercase, optional accent folding, then alphanumeric
    runs joined by single spaces.

    Apostrophes are *elided* rather than treated as separators, so ``Don't Stop``
    and ``Dont Stop`` produce the same key. The Indexer strips apostrophes when
    it sanitises filenames, so without this a wanted row would fail to match the
    very file it names — a false "missing" and a needless re-download.
    """
    if not text:
        return ""
    value = fold_text(text)
    if fold_accents:
        value = strip_accents(value)
    value = value.replace("'", "").replace("\u2019", "")
    return " ".join(_WORD_RE.findall(value.lower()))


# ── modifier extraction ──────────────────────────────────────────────────────


def _extract_bracketed(text: str, lexicon: Lexicon) -> Tuple[str, List[Modifier]]:
    """Pull classified bracketed groups out of ``text``.

    A group classified TITLE_PART is left in place — ``Running Up That Hill
    (A Deal With God)`` must keep its parenthetical, while ``Cupid (Twin
    Version)`` must lose it. Bracket type cannot tell those apart, so the
    lexicon's allow-list does.
    """
    mods: List[Modifier] = []
    removed: List[Tuple[int, int]] = []
    for match in _BRACKET_RE.finditer(text):
        inner = match.group(1).strip()
        if not inner:
            continue
        classified = lexicon.classify(inner, bracketed=True)
        if classified is None or classified.cls is ModifierClass.TITLE_PART:
            continue
        mods.append(classified)
        removed.append(match.span())
    if not removed:
        return text, mods
    kept = []
    last = 0
    for start, end in removed:
        kept.append(text[last:start])
        last = end
    kept.append(text[last:])
    return re.sub(r"\s+", " ", "".join(kept)).strip(" -–—·|"), mods


def _mask_brackets(text: str) -> str:
    """Blank out bracketed spans, preserving length so indices stay valid.

    Anything still in brackets at this point was deliberately kept — a title part
    like ``(A Deal With God)``. Without masking, the trailing scan reads inside
    it and "With God" trips the ``feat`` rule, amputating the real title.
    """
    chars = list(text)
    for match in _BRACKET_RE.finditer(text):
        for i in range(match.start(), match.end()):
            chars[i] = "\x00"
    return "".join(chars)


def _extract_trailing(text: str, lexicon: Lexicon) -> Tuple[str, List[Modifier]]:
    """Pull an unbracketed trailing modifier run off the end of ``text``.

    Handles ``Resonance but it's beats 3,3`` and ``Cupid - 2011 Remastered Radio
    Edit``. The whole tail from the first matching rule onward is treated as one
    region and then scanned for *every* rule that fires, so the "Radio Edit"
    half of that second example is not lost behind the "Remastered" half.
    """
    match = lexicon.first_match(_mask_brackets(text))
    if match is None or match.start() == 0:
        # start == 0 would mean the entire title is a modifier, which is never
        # what the user meant — leave it alone.
        return text, []
    core = text[: match.start()].strip(" -–—·|,")
    if not core:
        return text, []
    return core, list(lexicon.scan(text[match.start():]))


def _split_modifiers(title: str, lexicon: Lexicon) -> Tuple[str, Tuple[Modifier, ...]]:
    core, bracketed = _extract_bracketed(title, lexicon)
    core, trailing = _extract_trailing(core, lexicon)
    seen: set = set()
    mods: List[Modifier] = []
    for mod in bracketed + trailing:
        if mod.key in seen:
            continue
        seen.add(mod.key)
        mods.append(mod)
    return core.strip(), tuple(mods)


# ── splitting ────────────────────────────────────────────────────────────────


def looks_like_mashup(text: str) -> bool:
    """True when a display string joins two things rather than naming one.

    A property of the string itself, so callers can ask directly instead of
    inferring it from a candidate's ``origin`` — candidates get deduplicated by
    meaning, and an identically-worded higher-prior reading would swallow the
    mashup one and take the marker with it.
    """
    padded = f" {fold_text(text)} "
    if any(marker in padded for marker in MASHUP_MARKERS_CASED):
        return True
    lowered = padded.lower()
    return any(marker in lowered for marker in MASHUP_MARKERS_ANY_CASE)


#: Backwards-compatible private alias.
_looks_like_mashup = looks_like_mashup


def _candidate(
    artist: Optional[str],
    title: str,
    mods: Sequence[Modifier],
    prior: float,
    origin: str,
) -> Optional[SplitCandidate]:
    title = (title or "").strip(" -–—·|")
    artist = (artist or "").strip(" -–—·|") or None
    if not title or len(title) < 2:
        return None
    if artist is not None and len(artist) < 2:
        artist, prior = None, prior - 0.2
    return SplitCandidate(
        artist=artist,
        title=title,
        modifiers=tuple(mods),
        prior=max(0.0, min(1.0, prior)),
        origin=origin,
    )


def candidate_splits(
    display: str,
    *,
    lexicon: Lexicon,
    provided_artist: Optional[str] = None,
    provided_title: Optional[str] = None,
    channel_hint: Optional[str] = None,
    known_artists: Optional[frozenset] = None,
    max_candidates: int = 8,
) -> List[SplitCandidate]:
    """Return plausible readings of ``display``, best prior first.

    ``provided_artist`` / ``provided_title`` come from sources rich enough to
    supply them; they are emitted as the top candidate but do **not** suppress
    the others, because a "provided" artist is sometimes only a channel name.
    """
    text = fold_text(display)
    out: List[SplitCandidate] = []

    if provided_title:
        core, mods = _split_modifiers(fold_text(provided_title), lexicon)
        made = _candidate(fold_text(provided_artist or "") or None, core, mods, 1.0, "provided")
        if made:
            out.append(made)

    if not text:
        return out[:max_candidates]

    # An uploader/label tag in leading brackets is neither artist nor title.
    prefix_match = _LEADING_BRACKET_RE.match(text)
    if prefix_match:
        inner = prefix_match.group(1).strip()
        classified = lexicon.classify(inner, bracketed=True)
        is_modifier = classified is not None and classified.cls is not ModifierClass.TITLE_PART
        if not is_modifier:
            # Not a known modifier, so treat it as an uploader tag and drop it.
            text = text[prefix_match.end():].strip()

    if _looks_like_mashup(text) and not any(d in text for d in DELIMITERS):
        core, mods = _split_modifiers(text, lexicon)
        made = _candidate(None, core, mods, 0.6, "mashup")
        if made:
            out.append(made)
        return _finalize(out, max_candidates)

    for delimiter in DELIMITERS:
        if delimiter not in text:
            continue
        pieces = text.split(delimiter)
        for cut in range(1, len(pieces)):
            left = delimiter.join(pieces[:cut]).strip()
            right = delimiter.join(pieces[cut:]).strip()
            if not left or not right:
                continue
            core, mods = _split_modifiers(right, lexicon)
            forward = _candidate(left, core, mods, 0.8, f"delimiter:{delimiter.strip()}")
            if forward:
                out.append(_adjust(forward, known_artists, channel_hint))
            rev_core, rev_mods = _split_modifiers(left, lexicon)
            reverse = _candidate(right, rev_core, rev_mods, 0.35, f"reversed:{delimiter.strip()}")
            if reverse:
                out.append(_adjust(reverse, known_artists, channel_hint))

    core, mods = _split_modifiers(text, lexicon)
    whole = _candidate(None, core, mods, 0.15, "whole")
    if whole:
        out.append(whole)
    return _finalize(out, max_candidates)


def _adjust(
    cand: SplitCandidate,
    known_artists: Optional[frozenset],
    channel_hint: Optional[str],
) -> SplitCandidate:
    """Nudge a candidate's prior using cheap outside signals.

    Priors only order the probes — they never decide a verdict on their own.
    """
    bonus = 0.0
    key = compare_key(cand.artist)
    if known_artists and key and key in known_artists:
        bonus += 0.10
    if channel_hint and key and key == compare_key(channel_hint):
        bonus += 0.05
    if not bonus:
        return cand
    return SplitCandidate(
        artist=cand.artist,
        title=cand.title,
        modifiers=cand.modifiers,
        prior=min(1.0, cand.prior + bonus),
        origin=cand.origin,
    )


def _finalize(cands: Sequence[SplitCandidate], limit: int) -> List[SplitCandidate]:
    """Deduplicate by meaning, keep the strongest prior, and rank."""
    best: dict = {}
    for cand in cands:
        key = (compare_key(cand.artist), compare_key(cand.title), cand.modifier_keys())
        if key not in best or cand.prior > best[key].prior:
            best[key] = cand
    ordered = sorted(best.values(), key=lambda c: (-c.prior, c.origin))
    return ordered[:limit]
