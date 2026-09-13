"""Modifier lexicon for Playlist Gap.

A title is rarely just a title:

    Midnight City (feat. Susanne Sundfor) - 2011 Remastered Radio Edit
    |---- core ----| |------------- modifiers -------------------------|

The lexicon's job is to *classify* those modifiers rather than strip them, because
the difference between two modifier sets is the evidence that decides whether two
strings name the same recording. "(feat. X)" against nothing is the same song
labelled two ways; "(Chris Lake Remix)" against nothing is a different recording.

Classification is data, not code: a shipped JSON table merged with a per-library
override file, so a user whose library is 40% remixes can extend it without
touching matching logic.

Qt-free. The only collaborator (where the files live) is injectable.

See ``docs/playlist_gap_spec.md`` §6.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from playlist_gap_types import Modifier, ModifierClass

#: Shipped defaults. A user file with the same rule ``id`` overrides an entry.
DEFAULT_LEXICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "data", "playlist_gap_lexicon.json")

#: Where a library's own overrides live, relative to the library root.
USER_LEXICON_RELPATH = os.path.join("Docs", "playlist_gap_lexicon.json")

_WORD_RE = re.compile(r"[a-z0-9]+")


def _key(text: str) -> str:
    """Normalized comparison key for a modifier."""
    return " ".join(_WORD_RE.findall(text.lower()))


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: str
    cls: ModifierClass
    #: "any" — match anywhere; "bracketed" — only inside (), [] or {}.
    #: Broad patterns like ``audio`` or bare ``mix`` would otherwise eat words
    #: out of real titles ("Audio Video Disco"), so they are bracket-only.
    scope: str = "any"
    regex: re.Pattern = field(compare=False, repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "Rule":
        pattern = str(raw["pattern"])
        return cls(
            id=str(raw["id"]),
            pattern=pattern,
            cls=ModifierClass(str(raw.get("class", "ambiguous"))),
            scope=str(raw.get("scope", "any")),
            regex=re.compile(pattern, re.IGNORECASE),
        )

    def applies(self, *, bracketed: bool) -> bool:
        return bracketed or self.scope != "bracketed"


@dataclass
class Lexicon:
    """Ordered rules plus the two allow-lists that are not patterns."""

    rules: List[Rule] = field(default_factory=list)
    #: Parentheticals that are part of a canonical title, not modifiers.
    title_parts: frozenset = frozenset()
    #: Bracketed uploader/label tags to lift off the front of a display string.
    uploader_prefixes: frozenset = frozenset()

    # ── classification ───────────────────────────────────────────────────

    def is_title_part(self, text: str) -> bool:
        """True when a parenthetical is part of the song's real name.

        ``Running Up That Hill (A Deal With God)`` must not lose its
        parenthetical, while ``Cupid (Twin Version)`` must. Bracket type cannot
        tell them apart, so this is an explicit list — necessarily incomplete,
        hence user-editable.
        """
        return _key(text) in self.title_parts

    def is_uploader_prefix(self, text: str) -> bool:
        return _key(text) in self.uploader_prefixes

    def classify(self, text: str, *, bracketed: bool = False) -> Optional[Modifier]:
        """Classify one modifier string, or None if it matches no rule.

        Rules are tried in file order, so specific patterns must precede general
        ones — ``"<name> remix"`` before bare ``"remix"``, and both before the
        catch-all ``"mix|version|edit"``.
        """
        cleaned = text.strip().strip("()[]{}").strip(" -–—·")
        if not cleaned:
            return None
        if self.is_title_part(cleaned):
            return Modifier(raw=cleaned, cls=ModifierClass.TITLE_PART, key=_key(cleaned))
        for rule in self.rules:
            if not rule.applies(bracketed=bracketed):
                continue
            match = rule.regex.search(cleaned)
            if not match:
                continue
            actor = None
            if "actor" in (match.groupdict() or {}):
                captured = match.group("actor")
                actor = captured.strip() if captured else None
            return Modifier(
                raw=cleaned,
                cls=rule.cls,
                key=f"{rule.id}:{_key(actor)}" if actor else rule.id,
                actor=actor,
            )
        return None

    def classify_all(
        self, texts: Iterable[str], *, bracketed: bool = False
    ) -> Tuple[Modifier, ...]:
        found = (self.classify(text, bracketed=bracketed) for text in texts)
        return tuple(m for m in found if m is not None)

    def scan(self, region: str, *, bracketed: bool = False) -> Tuple[Modifier, ...]:
        """Collect every rule that fires inside ``region``, in rule order.

        Used for trailing runs like "2011 Remastered Radio Edit", where taking
        only the first match would lose the ambiguity of "Radio Edit".
        """
        out: List[Modifier] = []
        seen: set = set()
        for rule in self.rules:
            if not rule.applies(bracketed=bracketed):
                continue
            match = rule.regex.search(region)
            if not match:
                continue
            actor = None
            if "actor" in (match.groupdict() or {}):
                captured = match.group("actor")
                actor = captured.strip() if captured else None
            key = f"{rule.id}:{_key(actor)}" if actor else rule.id
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Modifier(raw=match.group(0).strip(), cls=rule.cls, key=key, actor=actor)
            )
        return tuple(out)

    def first_match(self, text: str, *, bracketed: bool = False):
        """Earliest-starting rule match in ``text``, or None."""
        best = None
        for rule in self.rules:
            if not rule.applies(bracketed=bracketed):
                continue
            match = rule.regex.search(text)
            if match and (best is None or match.start() < best.start()):
                best = match
        return best


# ── loading ──────────────────────────────────────────────────────────────────


def _merge(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    """Override rules by id, keeping base order; union the allow-lists."""
    merged = dict(base)
    by_id = {str(r["id"]): dict(r) for r in base.get("rules", [])}  # type: ignore[union-attr]
    order = list(by_id)
    for raw in override.get("rules", []) or []:
        rule_id = str(raw["id"])
        if rule_id not in by_id:
            order.append(rule_id)
        by_id[rule_id] = dict(raw)
    merged["rules"] = [by_id[rid] for rid in order]
    for list_key in ("title_parts", "uploader_prefixes"):
        merged[list_key] = list(base.get(list_key, []) or []) + list(
            override.get(list_key, []) or []
        )
    return merged


def _build(raw: Dict[str, object]) -> Lexicon:
    rules: List[Rule] = []
    for entry in raw.get("rules", []) or []:
        try:
            rules.append(Rule.from_dict(entry))  # type: ignore[arg-type]
        except (KeyError, ValueError, re.error):
            # A malformed user rule must not take the whole lexicon down.
            continue
    return Lexicon(
        rules=rules,
        title_parts=frozenset(_key(str(t)) for t in raw.get("title_parts", []) or []),
        uploader_prefixes=frozenset(
            _key(str(t)) for t in raw.get("uploader_prefixes", []) or []
        ),
    )


def _read_json(path: str) -> Dict[str, object]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_lexicon(
    library_root: str | None = None,
    *,
    default_path: str = DEFAULT_LEXICON_PATH,
    user_path: str | None = None,
) -> Lexicon:
    """Load the shipped lexicon, merged with a library's overrides if present."""
    base = _read_json(default_path)
    if user_path is None and library_root:
        user_path = os.path.join(library_root, USER_LEXICON_RELPATH)
    override = _read_json(user_path) if user_path else {}
    return _build(_merge(base, override) if override else base)


def modifier_delta(
    wanted: Sequence[Modifier], candidate: Sequence[Modifier]
) -> Tuple[Tuple[Modifier, ...], Tuple[Modifier, ...]]:
    """Return (only-on-wanted, only-on-candidate), ignoring TITLE_PART.

    Title parts belong to the name, so they are compared as part of the core
    title rather than as a modifier difference.
    """
    def real(mods: Sequence[Modifier]) -> Dict[str, Modifier]:
        return {m.key: m for m in mods if m.cls is not ModifierClass.TITLE_PART}

    w, c = real(wanted), real(candidate)
    return (
        tuple(w[k] for k in w if k not in c),
        tuple(c[k] for k in c if k not in w),
    )
