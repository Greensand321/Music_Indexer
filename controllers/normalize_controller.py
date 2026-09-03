"""Canonical genre mapping: scan raw genres, apply a mapping, preview-first.

Qt-free. The heavy collaborators — file discovery, the tag reader and the tag
writer — are injectable so the planning and apply rules can be tested without
mutagen, musicbrainzngs or a library on disk; the defaults are imported lazily
inside the functions that need them.

The workflow this backs:

1. ``scan_raw_genres``  — collect every distinct genre string in the library.
2. (outside the app)    — paste that list into an LLM with ``PROMPT_TEMPLATE``
                          and get back a JSON mapping of raw -> canonical.
3. ``plan_genre_normalization`` — dry run: which files would change, and how.
4. ``apply_genre_changes``      — write exactly the planned changes, nothing more.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

PROMPT_TEMPLATE = """
I will provide a list of raw music genres (one per line). Your task is to group and map each raw genre into a canonical key in JSON format, for example:

{
  "rock & roll": ["Rock"],
  "future bass": ["Future Bass","Electronic"],
  "indie rock": ["Indie Rock","Rock"],
  "90s": ["invalid"]
}

Follow these guidelines:

• For each raw genre key, return an array of one or more canonical genre names as the value.
• If a genre has clearly defined subgenres, list both the subgenre and its parent(s) (e.g. "future bass": ["Future Bass","Electronic"]).
• Split and list merged terms separately (e.g. "hiphoprap": ["Hip-Hop","Rap"]).
• Map non-genres to ["invalid"].
• Ask clarifying questions if any terms are ambiguous.
"""

#: Mapping value (from the prompt above) meaning "this is not a genre; drop it".
INVALID_MARKER = "invalid"

_SKIP = object()

MAPPING_FILENAME = ".genre_mapping.json"


def mapping_path(folder: str) -> str:
    """Return where a library's saved genre mapping lives."""
    return os.path.join(folder, "Docs", MAPPING_FILENAME)


def load_mapping(folder: str) -> tuple[Dict[str, object], str]:
    """Return ``(mapping, path)``; the mapping is empty if none is saved."""
    path = mapping_path(folder)
    mapping: Dict[str, object] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                mapping = loaded
        except Exception:
            mapping = {}
    return mapping, path


def save_mapping(folder: str, mapping: Dict[str, object]) -> str:
    """Save mapping JSON and return its path."""
    path = mapping_path(folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return path


def _split_and_clean(raw: str) -> list[str]:
    """Split a raw genre string on common delimiters and return cleaned parts."""

    if not isinstance(raw, str):
        return []

    parts = re.split(r"[;,/]+", raw)
    cleaned = []
    for part in parts:
        part = part.strip()
        if part:
            cleaned.append(part)
    return cleaned


def clean_mapping(mapping: Dict[str, object]) -> Dict[str, list[str] | None]:
    """Normalise a pasted/loaded mapping into ``{raw: [canonical, ...] | None}``.

    ``None`` means "drop this genre". Both an explicit ``null`` and the
    ``["invalid"]`` marker the prompt asks for are treated that way — the legacy
    app stripped nulls out before saving and then wrote the literal word
    "invalid" into files as if it were a genre, which is the opposite of what
    the prompt promises the user.
    """
    cleaned: Dict[str, list[str] | None] = {}
    for raw_key, raw_value in (mapping or {}).items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if not key:
            continue

        if raw_value is None:
            cleaned[key] = None
            continue

        values: list[str] = []
        raw_values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in raw_values:
            if value is None:
                continue
            values.extend(_split_and_clean(str(value)))

        if not values or any(v.casefold() == INVALID_MARKER for v in values):
            cleaned[key] = None
        else:
            cleaned[key] = values
    return cleaned


def _prepare_mapping(mapping: Dict[str, object]) -> Dict[str, list[str] | object]:
    """Return a case-insensitive lookup table ready for normalization."""

    prepared: Dict[str, list[str] | object] = {}
    for key, value in clean_mapping(mapping).items():
        prepared[key.casefold()] = _SKIP if value is None else value
    return prepared


def normalize_genres(genres: Iterable[str], mapping: Dict[str, object]) -> list[str]:
    """Return *genres* rewritten through *mapping*, de-duplicated, order kept.

    Unmapped entries fall through unchanged (after splitting on ``;,/``) so a
    partial mapping still leaves a library no worse than it started.
    """

    normalized: list[str] = []
    seen: set[str] = set()
    prepared_mapping = _prepare_mapping(mapping)

    for raw in genres:
        if not isinstance(raw, str):
            continue

        cleaned = raw.strip()
        if not cleaned:
            continue

        lookup_key = cleaned.casefold()
        mapped = prepared_mapping.get(lookup_key)
        if mapped is _SKIP:
            continue

        values = mapped if mapped is not None else _split_and_clean(cleaned)

        for value in values:
            key = value.casefold()
            if value and key not in seen:
                normalized.append(value)
                seen.add(key)

    return normalized


def get_raw_genres(records):
    """
    Given a list of FileRecord objects, return a sorted, deduplicated list of
    all raw genre strings (from rec.old_genres and rec.new_genres).
    """
    raw_set = set()
    for rec in records:
        # include whatever fields you want—typically the pre-normalized tags:
        raw_set.update(rec.old_genres)
        # if you want newly suggested genres too:
        raw_set.update(rec.new_genres)
    return sorted(raw_set)


def _genre_list(raw) -> list[str]:
    """Coerce a tag reader's ``genre`` value into a list of strings."""
    if raw in (None, ""):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(v) for v in raw if isinstance(v, str) and v.strip()]
    return [str(raw)]


# ── Injectable collaborators ─────────────────────────────────────────────────
# Real defaults are imported lazily: tagfix_controller pulls in mutagen and
# update_genres imports musicbrainzngs at module level, neither of which a
# headless test (or a machine without them) should need to plan a change.

ProgressCallback = Callable[[int, int], None]


def _default_discover(folder: str) -> List[str]:
    from controllers.tagfix_controller import discover_files

    return discover_files(folder)


def _default_read_tags(path: str) -> Dict[str, object]:
    from utils.audio_metadata_reader import read_tags

    return read_tags(path)


def _default_writer(path: str, genres: Sequence[str]) -> bool:
    from update_genres import update_genre_tag

    return bool(update_genre_tag(path, list(genres)))


def scan_raw_genres(
    folder: str,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    files: Optional[Sequence[str]] = None,
    read_tags: Optional[Callable[[str], Dict[str, object]]] = None,
) -> list[str]:
    """Return a sorted, de-duplicated list of every raw genre in *folder*.

    Combined entries are split on ``;``, ``,`` and ``/`` so "Hip-Hop/Rap" shows
    up as two raw genres — that is what the mapping needs to see.
    """
    progress = progress_callback or (lambda _i, _t: None)
    paths = list(files) if files is not None else _default_discover(folder)
    reader = read_tags or _default_read_tags

    total = len(paths)
    raw_set: set[str] = set()
    progress(0, total)
    for idx, path in enumerate(paths, start=1):
        progress(idx, total)
        for entry in _genre_list(reader(path).get("genre")):
            for part in _split_and_clean(entry):
                raw_set.add(part)
    return sorted(raw_set)


# ── Preview-first apply ───────────────────────────────────────────────────────


@dataclass
class GenreChange:
    """One file whose genre tag would be rewritten."""

    path: str
    before: list[str]
    after: list[str]


@dataclass
class NormalizationPlan:
    """Result of a dry run: what would change, and how much was looked at."""

    changes: list[GenreChange] = field(default_factory=list)
    scanned: int = 0        # files examined
    with_genres: int = 0    # files that had any genre tag at all

    @property
    def unchanged(self) -> int:
        return self.with_genres - len(self.changes)


def plan_genre_normalization(
    folder: str,
    mapping: Dict[str, object],
    progress_callback: Optional[ProgressCallback] = None,
    *,
    files: Optional[Sequence[str]] = None,
    read_tags: Optional[Callable[[str], Dict[str, object]]] = None,
) -> NormalizationPlan:
    """Dry run: compute every file whose genres *mapping* would change.

    Reads tags, never writes. Files with no genre tag are skipped — there is
    nothing to normalise — and files whose normalised genres equal their
    current ones are counted but not planned, so the plan is exactly the set of
    writes :func:`apply_genre_changes` will perform.
    """
    progress = progress_callback or (lambda _i, _t: None)
    paths = list(files) if files is not None else _default_discover(folder)
    reader = read_tags or _default_read_tags

    plan = NormalizationPlan(scanned=len(paths))
    progress(0, plan.scanned)
    for idx, path in enumerate(paths, start=1):
        progress(idx, plan.scanned)
        before = _genre_list(reader(path).get("genre"))
        if not before:
            continue
        plan.with_genres += 1
        after = normalize_genres(before, mapping)
        if after != before:
            plan.changes.append(GenreChange(path=path, before=before, after=after))
    return plan


@dataclass
class ApplyResult:
    applied: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    cancelled: bool = False


def apply_genre_changes(
    changes: Sequence[GenreChange],
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event=None,
    *,
    writer: Optional[Callable[[str, Sequence[str]], bool]] = None,
) -> ApplyResult:
    """Write the genres in *changes* to their files — nothing else.

    This deliberately takes the plan rather than re-reading the library: the
    execution step must perform exactly what the user previewed, not a fresh
    decision made after they clicked. A cancellation stops after the current
    file; files already written stay written and are reported as such.
    """
    progress = progress_callback or (lambda _i, _t: None)
    write = writer or _default_writer

    result = ApplyResult()
    total = len(changes)
    progress(0, total)
    for idx, change in enumerate(changes, start=1):
        if cancel_event is not None and cancel_event.is_set():
            result.cancelled = True
            break
        ok = False
        try:
            ok = write(change.path, change.after)
        except Exception:  # noqa: BLE001 - one bad file must not abort the batch
            ok = False
        (result.applied if ok else result.failed).append(change.path)
        progress(idx, total)
    return result
