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

    Each entry is looked up whole first ("hip hop/rap" can be mapped as a
    unit). If the whole string is not mapped, it is split on ``;,/`` and
    *each part* is looked up — the scan step lists split parts, so that is
    what a mapping built from it contains. Unmapped parts fall through
    unchanged so a partial mapping still leaves a library no worse than it
    started; parts mapped to ``["invalid"]``/``null`` are dropped.
    """

    normalized: list[str] = []
    seen: set[str] = set()
    prepared_mapping = _prepare_mapping(mapping)

    def _emit(values: Iterable[str]) -> None:
        for value in values:
            key = value.casefold()
            if value and key not in seen:
                normalized.append(value)
                seen.add(key)

    for raw in genres:
        if not isinstance(raw, str):
            continue

        cleaned = raw.strip()
        if not cleaned:
            continue

        mapped = prepared_mapping.get(cleaned.casefold())
        if mapped is _SKIP:
            continue
        if mapped is not None:
            _emit(mapped)
            continue

        for part in _split_and_clean(cleaned):
            part_mapped = prepared_mapping.get(part.casefold())
            if part_mapped is _SKIP:
                continue
            _emit(part_mapped if part_mapped is not None else [part])

    return normalized


def add_genres(existing: Sequence[str], extra: Sequence[str]) -> list[str]:
    """``existing`` unchanged, followed by each genre of ``extra`` it lacks.

    Purely additive: nothing already on the file is dropped, reordered or
    rewritten, and a genre present in any letter case is not added twice.
    Used by every tool that *adds* genres (Tag Fixer, Fill from MusicBrainz).
    """
    merged = [g for g in existing if isinstance(g, str)]
    seen = {g.strip().casefold() for g in merged}
    for genre in extra or []:
        if not isinstance(genre, str):
            continue
        cleaned = genre.strip()
        if cleaned and cleaned.casefold() not in seen:
            merged.append(cleaned)
            seen.add(cleaned.casefold())
    return merged


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
    """``{"genre": [every genre value]}`` — never just the first one.

    ``read_tags()["genre"]`` is the first value only; planning a rewrite from
    it would silently delete the others (a FLAC tagged Rock + Pop would be
    written back as just the mapped Rock).
    """
    from utils.audio_metadata_reader import read_genres

    return {"genre": read_genres(path)}


def _default_writer(path: str, genres: Sequence[str]) -> bool:
    """Write the genres; raise with the reason on failure (logged by the caller)."""
    from update_genres import write_genre_tag

    write_genre_tag(path, list(genres))
    return True


def _default_writable_check(path: str) -> Optional[str]:
    from utils.tag_writer import check_writable

    return check_writable(path)


def _default_lookup(path: str):
    from update_genres import lookup_musicbrainz_genres

    return lookup_musicbrainz_genres(path)


def scan_raw_genres(
    folder: str,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    files: Optional[Sequence[str]] = None,
    read_tags: Optional[Callable[[str], Dict[str, object]]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> list[str]:
    """Return a sorted, de-duplicated list of every raw genre in *folder*.

    Combined entries are split on ``;``, ``,`` and ``/`` so "Hip-Hop/Rap" shows
    up as two raw genres — that is what the mapping needs to see. Every genre
    value of a file is read, not only the first.
    """
    progress = progress_callback or (lambda _i, _t: None)
    paths = list(files) if files is not None else _default_discover(folder)
    reader = read_tags or _default_read_tags

    total = len(paths)
    raw_set: set[str] = set()
    progress(0, total)
    for idx, path in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
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
    #: Optional context shown in the preview (e.g. the MusicBrainz match).
    note: str = ""


@dataclass
class NormalizationPlan:
    """Result of a dry run: what would change, and how much was looked at."""

    changes: list[GenreChange] = field(default_factory=list)
    scanned: int = 0        # files examined
    with_genres: int = 0    # files that had any genre tag at all
    #: Files that *would* change but can't be written (read-only, a format
    #: that can't hold tags…): ``(path, reason)``. Shown in the preview and
    #: left out of ``changes``, so they don't fail at apply time and then
    #: reappear in every preview.
    blocked: list[tuple[str, str]] = field(default_factory=list)

    @property
    def unchanged(self) -> int:
        return self.with_genres - len(self.changes) - len(self.blocked)


def _resolve_writable_check(writable_check, read_tags):
    """The default writability probe touches the real disk, so it is only
    used when the reader is the real one too (injected readers mean fake
    paths in tests)."""
    if writable_check is not None:
        return writable_check
    if read_tags is None:
        return _default_writable_check
    return None


def plan_genre_normalization(
    folder: str,
    mapping: Dict[str, object],
    progress_callback: Optional[ProgressCallback] = None,
    *,
    files: Optional[Sequence[str]] = None,
    read_tags: Optional[Callable[[str], Dict[str, object]]] = None,
    writable_check: Optional[Callable[[str], Optional[str]]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> NormalizationPlan:
    """Dry run: compute every file whose genres *mapping* would change.

    Reads tags, never writes. Files with no genre tag are skipped — there is
    nothing to normalise — and files whose normalised genres equal their
    current ones are counted but not planned, so the plan is exactly the set of
    writes :func:`apply_genre_changes` will perform. Every genre value of a
    file is read (not just the first), so the preview shows — and apply
    writes — the complete list.
    """
    progress = progress_callback or (lambda _i, _t: None)
    paths = list(files) if files is not None else _default_discover(folder)
    reader = read_tags or _default_read_tags
    check = _resolve_writable_check(writable_check, read_tags)

    plan = NormalizationPlan(scanned=len(paths))
    progress(0, plan.scanned)
    for idx, path in enumerate(paths, start=1):
        if should_cancel is not None and should_cancel():
            break
        progress(idx, plan.scanned)
        before = _genre_list(reader(path).get("genre"))
        if not before:
            continue
        plan.with_genres += 1
        after = normalize_genres(before, mapping)
        if after != before:
            reason = check(path) if check is not None else None
            if reason:
                plan.blocked.append((path, reason))
            else:
                plan.changes.append(GenreChange(path=path, before=before, after=after))
    return plan


@dataclass
class ApplyResult:
    applied: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    cancelled: bool = False
    #: Files left alone because they no longer held the previewed genres.
    skipped: list[str] = field(default_factory=list)
    #: Where the before/after record of this run was written (if anywhere).
    log_path: str = ""


def apply_genre_changes(
    changes: Sequence[GenreChange],
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event=None,
    *,
    writer: Optional[Callable[[str, Sequence[str]], bool]] = None,
    read_tags: Optional[Callable[[str], Dict[str, object]]] = None,
    change_log=None,
) -> ApplyResult:
    """Write the genres in *changes* to their files — nothing else.

    This deliberately takes the plan rather than re-reading the library: the
    execution step must perform exactly what the user previewed, not a fresh
    decision made after they clicked. Before each write the file's current
    genres are compared with the previewed ``before`` (when writing to real
    files, or when ``read_tags`` is given): a file that changed since the
    preview is skipped, never overwritten blind. A cancellation stops after
    the current file; files already written stay written and are reported as
    such. ``change_log`` (a :class:`utils.tag_change_log.TagChangeLog`), when
    given, receives every file's before/after values and outcome.
    """
    progress = progress_callback or (lambda _i, _t: None)
    write = writer or _default_writer
    verify_reader = read_tags or (_default_read_tags if writer is None else None)

    result = ApplyResult()
    if change_log is not None:
        result.log_path = getattr(change_log, "path", "") or ""
    total = len(changes)
    progress(0, total)
    for idx, change in enumerate(changes, start=1):
        if cancel_event is not None and cancel_event.is_set():
            result.cancelled = True
            break
        record = {"genre": (list(change.before), list(change.after))}
        if verify_reader is not None:
            try:
                current = _genre_list(verify_reader(change.path).get("genre"))
            except Exception:  # noqa: BLE001
                current = None
            if current != list(change.before):
                result.skipped.append(change.path)
                if change_log is not None:
                    change_log.record(
                        change.path, record, "skipped",
                        "genres changed on disk since the preview",
                    )
                progress(idx, total)
                continue
        ok = False
        error = None
        try:
            ok = write(change.path, change.after)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            ok = False
            error = str(exc)
        (result.applied if ok else result.failed).append(change.path)
        if change_log is not None:
            change_log.record(
                change.path, record, "written" if ok else "failed",
                None if ok else (error or "write failed"),
            )
        progress(idx, total)
    return result


# ── Fill from MusicBrainz, preview-first ──────────────────────────────────────
#
# The Fill tab used to write each file the moment its lookup returned — no
# preview, no confirmation — and its "dry run" did no lookups, so it could not
# show what would be written. Now: lookups produce a FillPlan (no writes), the
# user reviews it, and ``apply_genre_changes`` writes exactly its changes.

#: Files that already carry this many genre values are left untouched.
FILL_SKIP_AT = 2


@dataclass
class FillLookup:
    """What one MusicBrainz lookup found for a file."""

    genres: list[str] = field(default_factory=list)
    #: "matched", "no_match", "no_tags", "missing_artist_title" or "error"
    status: str = "no_match"
    detail: str = ""
    #: True when a network request was made (used for rate limiting).
    queried: bool = True


@dataclass
class FillPlan(NormalizationPlan):
    """Dry run of Fill from MusicBrainz."""

    skipped_multi: list[str] = field(default_factory=list)   # already 2+ genres
    no_match: list[tuple[str, str]] = field(default_factory=list)  # (path, why)
    errors: list[tuple[str, str]] = field(default_factory=list)    # (path, error)
    cancelled: bool = False

    @property
    def unchanged(self) -> int:  # files looked up whose genres would not change
        return max(
            0,
            self.scanned - len(self.changes) - len(self.blocked) - len(self.skipped_multi)
            - len(self.no_match) - len(self.errors),
        )


def plan_genre_fill(
    folder: str,
    progress_callback: Optional[ProgressCallback] = None,
    *,
    files: Optional[Sequence[str]] = None,
    read_tags: Optional[Callable[[str], Dict[str, object]]] = None,
    lookup: Optional[Callable[[str], FillLookup]] = None,
    writable_check: Optional[Callable[[str], Optional[str]]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    rate_limit: Optional[float] = None,
    sleep: Optional[Callable[[float], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> FillPlan:
    """Dry run: look every file up and plan the genre additions. Never writes.

    Rules (stated in the UI too):

    * files that already have ``FILL_SKIP_AT`` (2) or more genres are skipped;
    * otherwise MusicBrainz's top community tags are **added after** the
      existing genre — nothing already on the file is removed or replaced
      (:func:`add_genres`);
    * a file whose genres would not change is not planned.

    ``rate_limit`` seconds are waited between network lookups (MusicBrainz
    allows about one request a second); the wait is cancellable.
    """
    import time as _time

    progress = progress_callback or (lambda _i, _t: None)
    log = log_callback or (lambda _m: None)
    paths = list(files) if files is not None else _default_discover(folder)
    reader = read_tags or _default_read_tags
    look = lookup or _default_lookup
    check = _resolve_writable_check(writable_check, read_tags)
    if rate_limit is None:
        try:
            from update_genres import RATE_LIMIT_SECONDS as rate_limit  # type: ignore
        except Exception:  # noqa: BLE001
            rate_limit = 1.0
    nap = sleep or _time.sleep

    def _cancelled() -> bool:
        return bool(should_cancel is not None and should_cancel())

    def _wait(seconds: float) -> None:
        # Sleep in short slices so Cancel is honoured within ~0.1 s.
        remaining = max(0.0, float(seconds))
        while remaining > 1e-9 and not _cancelled():
            chunk = min(0.1, remaining)
            nap(chunk)
            remaining -= chunk

    plan = FillPlan(scanned=len(paths))
    progress(0, plan.scanned)
    queried_before = False
    for idx, path in enumerate(paths, start=1):
        if _cancelled():
            plan.cancelled = True
            break
        name = os.path.basename(path)
        existing = _genre_list(reader(path).get("genre"))
        if existing:
            plan.with_genres += 1
        if len(existing) >= FILL_SKIP_AT:
            plan.skipped_multi.append(path)
            log(f"{name} → skipped (already has genres: {'; '.join(existing)})")
            progress(idx, plan.scanned)
            continue
        if queried_before and rate_limit:
            _wait(float(rate_limit))
            if _cancelled():
                plan.cancelled = True
                break
        try:
            found = look(path)
        except Exception as exc:  # noqa: BLE001 - one failed lookup must not end the run
            found = FillLookup(status="error", detail=str(exc))
        queried_before = queried_before or bool(found.queried)
        if found.status == "error":
            plan.errors.append((path, found.detail))
            log(f"{name} → lookup error: {found.detail}")
        elif not found.genres:
            plan.no_match.append((path, found.detail or found.status))
            log(f"{name} → {found.detail or found.status.replace('_', ' ')}")
        else:
            after = add_genres(existing, found.genres)
            if after != existing:
                reason = check(path) if check is not None else None
                if reason:
                    plan.blocked.append((path, reason))
                    log(f"{name} → can't write: {reason}")
                else:
                    plan.changes.append(GenreChange(path, existing, after, found.detail))
                    log(f"{name} → would add {', '.join(after[len(existing):])}")
        progress(idx, plan.scanned)
    return plan
