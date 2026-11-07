"""File Cleanup — strip OS duplicate-copy artifacts from audio filenames.

Operating-system file managers leave names such as ``Song (1).mp3``
(browsers / Windows), ``Song - Copy.mp3`` (Windows Explorer) or
``Song copy.mp3`` / ``Song copy 2.mp3`` (macOS Finder). This module renames
those back to ``Song.mp3``, **preview-first**:

1. :func:`build_cleanup_plan` walks the library (skipping the reserved
   top-level folders) and returns a :class:`CleanupPlan` — nothing on disk
   changes.
2. The caller shows the plan (old → new, collisions flagged) and asks for
   confirmation.
3. :func:`execute_cleanup_plan` performs exactly the renames in the plan,
   never overwriting anything, and writes a JSON-lines action log to
   ``<library>/Docs/`` *as it goes* so :func:`undo_cleanup` can reverse it.

The patterns are deliberately conservative. Only these trailing artifacts are
recognised, and only when what remains is a non-empty name:

* `` (N)`` with N from 1 to 99 — so ``Live at Wembley (1986)`` is left alone;
* `` - Copy`` / `` - Copy (N)`` (Windows Explorer);
* `` copy`` / `` copy N`` in lower case (macOS Finder) — so a title like
  ``Carbon Copy`` is left alone, and ``Xerocopy`` never matches.

When a title reader is supplied (the GUI passes one that reads tags), a file
whose own title tag already ends with the "artifact" is skipped: the suffix
is part of the real title.

This module is Qt-free so it can be tested and reused without the GUI.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable

from utils.reserved_folders import DOCS, walk_library

AUDIO_EXTS = frozenset({".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"})

# Each pattern must match the *whole* stem; ``base`` is what is kept.
_ARTIFACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Windows Explorer: "Song - Copy", "Song - Copy (2)"
    ("windows copy", re.compile(r"^(?P<base>.*\S) - Copy(?: \((?:[1-9]\d?)\))?$", re.IGNORECASE)),
    # macOS Finder: "Song copy", "Song copy 2" (Finder always writes lower case)
    ("finder copy", re.compile(r"^(?P<base>.*\S) copy(?: (?:[1-9]\d?))?$")),
    # Browser / Explorer counter: "Song (1)" … "Song (99)"
    ("counter", re.compile(r"^(?P<base>.*\S) \((?:[1-9]\d?)\)$")),
)

STATUS_RENAME = "rename"
STATUS_COLLISION = "collision"

LOG_PREFIX = "file_cleanup_"


def clean_stem(stem: str) -> tuple[str, str] | None:
    """Return ``(new_stem, rule)`` if ``stem`` ends in an OS copy artifact.

    Returns ``None`` when nothing should change. Only one artifact is removed,
    and the remaining name must still contain a letter (so "01 - Copy",
    track 01 titled "Copy", is left alone).
    """
    for rule, pattern in _ARTIFACT_PATTERNS:
        match = pattern.match(stem)
        if match:
            base = match.group("base").strip()
            # Keep at least one letter: "01 - Copy" is track 01 titled "Copy",
            # not a copy of a file called "01".
            if any(ch.isalpha() for ch in base):
                return base, rule
    return None


@dataclass
class RenameItem:
    src: str
    dst: str
    rule: str
    status: str = STATUS_RENAME
    note: str = ""

    @property
    def is_rename(self) -> bool:
        return self.status == STATUS_RENAME


@dataclass
class CleanupPlan:
    library_root: str
    items: list[RenameItem] = field(default_factory=list)
    scanned: int = 0
    kept_by_title: int = 0
    cancelled: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def renames(self) -> list[RenameItem]:
        return [i for i in self.items if i.is_rename]

    @property
    def collisions(self) -> list[RenameItem]:
        return [i for i in self.items if i.status == STATUS_COLLISION]


@dataclass
class CleanupResult:
    renamed: dict[str, str] = field(default_factory=dict)   # old -> new
    failed: list[tuple[str, str]] = field(default_factory=list)  # (old, reason)
    skipped_collisions: int = 0
    cancelled: bool = False
    log_path: str | None = None


def _title_has_suffix(title: object, stem: str, new_stem: str) -> bool:
    """True when the tag title itself ends with the text we would strip."""
    if not title:
        return False
    if isinstance(title, (list, tuple)):
        title = title[0] if title else ""
    suffix = stem[len(new_stem):].strip().lstrip("-").strip().casefold()
    return bool(suffix) and str(title).strip().casefold().endswith(suffix)


def build_cleanup_plan(
    library_root: str,
    *,
    read_title: Callable[[str], object] | None = None,
    cancel_event: threading.Event | None = None,
    progress: Callable[[int], None] | None = None,
    extensions: Iterable[str] = AUDIO_EXTS,
) -> CleanupPlan:
    """Compute (but do not perform) the cleanup renames for ``library_root``."""
    exts = {e.lower() for e in extensions}
    plan = CleanupPlan(library_root=os.path.abspath(library_root))
    proposals: list[RenameItem] = []

    for dirpath, _dirs, files in walk_library(plan.library_root):
        if cancel_event is not None and cancel_event.is_set():
            plan.cancelled = True
            break
        for filename in sorted(files):
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in exts:
                continue
            plan.scanned += 1
            if progress is not None and plan.scanned % 200 == 0:
                progress(plan.scanned)
            cleaned = clean_stem(stem)
            if cleaned is None:
                continue
            new_stem, rule = cleaned
            src = os.path.join(dirpath, filename)
            if read_title is not None:
                try:
                    title = read_title(src)
                except Exception:  # noqa: BLE001 - unreadable tags: decide on the name alone
                    title = None
                if _title_has_suffix(title, stem, new_stem):
                    plan.kept_by_title += 1
                    continue
            proposals.append(RenameItem(src, os.path.join(dirpath, new_stem + ext), rule))

    # Flag collisions: a target that already exists, or two files that would
    # end up with the same name. Never guess which one is "right".
    by_target: dict[str, list[RenameItem]] = {}
    for item in proposals:
        by_target.setdefault(os.path.normcase(item.dst), []).append(item)
    for item in proposals:
        if os.path.lexists(item.dst):
            item.status = STATUS_COLLISION
            item.note = "a file with the new name already exists"
        elif len(by_target[os.path.normcase(item.dst)]) > 1:
            item.status = STATUS_COLLISION
            item.note = "several files would get this name"
    plan.items = proposals
    return plan


def _new_log_path(library_root: str) -> str:
    docs = os.path.join(library_root, DOCS)
    os.makedirs(docs, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(docs, f"{LOG_PREFIX}{stamp}.jsonl")
    n = 1
    while os.path.exists(path):
        path = os.path.join(docs, f"{LOG_PREFIX}{stamp}_{n}.jsonl")
        n += 1
    return path


def _append(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def execute_cleanup_plan(
    plan: CleanupPlan,
    *,
    cancel_event: threading.Event | None = None,
    on_item: Callable[[RenameItem, str], None] | None = None,
) -> CleanupResult:
    """Perform exactly the ``rename`` items of ``plan``; never overwrite.

    Every action is appended to ``<library>/Docs/file_cleanup_<ts>.jsonl``
    as it happens, so the log is complete even if the run is interrupted.
    ``on_item(item, outcome)`` is called after each item (outcome is
    ``"renamed"``, ``"failed: …"`` or ``"skipped: …"``).
    """
    result = CleanupResult(skipped_collisions=len(plan.collisions))
    renames = plan.renames
    if not renames:
        return result
    result.log_path = _new_log_path(plan.library_root)
    with open(result.log_path, "w", encoding="utf-8") as log:
        _append(log, {
            "type": "header", "tool": "file_cleanup", "library_root": plan.library_root,
            "planned_at": plan.created_at,
            "executed_at": datetime.now().isoformat(timespec="seconds"),
            "planned_renames": len(renames), "collisions_skipped": len(plan.collisions),
        })
        for item in plan.collisions:
            _append(log, {"type": "skipped", "old": item.src, "new": item.dst, "reason": item.note})
        for item in renames:
            if cancel_event is not None and cancel_event.is_set():
                result.cancelled = True
                _append(log, {"type": "cancelled"})
                break
            if not os.path.exists(item.src):
                outcome = "skipped: source no longer exists"
                result.failed.append((item.src, outcome))
                _append(log, {"type": "skipped", "old": item.src, "new": item.dst, "reason": outcome})
            elif os.path.lexists(item.dst):
                outcome = "skipped: target appeared after the preview"
                result.failed.append((item.src, outcome))
                _append(log, {"type": "skipped", "old": item.src, "new": item.dst, "reason": outcome})
            else:
                try:
                    os.rename(item.src, item.dst)
                except OSError as exc:
                    outcome = f"failed: {exc}"
                    result.failed.append((item.src, outcome))
                    _append(log, {"type": "failed", "old": item.src, "new": item.dst, "reason": str(exc)})
                else:
                    outcome = "renamed"
                    result.renamed[item.src] = item.dst
                    _append(log, {"type": "rename", "old": item.src, "new": item.dst})
            if on_item is not None:
                on_item(item, outcome)
    return result


def undo_cleanup(log_path: str) -> tuple[int, list[tuple[str, str]]]:
    """Reverse the renames recorded in a cleanup log (newest first).

    Returns ``(restored_count, problems)``; a rename is only reversed when the
    new file still exists and the old name is free.
    """
    with open(log_path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    restored = 0
    problems: list[tuple[str, str]] = []
    for rec in reversed([r for r in records if r.get("type") == "rename"]):
        old, new = rec["old"], rec["new"]
        if not os.path.exists(new):
            problems.append((new, "renamed file no longer exists"))
        elif os.path.lexists(old):
            problems.append((old, "original name is taken"))
        else:
            try:
                os.rename(new, old)
                restored += 1
            except OSError as exc:
                problems.append((new, str(exc)))
    return restored, problems
