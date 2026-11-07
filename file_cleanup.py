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
   ``<library>/Docs/`` *as it goes* so it can be reversed: the GUI's "Undo
   Last Cleanup" finds the newest log with :func:`latest_undoable_log`,
   previews :func:`plan_undo` and runs exactly that with
   :func:`execute_undo` (which records each restore in the same log).

Renames never overwrite: :func:`rename_no_overwrite` uses ``os.link`` +
``os.unlink`` on POSIX so a file that appears after the check is never
replaced, and collisions are detected case-insensitively everywhere
(:func:`collision_key`), since macOS and exFAT/FAT volumes ignore case.

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

import errno
import json
import os
import re
import sys
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
    # Every name in each scanned folder, case-folded: a target that differs
    # from an existing file only in letter case is the same file on macOS,
    # Windows and exFAT/FAT cards (os.path.normcase only folds on Windows).
    folder_names: dict[str, set[str]] = {}

    for dirpath, _dirs, files in walk_library(plan.library_root):
        if cancel_event is not None and cancel_event.is_set():
            plan.cancelled = True
            break
        folder_names[dirpath] = {f.casefold() for f in files}
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
    # end up with the same name. Never guess which one is "right". Names are
    # compared case-insensitively on every platform, so a plan made on Linux
    # is just as safe when the library lives on a case-insensitive disk.
    by_target: dict[str, list[RenameItem]] = {}
    for item in proposals:
        by_target.setdefault(collision_key(item.dst), []).append(item)
    for item in proposals:
        dst_name = os.path.basename(item.dst).casefold()
        src_name = os.path.basename(item.src).casefold()
        siblings = folder_names.get(os.path.dirname(item.src), set())
        if os.path.lexists(item.dst) or (dst_name in siblings and dst_name != src_name):
            item.status = STATUS_COLLISION
            item.note = "a file with the new name already exists"
        elif len(by_target[collision_key(item.dst)]) > 1:
            item.status = STATUS_COLLISION
            item.note = "several files would get this name"
    plan.items = proposals
    return plan


def collision_key(path: str) -> str:
    """Key under which two paths name the same file on *some* filesystem.

    ``os.path.normcase`` folds case only on Windows, but macOS (APFS/HFS+)
    and exFAT/FAT drives are case-insensitive too, so the key is case-folded
    everywhere.
    """
    return os.path.normpath(os.path.abspath(path)).casefold()


class TargetExists(OSError):
    """The rename target appeared after the check: nothing was overwritten."""


def rename_no_overwrite(src: str, dst: str) -> None:
    """Rename ``src`` to ``dst`` without ever replacing an existing ``dst``.

    * Windows: ``os.rename`` already refuses an existing target.
    * POSIX: ``os.link`` + ``os.unlink`` — creating the link fails atomically
      if ``dst`` exists (including a name that differs only in case on a
      case-insensitive volume), so there is no window between the check and
      the rename. Filesystems without hard links (FAT/exFAT, some network
      shares) fall back to a re-check immediately before ``os.rename``.

    Raises :class:`TargetExists` when ``dst`` is taken, ``OSError`` otherwise.
    """
    if sys.platform == "win32":
        try:
            os.rename(src, dst)
        except FileExistsError as exc:
            raise TargetExists(errno.EEXIST, "target already exists", dst) from exc
        return
    if not os.path.islink(src):
        try:
            os.link(src, dst)
        except FileExistsError as exc:
            raise TargetExists(errno.EEXIST, "target already exists", dst) from exc
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP,
                                 errno.EMLINK, errno.EXDEV, errno.ENOSYS):
                raise
            # No hard links here: fall through to the re-checked rename.
        else:
            try:
                os.unlink(src)
            except OSError:
                # Undo the extra name so the library looks as it did.
                try:
                    os.unlink(dst)
                except OSError:
                    pass
                raise
            return
    if os.path.lexists(dst):
        raise TargetExists(errno.EEXIST, "target already exists", dst)
    os.rename(src, dst)


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
                    rename_no_overwrite(item.src, item.dst)
                except TargetExists:
                    outcome = "skipped: target appeared after the preview"
                    result.failed.append((item.src, outcome))
                    _append(log, {"type": "skipped", "old": item.src, "new": item.dst,
                                  "reason": outcome})
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


# ── Undo ──────────────────────────────────────────────────────────────────


@dataclass
class UndoPlan(CleanupPlan):
    """What undoing one cleanup log would do — same shape as a cleanup plan.

    Each item renames ``src`` (the cleaned name, as it is now) back to
    ``dst`` (the original name). Items whose file is gone, or whose original
    name is taken again, are ``collision`` rows that will be skipped.
    """

    log_path: str = ""


def _read_log(log_path: str) -> list[dict]:
    with open(log_path, encoding="utf-8") as handle:
        records = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue  # a line cut short by a crash
        return records


def _pending_undo(records: list[dict]) -> list[dict]:
    """Rename records of a log that have not been undone yet."""
    restored = {
        (r.get("old"), r.get("new"))
        for r in records
        if r.get("type") == "undo" and r.get("outcome") == "restored"
    }
    return [
        r for r in records
        if r.get("type") == "rename" and (r.get("old"), r.get("new")) not in restored
    ]


def list_cleanup_logs(library_root: str) -> list[str]:
    """Every ``Docs/file_cleanup_*.jsonl`` of the library, newest first."""
    docs = os.path.join(os.path.abspath(library_root), DOCS)
    try:
        names = [n for n in os.listdir(docs) if n.startswith(LOG_PREFIX) and n.endswith(".jsonl")]
    except OSError:
        return []
    paths = [os.path.join(docs, n) for n in names]

    def _key(p: str):
        try:
            return (os.path.getmtime(p), p)
        except OSError:
            return (0.0, p)

    return sorted(paths, key=_key, reverse=True)


def latest_undoable_log(library_root: str) -> str | None:
    """The newest cleanup log that still has renames to undo, or ``None``.

    Undoing it and asking again walks back to the cleanup before it.
    """
    for path in list_cleanup_logs(library_root):
        try:
            if _pending_undo(_read_log(path)):
                return path
        except OSError:
            continue
    return None


def plan_undo(log_path: str) -> UndoPlan:
    """Plan (but do not perform) undoing the renames recorded in ``log_path``.

    Newest rename first. Nothing on disk changes.
    """
    records = _read_log(log_path)
    header = next((r for r in records if r.get("type") == "header"), {})
    library_root = header.get("library_root") or os.path.dirname(
        os.path.dirname(os.path.abspath(log_path))
    )
    plan = UndoPlan(library_root=library_root, log_path=log_path)
    pending = list(reversed(_pending_undo(records)))
    plan.scanned = len(pending)
    by_target: dict[str, list[RenameItem]] = {}
    for rec in pending:
        item = RenameItem(src=rec["new"], dst=rec["old"], rule="undo")
        if not os.path.exists(item.src):
            item.status = STATUS_COLLISION
            item.note = "the renamed file is no longer there"
        elif os.path.lexists(item.dst) and collision_key(item.dst) != collision_key(item.src):
            item.status = STATUS_COLLISION
            item.note = "the original name is taken"
        by_target.setdefault(collision_key(item.dst), []).append(item)
        plan.items.append(item)
    for items in by_target.values():
        if len(items) > 1:
            for item in items:
                if item.is_rename:
                    item.status = STATUS_COLLISION
                    item.note = "several files would get this name"
    return plan


def execute_undo(
    plan: UndoPlan,
    *,
    cancel_event: threading.Event | None = None,
    on_item: Callable[[RenameItem, str], None] | None = None,
) -> CleanupResult:
    """Perform exactly the ``rename`` items of an :class:`UndoPlan`.

    Never overwrites. Each outcome is appended to the original cleanup log
    (``{"type": "undo", ...}``) as it happens, so a later undo only retries
    what was not restored. ``result.renamed`` maps *cleaned name -> original
    name*, ready to hand to ``playlist_generator.update_playlists``.
    """
    result = CleanupResult(skipped_collisions=len(plan.collisions), log_path=plan.log_path)
    renames = plan.renames
    if not renames:
        return result
    with open(plan.log_path, "a", encoding="utf-8") as log:
        _append(log, {
            "type": "undo_header",
            "executed_at": datetime.now().isoformat(timespec="seconds"),
            "planned_restores": len(renames), "collisions_skipped": len(plan.collisions),
        })
        for item in renames:
            if cancel_event is not None and cancel_event.is_set():
                result.cancelled = True
                _append(log, {"type": "undo_cancelled"})
                break
            if not os.path.exists(item.src):
                outcome = "skipped: the renamed file is no longer there"
            else:
                try:
                    rename_no_overwrite(item.src, item.dst)
                except TargetExists:
                    outcome = "skipped: the original name is taken"
                except OSError as exc:
                    outcome = f"failed: {exc}"
                else:
                    outcome = "renamed"
            if outcome == "renamed":
                result.renamed[item.src] = item.dst
                _append(log, {"type": "undo", "old": item.dst, "new": item.src,
                              "outcome": "restored"})
            else:
                result.failed.append((item.src, outcome))
                _append(log, {"type": "undo", "old": item.dst, "new": item.src,
                              "outcome": outcome})
            if on_item is not None:
                on_item(item, outcome)
    return result


def undo_cleanup(log_path: str) -> tuple[int, list[tuple[str, str]]]:
    """Reverse the renames recorded in a cleanup log (newest first).

    Returns ``(restored_count, problems)``; a rename is only reversed when the
    renamed file still exists and the original name is free. Playlists are not
    touched here — the GUI's "Undo last cleanup" also points playlists back.
    """
    plan = plan_undo(log_path)
    result = execute_undo(plan)
    problems = [(i.src if "no longer" in i.note else i.dst,
                 "renamed file no longer exists" if "no longer" in i.note
                 else "original name is taken")
                for i in plan.collisions]
    problems += [(path, reason) for path, reason in result.failed]
    return len(result.renamed), problems
