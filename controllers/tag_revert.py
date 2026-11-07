"""Undo a Tag Fixer or Genre Normalizer apply from its change log.

Every tag write is logged to ``Docs/tag_changes_<tool>_<timestamp>.jsonl``
(``utils.tag_change_log``) with each field's value before and after. This
module turns one of those logs back into a write, preview-first like every
other file-changing workflow:

1. :func:`list_undoable_logs` — the library's change logs that wrote
   something, newest first;
2. :func:`plan_revert` — per file and field, the value to restore. A file is
   only planned when every logged field still holds exactly what the log
   says was written; a file changed since (by anyone) is skipped and
   reported as a conflict, a file no longer there as missing, a file that
   already holds the old values as already reverted. Reads only;
3. :func:`apply_revert` — writes exactly the plan through
   ``utils.tag_writer`` (every value of a multi-value field; a field that
   did not exist before is removed again), re-checking each file just
   before writing, and logs the revert itself (tool ``revert``) so a revert
   can be reverted.

Field names are the ones the tools log: ``artist`` / ``title`` / ``album``
(Tag Fixer), ``genres`` (Tag Fixer) and ``genre`` (Genre Normalizer, Fill).
WAV/AIFF files with ID3 tags are read and written through their ID3 frames.

Qt-free and Tk-free. A GUI runs :func:`plan_revert` and :func:`apply_revert`
in a worker thread, holding ``controllers.tag_write_guard`` and the library
write lock while applying.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from utils.path_helpers import ensure_long_path

#: Tool name of the change logs a revert writes.
REVERT_TOOL = "revert"

_LOG_NAME = re.compile(
    r"^tag_changes_(?P<tool>.+)_(?P<stamp>\d{8}-\d{6})(?:_(?P<n>\d+))?\.jsonl$"
)

# Logged field name -> the field the tag writer / easy interface uses.
_WRITE_KEY = {"artist": "artist", "title": "title", "album": "album", "genre": "genre", "genres": "genre"}
_TEXT_FIELDS = ("artist", "title", "album")
_ID3_TEXT_FRAMES = {"artist": "TPE1", "title": "TIT2", "album": "TALB"}

ProgressCallback = Callable[[int, int], None]


# ── Listing ──────────────────────────────────────────────────────────────────


@dataclass
class ChangeLogInfo:
    """One change log under ``Docs/``."""

    path: str
    tool: str
    #: ``YYYYMMDD-HHMMSS`` from the file name (when the apply started)
    stamp: str
    written: int = 0
    failed: int = 0
    skipped: int = 0
    #: distinct files with at least one written entry
    files: int = 0
    #: for a revert log: the file name of the log it reverted
    reverts: Optional[str] = None
    #: paths of revert logs that reverted this log (newest first)
    reverted_by: List[str] = field(default_factory=list)
    #: lines that could not be parsed (e.g. a crash mid-write)
    bad_lines: int = 0

    @property
    def undoable(self) -> bool:
        return self.written > 0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def _read_entries(path: str) -> Tuple[List[dict], int]:
    """Entries of a change log, tolerating a torn last line. ``(entries, bad)``."""
    entries: List[dict] = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                bad += 1
                continue
            if isinstance(entry, dict) and isinstance(entry.get("changes"), dict) and entry.get("path"):
                entries.append(entry)
            else:
                bad += 1
    return entries, bad


def _sort_key(info: ChangeLogInfo) -> tuple:
    m = _LOG_NAME.match(info.name)
    n = int(m.group("n") or 1) if m else 0
    try:
        mtime = os.path.getmtime(info.path)
    except OSError:
        mtime = 0.0
    return (info.stamp, n, mtime)


def list_change_logs(library_root: str) -> List[ChangeLogInfo]:
    """Every tag change log in ``<library_root>/Docs``, newest first."""
    docs = os.path.join(library_root, "Docs")
    try:
        names = os.listdir(docs)
    except OSError:
        return []
    infos: List[ChangeLogInfo] = []
    for name in names:
        m = _LOG_NAME.match(name)
        if not m:
            continue
        path = os.path.join(docs, name)
        info = ChangeLogInfo(path=path, tool=m.group("tool"), stamp=m.group("stamp"))
        try:
            entries, info.bad_lines = _read_entries(path)
        except OSError:
            continue
        files = set()
        for entry in entries:
            status = entry.get("status")
            if status == "written":
                info.written += 1
                files.add(entry["path"])
            elif status == "failed":
                info.failed += 1
            else:
                info.skipped += 1
            if entry.get("reverts") and info.reverts is None:
                info.reverts = str(entry["reverts"])
        info.files = len(files)
        infos.append(info)
    infos.sort(key=_sort_key, reverse=True)
    by_name = {info.name: info for info in infos}
    for info in infos:  # newest first, so reverted_by ends up newest first
        if info.reverts and info.reverts in by_name and info.written:
            by_name[info.reverts].reverted_by.append(info.path)
    return infos


def list_undoable_logs(library_root: str) -> List[ChangeLogInfo]:
    """The change logs that wrote at least one file, newest first.

    Revert logs are included (a revert can be reverted); ``reverted_by``
    says whether a log was already undone.
    """
    return [info for info in list_change_logs(library_root) if info.undoable]


# ── Reading what a file holds now ────────────────────────────────────────────


def _values(value: object) -> List[str]:
    """A logged or read value as a list of strings: ``None``/``""`` -> ``[]``."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v).strip() != ""]
    text = str(value)
    return [text] if text.strip() else []


def _comparable(value: object) -> List[str]:
    return [v.strip() for v in _values(value)]


def _default_opener(path: str):
    from mutagen import File as MutagenFile  # type: ignore

    return MutagenFile(path, easy=True)


def read_current_values(
    path: str, fields: Iterable[str], *, opener: Optional[Callable[[str], object]] = None
) -> Dict[str, List[str]]:
    """Every value ``path`` holds now for each logged field name.

    Raises ``OSError`` / mutagen errors when the file cannot be read, and
    ``ValueError`` for an unsupported format.
    """
    fields = list(fields)
    out: Dict[str, List[str]] = {}
    text_fields = [f for f in fields if f in _TEXT_FIELDS]
    if text_fields:
        audio = (opener or _default_opener)(ensure_long_path(path))
        if audio is None:
            raise ValueError("unsupported audio format")
        tags = getattr(audio, "tags", None)
        for name in text_fields:
            values: List[str] = []
            if tags is not None:
                if hasattr(tags, "getall"):  # raw ID3 in WAV / AIFF
                    for frame in tags.getall(_ID3_TEXT_FRAMES[name]):
                        values.extend(_values(list(getattr(frame, "text", []) or [])))
                elif name in tags:
                    values = _values(tags[name])
            out[name] = values
    if any(f in ("genre", "genres") for f in fields):
        from utils.audio_metadata_reader import read_genres_checked

        genres, error = read_genres_checked(ensure_long_path(path))
        if error:
            raise ValueError(error)
        for name in ("genre", "genres"):
            if name in fields:
                out[name] = list(genres)
    unknown = [f for f in fields if f not in _WRITE_KEY]
    if unknown:
        raise ValueError(f"don't know how to read {', '.join(sorted(unknown))}")
    return out


# ── Planning ─────────────────────────────────────────────────────────────────


@dataclass
class FieldRevert:
    """One field of one file: what it holds now and what it gets back."""

    current: List[str]
    #: the values to write back; ``[]`` means the field did not exist and is removed
    restore: List[str]


@dataclass
class FileRevert:
    path: str
    fields: Dict[str, FieldRevert]


@dataclass
class RevertConflict:
    """A file whose tags changed since the logged write: not touched."""

    path: str
    #: field -> (what the log says was written, what the file holds now)
    fields: Dict[str, Tuple[List[str], List[str]]]


@dataclass
class RevertPlan:
    log_path: str
    library_root: str
    tool: str
    items: List[FileRevert] = field(default_factory=list)
    conflicts: List[RevertConflict] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    already_reverted: List[str] = field(default_factory=list)
    unreadable: List[Tuple[str, str]] = field(default_factory=list)
    #: logged entries that wrote nothing (failed / skipped at the time)
    not_written: int = 0
    bad_lines: int = 0
    cancelled: bool = False

    def counts(self) -> Dict[str, int]:
        return {
            "files": len(self.items),
            "fields": sum(len(item.fields) for item in self.items),
            "conflicts": len(self.conflicts),
            "missing": len(self.missing),
            "already_reverted": len(self.already_reverted),
            "unreadable": len(self.unreadable),
            "not_written": self.not_written,
        }


def _library_root_for(log_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(log_path)))


def _net_changes(entries: Sequence[dict]) -> Tuple[Dict[str, Dict[str, Tuple[object, object]]], int, List[str]]:
    """Per path, per field: (first logged before, last logged after).

    Only ``written`` entries count; a file written twice in one run is
    reverted to what it held before the first write.
    """
    order: List[str] = []
    net: Dict[str, Dict[str, Tuple[object, object]]] = {}
    not_written = 0
    for entry in entries:
        if entry.get("status") != "written":
            not_written += 1
            continue
        path = str(entry["path"])
        if path not in net:
            net[path] = {}
            order.append(path)
        for name, change in entry["changes"].items():
            if not isinstance(change, dict):
                continue
            before, after = change.get("before"), change.get("after")
            if name in net[path]:
                before = net[path][name][0]
            net[path][name] = (before, after)
    return net, not_written, order


def plan_revert(
    log_path: str,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    reader: Optional[Callable[[str, Iterable[str]], Dict[str, List[str]]]] = None,
) -> RevertPlan:
    """Work out what undoing ``log_path`` would write. Reads only.

    ``reader(path, fields)`` (default :func:`read_current_values`) returns
    every value each field holds now.
    """
    read = reader or read_current_values
    entries, bad = _read_entries(log_path)
    m = _LOG_NAME.match(os.path.basename(log_path))
    tool = m.group("tool") if m else (entries[0].get("tool", "") if entries else "")
    plan = RevertPlan(log_path=log_path, library_root=_library_root_for(log_path), tool=str(tool), bad_lines=bad)
    net, plan.not_written, order = _net_changes(entries)
    progress = progress_callback or (lambda _i, _t: None)
    total = len(order)
    progress(0, total)
    for idx, path in enumerate(order, start=1):
        if cancel_event is not None and cancel_event.is_set():
            plan.cancelled = True
            break
        changes = {name: c for name, c in net[path].items() if name in _WRITE_KEY}
        if not changes:
            progress(idx, total)
            continue
        if not os.path.exists(ensure_long_path(path)):
            plan.missing.append(path)
            progress(idx, total)
            continue
        try:
            current = read(path, list(changes))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the plan
            plan.unreadable.append((path, str(exc) or exc.__class__.__name__))
            progress(idx, total)
            continue
        conflicts: Dict[str, Tuple[List[str], List[str]]] = {}
        fields: Dict[str, FieldRevert] = {}
        reverted = True
        for name, (before, after) in changes.items():
            now = list(current.get(name, []))
            if _comparable(now) == _comparable(before):
                fields[name] = FieldRevert(current=now, restore=_values(before))
                continue
            reverted = False
            if _comparable(now) != _comparable(after):
                conflicts[name] = (_values(after), now)
            fields[name] = FieldRevert(current=now, restore=_values(before))
        if conflicts:
            plan.conflicts.append(RevertConflict(path=path, fields=conflicts))
        elif reverted:
            plan.already_reverted.append(path)
        else:
            plan.items.append(FileRevert(
                path=path,
                fields={n: f for n, f in fields.items() if _comparable(f.current) != _comparable(f.restore)},
            ))
        progress(idx, total)
    return plan


# ── Applying ─────────────────────────────────────────────────────────────────


@dataclass
class RevertResult:
    reverted: List[str] = field(default_factory=list)
    #: (path, why) — changed again since the revert preview, or vanished
    skipped: List[Tuple[str, str]] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)
    cancelled: bool = False
    log_path: str = ""


def _default_writer(path: str, values: Dict[str, Optional[List[str]]]) -> None:
    from utils.tag_writer import write_fields

    write_fields(path, values)


def apply_revert(
    plan: RevertPlan,
    *,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_event: Optional[threading.Event] = None,
    reader: Optional[Callable[[str, Iterable[str]], Dict[str, List[str]]]] = None,
    writer: Optional[Callable[[str, Dict[str, Optional[List[str]]]], None]] = None,
    log_dir_root: Optional[str] = None,
) -> RevertResult:
    """Write exactly ``plan.items`` and log it as a ``revert`` change log.

    Each file is re-read just before writing: if any planned field no longer
    holds the value the preview showed, the file is skipped, never written
    blind. A cancellation stops before the next file; files already reverted
    stay reverted and are logged. ``log_dir_root`` overrides the library
    whose ``Docs/`` receives the revert log (default: the log's library).
    """
    from utils.tag_change_log import TagChangeLog

    read = reader or read_current_values
    write = writer or _default_writer
    progress = progress_callback or (lambda _i, _t: None)
    result = RevertResult()
    total = len(plan.items)
    progress(0, total)
    if not total:
        return result  # nothing to write: no (empty) revert log either
    with TagChangeLog(log_dir_root or plan.library_root, REVERT_TOOL, reverts=plan.log_path) as log:
        result.log_path = log.path
        for idx, item in enumerate(plan.items, start=1):
            if cancel_event is not None and cancel_event.is_set():
                result.cancelled = True
                break
            record = {name: (f.current, f.restore) for name, f in item.fields.items()}
            if not item.fields:
                progress(idx, total)
                continue
            why = None
            if not os.path.exists(ensure_long_path(item.path)):
                why = "file not found"
            else:
                try:
                    now = read(item.path, list(item.fields))
                except Exception as exc:  # noqa: BLE001
                    now = None
                    why = f"could not read the file ({exc})"
                if now is not None:
                    changed = [
                        name for name, f in item.fields.items()
                        if _comparable(now.get(name, [])) != _comparable(f.current)
                    ]
                    if changed:
                        why = f"{', '.join(changed)} changed since the preview"
            if why:
                result.skipped.append((item.path, why))
                log.record(item.path, record, "skipped", why)
                progress(idx, total)
                continue
            values: Dict[str, Optional[List[str]]] = {}
            for name, f in item.fields.items():
                values[_WRITE_KEY[name]] = list(f.restore) if f.restore else None
            try:
                write(item.path, values)
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
                message = str(exc) or exc.__class__.__name__
                result.failed.append((item.path, message))
                log.record(item.path, record, "failed", message)
                progress(idx, total)
                continue
            result.reverted.append(item.path)
            log.record(item.path, record, "written")
            progress(idx, total)
    return result
