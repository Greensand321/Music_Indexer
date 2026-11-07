import os
import sys
import argparse
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterable, Callable, Dict, List, Optional, Sequence, Tuple, Union
import importlib
from mutagen import File as MutagenFile
from utils.audio_metadata_reader import read_tags, read_tags_checked, read_genres_checked
from utils.path_helpers import ensure_long_path
from utils.reserved_folders import MANUAL_REVIEW, RESERVED_FOLDERS, walk_library
from utils.tag_writer import SUPPORTED_TAG_EXTS, assign_fields
# Additive genre merge shared with Fill from MusicBrainz (Qt-free, stdlib-only).
from controllers.normalize_controller import add_genres

from plugins.base import MetadataPlugin, SourceStatus

import sqlite3
from crash_watcher import record_event
from crash_logger import watcher

# ─── Database Helpers ─────────────────────────────────────────────────────
def init_db(path: str):
    """Create the SQLite database schema if needed."""
    db_folder = os.path.dirname(path)
    os.makedirs(db_folder, exist_ok=True)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        """
      CREATE TABLE IF NOT EXISTS files (
        path        TEXT PRIMARY KEY,
        status      TEXT,
        score       REAL,
        old_artist  TEXT, new_artist TEXT,
        old_title   TEXT, new_title  TEXT,
        old_album   TEXT, new_album  TEXT,
        old_genres  TEXT, new_genres TEXT
      );
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_status ON files(status);")
    c.execute(
        """
      CREATE TABLE IF NOT EXISTS fingerprints (
        path TEXT PRIMARY KEY,
        duration INT,
        fingerprint TEXT
      );
        """
    )
    conn.commit()
    conn.close()

# ─── Configuration ────────────────────────────────────────────────────────
ACOUSTID_API_KEY       = "eBOqCZhyAx"
ACOUSTID_APP_NAME      = "SoundVaultTagFixer"
ACOUSTID_APP_VERSION   = "1.0.0"
SUPPORTED_EXTS         = set(SUPPORTED_TAG_EXTS)  # shared with the Genre Normalizer

#: Top-level library folders the tag tools never scan or rewrite: the user's
#: "Not Sorted" exclusion zone, quarantined duplicate losers, Trash, Docs and
#: Playlists. "Manual Review" stays in scope on purpose — it holds the tracks
#: the indexer parked for missing metadata, which is exactly what this tool
#: repairs (and nothing is written without a preview and a confirmation).
TAG_SCAN_SKIPPED_FOLDERS = tuple(f for f in RESERVED_FOLDERS if f != MANUAL_REVIEW)

# ─── Plugin Discovery ─────────────────────────────────────────────────────
# Explicit module list rather than pkgutil.iter_modules(['plugins']): a
# filesystem scan relative to the working directory can't find the plugins
# package once it's bundled into a frozen executable.
_PLUGIN_MODULE_NAMES = ["acoustid_plugin", "api_service", "lastfm", "musicbrainz_plugin", "test_plugin"]

PLUGINS: List[MetadataPlugin] = []
for name in _PLUGIN_MODULE_NAMES:
    try:
        module = importlib.import_module(f'plugins.{name}')
    except Exception:
        continue
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and issubclass(obj, MetadataPlugin) and obj is not MetadataPlugin:
            PLUGINS.append(obj())

# ─── Data Classes ────────────────────────────────────────────────────────

#: A text tag's current value: ``None`` (absent), one ``str``, or the full
#: ``list`` of values when the file holds several.
FieldValue = Union[str, List[str]]

@dataclass
class FileRecord:
    """Representation of a single audio file and proposed tags."""

    path: Path
    status: str
    score: Optional[float] = None
    #: What the file holds now. A single value is a ``str``; a field with
    #: several values (``["Artist A", "Artist B"]``) is the full ``list`` so
    #: the preview, the change log and the "changed since preview" check see
    #: every value, not just the first.
    old_artist: Optional[FieldValue] = None
    new_artist: Optional[str] = None
    old_title: Optional[FieldValue] = None
    new_title: Optional[str] = None
    old_album: Optional[FieldValue] = None
    new_album: Optional[str] = None
    old_genres: List[str] = field(default_factory=list)
    new_genres: List[str] = field(default_factory=list)
    #: Why an ``unmatched`` record has no proposal ("AcoustID: network error").
    note: str = ""

#: Fields the Tag Fixer can propose and write, in display order.
TAG_FIELDS = ("artist", "title", "album", "genres")

# Score thresholds
MIN_AUTOMATIC_SCORE   = 0.90   # ≥90% → apply automatically
MIN_INTERACTIVE_SCORE = 0.75   # ≥75% & <90% → prompt user

# ─── Utility Functions ────────────────────────────────────────────────────
def is_remix(audio_path):
    """Return True if filename or existing title suggests a remix."""
    if "remix" in os.path.basename(audio_path).lower():
        return True
    tags = read_tags(ensure_long_path(audio_path))
    title = tags.get("title")
    if isinstance(title, str) and "remix" in title.lower():
        return True
    return False

def find_files(root):
    """Return a list of audio files under `root` (file or directory).

    The reserved top-level folders in ``TAG_SCAN_SKIPPED_FOLDERS`` are never
    entered, so quarantined duplicates and the user's "Not Sorted" zone are
    not proposed or rewritten.
    """
    if os.path.isfile(root):
        return [root]
    audio_files = []
    for dirpath, _, files in walk_library(root, TAG_SCAN_SKIPPED_FOLDERS):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
                audio_files.append(os.path.join(dirpath, fname))
    return audio_files


#: Text fields the Tag Fixer reads in full and can rewrite.
_TEXT_FIELDS = ("artist", "title", "album")
# ID3 frames for containers mutagen doesn't wrap in EasyID3 (WAV, AIFF).
_ID3_TEXT_FRAMES = {"artist": "TPE1", "title": "TIT2", "album": "TALB"}


def _value_list(value: object) -> List[str]:
    """``None`` / ``""`` -> ``[]``, ``"x"`` -> ``["x"]``, a list -> its strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None and str(v) != ""]
    text = str(value)
    return [text] if text else []


def _shape(values: Sequence[str]) -> Optional[FieldValue]:
    """The ``FileRecord.old_*`` form of ``values``: None, one str, or the list."""
    values = list(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def _read_text_field_lists(path: str) -> Optional[Dict[str, List[str]]]:
    """Every value of artist / title / album on disk, read with mutagen.

    ``utils.audio_metadata_reader.read_tags`` returns only the *first* value
    of each field, and the Tag Fixer writes a whole field: a second artist
    the preview never showed would be deleted. Returns ``None`` when mutagen
    can't open the file (the caller then falls back to ``read_tags``).
    """
    try:
        audio = MutagenFile(ensure_long_path(path), easy=True)
    except Exception:  # noqa: BLE001
        return None
    if audio is None:
        return None
    out: Dict[str, List[str]] = {name: [] for name in _TEXT_FIELDS}
    tags = getattr(audio, "tags", None)
    if tags is None:
        return out
    try:
        if hasattr(tags, "getall"):  # raw ID3 in WAV / AIFF
            for name, frame_id in _ID3_TEXT_FRAMES.items():
                for frame in tags.getall(frame_id):
                    out[name].extend(_value_list(list(getattr(frame, "text", []) or [])))
        else:
            for name in _TEXT_FIELDS:
                try:
                    raw = tags[name] if name in tags else None
                except Exception:  # noqa: BLE001 - e.g. a key the easy map lacks
                    raw = None
                out[name] = _value_list(raw)
    except Exception:  # noqa: BLE001
        return None
    return out


def _text_values(path: str, first_values: Dict[str, object]) -> Dict[str, Optional[FieldValue]]:
    """``{field: value}`` for artist/title/album with every value kept.

    ``first_values`` is ``read_tags``' result, used for a field the full
    reader found nothing for (or when mutagen couldn't open the file).
    """
    lists = _read_text_field_lists(path)
    out: Dict[str, Optional[FieldValue]] = {}
    for name in _TEXT_FIELDS:
        full = lists.get(name) if lists is not None else None
        out[name] = _shape(full) if full else _shape(_value_list(first_values.get(name)))
    return out


def _same_value(a: object, b: object) -> bool:
    return _value_list(a) == _value_list(b)


def _db_text(value: object) -> Optional[str]:
    values = _value_list(value)
    return "; ".join(values) if values else None


def planned_changes(
    rec: "FileRecord", fields: Optional[Iterable[str]] = None
) -> Dict[str, Tuple[object, object]]:
    """``{field: (before, after)}`` — exactly what applying ``rec`` would write.

    The single source of truth for both the preview rows and the writer, so
    the preview can never show something apply would not do (or vice versa).
    A field is planned only when the lookup returned a non-empty value that
    differs from the file's; genres are planned only when the lookup adds at
    least one genre (see :func:`add_genres`). Never blanks a field.

    Artist / title / album proposals **replace every value** of the field
    with the one proposed value. ``before`` is the full current value — a
    list when the file holds several (``["Artist A", "Artist B"]``), which
    the preview shows as "Artist A; Artist B → New Artist" — so the preview
    shows exactly what the write removes. A proposal equal to the current
    values joined with "; " is treated as no change.
    """
    wanted = set(TAG_FIELDS if fields is None else fields)
    out: Dict[str, Tuple[object, object]] = {}
    for name, old, new in (
        ("artist", rec.old_artist, rec.new_artist),
        ("title", rec.old_title, rec.new_title),
        ("album", rec.old_album, rec.new_album),
    ):
        if name not in wanted or new is None:
            continue
        new_text = str(new).strip()
        if not new_text:
            continue
        old_values = _value_list(old)
        if old_values == [new_text] or (len(old_values) > 1 and "; ".join(old_values) == new_text):
            continue
        out[name] = (old, new_text)
    if "genres" in wanted and rec.new_genres:
        before = list(rec.old_genres or [])
        after = add_genres(before, rec.new_genres)
        if after != before:
            out["genres"] = (before, after)
    return out


@dataclass
class ScanDiagnostics:
    """What a scan could use, and why files went unmatched."""

    sources: List[SourceStatus] = field(default_factory=list)
    errors: Counter = field(default_factory=Counter)
    scanned: int = 0
    matched: int = 0
    unmatched: int = 0
    cancelled: bool = False

    @property
    def active(self) -> List[SourceStatus]:
        return [s for s in self.sources if s.active]

    @property
    def problems(self) -> List[str]:
        return [f"{s.name}: {s.problem}" for s in self.sources if s.problem]

    def sources_line(self) -> str:
        named = list(self.sources)
        if not named:
            return "Lookup sources: none"
        parts = []
        for s in named:
            if s.active:
                parts.append(f"{s.name} ({s.detail})" if s.detail else s.name)
        off = [s.name for s in named if not s.active and s.problem]
        line = "Lookup sources: " + (", ".join(parts) if parts else "none available")
        if off:
            line += " · unavailable: " + ", ".join(off)
        return line

    def unmatched_reason(self) -> str:
        """Plain-words answer to "why did nothing match?"."""
        if not any(s.active for s in self.sources):
            problems = self.problems
            if problems:
                return "No lookup source could run — " + "; ".join(problems)
            return "No lookup source is configured."
        if self.errors:
            top = ", ".join(f"{reason} ({n})" for reason, n in self.errors.most_common(3))
            return f"Lookups failed: {top}"
        return "The online services returned no confident match."


def _current_values(path: str) -> Tuple[Optional[Dict[str, object]], str]:
    """Re-read what ``path`` holds now: ``(values, error)``."""
    tags, error = read_tags_checked(ensure_long_path(path))
    if error and error.startswith("read failed"):
        return None, error
    genres, gerror = read_genres_checked(ensure_long_path(path))
    if gerror and gerror.startswith("read failed"):
        return None, gerror
    values: Dict[str, object] = dict(_text_values(path, tags))
    values["genres"] = genres
    return values, ""


def _write_record(
    path: str,
    proposal: FileRecord,
    fields: Optional[Iterable[str]],
    log_callback: Callable[[str], None],
    change_log=None,
) -> Tuple[str, Dict[str, Tuple[object, object]], str]:
    """Write ``proposal``'s planned changes for ``fields``.

    Returns ``(status, changes, message)`` with ``status`` one of
    ``"written"``, ``"failed"``, ``"skipped"`` (stale: the file changed since
    the preview) or ``"nothing"`` (no planned change for those fields).
    Never raises for a per-file problem: one bad file must not abort a batch.
    """
    changes = planned_changes(proposal, fields)
    if not changes:
        return "nothing", changes, ""

    def _finish(status: str, message: str) -> Tuple[str, Dict[str, Tuple[object, object]], str]:
        if change_log is not None:
            change_log.record(path, changes, status, None if status == "written" else message)
        if message:
            log_callback(message)
        return status, changes, message

    try:
        # What executes must be what was previewed: if the file no longer
        # holds the "before" values the preview showed (another tool, or the
        # user, changed it since), skip it rather than overwrite blind.
        current, error = _current_values(path)
        if current is None:
            return _finish("failed", f"Failed to read {path}: {error}")
        # Full value lists on both sides: a second artist added since the
        # preview makes the file stale too.
        stale = [
            f for f, (before, _after) in changes.items()
            if not _same_value(current.get(f), before)
        ]
        if stale:
            return _finish(
                "skipped",
                f"Skipped {path}: {', '.join(stale)} changed on disk since the preview — scan again",
            )
        try:
            audio = MutagenFile(ensure_long_path(path), easy=True)
        except Exception as e:  # noqa: BLE001
            return _finish("failed", f"Failed to read {path}: {e}")
        if audio is None:
            return _finish("failed", f"Failed to save {path}: unsupported audio format")
        values = {
            ("genre" if f == "genres" else f): (list(after) if f == "genres" else [after])
            for f, (_before, after) in changes.items()
        }
        assign_fields(audio, values)
        audio.save()
    except Exception as e:  # noqa: BLE001 - e.g. ID3-in-WAV quirks, read-only files
        record_event(f"tag_fixer: failed to save {path}")
        return _finish("failed", f"Failed to save {path}: {e}")
    record_event(f"tag_fixer: updated tags for {path}")
    return _finish("written", f"Updated tags for {path}")


@watcher.traced
def update_tags(path: str, proposal: FileRecord, fields: List[str], log_callback, change_log=None):
    """Write selected tags from ``proposal`` into ``path``. Return True if saved.

    Writes exactly :func:`planned_changes` for ``fields``: genres as a real
    multi-value list (existing genres kept, new ones appended), never a
    joined ``"a; b"`` string, and never a genre write when the lookup found
    no genres. Per-file errors are logged and reported as ``False``.
    """
    record_event(f"tag_fixer: updating tags for {path}")
    status, _changes, _msg = _write_record(path, proposal, fields, log_callback, change_log)
    return status == "written"


@dataclass
class TagApplyResult:
    """Outcome of an apply run, per file."""

    applied: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    cancelled: bool = False
    #: path -> the fields actually written there
    written_fields: Dict[str, List[str]] = field(default_factory=dict)
    log_path: str = ""


def apply_tag_changes(
    items: Iterable[Tuple[FileRecord, Iterable[str]]],
    *,
    log_callback: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    change_log=None,
) -> TagApplyResult:
    """Write each ``(record, fields)`` pair — exactly those fields, nothing else.

    This is what the review grid's per-row ticks map to: a file can have its
    artist written and its genre left alone. Cancellation stops before the
    next file; files already written stay written and are reported.
    """
    items = [(rec, list(fields)) for rec, fields in items]
    if log_callback is None:
        def log_callback(msg: str):
            print(msg)
    result = TagApplyResult()
    total = len(items)
    for idx, (rec, fields) in enumerate(items, start=1):
        if should_cancel is not None and should_cancel():
            result.cancelled = True
            break
        status, changes, _msg = _write_record(str(rec.path), rec, fields, log_callback, change_log)
        if status == "written":
            result.applied.append(str(rec.path))
            result.written_fields[str(rec.path)] = list(changes)
        elif status == "failed":
            result.failed.append(str(rec.path))
        elif status == "skipped":
            result.skipped.append(str(rec.path))
        if progress_callback is not None:
            progress_callback(idx, total)
    record_event(
        f"tag_fixer: applied {len(result.applied)} proposals "
        f"({len(result.failed)} failed, {len(result.skipped)} skipped)"
    )
    return result

def prompt_user_about_tags(f, old_artist, old_title, new_tags):
    """Show side-by-side old vs new and ask Y/N."""
    new_artist = new_tags["artist"] or "—"
    new_title  = new_tags["title"] or "—"
    # Several values show as "A; B" (FileRecord keeps multi-value fields as lists).
    old_artist = _db_text(old_artist)
    old_title = _db_text(old_title)
    print(f"\nFile: {f}")
    print(f"{'Field':10} │ {'Current':30} │ {'New from AcoustID':30}")
    print("-" * 75)
    print(f"{'Artist':10} │ {old_artist or '—':30} │ {new_artist:30}")
    print(f"{'Title':10} │ {old_title  or '—':30} │ {new_title:30}")
    print()
    resp = input("Apply these changes? [y/N]: ").strip().lower()
    return (resp == "y")

# ─── Main Tag-Fixing Logic ────────────────────────────────────────────────
@watcher.traced
def build_file_records(
    root: str,
    *,
    db_conn: sqlite3.Connection,
    show_all: bool = False,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    progress_detail: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    diagnostics: ScanDiagnostics | None = None,
) -> List[FileRecord]:
    """Return a list of ``FileRecord`` objects for ``root``.

    ``progress_callback(idx)`` receives the 1-based file index (legacy
    contract); ``progress_detail(idx, total)`` receives both, for a real
    percentage. ``should_cancel()`` is polled before every file — a scan of
    thousands of files making network calls must stop when asked, not run
    to the end and then discard the result. ``diagnostics`` (optional) is
    filled with which lookup sources ran and why files went unmatched.
    """
    record_event(f"tag_fixer: building records for {root}")

    if log_callback is None:
        def log_callback(msg: str):
            print(msg)
    diag = diagnostics if diagnostics is not None else ScanDiagnostics()

    records: List[FileRecord] = []

    # Pre-flight every source once, so a missing fpcalc, a missing key or no
    # network is reported up front (and those lookups are actually skipped)
    # instead of every file silently coming back "unmatched".
    active_plugins: List[MetadataPlugin] = []
    for plugin in PLUGINS:
        name = getattr(plugin, "name", None)
        try:
            if hasattr(plugin, "preflight"):
                source = plugin.preflight()
            else:
                source = SourceStatus(name or type(plugin).__name__, True)
        except Exception as exc:  # noqa: BLE001
            source = SourceStatus(name or type(plugin).__name__, False, "", f"check failed: {exc}")
        if name:
            diag.sources.append(source)
        if source.active:
            active_plugins.append(plugin)
        if source.problem:
            log_callback(f"\u26a0 {source.name}: {source.problem}")
    if diag.sources:
        log_callback(diag.sources_line())

    existing_status = dict(db_conn.execute("SELECT path, status FROM files"))

    files = find_files(root)
    total = len(files)

    def _progress(i: int) -> None:
        if progress_callback:
            progress_callback(i)
        if progress_detail:
            progress_detail(i, total)

    for idx, f in enumerate(files, start=1):
        if should_cancel is not None and should_cancel():
            diag.cancelled = True
            log_callback("Scan cancelled.")
            break
        status = existing_status.get(f, "new")

        if status == "applied" and not show_all:
            _progress(idx)
            continue

        if is_remix(f) and not show_all:
            _progress(idx)
            continue

        log_callback(f"Processing {f}")

        tags, read_error = read_tags_checked(ensure_long_path(f))
        if read_error and read_error.startswith("read failed"):
            # Corrupt/unreadable file: skip it rather than proposing tags for
            # (and running metadata lookups on) something we can't write to.
            log_callback(f"Failed to read {f}: {read_error}")
            _progress(idx)
            continue
        # Every value of each text field, not just the first: a proposal
        # replaces the whole field, so the preview must show all of it.
        text_values = _text_values(f, tags)
        old_artist = text_values["artist"]
        old_title = text_values["title"]
        old_album = text_values["album"]
        # Every genre value, not just the first: the genre write appends to
        # this list, so a value missing here would be deleted on apply.
        old_genres, _genre_error = read_genres_checked(ensure_long_path(f))
        diag.scanned += 1

        rec = FileRecord(
            path=Path(f),
            status=status,
            score=None,
            old_artist=old_artist,
            new_artist=None,
            old_title=old_title,
            new_title=None,
            old_album=old_album,
            new_album=None,
            old_genres=list(old_genres),
            new_genres=[],
        )

        best_result = None
        best_score = 0.0
        merged_genres: List[str] = []
        failures: List[str] = []
        for plugin in active_plugins:
            label = getattr(plugin, "name", None) or type(plugin).__name__
            try:
                result = plugin.identify(f)
            except Exception as exc:  # noqa: BLE001 - one source failing must not end the scan
                reason = str(exc).strip() or type(exc).__name__
                diag.errors[f"{label}: {reason}"] += 1
                failures.append(f"{label}: {reason}")
                continue
            if not result:
                continue
            enrichment_error = result.get("enrichment_error")
            if enrichment_error:
                diag.errors[f"{label}: {enrichment_error}"] += 1
            try:
                score = float(result.get('score') or 0)
            except (TypeError, ValueError):
                score = 0.0
            # Genres only from matches confident enough to propose at all —
            # the same threshold that gates artist/title/album below.
            if result.get('genres') and score >= MIN_INTERACTIVE_SCORE:
                merged_genres = add_genres(merged_genres, list(result['genres']))
            if score > best_score:
                best_score = score
                best_result = result

        if best_result and best_score >= MIN_INTERACTIVE_SCORE:
            rec.new_artist = best_result.get('artist', rec.new_artist)
            rec.new_title  = best_result.get('title', rec.new_title)
            rec.new_album  = best_result.get('album', rec.new_album)
            if merged_genres:
                rec.new_genres = list(merged_genres)
            rec.score      = best_score

        _progress(idx)

        if rec.new_artist is None and rec.new_title is None and rec.new_album is None and not rec.new_genres:
            rec.status = "unmatched"
            diag.unmatched += 1
            if failures:
                rec.note = "; ".join(failures)
            elif not active_plugins or not any(s.active for s in diag.sources):
                rec.note = "no lookup source available"
            elif best_result is not None:
                rec.note = f"low-confidence match ({best_score:.2f})"
            else:
                rec.note = "no match found"
        else:
            diag.matched += 1
            if not planned_changes(rec):
                rec.status = "no_diff"
        records.append(rec)
        genres_old = ";".join(rec.old_genres)
        genres_new = ";".join(rec.new_genres)
        vals = (
            str(rec.path), rec.status, rec.score,
            _db_text(rec.old_artist), rec.new_artist,
            _db_text(rec.old_title), rec.new_title,
            _db_text(rec.old_album), rec.new_album,
            genres_old, genres_new,
        )
        db_conn.execute(
            "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            vals,
        )

    db_conn.commit()
    record_event(
        f"tag_fixer: built {len(records)} records for {root}"
    )
    return records


@watcher.traced
def apply_tag_proposals(
    selected: Iterable[FileRecord],
    *,
    fields: List[str] | None = None,
    log_callback: Callable[[str], None] | None = None,
    result: TagApplyResult | None = None,
) -> int:
    """Apply ``selected`` proposals and return number of files updated.

    Same ``fields`` for every record (legacy contract); see
    :func:`apply_tag_changes` for per-record field selection. Pass a
    ``TagApplyResult`` as ``result`` to learn which files succeeded.
    """
    selected = list(selected)
    record_event(f"tag_fixer: applying {len(selected)} proposals")

    if fields is None:
        fields = ["artist", "title"]

    outcome = apply_tag_changes(
        [(p, fields) for p in selected], log_callback=log_callback
    )
    if result is not None:
        result.applied[:] = outcome.applied
        result.failed[:] = outcome.failed
        result.skipped[:] = outcome.skipped
        result.written_fields.update(outcome.written_fields)
    return len(outcome.applied)


@watcher.traced
def fix_tags(target, log_callback=None, interactive=False):
    """Fill missing tags for files in target using AcoustID."""
    if log_callback is None:
        def log_callback(msg):
            print(msg)

    record_event(f"tag_fixer: fix_tags starting on {target}")

    files = find_files(target)
    if not files:
        log_callback("No audio files found.")
        record_event("tag_fixer: no audio files found")
        return {"processed": 0, "updated": 0}

    base_folder = target if os.path.isdir(target) else os.path.dirname(target)
    docs_dir = os.path.join(base_folder, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    db_path = os.path.join(docs_dir, ".soundvault.db")
    init_db(db_path)

    db_folder = os.path.dirname(db_path)
    os.makedirs(db_folder, exist_ok=True)
    db_conn = sqlite3.connect(db_path)
    records = build_file_records(
        target,
        db_conn=db_conn,
        show_all=False,
        log_callback=log_callback,
    )
    db_conn.commit()
    db_conn.close()
    selected = [p for p in records if p.status not in ("no_diff", "unmatched")]

    if interactive:
        selected = []
        for p in [r for r in records if r.status not in ("no_diff", "unmatched")]:
            apply_change = prompt_user_about_tags(
                str(p.path),
                p.old_artist,
                p.old_title,
                {"artist": p.new_artist, "title": p.new_title}
            )
            if apply_change:
                selected.append(p)

    outcome = TagApplyResult()
    updated = apply_tag_proposals(
        selected,
        fields=["artist", "title"],
        log_callback=log_callback,
        result=outcome,
    )

    db_folder = os.path.dirname(db_path)
    os.makedirs(db_folder, exist_ok=True)
    db_conn = sqlite3.connect(db_path)
    # Only files whose write succeeded are "applied"; a failed write stays a
    # candidate for the next run.
    applied_set = set(outcome.applied)
    for rec in records:
        if str(rec.path) in applied_set:
            status = "applied"
        elif rec.status == "no_diff":
            status = "no_diff"
        else:
            status = "skipped"
        rec.status = status
        db_conn.execute(
            "UPDATE files SET status=? WHERE path=?",
            (status, str(rec.path)),
        )

    db_conn.commit()
    db_conn.close()
    result = {"processed": len(files), "updated": updated}
    record_event(
        f"tag_fixer: fix_tags finished processed={len(files)} updated={updated}"
    )
    return result

# ─── CLI Entry Point ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fill missing audio tags using AcoustID"
    )
    parser.add_argument("target", help="file or folder to process")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="show and confirm tag changes for medium-confidence matches",
    )
    args = parser.parse_args()

    try:
        summary = fix_tags(args.target, interactive=args.interactive)
        print(f"\nProcessed {summary['processed']} files, updated {summary['updated']}.")
    except RuntimeError as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
