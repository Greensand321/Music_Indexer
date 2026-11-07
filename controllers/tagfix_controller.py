import os
import json
import sqlite3
from typing import Callable, Iterable, List, Optional, Tuple

from tag_fixer import (
    build_file_records,
    init_db,
    find_files,
    apply_tag_proposals,
    apply_tag_changes,
    FileRecord,
    ScanDiagnostics,
    TagApplyResult,
)
from crash_watcher import record_event
from crash_logger import watcher


@watcher.traced
def prepare_library(folder: str) -> tuple[str, dict]:
    """Initialize DB and load saved genre mapping."""
    record_event(f"tagfix_controller: preparing library {folder}")
    docs_dir = os.path.join(folder, "Docs")
    os.makedirs(docs_dir, exist_ok=True)
    db_path = os.path.join(docs_dir, ".soundvault.db")
    init_db(db_path)

    mapping_path = os.path.join(docs_dir, ".genre_mapping.json")
    mapping = {}
    if os.path.isfile(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception:
            mapping = {}
    record_event("tagfix_controller: library prepared")
    return db_path, mapping


@watcher.traced
def discover_files(folder: str) -> List[str]:
    """Return supported audio files under ``folder`` (reserved folders skipped)."""
    record_event(f"tagfix_controller: discovering files in {folder}")
    files = find_files(folder)
    record_event(f"tagfix_controller: found {len(files)} files")
    return files


@watcher.traced
def gather_records(
    folder: str,
    db_path: str,
    show_all: bool,
    progress_callback: Callable[[int], None] | None,
    log_callback: Callable[[str], None] | None = None,
    *,
    progress_detail: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    diagnostics: ScanDiagnostics | None = None,
) -> List[FileRecord]:
    """Build FileRecord objects for ``folder``.

    ``progress_callback`` gets the file index (legacy); ``progress_detail``
    gets ``(index, total)``; ``should_cancel`` is polled per file.
    """
    record_event(f"tagfix_controller: gathering records for {folder}")
    db_folder = os.path.dirname(db_path)
    os.makedirs(db_folder, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        records = build_file_records(
            folder,
            db_conn=conn,
            show_all=show_all,
            log_callback=log_callback or (lambda m: None),
            progress_callback=progress_callback,
            progress_detail=progress_detail,
            should_cancel=should_cancel,
            diagnostics=diagnostics,
        )
        conn.commit()
    finally:
        conn.close()
    record_event(f"tagfix_controller: gathered {len(records)} records")
    return records


def _mark_statuses(db_path: str, all_records: Iterable[FileRecord], applied_paths: set) -> None:
    """Persist per-file status. Only files whose write succeeded are "applied"."""
    db_folder = os.path.dirname(db_path)
    os.makedirs(db_folder, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        for rec in all_records:
            if str(rec.path) in applied_paths:
                new_status = 'applied'
            elif rec.status == 'no_diff':
                new_status = 'no_diff'
            else:
                new_status = 'skipped'
            rec.status = new_status
            conn.execute(
                "UPDATE files SET status=? WHERE path=?",
                (new_status, str(rec.path)),
            )
        conn.commit()
    finally:
        conn.close()


@watcher.traced
def apply_selected_fields(
    selections: Iterable[Tuple[FileRecord, Iterable[str]]],
    all_records: List[FileRecord],
    db_path: str,
    *,
    log_callback: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    library_root: Optional[str] = None,
) -> TagApplyResult:
    """Write exactly the ticked ``(record, fields)`` pairs and update the DB.

    Every write (and every failure or skip) is recorded with its before and
    after values in ``Docs/tag_changes_tagfixer_<timestamp>.jsonl`` under the
    library root (derived from ``db_path``'s ``Docs`` folder when not given).
    """
    from utils.tag_change_log import TagChangeLog

    selections = [(rec, list(fields)) for rec, fields in selections]
    record_event(f"tagfix_controller: applying {len(selections)} selections")
    root = library_root or os.path.dirname(os.path.dirname(os.path.abspath(db_path)))
    with TagChangeLog(root, "tagfixer") as change_log:
        result = apply_tag_changes(
            selections,
            log_callback=log_callback,
            should_cancel=should_cancel,
            progress_callback=progress_callback,
            change_log=change_log,
        )
        result.log_path = change_log.path
    _mark_statuses(db_path, all_records, set(result.applied))
    record_event(f"tagfix_controller: applied proposals to {len(result.applied)} files")
    return result


@watcher.traced
def apply_proposals(selected: List[FileRecord], all_records: List[FileRecord], db_path: str, fields: List[str], log_callback: Callable[[str], None]) -> int:
    """Apply tag proposals (same ``fields`` for every record) and update DB.

    Returns the number of files actually written. Only those are marked
    ``applied``; a failed write is not.
    """
    result = apply_selected_fields(
        [(rec, fields) for rec in selected],
        all_records,
        db_path,
        log_callback=log_callback,
    )
    return len(result.applied)
