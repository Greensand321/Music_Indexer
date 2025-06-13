import os
import json
import sqlite3
from typing import List, Callable

from tag_fixer import build_file_records, init_db, find_files, apply_tag_proposals, FileRecord


def prepare_library(folder: str) -> tuple[str, dict]:
    """Initialize DB and load saved genre mapping."""
    db_path = os.path.join(folder, ".soundvault.db")
    init_db(db_path)

    mapping_path = os.path.join(folder, ".genre_mapping.json")
    mapping = {}
    if os.path.isfile(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
        except Exception:
            mapping = {}
    return db_path, mapping


def discover_files(folder: str) -> List[str]:
    """Return supported audio files under ``folder``."""
    return find_files(folder)


def gather_records(folder: str, db_path: str, show_all: bool, progress_callback: Callable[[int], None] | None) -> List[FileRecord]:
    """Build FileRecord objects for ``folder``."""
    conn = sqlite3.connect(db_path)
    records = build_file_records(
        folder,
        db_conn=conn,
        show_all=show_all,
        log_callback=lambda m: None,
        progress_callback=progress_callback,
    )
    conn.commit()
    conn.close()
    return records


def apply_proposals(selected: List[FileRecord], all_records: List[FileRecord], db_path: str, fields: List[str], log_callback: Callable[[str], None]) -> int:
    """Apply tag proposals and update DB."""
    count = apply_tag_proposals(selected, fields=fields, log_callback=log_callback)
    selected_paths = {rec.path for rec in selected}
    conn = sqlite3.connect(db_path)
    for rec in all_records:
        if rec.path in selected_paths:
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
    conn.close()
    return count
