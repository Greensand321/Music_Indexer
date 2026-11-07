"""Append-only record of every tag rewrite, kept under the library's ``Docs/``.

Tag edits happen in place, so once a file is saved its old values are gone.
Each apply run of the Tag Fixer or the Genre Normalizer therefore writes a
JSON-lines file ``Docs/tag_changes_<tool>_<timestamp>.jsonl`` with one line
per file: what each field held before, what was written, and whether the
write succeeded. That is enough to see — and undo, with
``controllers.tag_revert`` — exactly what an apply changed. A revert is
itself logged (tool ``revert``, every entry naming the log it reverts in
``"reverts"``), so a revert can be reverted too.

Lines are flushed as they are written, so the log survives a crash mid-run.
Backend only: no Qt, no Tk.
"""
from __future__ import annotations

import json
import os
import time
from typing import Mapping, Optional, Sequence, Tuple

FieldChange = Tuple[object, object]  # (before, after)


class TagChangeLog:
    """One apply run's change log. Use as a context manager or call close()."""

    def __init__(self, library_root: str, tool: str, *, reverts: Optional[str] = None) -> None:
        self.tool = tool
        #: file name of the change log this run undoes (revert runs only)
        self.reverts = os.path.basename(reverts) if reverts else None
        docs = os.path.join(library_root, "Docs")
        os.makedirs(docs, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join(docs, f"tag_changes_{tool}_{stamp}")
        path = f"{base}.jsonl"
        n = 1
        while os.path.exists(path):
            n += 1
            path = f"{base}_{n}.jsonl"
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")
        self.written = 0
        self.failed = 0
        self.skipped = 0

    def record(
        self,
        path: str,
        changes: Mapping[str, FieldChange],
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """``status`` is ``"written"``, ``"failed"`` or ``"skipped"``."""
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tool": self.tool,
            "path": str(path),
            "status": status,
            "changes": {
                field: {"before": _jsonable(before), "after": _jsonable(after)}
                for field, (before, after) in changes.items()
            },
        }
        if error:
            entry["error"] = error
        if self.reverts:
            entry["reverts"] = self.reverts
        if status == "written":
            self.written += 1
        elif status == "failed":
            self.failed += 1
        else:
            self.skipped += 1
        if self._fh is not None:
            self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "TagChangeLog":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Sequence):
        return [_jsonable(v) for v in value]
    return str(value)


def read_change_log(path: str) -> list[dict]:
    """Parse a change log back into a list of entries (for tests and tools)."""
    entries = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
