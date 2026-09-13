"""Turn Playlist Gap results into the thing the user actually uses.

The deliverable is the missing list, and its format should suit how it gets
used: pasted into a downloader. The plain-text writer is therefore the default
and the least glamorous thing here.

Qt-free, writes only to paths it is given. See ``docs/playlist_gap_spec.md`` §12.
"""
from __future__ import annotations

import csv
import io
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from playlist_gap_types import GapResult, Verdict


def missing_rows(results: Iterable[GapResult]) -> List[GapResult]:
    """Rows the user should go download, in the source's own order."""
    rows = [r for r in results if r.verdict is Verdict.MISSING]
    rows.sort(key=lambda r: (r.wanted.position if r.wanted.position is not None else 0))
    return rows


def _label(result: GapResult) -> str:
    return result.wanted.label()


def render_text(results: Iterable[GapResult], *, group_by: str = "flat") -> str:
    """One ``Artist - Title`` per line. The default, and the most used.

    ``group_by="album"`` annotates how much of each album is missing, because
    grabbing a whole album once is often easier and better quality than chasing
    six singles.
    """
    rows = missing_rows(results)
    if not rows:
        return "# Nothing missing — you already have everything on this list.\n"

    if group_by == "flat":
        return "".join(f"{_label(r)}\n" for r in rows)

    key = (lambda r: r.wanted.album or "") if group_by == "album" else (
        lambda r: r.wanted.artist or "")
    groups: "OrderedDict[str, List[GapResult]]" = OrderedDict()
    for row in rows:
        groups.setdefault(key(row) or "", []).append(row)

    out = io.StringIO()
    loose = groups.pop("", [])
    for name, members in groups.items():
        out.write(f"# {name}  ({len(members)} missing)\n")
        for row in members:
            out.write(f"{_label(row)}\n")
        out.write("\n")
    if loose:
        out.write(f"# Singles & one-offs  ({len(loose)})\n")
        for row in loose:
            out.write(f"{_label(row)}\n")
    return out.getvalue()


CSV_FIELDS: Tuple[str, ...] = (
    "row_id", "artist", "title", "display", "album", "duration",
    "verdict", "reason_code", "reason", "rung", "playlists", "best_match",
)


def render_csv(results: Iterable[GapResult], *, only_missing: bool = True) -> str:
    """Everything, for a spreadsheet or a diff against a later run."""
    rows = missing_rows(results) if only_missing else list(results)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    for result in rows:
        best = result.best
        writer.writerow({
            "row_id": result.wanted.row_id,
            "artist": result.wanted.artist or "",
            "title": result.wanted.title or "",
            "display": result.wanted.display,
            "album": result.wanted.album or "",
            "duration": result.wanted.duration if result.wanted.duration else "",
            "verdict": result.verdict.value,
            "reason_code": result.reason_code.value,
            "reason": result.reason_text,
            "rung": result.rung.value,
            "playlists": "; ".join(p for p in result.wanted.playlists if p),
            "best_match": best.track.path if best else "",
        })
    return buffer.getvalue()


def write_text(path: str, results: Iterable[GapResult], *, group_by: str = "flat") -> str:
    return _write(path, render_text(results, group_by=group_by))


def write_csv(path: str, results: Iterable[GapResult], *, only_missing: bool = True) -> str:
    return _write(path, render_csv(results, only_missing=only_missing))


def _write(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return path


def summary_lines(results: Sequence[GapResult], counts: Dict[str, int]) -> List[str]:
    """Short, plain-language run summary for the log drawer and the report."""
    total = counts.get("total", len(results))
    owned = counts.get(Verdict.OWNED.value, 0)
    share = f"{owned / total:.0%}" if total else "0%"
    lines = [
        f"{total} wanted tracks compared against the library.",
        f"{owned} already owned ({share} of the list).",
        f"{counts.get(Verdict.UNSURE.value, 0)} need confirmation.",
        f"{counts.get(Verdict.MISSING.value, 0)} to download.",
    ]
    unavailable = counts.get(Verdict.UNAVAILABLE.value, 0)
    if unavailable:
        lines.append(f"{unavailable} unavailable upstream (deleted or private).")
    ignored = counts.get(Verdict.IGNORED.value, 0)
    if ignored:
        lines.append(f"{ignored} on your never-want list.")
    return lines
