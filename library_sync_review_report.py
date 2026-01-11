"""Export and reporting helpers for the Library Sync review experience."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

import library_sync
from library_sync import MatchResult, MatchStatus
from library_sync_review_state import ReviewStateStore, quality_delta

DEFAULT_REPORT_VERSION = 1


class ReportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"

    @classmethod
    def from_value(cls, value: str) -> "ReportFormat":
        try:
            return cls(value.lower())
        except Exception as exc:
            raise ValueError(f"Unsupported report format: {value}") from exc


REQUIRED_FIELDS = [
    "schema_version",
    "incoming_path",
    "existing_path",
    "status",
    "distance",
    "threshold_used",
    "incoming_quality_score",
    "existing_quality_score",
    "quality_delta",
    "user_flag",
    "notes",
    "timestamp",
    "scan_config_hash",
]

PreviewHandler = Callable[[Dict[str, int]], bool]


@dataclass
class ExportRecord:
    """Row captured in the export plan/report."""

    schema_version: int
    incoming_path: str
    existing_path: str | None
    status: str
    distance: float | None
    threshold_used: float
    incoming_quality_score: int
    existing_quality_score: int | None
    quality_delta: int | None
    user_flag: str | None
    notes: str
    timestamp: str
    scan_config_hash: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "incoming_path": self.incoming_path,
            "existing_path": self.existing_path,
            "status": self.status,
            "distance": self.distance,
            "threshold_used": self.threshold_used,
            "incoming_quality_score": self.incoming_quality_score,
            "existing_quality_score": self.existing_quality_score,
            "quality_delta": self.quality_delta,
            "user_flag": self.user_flag,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "scan_config_hash": self.scan_config_hash,
        }


def _normalize_scan_config(scan_config: Mapping[str, object] | object) -> Dict[str, object]:
    if hasattr(scan_config, "to_dict"):
        cfg = getattr(scan_config, "to_dict")()
        if isinstance(cfg, Mapping):
            return dict(cfg)
    if isinstance(scan_config, Mapping):
        return dict(scan_config)
    raise TypeError("scan_config must be a mapping or expose a to_dict method.")


def calculate_scan_config_hash(scan_config: Mapping[str, object]) -> str:
    """Return a deterministic hash for the current scan configuration."""
    canonical = json.dumps(scan_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def summarize_match_results(results: Iterable[MatchResult], flags: ReviewStateStore) -> Dict[str, int]:
    """Summarize review outcomes for preview prior to export."""
    new_count = 0
    collision_count = 0
    low_conf_count = 0
    flagged_copy = 0
    flagged_replace = 0

    for res in results:
        if res.status == MatchStatus.NEW:
            new_count += 1
        elif res.status in (MatchStatus.COLLISION, MatchStatus.EXACT_MATCH):
            collision_count += 1
        elif res.status == MatchStatus.LOW_CONFIDENCE:
            low_conf_count += 1

        if flags.is_copy_flagged(res.incoming.track_id):
            flagged_copy += 1
        if flags.replace_target(res.incoming.track_id):
            flagged_replace += 1

    return {
        "new": new_count,
        "collisions": collision_count,
        "low_confidence": low_conf_count,
        "flagged_copy": flagged_copy,
        "flagged_replace": flagged_replace,
    }


def build_export_records(
    results: Sequence[MatchResult],
    flags: ReviewStateStore,
    scan_config: Mapping[str, object],
    report_version: int = DEFAULT_REPORT_VERSION,
    notes: MutableMapping[str, str] | None = None,
    timestamp: datetime | None = None,
) -> tuple[list[ExportRecord], Dict[str, str]]:
    """Create structured export rows without touching the filesystem."""
    normalized_cfg = _normalize_scan_config(scan_config)
    scan_hash = calculate_scan_config_hash(normalized_cfg)
    ts = timestamp or datetime.now(timezone.utc)
    ts_str = ts.isoformat()
    note_map = notes if notes is not None else flags.flags.notes

    rows: list[ExportRecord] = []
    for res in results:
        incoming_id = res.incoming.track_id
        user_flag: str | None = None
        if flags.replace_target(incoming_id):
            user_flag = "replace"
        elif flags.is_copy_flagged(incoming_id):
            user_flag = "copy"

        row = ExportRecord(
            schema_version=report_version,
            incoming_path=res.incoming.path,
            existing_path=res.existing.path if res.existing else None,
            status=res.status.value,
            distance=res.distance,
            threshold_used=res.threshold_used,
            incoming_quality_score=res.incoming_score,
            existing_quality_score=res.existing_score,
            quality_delta=quality_delta(res),
            user_flag=user_flag,
            notes=note_map.get(incoming_id, "") if hasattr(note_map, "get") else "",
            timestamp=ts_str,
            scan_config_hash=scan_hash,
        )
        rows.append(row)

    metadata = {"timestamp": ts_str, "scan_config_hash": scan_hash}
    return rows, metadata


def export_report(
    output_path: str,
    results: Sequence[MatchResult],
    flags: ReviewStateStore,
    scan_config: Mapping[str, object],
    fmt: str = ReportFormat.JSON.value,
    report_version: int = DEFAULT_REPORT_VERSION,
    notes: MutableMapping[str, str] | None = None,
    timestamp: datetime | None = None,
    preview_handler: PreviewHandler | None = None,
) -> str:
    """Write a CSV or JSON report of the review state.

    The export routine deliberately avoids any move/replace logic and only
    writes the requested report file after the caller has previewed counts.
    """
    summary = summarize_match_results(results, flags)
    if preview_handler:
        proceed = preview_handler(summary)
        if not proceed:
            return ""

    records, metadata = build_export_records(
        results,
        flags,
        scan_config,
        report_version=report_version,
        notes=notes,
        timestamp=timestamp,
    )
    format_enum = ReportFormat.from_value(fmt)

    if format_enum is ReportFormat.JSON:
        payload = {
            "schema_version": report_version,
            "generated_at": metadata["timestamp"],
            "scan_config_hash": metadata["scan_config_hash"],
            "items": [rec.to_dict() for rec in records],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec.to_dict())

    return output_path


def _resolve_thresholds(
    threshold: float | None,
    thresholds: Mapping[str, float] | None,
) -> Dict[str, float]:
    if thresholds is not None:
        return {str(k): float(v) for k, v in thresholds.items()}
    if threshold is not None:
        return {"default": float(threshold)}
    return dict(library_sync.DEFAULT_FP_THRESHOLDS)


def _verdict_for_result(result: MatchResult) -> str:
    if result.existing is None:
        return "New (no match)"
    if result.quality_label:
        return result.quality_label
    if result.existing_score is None:
        return "Match"
    return "Replace" if result.incoming_score > (result.existing_score or 0) else "Keep Existing"


def _mixed_codec_info(
    result: MatchResult,
    mixed_codec_boost: float,
) -> tuple[float, bool]:
    if result.existing is None:
        return 0.0, False
    ext_a = result.incoming.ext
    ext_b = result.existing.ext
    if not ext_a or not ext_b or ext_a == ext_b:
        return 0.0, False
    lossless_a = ext_a in library_sync.LOSSLESS_EXTS
    lossless_b = ext_b in library_sync.LOSSLESS_EXTS
    if lossless_a != lossless_b:
        return float(mixed_codec_boost), True
    return 0.0, False


def export_review_report_html(
    output_html_path: str,
    results: Sequence[MatchResult],
    *,
    library_root: str,
    incoming_root: str,
    thresholds: Mapping[str, float],
    mixed_codec_threshold_boost: float,
    format_priority: Mapping[str, int],
    generated_at: datetime | None = None,
) -> str:
    """Write an HTML report mirroring the duplicate finder report styling."""
    esc = lambda value: html.escape(str(value))
    ts = generated_at or datetime.now(timezone.utc)
    counts = summarize_match_results(results, ReviewStateStore())
    html_lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8' />",
        "<title>Library Sync Review Report</title>",
        "<style>",
        "body{font-family:Arial, sans-serif; margin:24px; color:#222;}",
        "h1{font-size:20px; margin-bottom:6px;}",
        "h2{font-size:16px; margin-top:24px;}",
        "table{border-collapse:collapse; width:100%; margin-top:8px;}",
        "th,td{border:1px solid #ddd; padding:8px; text-align:left; vertical-align:top;}",
        "th{background:#f4f4f4; width:220px;}",
        ".badge{display:inline-block; padding:2px 8px; border-radius:12px; font-size:12px; margin-left:6px;}",
        ".ok{background:#d4f4dd; color:#116329;}",
        ".fail{background:#ffe0e0; color:#8a1f1f;}",
        ".blocked{background:#fff1c1; color:#7a5c00;}",
        ".path{font-family:monospace; word-break:break-all;}",
        ".meta{color:#555; font-size:12px; margin-top:4px;}",
        ".muted{color:#666; font-size:12px;}",
        ".actions{display:flex; gap:10px; flex-wrap:wrap; margin:12px 0;}",
        ".action-btn{border:1px solid #ddd; background:#fff; padding:6px 12px; border-radius:6px; cursor:pointer;}",
        ".action-btn.primary{background:#116329; color:#fff; border-color:#116329;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Library Sync Review Report</h1>",
        f"<div class='muted'>Generated: {esc(ts.isoformat())}</div>",
        "<div class='actions'>",
        "<button class='action-btn primary' data-action='replace-all'>Replace All</button>",
        "<button class='action-btn' data-action='replace-better'>Replace with Better</button>",
        "<button class='action-btn' data-action='cancel'>Cancel</button>",
        "</div>",
        "<h2>Summary</h2>",
        "<table>",
        f"<tr><th>Library Root</th><td class='path'>{esc(library_root)}</td></tr>",
        f"<tr><th>Incoming Folder</th><td class='path'>{esc(incoming_root)}</td></tr>",
        f"<tr><th>New</th><td>{esc(counts.get('new', 0))}</td></tr>",
        f"<tr><th>Collisions / Exact Matches</th><td>{esc(counts.get('collisions', 0))}</td></tr>",
        f"<tr><th>Low Confidence</th><td>{esc(counts.get('low_confidence', 0))}</td></tr>",
        "</table>",
        "<h2>Threshold Settings Snapshot</h2>",
        "<table>",
        "<tr><th>Setting</th><th>Value</th></tr>",
    ]

    for key, value in sorted(thresholds.items(), key=lambda item: str(item[0])):
        html_lines.append(f"<tr><td>{esc(key)}</td><td>{esc(value)}</td></tr>")
    html_lines.append(
        f"<tr><td>mixed_codec_threshold_boost</td><td>{esc(mixed_codec_threshold_boost)}</td></tr>"
    )
    html_lines.append("</table>")

    html_lines.extend(
        [
            "<h2>Format Priority Snapshot</h2>",
            "<table>",
            "<tr><th>Format</th><th>Priority</th></tr>",
        ]
    )
    for key, value in sorted(format_priority.items(), key=lambda item: str(item[0])):
        html_lines.append(f"<tr><td>{esc(key)}</td><td>{esc(value)}</td></tr>")
    html_lines.append("</table>")

    status_classes = {
        MatchStatus.NEW: "blocked",
        MatchStatus.COLLISION: "ok",
        MatchStatus.EXACT_MATCH: "ok",
        MatchStatus.LOW_CONFIDENCE: "fail",
    }
    for idx, res in enumerate(results, start=1):
        verdict = _verdict_for_result(res)
        badge_class = status_classes.get(res.status, "blocked")
        mixed_boost, mixed_applied = _mixed_codec_info(res, mixed_codec_threshold_boost)
        html_lines.extend(
            [
                f"<h2>Incoming Track {idx}</h2>",
                "<table>",
                f"<tr><th>Status</th><td>{esc(res.status.value)}<span class='badge {badge_class}'>{esc(res.status.value)}</span></td></tr>",
                f"<tr><th>Verdict</th><td>{esc(verdict)}</td></tr>",
                f"<tr><th>Incoming Path</th><td class='path'>{esc(res.incoming.path)}</td></tr>",
                f"<tr><th>Existing Match</th><td class='path'>{esc(res.existing.path) if res.existing else 'None'}</td></tr>",
                f"<tr><th>Fingerprint Distance</th><td>{esc(f'{res.distance:.4f}' if res.distance is not None else 'n/a')}</td></tr>",
                f"<tr><th>Threshold Used</th><td>{esc(f'{res.threshold_used:.4f}')}</td></tr>",
                f"<tr><th>Mixed-codec Boost</th><td>{esc(f'{mixed_boost:.4f}')} (applied: {esc('Yes' if mixed_applied else 'No')})</td></tr>",
                f"<tr><th>Format Priority Score</th><td>Incoming: {esc(res.incoming_score)} | Existing: {esc(res.existing_score) if res.existing_score is not None else 'n/a'}</td></tr>",
                "</table>",
            ]
        )

    html_lines.extend(["</body>", "</html>"])

    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    return output_html_path


def generate_review_report_html(
    library_root: str,
    incoming_folder: str,
    db_path: str,
    output_html_path: str,
    *,
    threshold: float | None = None,
    thresholds: Mapping[str, float] | None = None,
    fmt_priority: Mapping[str, int] | None = None,
    mixed_codec_threshold_boost: float | None = None,
    fingerprint_settings: Mapping[str, object] | None = None,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    cancel_event=None,
) -> Dict[str, object]:
    """Scan, match, and emit the review HTML report using compare_libraries."""
    threshold_map = _resolve_thresholds(threshold, thresholds)
    fmt_priority = fmt_priority or library_sync.FORMAT_PRIORITY
    if mixed_codec_threshold_boost is None:
        cfg = library_sync.idx_config.load_config()
        mixed_codec_threshold_boost = float(
            cfg.get("mixed_codec_threshold_boost", library_sync.MIXED_CODEC_THRESHOLD_BOOST)
        )

    result = library_sync.compare_libraries(
        library_root,
        incoming_folder,
        db_path,
        threshold=threshold,
        fmt_priority=fmt_priority,
        thresholds=threshold_map,
        mixed_codec_threshold_boost=mixed_codec_threshold_boost,
        fingerprint_settings=fingerprint_settings,
        log_callback=log_callback,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        include_match_objects=True,
    )
    match_objects = result.get("match_objects", [])
    export_review_report_html(
        output_html_path,
        match_objects,
        library_root=library_root,
        incoming_root=incoming_folder,
        thresholds=threshold_map,
        mixed_codec_threshold_boost=float(mixed_codec_threshold_boost),
        format_priority=fmt_priority,
    )
    result["report_path"] = output_html_path
    return result
