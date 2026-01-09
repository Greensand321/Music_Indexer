"""Export and reporting helpers for the Library Sync review experience."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

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
