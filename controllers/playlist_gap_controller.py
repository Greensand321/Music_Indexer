"""Orchestration for Playlist Gap: read a source, snapshot, match, remember.

Thin by design. Every decision of substance lives in the backend modules; this
just sequences them, threads progress callbacks through, and owns the one piece
of cross-module policy — that a run's results are written to the ledger and that
stored human decisions are applied on the way back.

Qt-free, so the workspace's worker threads can call it directly and the whole
pipeline stays testable without a GUI.

See ``docs/playlist_gap_spec.md`` §1.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from playlist_gap_ledger import DECIDED_AUTO, Ledger, WantedDiff, ledger_path_for
from playlist_gap_lexicon import Lexicon, load_lexicon
from playlist_gap_match import DEFAULT_SCORER, Scorer, match_all, summarize
from playlist_gap_report import summary_lines
from playlist_gap_snapshot import DEFAULT_INCLUDE, FolderPolicy, LibrarySnapshot, build_snapshot
from playlist_gap_sources import CsvSource, SourceSpec, merge_wanted
from playlist_gap_types import (
    DEFAULT_THRESHOLDS,
    GapResult,
    SourceCapabilities,
    Thresholds,
    WantedTrack,
)

Progress = Callable[[int, int, str], None]
Log = Callable[[str], None]


def _noop_log(_message: str) -> None:
    pass


@dataclass
class RunResult:
    """Everything one comparison produced."""

    spec: SourceSpec
    results: List[GapResult] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    capabilities: SourceCapabilities = field(default_factory=SourceCapabilities)
    snapshot: Optional[LibrarySnapshot] = None
    diff: Optional[WantedDiff] = None
    warnings: List[str] = field(default_factory=list)
    column_mapping: Dict[str, str] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def elapsed(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def summary(self) -> List[str]:
        return summary_lines(self.results, self.counts)


def read_source(
    spec: SourceSpec,
    *,
    progress: Optional[Progress] = None,
):
    """Read a wanted list. Phase 1 supports CSV; other kinds raise clearly."""
    if spec.kind != "csv":
        raise NotImplementedError(
            f"Source kind {spec.kind!r} is not wired yet — Phase 1 reads CSV exports. "
            "Use a CSV export for now; the importer is pluggable so richer sources "
            "drop in without changing the matcher."
        )
    return CsvSource().read(spec, progress=progress)


def load_snapshot(
    library_root: str,
    *,
    policy: FolderPolicy = DEFAULT_INCLUDE,
    cache_db: Optional[str] = None,
    progress: Optional[Progress] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> LibrarySnapshot:
    """Index the library under this feature's own inclusion policy."""
    return build_snapshot(
        library_root,
        policy=policy,
        cache_db=cache_db,
        progress=progress,
        should_cancel=should_cancel,
    )


def run_source(
    spec: SourceSpec,
    library_root: str,
    *,
    snapshot: Optional[LibrarySnapshot] = None,
    lexicon: Optional[Lexicon] = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    scorer: Scorer = DEFAULT_SCORER,
    ledger: Optional[Ledger] = None,
    own_ledger: bool = True,
    cache_db: Optional[str] = None,
    progress: Optional[Progress] = None,
    log: Log = _noop_log,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> RunResult:
    """Read, snapshot, match and remember — the whole of pass 1 for one source."""
    started = time.time()
    spec.ensure_id()
    out = RunResult(spec=spec, started_at=started)

    log(f"Reading “{spec.name or spec.location}”…")
    read = read_source(spec, progress=progress)
    out.capabilities = read.capabilities
    out.warnings = list(read.warnings)
    out.column_mapping = dict(read.mapping)
    log(f"{len(read.tracks)} rows · fields available: {read.capabilities.describe()}")
    if read.blank_rows:
        log(f"{read.blank_rows} row(s) have no title — kept and reported, never dropped.")

    if snapshot is None:
        log("Reading the library…")
        snapshot = load_snapshot(
            library_root, cache_db=cache_db, progress=progress, should_cancel=should_cancel
        )
    out.snapshot = snapshot
    log(f"Library snapshot: {len(snapshot)} tracks · " + " · ".join(
        f"{name} {count}" for name, count in sorted(snapshot.counts.items())
    ))

    close_ledger = False
    if ledger is None and own_ledger:
        ledger = Ledger(ledger_path_for(library_root))
        close_ledger = True

    try:
        if ledger is not None:
            ledger.upsert_source(spec.source_id, spec.name, spec.kind, spec.location)
            out.diff = ledger.sync_wanted(spec.source_id, read.tracks)
            if out.diff.added:
                log(f"{out.diff.new_count} new since the last run of this source.")
            if out.diff.removed_upstream:
                log(f"{len(out.diff.removed_upstream)} row(s) have left the playlist upstream.")

        log("Comparing…")
        out.results = match_all(
            read.tracks,
            snapshot,
            caps=read.capabilities,
            lexicon=lexicon or load_lexicon(library_root),
            thresholds=thresholds,
            scorer=scorer,
            decisions=ledger.decisions() if ledger is not None else None,
            progress=progress,
            should_cancel=should_cancel,
        )
        out.counts = summarize(out.results)
        out.finished_at = time.time()

        if ledger is not None:
            # Automatic verdicts are recorded but carry the lowest authority, so
            # a human answer from a previous run is never overwritten.
            for result in out.results:
                if result.auto:
                    ledger.record(
                        result.wanted.row_id,
                        result.verdict,
                        by=DECIDED_AUTO,
                        reason_code=result.reason_code.value,
                        matched_path=result.best.track.path if result.best else None,
                    )
            ledger.record_run(
                uuid.uuid4().hex[:12], spec.source_id, json.dumps(out.counts),
                out.started_at, out.finished_at,
            )
            ledger.touch_source(spec.source_id)
    finally:
        if close_ledger and ledger is not None:
            ledger.close()

    for line in out.summary():
        log(line)
    return out


def run_all(
    specs: Sequence[SourceSpec],
    library_root: str,
    **kwargs,
) -> RunResult:
    """Compare several saved sources against one shared library snapshot.

    The snapshot is the expensive part and is identical across sources, so it is
    built once. Rows are merged by identity, so a track wanted by three playlists
    appears once in the download list rather than three times.
    """
    if not specs:
        raise ValueError("run_all needs at least one source")
    snapshot = kwargs.pop("snapshot", None)
    if snapshot is None:
        snapshot = load_snapshot(
            library_root,
            cache_db=kwargs.get("cache_db"),
            progress=kwargs.get("progress"),
            should_cancel=kwargs.get("should_cancel"),
        )
    runs = [run_source(spec, library_root, snapshot=snapshot, **kwargs) for spec in specs]

    combined = RunResult(
        spec=SourceSpec(name="All saved sources", kind="multi"),
        snapshot=snapshot,
        started_at=min(r.started_at for r in runs),
        finished_at=max(r.finished_at for r in runs),
    )
    merged = merge_wanted([[r.wanted for r in run.results] for run in runs])
    keep = {track.row_id for track in merged}
    seen: set = set()
    for run in runs:
        combined.warnings.extend(run.warnings)
        for result in run.results:
            if result.wanted.row_id in keep and result.wanted.row_id not in seen:
                seen.add(result.wanted.row_id)
                combined.results.append(result)
    combined.counts = summarize(combined.results)
    return combined
