"""Durable memory for Playlist Gap.

The problem this feature solves recurs every few months, so a tool that answers
perfectly but forgets everything has solved a third of it. The ledger is what
makes run two cheaper than run one: it remembers what each wanted row is, what
was decided about it, whether a human or the matcher decided, what has been
exported but not yet verified, and which rows the user never wants.

SQLite rather than JSON: the project already uses it (``fingerprint_cache``), row
counts run to thousands per source, and a partial write during a long triage
session must not corrupt decisions already made.

It lives under the *library's* ``Docs/`` so it travels with the music, while
saved sources live in config and travel with the app; the two join on
``source_id``, which is stored in both.

Qt-free. See ``docs/playlist_gap_spec.md`` §8.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from playlist_gap_types import Verdict, WantedTrack

LEDGER_RELPATH = os.path.join("Docs", "playlist_gap.sqlite3")

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS sources (
        source_id   TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        kind        TEXT NOT NULL,
        location    TEXT,
        last_run_at REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS wanted (
        row_id     TEXT NOT NULL,
        source_id  TEXT NOT NULL,
        display    TEXT NOT NULL,
        artist     TEXT,
        title      TEXT,
        album      TEXT,
        duration   INTEGER,
        isrc       TEXT,
        video_id   TEXT,
        first_seen REAL NOT NULL,
        last_seen  REAL NOT NULL,
        removed_at REAL,
        PRIMARY KEY (row_id, source_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS decisions (
        row_id      TEXT PRIMARY KEY,
        verdict     TEXT NOT NULL,
        decided_by  TEXT NOT NULL,
        reason_code TEXT,
        matched_path TEXT,
        note        TEXT,
        decided_at  REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS pending (
        row_id       TEXT PRIMARY KEY,
        exported_at  REAL NOT NULL,
        verified_at  REAL,
        outcome      TEXT,
        arrived_path TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS learned_rules (
        rule_id    TEXT PRIMARY KEY,
        kind       TEXT NOT NULL,
        payload    TEXT NOT NULL,
        hits       INTEGER DEFAULT 0,
        created_at REAL NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id      TEXT PRIMARY KEY,
        source_id   TEXT NOT NULL,
        started_at  REAL,
        finished_at REAL,
        counts      TEXT
    );
    """,
)

#: Who decided. Precedence: a human always beats the matcher; the verification
#: pass beats the matcher but not a human.
DECIDED_AUTO = "auto"
DECIDED_USER = "user"
DECIDED_VERIFY = "verify"
_AUTHORITY = {DECIDED_AUTO: 0, DECIDED_VERIFY: 1, DECIDED_USER: 2}


@dataclass
class Decision:
    row_id: str
    verdict: Verdict
    decided_by: str = DECIDED_AUTO
    reason_code: Optional[str] = None
    matched_path: Optional[str] = None
    note: Optional[str] = None
    decided_at: float = 0.0


@dataclass
class WantedDiff:
    """What changed in a wanted list since the last run.

    ``added`` is the answer to the question the user actually asks —
    "what's new in this playlist since I last checked?"
    """

    added: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    removed_upstream: List[str] = field(default_factory=list)

    @property
    def new_count(self) -> int:
        return len(self.added)


def ledger_path_for(library_root: str) -> str:
    return os.path.join(library_root, LEDGER_RELPATH)


class Ledger:
    """Read/write access to one library's Playlist Gap memory."""

    def __init__(self, db_path: str, *, clock=time.time) -> None:
        self.db_path = db_path
        self._clock = clock
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._initialize()

    # ── lifecycle ────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        self._conn.execute("PRAGMA journal_mode=WAL")
        for statement in SCHEMA:
            self._conn.execute(statement)
        self._upgrade()
        self._conn.commit()

    def _upgrade(self) -> None:
        """Additive migrations only.

        The ledger is the one thing here that cannot be regenerated — it holds
        human decisions — so a column is added, never dropped or rewritten.
        """
        for table, column, ddl in (
            ("wanted", "removed_at", "ALTER TABLE wanted ADD COLUMN removed_at REAL"),
            ("decisions", "matched_path", "ALTER TABLE decisions ADD COLUMN matched_path TEXT"),
            ("pending", "arrived_path", "ALTER TABLE pending ADD COLUMN arrived_path TEXT"),
        ):
            existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self._conn.execute(ddl)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── sources ──────────────────────────────────────────────────────────

    def upsert_source(self, source_id: str, name: str, kind: str, location: str) -> str:
        self._conn.execute(
            "INSERT INTO sources (source_id, name, kind, location) VALUES (?,?,?,?) "
            "ON CONFLICT(source_id) DO UPDATE SET name=excluded.name, "
            "kind=excluded.kind, location=excluded.location",
            (source_id, name, kind, location),
        )
        self._conn.commit()
        return source_id

    def sources(self) -> List[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM sources ORDER BY name"))

    def touch_source(self, source_id: str) -> None:
        self._conn.execute(
            "UPDATE sources SET last_run_at = ? WHERE source_id = ?",
            (self._clock(), source_id),
        )
        self._conn.commit()

    # ── wanted rows ──────────────────────────────────────────────────────

    def sync_wanted(self, source_id: str, rows: Sequence[WantedTrack]) -> WantedDiff:
        """Record this run's rows and report what changed since the last one."""
        now = self._clock()
        known = {
            row["row_id"]
            for row in self._conn.execute(
                "SELECT row_id FROM wanted WHERE source_id = ? AND removed_at IS NULL",
                (source_id,),
            )
        }
        seen = set()
        diff = WantedDiff()

        for track in rows:
            seen.add(track.row_id)
            if track.row_id in known:
                diff.unchanged.append(track.row_id)
            else:
                diff.added.append(track.row_id)
            self._conn.execute(
                "INSERT INTO wanted (row_id, source_id, display, artist, title, album,"
                " duration, isrc, video_id, first_seen, last_seen, removed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)"
                " ON CONFLICT(row_id, source_id) DO UPDATE SET"
                "   display=excluded.display, artist=excluded.artist,"
                "   title=excluded.title, album=excluded.album,"
                "   duration=excluded.duration, isrc=excluded.isrc,"
                "   video_id=excluded.video_id, last_seen=excluded.last_seen,"
                "   removed_at=NULL",
                (
                    track.row_id, source_id, track.display, track.artist, track.title,
                    track.album, track.duration, track.isrc, track.video_id, now, now,
                ),
            )

        for row_id in known - seen:
            diff.removed_upstream.append(row_id)
            self._conn.execute(
                "UPDATE wanted SET removed_at = ? WHERE row_id = ? AND source_id = ?",
                (now, row_id, source_id),
            )
        self._conn.commit()
        return diff

    # ── decisions ────────────────────────────────────────────────────────

    def decision_for(self, row_id: str) -> Optional[Decision]:
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE row_id = ?", (row_id,)
        ).fetchone()
        return self._to_decision(row) if row else None

    def decisions(self, *, human_only: bool = True) -> Dict[str, Tuple[Verdict, str]]:
        """Stored decisions keyed by row_id, for the matcher to apply.

        Only decisions with real authority are returned by default — replaying
        the matcher's own previous guesses would freeze in whatever it got wrong.
        """
        query = "SELECT * FROM decisions"
        if human_only:
            query += " WHERE decided_by IN ('user', 'verify')"
        out: Dict[str, Tuple[Verdict, str]] = {}
        for row in self._conn.execute(query):
            decision = self._to_decision(row)
            out[decision.row_id] = (decision.verdict, decision.note or "")
        return out

    def record(
        self,
        row_id: str,
        verdict: Verdict,
        *,
        by: str = DECIDED_USER,
        reason_code: Optional[str] = None,
        matched_path: Optional[str] = None,
        note: Optional[str] = None,
    ) -> bool:
        """Store a decision. Returns False when a stronger one already stands."""
        existing = self.decision_for(row_id)
        if existing and _AUTHORITY.get(by, 0) < _AUTHORITY.get(existing.decided_by, 0):
            return False
        self._conn.execute(
            "INSERT INTO decisions (row_id, verdict, decided_by, reason_code,"
            " matched_path, note, decided_at) VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(row_id) DO UPDATE SET verdict=excluded.verdict,"
            "   decided_by=excluded.decided_by, reason_code=excluded.reason_code,"
            "   matched_path=excluded.matched_path, note=excluded.note,"
            "   decided_at=excluded.decided_at",
            (row_id, verdict.value, by, reason_code, matched_path, note, self._clock()),
        )
        self._conn.commit()
        return True

    def record_many(self, updates: Iterable[Tuple[str, Verdict]], *, by: str = DECIDED_USER,
                    note: Optional[str] = None) -> int:
        """Apply a bulk triage action as one transaction."""
        count = 0
        for row_id, verdict in updates:
            if self.record(row_id, verdict, by=by, note=note):
                count += 1
        return count

    def forget(self, row_id: str) -> None:
        self._conn.execute("DELETE FROM decisions WHERE row_id = ?", (row_id,))
        self._conn.commit()

    # ── export / verification bookkeeping ────────────────────────────────

    def mark_exported(self, row_ids: Sequence[str]) -> None:
        now = self._clock()
        for row_id in row_ids:
            self._conn.execute(
                "INSERT INTO pending (row_id, exported_at) VALUES (?, ?)"
                " ON CONFLICT(row_id) DO UPDATE SET exported_at=excluded.exported_at,"
                "   verified_at=NULL, outcome=NULL, arrived_path=NULL",
                (row_id, now),
            )
        self._conn.commit()

    def outstanding(self) -> List[str]:
        """Rows exported but not yet verified — the second pass's worklist."""
        return [
            row["row_id"]
            for row in self._conn.execute(
                "SELECT row_id FROM pending WHERE verified_at IS NULL"
            )
        ]

    def mark_verified(self, row_id: str, outcome: str, path: Optional[str] = None) -> None:
        self._conn.execute(
            "UPDATE pending SET verified_at = ?, outcome = ?, arrived_path = ?"
            " WHERE row_id = ?",
            (self._clock(), outcome, path, row_id),
        )
        self._conn.commit()

    def reopen(self, row_id: str, note: str) -> None:
        """Put a row back on the missing list.

        Used when a download turns out to be the wrong recording: the row must
        stop counting as satisfied, or it is never asked about again.
        """
        self.record(row_id, Verdict.MISSING, by=DECIDED_VERIFY, note=note)
        # It leaves the pending queue: it is no longer awaiting verification, it
        # is awaiting a re-download. Re-exporting puts it back.
        self._conn.execute("DELETE FROM pending WHERE row_id = ?", (row_id,))
        self._conn.commit()

    # ── runs ─────────────────────────────────────────────────────────────

    def record_run(self, run_id: str, source_id: str, counts: str,
                   started_at: float, finished_at: float) -> None:
        self._conn.execute(
            "INSERT INTO runs (run_id, source_id, started_at, finished_at, counts)"
            " VALUES (?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET"
            "   finished_at=excluded.finished_at, counts=excluded.counts",
            (run_id, source_id, started_at, finished_at, counts),
        )
        self._conn.commit()

    def last_run_at(self, source_id: str) -> Optional[float]:
        row = self._conn.execute(
            "SELECT last_run_at FROM sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        return row["last_run_at"] if row else None

    # ── identity migration ───────────────────────────────────────────────

    def migrate_row_id(self, old_row_id: str, new_row_id: str) -> bool:
        """Move a decision when a row gains a real id.

        A row first seen through a CSV gets a hashed ``s:`` id; the same track
        read later through a source that supplies a video id gets ``yt:``. Moving
        the decision keeps the user's answer instead of asking again.
        """
        if old_row_id == new_row_id:
            return False
        existing = self.decision_for(old_row_id)
        if existing is None or self.decision_for(new_row_id) is not None:
            return False
        with self._conn:
            self._conn.execute(
                "UPDATE decisions SET row_id = ? WHERE row_id = ?", (new_row_id, old_row_id)
            )
            self._conn.execute(
                "UPDATE pending SET row_id = ? WHERE row_id = ?", (new_row_id, old_row_id)
            )
        return True

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _to_decision(row: sqlite3.Row) -> Decision:
        return Decision(
            row_id=row["row_id"],
            verdict=Verdict(row["verdict"]),
            decided_by=row["decided_by"],
            reason_code=row["reason_code"],
            matched_path=row["matched_path"],
            note=row["note"],
            decided_at=row["decided_at"],
        )
