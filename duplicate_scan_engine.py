from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import librosa
import numpy as np

from near_duplicate_detector import fingerprint_distance, _parse_fp
from simple_duplicate_finder import SUPPORTED_EXTS
import chromaprint_utils

logger = logging.getLogger(__name__)

LogCallback = Callable[[str], None]


@dataclass
class DuplicateScanConfig:
    sample_rate: int = 11025
    max_analysis_sec: float = 120.0
    duration_tolerance_ms: int = 2000
    duration_tolerance_ratio: float = 0.01
    rms_tolerance_db: float | None = 6.0
    centroid_tolerance: float | None = 1500.0
    rolloff_tolerance: float | None = None
    fp_bands: int = 8
    min_band_collisions: int = 2
    fp_distance_threshold: float = 0.2
    chroma_max_offset_frames: int = 12
    chroma_match_threshold: float = 0.82
    chroma_possible_threshold: float = 0.72


@dataclass
class DuplicateScanSummary:
    tracks_total: int
    headers_updated: int
    fingerprints_updated: int
    edges_written: int
    groups_written: int


def _log(log_callback: LogCallback | None, message: str) -> None:
    if log_callback:
        log_callback(message)
    else:
        logger.info(message)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS audio_header (
            track_id TEXT PRIMARY KEY,
            mtime REAL,
            size INTEGER,
            duration_ms INTEGER,
            rms_db REAL,
            centroid_mean REAL,
            rolloff_mean REAL,
            updated_at REAL
        );

        CREATE TABLE IF NOT EXISTS audio_fingerprint (
            track_id TEXT PRIMARY KEY,
            mtime REAL,
            size INTEGER,
            fp_blob BLOB,
            fp_len INTEGER,
            fp_version TEXT
        );

        CREATE TABLE IF NOT EXISTS fp_lsh (
            band_hash TEXT NOT NULL,
            band_index INTEGER NOT NULL,
            track_id TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dup_edges (
            track_id_a TEXT NOT NULL,
            track_id_b TEXT NOT NULL,
            score REAL NOT NULL,
            method TEXT NOT NULL,
            verified_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dup_groups (
            group_id TEXT NOT NULL,
            track_id TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_audio_header_duration ON audio_header(duration_ms);
        CREATE INDEX IF NOT EXISTS idx_audio_header_centroid ON audio_header(centroid_mean);
        CREATE INDEX IF NOT EXISTS idx_fp_lsh_band ON fp_lsh(band_hash, band_index);
        CREATE INDEX IF NOT EXISTS idx_fp_lsh_track ON fp_lsh(track_id);
        CREATE INDEX IF NOT EXISTS idx_dup_edges_pair ON dup_edges(track_id_a, track_id_b);
        CREATE INDEX IF NOT EXISTS idx_dup_groups_track ON dup_groups(track_id);
        """
    )


def _list_audio_files(root: str) -> list[str]:
    paths: list[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTS:
                paths.append(os.path.join(base, name))
    return sorted(paths)


def _compute_audio_header(
    path: str,
    sample_rate: int,
    max_analysis_sec: float,
) -> tuple[int, float, float, float]:
    y, sr = librosa.load(path, sr=sample_rate, mono=True, duration=max_analysis_sec)
    if y.size == 0:
        raise ValueError("empty audio")
    duration_ms = int(round(librosa.get_duration(y=y, sr=sr) * 1000))
    rms = librosa.feature.rms(y=y)[0]
    rms_db = float(librosa.amplitude_to_db(np.mean(rms), ref=1.0))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = float(np.mean(rolloff))
    return duration_ms, rms_db, centroid_mean, rolloff_mean


def _fingerprint_to_bytes(fp: str) -> bytes:
    return fp.encode("utf-8")


def _fingerprint_band_hashes(fp: str, bands: int) -> list[tuple[int, str]]:
    parsed = _parse_fp(fp)
    if not parsed:
        return []
    kind, data = parsed
    if kind == "ints":
        items: Sequence[int] = data  # type: ignore[assignment]
        band_size = max(1, len(items) // bands)
        hashes = []
        for idx in range(bands):
            start = idx * band_size
            if start >= len(items):
                break
            end = len(items) if idx == bands - 1 else start + band_size
            slice_items = items[start:end]
            if not slice_items:
                continue
            payload = ",".join(str(v) for v in slice_items).encode("utf-8")
            hashes.append((idx, hashlib.sha1(payload).hexdigest()))
        return hashes

    blob: bytes = data  # type: ignore[assignment]
    band_size = max(1, len(blob) // bands)
    hashes = []
    for idx in range(bands):
        start = idx * band_size
        if start >= len(blob):
            break
        end = len(blob) if idx == bands - 1 else start + band_size
        payload = blob[start:end]
        if not payload:
            continue
        hashes.append((idx, hashlib.sha1(payload).hexdigest()))
    return hashes


def _chroma_sequence(
    path: str,
    sample_rate: int,
    max_analysis_sec: float,
    hop_length: int = 512,
) -> np.ndarray:
    y, sr = librosa.load(path, sr=sample_rate, mono=True, duration=max_analysis_sec)
    if y.size == 0:
        raise ValueError("empty audio")
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)
    return chroma


def _alignment_score(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    max_offset_frames: int,
) -> float:
    if seq_a.size == 0 or seq_b.size == 0:
        return 0.0
    max_score = 0.0
    for offset in range(-max_offset_frames, max_offset_frames + 1):
        if offset < 0:
            a = seq_a[:, :offset]
            b = seq_b[:, -offset:]
        elif offset > 0:
            a = seq_a[:, offset:]
            b = seq_b[:, :-offset]
        else:
            a = seq_a
            b = seq_b
        if a.shape[1] == 0 or b.shape[1] == 0:
            continue
        n = min(a.shape[1], b.shape[1])
        if n == 0:
            continue
        a_seg = a[:, :n]
        b_seg = b[:, :n]
        score = float(np.mean(np.sum(a_seg * b_seg, axis=0)))
        if score > max_score:
            max_score = score
    return max_score


def _stage1_candidates(
    conn: sqlite3.Connection,
    target_id: str,
    config: DuplicateScanConfig,
) -> set[str]:
    row = conn.execute(
        "SELECT duration_ms, rms_db, centroid_mean, rolloff_mean FROM audio_header WHERE track_id=?",
        (target_id,),
    ).fetchone()
    if not row:
        return set()
    duration_ms, rms_db, centroid_mean, rolloff_mean = row
    if duration_ms is None:
        return set()
    duration_ms = int(duration_ms)
    tolerance = max(
        config.duration_tolerance_ms,
        int(round(config.duration_tolerance_ratio * duration_ms)),
    )

    conditions = ["track_id != ?", "ABS(duration_ms - ?) <= ?"]
    params: list[object] = [target_id, duration_ms, tolerance]
    if config.centroid_tolerance is not None and centroid_mean is not None:
        conditions.append("ABS(centroid_mean - ?) <= ?")
        params.extend([centroid_mean, config.centroid_tolerance])
    if config.rms_tolerance_db is not None and rms_db is not None:
        conditions.append("ABS(rms_db - ?) <= ?")
        params.extend([rms_db, config.rms_tolerance_db])
    if config.rolloff_tolerance is not None and rolloff_mean is not None:
        conditions.append("ABS(rolloff_mean - ?) <= ?")
        params.extend([rolloff_mean, config.rolloff_tolerance])

    query = f"SELECT track_id FROM audio_header WHERE {' AND '.join(conditions)}"
    rows = conn.execute(query, params).fetchall()
    return {r[0] for r in rows}


def _stage2_candidates(
    conn: sqlite3.Connection,
    target_id: str,
    stage1: set[str],
    config: DuplicateScanConfig,
) -> set[str]:
    row = conn.execute(
        "SELECT fp_blob FROM audio_fingerprint WHERE track_id=?",
        (target_id,),
    ).fetchone()
    if not row or row[0] is None:
        return set()
    fp_text = row[0].decode("utf-8")
    bands = _fingerprint_band_hashes(fp_text, config.fp_bands)
    if not bands:
        return set()

    collision_counts: dict[str, int] = {}
    for band_index, band_hash in bands:
        for (track_id,) in conn.execute(
            "SELECT track_id FROM fp_lsh WHERE band_hash=? AND band_index=?",
            (band_hash, band_index),
        ):
            if track_id == target_id:
                continue
            collision_counts[track_id] = collision_counts.get(track_id, 0) + 1

    candidates = {
        track_id
        for track_id, count in collision_counts.items()
        if count >= config.min_band_collisions
    }
    if stage1:
        candidates &= stage1
    return candidates


def _fetch_fingerprint(conn: sqlite3.Connection, track_id: str) -> str | None:
    row = conn.execute(
        "SELECT fp_blob FROM audio_fingerprint WHERE track_id=?",
        (track_id,),
    ).fetchone()
    if not row or row[0] is None:
        return None
    return row[0].decode("utf-8")


def _cleanup_missing(conn: sqlite3.Connection, track_ids: set[str]) -> None:
    existing = {row[0] for row in conn.execute("SELECT track_id FROM audio_header")}
    missing = existing - track_ids
    if not missing:
        return
    for track_id in missing:
        conn.execute("DELETE FROM audio_header WHERE track_id=?", (track_id,))
        conn.execute("DELETE FROM audio_fingerprint WHERE track_id=?", (track_id,))
        conn.execute("DELETE FROM fp_lsh WHERE track_id=?", (track_id,))


def _update_headers(
    conn: sqlite3.Connection,
    paths: Iterable[str],
    config: DuplicateScanConfig,
    log_callback: LogCallback | None,
) -> int:
    updated = 0
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        row = conn.execute(
            "SELECT mtime, size FROM audio_header WHERE track_id=?",
            (path,),
        ).fetchone()
        if row and row[0] == stat.st_mtime and row[1] == stat.st_size:
            continue
        try:
            duration_ms, rms_db, centroid_mean, rolloff_mean = _compute_audio_header(
                path, config.sample_rate, config.max_analysis_sec
            )
        except Exception as exc:
            _log(log_callback, f"⚠ Skipped header for {path}: {exc}")
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO audio_header (
                track_id, mtime, size, duration_ms, rms_db, centroid_mean, rolloff_mean, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                path,
                stat.st_mtime,
                stat.st_size,
                duration_ms,
                rms_db,
                centroid_mean,
                rolloff_mean,
                time.time(),
            ),
        )
        updated += 1
    return updated


def _update_fingerprints(
    conn: sqlite3.Connection,
    paths: Iterable[str],
    config: DuplicateScanConfig,
    log_callback: LogCallback | None,
) -> int:
    updated = 0
    for path in paths:
        try:
            stat = os.stat(path)
        except OSError:
            continue
        row = conn.execute(
            "SELECT mtime, size FROM audio_fingerprint WHERE track_id=?",
            (path,),
        ).fetchone()
        if row and row[0] == stat.st_mtime and row[1] == stat.st_size:
            continue
        try:
            fp = chromaprint_utils.fingerprint_fpcalc(
                path,
                trim=True,
                start_sec=0.0,
                duration_sec=config.max_analysis_sec,
            )
        except chromaprint_utils.FingerprintError as exc:
            _log(log_callback, f"⚠ Fingerprint failed for {path}: {exc}")
            continue
        if not fp:
            _log(log_callback, f"⚠ No fingerprint for {path}")
            continue
        fp_blob = _fingerprint_to_bytes(fp)
        fp_len = len(fp.split())
        conn.execute(
            """
            INSERT OR REPLACE INTO audio_fingerprint (
                track_id, mtime, size, fp_blob, fp_len, fp_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (path, stat.st_mtime, stat.st_size, fp_blob, fp_len, "chromaprint-v1"),
        )
        conn.execute("DELETE FROM fp_lsh WHERE track_id=?", (path,))
        for band_index, band_hash in _fingerprint_band_hashes(fp, config.fp_bands):
            conn.execute(
                "INSERT INTO fp_lsh (band_hash, band_index, track_id) VALUES (?, ?, ?)",
                (band_hash, band_index, path),
            )
        updated += 1
    return updated


def _write_dup_groups(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT track_id_a, track_id_b FROM dup_edges ORDER BY track_id_a"
    ).fetchall()
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in rows:
        union(a, b)

    groups: dict[str, list[str]] = {}
    for track_id in parent:
        root = find(track_id)
        groups.setdefault(root, []).append(track_id)

    conn.execute("DELETE FROM dup_groups")
    group_id = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        group_id += 1
        gid = f"group-{group_id:04d}"
        for track_id in members:
            conn.execute(
                "INSERT INTO dup_groups (group_id, track_id) VALUES (?, ?)",
                (gid, track_id),
            )
    return group_id


def run_duplicate_scan(
    library_path: str,
    db_path: str,
    config: DuplicateScanConfig | None = None,
    log_callback: LogCallback | None = None,
) -> DuplicateScanSummary:
    config = config or DuplicateScanConfig()
    paths = _list_audio_files(library_path)
    _log(log_callback, f"Found {len(paths)} audio files.")

    with _connect(db_path) as conn:
        ensure_schema(conn)
        _cleanup_missing(conn, set(paths))
        _log(log_callback, "Stage 1: updating audio headers...")
        headers_updated = _update_headers(conn, paths, config, log_callback)
        _log(log_callback, f"Header updates: {headers_updated}")

        _log(log_callback, "Stage 2: updating fingerprints + LSH...")
        fingerprints_updated = _update_fingerprints(conn, paths, config, log_callback)
        _log(log_callback, f"Fingerprints updated: {fingerprints_updated}")

        _log(log_callback, "Stage 3: scanning candidates...")
        conn.execute("DELETE FROM dup_edges")
        chroma_cache: dict[str, np.ndarray] = {}
        edges_written = 0

        for idx, target_id in enumerate(paths):
            stage1 = _stage1_candidates(conn, target_id, config)
            if not stage1:
                continue
            stage2 = _stage2_candidates(conn, target_id, stage1, config)
            if not stage2:
                continue
            fp_target = _fetch_fingerprint(conn, target_id)
            if not fp_target:
                continue
            for cand_id in sorted(stage2):
                if cand_id <= target_id:
                    continue
                fp_cand = _fetch_fingerprint(conn, cand_id)
                if not fp_cand:
                    continue
                dist = fingerprint_distance(fp_target, fp_cand)
                if dist > config.fp_distance_threshold:
                    continue
                try:
                    seq_a = chroma_cache.get(target_id)
                    if seq_a is None:
                        seq_a = _chroma_sequence(
                            target_id, config.sample_rate, config.max_analysis_sec
                        )
                        chroma_cache[target_id] = seq_a
                    seq_b = chroma_cache.get(cand_id)
                    if seq_b is None:
                        seq_b = _chroma_sequence(
                            cand_id, config.sample_rate, config.max_analysis_sec
                        )
                        chroma_cache[cand_id] = seq_b
                    score = _alignment_score(
                        seq_a, seq_b, config.chroma_max_offset_frames
                    )
                except Exception as exc:
                    _log(log_callback, f"⚠ Stage 3 failed for {cand_id}: {exc}")
                    continue

                if score >= config.chroma_match_threshold:
                    verdict = "match"
                elif score >= config.chroma_possible_threshold:
                    verdict = "possible"
                else:
                    continue

                conn.execute(
                    """
                    INSERT INTO dup_edges (track_id_a, track_id_b, score, method, verified_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (target_id, cand_id, score, verdict, time.time()),
                )
                edges_written += 1
            if (idx + 1) % 50 == 0:
                _log(log_callback, f"Scanned {idx + 1}/{len(paths)} tracks...")

        groups_written = _write_dup_groups(conn)
        _log(log_callback, f"Duplicate groups: {groups_written}")

    return DuplicateScanSummary(
        tracks_total=len(paths),
        headers_updated=headers_updated,
        fingerprints_updated=fingerprints_updated,
        edges_written=edges_written,
        groups_written=groups_written,
    )
