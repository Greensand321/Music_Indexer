import os
import sqlite3
from typing import Callable, Optional


def _ensure_db(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
            path TEXT PRIMARY KEY,
            mtime REAL,
            duration INT,
            fingerprint TEXT
        );
        """
    )
    return conn


def get_fingerprint(
    path: str,
    db_path: str,
    compute_func: Callable[[str], tuple[int | None, str | None]]
) -> Optional[str]:
    """Return fingerprint for path using cache; compute if missing."""
    conn = _ensure_db(db_path)
    mtime = os.path.getmtime(path)
    row = conn.execute(
        "SELECT mtime, fingerprint FROM fingerprints WHERE path=?",
        (path,),
    ).fetchone()
    if row and abs(row[0] - mtime) < 1e-6:
        fp = row[1]
        conn.close()
        if isinstance(fp, (bytes, bytearray)):
            try:
                return fp.decode("utf-8")
            except Exception:
                return fp.decode("latin1", errors="ignore")
        return fp

    duration, fp_hash = compute_func(path)
    if fp_hash is not None:
        conn.execute(
            "INSERT OR REPLACE INTO fingerprints (path, mtime, duration, fingerprint) VALUES (?, ?, ?, ?)",
            (path, mtime, duration, fp_hash),
        )
        conn.commit()
    conn.close()
    return fp_hash


def flush_cache(db_path: str) -> None:
    if not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS fingerprints")
    conn.commit()
    conn.close()
