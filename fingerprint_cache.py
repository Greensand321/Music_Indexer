import os
import sqlite3
import time
from typing import Callable, Optional
from functools import lru_cache
from utils.path_helpers import ensure_long_path

verbose: bool = True


def _dlog(label: str, msg: str, cb: Optional[Callable[[str], None]] = None) -> None:
    if not verbose:
        return
    ts = time.strftime("%H:%M:%S")
    line = f"{ts} [{label}] {msg}"
    if cb:
        cb(line)
    else:
        print(line)


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

    # --- Handle schema upgrades -------------------------------------------
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(fingerprints)")
    }
    if "mtime" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN mtime REAL")
    if "duration" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN duration INT")
    if "fingerprint" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN fingerprint TEXT")
    conn.commit()

    return conn


@lru_cache(maxsize=128)
def get_fingerprint(
    path: str,
    db_path: str,
    compute_func: Callable[[str], tuple[int | None, str | None]],
    log_callback: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Return fingerprint for path using cache; compute if missing."""
    if log_callback is None:
        log_callback = lambda msg: None

    path = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = _ensure_db(db_path)
    try:
        mtime = os.path.getmtime(path)
    except OSError as e:
        log_callback(f"! Could not stat {path}: {e}")
        conn.close()
        return None
    row = conn.execute(
        "SELECT mtime, fingerprint FROM fingerprints WHERE path=?",
        (path,),
    ).fetchone()
    if row and abs(row[0] - mtime) < 1e-6:
        fp = row[1]
        _dlog("FP", f"cache hit {path}", log_callback)
        conn.close()
        if isinstance(fp, (bytes, bytearray)):
            try:
                fp = fp.decode("utf-8")
            except Exception:
                fp = fp.decode("latin1", errors="ignore")
        _dlog("FP", f"fingerprint={fp} prefix={fp[:16]}", log_callback)
        return fp

    _dlog("FP", f"cache miss {path}", log_callback)
    duration, fp_hash = compute_func(path)
    if fp_hash is not None:
        _dlog("FP", f"computed fingerprint {fp_hash} prefix={fp_hash[:16]}", log_callback)
        conn.execute(
            "INSERT OR REPLACE INTO fingerprints (path, mtime, duration, fingerprint) VALUES (?, ?, ?, ?)",
            (path, mtime, duration, fp_hash),
        )
        conn.commit()
    conn.close()
    return fp_hash


def flush_cache(db_path: str) -> None:
    get_fingerprint.cache_clear()
    if not os.path.exists(db_path):
        return
    try:
        os.remove(db_path)
    except Exception:
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS fingerprints")
        conn.commit()
        conn.close()

