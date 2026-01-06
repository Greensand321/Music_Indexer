import os
import sqlite3
import time
from typing import Callable, Dict, Optional
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
            size INTEGER,
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
    if "size" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN size INTEGER")
    if "duration" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN duration INT")
    if "fingerprint" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN fingerprint TEXT")
    conn.commit()

    return conn


def get_fingerprint(
    path: str,
    db_path: str,
    compute_func: Callable[[str], tuple[int | None, str | None]],
    log_callback: Optional[Callable[[str], None]] = None,
    trace: Optional[Dict[str, object]] = None,
) -> Optional[str]:
    """Return fingerprint for path using cache; compute if missing."""
    if log_callback is None:
        log_callback = lambda msg: None
    if trace is None:
        trace = {}

    path = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = _ensure_db(db_path)
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    except OSError as e:
        log_callback(f"! Could not stat {path}: {e}")
        trace["source"] = "stat_error"
        trace["error"] = str(e)
        conn.close()
        return None
    row = conn.execute(
        "SELECT mtime, size, fingerprint FROM fingerprints WHERE path=?",
        (path,),
    ).fetchone()
    cached_mtime = row[0] if row else None
    cached_size = row[1] if row else None
    if row and abs(cached_mtime - mtime) < 1e-6 and int(cached_size or 0) == int(size):
        fp = row[2]
        _dlog("FP", f"cache hit {path}", log_callback)
        conn.close()
        if isinstance(fp, (bytes, bytearray)):
            try:
                fp = fp.decode("utf-8")
            except Exception:
                fp = fp.decode("latin1", errors="ignore")
        _dlog("FP", f"fingerprint={fp} prefix={fp[:16]}", log_callback)
        trace["source"] = "cache"
        trace["error"] = ""
        return fp
    if row:
        _dlog(
            "FP",
            f"cache invalidated {path} stored_mtime={cached_mtime} stored_size={cached_size} new_mtime={mtime} new_size={size}",
            log_callback,
        )

    _dlog("FP", f"cache miss {path}", log_callback)
    duration, fp_hash = compute_func(path)
    if fp_hash is not None:
        _dlog("FP", f"computed fingerprint {fp_hash} prefix={fp_hash[:16]}", log_callback)
        conn.execute(
            "INSERT OR REPLACE INTO fingerprints (path, mtime, size, duration, fingerprint) VALUES (?, ?, ?, ?, ?)",
            (path, mtime, size, duration, fp_hash),
        )
        conn.commit()
    conn.close()
    if fp_hash:
        trace["source"] = "computed"
        trace["error"] = ""
    else:
        trace["source"] = "missing"
        trace["error"] = "fingerprint unavailable"
    return fp_hash


def get_cached_fingerprint(
    path: str,
    db_path: str,
    log_callback: Optional[Callable[[str], None]] = None,
    trace: Optional[Dict[str, object]] = None,
    *,
    retries: int = 3,
    retry_delay: float = 0.05,
) -> Optional[str]:
    """Return cached fingerprint for path without computing new fingerprints."""
    if log_callback is None:
        log_callback = lambda msg: None
    if trace is None:
        trace = {}

    path = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    for attempt in range(retries):
        conn: sqlite3.Connection | None = None
        try:
            conn = _ensure_db(db_path)
            try:
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
            except OSError as e:
                log_callback(f"! Could not stat {path}: {e}")
                trace["source"] = "stat_error"
                trace["error"] = str(e)
                return None
            row = conn.execute(
                "SELECT mtime, size, fingerprint FROM fingerprints WHERE path=?",
                (path,),
            ).fetchone()
            cached_mtime = row[0] if row else None
            cached_size = row[1] if row else None
            if row and abs(cached_mtime - mtime) < 1e-6 and int(cached_size or 0) == int(size):
                fp = row[2]
                _dlog("FP", f"cache hit {path}", log_callback)
                if isinstance(fp, (bytes, bytearray)):
                    try:
                        fp = fp.decode("utf-8")
                    except Exception:
                        fp = fp.decode("latin1", errors="ignore")
                _dlog("FP", f"fingerprint={fp} prefix={fp[:16]}", log_callback)
                trace["source"] = "cache"
                trace["error"] = ""
                return fp
            trace["source"] = "missing"
            trace["error"] = ""
            return None
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or attempt >= retries - 1:
                log_callback(f"! Fingerprint cache read failed: {e}")
                trace["source"] = "cache_error"
                trace["error"] = str(e)
                return None
            time.sleep(retry_delay)
        finally:
            if conn is not None:
                conn.close()
    return None


def store_fingerprint(
    path: str,
    db_path: str,
    duration: int | None,
    fingerprint: str | None,
    log_callback: Optional[Callable[[str], None]] = None,
    *,
    retries: int = 3,
    retry_delay: float = 0.05,
) -> bool:
    """Persist a fingerprint in the cache without computing it."""
    if fingerprint is None:
        return False
    if log_callback is None:
        log_callback = lambda msg: None

    path = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    for attempt in range(retries):
        conn: sqlite3.Connection | None = None
        try:
            conn = _ensure_db(db_path)
            try:
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
            except OSError as e:
                log_callback(f"! Could not stat {path}: {e}")
                return False
            conn.execute(
                "INSERT OR REPLACE INTO fingerprints (path, mtime, size, duration, fingerprint) VALUES (?, ?, ?, ?, ?)",
                (path, mtime, size, duration, fingerprint),
            )
            conn.commit()
            return True
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or attempt >= retries - 1:
                log_callback(f"! Fingerprint cache write failed: {e}")
                return False
            time.sleep(retry_delay)
        finally:
            if conn is not None:
                conn.close()
    return False


def flush_cache(db_path: str) -> None:
    if not os.path.exists(db_path):
        return
    try:
        os.remove(db_path)
    except Exception:
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS fingerprints")
        conn.commit()
        conn.close()
