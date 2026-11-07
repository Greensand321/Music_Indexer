import os
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional
from .utils.path_helpers import ensure_long_path

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


# In-process memo of recent lookups. It used to be ``@lru_cache`` on
# ``get_fingerprint`` itself, keyed on the path (and callbacks) only, so a file
# edited mid-process kept its old fingerprint. Entries are now keyed on the
# file's (mtime, size) as well, exactly like the SQLite rows.
_MEMO_MAX = 128
_memo: "OrderedDict[tuple[str, str, float, int], str]" = OrderedDict()
_memo_lock = threading.Lock()


def _memo_get(db_path: str, path: str, mtime: float, size: int) -> Optional[str]:
    key = (db_path, path, mtime, int(size))
    with _memo_lock:
        fp = _memo.get(key)
        if fp is not None:
            _memo.move_to_end(key)
        return fp


def _memo_put(db_path: str, path: str, mtime: float, size: int, fp: str) -> None:
    key = (db_path, path, mtime, int(size))
    with _memo_lock:
        _memo[key] = fp
        _memo.move_to_end(key)
        while len(_memo) > _MEMO_MAX:
            _memo.popitem(last=False)


def _memo_clear() -> None:
    with _memo_lock:
        _memo.clear()


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
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    except OSError as e:
        log_callback(f"! Could not stat {path}: {e}")
        return None
    fp = _memo_get(db_path, path, mtime, size)
    if fp is not None:
        _dlog("FP", f"cache hit (memory) {path}", log_callback)
        return fp
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = _ensure_db(db_path)
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
        if fp:
            _dlog("FP", f"fingerprint={fp} prefix={fp[:16]}", log_callback)
            _memo_put(db_path, path, mtime, size, fp)
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
        _memo_put(db_path, path, mtime, size, fp_hash)
    conn.close()
    return fp_hash


# ``get_fingerprint.cache_clear()`` existed while it was an ``lru_cache``.
get_fingerprint.cache_clear = _memo_clear  # type: ignore[attr-defined]


def flush_cache(db_path: str) -> None:
    _memo_clear()
    if not os.path.exists(db_path):
        return
    try:
        os.remove(db_path)
    except Exception:
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS fingerprints")
        conn.commit()
        conn.close()
