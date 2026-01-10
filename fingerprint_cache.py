import os
import queue
import sqlite3
import threading
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


_writer_lock = threading.Lock()
_writer: "FingerprintWriter | None" = None
_writer_db_path: str | None = None


def _initialize_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
            path TEXT PRIMARY KEY,
            mtime REAL,
            size INTEGER,
            duration INT,
            fingerprint TEXT,
            ext TEXT,
            bitrate INT,
            sample_rate INT,
            bit_depth INT,
            normalized_artist TEXT,
            normalized_title TEXT,
            normalized_album TEXT
        );
        """
    )

    # --- Handle schema upgrades -------------------------------------------
    cols = {row[1] for row in conn.execute("PRAGMA table_info(fingerprints)")}
    if "mtime" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN mtime REAL")
    if "size" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN size INTEGER")
    if "duration" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN duration INT")
    if "fingerprint" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN fingerprint TEXT")
    if "ext" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN ext TEXT")
    if "bitrate" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN bitrate INT")
    if "sample_rate" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN sample_rate INT")
    if "bit_depth" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN bit_depth INT")
    if "normalized_artist" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN normalized_artist TEXT")
    if "normalized_title" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN normalized_title TEXT")
    if "normalized_album" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN normalized_album TEXT")
    conn.commit()


class FingerprintWriter:
    def __init__(
        self,
        db_path: str,
        *,
        batch_size: int = 50,
        flush_interval: float = 1.0,
    ) -> None:
        self._db_path = db_path
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run,
            name="FingerprintCacheWriter",
            daemon=True,
        )
        self._thread.start()

    def enqueue(
        self,
        path: str,
        mtime: float,
        size: int,
        duration: int | None,
        fingerprint: str,
        *,
        ext: str | None = None,
        bitrate: int | None = None,
        sample_rate: int | None = None,
        bit_depth: int | None = None,
        normalized_artist: str | None = None,
        normalized_title: str | None = None,
        normalized_album: str | None = None,
    ) -> None:
        self._queue.put(
            (
                "write",
                (
                    path,
                    mtime,
                    size,
                    duration,
                    fingerprint,
                    ext,
                    bitrate,
                    sample_rate,
                    bit_depth,
                    normalized_artist,
                    normalized_title,
                    normalized_album,
                ),
            )
        )

    def flush(self, timeout: float | None = None) -> None:
        event = threading.Event()
        self._queue.put(("flush", event))
        event.wait(timeout=timeout)

    def shutdown(self) -> None:
        event = threading.Event()
        self._queue.put(("shutdown", event))
        self._thread.join()
        event.wait(timeout=1.0)

    def _run(self) -> None:
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            _initialize_db(conn)
            pending: list[
                tuple[
                    str,
                    float,
                    int,
                    int | None,
                    str,
                    str | None,
                    int | None,
                    int | None,
                    int | None,
                    str | None,
                    str | None,
                    str | None,
                ]
            ] = []
            last_flush = time.monotonic()
            while True:
                timeout = self._flush_interval - (time.monotonic() - last_flush)
                if timeout < 0:
                    timeout = 0
                try:
                    kind, payload = self._queue.get(timeout=timeout)
                except queue.Empty:
                    kind = ""
                    payload = None

                if kind == "write":
                    pending.append(payload)  # type: ignore[arg-type]
                    if len(pending) >= self._batch_size:
                        self._flush_pending(conn, pending)
                        pending.clear()
                        last_flush = time.monotonic()
                elif kind == "flush":
                    self._flush_pending(conn, pending)
                    pending.clear()
                    last_flush = time.monotonic()
                    if isinstance(payload, threading.Event):
                        payload.set()
                elif kind == "shutdown":
                    self._flush_pending(conn, pending)
                    pending.clear()
                    self._drain_remaining(conn)
                    if isinstance(payload, threading.Event):
                        payload.set()
                    break
                elif kind == "":
                    if pending:
                        self._flush_pending(conn, pending)
                        pending.clear()
                        last_flush = time.monotonic()
        finally:
            if conn is not None:
                conn.close()

    def _flush_pending(
        self,
        conn: sqlite3.Connection,
        pending: list[
            tuple[
                str,
                float,
                int,
                int | None,
                str,
                str | None,
                int | None,
                int | None,
                int | None,
                str | None,
                str | None,
                str | None,
            ]
        ],
    ) -> None:
        if not pending:
            return
        for attempt in range(3):
            try:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO fingerprints (
                        path,
                        mtime,
                        size,
                        duration,
                        fingerprint,
                        ext,
                        bitrate,
                        sample_rate,
                        bit_depth,
                        normalized_artist,
                        normalized_title,
                        normalized_album
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    pending,
                )
                conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt >= 2:
                    _dlog("FP", f"cache write failed: {exc}")
                    return
                time.sleep(0.05)

    def _drain_remaining(self, conn: sqlite3.Connection) -> None:
        pending: list[
            tuple[
                str,
                float,
                int,
                int | None,
                str,
                str | None,
                int | None,
                int | None,
                int | None,
                str | None,
                str | None,
                str | None,
            ]
        ] = []
        while True:
            try:
                kind, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "write":
                pending.append(payload)  # type: ignore[arg-type]
            elif kind in {"flush", "shutdown"} and isinstance(payload, threading.Event):
                payload.set()
        if pending:
            self._flush_pending(conn, pending)


def _get_writer(db_path: str) -> FingerprintWriter:
    global _writer, _writer_db_path
    with _writer_lock:
        if _writer is None or _writer_db_path != db_path:
            if _writer is not None:
                _writer.shutdown()
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            _writer_db_path = db_path
            _writer = FingerprintWriter(db_path)
        return _writer


def _open_readonly_connection(db_path: str) -> sqlite3.Connection | None:
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _decode_fingerprint(fp: object) -> str | None:
    if not isinstance(fp, (bytes, bytearray)):
        return fp  # type: ignore[return-value]
    try:
        return fp.decode("utf-8")
    except Exception:
        return fp.decode("latin1", errors="ignore")


def _ensure_db(db_path: str) -> sqlite3.Connection:
    """Legacy helper for callers that need a ready connection."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    _initialize_db(conn)
    return conn


def ensure_fingerprint_cache(db_path: str) -> None:
    """Ensure the fingerprint cache schema exists and is up to date."""
    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        _initialize_db(conn)
    finally:
        conn.close()


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
    conn = _open_readonly_connection(db_path)
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    except OSError as e:
        log_callback(f"! Could not stat {path}: {e}")
        trace["source"] = "stat_error"
        trace["error"] = str(e)
        if conn is not None:
            conn.close()
        return None
    row = None
    if conn is not None:
        row = conn.execute(
            "SELECT mtime, size, fingerprint FROM fingerprints WHERE path=?",
            (path,),
        ).fetchone()
    cached_mtime = row[0] if row else None
    cached_size = row[1] if row else None
    if row and abs(cached_mtime - mtime) < 1e-6 and int(cached_size or 0) == int(size):
        fp = row[2]
        _dlog("FP", f"cache hit {path}", log_callback)
        if conn is not None:
            conn.close()
        fp = _decode_fingerprint(fp)
        if fp is None:
            trace["source"] = "missing"
            trace["error"] = "fingerprint unavailable"
            return None
        _dlog(
            "FP",
            f"fingerprint_prefix={fp[:16]} len={len(fp)}",
            log_callback,
        )
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
        _dlog(
            "FP",
            f"computed fingerprint prefix={fp_hash[:16]} len={len(fp_hash)}",
            log_callback,
        )
        _get_writer(db_path).enqueue(path, mtime, size, duration, fp_hash)
    if conn is not None:
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
            conn = _open_readonly_connection(db_path)
            if conn is None:
                trace["source"] = "missing"
                trace["error"] = ""
                return None
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
                fp = _decode_fingerprint(fp)
                if fp is None:
                    trace["source"] = "missing"
                    trace["error"] = "fingerprint unavailable"
                    return None
                _dlog(
                    "FP",
                    f"fingerprint_prefix={fp[:16]} len={len(fp)}",
                    log_callback,
                )
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


def get_cached_fingerprint_metadata(
    path: str,
    db_path: str,
    log_callback: Optional[Callable[[str], None]] = None,
    trace: Optional[Dict[str, object]] = None,
    *,
    retries: int = 3,
    retry_delay: float = 0.05,
    lock_retry_attempts: int = 0,
    lock_retry_delay: float = 0.1,
    lock_retry_backoff: float = 2.0,
) -> tuple[str | None, dict[str, object] | None]:
    """Return cached fingerprint and metadata for path without computing new fingerprints."""
    if log_callback is None:
        log_callback = lambda msg: None
    if trace is None:
        trace = {}

    path = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    total_attempts = max(retries, 1) + max(lock_retry_attempts, 0)
    lock_delay = max(lock_retry_delay, 0.0)
    for attempt in range(total_attempts):
        conn: sqlite3.Connection | None = None
        try:
            conn = _open_readonly_connection(db_path)
            if conn is None:
                trace["source"] = "missing"
                trace["error"] = ""
                return None, None
            try:
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
            except OSError as e:
                log_callback(f"! Could not stat {path}: {e}")
                trace["source"] = "stat_error"
                trace["error"] = str(e)
                return None, None
            row = conn.execute(
                """
                SELECT
                    mtime,
                    size,
                    fingerprint,
                    ext,
                    bitrate,
                    sample_rate,
                    bit_depth,
                    normalized_artist,
                    normalized_title,
                    normalized_album
                FROM fingerprints
                WHERE path=?
                """,
                (path,),
            ).fetchone()
            cached_mtime = row[0] if row else None
            cached_size = row[1] if row else None
            if row and abs(cached_mtime - mtime) < 1e-6 and int(cached_size or 0) == int(size):
                fp = _decode_fingerprint(row[2])
                _dlog("FP", f"cache hit {path}", log_callback)
                if fp is None:
                    trace["source"] = "missing"
                    trace["error"] = "fingerprint unavailable"
                    return None, None
                _dlog(
                    "FP",
                    f"fingerprint_prefix={fp[:16]} len={len(fp)}",
                    log_callback,
                )
                trace["source"] = "cache"
                trace["error"] = ""
                metadata = {
                    "ext": row[3],
                    "bitrate": row[4],
                    "sample_rate": row[5],
                    "bit_depth": row[6],
                    "normalized_artist": row[7],
                    "normalized_title": row[8],
                    "normalized_album": row[9],
                }
                return fp, metadata
            trace["source"] = "missing"
            trace["error"] = ""
            return None, None
        except sqlite3.OperationalError as e:
            locked = "locked" in str(e).lower()
            if not locked or attempt >= total_attempts - 1:
                log_callback(f"! Fingerprint cache read failed: {e}")
                trace["source"] = "cache_error"
                trace["error"] = str(e)
                return None, None
            if locked and attempt >= retries - 1:
                trace["lock_contention"] = True
                time.sleep(lock_delay)
                lock_delay *= max(lock_retry_backoff, 1.0)
            else:
                time.sleep(retry_delay)
        finally:
            if conn is not None:
                conn.close()
    return None, None


def store_fingerprint(
    path: str,
    db_path: str,
    duration: int | None,
    fingerprint: str | None,
    log_callback: Optional[Callable[[str], None]] = None,
    *,
    ext: str | None = None,
    bitrate: int | None = None,
    sample_rate: int | None = None,
    bit_depth: int | None = None,
    normalized_artist: str | None = None,
    normalized_title: str | None = None,
    normalized_album: str | None = None,
    retries: int = 3,
    retry_delay: float = 0.05,
    flush: bool = False,
) -> bool:
    """Persist a fingerprint in the cache without computing it."""
    if fingerprint is None:
        return False
    if log_callback is None:
        log_callback = lambda msg: None

    path = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    for attempt in range(retries):
        try:
            try:
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
            except OSError as e:
                log_callback(f"! Could not stat {path}: {e}")
                return False
            _get_writer(db_path).enqueue(
                path,
                mtime,
                size,
                duration,
                fingerprint,
                ext=ext,
                bitrate=bitrate,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                normalized_artist=normalized_artist,
                normalized_title=normalized_title,
                normalized_album=normalized_album,
            )
            if flush:
                flush_fingerprint_writes(db_path)
            return True
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower() or attempt >= retries - 1:
                log_callback(f"! Fingerprint cache write failed: {e}")
                return False
            time.sleep(retry_delay)
    return False


def flush_fingerprint_writes(db_path: str, timeout: float | None = None) -> None:
    """Force a flush of queued fingerprint writes for immediate persistence."""
    if not os.path.exists(db_path):
        return
    _get_writer(db_path).flush(timeout=timeout)


def shutdown_fingerprint_writer() -> None:
    """Drain queued fingerprint writes; call during application shutdown (e.g., exit handler)."""
    global _writer, _writer_db_path
    with _writer_lock:
        if _writer is None:
            return
        _writer.shutdown()
        _writer = None
        _writer_db_path = None


def flush_cache(db_path: str) -> None:
    if not os.path.exists(db_path):
        return
    shutdown_fingerprint_writer()
    try:
        os.remove(db_path)
    except Exception:
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS fingerprints")
        conn.commit()
        conn.close()
