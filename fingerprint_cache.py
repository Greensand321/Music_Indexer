import json
import os
import queue
import sqlite3
import threading
import time
from collections import OrderedDict
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

# In-process memo of fingerprints this process has computed, stored or read,
# keyed by (db_path, path) and validated against the file's mtime/size on every
# lookup.  Writes to SQLite go through the batching ``FingerprintWriter`` and
# may not be on disk yet, so without this a second ``get_fingerprint`` call for
# the same file moments later would miss the cache and recompute the (expensive)
# fingerprint.  It also lets repeat lookups skip opening SQLite altogether.
_MEMO_MAXSIZE = 4096
_memo_lock = threading.Lock()
_memo: "OrderedDict[tuple[str, str], tuple[float, int, str]]" = OrderedDict()


def _memo_get(db_path: str, path: str, mtime: float, size: int) -> str | None:
    key = (db_path, path)
    with _memo_lock:
        entry = _memo.get(key)
        if entry is None:
            return None
        cached_mtime, cached_size, fp = entry
        if abs(cached_mtime - mtime) >= 1e-6 or int(cached_size) != int(size):
            del _memo[key]
            return None
        _memo.move_to_end(key)
        return fp


def _memo_put(db_path: str, path: str, mtime: float, size: int, fp: str) -> None:
    key = (db_path, path)
    with _memo_lock:
        _memo[key] = (mtime, int(size), fp)
        _memo.move_to_end(key)
        while len(_memo) > _MEMO_MAXSIZE:
            _memo.popitem(last=False)


def _memo_clear(db_path: str | None = None) -> None:
    with _memo_lock:
        if db_path is None:
            _memo.clear()
            return
        for key in [k for k in _memo if k[0] == db_path]:
            del _memo[key]


def _is_corruption(exc: BaseException) -> bool:
    """A damaged/non-SQLite file — not a busy one ("database is locked")."""
    return isinstance(exc, sqlite3.DatabaseError) and not isinstance(exc, sqlite3.OperationalError)


def _set_aside_corrupt_db(db_path: str, exc: BaseException) -> str | None:
    """Rename a corrupt cache (and its -wal/-shm files) out of the way.

    The cache only holds recomputable fingerprints, but it is renamed rather
    than deleted so nothing is destroyed. Returns the new name, or None.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = f"{db_path}.corrupt-{stamp}"
    counter = 1
    while os.path.exists(target):
        target = f"{db_path}.corrupt-{stamp}-{counter}"
        counter += 1
    try:
        os.replace(db_path, target)
    except OSError:
        return None
    for suffix in ("-wal", "-shm"):
        try:
            os.replace(db_path + suffix, target + suffix)
        except OSError:
            pass
    _memo_clear(db_path)
    _dlog("FP", f"fingerprint cache {db_path} was unreadable ({exc}); moved it to {target} and started a new one")
    return target


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
            channels INT,
            codec TEXT,
            container TEXT,
            tags_json TEXT,
            artwork_json TEXT,
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
    if "channels" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN channels INT")
    if "codec" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN codec TEXT")
    if "container" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN container TEXT")
    if "tags_json" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN tags_json TEXT")
    if "artwork_json" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN artwork_json TEXT")
    if "normalized_artist" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN normalized_artist TEXT")
    if "normalized_title" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN normalized_title TEXT")
    if "normalized_album" not in cols:
        conn.execute("ALTER TABLE fingerprints ADD COLUMN normalized_album TEXT")
    conn.commit()


_INSERT_SQL = """
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
        channels,
        codec,
        container,
        tags_json,
        artwork_json,
        normalized_artist,
        normalized_title,
        normalized_album
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


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
        # Open the connection and create the schema synchronously, so that once
        # a writer exists the table exists too.  Doing this inside the worker
        # thread left a window where the DB file existed without a
        # ``fingerprints`` table, and readers failed with "no such table".
        # It also surfaces an unopenable DB to the caller instead of silently
        # killing the worker thread (which made ``flush()`` wait forever).
        self._conn = _connect_initialized(db_path, check_same_thread=False)
        self._state_lock = threading.Lock()
        self._closed = False
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
        channels: int | None = None,
        codec: str | None = None,
        container: str | None = None,
        tags_json: str | None = None,
        artwork_json: str | None = None,
        normalized_artist: str | None = None,
        normalized_title: str | None = None,
        normalized_album: str | None = None,
    ) -> None:
        row = (
            path,
            mtime,
            size,
            duration,
            fingerprint,
            ext,
            bitrate,
            sample_rate,
            bit_depth,
            channels,
            codec,
            container,
            tags_json,
            artwork_json,
            normalized_artist,
            normalized_title,
            normalized_album,
        )
        with self._state_lock:
            if not self._closed and self._thread.is_alive():
                self._queue.put(("write", row))
                return
        # This writer was shut down (another cache became current) or its
        # thread died: write the row directly instead of leaving it in a
        # queue nobody reads any more.
        conn = sqlite3.connect(self._db_path)
        try:
            self._flush_pending(conn, [row])
        finally:
            conn.close()

    def is_alive(self) -> bool:
        return self._thread.is_alive() and not self._closed

    def flush(self, timeout: float | None = None) -> None:
        if not self._thread.is_alive():
            return  # nothing will ever answer; waiting with timeout=None hung forever
        event = threading.Event()
        self._queue.put(("flush", event))
        event.wait(timeout=timeout)

    def shutdown(self) -> None:
        with self._state_lock:
            self._closed = True
        event = threading.Event()
        self._queue.put(("shutdown", event))
        self._thread.join()
        event.wait(timeout=1.0)

    def _run(self) -> None:
        conn: sqlite3.Connection | None = self._conn
        try:
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
                    int | None,
                    str | None,
                    str | None,
                    str | None,
                    str | None,
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
                    # Always restart the interval: without this an idle
                    # writer called get(timeout=0) in a tight loop forever
                    # (a 100% CPU thread that starved the GUI of the GIL).
                    last_flush = time.monotonic()
        except Exception as exc:  # pragma: no cover - _flush_pending already contains sqlite errors
            _dlog("FP", f"fingerprint cache writer stopped: {exc}")
        finally:
            with self._state_lock:
                self._closed = True
            # Write what is still queued, and release anybody waiting on a
            # flush that will now never run.
            leftovers = []
            while True:
                try:
                    kind, payload = self._queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "write":
                    leftovers.append(payload)
                elif isinstance(payload, threading.Event):
                    payload.set()
            if leftovers and conn is not None:
                try:
                    self._flush_pending(conn, leftovers)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - best effort
                    pass
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
                int | None,
                str | None,
                str | None,
                str | None,
                str | None,
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
                conn.executemany(_INSERT_SQL, pending)
                conn.commit()
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt >= 2:
                    _dlog("FP", f"cache write failed: {exc}")
                    return
                time.sleep(0.05)
            except sqlite3.Error as exc:
                # One row sqlite cannot bind (or a damaged file) used to raise
                # out of here and kill the writer thread: every later write
                # was silently lost and flush(timeout=None) hung forever.
                # Write the rows one by one, skipping only the bad ones.
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                for row in pending:
                    try:
                        conn.execute(_INSERT_SQL, row)
                    except sqlite3.Error as row_exc:
                        _dlog("FP", f"cache write skipped for {row[0]!r}: {row_exc} ({exc})")
                try:
                    conn.commit()
                except sqlite3.Error as commit_exc:
                    _dlog("FP", f"cache write failed: {commit_exc}")
                return

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
                int | None,
                str | None,
                str | None,
                str | None,
                str | None,
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


def _connect_initialized(db_path: str, **kwargs) -> sqlite3.Connection:
    """Open ``db_path`` with the schema in place; a corrupt file is set aside once."""
    for attempt in range(2):
        conn = sqlite3.connect(db_path, **kwargs)
        try:
            _initialize_db(conn)
            return conn
        except Exception as exc:
            conn.close()
            if attempt == 0 and _is_corruption(exc) and _set_aside_corrupt_db(db_path, exc):
                continue
            raise
    raise AssertionError("unreachable")  # pragma: no cover


def _get_writer(db_path: str) -> FingerprintWriter:
    global _writer, _writer_db_path
    with _writer_lock:
        if _writer is None or _writer_db_path != db_path or not _writer.is_alive():
            if _writer is not None:
                _writer.shutdown()
                _writer = None
                _writer_db_path = None
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            # Only record the writer once it was created successfully, so a
            # failed open is retried next time rather than leaving a shut-down
            # writer registered for this path.
            _writer = FingerprintWriter(db_path)
            _writer_db_path = db_path
        return _writer


def _open_readonly_connection(db_path: str) -> sqlite3.Connection | None:
    """Open ``db_path`` for reading, or return ``None`` if there is no cache yet.

    A DB file can exist without the ``fingerprints`` table (an empty file, or
    one created by ``sqlite3.connect`` elsewhere before the schema was
    written).  Querying it would raise ``no such table``; it is simply an
    empty cache, so it is reported the same way as a missing file.
    """
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA query_only = ON")
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fingerprints'"
        ).fetchone()
    except Exception:
        conn.close()
        raise
    if not has_table:
        conn.close()
        return None
    return conn


def _row_is_current(cached_mtime: object, cached_size: object, mtime: float, size: int) -> bool:
    """True when a cache row was written for the file as it is now.

    Rows from before the ``mtime``/``size`` columns existed have NULLs there;
    comparing them raised ``TypeError`` out of the cached-lookup helpers.
    """
    if cached_mtime is None or cached_size is None:
        return False
    try:
        return abs(float(cached_mtime) - mtime) < 1e-6 and int(cached_size) == int(size)
    except (TypeError, ValueError):
        return False


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
    """Ensure the fingerprint cache schema exists and is up to date.

    A corrupt cache file (``file is not a database``, ``disk image is
    malformed``) used to make every duplicate scan fail; it is now renamed to
    ``<name>.corrupt-<time>`` and a fresh cache is started.
    """
    if not os.path.exists(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _connect_initialized(db_path).close()


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

    # The cache key and every OS call use the long-path form; the caller's
    # own path is what compute_func, logs and messages see (no ``\\?\``).
    key = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        mtime = os.path.getmtime(key)
        size = os.path.getsize(key)
    except OSError as e:
        log_callback(f"! Could not stat {path}: {e}")
        trace["source"] = "stat_error"
        trace["error"] = str(e)
        return None

    fp = _memo_get(db_path, key, mtime, size)
    if fp is not None:
        _dlog("FP", f"cache hit (memory) {path}", log_callback)
        trace["source"] = "cache"
        trace["error"] = ""
        return fp

    row = None
    conn: sqlite3.Connection | None = None
    try:
        conn = _open_readonly_connection(db_path)
        if conn is not None:
            row = conn.execute(
                "SELECT mtime, size, fingerprint FROM fingerprints WHERE path=?",
                (key,),
            ).fetchone()
    except sqlite3.Error as e:
        # An unreadable cache is a cache miss, not a reason to fail the lookup.
        log_callback(f"! Fingerprint cache read failed: {e}")
        row = None
    finally:
        if conn is not None:
            conn.close()

    cached_mtime = row[0] if row else None
    cached_size = row[1] if row else None
    if row and _row_is_current(cached_mtime, cached_size, mtime, size):
        _dlog("FP", f"cache hit {path}", log_callback)
        fp = _decode_fingerprint(row[2])
        if fp is None:
            trace["source"] = "missing"
            trace["error"] = "fingerprint unavailable"
            return None
        _dlog(
            "FP",
            f"fingerprint_prefix={fp[:16]} len={len(fp)}",
            log_callback,
        )
        _memo_put(db_path, key, mtime, size, fp)
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
        _memo_put(db_path, key, mtime, size, fp_hash)
        try:
            _get_writer(db_path).enqueue(key, mtime, size, duration, fp_hash)
        except (sqlite3.Error, OSError) as e:
            log_callback(f"! Fingerprint cache write failed: {e}")
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

    # The cache key and every OS call use the long-path form; the caller's
    # own path is what compute_func, logs and messages see (no ``\\?\``).
    key = ensure_long_path(path)
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
                mtime = os.path.getmtime(key)
                size = os.path.getsize(key)
            except OSError as e:
                log_callback(f"! Could not stat {path}: {e}")
                trace["source"] = "stat_error"
                trace["error"] = str(e)
                return None
            row = conn.execute(
                "SELECT mtime, size, fingerprint FROM fingerprints WHERE path=?",
                (key,),
            ).fetchone()
            if row and _row_is_current(row[0], row[1], mtime, size):
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
        except sqlite3.Error as e:
            # A corrupt or half-upgraded cache is a cache miss, never a crash.
            if not isinstance(e, sqlite3.OperationalError) or "locked" not in str(e).lower() or attempt >= retries - 1:
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
) -> tuple[str | None, dict[str, object] | None]:
    """Return cached fingerprint and metadata for path without computing new fingerprints."""
    if log_callback is None:
        log_callback = lambda msg: None
    if trace is None:
        trace = {}

    def _load_json(payload: str | None) -> object | None:
        if not payload:
            return None
        try:
            return json.loads(payload)
        except Exception:
            return None

    # The cache key and every OS call use the long-path form; the caller's
    # own path is what compute_func, logs and messages see (no ``\\?\``).
    key = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    for attempt in range(retries):
        conn: sqlite3.Connection | None = None
        try:
            conn = _open_readonly_connection(db_path)
            if conn is None:
                trace["source"] = "missing"
                trace["error"] = ""
                return None, None
            try:
                mtime = os.path.getmtime(key)
                size = os.path.getsize(key)
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
                    channels,
                    codec,
                    container,
                    tags_json,
                    artwork_json,
                    normalized_artist,
                    normalized_title,
                    normalized_album
                FROM fingerprints
                WHERE path=?
                """,
                (key,),
            ).fetchone()
            if row and _row_is_current(row[0], row[1], mtime, size):
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
                    "channels": row[7],
                    "codec": row[8],
                    "container": row[9],
                    "tags": _load_json(row[10]),
                    "artwork": _load_json(row[11]),
                    "normalized_artist": row[12],
                    "normalized_title": row[13],
                    "normalized_album": row[14],
                }
                return fp, metadata
            trace["source"] = "missing"
            trace["error"] = ""
            return None, None
        except sqlite3.Error as e:
            # A corrupt or half-upgraded cache is a cache miss, never a crash.
            if not isinstance(e, sqlite3.OperationalError) or "locked" not in str(e).lower() or attempt >= retries - 1:
                log_callback(f"! Fingerprint cache read failed: {e}")
                trace["source"] = "cache_error"
                trace["error"] = str(e)
                return None, None
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
    channels: int | None = None,
    codec: str | None = None,
    container: str | None = None,
    tags: dict[str, object] | None = None,
    artwork: list[dict[str, object]] | None = None,
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

    # The cache key and every OS call use the long-path form; the caller's
    # own path is what compute_func, logs and messages see (no ``\\?\``).
    key = ensure_long_path(path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    for attempt in range(retries):
        try:
            try:
                mtime = os.path.getmtime(key)
                size = os.path.getsize(key)
            except OSError as e:
                log_callback(f"! Could not stat {path}: {e}")
                return False
            _get_writer(db_path).enqueue(
                key,
                mtime,
                size,
                duration,
                fingerprint,
                ext=ext,
                bitrate=bitrate,
                sample_rate=sample_rate,
                bit_depth=bit_depth,
                channels=channels,
                codec=codec,
                container=container,
                tags_json=json.dumps(tags, default=str) if tags is not None else None,
                artwork_json=json.dumps(artwork, default=str) if artwork is not None else None,
                normalized_artist=normalized_artist,
                normalized_title=normalized_title,
                normalized_album=normalized_album,
            )
            _memo_put(db_path, key, mtime, size, fingerprint)
            if flush:
                flush_fingerprint_writes(db_path)
            return True
        except sqlite3.Error as e:
            if not isinstance(e, sqlite3.OperationalError) or "locked" not in str(e).lower() or attempt >= retries - 1:
                log_callback(f"! Fingerprint cache write failed: {e}")
                return False
            time.sleep(retry_delay)
        except (TypeError, ValueError) as e:  # tags that json cannot encode
            log_callback(f"! Fingerprint cache write failed for {path}: {e}")
            return False
    return False


def flush_fingerprint_writes(db_path: str, timeout: float | None = None) -> None:
    """Force a flush of queued fingerprint writes for immediate persistence."""
    with _writer_lock:
        writer = _writer if _writer_db_path == db_path else None
    if writer is None:
        # Writes are only ever queued on the current writer; a different (or
        # no) writer means nothing is pending for this DB.
        return
    writer.flush(timeout=timeout)


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
    _memo_clear(db_path)
    shutdown_fingerprint_writer()
    if not os.path.exists(db_path):
        return
    try:
        os.remove(db_path)
    except Exception:
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS fingerprints")
        conn.commit()
        conn.close()
