import os
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional
from .near_duplicate_detector import _fp_text, has_silence_padding
from .utils.path_helpers import ensure_long_path

verbose: bool = True

#: ``fp_scheme`` of rows written now: fingerprints of audio cut to the
#: fingerprint window but never padded (ported from the root cache). Library
#: Sync's generator fingerprints through ``audio_norm.normalize_for_fp`` with
#: a window (120 s by default), which used to pad every shorter track with
#: silence, so the prints of *different* short tracks matched each other.
#: Rows from before the column are NULL; those whose print ends in a run of
#: padding silence are treated as stale (see :func:`fingerprint_trusted`),
#: every other old row stays valid, so the library is not re-fingerprinted.
FP_SCHEME_UNPADDED = 2


def _dlog(label: str, msg: str, cb: Optional[Callable[[str], None]] = None) -> None:
    if not verbose:
        return
    ts = time.strftime("%H:%M:%S")
    line = f"{ts} [{label}] {msg}"
    if cb:
        cb(line)
    else:
        print(line)


def _initialize_db(conn: sqlite3.Connection) -> None:
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
    if "decoded" not in cols:
        # Library Sync's decoded Chromaprint items (see pack_decoded). Old rows
        # get NULL and are decoded on first use, then written back.
        conn.execute("ALTER TABLE fingerprints ADD COLUMN decoded BLOB")
    if "fp_scheme" not in cols:
        # NULL for every existing row: "written before padding was dropped".
        conn.execute("ALTER TABLE fingerprints ADD COLUMN fp_scheme INTEGER")
    conn.commit()


# ── Corrupt cache files ──────────────────────────────────────────────────────
#
# A damaged cache file ("file is not a database", "database disk image is
# malformed") made ``_ensure_db`` raise, so every Library Sync scan failed
# until the user found and deleted the file. It only holds recomputable
# fingerprints, so it is renamed out of the way (never deleted) and a fresh
# cache is started — ported from the root cache.


def _is_corruption(exc: BaseException) -> bool:
    """A damaged/non-SQLite file — not a busy one ("database is locked")."""
    return isinstance(exc, sqlite3.DatabaseError) and not isinstance(exc, sqlite3.OperationalError)


def _set_aside_corrupt_db(db_path: str, exc: BaseException) -> Optional[str]:
    """Rename a corrupt cache (and its -wal/-shm/-journal files) to
    ``<name>.corrupt-<time>``. Returns the new name, or None."""
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
    for suffix in ("-wal", "-shm", "-journal"):
        try:
            os.replace(db_path + suffix, target + suffix)
        except OSError:
            pass
    _memo_clear()
    _dlog("FP", f"fingerprint cache {db_path} was unreadable ({exc}); moved it to {target} and started a new one")
    return target


def _ensure_db(db_path: str) -> sqlite3.Connection:
    """Open Library Sync's cache with the schema in place.

    A corrupt file is set aside (:func:`_set_aside_corrupt_db`) and a new
    cache created, once; any other error (a locked or read-only file) is
    raised as before.
    """
    folder = os.path.dirname(db_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    for attempt in range(2):
        conn = sqlite3.connect(db_path)
        try:
            _initialize_db(conn)
            return conn
        except Exception as exc:
            conn.close()
            if attempt == 0 and _is_corruption(exc) and _set_aside_corrupt_db(db_path, exc):
                continue
            raise
    raise AssertionError("unreachable")  # pragma: no cover


# ── Row validity ─────────────────────────────────────────────────────────────


def row_is_current(cached_mtime: object, cached_size: object, mtime: float, size: int) -> bool:
    """True when a cache row was written for the file as it is now.

    Rows from before the ``mtime``/``size`` columns have NULLs there; the old
    ``abs(cached_mtime - mtime)`` raised ``TypeError`` on them mid-scan.
    """
    if cached_mtime is None or cached_size is None:
        return False
    try:
        return abs(float(cached_mtime) - float(mtime)) < 1e-6 and int(cached_size) == int(size)
    except (TypeError, ValueError):
        return False


def fingerprint_trusted(fp: object, scheme: object) -> bool:
    """False for a pre-fix row holding a silence-padded short-track print.

    Rows written now (by this module, the vendored generator or
    ``library_sync``) carry ``fp_scheme`` and are always trusted, so a track
    that genuinely ends in silence is fingerprinted again at most once. Only
    old rows (NULL scheme) whose print ends in a run of padding are
    distrusted: a cache miss, so that file is fingerprinted again.
    """
    try:
        if scheme is not None and int(scheme) >= FP_SCHEME_UNPADDED:
            return True
    except (TypeError, ValueError):
        pass
    return not has_silence_padding(fp)


def store_fingerprint_row(
    conn: sqlite3.Connection,
    path: str,
    mtime: float,
    size: int,
    duration: object,
    fingerprint: Optional[str],
) -> None:
    """Write a freshly computed fingerprint for ``path`` (no commit).

    Every writer of a *new* print goes through here: the row is stamped with
    :data:`FP_SCHEME_UNPADDED` and its ``decoded`` blob is cleared in the same
    statement, so a decode of the previous print never sits next to the new
    one (its text digest would make :func:`unpack_decoded` reject it anyway).
    Raises ``sqlite3.Error`` like ``execute``.
    """
    conn.execute(
        "INSERT OR REPLACE INTO fingerprints (path, mtime, size, duration, fingerprint, fp_scheme, decoded) "
        "VALUES (?, ?, ?, ?, ?, ?, NULL)",
        (path, mtime, size, duration, fingerprint, FP_SCHEME_UNPADDED),
    )


# ── Decoded fingerprints ─────────────────────────────────────────────────────
#
# Library Sync's candidate index needs every library print *decoded* (raw
# 32-bit Chromaprint items); decoding is pure Python, ~1.5 ms per print, so a
# 20k-track library cost ~30 s on every Build Plan. The decoded form is kept
# in the ``decoded`` column next to the compressed ``fingerprint``. A row is
# only trusted for a file whose path, mtime and size still match (exactly as
# for the fingerprint), and the blob also carries a digest of the compressed
# text it was decoded from, so a fingerprint rewritten by any other statement
# can never be paired with a stale decode. Every write of a new fingerprint
# (:func:`store_fingerprint_row`) resets the column to NULL.
#
# Blob layout: version (1 byte) | blake2b-64 digest of the stripped text (8) |
# algorithm (1) | items as packed little-endian uint32. A blob with nothing
# after the digest records "not a Chromaprint print" (so it is not retried).

DECODED_BLOB_VERSION = 1
_DIGEST_SIZE = 8


def _text_digest(text: str) -> bytes:
    import hashlib

    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=_DIGEST_SIZE).digest()


def pack_decoded(text: str, decoded: "tuple[int, bytes] | None") -> bytes:
    """Serialise ``decoded`` (``(algorithm, packed_items)`` or ``None``) for ``text``."""
    head = bytes([DECODED_BLOB_VERSION]) + _text_digest(text)
    if decoded is None:
        return head
    algorithm, packed = decoded
    return head + bytes([int(algorithm) & 0xFF]) + bytes(packed)


_NOT_STORED = object()


def unpack_decoded(text: str, blob: object) -> object:
    """``(algorithm, packed)`` or ``None`` stored for ``text``; ``_NOT_STORED``
    when ``blob`` is missing, malformed or was decoded from another text."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return _NOT_STORED
    blob = bytes(blob)
    head = 1 + _DIGEST_SIZE
    if len(blob) < head or blob[0] != DECODED_BLOB_VERSION:
        return _NOT_STORED
    if blob[1:head] != _text_digest(text):
        return _NOT_STORED
    if len(blob) == head:
        return None
    packed = blob[head + 1:]
    if len(packed) % 4:
        return _NOT_STORED
    return blob[head], packed


def is_stored(value: object) -> bool:
    return value is not _NOT_STORED


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
    """Return fingerprint for path using cache; compute if missing.

    An unreadable row is a cache miss, and a failed cache write only costs
    the cache entry, never the fingerprint. An old row holding a
    silence-padded print (:func:`fingerprint_trusted`) is recomputed.
    """
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
    conn = _ensure_db(db_path)
    try:
        try:
            row = conn.execute(
                "SELECT mtime, size, fingerprint, fp_scheme FROM fingerprints WHERE path=?",
                (path,),
            ).fetchone()
        except sqlite3.DatabaseError as e:
            log_callback(f"! Fingerprint cache read failed: {e}")
            row = None
        cached_mtime = row[0] if row else None
        cached_size = row[1] if row else None
        if row and row_is_current(cached_mtime, cached_size, mtime, size):
            fp = _fp_text(row[2])
            if fingerprint_trusted(fp, row[3]):
                _dlog("FP", f"cache hit {path}", log_callback)
                if fp:
                    _dlog("FP", f"fingerprint={fp} prefix={fp[:16]}", log_callback)
                    _memo_put(db_path, path, mtime, size, fp)
                return fp
            log_callback(f"Re-fingerprinting {path}: its cached fingerprint was made with silence padding.")
        elif row:
            _dlog(
                "FP",
                f"cache invalidated {path} stored_mtime={cached_mtime} stored_size={cached_size} new_mtime={mtime} new_size={size}",
                log_callback,
            )

        _dlog("FP", f"cache miss {path}", log_callback)
        duration, fp_hash = compute_func(path)
        fp_hash = _fp_text(fp_hash)
        if fp_hash is not None:
            _dlog("FP", f"computed fingerprint {fp_hash} prefix={fp_hash[:16]}", log_callback)
            _memo_put(db_path, path, mtime, size, fp_hash)
            try:
                store_fingerprint_row(conn, path, mtime, size, duration, fp_hash)
                conn.commit()
            except sqlite3.Error as e:
                log_callback(f"! Fingerprint cache write failed: {e}")
        return fp_hash
    finally:
        conn.close()


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
