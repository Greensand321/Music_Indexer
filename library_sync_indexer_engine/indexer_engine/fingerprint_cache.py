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
    if "decoded" not in cols:
        # Library Sync's decoded Chromaprint items (see pack_decoded). Old rows
        # get NULL and are decoded on first use, then written back.
        conn.execute("ALTER TABLE fingerprints ADD COLUMN decoded BLOB")
    conn.commit()

    return conn


# ── Decoded fingerprints ─────────────────────────────────────────────────────
#
# Library Sync's candidate index needs every library print *decoded* (raw
# 32-bit Chromaprint items); decoding is pure Python, ~1.5 ms per print, so a
# 20k-track library cost ~30 s on every Build Plan. The decoded form is kept
# in the ``decoded`` column next to the compressed ``fingerprint``. A row is
# only trusted for a file whose path, mtime and size still match (exactly as
# for the fingerprint), and the blob also carries a digest of the compressed
# text it was decoded from, so a fingerprint rewritten by any other statement
# can never be paired with a stale decode. Every ``INSERT OR REPLACE`` of a
# new fingerprint resets the column to NULL.
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
