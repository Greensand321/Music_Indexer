import os
import sqlite3
import tempfile
import threading
from typing import Callable, Iterable, List, Tuple
from pydub import AudioSegment, silence


SILENCE_THRESH = -50  # dBFS used for silence detection


def _with_thresh(func, *args, **kwargs):
    try:
        return func(*args, silence_threshold=SILENCE_THRESH, **kwargs)
    except TypeError:
        return func(*args, silence_thresh=SILENCE_THRESH, **kwargs)
from concurrent.futures import ProcessPoolExecutor
import acoustid
from utils.path_helpers import ensure_long_path

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}


def _trim_silence(path: str) -> str:
    """Return path to temporary file with leading/trailing silence removed."""
    try:
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
        audio = AudioSegment.from_file(path)
        segs = _with_thresh(silence.detect_nonsilent, audio, min_silence_len=500)
        if not segs:
            return path
        start = segs[0][0]
        end = segs[-1][1]
        trimmed = audio[start:end]
        fd, tmp = tempfile.mkstemp(suffix=f".{ext}")
        os.close(fd)
        trimmed.export(tmp, format=ext)
        return tmp
    except Exception:
        return path


def compute_fingerprint_for_file(args: Tuple[str, str, bool]) -> Tuple[str, int | None, str | None, str | None]:
    """Compute fingerprint for a single file. Returns (path, duration, fp, error)."""
    path, _db_path, trim = args
    path = ensure_long_path(path)
    tmp = None
    try:
        target = path
        if trim:
            tmp = _trim_silence(path)
            target = tmp
        duration, fp_hash = acoustid.fingerprint_file(target)
        if tmp and tmp != path:
            os.remove(tmp)
        return path, duration, fp_hash, None
    except Exception as e:  # pragma: no cover - just to be safe
        if tmp and tmp != path:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return path, None, None, str(e)


def _load_cached_entries(conn: sqlite3.Connection) -> dict[str, Tuple[float, int | None, str | None]]:
    """Return a mapping of path -> (mtime, duration, fingerprint)."""

    cursor = conn.execute(
        "SELECT path, mtime, duration, fingerprint FROM fingerprints"
    )
    return {row[0]: (row[1], row[2], row[3]) for row in cursor.fetchall()}


def _gather_audio_files(root_path: str | Iterable[str]) -> List[str]:
    """Return a list of audio files either from a directory walk or iterable."""

    if isinstance(root_path, str):
        audio_files: List[str] = []
        for dirpath, _, files in os.walk(root_path):
            rel_dir = os.path.relpath(dirpath, root_path)
            if {"not sorted", "playlists"} & {p.lower() for p in rel_dir.split(os.sep)}:
                continue
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTS:
                    audio_files.append(os.path.join(dirpath, fname))
        return audio_files

    return [p for p in root_path if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS]


def compute_fingerprints_parallel(
    root_path: str | Iterable[str],
    db_path: str,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    max_workers: int | None = None,
    trim_silence: bool = False,
    phase: str = "A",
    cancel_event: threading.Event | None = None,
) -> list[tuple[str, int | None, str | None]]:
    """Compute fingerprints and return ``(path, duration, fingerprint)`` results.

    ``root_path`` may be a directory or an iterable of specific file paths. Existing
    fingerprints in ``db_path`` are reused when their mtime matches, allowing
    subsequent scans to skip unchanged files without sacrificing quality.
    """
    if log_callback is None:
        log_callback = print

    if progress_callback is None:
        def progress_callback(current: int, total: int, msg: str, _phase: str) -> None:
            pass

    if cancel_event is None:
        cancel_event = threading.Event()

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    db_folder = os.path.dirname(db_path)
    os.makedirs(db_folder, exist_ok=True)
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

    results: list[tuple[str, int | None, str | None]] = []
    audio_files = _gather_audio_files(root_path)

    total = len(audio_files)
    progress_callback(0, total, "Fingerprinting", phase)

    cached = _load_cached_entries(conn)
    pending: list[str] = []
    completed = 0

    for path in audio_files:
        if cancel_event.is_set():
            log_callback("Cancellation requested before fingerprint scheduling")
            break
        mtime = os.path.getmtime(ensure_long_path(path))
        cache_entry = cached.get(path)
        if cache_entry and cache_entry[2] is not None and cache_entry[0] == mtime:
            duration, fp_hash = cache_entry[1], cache_entry[2]
            results.append((path, duration, fp_hash))
            completed += 1
            progress_callback(completed, total, path, phase)
            continue
        pending.append(path)

    work_items = [(p, db_path, trim_silence) for p in pending]

    if pending:
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(compute_fingerprint_for_file, item) for item in work_items]
            for fut in futures:
                if cancel_event.is_set():
                    cancelled = 0
                    for remaining in futures:
                        if not remaining.done() and remaining.cancel():
                            cancelled += 1
                    log_callback(f"Cancellation requested; attempting to stop after current file ({cancelled} cancelled)")
                    break

                path, duration, fp_hash, err = fut.result()
                completed += 1
                if err:
                    log_callback(f"   ! Failed fingerprint {path}: {err}")
                    progress_callback(completed, total, path, phase)
                    if cancel_event.is_set():
                        break
                    continue
                mtime = os.path.getmtime(ensure_long_path(path))
                conn.execute(
                    "INSERT OR REPLACE INTO fingerprints (path, mtime, duration, fingerprint) VALUES (?, ?, ?, ?)",
                    (path, mtime, duration, fp_hash),
                )
                results.append((path, duration, fp_hash))
                log_callback(f"Fingerprinted {path}")
                progress_callback(completed, total, path, phase)
                if completed % 50 == 0 or completed == total:
                    log_callback(f"   • Fingerprinting {completed}/{total}")
                if cancel_event.is_set():
                    log_callback("Cancellation requested; stopping after current fingerprint")
                    break
    else:
        log_callback("All fingerprints loaded from cache")

    conn.commit()
    conn.close()
    return results



def compute_fingerprints(
    root_path: str,
    db_path: str,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    max_workers: int | None = None,
    trim_silence: bool = False,
    phase: str = "A",
    cancel_event: threading.Event | None = None,
) -> None:
    """Backward-compatible wrapper around :func:`compute_fingerprints_parallel`."""
    return compute_fingerprints_parallel(
        root_path,
        db_path,
        log_callback,
        progress_callback,
        max_workers,
        trim_silence,
        phase,
        cancel_event,
    )
