import os
import sqlite3
from typing import Callable, Tuple
from concurrent.futures import ProcessPoolExecutor
import acoustid

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}


def compute_fingerprint_for_file(args: Tuple[str, str]) -> Tuple[str, int | None, str | None, str | None]:
    """Compute fingerprint for a single file. Returns (path, duration, fp, error)."""
    path, _db_path = args
    try:
        duration, fp_hash = acoustid.fingerprint_file(path)
        return path, duration, fp_hash, None
    except Exception as e:  # pragma: no cover - just to be safe
        return path, None, None, str(e)


def compute_fingerprints_parallel(
    root_path: str,
    db_path: str,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int | None = None,
) -> None:
    """Walk ``root_path`` and compute fingerprints using multiple processes."""
    if log_callback is None:
        def log_callback(msg: str) -> None:
            pass

    if progress_callback is None:
        def progress_callback(current: int, total: int, msg: str) -> None:
            pass

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    db_folder = os.path.dirname(db_path)
    os.makedirs(db_folder, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fingerprints (
          path TEXT PRIMARY KEY,
          duration INT,
          fingerprint TEXT
        );
        """
    )

    audio_files = []
    for dirpath, _, files in os.walk(root_path):
        rel_dir = os.path.relpath(dirpath, root_path)
        if "Not Sorted" in rel_dir.split(os.sep):
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                audio_files.append(os.path.join(dirpath, fname))

    total = len(audio_files)
    progress_callback(0, total, "Fingerprinting")
    work_items = [(p, db_path) for p in audio_files]

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for idx, (path, duration, fp_hash, err) in enumerate(exe.map(compute_fingerprint_for_file, work_items), start=1):
            if err:
                log_callback(f"   ! Failed fingerprint {path}: {err}")
                continue
            conn.execute(
                "INSERT OR REPLACE INTO fingerprints (path, duration, fingerprint) VALUES (?, ?, ?)",
                (path, duration, fp_hash),
            )
            log_callback(f"Fingerprinted {path}")
            progress_callback(idx, total, path)
            if idx % 50 == 0 or idx == total:
                log_callback(f"   • Fingerprinting {idx}/{total}")

    conn.commit()
    conn.close()



def compute_fingerprints(
    root_path: str,
    db_path: str,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int | None = None,
) -> None:
    """Backward-compatible wrapper around :func:`compute_fingerprints_parallel`."""
    compute_fingerprints_parallel(root_path, db_path, log_callback, progress_callback, max_workers)

