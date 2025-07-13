import os
import sqlite3
from typing import Callable, Tuple
import tempfile
from pydub import AudioSegment, silence
from concurrent.futures import ProcessPoolExecutor
import acoustid

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}


def _trim_silence(path: str) -> str:
    """Return path to temporary file with leading/trailing silence removed."""
    try:
        ext = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
        audio = AudioSegment.from_file(path)
        segs = silence.detect_nonsilent(audio, min_silence_len=500, silence_thresh=-50)
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


def compute_fingerprints_parallel(
    root_path: str,
    db_path: str,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    max_workers: int | None = None,
    trim_silence: bool = False,
    phase: str = "A",
) -> None:
    """Walk ``root_path`` and compute fingerprints using multiple processes."""
    if log_callback is None:
        def log_callback(msg: str) -> None:
            pass

    if progress_callback is None:
        def progress_callback(current: int, total: int, msg: str, _phase: str) -> None:
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
          mtime REAL,
          duration INT,
          fingerprint TEXT
        );
        """
    )

    audio_files = []
    idx = 0
    for dirpath, _, files in os.walk(root_path):
        rel_dir = os.path.relpath(dirpath, root_path)
        if "Not Sorted" in rel_dir.split(os.sep):
            continue
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                full = os.path.join(dirpath, fname)
                audio_files.append(full)
                idx += 1
                progress_callback(idx, 0, os.path.relpath(full, root_path), phase)

    total = len(audio_files)
    progress_callback(0, total, "Fingerprinting", phase)
    work_items = [(p, db_path, trim_silence) for p in audio_files]

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for idx, (path, duration, fp_hash, err) in enumerate(exe.map(compute_fingerprint_for_file, work_items), start=1):
            if err:
                log_callback(f"   ! Failed fingerprint {path}: {err}")
                continue
            mtime = os.path.getmtime(path)
            conn.execute(
                "INSERT OR REPLACE INTO fingerprints (path, mtime, duration, fingerprint) VALUES (?, ?, ?, ?)",
                (path, mtime, duration, fp_hash),
            )
            log_callback(f"Fingerprinted {path}")
            progress_callback(idx, total, path, phase)
            if idx % 50 == 0 or idx == total:
                log_callback(f"   • Fingerprinting {idx}/{total}")

    conn.commit()
    conn.close()



def compute_fingerprints(
    root_path: str,
    db_path: str,
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    max_workers: int | None = None,
    trim_silence: bool = False,
    phase: str = "A",
) -> None:
    """Backward-compatible wrapper around :func:`compute_fingerprints_parallel`."""
    compute_fingerprints_parallel(root_path, db_path, log_callback, progress_callback, max_workers, trim_silence, phase)

