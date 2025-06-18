import os
import sqlite3
from typing import Callable
import acoustid

SUPPORTED_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg"}


def compute_fingerprints(root_path: str, db_path: str, log_callback: Callable[[str], None] | None = None) -> None:
    """Walk ``root_path`` and compute fingerprints for all audio files."""
    if log_callback is None:
        def log_callback(msg: str) -> None:
            pass

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
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                audio_files.append(os.path.join(dirpath, fname))

    total = len(audio_files)
    for idx, path in enumerate(audio_files, start=1):
        if idx % 50 == 0 or idx == total:
            log_callback(f"   • Fingerprinting {idx}/{total}")
        try:
            duration, fp_hash = acoustid.fingerprint_file(path)
            conn.execute(
                "INSERT OR REPLACE INTO fingerprints (path, duration, fingerprint) VALUES (?, ?, ?)",
                (path, duration, fp_hash),
            )
        except Exception as e:
            log_callback(f"   ! Failed fingerprint {path}: {e}")

    conn.commit()
    conn.close()

