import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Callable
from fingerprint_cache import flush_fingerprint_writes, get_fingerprint


def prewarm_cache(paths: Iterable[str], db_path: str, compute_func: Callable[[str], tuple[int | None, str | None]] | None = None, max_workers: int = 2):
    """Warm fingerprint cache for ``paths`` in background."""
    if compute_func is None:
        def compute_func(p: str) -> tuple[int | None, str | None]:
            try:
                import acoustid
                return acoustid.fingerprint_file(p)
            except Exception:
                return None, None

    def worker() -> None:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            for p in paths:
                exe.submit(get_fingerprint, p, db_path, compute_func)
        # ``get_fingerprint`` only queues its writes on the batching cache
        # writer.  Flush them before this thread ends, so "the prewarm thread
        # has finished" really means "the fingerprints are in the DB".
        flush_fingerprint_writes(db_path)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return t
