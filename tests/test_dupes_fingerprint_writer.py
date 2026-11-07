"""The fingerprint-cache writer thread must sleep when idle.

``FingerprintWriter._run`` never reset its flush timer when it woke up with
nothing pending, so once ``flush_interval`` had passed it called
``queue.get(timeout=0)`` in a tight loop for the rest of the process — a
100 % CPU thread that starved the Qt GUI thread of the GIL (window builds
went from ~0.4 s to ~45 s in the test suite after one Duplicates scan).
"""
from __future__ import annotations

import queue
import time

from fingerprint_cache import FingerprintWriter


class _CountingQueue(queue.Queue):
    def __init__(self) -> None:
        super().__init__()
        self.gets = 0

    def get(self, block=True, timeout=None):  # noqa: D401
        self.gets += 1
        return super().get(block, timeout)


def test_idle_writer_does_not_spin(tmp_path):
    writer = FingerprintWriter(str(tmp_path / "fp.db"), flush_interval=0.05)
    counting = _CountingQueue()
    writer._queue.put(("flush", None))       # wake the thread so it picks up the new queue
    time.sleep(0.1)
    writer._queue = counting
    try:
        writer.enqueue("/a.mp3", 1.0, 1, None, "fp")
        time.sleep(0.6)                      # ~12 flush intervals of idleness
        gets = counting.gets
    finally:
        writer.shutdown()
    # A sleeping writer wakes about once per interval; a spinning one
    # calls get() hundreds of thousands of times in 0.6 s.
    assert gets < 60, gets


def test_writes_still_flush_on_the_interval(tmp_path):
    import sqlite3

    db = tmp_path / "fp.db"
    writer = FingerprintWriter(str(db), flush_interval=0.05)
    try:
        writer.enqueue("/a.mp3", 1.0, 1, None, "fp")
        deadline = time.monotonic() + 3
        rows = []
        while time.monotonic() < deadline and not rows:
            time.sleep(0.05)
            with sqlite3.connect(db) as conn:
                rows = conn.execute("SELECT path FROM fingerprints").fetchall()
        assert rows == [("/a.mp3",)]
    finally:
        writer.shutdown()
