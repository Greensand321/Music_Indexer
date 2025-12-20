import os
import sqlite3
import time
import types
import sys

# Stub heavy dependencies before importing the module under test
acoustid_stub = types.ModuleType("acoustid")
acoustid_stub.fingerprint_file = lambda path: (0, "stub-fp")
sys.modules["acoustid"] = acoustid_stub

silence_stub = types.ModuleType("pydub.silence")
silence_stub.detect_nonsilent = lambda *args, **kwargs: []

AudioSegment = type("AudioSegment", (), {"from_file": staticmethod(lambda path: None)})
pydub_stub = types.ModuleType("pydub")
pydub_stub.AudioSegment = AudioSegment
pydub_stub.silence = silence_stub
sys.modules["pydub"] = pydub_stub
sys.modules["pydub.silence"] = silence_stub

import fingerprint_generator


SCHEMA = """
CREATE TABLE IF NOT EXISTS fingerprints (
    path TEXT PRIMARY KEY,
    mtime REAL,
    duration INT,
    fingerprint TEXT
);
"""


def _init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def test_cached_fingerprints_reused(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_text("data")
    db = tmp_path / "fp.db"
    conn = _init_db(db)
    mtime = os.path.getmtime(audio)
    conn.execute(
        "INSERT OR REPLACE INTO fingerprints (path, mtime, duration, fingerprint) VALUES (?, ?, ?, ?)",
        (str(audio), mtime, 111, "fp-cache"),
    )
    conn.commit()
    conn.close()

    logs: list[str] = []
    progress: list[tuple[int, int, str]] = []

    results = fingerprint_generator.compute_fingerprints_parallel(
        [str(audio)],
        str(db),
        log_callback=logs.append,
        progress_callback=lambda c, t, p, _ph: progress.append((c, t, p)),
        max_workers=1,
    )

    assert results == [(str(audio), 111, "fp-cache")]
    assert any("All fingerprints loaded from cache" in line for line in logs)
    assert (1, 1, str(audio)) in progress


class DummyExecutor:
    def __init__(self, *_, **__):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, iterable):
        return map(fn, iterable)

    def submit(self, fn, arg):
        class _Future:
            def __init__(self, fn, arg):
                self._fn = fn
                self._arg = arg
                self._result = None
                self._executed = False

            def result(self):
                if not self._executed:
                    self._result = self._fn(self._arg)
                    self._executed = True
                return self._result

            def done(self):
                return self._executed

            def cancel(self):
                return False

        return _Future(fn, arg)


def test_mtime_change_triggers_recompute(monkeypatch, tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_text("old")
    db = tmp_path / "fp.db"
    conn = _init_db(db)
    old_mtime = os.path.getmtime(audio)
    conn.execute(
        "INSERT OR REPLACE INTO fingerprints (path, mtime, duration, fingerprint) VALUES (?, ?, ?, ?)",
        (str(audio), old_mtime, 5, "old-fp"),
    )
    conn.commit()
    conn.close()

    def fake_compute(args):
        path, _db_path, _trim = args
        return path, 9, "new-fp", None

    # ensure filesystem mtime advances
    time.sleep(0.01)
    audio.write_text("new")

    monkeypatch.setattr(fingerprint_generator, "ProcessPoolExecutor", DummyExecutor)
    monkeypatch.setattr(fingerprint_generator, "compute_fingerprint_for_file", fake_compute)

    results = fingerprint_generator.compute_fingerprints_parallel(
        [str(audio)],
        str(db),
        log_callback=lambda _msg: None,
        progress_callback=lambda *_args, **_kwargs: None,
        max_workers=1,
    )

    assert results == [(str(audio), 9, "new-fp")]
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT mtime, duration, fingerprint FROM fingerprints WHERE path=?",
        (str(audio),),
    ).fetchone()
    conn.close()

    assert row[2] == "new-fp"
    assert row[0] == os.path.getmtime(audio)
