"""Indexer "Max workers": a bounded tag-reading thread pool (area: improve).

``max_workers`` used to be accepted and ignored. It now reads tags on a
bounded thread pool; the plan must be identical to a serial one, in the same
order, cancellable, and leave no thread behind. Each file's tags are also
read once per scan now (the artist pre-scan used to read every file and the
plan phase read it again).
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

import pytest

import music_indexer_api as api
from indexer_control import IndexCancelled
from library_factory import TrackSpec, build_library


def _specs():
    specs = []
    for i in range(48):
        artist = ["Aurora Lines", "aurora lines", "Basalt", "Cobalt Choir & Friends", "Delta"][i % 5]
        specs.append(TrackSpec(
            artist, f"Song {i:02d}", f"Album {i % 3}", i % 7 + 1, str(2000 + i % 4), "Rock",
            ext=[".mp3", ".flac", ".m4a", ".ogg"][i % 4], freq=200 + i * 9,
            relpath=f"Incoming/{i:02d}.{['mp3', 'flac', 'm4a', 'ogg'][i % 4]}" if i % 2 else None,
        ))
    # collisions (same artist/title/album twice), remixes, an untagged file
    specs.append(TrackSpec("Basalt", "Song 02", "Album 2", 3, "2002", "Rock", freq=901, relpath="dup/a.mp3"))
    specs.append(TrackSpec("Basalt", "Song 02", "Album 2", 3, "2002", "Rock", freq=902, relpath="dup/b.mp3"))
    specs.append(TrackSpec("Delta", "Song (Remix)", "Remixes", 1, "2010", "Pop", freq=903))
    specs.append(TrackSpec(None, None, freq=904, relpath="loose/untagged.mp3", write_tags=False))
    return specs


@pytest.fixture(scope="module")
def _template(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("idx_workers") / "lib", _specs()).root


@pytest.fixture
def library(_template, tmp_path):
    dest = tmp_path / "lib"
    shutil.copytree(_template, dest)
    return dest


def _plan_view(plan, root):
    rel = lambda p: Path(p).relative_to(root).as_posix()
    return (
        [(rel(s), rel(d)) for s, d in plan.moves.items()],
        {rel(k): v for k, v in plan.tag_index.items()},
        list(plan.decision_log),
        {rel(k): rel(v) for k, v in plan.leftovers.items()},
        [rel(d) for d in plan.empty_dirs],
    )


def test_plan_is_identical_for_every_worker_count(library):
    serial = _plan_view(api.build_indexer_plan(str(library)), library)
    assert len(serial[0]) == 52
    for workers in (1, 2, 4, 8, 64):
        assert _plan_view(api.build_indexer_plan(str(library), max_workers=workers), library) == serial, workers


def test_progress_and_log_sequences_do_not_depend_on_workers(library):
    def run(workers):
        logs, progress = [], []
        api.build_indexer_plan(
            str(library), logs.append, lambda c, t, m, ph: progress.append((c, t, m, ph)),
            max_workers=workers,
        )
        return logs, progress

    assert run(None) == run(4)


def test_workers_really_run_in_parallel_threads(library, monkeypatch):
    names = set()
    real = api.get_tags

    def spy(path):
        names.add(threading.current_thread().name)
        return real(path)

    monkeypatch.setattr(api, "get_tags", spy)
    api.build_indexer_plan(str(library))
    assert names == {threading.current_thread().name}      # default: serial, as before
    names.clear()
    api.build_indexer_plan(str(library), max_workers=4)
    assert names and all(n.startswith("indexer-tags") for n in names)
    assert not [t for t in threading.enumerate() if t.name.startswith("indexer-tags")]


@pytest.mark.parametrize("workers", [None, 4])
def test_each_file_is_read_once_per_scan(library, monkeypatch, workers):
    calls = []
    real = api.get_tags
    monkeypatch.setattr(api, "get_tags", lambda p: calls.append(p) or real(p))
    plan = api.build_indexer_plan(str(library), max_workers=workers)
    assert sorted(calls) == sorted(plan.moves)              # used to be every file twice


def test_cancel_stops_the_pool_and_leaves_no_thread(library, monkeypatch):
    big = library / "Incoming" / "more"
    big.mkdir()
    src = library / "dup" / "a.mp3"
    for i in range(400):
        shutil.copyfile(src, big / f"copy{i:03d}.mp3")
    cancel = threading.Event()
    calls = []
    real = api.get_tags

    def spy(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(api, "get_tags", spy)

    def progress(current, _total, _msg, _phase):
        if current >= 100:
            cancel.set()

    with pytest.raises(IndexCancelled):
        api.build_indexer_plan(str(library), None, progress, cancel_event=cancel, max_workers=4)
    # The pre-scan reports progress every 100 files; the cancel set at 100 is
    # seen at 200. Only the read-ahead window (8 x workers) is read past it.
    assert 200 <= len(calls) <= 200 + 4 * 8 + 4 < 452
    assert not [t for t in threading.enumerate() if t.name.startswith("indexer-tags")]


def test_a_failing_read_raises_like_serial(library, monkeypatch):
    def boom(path):
        if path.endswith("untagged.mp3"):
            raise RuntimeError("unreadable")
        return {"artist": "A", "title": "T", "album": None, "year": None, "track": None,
                "genre": None, "disc_value": None}

    monkeypatch.setattr(api, "get_tags", boom)
    for workers in (None, 4):
        with pytest.raises(RuntimeError, match="unreadable"):
            api.build_indexer_plan(str(library), max_workers=workers)
    assert not [t for t in threading.enumerate() if t.name.startswith("indexer-tags")]


def test_run_full_indexer_and_dry_run_html_pass_max_workers(library, monkeypatch):
    seen = []
    real = api.build_indexer_plan

    def spy(*args, **kwargs):
        seen.append(kwargs.get("max_workers"))
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "build_indexer_plan", spy)
    out = library / "Docs" / "MusicIndex.html"
    res = api.run_full_indexer(str(library), str(out), dry_run_only=True, max_workers=4)
    assert res["dry_run"] and out.exists()
    api.build_dry_run_html(str(library), str(out), max_workers=3)
    assert seen == [4, 3]
    # preview-first contract unchanged: a dry run moves nothing
    assert _plan_view(res["plan"], library)[0] == _plan_view(real(str(library)), library)[0]


@pytest.mark.parametrize(
    "value, expected",
    [(None, 1), (0, 1), (1, 1), (-3, 1), ("x", 1), (4, 4), ("4", 4), (10_000, api.MAX_TAG_WORKERS)],
)
def test_worker_count_is_bounded(value, expected):
    assert api._tag_workers(value) == expected
