"""Library Sync's plan builder (the vendored indexer engine run on the incoming
folder) must plan every incoming track and must not write into the incoming
folder.

* ``.opus`` was missing from the vendored engine's extension lists, so a new
  Opus track never reached the plan (and the preview tree drew one as a
  folder).
* Only ``incoming/Music`` was walked when it existed, so a track beside it
  (``incoming/Extra/…``) was silently left out, although the scan saw it.
* A track already sitting at its routed place inside the incoming folder was
  skipped as "destination unchanged" — for Library Sync that place is in
  *another* folder, so it was silently left out too.
* Building a plan created ``incoming/Docs`` (and ``flush_cache`` deleted an
  incoming ``Docs/.soundvault.db``).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import library_sync
from library_factory import ExtraFile, TrackSpec, build_library
from library_sync_indexer_engine import fingerprint_generator as vgen
from library_sync_indexer_engine import music_indexer_api as vidx
from utils.audio_metadata_reader import read_tags


def _snapshot(root: Path) -> dict[str, str]:
    """Every directory and file under ``root``: path -> 'dir' or sha256."""
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            out[os.path.relpath(os.path.join(dirpath, d), root) + "/"] = "dir"
        for f in filenames:
            full = os.path.join(dirpath, f)
            with open(full, "rb") as fh:
                out[os.path.relpath(full, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


@pytest.fixture
def library(tmp_path):
    return build_library(tmp_path / "library", [
        TrackSpec("Kept Artist", "Old Song", "Old Album", 1, "2001", ext=".mp3", freq=300),
    ]).root


def _plan(library: Path, incoming: Path, **kw):
    return library_sync.compute_library_sync_plan(str(library), str(incoming), transfer_mode="copy", **kw)


def _sources(plan) -> set[str]:
    return {Path(item.source).name for item in plan.items}


def test_opus_is_a_supported_extension_everywhere():
    assert ".opus" in vidx.SUPPORTED_EXTS
    assert ".opus" in vgen.SUPPORTED_EXTS


def test_new_opus_track_is_planned(library, tmp_path):
    incoming = tmp_path / "incoming"
    build_library(incoming, [
        TrackSpec("Opus Artist", "Night Bus", "Late", 1, "2022", ext=".opus", freq=500, relpath="drop/night.opus"),
        TrackSpec("Flac Artist", "Day Tram", "Early", 1, "2021", ext=".flac", freq=600, relpath="drop/day.flac"),
    ])
    plan = _plan(library, incoming)
    assert _sources(plan) == {"night.opus", "day.flac"}
    opus = next(i for i in plan.items if i.source.endswith("night.opus"))
    assert opus.destination.endswith(".opus")
    assert library_sync._rel_inside(opus.destination, str(library)) is not None


def test_preview_draws_an_opus_track_as_a_file(library, tmp_path):
    incoming = tmp_path / "incoming"
    build_library(incoming, [
        TrackSpec("Opus Artist", "Night Bus", "Late", 1, "2022", ext=".opus", freq=500, relpath="night.opus"),
    ])
    out = tmp_path / "out" / "preview.html"
    out.parent.mkdir()
    plan = library_sync.build_library_sync_preview(str(library), str(incoming), str(out), transfer_mode="copy")
    dest_name = Path(plan.items[0].destination).name
    html = out.read_text(encoding="utf-8")
    assert dest_name in html
    assert f"{dest_name}/" not in html


def test_tracks_beside_incoming_music_are_planned(library, tmp_path):
    incoming = tmp_path / "incoming"
    build_library(incoming, [
        TrackSpec("Inside Artist", "In Song", "In", 1, "2020", ext=".flac", freq=410, relpath="Music/in.flac"),
        TrackSpec("Beside Artist", "Out Song", "Out", 1, "2019", ext=".mp3", freq=420, relpath="Extra/out.mp3"),
        TrackSpec("Loose Artist", "Top Song", "Top", 1, "2018", ext=".m4a", freq=430, relpath="top.m4a"),
        # Reserved folders are not incoming tracks (the scan skips them too).
        TrackSpec("Q Artist", "Q Song", "Q", 1, "2017", ext=".mp3", freq=440, relpath="Quarantine/q.mp3"),
        TrackSpec("D Artist", "D Song", "D", 1, "2016", ext=".mp3", freq=450, relpath="Docs/d.mp3"),
    ])
    plan = _plan(library, incoming)
    assert _sources(plan) == {"in.flac", "out.mp3", "top.m4a"}
    for item in plan.items:
        # Routed into the library's music tree, never back into incoming.
        assert library_sync._rel_inside(item.destination, str(library)) is not None
        assert library_sync._rel_inside(item.destination, str(incoming)) is None


def test_track_already_at_its_route_is_planned(library, tmp_path):
    spec = dict(artist="Placed Artist", title="Placed Song", album="Placed", track=1, year="2015",
                ext=".flac", freq=470)
    probe = tmp_path / "probe"
    build_library(probe, [TrackSpec(**spec, relpath="loose.flac")])
    [(src, dst)] = vidx.compute_moves_and_tag_index(str(probe), dry_run=True)[0].items()
    routed_rel = os.path.relpath(dst, probe)

    incoming = tmp_path / "incoming"
    build_library(incoming, [TrackSpec(**spec, relpath=routed_rel)])
    plan = _plan(library, incoming)
    assert len(plan.items) == 1, plan.decision_log
    item = plan.items[0]
    assert item.decision == library_sync.ExecutionDecision.COPY
    assert os.path.relpath(item.destination, library) == routed_rel


@pytest.mark.parametrize("with_docs", [False, True])
@pytest.mark.parametrize("flush", [False, True])
def test_planning_leaves_the_incoming_tree_unchanged(library, tmp_path, with_docs, flush):
    incoming = tmp_path / "incoming"
    extras = [ExtraFile("Docs/.soundvault.db", b"user cache"), ExtraFile("Docs/notes.txt", b"hi")] if with_docs else []
    build_library(incoming, [
        TrackSpec("A Artist", "A Song", "A", 1, "2020", ext=".flac", freq=510, relpath="Music/a.flac"),
        TrackSpec("B Artist", "B Song", None, None, None, ext=".opus", freq=520, relpath="b.opus"),
        TrackSpec(None, None, ext=".mp3", freq=530, relpath="untagged.mp3", write_tags=False),
    ], extras)
    before = _snapshot(incoming)
    library_before = _snapshot(library)

    out = tmp_path / "out" / "preview.html"
    out.parent.mkdir()
    plan = library_sync.build_library_sync_preview(
        str(library), str(incoming), str(out), transfer_mode="move", flush_cache=flush,
    )
    assert len(plan.items) == 3
    assert _snapshot(incoming) == before
    assert not (incoming / "Docs").exists() or with_docs
    assert _snapshot(library) == library_before    # planning writes nothing into the library either


def test_every_scanned_incoming_track_reaches_the_plan(library, tmp_path, monkeypatch):
    """End to end: compare_libraries' scan and the plan see the same files."""
    def fake(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kw):
        return [(p, 1, ",".join(str(ord(c)) for c in (read_tags(p).get("title") or Path(p).stem))) for p in paths]

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", fake)
    incoming = tmp_path / "incoming"
    build_library(incoming, [
        TrackSpec("C Artist", "C Song", "C", 1, "2020", ext=".flac", freq=610, relpath="Music/c.flac"),
        TrackSpec("D Artist", "D Song", "D", 1, "2020", ext=".opus", freq=620, relpath="Music/d.opus"),
        TrackSpec("E Artist", "E Song", "E", 1, "2020", ext=".mp3", freq=630, relpath="Other/e.mp3"),
    ])
    result = library_sync.compare_libraries(
        str(library), str(incoming), str(tmp_path / "fp.db"), include_match_objects=True,
    )
    scanned = {Path(m.incoming.path).name for m in result["match_objects"]}
    assert scanned == {"c.flac", "d.opus", "e.mp3"}
    plan = _plan(library, incoming, match_results=result["match_objects"])
    assert _sources(plan) == scanned
    assert all(i.decision == library_sync.ExecutionDecision.COPY for i in plan.items), [
        (i.source, i.decision, i.reason) for i in plan.items
    ]
