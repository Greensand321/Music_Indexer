"""End-to-end: read a CSV, snapshot a library, match, remember, export."""
from __future__ import annotations

import os

import pytest

from controllers.playlist_gap_controller import load_snapshot, run_all, run_source
from playlist_gap_ledger import DECIDED_USER, Ledger, ledger_path_for
from playlist_gap_report import render_text
from playlist_gap_sources import SourceSpec
from playlist_gap_types import Verdict

EXPORT = (
    "Track name,Artist name,Album,Playlist name,Type,ISRC\n"
    "FIFTY FIFTY - Cupid,,,Liked videos,Favorite,\n"
    "The Sways - Someday We Will Dream About Today,,,Liked videos,Favorite,\n"
    "Yotto - Radiate,,,Liked videos,Favorite,\n"
    ",,,Liked videos,Favorite,\n"
)

LIBRARY = {
    "By Artist/FIFTY FIFTY/Cupid.flac": {"artist": "FIFTY FIFTY", "title": "Cupid"},
    "Not Sorted/The Sways - Someday We Will Dream About Today.opus": {},
}


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "library"
    for rel in LIBRARY:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    import playlist_gap_snapshot as snapshot_module

    def fake_tags(path):
        rel = os.path.relpath(path, str(root)).replace(os.sep, "/")
        return dict(LIBRARY.get(rel, {}))

    monkeypatch.setattr(snapshot_module, "_default_tag_reader", fake_tags)
    return str(root)


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text(EXPORT, encoding="utf-8")
    return SourceSpec(name="Liked videos", kind="csv", location=str(path))


def test_full_run(library, spec):
    run = run_source(spec, library)
    assert run.counts["total"] == 4
    assert run.counts["owned"] >= 1
    assert run.counts["unavailable"] == 1
    assert run.capabilities.isrc is False
    assert run.diff.new_count == 4


def test_untagged_file_matches_by_filename(library, spec):
    run = run_source(spec, library)
    by_display = {r.wanted.display: r for r in run.results}
    owned = by_display["The Sways - Someday We Will Dream About Today"]
    assert owned.verdict is Verdict.OWNED


def test_download_list_excludes_what_you_own(library, spec):
    text = render_text(run_source(spec, library).results)
    assert "Yotto - Radiate" in text
    assert "Someday We Will Dream About Today" not in text


def test_rerunning_reports_no_new_rows(library, spec):
    run_source(spec, library)
    second = run_source(spec, library)
    assert second.diff.new_count == 0
    assert second.diff.unchanged


def test_a_user_decision_survives_the_next_run(library, spec):
    """The point of the ledger: never re-ask a settled question."""
    first = run_source(spec, library)
    radiate = next(r for r in first.results if "Radiate" in r.wanted.display)

    ledger = Ledger(ledger_path_for(library))
    try:
        ledger.record(radiate.wanted.row_id, Verdict.IGNORED, by=DECIDED_USER,
                      note="never want this")
    finally:
        ledger.close()

    second = run_source(spec, library)
    again = next(r for r in second.results if "Radiate" in r.wanted.display)
    assert again.verdict is Verdict.IGNORED
    assert again.auto is False
    assert "Radiate" not in render_text(second.results)


def test_snapshot_can_be_shared_across_runs(library, spec):
    snapshot = load_snapshot(library)
    run = run_source(spec, library, snapshot=snapshot)
    assert run.snapshot is snapshot


def test_run_all_merges_and_deduplicates(library, tmp_path, spec):
    other = tmp_path / "other.csv"
    other.write_text(
        "Track name,Playlist name\nYotto - Radiate,Deep house\nLane 8 - Shatter,Deep house\n",
        encoding="utf-8",
    )
    second = SourceSpec(name="Deep house", kind="csv", location=str(other))
    combined = run_all([spec, second], library)
    displays = [r.wanted.display for r in combined.results]
    assert displays.count("Yotto - Radiate") == 1
    assert "Lane 8 - Shatter" in displays


def test_run_all_needs_a_source(library):
    with pytest.raises(ValueError):
        run_all([], library)


def test_an_unwired_source_kind_says_so_clearly(library, tmp_path):
    spec = SourceSpec(name="x", kind="ytmusic", location="https://music.youtube.com/playlist?list=LM")
    with pytest.raises(NotImplementedError) as excinfo:
        run_source(spec, library)
    assert "CSV" in str(excinfo.value)


def test_logging_reports_what_the_source_could_supply(library, spec):
    lines = []
    run_source(spec, library, log=lines.append)
    assert any("fields available" in line for line in lines)
    assert any("never dropped" in line for line in lines)


def test_cancellation_is_honoured(library, spec):
    run = run_source(spec, library, should_cancel=lambda: True)
    assert run.results == []
