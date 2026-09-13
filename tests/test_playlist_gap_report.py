"""The download list — the thing the user actually acts on."""
from __future__ import annotations

import csv
import io

import pytest

from playlist_gap_report import (
    missing_rows,
    render_csv,
    render_text,
    summary_lines,
    write_csv,
    write_text,
)
from playlist_gap_types import GapResult, ReasonCode, Verdict, WantedTrack


def result(display, verdict, *, artist=None, title=None, album=None, position=0):
    return GapResult(
        wanted=WantedTrack(
            row_id=display, display=display, artist=artist, title=title,
            album=album, position=position, playlists=("Liked videos",),
        ),
        verdict=verdict,
        reason_text="because",
        reason_code=ReasonCode.NO_CANDIDATE,
    )


@pytest.fixture
def results():
    return [
        result("Yotto - Radiate", Verdict.MISSING, artist="Yotto", title="Radiate",
               album="Erased Dreams", position=0),
        result("Yotto - Nova", Verdict.MISSING, artist="Yotto", title="Nova",
               album="Erased Dreams", position=1),
        result("FIFTY FIFTY - Cupid", Verdict.OWNED, position=2),
        result("Lane 8 - Shatter", Verdict.MISSING, position=3),
        result("Self Aware x Babydoll", Verdict.UNSURE, position=4),
        result("", Verdict.UNAVAILABLE, position=5),
    ]


def test_only_missing_rows_are_listed(results):
    assert len(missing_rows(results)) == 3


def test_source_order_is_preserved(results):
    assert [r.wanted.display for r in missing_rows(results)] == [
        "Yotto - Radiate", "Yotto - Nova", "Lane 8 - Shatter",
    ]


def test_flat_text_is_one_label_per_line(results):
    lines = render_text(results).strip().splitlines()
    assert lines == ["Yotto - Radiate", "Yotto - Nova", "Lane 8 - Shatter"]


def test_label_prefers_artist_and_title(results):
    assert "Yotto - Radiate" in render_text(results)


def test_nothing_missing_says_so():
    text = render_text([result("x", Verdict.OWNED)])
    assert "Nothing missing" in text


def test_album_grouping_annotates_how_much_is_missing(results):
    text = render_text(results, group_by="album")
    assert "# Erased Dreams  (2 missing)" in text
    assert "# Singles & one-offs  (1)" in text


def test_artist_grouping(results):
    text = render_text(results, group_by="artist")
    assert "# Yotto  (2 missing)" in text


def test_csv_has_a_stable_header(results):
    rows = list(csv.DictReader(io.StringIO(render_csv(results))))
    assert len(rows) == 3
    assert rows[0]["artist"] == "Yotto"
    assert rows[0]["verdict"] == "missing"
    assert rows[0]["playlists"] == "Liked videos"


def test_csv_can_include_everything(results):
    rows = list(csv.DictReader(io.StringIO(render_csv(results, only_missing=False))))
    assert len(rows) == len(results)


def test_writers_create_directories(tmp_path, results):
    text_path = write_text(str(tmp_path / "nested" / "list.txt"), results)
    csv_path = write_csv(str(tmp_path / "nested" / "list.csv"), results)
    assert open(text_path, encoding="utf-8").read().startswith("Yotto")
    assert "row_id" in open(csv_path, encoding="utf-8").read()


def test_summary_lines_are_plain_language(results):
    counts = {"total": 6, "owned": 1, "unsure": 1, "missing": 3, "unavailable": 1, "ignored": 0}
    lines = summary_lines(results, counts)
    assert "6 wanted tracks compared against the library." in lines
    assert "1 already owned (17% of the list)." in lines
    assert "3 to download." in lines
    assert any("unavailable upstream" in line for line in lines)


def test_summary_omits_empty_categories(results):
    counts = {"total": 1, "owned": 1, "unsure": 0, "missing": 0, "unavailable": 0, "ignored": 0}
    assert not any("never-want" in line for line in summary_lines(results, counts))


def test_summary_handles_an_empty_run():
    assert summary_lines([], {"total": 0}) [0].startswith("0 wanted")
