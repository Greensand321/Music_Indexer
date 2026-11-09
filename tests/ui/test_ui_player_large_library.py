"""The Player with a large library (R3 performance audit).

The library table used ResizeToContents for Artist and Album: Qt re-measured
up to 1,000 rows through the Python delegate on every batch, sort, filter
and resize, so loading 5,000 tracks froze the window for ~0.35 s, a sort for
~0.17 s and each search keystroke for 35-90 ms. The columns now fit their
widest text, measured once per distinct string as rows arrive, and the
search reads a cached lower-cased text per track.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6 import QtGui

from gui.workspaces import player as player_mod
from player_helpers import make_player_with_bar
from ui_harness import FrameGapMeter, process_events

N = 5000


def _rows(root: Path, n: int = N) -> list[dict]:
    rows = []
    for i in range(n):
        a, b, t = i // 100, (i // 10) % 10, i % 10 + 1
        path = str(root / f"Artist {a:02d}" / f"Album {b}" / f"{t:02d} Track {i}.mp3")
        rows.append({"track_num": str(t), "title": f"Track {i} (Extended Mix)",
                     "artist": f"The Long Artist Name {a:02d}", "album": f"Album {b} (Deluxe)",
                     "dur": "3:45", "path": path, "track_int": t, "dur_s": 225})
    return rows


def _load(ws, rows, batch=250) -> None:
    for i in range(0, len(rows), batch):
        ws._on_tracks_ready(ws._scan_gen, rows[i:i + batch])
    ws._on_scan_complete(ws._scan_gen, len(rows))


def test_artist_and_album_fit_their_text_without_resize_to_contents(qtbot, tmp_path):
    ws, _, _ = make_player_with_bar(qtbot)
    ws.resize(1100, 700)
    process_events(20)
    rows = _rows(tmp_path, 40)
    rows[7]["artist"] = "An Artist With A Rather Long Name"
    _load(ws, rows)
    process_events(20)
    table = ws._lib_table
    hh = table.horizontalHeader()
    mode = hh.sectionResizeMode
    assert mode(player_mod._COL_ARTIST) != hh.ResizeMode.ResizeToContents
    assert mode(player_mod._COL_ALBUM) != hh.ResizeMode.ResizeToContents
    fm = QtGui.QFontMetrics(player_mod._TrackTableDelegate.font_for(player_mod._COL_ARTIST))
    widest = fm.horizontalAdvance("An Artist With A Rather Long Name")
    cap = int(table.width() * ws._FIT_SHARE)
    assert table.columnWidth(player_mod._COL_ARTIST) >= min(widest, cap - 22)
    assert table.columnWidth(player_mod._COL_ARTIST) <= max(cap, 80)
    # Title keeps the rest of the row.
    assert table.columnWidth(player_mod._COL_TITLE) > table.columnWidth(player_mod._COL_ARTIST)
    # A new scan starts the widths over.
    ws._library_path = str(tmp_path)
    ws._reset_column_fit()
    assert table.columnWidth(player_mod._COL_ARTIST) < min(widest, cap - 22)


@pytest.mark.parametrize("query, shown", [("name 01", 100), ("album 3 (deluxe", 500),
                                          ("track 4999 ", 1), ("zzz", 0), ("", N)])
def test_search_still_matches_title_artist_and_album(qtbot, tmp_path, query, shown):
    ws, _, _ = make_player_with_bar(qtbot)
    _load(ws, _rows(tmp_path))
    ws._search.setText(query)
    visible = sum(not ws._lib_table.isRowHidden(r) for r in range(ws._lib_table.rowCount()))
    assert visible == shown


def test_five_thousand_tracks_load_search_and_sort_without_freezing(qtbot, tmp_path):
    """Generous budgets that hold on a slow CI runner; the structural test
    above is the strict guard. Measured offscreen here: loading in 250-row
    batches max 27 ms (p95 10), a keystroke 15 ms, a sort 53 ms; with
    ResizeToContents 78 ms (p95 70, ~50 dropped frames), 38 ms and 114 ms."""
    ws, _, _ = make_player_with_bar(qtbot)
    ws.resize(1100, 700)
    process_events(20)
    rows = _rows(tmp_path)
    with FrameGapMeter() as load:
        for i in range(0, N, 250):
            ws._on_tracks_ready(ws._scan_gen, rows[i:i + 250])
            process_events(0)
        ws._on_scan_complete(ws._scan_gen, N)
        process_events(20)
    with FrameGapMeter() as search:
        for text in ("n", "na", "name", "name 0", "name 01", ""):
            ws._search.setText(text)
            process_events(0)
    with FrameGapMeter() as sort:
        ws._lib_table.sortItems(player_mod._COL_TITLE)
        process_events(10)
    print(f"\n[perf] 5,000 tracks: load {load.summary()}; search {search.summary()}; "
          f"sort {sort.summary()}")
    assert ws._lib_table.rowCount() == N
    assert load.max_gap_ms < 150
    assert search.max_gap_ms < 80
    assert sort.max_gap_ms < 200
