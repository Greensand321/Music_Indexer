"""Metadata page in the Liquid Glass look (R2): every section fits the
minimum window with no sideways scrolling, and none repeats the page title."""
from __future__ import annotations

import pytest

from gui.compat import QtWidgets
from ui_harness import process_events

SECTIONS = ("metadata/tag_fixer", "metadata/fill_genres",
            "metadata/normalize_genres", "metadata/undo_tags")


def _hscroll_max(panel) -> int:
    return panel._scroll_area.horizontalScrollBar().maximum()


@pytest.mark.parametrize("size", [(900, 600), (1300, 860), (1600, 1000)])
def test_every_section_fits_without_horizontal_scroll(main_window, library, size):
    main_window.resize(*size)
    # Only the Metadata panels get the library (a long path to elide); the
    # window-wide library switch would also rebuild the album-art backdrop.
    for panel in ("tag_fixer", "genres", "tag_history"):
        main_window._workspaces[panel].set_library_path(str(library.root))
    for key in SECTIONS:
        main_window._on_nav_changed(key)
        process_events(30)
        panel = main_window.current_panel()
        assert _hscroll_max(panel) == 0, (key, size)
        inner = panel._scroll_area.widget()
        assert inner.minimumSizeHint().width() <= panel._scroll_area.viewport().width(), key


def test_no_repeated_title_and_one_line_purpose(main_window):
    for key in SECTIONS:
        main_window._on_nav_changed(key)
        panel = main_window.current_panel()
        assert panel.findChildren(QtWidgets.QLabel, "sectionTitle") == [], key
        visible = [w for w in panel.findChildren(QtWidgets.QLabel, "sectionSubtitle")
                   if w.isVisibleTo(panel)]
        assert len(visible) == 1, key
        hero = [w for w in panel.findChildren(QtWidgets.QFrame, "headerCard") if w.isVisibleTo(panel)]
        assert len(hero) == 1, key     # one hero card per view


def test_filled_review_grid_fits_the_minimum_width(make_workspace, tmp_path, monkeypatch, qtbot):
    """The Tag Fixer with proposals, at the width a 900 px window leaves it."""
    from _helpers.meta_tags import build_genre_library
    from library_factory import ffmpeg_available
    from meta_helpers import install_source, wait_for

    if not ffmpeg_available():
        pytest.skip("ffmpeg is needed to build the test files")
    build_genre_library(tmp_path / "lib", [
        ("A very long artist name - a very long title of a song.flac", "Art", "Song", ["Rock"]),
    ])
    install_source(monkeypatch, {"Song": {"artist": "A rather long corrected artist name",
                                          "genres": ["Electronic", "Ambient"], "score": 1.0}})
    ws = make_workspace("tag_fixer", library_path=tmp_path / "lib")
    ws.resize(652, 600)
    ws._scan_btn.click()
    wait_for(qtbot, lambda: ws._worker is None)
    process_events(30)
    assert ws._table.topLevelItemCount() == 2
    assert _hscroll_max(ws) == 0
    assert ws._table.horizontalScrollBar().maximum() == 0
