"""Cross-area wiring in the main window (pieces fixed on separate branches)."""
from __future__ import annotations

from ui_harness import process_events


def test_page_volume_changes_from_any_source_move_the_bar_knob(main_window):
    player = main_window._workspaces["player"]
    bar = main_window._now_playing_bar
    # setValue mimics wheel / keyboard / groove clicks (no sliderMoved).
    player._vol_slider.setValue(37)
    process_events(5)
    assert bar._vol_slider.value() == 37


def test_bar_volume_reaches_the_page_without_echo(main_window):
    player = main_window._workspaces["player"]
    bar = main_window._now_playing_bar
    bar._vol_slider.setValue(64)
    process_events(5)
    assert player._vol_slider.value() == 64
    assert bar._vol_slider.value() == 64
