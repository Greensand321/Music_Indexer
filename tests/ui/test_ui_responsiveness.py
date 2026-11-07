"""Responsiveness baselines — how long the GUI thread freezes for common actions.

The redesign's "perfect motion" goal is measurable: an animation can only be
smooth if nothing blocks the event loop for more than a frame (~17 ms). These
tests record the freeze for core shell actions and fail only on gross
regressions (ceilings are deliberately generous so slow CI machines pass).
Measured values are printed (run with ``-s``) — tighten the ceilings as the
redesign lands. Baseline on the dev container, 2026-11-07: theme switch
≈1000 ms (DS-04), workspace switch ≈10-60 ms, window build ≈350 ms.
"""
from __future__ import annotations

import time

import pytest

from gui.main_window import _WORKSPACE_MAP
from ui_harness import FrameGapMeter, process_events

CEILING_MS = {
    "window_build": 6000,
    "workspace_switch": 1500,
    "theme_switch": 4000,
}


def test_window_build_time(qtbot):
    from gui.main_window import AlphaDEXWindow

    start = time.perf_counter()
    window = AlphaDEXWindow()
    elapsed = (time.perf_counter() - start) * 1000
    qtbot.addWidget(window)
    print(f"\n[perf] AlphaDEXWindow() built in {elapsed:.0f} ms")
    assert elapsed < CEILING_MS["window_build"]


def test_workspace_switch_freeze(main_window):
    worst = ("", 0.0)
    for key in _WORKSPACE_MAP:
        with FrameGapMeter() as meter:
            main_window._on_nav_changed(key)
            process_events(30)
        print(f"[perf] switch → {key:<13} {meter.summary()}")
        if meter.max_gap_ms > worst[1]:
            worst = (key, meter.max_gap_ms)
    assert worst[1] < CEILING_MS["workspace_switch"], worst


@pytest.mark.parametrize("theme", ["pearl", "midnight"])
def test_theme_switch_freeze(main_window, theme):
    from gui.themes.manager import get_manager

    with FrameGapMeter() as meter:
        get_manager().apply(theme)
        process_events(30)
    print(f"[perf] theme → {theme:<9} {meter.summary()}")
    assert meter.max_gap_ms < CEILING_MS["theme_switch"]
