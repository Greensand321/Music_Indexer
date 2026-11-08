"""Cross-area wiring in the main window (pieces fixed on separate branches)."""
from __future__ import annotations

from ui_harness import process_events


def test_player_volume_changes_from_any_source_move_the_bar_knob(main_window):
    player = main_window._workspaces["player"]
    bar = main_window._now_playing_bar
    player.set_volume(37)              # the Player's public API (no page slider)
    process_events(5)
    assert bar._vol_slider.value() == 37


def test_bar_volume_reaches_the_page_without_echo(main_window):
    player = main_window._workspaces["player"]
    bar = main_window._now_playing_bar
    bar._vol_slider.setValue(64)       # mimics wheel / keyboard / groove clicks
    process_events(5)
    assert player.volume() == 64
    assert bar._vol_slider.value() == 64


def test_saving_settings_applies_cached_appearance_flags_immediately(main_window, monkeypatch):
    """The gradient flag and reduced-motion flag are cached (DS-03); saving
    Settings must refresh both instead of waiting for the next theme switch."""
    import config
    from gui.themes import motion
    from gui.widgets import gradient_bg

    cfg = config.load_config()
    cfg["bg_gradient_enabled"] = not gradient_bg._gradient_enabled()
    cfg["reduced_motion"] = True
    config.save_config(cfg)
    monkeypatch.delenv("ALPHADEX_REDUCED_MOTION", raising=False)

    main_window._on_settings_saved()
    assert gradient_bg._gradient_enabled() == cfg["bg_gradient_enabled"]
    assert motion.reduced_motion() is True

    cfg["reduced_motion"] = False
    config.save_config(cfg)
    main_window._on_settings_saved()
    assert motion.reduced_motion() is False
