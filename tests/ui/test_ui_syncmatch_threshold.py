"""Library Sync page: the match threshold on the rescaled metric (area: syncmatch).

Library Sync's fingerprint distance is now the bit error rate of the decoded
Chromaprint prints (same recording ≈ 0.01, different recordings ≥ 0.43) and
its default threshold is 0.2. The page still offered "0.3", and a
``sync_threshold`` saved under the old metric (0.3 / 0.35 were its
defaults) would now match unrelated tracks. These tests pin the page's
default, the one-time migration of an old saved default, the warning for a
threshold high enough to match unrelated tracks, and a scan through the page
on a 73-track mixed-codec library.
"""
from __future__ import annotations

import json

import pytest
from PySide6 import QtCore

import config
from library_sync_indexer_engine import config as vconfig

ALL_WORKERS = ("_worker", "_build_worker", "_exec_worker")


def _write_config(values: dict) -> None:
    with open(config.CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(values, fh)


def _read_config() -> dict:
    with open(config.CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _page(make_workspace, library_path=""):
    ws = make_workspace("library_sync", library_path=library_path)
    logs = []
    ws.log_message.connect(lambda message, level: logs.append((level, message)))
    ws.flush_pending_logs()
    ws._test_logs = logs
    return ws


def test_root_config_mirrors_the_engine_thresholds():
    """The page and the legacy panel read ``config``; Library Sync matches with
    the engine's values. They used to say 0.3 / 0.35 against the engine's 0.2."""
    assert config.DEFAULT_FP_THRESHOLDS == vconfig.DEFAULT_FP_THRESHOLDS
    import library_sync

    assert config.SYNC_DIFFERENT_RECORDING_FLOOR == library_sync.DIFFERENT_RECORDING_FLOOR


def test_default_threshold_is_the_engine_default(make_workspace):
    ws = _page(make_workspace)
    assert ws._threshold_entry.text() == "0.2"
    thresholds, error = ws._parse_thresholds()
    assert error is None and thresholds == {"default": vconfig.DEFAULT_FP_THRESHOLDS["default"]}
    ws._threshold_entry.setText("")
    assert ws._parse_thresholds()[0] == {"default": 0.2}
    assert "0.3" not in ws._overrides_text.placeholderText()
    assert not ws._threshold_warning.isVisible()


@pytest.mark.parametrize("saved", ["0.3", "0.35", "0.30", 0.35])
def test_old_default_saved_threshold_is_migrated_once(make_workspace, saved):
    _write_config({"sync_threshold": saved, "sync_incoming": "/somewhere"})
    ws = _page(make_workspace)
    assert ws._threshold_entry.text() == "0.2"
    stored = _read_config()
    assert stored["sync_threshold"] == "0.2"
    assert stored["sync_threshold_scale"] == config.SYNC_THRESHOLD_SCALE
    assert stored["sync_incoming"] == "/somewhere"         # the rest of the config survives
    why = [m for _level, m in ws._test_logs if "threshold" in m]
    assert len(why) == 1 and "rescaled" in why[0] and str(saved) in why[0]
    assert not ws._threshold_warning.isVisible()

    # The new value carries the marker: a later 0.3 chosen on the new scale stays.
    _write_config({**stored, "sync_threshold": "0.3"})
    again = _page(make_workspace)
    assert again._threshold_entry.text() == "0.3"
    assert _read_config()["sync_threshold"] == "0.3"
    assert not [m for _level, m in again._test_logs if "rescaled" in m]


def test_other_saved_threshold_is_left_alone(make_workspace):
    _write_config({"sync_threshold": "0.25"})
    before = _read_config()
    ws = _page(make_workspace)
    assert ws._threshold_entry.text() == "0.25"
    assert _read_config() == before                          # not rewritten
    assert not ws._threshold_warning.isVisible()
    assert not [m for _level, m in ws._test_logs if "rescaled" in m]


def test_saved_threshold_that_matches_unrelated_tracks_is_kept_but_warned(make_workspace):
    _write_config({"sync_threshold": "0.45"})
    ws = _page(make_workspace)
    assert ws._threshold_entry.text() == "0.45"
    assert _read_config()["sync_threshold"] == "0.45"
    assert ws._threshold_warning.isVisible()
    text = ws._threshold_warning.text()
    assert "0.45" in text and "unrelated tracks" in text and "0.2" in text


def test_warning_follows_the_threshold_and_overrides(make_workspace):
    ws = _page(make_workspace)
    ws._threshold_entry.setText("0.4")
    assert ws._threshold_warning.isVisible()
    ws._threshold_entry.setText("0.39")
    assert not ws._threshold_warning.isVisible()
    ws._overrides_text.setPlainText(".flac=0.1\nmp3=0.5")
    assert ws._threshold_warning.isVisible() and ".mp3" in ws._threshold_warning.text()
    ws._overrides_text.setPlainText("garbage")               # a parse error is reported on Scan
    assert not ws._threshold_warning.isVisible()


def test_save_session_marks_the_threshold_scale(make_workspace):
    ws = _page(make_workspace)
    ws._threshold_entry.setText("0.3")
    ws._on_save_session()
    stored = _read_config()
    assert stored["sync_threshold"] == "0.3"
    assert stored["sync_threshold_scale"] == config.SYNC_THRESHOLD_SCALE
    again = _page(make_workspace)
    assert again._threshold_entry.text() == "0.3"            # a deliberate 0.3 is not "migrated"


def test_scan_through_the_page_recognises_every_copy(make_workspace, tmp_path, monkeypatch, qtbot):
    from library_factory import ffmpeg_available

    if not ffmpeg_available():
        pytest.skip("needs ffmpeg to build MP3/FLAC/M4A/Ogg/Opus files")
    from _helpers.syncmatch_library import build_sync_scenario, install_title_fingerprints
    from library_helpers import wait_idle

    install_title_fingerprints(monkeypatch)
    sc = build_sync_scenario(tmp_path / "case")
    ws = _page(make_workspace, library_path=sc.library)
    ws._incoming_entry.setText(str(sc.incoming))
    qtbot.mouseClick(ws._scan_btn, QtCore.Qt.MouseButton.LeftButton)
    wait_idle(qtbot, ws, ALL_WORKERS)

    rows = {ws._incoming_table.topLevelItem(i).text(0): ws._incoming_table.topLevelItem(i).text(1)
            for i in range(ws._incoming_table.topLevelItemCount())}
    assert rows == {
        "song_a.flac": "COLLISION",
        "late.mp3": "COLLISION",
        "shifted.mp3": "COLLISION",
        "brand_new.flac": "NEW",
    }
    summary = ws._match_summary_lbl.text()
    assert "1 new" in summary and "3 already in library (1 possible upgrade(s))" in summary
    assert "0 low confidence" in summary
