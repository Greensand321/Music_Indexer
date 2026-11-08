"""Settings dialog (SHELL-09/16/23) and Crash Log viewer (SHELL-21)."""
from __future__ import annotations

import threading

import pytest
from PySide6 import QtWidgets

import gui.dialogs.settings_drawer as sd
from gui.dialogs.settings_drawer import SettingsDrawer
from shell_helpers import wait_until


@pytest.fixture
def drawer(qtbot):
    dlg = SettingsDrawer()
    qtbot.addWidget(dlg)
    return dlg


def test_every_control_persists_under_its_key(drawer, tmp_path):
    import config

    lib = tmp_path / "Lib"
    lib.mkdir()
    drawer._service_combo.setCurrentText("AcoustID")
    drawer._api_key_entry.setText("KEY123")      # the AcoustID key...
    drawer._service_combo.setCurrentText("MusicBrainz")   # ...survives switching away
    drawer._mb_app.setText("App")
    drawer._mb_ver.setText("9")
    drawer._mb_contact.setText("me@example.com")
    drawer._library_entry.setText(str(lib))
    drawer._near_dup_spin.setValue(0.25)
    drawer._exact_dup_spin.setValue(0.15)       # above the old 0.1 cap
    drawer._backdrop_combo.setCurrentIndex(drawer._backdrop_combo.findData("gradient"))
    drawer._reduce_motion_cb.setChecked(True)
    drawer._on_save()

    cfg = config.load_config()
    assert cfg["metadata_service"] == "MusicBrainz"
    assert cfg["metadata_api_key"] == "KEY123"
    assert cfg["musicbrainz_useragent"] == {"app": "App", "version": "9", "contact": "me@example.com"}
    assert cfg["library_root"] == str(lib)
    assert cfg["near_duplicate_threshold"] == pytest.approx(0.25)
    assert cfg["exact_duplicate_threshold"] == pytest.approx(0.15)
    assert cfg["backdrop_mode"] == "gradient"
    assert cfg["bg_gradient_enabled"] is True       # legacy key: not Plain
    assert cfg["reduced_motion"] is True

    again = SettingsDrawer()
    try:
        assert again._service_combo.currentText() == "MusicBrainz"
        assert again._exact_dup_spin.value() == pytest.approx(0.15)   # not clamped on re-open
        assert again._backdrop_combo.currentData() == "gradient"
        assert again._reduce_motion_cb.isChecked() is True
    finally:
        again.deleteLater()


def test_nonexistent_default_library_is_refused(drawer, tmp_path, dialogs):
    import config

    saved = []
    drawer.settings_saved.connect(lambda: saved.append(1))
    drawer._library_entry.setText(str(tmp_path / "nope"))
    drawer._on_save()
    assert saved == []
    assert "not found" in dialogs.calls_of("warning")[0].title.lower()
    assert config.load_config()["library_root"] == ""


def test_lastfm_shows_the_api_key_field(drawer):
    drawer.show()
    drawer._service_combo.setCurrentText("Last.fm")
    assert drawer._acoustid_frame.isVisible()
    assert "Last.fm" in drawer._api_key_entry.placeholderText()
    drawer._service_combo.setCurrentText("MusicBrainz")
    assert not drawer._acoustid_frame.isVisible() and drawer._mb_frame.isVisible()


def test_connection_test_reads_widgets_on_the_gui_thread(drawer, monkeypatch):
    calls = []

    def fake_check(service, api_key, useragent):
        calls.append((service, api_key, useragent, threading.current_thread() is threading.main_thread()))
        return True, f"{service}: fine"

    monkeypatch.setattr(sd, "check_service_connection", fake_check)
    drawer._service_combo.setCurrentText("AcoustID")
    drawer._api_key_entry.setText("abc")
    drawer._on_test()
    assert wait_until(lambda: drawer._test_btn.isEnabled(), 3000)
    assert calls == [("AcoustID", "abc", ("", "", ""), False)]
    assert drawer._test_status.text() == "AcoustID: fine"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def test_acoustid_check_detects_a_rejected_key(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(
        400, {"status": "error", "error": {"code": 4, "message": "invalid API key"}}))
    ok, msg = sd.check_service_connection("AcoustID", "bad")
    assert not ok and "rejected" in msg

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(
        400, {"status": "error", "error": {"code": 3, "message": "invalid fingerprint"}}))
    ok, msg = sd.check_service_connection("AcoustID", "good")
    assert ok

    ok, msg = sd.check_service_connection("AcoustID", "")
    assert not ok and "API key" in msg


def test_lastfm_check(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(
        403, {"error": 10, "message": "Invalid API key - You must be granted a valid key"}))
    ok, msg = sd.check_service_connection("Last.fm", "bad")
    assert not ok and "Invalid API key" in msg


def test_crash_log_viewer_reads_the_qt_crash_log(qtbot):
    import os

    import crash_logger
    from gui.dialogs.crash_log_dialog import CrashLogDialog

    dlg = CrashLogDialog()
    qtbot.addWidget(dlg)
    assert "No crash log found" in dlg._text.toPlainText()
    assert crash_logger.default_log_path() in dlg._text.toPlainText()

    os.makedirs(os.path.dirname(crash_logger.default_log_path()), exist_ok=True)
    with open(crash_logger.default_log_path(), "w", encoding="utf-8") as fh:
        fh.write("2026-11-07 [MainThread] Unhandled exception:\nValueError: boom\n")
    dlg2 = CrashLogDialog()
    qtbot.addWidget(dlg2)
    assert "ValueError: boom" in dlg2._text.toPlainText()


def test_unchanged_missing_default_library_does_not_block_saving(qtbot, tmp_path, dialogs):
    import config
    from shell_helpers import write_config

    gone = str(tmp_path / "unplugged")
    write_config(library_root=gone)
    dlg = SettingsDrawer()
    qtbot.addWidget(dlg)
    dlg._api_key_entry.setText("NEWKEY")
    dlg._on_save()
    assert dialogs.calls_of("warning") == []
    assert dlg.library_changed is False
    cfg = config.load_config()
    assert cfg["metadata_api_key"] == "NEWKEY" and cfg["library_root"] == gone


def test_acoustid_and_lastfm_keys_are_stored_separately(drawer):
    """The key field used to save every service's key into metadata_api_key,
    so a Last.fm key replaced the AcoustID key (and Last.fm never read it)."""
    import config

    drawer._service_combo.setCurrentText("AcoustID")
    drawer._api_key_entry.setText("ACOUSTID-KEY")
    drawer._service_combo.setCurrentText("Last.fm")
    assert drawer._api_key_entry.text() == ""          # Last.fm has its own key
    drawer._api_key_entry.setText("LASTFM-KEY")
    drawer._service_combo.setCurrentText("AcoustID")
    assert drawer._api_key_entry.text() == "ACOUSTID-KEY"
    drawer._service_combo.setCurrentText("Last.fm")
    drawer._on_save()

    cfg = config.load_config()
    assert cfg["metadata_api_key"] == "ACOUSTID-KEY"
    assert cfg["lastfm_api_key"] == "LASTFM-KEY"
    assert cfg["metadata_service"] == "Last.fm"


def test_lastfm_source_reads_the_key_saved_in_settings(monkeypatch):
    import config
    from plugins import lastfm

    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    monkeypatch.setattr(lastfm, "API_KEY", None)
    config.save_config({"metadata_service": "Last.fm", "lastfm_api_key": "FROM-SETTINGS"})
    assert lastfm._api_key() == "FROM-SETTINGS"
    status = lastfm.LastfmPlugin().preflight()
    assert status.active and "Settings" in status.detail
    monkeypatch.setenv("LASTFM_API_KEY", "FROM-ENV")
    assert lastfm._api_key() == "FROM-ENV"           # the environment still wins
