import crash_watcher

def test_crash_report_written(monkeypatch, tmp_path):
    crash_watcher._watcher = None  # ensure clean state
    crash_watcher.clean_shutdown = False
    crash_watcher.start()
    crash_watcher.record_event("test event")
    crash_watcher.set_library_path(tmp_path)
    monkeypatch.setattr(crash_watcher, "tk", None)
    monkeypatch.setattr(crash_watcher, "ScrolledText", None)
    crash_watcher._handle_exit()
    report = tmp_path / "Docs" / "crash_report.txt"
    assert report.exists()
    assert "test event" in report.read_text()
    crash_watcher.clean_shutdown = True
    crash_watcher._watcher = None
