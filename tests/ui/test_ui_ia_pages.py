"""The adopted information architecture in the running window: pages,
sections, old-key routing, the Experimental pills, and the new pages (Home,
Activity, Settings, Sound Map actions, Mirror → Verify)."""
from __future__ import annotations

import os
import threading
import time

import pytest
from PySide6 import QtCore, QtWidgets

from shell_helpers import make_album_tree, wait_until
from ui_harness import process_events


# ── Routing ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key, page, section, panel", [
    ("similarity", "duplicates", "compare", "similarity"),
    ("tag_fixer", "metadata", "tag_fixer", "tag_fixer"),
    ("genres", "metadata", "fill_genres", "genres"),
    ("clustered", "sound_map", "analyse", "clustered"),
    ("graph", "sound_map", "map", "graph"),
    ("compression", "mirror", "build", "compression"),
    ("tools", "exports", "exports", "exports"),
    ("duplicates", "duplicates", "find", "duplicates"),
    ("indexer", "indexer", "indexer", "indexer"),
    ("settings/appearance", "settings", "appearance", "settings"),
    ("bucketing", "duplicates", "bucketing", "bucketing"),
])
def test_old_and_new_keys_route_to_page_and_section(main_window, key, page, section, panel):
    main_window._on_nav_changed(key)
    process_events(5)
    current = main_window.current_page()
    assert current is main_window._pages[page]
    assert current.current_section() == section
    assert main_window.current_panel() is main_window._workspaces[panel]
    assert main_window._workspaces[panel].isVisible()
    assert main_window._sidebar._active_key == page


def test_navigate_requested_from_analyse_opens_the_map(main_window):
    main_window._on_nav_changed("clustered")
    main_window._workspaces["clustered"]._open_graph()      # "📊 Open the map"
    page = main_window.current_page()
    assert page is main_window._pages["sound_map"] and page.current_section() == "map"
    assert main_window.statusBar().currentMessage() == "Sound Map · Map"


def test_sidebar_keeps_a_pages_section_but_panel_keys_pick_theirs(main_window, qtbot):
    main_window._on_nav_changed("similarity")
    main_window.select_page("home")
    qtbot.mouseClick(main_window._sidebar._buttons["duplicates"], QtCore.Qt.MouseButton.LeftButton)
    process_events(5)
    page = main_window._pages["duplicates"]
    assert main_window.current_page() is page and page.current_section() == "compare"
    main_window._on_nav_changed("duplicates")       # the Duplicate Finder panel
    assert page.current_section() == "find"


def test_section_tab_click_switches_panel_and_status(main_window, monkeypatch):
    graph = main_window._workspaces["graph"]
    seen = []
    monkeypatch.setattr(graph, "on_activated", lambda: seen.append(1), raising=False)
    main_window._on_nav_changed("sound_map/analyse")
    page = main_window._pages["sound_map"]
    page.tab_bar().setCurrentIndex(1)
    process_events(5)
    assert page.current_section() == "map"
    assert main_window.current_panel() is graph
    assert main_window.statusBar().currentMessage() == "Sound Map · Map"
    assert seen == [1], "the Map's on_activated hook runs once when its tab is chosen"


def test_genre_sections_drive_the_genre_panel(main_window):
    genres = main_window._workspaces["genres"]
    assert not genres._tabs.tabBar().isVisibleTo(genres), "the page's tabs replace the panel's"
    main_window._on_nav_changed("metadata/normalize_genres")
    assert genres._tabs.currentIndex() == 1 and genres.current_section() == "normalize"
    main_window.select_page("home")
    main_window._on_nav_changed("genres")           # old key: the view it shows
    assert main_window.current_page().current_section() == "normalize_genres"
    main_window._on_nav_changed("fill_genres")
    assert genres._tabs.currentIndex() == 0


# ── Experimental pills ──────────────────────────────────────────────────────

PROTOTYPES = {"fuzzy_finder", "scan_engine", "bucketing"}


def test_experimental_pills_mark_exactly_the_three_prototypes(main_window):
    from gui.widgets.pill import ExperimentalPill

    tab_pills = {}
    for page in main_window._pages.values():
        for section, pill in page.experimental_pills().items():
            tab_pills[(page.key, section)] = pill
    assert set(tab_pills) == {("duplicates", key) for key in PROTOTYPES}

    panel_pills = {key for key, ws in main_window._workspaces.items()
                   if getattr(ws, "pill", None) is not None}
    assert panel_pills == PROTOTYPES

    every_pill = main_window.findChildren(ExperimentalPill)
    assert len(every_pill) == 6, "a tab pill and a title pill per prototype, nothing else"
    assert all(p.text() == "Experimental" for p in every_pill)

    page = main_window._pages["duplicates"]
    for key in PROTOTYPES:
        main_window._on_nav_changed(f"duplicates/{key}")
        process_events(5)
        panel = main_window._workspaces[key]
        assert panel.pill.isVisible() and tab_pills[("duplicates", key)].isVisible()
        assert "not ready to test" in panel._note.text()
    for key in ("find", "compare", "pair_review"):
        assert key not in page.experimental_pills()


# ── Home ────────────────────────────────────────────────────────────────────

def _home_library(root):
    """A library with an Indexer layout, reserved folders and one run of each kind."""
    root = make_album_tree(root, artists=1, albums=1, tracks=1)
    os.makedirs(os.path.join(root, "By Artist"))

    def touch(rel, age_s=0):
        path = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(b"x")
        if age_s:
            stamp = time.time() - age_s
            os.utime(path, (stamp, stamp))
        return path

    touch("Manual Review/a.mp3")
    touch("Manual Review/b.flac")
    touch("Quarantine/Run/x.mp3")
    touch("Not Sorted/notes.txt")                 # not audio
    touch("Docs/indexer_executed_20260101_000000.txt", 500)
    touch("Docs/indexer_executed_20260102_000000.txt", 100)     # the newer one
    touch("Docs/duplicate_execution_reports/consolidation_run_1/reports/execution_report.html", 50)
    touch("Docs/LibrarySyncExecuted_run1.html", 40)
    touch("Docs/tag_changes_tagfixer_20260101-000000.jsonl", 30)
    touch("Docs/tag_changes_genres_20260101-000000.jsonl", 20)
    touch("Docs/file_cleanup_20260101.jsonl", 10)
    return root


def test_library_health_scan_counts_and_finds_the_latest_runs(tmp_path):
    from gui.workspaces.home import scan_library_health

    root = _home_library(tmp_path / "Lib")
    health = scan_library_health(root)
    assert health.layout_ok
    assert health.folder_counts == {"Manual Review": 2, "Quarantine": 1, "Not Sorted": 0}
    kinds = [(r.kind, r.detail, os.path.basename(r.path)) for r in health.recent_runs]
    assert kinds == [                                    # newest first
        ("File cleanup", "", "file_cleanup_20260101.jsonl"),
        ("Tag changes", "genres", "tag_changes_genres_20260101-000000.jsonl"),
        ("Tag changes", "tagfixer", "tag_changes_tagfixer_20260101-000000.jsonl"),
        ("Library Sync run", "", "LibrarySyncExecuted_run1.html"),
        ("Duplicate cleanup", "", "execution_report.html"),
        ("Indexer run", "", "indexer_executed_20260102_000000.txt"),
    ]
    assert scan_library_health(root, lambda: True) is None   # stops when asked


def test_home_scan_is_a_read_only_job(make_workspace, tmp_path, threads, monkeypatch):
    import gui.workers as workers

    started = []
    real = workers.start_worker

    def recording(thread, **kwargs):
        started.append((type(thread).__name__, kwargs))
        return real(thread, **kwargs)

    monkeypatch.setattr(workers, "start_worker", recording)
    root = _home_library(tmp_path / "Lib")
    ws = make_workspace("home", library_path=root)
    assert started and started[0][0] == "HomeScanWorker"
    kwargs = started[0][1]
    assert kwargs["ask_before_quit"] is False and kwargs["group"] is None
    assert wait_until(lambda: ws.health() is not None, 5000)
    assert threads.wait_all(5000) == []
    process_events(10)
    texts = [w.text() for w in ws.findChildren(QtWidgets.QLabel)]
    assert any("Manual Review/ — 2 tracks" in t for t in texts)
    assert any(t.startswith("Indexer run — ") for t in texts)
    assert ws._layout_label.text().startswith("✓")


def test_home_without_a_library_starts_nothing(make_workspace, threads):
    ws = make_workspace("home")
    assert threads.started == []
    assert ws._path_label.text() == "No library selected"
    ws.on_activated()
    ws.refresh()
    assert threads.started == []


def test_home_handles_an_empty_library(make_workspace, tmp_path, threads):
    empty = tmp_path / "Empty"
    empty.mkdir()
    ws = make_workspace("home", library_path=empty)
    assert wait_until(lambda: ws.health() is not None, 5000)
    assert threads.wait_all(5000) == []
    process_events(10)
    health = ws.health()
    assert not health.layout_ok and health.recent_runs == []
    assert set(health.folder_counts.values()) == {None}
    texts = [w.text() for w in ws.findChildren(QtWidgets.QLabel)]
    assert any(t.startswith("No runs yet") for t in texts)
    assert any("not created yet" in t for t in texts)
    assert ws._layout_label.text().startswith("✗")


def test_home_ignores_a_stale_scan(make_workspace, tmp_path, threads):
    from gui.workspaces.home import LibraryHealth

    root = make_album_tree(tmp_path / "Lib")
    ws = make_workspace("home", library_path=root)
    assert wait_until(lambda: ws.health() is not None, 5000)
    assert threads.wait_all(5000) == []
    ws._on_health_ready(ws._generation - 1, LibraryHealth(root="/elsewhere", layout_ok=True))
    assert ws.health().root == root


def test_home_open_buttons_and_quick_actions(main_window, tmp_path, threads, monkeypatch):
    import gui.workspaces.home as home_mod

    opened = []
    monkeypatch.setattr(home_mod, "open_path", lambda p: opened.append(p) or True)
    root = _home_library(tmp_path / "Lib")
    main_window.set_library(root)
    home = main_window._workspaces["home"]
    assert wait_until(lambda: home.health() is not None, 5000)
    assert threads.wait_all(10_000) == []
    process_events(10)
    assert "tracks" in home._stats_label.text(), "the top bar's stats reach Home"

    open_buttons = [b for b in home.findChildren(QtWidgets.QPushButton) if b.property("openPath")]
    open_buttons[0].click()
    assert opened and os.path.exists(opened[0])

    home._action_buttons["duplicates/find"].click()
    assert main_window.current_page().current_section() == "find"
    home._action_buttons["metadata/tag_fixer"].click()
    assert main_window.current_panel() is main_window._workspaces["tag_fixer"]
    assert threads.wait_all(10_000) == []


# ── Activity ────────────────────────────────────────────────────────────────

class _GateWorker(QtCore.QThread):
    def __init__(self):
        super().__init__()
        self.stop = threading.Event()
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        self.stop.set()

    def run(self):
        self.stop.wait(10)


def test_activity_lists_and_cancels_a_job(main_window, qtbot, threads):
    from gui.workers import start_worker

    main_window._on_nav_changed("activity")
    activity = main_window._workspaces["activity"]
    worker = _GateWorker()
    start_worker(worker, owner=main_window._workspaces["indexer"], name="Test job")
    process_events(10)
    table = activity._jobs_table
    rows = [table.item(r, 0).text() for r in range(table.rowCount())]
    assert "Test job" in rows
    row = rows.index("Test job")
    assert table.item(row, 1).text() == "Indexer"
    assert table.item(row, 2).text().endswith(" s")
    assert main_window._sidebar._buttons["activity"].badge == 1

    cancel = table.cellWidget(row, 3)
    cancel.click()
    assert worker.cancelled
    assert cancel.text() == "Stopping…" and not cancel.isEnabled()
    assert threads.wait_all(5000) == []
    assert wait_until(lambda: activity._jobs_empty.isVisible(), 3000)
    assert main_window._sidebar._buttons["activity"].badge == 0


def test_activity_keeps_the_log_and_the_problems(main_window):
    activity = main_window._workspaces["activity"]
    main_window._workspaces["indexer"]._log("first line", "info")
    main_window._workspaces["indexer"]._log("something odd", "warn")
    history = activity._history_text.toPlainText()
    assert "first line" in history and "something odd" in history
    assert "something odd" in activity._problems.toPlainText()
    assert "first line" not in activity._problems.toPlainText()
    activity._log_filter.setCurrentIndex(1)            # warnings and errors only
    assert "first line" not in activity._history_text.toPlainText()
    assert "something odd" in activity._history_text.toPlainText()
    # The log drawer keeps working too.
    assert "first line" in main_window._log_drawer._text.toPlainText()


def test_activity_shows_the_crash_log(make_workspace):
    import crash_logger

    ws = make_workspace("activity")
    assert "No crash log found" in ws._crash_text.toPlainText()
    os.makedirs(os.path.dirname(crash_logger.default_log_path()), exist_ok=True)
    with open(crash_logger.default_log_path(), "w", encoding="utf-8") as fh:
        fh.write("2026-11-07 [MainThread] Unhandled exception:\nValueError: boom\n")
    ws.refresh_problems()
    assert "ValueError: boom" in ws._crash_text.toPlainText()


# ── Settings page ───────────────────────────────────────────────────────────

def test_settings_page_saves_under_the_same_keys(make_workspace, tmp_path):
    import config

    lib = tmp_path / "Lib"
    lib.mkdir()
    ws = make_workspace("settings")
    ws._service_combo.setCurrentText("AcoustID")
    ws._api_key_entry.setText("KEY123")
    ws._service_combo.setCurrentText("Last.fm")
    ws._api_key_entry.setText("LFM")
    ws._service_combo.setCurrentText("MusicBrainz")
    ws._mb_app.setText("App")
    ws._mb_ver.setText("9")
    ws._mb_contact.setText("me@example.com")
    ws._library_entry.setText(str(lib))
    ws._near_dup_spin.setValue(0.25)
    ws._exact_dup_spin.setValue(0.15)
    ws._bg_gradient_cb.setChecked(False)
    assert ws._on_save()

    cfg = config.load_config()
    assert cfg["metadata_service"] == "MusicBrainz"
    assert cfg["metadata_api_key"] == "KEY123"
    assert cfg["lastfm_api_key"] == "LFM"
    assert cfg["musicbrainz_useragent"] == {"app": "App", "version": "9", "contact": "me@example.com"}
    assert cfg["library_root"] == str(lib)
    assert cfg["near_duplicate_threshold"] == pytest.approx(0.25)
    assert cfg["exact_duplicate_threshold"] == pytest.approx(0.15)
    assert cfg["bg_gradient_enabled"] is False
    assert ws.library_changed is True


def test_settings_page_save_runs_the_main_window_flow(main_window, monkeypatch):
    import gui.widgets.gradient_bg as gb

    calls = []
    indexer = main_window._workspaces["indexer"]
    monkeypatch.setattr(indexer, "on_settings_saved", lambda: calls.append(1))
    main_window._on_nav_changed("settings/appearance")
    settings = main_window._workspaces["settings"]
    settings._bg_gradient_cb.setChecked(False)
    settings._save_btn.click()
    assert calls == [1]
    assert gb._gradient_enabled() is False, "cached appearance settings were refreshed"
    assert "Saved" in settings._save_state.text()


def test_settings_page_refuses_a_missing_library(make_workspace, tmp_path, dialogs):
    import config

    ws = make_workspace("settings")
    saved = []
    ws.settings_saved.connect(lambda: saved.append(1))
    ws._library_entry.setText(str(tmp_path / "nope"))
    assert ws._on_save() is False
    assert saved == []
    assert "not found" in dialogs.calls_of("warning")[0].title.lower()
    assert config.load_config().get("library_root", "") == ""


def test_settings_page_keeps_edits_and_picks_up_outside_changes(main_window):
    import config

    settings = main_window._workspaces["settings"]
    main_window._on_nav_changed("settings/general")
    settings._near_dup_spin.setValue(0.33)               # unsaved edit
    main_window.select_page("home")
    cfg = config.load_config()
    cfg["exact_duplicate_threshold"] = 0.05
    config.save_config(cfg)
    main_window.select_page("settings")
    assert settings._near_dup_spin.value() == pytest.approx(0.33), "unsaved edits survive"
    settings._revert_btn.click()
    assert settings._exact_dup_spin.value() == pytest.approx(0.05)
    assert settings._near_dup_spin.value() != pytest.approx(0.33)

    cfg["exact_duplicate_threshold"] = 0.07                # changed elsewhere, no edits here
    config.save_config(cfg)
    main_window.select_page("home")
    main_window.select_page("settings")
    assert settings._exact_dup_spin.value() == pytest.approx(0.07)


def test_appearance_theme_picker_applies_instantly(main_window):
    from gui.themes.manager import get_manager

    main_window._on_theme_requested()
    picker = main_window._workspaces["settings"].theme_picker
    picker._apply_theme("pearl")
    assert get_manager().current.key == "pearl"
    selected = [sw.tokens.key for sw in picker._all_swatches if sw.isChecked()]
    assert selected == ["pearl"]


# ── Sound Map ───────────────────────────────────────────────────────────────

def test_sound_map_quality_report_follows_the_latest_analysis(main_window, dialogs):
    page = main_window._pages["sound_map"]
    analysis = main_window._workspaces["clustered"]
    assert not page._report_btn.isEnabled()
    analysis._set_result({"metrics": {"silhouette_score": 0.61}, "cluster_info": {}})
    assert page._report_btn.isEnabled()
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Rejected)
    main_window._on_nav_changed("graph")
    page._report_btn.click()
    (call,) = dialogs.calls_of("exec")
    assert type(call.widget).__name__ == "ClusterQualityReportDialog"
    analysis._set_result(None)
    assert not page._report_btn.isEnabled()


def test_sound_map_has_a_place_for_page_actions(main_window):
    page = main_window._pages["sound_map"]
    btn = QtWidgets.QPushButton("Save clusters as playlists")
    page.add_action(btn)
    assert btn.parentWidget() is page._bar
    main_window._on_nav_changed("sound_map/map")
    assert btn.isVisible()


def test_single_section_pages_have_no_section_bar(main_window):
    for key in ("home", "indexer", "player", "activity", "help"):
        page = main_window._pages[key]
        assert page.tab_bar() is None
        with pytest.raises(RuntimeError):
            page.add_action(QtWidgets.QPushButton("x"))


# ── Mirror → Verify ─────────────────────────────────────────────────────────

def test_mirror_verify_embeds_both_testers(make_workspace, tmp_path):
    ws = make_workspace("mirror_verify")
    assert set(ws.testers) == {"opus", "m4a"}
    bad = tmp_path / "garbage.opus"
    bad.write_bytes(b"not audio")
    ws.testers["opus"]._load_file(str(bad))
    assert "Could not read tags" in ws.testers["opus"]._meta_text.toPlainText()
    assert ws.findChildren(QtWidgets.QDialog) == [], "embedded, not a dialog"
