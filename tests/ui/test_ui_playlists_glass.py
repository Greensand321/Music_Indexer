"""The Playlists page in Liquid Glass (redesign R2): a hero card for the
generator, options in a well, the preview as glass rows with a summary,
an empty state before the first run, glyph buttons and no overflow."""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets

from gui.workspaces.playlists import _ROLE_BADGE
from playlists_helpers import make_library
from ui_harness import process_events


def _ws(make_workspace, tmp_path):
    lib = make_library(tmp_path / "lib")
    return make_workspace("playlists", library_path=lib), lib


def _wait_idle(ws, qtbot, timeout=5000):
    qtbot.waitUntil(lambda: not ws.is_busy(), timeout=timeout)
    process_events(20)


def test_no_repeated_title_and_one_hero_card(make_workspace, tmp_path):
    ws, _ = _ws(make_workspace, tmp_path)
    texts = [lbl.text() for lbl in ws.findChildren(QtWidgets.QLabel)]
    assert "Playlist Generator" not in texts, "the shell header already says Playlists"
    assert not any(lbl.objectName() == "sectionTitle" for lbl in ws.findChildren(QtWidgets.QLabel))
    heroes = [c for c in ws._cards if c.objectName() == "headerCard"]
    assert len(heroes) == 1
    for old in ("Clustered", "Music Graph", "Compression"):
        assert not any(old in t for t in texts)


def test_empty_state_before_the_first_run(make_workspace, tmp_path):
    ws, _ = _ws(make_workspace, tmp_path)
    assert ws._results_stack.currentWidget() is ws._results_empty
    assert ws._results_empty.title() == "No playlists yet"
    assert not ws._log_area.isVisible()
    ws._details_btn.click()
    assert ws._log_area.isVisible()


def test_each_kind_shows_its_own_options_and_button(make_workspace, tmp_path):
    ws, _ = _ws(make_workspace, tmp_path)
    seen = []
    for key in ("folder", "tempo", "autodj", "repair"):
        ws.set_mode(key)
        assert ws.mode() == key
        seen.append(ws._run_btn.text())
        assert ws._options.currentIndex() == ("folder", "tempo", "autodj", "repair").index(key)
    assert len(set(seen)) == 4
    assert ws._run_buttons == [ws._run_btn]


def test_the_primary_button_runs_the_chosen_kind(make_workspace, tmp_path, monkeypatch):
    ws, _ = _ws(make_workspace, tmp_path)
    started = []
    monkeypatch.setattr(ws, "_on_run", started.append)
    for key in ("repair", "tempo"):
        ws.set_mode(key)
        ws._run_btn.click()
    assert started == ["repair", "tempo"]


def test_preview_shows_glass_rows_and_a_summary(make_workspace, tmp_path, qtbot, dialogs):
    ws, lib = _ws(make_workspace, tmp_path)
    existing = lib / "Playlists" / "Artist A_Album 1.m3u"
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("user edits\n", encoding="utf-8")
    dialogs.answer_no()
    ws._on_run("folder")
    _wait_idle(ws, qtbot)
    assert ws._results_stack.currentIndex() == 1
    n = ws._results_list.count()
    assert n >= 2
    assert ws._stats.tile("playlists").value() == str(n)
    assert ws._stats.tile("replaced").value() == "1"
    assert ws._stats.tile("replaced").tone() == "warning"
    badges = sorted(ws._results_list.item(i).data(_ROLE_BADGE) for i in range(n))
    assert "replace" in badges and "new" in badges
    assert ws._results_note.text() == "Nothing was written."


def test_written_rows_say_so(make_workspace, tmp_path, qtbot, dialogs):
    ws, lib = _ws(make_workspace, tmp_path)
    dialogs.answer_yes()
    ws._on_run("folder")
    _wait_idle(ws, qtbot)
    _wait_idle(ws, qtbot)
    badges = {ws._results_list.item(i).data(_ROLE_BADGE) for i in range(ws._results_list.count())}
    assert badges == {"written"}
    assert ws._results_title.text() == "Written"
    assert ws._results_note.text().startswith("Wrote ")


def test_cancel_and_errors_show_in_the_preview_card(make_workspace, tmp_path, qtbot, dialogs):
    ws, _ = _ws(make_workspace, tmp_path)
    ws._autodj_seed.setText(str(tmp_path / "not-in-library.mp3"))
    ws._on_run("autodj")
    _wait_idle(ws, qtbot)
    assert ws._results_stack.currentWidget() is ws._results_empty
    assert ws._results_empty.title() == "The playlists could not be prepared"
    assert "Seed track is not in the library" in ws._results_empty.message()


def test_glyph_buttons_and_no_emoji(make_workspace, tmp_path):
    ws, _ = _ws(make_workspace, tmp_path)
    for btn in ws.findChildren(QtWidgets.QAbstractButton):
        assert not any(ch in btn.text() for ch in "🎵🔧✕💾📂"), btn.text()
        if not btn.text() and isinstance(btn, QtWidgets.QPushButton) and not btn.isHidden():
            assert btn.property("glyphName") and btn.toolTip() and btn.accessibleName()
    assert all(not w.styleSheet() for w in ws.findChildren(QtWidgets.QWidget))


@pytest.mark.parametrize("size", [(900, 600), (1300, 860), (1600, 1000)])
def test_playlists_page_has_no_horizontal_scroll(main_window, size):
    main_window.resize(*size)
    main_window._on_nav_changed("playlists")
    process_events(30)
    ws = main_window._workspaces["playlists"]
    for key in ("folder", "tempo", "autodj", "repair"):
        ws.set_mode(key)
        process_events(10)
        assert ws._scroll_area.horizontalScrollBar().maximum() == 0, key
