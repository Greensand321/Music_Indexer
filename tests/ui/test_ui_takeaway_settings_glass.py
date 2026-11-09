"""Liquid Glass R2 for Mirror (Build / Verify), Settings and the dialog chrome.

Layout and look contracts (behaviour stays covered by the older tests):
no repeated page titles, one primary per card, paths as elided rows with
glyph buttons, summaries as stat tiles, problems as glass rows, the service
rows and the backdrop pill, a clean responsive swatch grid, no horizontal
scrolling at the minimum window, and dialogs that are opaque but part of the
glass family (palette roles, raised content over a footer band, the default
button as the primary pill, painted message-box icons).
"""
from __future__ import annotations

import os

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from gui.themes.manager import get_manager


def _close(a: QtGui.QColor, b: QtGui.QColor, tol: int = 3) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a.getRgb()[:3], b.getRgb()[:3]))


def _labels(widget, name):
    return [lbl for lbl in widget.findChildren(QtWidgets.QLabel) if lbl.objectName() == name]


# ── Widths ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [(900, 600), (1300, 860)])
def test_my_pages_never_scroll_sideways(main_window, size):
    main_window.resize(*size)
    for key, panel in (("mirror/build", "compression"), ("mirror/verify", "mirror_verify"),
                       ("settings/services", "settings"), ("settings/general", "settings"),
                       ("settings/appearance", "settings")):
        main_window._on_nav_changed(key)
        ws = main_window._workspaces[panel]
        QtWidgets.QApplication.processEvents()
        bar = ws._scroll_area.horizontalScrollBar()
        assert bar.maximum() == 0, f"{key} scrolls sideways at {size}"


def test_verify_cards_stack_when_narrow(make_workspace):
    ws = make_workspace("mirror_verify")
    ws.resize(1100, 760)
    QtWidgets.QApplication.processEvents()
    assert ws._row.direction() == QtWidgets.QBoxLayout.Direction.LeftToRight
    ws.resize(640, 760)
    QtWidgets.QApplication.processEvents()
    assert ws._row.direction() == QtWidgets.QBoxLayout.Direction.TopToBottom


# ── Mirror → Build ────────────────────────────────────────────────────────────

def test_mirror_build_hero_paths_and_buttons(make_workspace, library):
    from gui.widgets.path_field import PathField

    ws = make_workspace("compression", library_path=library.root)
    assert not _labels(ws, "sectionTitle"), "the shell header already names the page"
    assert len(_labels(ws, "sectionSubtitle")) == 1
    assert ws._hero.objectName() == "headerCard"
    for field in (ws._source_edit, ws._dest_edit):
        assert isinstance(field, PathField)
        assert field.button.property("glyphName") == "folder"
        assert field.button.toolTip() and field.button.accessibleName()
    assert ws._source_edit.toolTip() == str(library.root), "the full path is the tooltip"
    primaries = [b for b in ws._hero.findChildren(QtWidgets.QPushButton)
                 if b.objectName() == "primaryBtn"]
    assert primaries == [ws._run_btn] and ws._run_btn.text() == "Build mirror"
    assert ws._update_cb.isChecked()
    assert ws._orphans_btn.objectName() == "dangerBtn"
    for btn, glyph in ((ws._estimate_btn, "search"), (ws._run_btn, "play"),
                       (ws._cancel_btn, "stop"), (ws._report_btn, "external"),
                       (ws._orphans_btn, "trash")):
        assert btn.property("glyphName") == glyph
    for btn in ws.findChildren(QtWidgets.QAbstractButton):
        assert not any(ord(ch) > 0x2000 and ch not in "…–—·" for ch in btn.text()), btn.text()
    visible = [w for w in ws.findChildren(QtWidgets.QLabel) if w.isVisibleTo(ws)]
    assert not any("Compression" in w.text() or "Utilities" in w.text() for w in visible)
    assert ws._empty.isVisibleTo(ws) and not ws._result_lbl.isVisibleTo(ws)


def test_mirror_run_summary_is_stat_tiles(make_workspace, library, tmp_path):
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(tmp_path / "Mirror"))
    ws._on_finished({
        "total": 40, "converted": 12, "copied": 25, "skipped": 3, "errors": 1, "stale": 4,
        "updated": 2, "cancelled": False, "error_files": [], "skipped_files": [],
        "orphan_files": [], "partials_removed": 1,
    })
    values = {key: ws._run_tile(key).value() for key, _l, _t in ws._RUN_TILES}
    assert values == {"converted": "12", "copied": "25", "skipped": "3", "errors": "1",
                      "updated": "2", "stale": "2", "partials": "1", "orphans": "0"}
    assert ws._run_tile("errors").tone() == "danger"
    assert ws._run_stats.isVisibleTo(ws) and ws._run_stats_more.isVisibleTo(ws)
    assert not ws._empty.isVisibleTo(ws)
    text = ws._result_lbl.text()
    assert text.startswith("Done with errors")
    assert "copied 25" not in text, "the counts live in the tiles"
    assert "2 file(s) older than their changed source were kept" in text, "explanations stay"


def test_orphan_confirmation_lists_at_most_fifteen(make_workspace, library, dialogs, tmp_path):
    ws = make_workspace("compression", library_path=library.root)
    dest = str(tmp_path / "Mirror")
    ws._dest_edit.setText(dest)
    paths = [os.path.join(dest, f"gone{i:02d}.opus") for i in range(20)]
    ws._set_orphans((str(library.root), dest, paths))
    assert ws._orphans_btn.isVisibleTo(ws) and ws._orphans_btn.text().startswith("Remove 20")
    dialogs.answer_no()
    ws._orphans_btn.click()
    text = dialogs.calls_of("question")[-1].text
    assert "gone14.opus" in text and "gone15.opus" not in text
    assert "and 5 more" in text
    assert ws._orphan_worker is None, "declining deletes nothing"


# ── PathField ────────────────────────────────────────────────────────────────

def test_path_field_behaves_like_a_line_edit(qtbot):
    from gui.widgets.path_field import PathField

    field = PathField("Nothing yet", clearable=True)
    qtbot.addWidget(field)
    field.resize(200, field.HEIGHT)
    field.show()
    seen, browse = [], []
    field.textChanged.connect(seen.append)
    field.browse_requested.connect(lambda: browse.append(1))
    long = "/very/long/path/" + "x" * 200 + "/Music"
    field.setText(long)
    assert field.text() == long and seen == [long]
    assert field.toolTip() == long and field.label.text() == long, "only painting elides"
    assert field.minimumSizeHint().width() < 200, "never widens its card"
    assert field.clear_button.isVisibleTo(field)
    field.clear_button.click()
    assert field.text() == "" and seen[-1] == ""
    assert not field.clear_button.isVisibleTo(field)
    assert field.label.text() == "" and field.label.placeholder() == "Nothing yet", \
        "an empty field paints its placeholder"
    assert field.toolTip() == "Nothing yet"
    field.button.click()
    qtbot.mouseClick(field, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(30, 20))
    assert browse == [1, 1]


# ── Mirror → Verify ───────────────────────────────────────────────────────────

def test_verify_shows_stats_and_problem_rows(make_workspace, tmp_path, monkeypatch):
    import gui.dialogs.media_tester_dialog as mt

    ws = make_workspace("mirror_verify")
    assert not _labels(ws, "sectionTitle")
    tester = ws.testers["opus"]
    assert tester._empty.isVisibleTo(tester) and not tester._results.isVisibleTo(tester)

    bad = tmp_path / "garbage.opus"
    bad.write_bytes(b"not audio")
    tester._load_file(str(bad))
    assert tester._file_entry.text() == str(bad)
    assert tester._results.isVisibleTo(tester) and not tester._empty.isVisibleTo(tester)
    (tone, text), = tester.problems()
    assert tone == "danger" and text.startswith("Could not read tags")
    assert tester._stats.tile("problems").value() == "1"
    assert tester._stats.tile("problems").tone() == "danger"

    monkeypatch.setattr(mt, "describe_media", lambda p: (["Title: Sparse"], None, None))
    tester._load_file(str(tmp_path / "sparse.opus"))
    assert tester.problems() == [("warning", "No artist tag"), ("warning", "No album tag"),
                                 ("warning", "No cover art")]
    assert tester._stats.tile("tags").value() == "1"
    assert tester._stats.tile("art").value() == "No"
    assert "Title: Sparse" in tester._meta_text.toPlainText()

    png = QtCore.QBuffer()
    png.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img = QtGui.QImage(8, 8, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor(0, 128, 255))
    img.save(png, "PNG")
    full = ["Title: T", "Artist: A", "Album: B"]
    monkeypatch.setattr(mt, "describe_media", lambda p: (full, bytes(png.data()), None))
    tester._load_file(str(tmp_path / "full.opus"))
    assert tester.problems() == []
    rows = tester._problem_rows
    assert [r.tone() for r in rows] == ["success"], "an all-clear row instead of nothing"
    assert tester._stats.tile("art").value() == "Yes"
    assert not tester._art_lbl.pixmap().isNull()


# ── Settings ─────────────────────────────────────────────────────────────────

def test_settings_header_row_and_cards(make_workspace):
    ws = make_workspace("settings")
    assert not [lbl for lbl in _labels(ws, "sectionTitle") if lbl.text() == "Settings"]
    subtitles = _labels(ws, "sectionSubtitle")
    assert len(subtitles) == 1 and len(subtitles[0].text()) < 90
    assert ws._save_btn.objectName() == "primaryBtn"
    primaries = [b for b in ws.findChildren(QtWidgets.QPushButton)
                 if b.objectName() == "primaryBtn" and b is not ws._save_btn]
    assert [b.text() for b in primaries] in ([], ["Select"], ["Active"]), "Auto's Select only"
    general = ws._tabs.widget(ws.SECTIONS.index("general"))
    assert general.isAncestorOf(ws._reduce_motion_cb), "Reduce motion lives in General"
    assert general.isAncestorOf(ws._library_entry)
    appearance = ws._tabs.widget(ws.SECTIONS.index("appearance"))
    assert appearance.isAncestorOf(ws._backdrop_combo)


def test_service_rows_keep_the_unavailable_services_disabled(make_workspace):
    ws = make_workspace("settings")
    rows = {row.text(): row for row in ws._service_combo.rows()}
    assert list(rows) == ["AcoustID", "MusicBrainz", "Last.fm", "Spotify", "Gracenote"]
    for name in ("Spotify", "Gracenote"):
        assert not rows[name].isEnabled()
        assert "not yet implemented" in rows[name].toolTip() and "ROADMAP" in rows[name].toolTip()
    ws._service_combo.setCurrentText("Spotify")
    assert ws._service_combo.currentText() == "AcoustID", "an unavailable service is never chosen"
    ws.show()
    rows["MusicBrainz"].click()
    assert ws._service_combo.currentText() == "MusicBrainz"
    assert ws._mb_frame.isVisibleTo(ws) and not ws._acoustid_frame.isVisibleTo(ws)
    rows["Last.fm"].click()
    assert ws._acoustid_frame.isVisibleTo(ws) and ws._api_key_entry.echoMode() == \
        QtWidgets.QLineEdit.EchoMode.Password


def test_backdrop_is_a_segmented_pill(make_workspace):
    ws = make_workspace("settings")
    bar = ws._backdrop_combo
    assert isinstance(bar, QtWidgets.QTabBar) and bar.objectName() == "segmentedChoice"
    assert [bar.tabText(i) for i in range(bar.count())] == ["Album art", "Gradient", "Plain"]
    assert [bar.tabData(i) for i in range(bar.count())] == ["art", "gradient", "plain"]
    bar.setCurrentIndex(bar.findData("gradient"))
    assert ws.form.backdrop_mode() == "gradient" and ws.form.is_dirty()


def test_threshold_rows_have_units_and_help(make_workspace):
    ws = make_workspace("settings")
    box = ws.form.thresholds_widget
    for spin in (ws._near_dup_spin, ws._exact_dup_spin):
        assert spin.toolTip() and box.isAncestorOf(spin)
    units = [lbl.text() for lbl in box.findChildren(QtWidgets.QLabel) if "0–1" in lbl.text()]
    assert len(units) == 2
    dots = [w for w in box.findChildren(QtWidgets.QWidget) if w.objectName() == "helpDot"]
    assert len(dots) == 2 and all(d.toolTip() for d in dots)


def test_test_connection_status_uses_a_tone_not_a_stylesheet(make_workspace):
    from gui.workspaces.label_tone import tone_of

    ws = make_workspace("settings")
    ws.form._apply_test_result(False, "AcoustID: the API key was rejected")
    assert tone_of(ws._test_status) == "danger" and ws._test_status.styleSheet() == ""
    colour = ws._test_status.palette().color(QtGui.QPalette.ColorRole.WindowText)
    assert colour == QtGui.QColor(get_manager().current.danger)


# ── Theme picker ─────────────────────────────────────────────────────────────

def test_picker_order_and_a_clean_swatch_grid(make_workspace):
    from gui.themes.picker import _SwatchCard

    ws = make_workspace("settings")
    ws.set_section("appearance")
    picker = ws.theme_picker
    lay = picker.layout()
    order = [lay.itemAt(i).widget() for i in range(lay.count()) if lay.itemAt(i).widget()]
    names = [w.objectName() or (w.text() if isinstance(w, QtWidgets.QLabel) else "") for w in order]
    assert names.index("glassRow") < names.index("autoCard") < names.index("sectionHeader")
    assert "◑" not in "".join(lbl.text() for lbl in picker.findChildren(QtWidgets.QLabel))

    def check(width):
        picker.resize(width, picker.heightForWidth(width) if picker.hasHeightForWidth() else 900)
        ws.resize(width + 120, 900)
        QtWidgets.QApplication.processEvents()
        grids = [w for w in picker.findChildren(QtWidgets.QWidget) if w.objectName() == "swatchGrid"]
        assert len(grids) == 2
        widths = set()
        for grid in grids:
            cards = grid.findChildren(_SwatchCard)
            rects = [c.geometry() for c in cards]
            for i, a in enumerate(rects):
                assert grid.rect().contains(a), "inside its grid"
                for b in rects[i + 1:]:
                    assert not a.intersects(b), "swatches never overlap"
            widths |= {r.width() for r in rects}
            assert len({r.height() for r in rects}) == 1
            lefts = sorted({r.left() for r in rects})
            assert all(r.left() in lefts[:len(lefts)] for r in rects)
            for card in cards:
                assert card.preview_rect().bottom() <= card.name_rect().top(), \
                    "the name never overlaps the preview"
        assert len(widths) == 1, "both groups share one cell width"
        return grids

    wide = check(980)
    rows_wide = len({c.y() for c in wide[0].findChildren(_SwatchCard)})
    narrow = check(520)
    rows_narrow = len({c.y() for c in narrow[0].findChildren(_SwatchCard)})
    assert rows_narrow > rows_wide, "the grid reflows to more rows when narrow"


# ── Dialog chrome ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("theme", ["glass", "midnight"])
def test_dialog_chrome_palette_and_roles(qtbot, dialogs, theme):
    from gui.themes.style import dialog_footer_top, dialog_surface_colours
    from gui.themes.tokens import TYPE

    mgr = get_manager()
    mgr.apply(theme)
    t = mgr.current
    base, raised, _line = dialog_surface_colours(t)
    window = QtGui.QPalette.ColorRole.Window
    assert base == QtGui.QColor(t.material.backdrop_base)
    assert raised == QtGui.QColor(t.card_bg)

    box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Question, "Remove Orphaned Files",
                                "Delete these 2 file(s) from the mirror?",
                                QtWidgets.QMessageBox.StandardButton.Yes
                                | QtWidgets.QMessageBox.StandardButton.No)
    box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
    plain = QtWidgets.QDialog()
    lay = QtWidgets.QVBoxLayout(plain)
    lay.addWidget(QtWidgets.QLabel("Body"))
    lay.addWidget(QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel))
    plain.resize(360, 220)
    for dlg in (box, plain):
        qtbot.addWidget(dlg)
        dlg.show()
        qtbot.waitExposed(dlg)
        colour = dlg.palette().color(window)
        assert colour.alpha() == 255, "top-level dialogs stay opaque"
        assert colour == base, "the window role is the material's backdrop_base"
        assert dlg.testAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
        footer = dialog_footer_top(dlg)
        assert footer is not None
        img = dlg.grab().toImage()
        assert _close(img.pixelColor(3, 3), raised), "raised content surface"
        assert _close(img.pixelColor(3, img.height() - 3), base), "footer band on the base fill"
        assert footer < dlg.findChild(QtWidgets.QDialogButtonBox).y()

    label = box.findChild(QtWidgets.QLabel, "qt_msgbox_label")
    assert label.font().pixelSize() == TYPE["body"].px
    assert label.palette().color(QtGui.QPalette.ColorRole.WindowText) == QtGui.QColor(t.text_primary)
    assert box.iconPixmap().width() == 40 * box.iconPixmap().devicePixelRatio()

    def fill(button):
        image = button.grab().toImage()
        return image.pixelColor(button.height() // 2, button.height() // 2)

    no = box.button(QtWidgets.QMessageBox.StandardButton.No)
    yes = box.button(QtWidgets.QMessageBox.StandardButton.Yes)
    assert _close(fill(no), QtGui.QColor(t.accent)), "the default button is the primary pill"
    assert not _close(fill(yes), QtGui.QColor(t.accent))
    assert yes.icon().isNull(), "no icons on dialog buttons"


def test_dialog_icons_and_mnemonics_are_painted():
    from gui.themes.style import _strip_mnemonic, message_icon

    t = get_manager().current
    for kind in ("information", "question", "warning", "critical"):
        icon = message_icon(kind, t)
        assert not icon.isNull() and icon.availableSizes()
    assert _strip_mnemonic("&Yes") == "Yes" and _strip_mnemonic("R&&B") == "R&B"


def test_my_dialogs_put_buttons_on_the_footer(qtbot, dialogs):
    from gui.dialogs.crash_log_dialog import CrashLogDialog
    from gui.dialogs.settings_drawer import SettingsDrawer
    from gui.themes.style import dialog_footer_top

    for dlg in (CrashLogDialog(), SettingsDrawer()):
        qtbot.addWidget(dlg)
        dlg.resize(640, 560)
        dlg.show()
        qtbot.waitExposed(dlg)
        box = dlg.findChild(QtWidgets.QDialogButtonBox)
        footer = dialog_footer_top(dlg)
        assert footer is not None and footer < box.y()
        others = [w for w in dlg.findChildren(QtWidgets.QWidget,
                                              options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly)
                  if w is not box and not w.isWindow() and w.isVisible()]
        assert all(w.geometry().bottom() < footer for w in others), "the band cuts nothing"
        assert not [w for w in dlg.findChildren(QtWidgets.QWidget) if w.styleSheet()]
    drawer = dlg
    assert drawer._save_btn.objectName() == "primaryBtn" and drawer._save_btn.isDefault()
