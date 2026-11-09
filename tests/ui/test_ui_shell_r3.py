"""Liquid Glass R3: the shell at the minimum window, style fixes, shared parts."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from gui import navigation
from gui.themes.manager import get_manager
from gui.themes.tokens import THEMES, TYPE
from ui_harness import process_events

REPO = Path(__file__).resolve().parents[2]
NARROW = (900, 600)


def _narrow(window, size=NARROW) -> None:
    window.resize(*size)
    process_events(20)


# ── Header at 900 px: stats first, then the search folds ─────────────────────

def test_header_keeps_the_library_chip_readable_at_the_minimum_width(main_window, library,
                                                                     threads):
    bar = main_window._top_bar
    main_window.set_library(str(library.root))
    assert threads.wait_all(10_000) == []
    process_events(20)
    assert bar.fit_mode() == bar.FULL
    assert bar._stats_box.isVisible() and bar.search_field().isVisible()
    assert not bar.search_button().isVisible()

    _narrow(main_window)
    assert bar.fit_mode() == bar.COMPACT
    assert not bar._stats_box.isVisible(), "the stats text goes first"
    assert not bar.search_field().isVisible() and bar.search_button().isVisible()
    assert bar.chip_width() >= bar.CHIP_MIN - 40, bar.chip_width()
    assert str(library.root) in bar._chip.toolTip() and "tracks" in bar._chip.toolTip(), \
        "the hidden stats live on in the chip's tooltip"
    btn = bar.search_button()
    assert btn.property("glyphName") == "search" and btn.accessibleName()

    main_window.resize(1300, 860)
    process_events(20)
    assert bar.fit_mode() == bar.FULL and bar.search_field().isVisible()


def test_header_drops_only_the_stats_when_that_is_enough(main_window, library, threads):
    bar = main_window._top_bar
    main_window.set_library(str(library.root))
    assert threads.wait_all(10_000) == []
    process_events(20)
    width = bar.width()
    # The widest header that is no longer FULL keeps its search pill.
    while width > 300 and bar._mode_for(width) == bar.FULL:
        width -= 4
    assert bar._mode_for(width) == bar.NO_STATS


def test_folded_search_opens_in_place_and_still_searches(main_window, library, threads):
    bar = main_window._top_bar
    main_window.set_library(str(library.root))
    assert threads.wait_all(10_000) == []
    _narrow(main_window)
    seen = []
    bar.search_requested.connect(seen.append)
    QtTest.QTest.mouseClick(bar.search_button(), QtCore.Qt.MouseButton.LeftButton)
    process_events(5)
    field = bar.search_field()
    assert field.isVisible() and not bar.search_button().isVisible()
    field.setText("aurora")
    field.returnPressed.emit()
    assert seen == ["aurora"]
    # Escape folds it back into the button.
    bar.open_search()
    QtTest.QTest.keyClick(field, QtCore.Qt.Key.Key_Escape)
    process_events(5)
    assert not field.isVisible() and bar.search_button().isVisible()


# ── Section tabs: compact, never a sideways scroll ───────────────────────────

def _pages_with_sections():
    return [p for p in navigation.PAGES if p.has_sections]


@pytest.mark.parametrize("width", [900, 1300])
def test_section_tabs_fit_without_scrolling(main_window, width):
    main_window.resize(width, 860 if width > 900 else 600)
    for spec in _pages_with_sections():
        main_window.select_page(spec.key)
        process_events(15)
        page = main_window.current_page()
        tabs = page.tab_bar()
        assert not tabs.usesScrollButtons()
        last = tabs.tabRect(tabs.count() - 1)
        assert last.right() <= tabs.width(), (spec.key, last, tabs.width())
        bar = page._bar
        assert bar.minimumSizeHint().width() <= bar.width(), spec.key
        scroll = page.window().findChild(QtWidgets.QScrollArea)
        assert scroll is None or scroll.horizontalScrollBar().maximum() == 0


def test_duplicates_tabs_go_compact_at_900_and_back(main_window):
    main_window.select_page("duplicates")
    page = main_window.current_page()
    tabs = page.tab_bar()
    process_events(10)
    assert not tabs.is_compact()
    assert all(not pill.is_dot() for pill in page.experimental_pills().values())
    _narrow(main_window)
    assert tabs.is_compact(), "six sections do not fit at full size in 900 px"
    pills = page.experimental_pills()
    assert len(pills) == 3 and all(pill.is_dot() for pill in pills.values())
    for key, pill in pills.items():
        index = page.section_keys().index(key)
        assert "experimental" in tabs.tabToolTip(index).lower(), "the dot keeps its meaning"
        assert pill.width() <= pill.DOT + 2
    for i in range(tabs.count()):
        assert tabs.tabRect(i).width() >= tabs.minimumTabSizeHint(i).width()
    for key in page.section_keys():
        page.show_section(key)
        process_events(5)
        current = tabs.currentIndex()
        assert tabs.tabRect(current).width() >= tabs._compact_width(current), \
            f"the current section ({key}) keeps its whole label"
        assert tabs.tabRect(tabs.count() - 1).right() <= tabs.width()
    page.show_section(page.section_keys()[0])
    main_window.resize(1300, 860)
    process_events(15)
    assert not tabs.is_compact()
    assert all(not pill.is_dot() for pill in page.experimental_pills().values())


def test_compact_tab_labels_elide_but_keep_their_text(qtbot):
    from gui.pages import SegmentedTabBar

    tabs = SegmentedTabBar()
    qtbot.addWidget(tabs)
    for label in ("Find duplicates", "Compare two files", "Pair review"):
        tabs.addTab(label)
    tabs.set_compact(True)
    full = sum(tabs.tabSizeHint(i).width() for i in range(tabs.count()))
    least = tabs.minimumSizeHint().width()
    assert least < full * 0.75, "labels may elide down to a few characters"
    tabs.resize(least, tabs.HEIGHT)
    tabs.show()
    process_events(5)
    assert [tabs.tabText(i) for i in range(3)] == ["Find duplicates", "Compare two files",
                                                    "Pair review"]
    assert tabs.tabRect(2).right() <= tabs.width()
    tabs.grab()   # paints the elided labels without error


# ── Style fixes ───────────────────────────────────────────────────────────────

class _Solid(QtWidgets.QWidget):
    COLOUR = QtGui.QColor(230, 60, 60)

    def paintEvent(self, event):  # noqa: N802
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), self.COLOUR)
        p.end()


def test_scroll_area_scrollbars_are_the_thin_glass_thumb(qtbot):
    """``QScrollArea > QWidget > QWidget`` also matched the scroll bars (in
    their containers), which made Qt draw old-style bars with arrow boxes."""
    get_manager().apply("glass")
    host = _Solid()
    qtbot.addWidget(host)
    host.resize(240, 200)
    lay = QtWidgets.QVBoxLayout(host)
    area = QtWidgets.QScrollArea()
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setWidgetResizable(True)
    inner = QtWidgets.QWidget()
    il = QtWidgets.QVBoxLayout(inner)
    for i in range(40):
        il.addWidget(QtWidgets.QLabel(f"row {i}"))
    area.setWidget(inner)
    lay.addWidget(area)
    host.show()
    process_events(5)
    bar = area.verticalScrollBar()
    bar.setValue(bar.maximum())           # the thumb at the bottom
    process_events(5)
    image = host.grab().toImage()
    top_left = bar.mapTo(host, QtCore.QPoint(0, 0))
    # Where an arrow box would be: only the backdrop (a transparent groove).
    for dy in range(2, 12):
        for dx in range(bar.width()):
            colour = image.pixelColor(top_left.x() + dx, top_left.y() + dy)
            assert colour == _Solid.COLOUR, (dx, dy, colour.name())


def test_log_text_paints_no_square_behind_its_rounded_inset(qtbot):
    get_manager().apply("glass")
    host = _Solid()
    qtbot.addWidget(host)
    host.resize(200, 120)
    lay = QtWidgets.QVBoxLayout(host)
    lay.setContentsMargins(10, 10, 10, 10)
    text = QtWidgets.QPlainTextEdit()
    text.setObjectName("logText")
    text.setPlainText("line\n" * 3)
    lay.addWidget(text)
    host.show()
    process_events(5)
    assert not text.viewport().autoFillBackground()
    vp = text.viewport().geometry()
    radius = 12
    # The viewport's corner lies inside the rounded corner.
    assert (radius - vp.left()) ** 2 + (radius - vp.top()) ** 2 < radius ** 2
    image = host.grab().toImage()
    origin = text.mapTo(host, QtCore.QPoint(0, 0))
    for x, y in ((1, 1), (text.width() - 2, 1), (1, text.height() - 2),
                 (text.width() - 2, text.height() - 2)):
        assert image.pixelColor(origin.x() + x, origin.y() + y) == _Solid.COLOUR, (x, y)
    inside = image.pixelColor(origin.x() + text.width() // 2, origin.y() + text.height() - 4)
    assert inside == QtGui.QColor(get_manager().current.log_bg)


@pytest.mark.parametrize("key", sorted(THEMES))
def test_progress_and_slider_grooves_show_on_their_card(key):
    from gui.themes.style import _TRACK_MIN_CONTRAST, _luminance, track_colour

    get_manager().apply(key)
    t = get_manager().current
    m = t.material
    groove = track_colour(t, m)
    card = QtGui.QColor(m.card.fill) if m.card.fill.startswith("#") else None
    if groove.alpha() < 255 or card is None:
        return     # translucent on glass
    lo, hi = sorted((_luminance(groove), _luminance(card)))
    assert (hi + 0.05) / (lo + 0.05) >= _TRACK_MIN_CONTRAST - 1e-6, key


def test_midnight_groove_is_no_longer_the_card_colour():
    from gui.themes.style import track_colour

    get_manager().apply("midnight")
    t = get_manager().current
    assert track_colour(t, t.material).name() != QtGui.QColor(t.progress_bg).name()


def test_log_strip_has_a_painted_shadow(main_window):
    backdrop = main_window._backdrop
    assert (main_window._log_drawer, "card") in backdrop._floating
    assert main_window._log_drawer.graphicsEffect() is None, "painted, never an effect"


# ── Shared parts ──────────────────────────────────────────────────────────────

def test_shared_parts_are_defined_once():
    """The private copies (metadata_views, organize_parts, dupes_parts) are
    gone; each shared part has one definition under gui/."""
    gui = REPO / "gui"
    assert not (gui / "workspaces" / "metadata_views.py").exists()
    assert not (gui / "workspaces" / "organize_parts.py").exists()
    assert not (gui / "widgets" / "dupes_parts.py").exists()
    source = "\n".join(p.read_text(encoding="utf-8") for p in gui.rglob("*.py")
                       if "__pycache__" not in p.parts)
    for definition in ("class FlowLayout", "class GlassRowDelegate", "class StepPills",
                       "class GlyphLabel", "class ToneBadge",
                       "class ReflowBox", "class ReflowSplitter", "def hero_title",
                       "def style_item_view", "class PathLabel", "class _StepPills"):
        count = len(re.findall(rf"^\s*{definition}\b", source, re.M))
        expected = 0 if definition in ("class PathLabel", "class _StepPills") else 1
        assert count == expected, (definition, count)
    assert "class TileGrid" in (gui / "widgets" / "glass_parts.py").read_text(encoding="utf-8")
    for role in ("organizeHeroTitle", "metaHeroTitle", "dupHeroTitle", "soundMapHeroTitle",
                 "playlistHeroTitle", "metaGlass", "dupStepRow", "organizeLog"):
        assert role not in source, role


def test_hero_title_keeps_the_title_size_under_the_stylesheet(main_window):
    main_window._on_nav_changed("duplicates")
    process_events(5)
    titles = [w for w in main_window.findChildren(QtWidgets.QLabel)
              if w.objectName() == "heroTitle"]
    assert len(titles) >= 8, "every hero card uses the shared role"
    for title in titles:
        assert title.font().pixelSize() == TYPE["title"].px, title.text()


def test_step_pills_shrink_the_other_steps_when_narrow(qtbot):
    from gui.widgets.glass_parts import StepPills

    steps = StepPills(["Choose folders", "Scan and match", "Build and execute"])
    qtbot.addWidget(steps)
    steps.set_current(1)
    full = steps.sizeHint().width()
    assert steps.minimumSizeHint().width() < full
    steps.resize(full, 30)
    steps.show()
    process_events(2)
    assert not steps.is_compact()
    steps.resize(full - 10, 30)
    process_events(2)
    assert steps.is_compact()
    labelled = [w > steps._H for w in steps._widths()]
    assert labelled[1] and sum(labelled) == 2, "one step gives way, nearest-first keep theirs"
    steps.resize(steps.minimumSizeHint().width(), 30)
    process_events(2)
    assert steps.is_compact()
    assert [w > steps._H for w in steps._widths()] == [False, True, False]
    steps.grab()
    assert steps.accessibleName() == "Step 2 of 3: Scan and match"


def test_one_hover_tracker_per_view(qtbot):
    from gui.widgets.glass_views import GlassRowDelegate, GlassTree, hover_tracker

    tree = GlassTree()
    qtbot.addWidget(tree)
    tree.setColumnCount(2)
    tree.setItemDelegateForColumn(1, GlassRowDelegate(tree))
    for i in range(3):
        tree.addTopLevelItem(QtWidgets.QTreeWidgetItem([f"row {i}", "x"]))
    tree.resize(300, 200)
    tree.show()
    process_events(2)
    assert len(tree.findChildren(type(hover_tracker(tree)))) == 1
    vp = tree.viewport()
    pos = tree.visualItemRect(tree.topLevelItem(1)).center()
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(pos),
                              QtCore.QPointF(vp.mapToGlobal(pos)), QtCore.Qt.MouseButton.NoButton,
                              QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier)
    QtWidgets.QApplication.sendEvent(vp, event)
    assert tree.hover_row() == 1


def test_path_label_and_field_share_one_label(qtbot):
    from gui.widgets.glass import ElidedLabel, path_label
    from gui.widgets.path_field import PathField

    label = path_label("", "Nothing chosen")
    qtbot.addWidget(label)
    assert label.elide_mode() == QtCore.Qt.TextElideMode.ElideMiddle
    assert label.accessibleName() == "Nothing chosen"
    label.setText("/music/a/very/long/path")
    assert label.toolTip() == "/music/a/very/long/path"
    named = path_label("")
    named.setAccessibleName("File A")
    named.setText("/x/y.flac")
    assert named.accessibleName() == "File A", "an owner's name is kept"
    field = PathField("Choose a folder")
    qtbot.addWidget(field)
    assert isinstance(field.label, ElidedLabel)
    assert field.label.placeholder() == "Choose a folder"


def test_flow_layout_wraps_instead_of_widening(qtbot):
    from gui.widgets.layouts import FlowLayout

    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    flow = FlowLayout(host, spacing=8, line_spacing=6)
    buttons = [QtWidgets.QPushButton(f"Button number {i}") for i in range(5)]
    for b in buttons:
        flow.addWidget(b)
    one_line = flow.sizeHint().width()
    assert flow.minimumSize().width() < one_line
    host.resize(one_line // 2, 300)
    host.show()
    process_events(2)
    assert len({b.y() for b in buttons}) > 1
    assert all(b.geometry().right() <= host.width() for b in buttons)
