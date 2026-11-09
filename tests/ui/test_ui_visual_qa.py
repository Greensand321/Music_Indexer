"""Visual QA regressions found in the R3 pass (every theme, light classic
themes especially)."""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from gui.themes.manager import get_manager
from gui.themes.tokens import THEMES
from ui_harness import process_events

STYLED = QtCore.Qt.WidgetAttribute.WA_StyleSheetTarget


def _page_scroll_areas(window) -> list[QtWidgets.QScrollArea]:
    return [a for a in window.findChildren(QtWidgets.QScrollArea)
            if a.objectName() == "workspaceScroll"]


@pytest.mark.parametrize("theme", ["glass", "pearl"])
def test_page_scroll_bars_are_painted_by_the_style_not_the_stylesheet(main_window, theme):
    """``QScrollArea > QWidget > QWidget`` (the transparent content rule)
    also matched the scroll bars inside ``qt_scrollarea_vcontainer``, and a
    stylesheet rule on a QScrollBar hands it to Qt's stylesheet painter:
    arrow boxes and a 1 px frame instead of AlphaDEXStyle's thin rounded
    thumb, on every scrolling page and the sidebar."""
    get_manager().apply(theme)
    for key in ("home", "duplicates", "player", "settings"):
        main_window._on_nav_changed(key)
    process_events(50)
    areas = main_window.findChildren(QtWidgets.QScrollArea)
    assert len(_page_scroll_areas(main_window)) >= 3
    for area in areas:
        for bar in (area.verticalScrollBar(), area.horizontalScrollBar()):
            bar.ensurePolished()
            assert not bar.testAttribute(STYLED), (
                f"{area.objectName() or type(area).__name__}: its scroll bar is stylesheet-painted")
        content = area.widget()
        if content is not None:
            content.ensurePolished()
            assert content.testAttribute(STYLED), "the content widget keeps its transparent rule"


def test_classic_hover_rows_follow_the_card(main_window):
    """Classic themes' hover overlay is a tint of the card (as the glass
    overlays are), not the sidebar's hover colour, which in Midnight *is*
    the card colour: Activity's "Earlier this session" rows were invisible
    in Midnight and heavy grey in Pearl."""
    for key in ("midnight", "pearl"):
        t = THEMES[key]
        assert t.material.hover.lower() != t.card_bg.lower()
        assert t.material.hover.lower() != t.sidebar_hover.lower()
