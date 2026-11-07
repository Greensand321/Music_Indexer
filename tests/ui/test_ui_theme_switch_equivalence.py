"""A live theme switch must look exactly like starting in that theme.

The fast theme switch (DS-04) re-polishes only some widgets. Qt's stylesheet
engine snapshots a widget's palette when it polishes a widget that has its
own inline stylesheet, so any such widget (and everything under it) that the
switch skipped keeps the *previous* theme's colours. Example: the sidebar
section headers stayed Midnight-white on Pearl, i.e. invisible.

This test builds one window, switches it to the target theme, builds a
second window directly in the target theme, and compares the resolved
palette colours of every widget pair. Any difference is a widget the switch
failed to reach.
"""
from __future__ import annotations

import pytest
from PySide6 import QtGui, QtWidgets

from ui_harness import process_events

ROLES = (
    QtGui.QPalette.ColorRole.WindowText,
    QtGui.QPalette.ColorRole.Window,
    QtGui.QPalette.ColorRole.Text,
    QtGui.QPalette.ColorRole.Base,
    QtGui.QPalette.ColorRole.ButtonText,
)


def _signature(window: QtWidgets.QWidget) -> dict[str, tuple[str, ...]]:
    """path → resolved colours, for every widget under ``window``."""
    out: dict[str, tuple[str, ...]] = {}

    def walk(w: QtWidgets.QWidget, path: str) -> None:
        pal = w.palette()
        out[path] = tuple(pal.color(r).name(QtGui.QColor.NameFormat.HexArgb) for r in ROLES)
        counts: dict[str, int] = {}
        for child in w.findChildren(QtWidgets.QWidget, options=QtCore_direct()):
            key = f"{type(child).__name__}#{child.objectName()}"
            idx = counts.get(key, 0)
            counts[key] = idx + 1
            walk(child, f"{path}/{key}[{idx}]")

    walk(window, type(window).__name__)
    return out


def QtCore_direct():  # noqa: N802 - tiny helper for readability above
    from PySide6 import QtCore
    return QtCore.Qt.FindChildOption.FindDirectChildrenOnly


@pytest.mark.parametrize("start,target", [("midnight", "pearl"), ("pearl", "midnight")])
def test_live_switch_matches_fresh_start(qtbot, start, target):
    import config
    from gui.main_window import AlphaDEXWindow
    from gui.themes.manager import get_manager

    mgr = get_manager()

    cfg = config.load_config()
    cfg["theme"] = start
    config.save_config(cfg)
    switched = AlphaDEXWindow()
    qtbot.addWidget(switched)
    switched.show()
    process_events(20)
    mgr.apply(target)
    process_events(20)
    a = _signature(switched)

    fresh = AlphaDEXWindow()   # load_persisted() now reads ``target``
    qtbot.addWidget(fresh)
    fresh.show()
    process_events(20)
    assert mgr.current.key == target
    b = _signature(fresh)

    common = sorted(set(a) & set(b))
    assert len(common) > 200, "widget trees should mostly line up"
    stale = [p for p in common if a[p] != b[p]]
    assert not stale, (
        f"{len(stale)} widget(s) kept colours from '{start}' after switching to "
        f"'{target}' (first 15):\n  " + "\n  ".join(stale[:15])
    )
