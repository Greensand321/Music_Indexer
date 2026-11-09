"""Page and section fades are snapshot fades (``gui.themes.motion.reveal``).

A QGraphicsOpacityEffect on a widget re-renders the widget's whole subtree on
every frame (Qt caches effect sources only for graphics items), so fading a
page of glass cards cost ~25 ms a frame offscreen (~40 fps). ``reveal`` now
renders the page once and fades that picture: ~7 ms a frame (R3 audit).
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from gui.themes import motion as mo
from ui_harness import FrameGapMeter, process_events

pytestmark = pytest.mark.motion(time_scale=1.0)


class _PaintCounter(QtCore.QObject):
    def __init__(self, widget: QtWidgets.QWidget) -> None:
        super().__init__(widget)
        self.count = 0
        widget.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QtCore.QEvent.Type.Paint:
            self.count += 1
        return False


@pytest.fixture(autouse=True)
def _real_motion(monkeypatch):
    monkeypatch.delenv(mo.ENV_VAR, raising=False)
    mo.set_reduced_motion(False)
    yield
    mo.set_reduced_motion(None)


def _overlays(window) -> list:
    return [w for w in window.findChildren(QtWidgets.QWidget) if isinstance(w, mo._RevealOverlay)]


def _card_of(page) -> QtWidgets.QWidget:
    from gui.widgets.glass import GlassCard
    cards = [c for c in page.findChildren(GlassCard) if c.isVisibleTo(page)]
    assert cards
    return cards[0]


def test_a_page_fade_renders_the_page_once_then_blits(main_window, motion):
    for key in ("help", "home", "help", "home"):
        main_window._on_nav_changed(key)
        motion.settle(2000)
    page = main_window._pages["help"]
    counter = _PaintCounter(_card_of(page))
    with FrameGapMeter() as meter:
        main_window._on_nav_changed("help")
        process_events(300)
    assert motion.settle(2000) == []
    frames = len(meter.gaps_ms)
    assert frames > 10, "the fade ran in real time"
    # The snapshot, the live widget taking over, and slack for a stray
    # update: nowhere near one paint per frame.
    assert counter.count <= 4, f"{counter.count} paints over {frames} frames"
    eff = page.graphicsEffect()
    assert eff is None or (not eff.isEnabled() and eff.opacity() == 1.0)
    assert not any(o.isVisible() for o in _overlays(main_window))


def test_a_section_fade_is_a_snapshot_too(main_window, motion):
    for key in ("duplicates/find", "duplicates/compare", "duplicates/find"):
        main_window._on_nav_changed(key)
        motion.settle(2000)
    panel = main_window._workspaces["similarity"]
    counter = _PaintCounter(_card_of(panel))
    main_window._on_nav_changed("duplicates/compare")
    process_events(250)
    assert motion.settle(2000) == []
    assert counter.count <= 4
    assert not any(o.isVisible() for o in _overlays(main_window))


def test_switching_away_mid_fade_leaves_no_picture_behind(main_window, motion):
    for key in ("help", "home", "indexer", "home"):
        main_window._on_nav_changed(key)
        motion.settle(2000)
    main_window._on_nav_changed("help")
    process_events(60)                       # mid-fade: the picture is up
    assert any(o.isVisible() for o in _overlays(main_window))
    main_window._on_nav_changed("indexer")   # the help page is hidden now
    visible = [o for o in _overlays(main_window) if o.isVisible()]
    assert all(o.target is main_window._pages["indexer"] for o in visible)
    assert motion.settle(2000) == []
    assert not any(o.isVisible() for o in _overlays(main_window))
    for key in ("help", "indexer"):
        eff = main_window._pages[key].graphicsEffect()
        assert eff is None or (not eff.isEnabled() and eff.opacity() == 1.0)


def test_the_picture_never_takes_input(main_window, motion):
    main_window._on_nav_changed("home")
    motion.settle(2000)
    main_window._on_nav_changed("help")
    process_events(60)
    for o in _overlays(main_window):
        assert o.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        assert o.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
    motion.settle(2000)
