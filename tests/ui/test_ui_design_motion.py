"""gui.themes.motion: tokens, reduced motion, and the safe animation helpers.

Also checks that the design-system widgets (hover mixin, sidebar pill, log
drawer) are built on the helpers: token durations, parented animations, no
growth on repeated triggers, correct end states (DS-05, DS-18).
"""
from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from gui.themes import motion as mo
from ui_harness import process_events, stuck_opacity


@pytest.fixture(autouse=True)
def _motion_state(monkeypatch):
    monkeypatch.delenv(mo.ENV_VAR, raising=False)
    mo.set_reduced_motion(None)
    mo.refresh_reduced_motion()
    yield
    mo.set_reduced_motion(None)
    mo.refresh_reduced_motion()


def _helper_children(obj):
    return [c for c in obj.children() if isinstance(c, QtCore.QAbstractAnimation)]


# ── Tokens ────────────────────────────────────────────────────────────────────

def test_tokens_and_easings():
    assert mo.DURATIONS == {
        "instant": 0, "micro": 120, "fast": 150, "base": 220, "slow": 320,
        "page": 420, "stagger": 38, "ambient": 3500, "flash": 1600,
    }
    assert set(mo.EASINGS) == {"standard", "exit", "emphasis", "ambient"}
    assert mo.easing("standard").type() == QtCore.QEasingCurve.Type.OutCubic
    assert mo.easing("exit").type() == QtCore.QEasingCurve.Type.InCubic
    emphasis = mo.easing("emphasis")
    assert emphasis.type() == QtCore.QEasingCurve.Type.OutBack
    assert emphasis.overshoot() == pytest.approx(mo.EMPHASIS_OVERSHOOT)
    assert mo.duration("base") == 220
    assert mo.duration(75) == 75
    # Stagger never exceeds the cap, however many items.
    assert mo.stagger_delay(31, 32) <= mo.STAGGER_MAX
    assert mo.stagger_delay(3, 4) == 3 * mo.STAGGER


def test_reduced_motion_sources(monkeypatch):
    assert mo.reduced_motion() is False
    monkeypatch.setenv(mo.ENV_VAR, "1")
    assert mo.reduced_motion() is True
    assert mo.duration("slow") == 0
    monkeypatch.setenv(mo.ENV_VAR, "0")
    assert mo.reduced_motion() is False

    from config import load_config, save_config
    cfg = load_config()
    cfg[mo.CONFIG_KEY] = True
    save_config(cfg)
    assert mo.reduced_motion() is False   # cached until refreshed
    mo.refresh_reduced_motion()
    assert mo.reduced_motion() is True

    mo.set_reduced_motion(False)           # explicit override wins
    assert mo.reduced_motion() is False


# ── animate_value ─────────────────────────────────────────────────────────────

@pytest.mark.motion(time_scale=1.0)
def test_animate_value_uses_token_parented_and_reaches_end(qtbot, motion):
    owner = QtWidgets.QWidget()
    qtbot.addWidget(owner)
    seen = []
    done = []
    anim = mo.animate_value(owner, "x", 0.0, 10.0, seen.append,
                                      duration="fast", on_finished=lambda: done.append(1))
    assert anim.parent() is owner
    assert anim.duration() == 150
    assert motion.records[-1].duration == 150
    assert motion.settle(2000) == []
    assert seen[-1] == pytest.approx(10.0)
    assert done == [1]
    assert motion.orphans() == []


def test_animate_value_is_reentrant(qtbot, motion):
    owner = QtWidgets.QWidget()
    qtbot.addWidget(owner)
    calls = []
    first = mo.animate_value(owner, "x", 0.0, 1.0, lambda v: None,
                                       on_finished=lambda: calls.append("first"))
    for _ in range(10):
        again = mo.animate_value(owner, "x", 0.5, 0.0, lambda v: calls.append(v) if v == 0.0 else None,
                                           on_finished=lambda: calls.append("last"))
    assert again is first, "the helper must reuse one animation per (owner, key)"
    assert len(_helper_children(owner)) == 1
    motion.settle(2000)
    assert "first" not in calls, "a superseded run must not fire its callback"
    assert calls.count("last") == 1


def test_animate_property_refuses_layout_owned_geometry(qtbot):
    host = QtWidgets.QWidget()
    qtbot.addWidget(host)
    lay = QtWidgets.QVBoxLayout(host)
    child = QtWidgets.QPushButton("x")
    lay.addWidget(child)
    with pytest.raises(ValueError):
        mo.animate_property(child, b"geometry", QtCore.QRect(0, 0, 10, 10))
    with pytest.raises(ValueError):
        mo.animate_property(child, b"maximumHeight", 0)
    # A child of a widget with no layout may be moved.
    free = QtWidgets.QWidget()
    qtbot.addWidget(free)
    loose = QtWidgets.QWidget(free)
    mo.animate_property(loose, b"pos", QtCore.QPoint(5, 7), duration="instant")
    assert loose.pos() == QtCore.QPoint(5, 7)


def test_reduced_motion_is_zero_duration_and_immediate(qtbot):
    mo.set_reduced_motion(True)
    owner = QtWidgets.QWidget()
    qtbot.addWidget(owner)
    seen, done = [], []
    anim = mo.animate_value(owner, "x", 0.0, 3.0, seen.append, duration="ambient",
                                on_finished=lambda: done.append(1))
    # No event processing needed: the end state and callback are applied now.
    assert anim.duration() == 0
    assert seen == [3.0] and done == [1]

    target = QtWidgets.QWidget()
    qtbot.addWidget(target)
    target.show()
    assert mo.fade(target, False) is not None
    assert not target.isVisible()
    assert target.graphicsEffect().opacity() == 1.0


# ── fade ─────────────────────────────────────────────────────────────────────

def test_fade_out_hides_and_resets_opacity(qtbot, motion):
    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    w.show()
    mo.fade(w, False)
    motion.settle(2000)
    assert not w.isVisible()
    eff = w.graphicsEffect()
    assert isinstance(eff, QtWidgets.QGraphicsOpacityEffect)
    assert eff.parent() is w, "the effect must be parented (PySide drops temporaries)"
    assert eff.opacity() == 1.0 and not eff.isEnabled()


def test_fade_in_ends_fully_opaque_even_when_interrupting_a_fade_out(qtbot, motion):
    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    w.show()
    mo.fade(w, False)
    mo.fade(w, True)          # re-entrant: newest request wins
    motion.settle(2000)
    assert w.isVisible()
    assert stuck_opacity(w) == []
    assert w.graphicsEffect().opacity() == 1.0
    assert not w.graphicsEffect().isEnabled(), "no off-screen rendering at rest"
    assert motion.orphans() == []


def test_fade_never_clobbers_another_effect(qtbot):
    w = QtWidgets.QWidget()
    qtbot.addWidget(w)
    blur = QtWidgets.QGraphicsBlurEffect(w)
    w.setGraphicsEffect(blur)
    w.show()
    assert mo.fade(w, False) is None
    assert w.graphicsEffect() is blur
    assert not w.isVisible()


# ── Consumers ────────────────────────────────────────────────────────────────

@pytest.mark.motion(time_scale=1.0)
def test_hover_mixin_uses_fast_token(qtbot, motion):
    from gui.themes.animations import AnimatedButton

    btn = AnimatedButton("Go")
    qtbot.addWidget(btn)
    btn._start_hover(True)
    assert btn._hover_anim.duration() == mo.FAST
    assert btn._hover_anim.parent() is btn
    motion.settle(2000)
    assert btn._hover_p == pytest.approx(1.0)
    for _ in range(5):
        btn._start_hover(False)
        btn._start_hover(True)
    assert len(_helper_children(btn)) == 1
    assert motion.orphans() == []


def test_programmatic_nav_leaves_no_hover_tint(qtbot, motion):
    """DS-05: activating items from code (Ctrl+1..9, navigate_requested)
    must not leave the previous item tinted."""
    from gui.widgets.sidebar import Sidebar

    sb = Sidebar()
    qtbot.addWidget(sb)
    sb.show()
    process_events(20)
    for key in ("duplicates", "player", "tools", "indexer"):
        sb.activate(key)
    motion.settle(2000)
    tinted = {k: b._hover_p for k, b in sb._buttons.items() if b._hover_p > 0 and not b.underMouse()}
    assert tinted == {}


@pytest.mark.motion(time_scale=1.0)
def test_sidebar_pill_uses_slow_emphasis_and_is_reused(qtbot, motion):
    from gui.widgets.sidebar import Sidebar

    sb = Sidebar()
    qtbot.addWidget(sb)
    sb.resize(360, 900)
    sb.show()
    process_events(30)
    for key in ("duplicates", "player", "exports"):
        sb.activate(key)
    anim = sb._nav_container._pill_anim
    assert anim.duration() == mo.SLOW
    assert anim.easingCurve().type() == QtCore.QEasingCurve.Type.OutBack
    assert len(_helper_children(sb._nav_container)) == 1
    motion.settle(3000)
    btn = sb._buttons["exports"]
    assert sb._nav_container._pill_y == pytest.approx(float(btn.y()))
    assert motion.orphans() == []


@pytest.mark.motion(time_scale=1.0)
def test_log_drawer_slides_full_distance_without_per_frame_relayout(qtbot, motion):
    """DS-18: the body takes its final height once, the sheet slides the full
    height, and collapsing gives the height back at the end."""
    from gui.widgets.log_drawer import LogDrawer, _DRAWER_EXPANDED

    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    lay.addWidget(QtWidgets.QWidget(), 1)
    drawer = LogDrawer()
    lay.addWidget(drawer)
    qtbot.addWidget(host)
    host.resize(800, 600)
    host.show()
    process_events(20)

    heights = []
    original = drawer._on_reveal

    def spy(v):
        heights.append(drawer._log_body.height())
        original(v)

    drawer._on_reveal = spy
    drawer.toggle()
    anim = mo.animation_for(drawer, "drawer")
    assert anim.duration() == mo.BASE
    motion.settle(2000)
    process_events(10)
    body = drawer._log_body
    assert body.height() == _DRAWER_EXPANDED
    assert set(heights) == {_DRAWER_EXPANDED}, "the layout height must not change per frame"
    assert body.sheet.y() == 0 and body.sheet.height() == _DRAWER_EXPANDED

    drawer.toggle()
    motion.settle(2000)
    process_events(10)
    assert drawer._log_body.height() == 0
    assert not drawer.is_expanded()
    assert motion.orphans() == []


def test_log_drawer_toggle_spam_ends_consistent(qtbot, motion):
    from gui.widgets.log_drawer import LogDrawer, _DRAWER_EXPANDED

    drawer = LogDrawer()
    qtbot.addWidget(drawer)
    drawer.show()
    for _ in range(7):           # odd count -> expanded
        drawer.toggle()
    motion.settle(2000)
    process_events(10)
    assert drawer.is_expanded()
    assert drawer._log_body.height() == _DRAWER_EXPANDED
    assert drawer._log_body.reveal == pytest.approx(1.0)
    assert len(_helper_children(drawer)) == 1

