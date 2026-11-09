"""Reduced motion: nothing in the app animates.

Every animation must honour Settings -> General -> Reduce motion: page and
section cross-fades, the R2 content reveals, the sidebar pill and section
thumb, the capsule's fade and keyboard pulse, the Player card's cover
cross-fade, status-label flashes and the landing. These tests run with
real-time motion (the harness does not compress durations) and fail when
any animation with a duration is started.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtGui, QtWidgets

from gui.navigation import PAGES
from gui.themes import motion as mo
from ui_harness import process_events

pytestmark = pytest.mark.motion(time_scale=1.0)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reduced(monkeypatch):
    monkeypatch.delenv(mo.ENV_VAR, raising=False)
    mo.set_reduced_motion(True)
    yield
    mo.set_reduced_motion(None)
    mo.refresh_reduced_motion()


def _moving(motion) -> list[str]:
    """Animations started with a duration (0 ms ones end inside start())."""
    return [f"{r.kind} {r.target}.{r.prop} {r.duration} ms" for r in motion.records if r.duration > 0]


def _nav_keys() -> list[str]:
    keys = []
    for page in PAGES:
        keys.append(page.key)
        if page.has_sections:
            keys += [f"{page.key}/{s.key}" for s in page.sections]
    return keys


def test_page_and_section_switches_start_no_animation(main_window, motion, library, threads):
    main_window.set_library(str(library.root))
    assert threads.wait_all(20_000) == []
    keys = _nav_keys()
    for key in keys + list(reversed(keys)):
        main_window._on_nav_changed(key)
        process_events(5)
    assert threads.wait_all(20_000) == []
    assert _moving(motion) == []
    assert motion.running() == []
    page = main_window.current_page()
    eff = page.graphicsEffect()
    assert eff is None or not eff.isEnabled()


def test_r2_reveals_start_no_animation(main_window, motion, library, threads):
    """The content reveals pages run when results appear (Sound Map's
    results, the Playlists preview), and ``motion.reveal`` itself."""
    main_window.set_library(str(library.root))
    assert threads.wait_all(20_000) == []
    analysis = main_window._workspaces["clustered"]
    main_window._on_nav_changed("sound_map/analyse")
    labels = np.array([0, 1, 0, 1, 2, 2, -1])
    result = {
        "labels": labels,
        "metrics": {"n_clusters": 3, "noise_count": 1, "noise_fraction": 1 / 7,
                    "silhouette_score": 0.41},
        "cluster_info": {0: {"size": 2}, 1: {"size": 2}, 2: {"size": 2}},
        "config": {"algorithm": "kmeans", "features": {"mfcc": True}, "k": 3},
        "failed": {},
    }
    analysis._set_result(result)
    analysis._show_results_panel(result)
    playlists = main_window._workspaces["playlists"]
    main_window._on_nav_changed("playlists")
    from types import SimpleNamespace
    plan = [SimpleNamespace(path=str(library.root / "Playlists" / "A.m3u"), entries=3, exists=False)]
    playlists._show_plan("folder", plan)            # reveal: the plan rows
    playlists._show_empty("Nothing to write", "")   # reveal: the empty state
    widget = QtWidgets.QLabel("x", main_window)
    widget.show()
    assert mo.reveal(widget) is None
    process_events(20)
    assert _moving(motion) == []
    assert motion.running() == []


def test_capsule_fade_and_keyboard_pulse_do_not_animate(main_window, motion):
    bar = main_window._now_playing_bar
    pm = QtGui.QPixmap(8, 8)
    pm.fill(QtGui.QColor("#335577"))
    bar.update_now_playing("Song", "Artist", "Album", pm)
    process_events(20)
    assert bar.isVisible()
    bar._kb_btn.toggle()
    bar._kb_btn.toggle()
    process_events(20)
    assert _moving(motion) == []
    assert not bar._kb_fx.isEnabled()


def test_player_cover_cross_fade_swaps_at_once(main_window, motion):
    from gui.workspaces.player import _LibraryCard

    card = _LibraryCard(main_window)
    a, b = QtGui.QPixmap(40, 40), QtGui.QPixmap(40, 40)
    a.fill(QtGui.QColor("#aa3355"))
    b.fill(QtGui.QColor("#3355aa"))
    card.set_art(a)
    card.set_art(b)
    assert not card._fading and card.has_art
    assert _moving(motion) == []


def test_status_flash_steps_instead_of_fading(qtbot, motion):
    from gui.workspaces.label_tone import flash_tone, tone_color, tone_of

    label = QtWidgets.QLabel("Done")
    qtbot.addWidget(label)
    flash_tone(label, "success", "secondary", duration=60)
    assert _moving(motion) == []
    pal = QtGui.QPalette.ColorRole.WindowText
    assert label.palette().color(pal) == tone_color("success")    # the status shows
    assert tone_of(label) == "secondary"
    qtbot.waitUntil(lambda: label.palette().color(pal) == tone_color("secondary"), timeout=2000)


def test_every_hand_built_animation_checks_reduced_motion():
    """Animations built outside the motion helpers must ask for their
    duration through ``motion.duration`` / ``reduced_motion`` (the helpers
    already do). ``gui/qt_launcher.py`` belongs to the legacy Tk app."""
    builders = {"QPropertyAnimation", "QVariantAnimation", "QSequentialAnimationGroup",
                "QParallelAnimationGroup", "QTimeLine"}
    skip = {REPO / "gui" / "themes" / "motion.py", REPO / "gui" / "qt_launcher.py"}
    offenders = []
    for path in sorted((REPO / "gui").rglob("*.py")):
        if path in skip:
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        builds = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr in builders]
        if builds and not any(k in src for k in ("motion.duration(", "reduced_motion(")):
            offenders.append(f"{path.relative_to(REPO)}:{builds[0].lineno}")
    assert offenders == []
