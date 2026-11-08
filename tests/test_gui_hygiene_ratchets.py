"""Static hygiene ratchets for the Qt GUI (no Qt needed to run).

The GUI redesign needs the codebase to converge on a few rules. Rather than
demanding everything is fixed at once, each rule has a *ceiling*: the count
measured when the ratchet was introduced. A change may lower a count, never
raise it. When you lower one, lower its ceiling here too so it stays locked.

Zero-tolerance rules (ceiling 0) guard bugs that have already been fixed.

Introduced 2026-11-07 at hex 237 / inline sheets 168 / font-size 81 /
direct QThread starts 28 / unparented animations 7; lowered the same day
after the phase-2 bug fixes (see docs/redesign/README.md "Harness"), and again
to the measured counts when the information architecture replaced the
Utilities page (hex 103, inline sheets 88, font-size 34), and again when the
Liquid Glass shell (redesign R1) moved the landing, sidebar, header, Now
Playing bar and log drawer onto painted tokens (hex 60, inline sheets 74,
font-size 28).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUI = REPO / "gui"

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
_THREADISH = re.compile(r"worker|thread|scanner|loader", re.I)
_ANIMATIONS = {
    "QPropertyAnimation": 3,         # (target, property, parent)
    "QVariantAnimation": 1,          # (parent)
    "QParallelAnimationGroup": 1,
    "QSequentialAnimationGroup": 1,
}

CEILINGS = {
    # Colours belong in gui/themes/tokens.py; pages should use tokens/roles.
    "hex_colours_outside_tokens": 60,
    # Inline stylesheets outside gui/themes bypass the design system and are
    # re-parsed on every theme switch.
    "inline_setStyleSheet_calls": 47,
    # Type sizes belong in the type ramp (gui/fonts/loader.py TypeScale).
    "font_size_literals_outside_themes": 12,
    # Zero tolerance since every page moved onto the JobRegistry: start QThreads
    # via WorkspaceBase._start_worker / gui.workers.start_worker so they are
    # kept alive until finished and stopped on quit.
    "direct_qthread_starts": 0,
    # Animations without a QObject parent can be garbage-collected mid-flight.
    # (The remaining 2 are in gui/qt_launcher.py, used only by the legacy Tk app.)
    "unparented_animations": 2,
    # Zero tolerance: lambdas connected to the ThemeManager singleton outlive
    # their widget and raise "Internal C++ object already deleted".
    "theme_changed_lambda_connections": 0,
}


def _gui_files():
    return sorted(p for p in GUI.rglob("*.py") if "__pycache__" not in p.parts)


def measure() -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {k: [] for k in CEILINGS}
    for path in _gui_files():
        rel = path.relative_to(REPO).as_posix()
        in_themes = rel.startswith("gui/themes/")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and not in_themes:
                for _ in _HEX.findall(node.value):
                    hits["hex_colours_outside_tokens"].append(f"{rel}:{node.lineno}")
                for _ in re.findall(r"font-size\s*:", node.value):
                    hits["font_size_literals_outside_themes"].append(f"{rel}:{node.lineno}")
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "setStyleSheet" and not in_themes:
                hits["inline_setStyleSheet_calls"].append(f"{rel}:{node.lineno}")
            if (
                name == "start" and isinstance(func, ast.Attribute) and not node.args
                and rel != "gui/workers.py"
            ):
                receiver = ast.unparse(func.value)
                if _THREADISH.search(receiver) and not receiver.startswith("threading."):
                    hits["direct_qthread_starts"].append(f"{rel}:{node.lineno} {receiver}.start()")
            if name in _ANIMATIONS and len(node.args) + len(node.keywords) < _ANIMATIONS[name]:
                hits["unparented_animations"].append(f"{rel}:{node.lineno} {name}")
            if (
                name == "connect" and isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Attribute) and func.value.attr == "theme_changed"
                and node.args and isinstance(node.args[0], ast.Lambda)
            ):
                hits["theme_changed_lambda_connections"].append(f"{rel}:{node.lineno}")
    return hits


def test_hygiene_ratchets():
    hits = measure()
    over = []
    for rule, ceiling in CEILINGS.items():
        count = len(hits[rule])
        if count > ceiling:
            where = "\n      ".join(hits[rule][:40])
            over.append(f"{rule}: {count} > ceiling {ceiling}\n      {where}")
    assert not over, "GUI hygiene ratchet exceeded (fix the new occurrences):\n  " + "\n  ".join(over)


def test_ratchet_detectors_work(tmp_path, monkeypatch):
    """The detectors must actually detect (guards against a silently-zero rule)."""
    fake_gui = tmp_path / "gui"
    fake_gui.mkdir()
    (fake_gui / "page.py").write_text(
        "from gui.compat import QtCore\n"
        "def build(w, mgr):\n"
        "    w.setStyleSheet('color: #ff00aa; font-size: 12px;')\n"
        "    mgr.theme_changed.connect(lambda t: w.update())\n"
        "    w._worker.start()\n"
        "    a = QtCore.QPropertyAnimation(w, b'pos')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(__import__(__name__), "REPO", tmp_path)
    monkeypatch.setattr(__import__(__name__), "GUI", fake_gui)
    hits = measure()
    assert {k: len(v) for k, v in hits.items()} == {
        "hex_colours_outside_tokens": 1,
        "inline_setStyleSheet_calls": 1,
        "font_size_literals_outside_themes": 1,
        "direct_qthread_starts": 1,
        "unparented_animations": 1,
        "theme_changed_lambda_connections": 1,
    }
