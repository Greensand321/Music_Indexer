"""Every Qt name the GUI references must exist in the installed PySide6.

Two shipped bugs were simply misspelled / Qt5-era names that only blow up
when a particular code path runs: ``QFont.Weight.SemiBold`` (should be
``DemiBold``; every selected tab label failed to paint) and
``QtWidgets.QShortcut`` (moved to QtGui in Qt 6; the Pair Review dialog
could never open). This test resolves *every* ``QtCore.*`` / ``QtGui.*`` /
``QtWidgets.*`` attribute chain in the GUI source against the real modules,
so this whole class of bug is caught without having to reach the code path.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

REPO = Path(__file__).resolve().parents[2]
GUI_FILES = sorted((REPO / "gui").rglob("*.py")) + [REPO / "alpha_dex_gui.py"]
ROOTS = {"QtCore": QtCore, "QtGui": QtGui, "QtWidgets": QtWidgets}

# Optional modules imported behind try/except in gui/compat.py.
try:  # pragma: no cover - environment dependent
    from PySide6 import QtWebChannel, QtWebEngineWidgets  # noqa: F401
    ROOTS["QtWebEngineWidgets"] = QtWebEngineWidgets
    ROOTS["QtWebChannel"] = QtWebChannel
except ImportError:
    pass


def _chains(tree: ast.AST):
    """Yield (dotted_chain, lineno) for maximal attribute chains on a Qt root."""
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        parent = parents.get(id(node))
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue  # not maximal
        parts = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id in ROOTS:
            parts.append(cur.id)
            yield ".".join(reversed(parts)), node.lineno


def _resolve(chain: str) -> str | None:
    """None if the chain resolves; else the first missing component."""
    root, *rest = chain.split(".")
    obj = ROOTS[root]
    for i, name in enumerate(rest):
        if not hasattr(obj, name):
            # Stop at instance-level attributes we cannot see on the class
            # (e.g. a method's return value attribute) — only Qt classes,
            # enums and enum members are checked.
            if i > 0 and not isinstance(obj, type):
                return None
            return ".".join([root, *rest[: i + 1]])
        obj = getattr(obj, name)
    return None


# PyQt6 fallback names used behind ``hasattr`` guards (gui/compat.py, qt_launcher.py).
ALLOWED_CHAINS = {"QtCore.pyqtSignal", "QtCore.pyqtSlot"}

# Known-broken references, keyed "file:chain" → bug ID in docs/redesign/.
# Fixing the bug means deleting its entry here; the test then guards it.
KNOWN_BROKEN = {
    # Fixing the import alone would expose DUP-08 (one-key permanent delete),
    # so it must be fixed together with that.
    "gui/dialogs/pair_review_dialog.py:QtWidgets.QShortcut": "DUP-06",
}


def _collect():
    problems, known_seen = [], set()
    for path in GUI_FILES:
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for chain, line in _chains(tree):
            if chain in ALLOWED_CHAINS:
                continue
            missing = _resolve(chain)
            if not missing:
                continue
            key = f"{rel}:{chain}"
            if key in KNOWN_BROKEN:
                known_seen.add(key)
                continue
            problems.append(f"{rel}:{line}: {chain} (missing {missing})")
    return problems, known_seen


def test_all_qt_attribute_references_resolve():
    problems, _ = _collect()
    assert not problems, "Qt names that do not exist in PySide6:\n" + "\n".join(problems)


def test_known_broken_list_is_current():
    """Once a known-broken reference is fixed, its entry must be removed."""
    _, seen = _collect()
    stale = sorted(set(KNOWN_BROKEN) - seen)
    assert not stale, f"Fixed — remove from KNOWN_BROKEN: {stale}"


@pytest.mark.parametrize("chain,ok", [
    ("QtGui.QFont.Weight.DemiBold", True),
    ("QtGui.QFont.Weight.SemiBold", False),
    ("QtWidgets.QShortcut", False),
    ("QtGui.QShortcut", True),
])
def test_resolver_self_check(chain, ok):
    assert (_resolve(chain) is None) is ok
