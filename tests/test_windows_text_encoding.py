"""Our own text files are read and written as UTF-8, on Windows too.

Without ``encoding=`` Python uses the locale codec, which on Windows is
cp1252: a report or log holding "✓", "→" or a Japanese file name can't be
written (UnicodeEncodeError), and a UTF-8 file we wrote can't be read back
(Windows CI: ``UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d``).

This scans the app's modules for text-mode ``open()`` / ``Path.open()`` /
``read_text()`` / ``write_text()`` calls without an explicit encoding.
``main_gui.py`` (the legacy Tkinter app) and tests are not scanned.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SKIP_DIRS = {"tests", "docs", ".git", ".venv", "venv", ".claude", "build", "dist", "mutagen_stub"}
_SKIP_FILES = {"main_gui.py"}
# Modules whose ``open`` is not the builtin text-file open.
_OTHER_OPENS = {"os", "Image", "wave", "webbrowser", "sqlite3", "tarfile", "zipfile", "gzip",
                "codecs", "aifc", "sunau", "subprocess", "QtCore", "QtGui", "QtWidgets", "QDesktopServices"}


def _app_modules():
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in _SKIP_DIRS or rel.name in _SKIP_FILES or rel.name.startswith("test_"):
            continue
        yield rel, path


def _mode(node: ast.Call, position: int):
    if len(node.args) > position and isinstance(node.args[position], ast.Constant):
        return node.args[position].value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return "r"


def _unencoded_text_io(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if any(kw.arg == "encoding" or kw.arg is None for kw in node.keywords):
            continue  # explicit encoding (or **kwargs we can't see into)
        if isinstance(func, ast.Name) and func.id == "open":
            mode = _mode(node, 1)
        elif isinstance(func, ast.Attribute) and func.attr == "open":
            if isinstance(func.value, ast.Name) and func.value.id in _OTHER_OPENS:
                continue
            if isinstance(func.value, ast.Attribute) and func.value.attr in _OTHER_OPENS:
                continue
            mode = _mode(node, 0)
        elif isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            mode = "r"
        else:
            continue
        if not isinstance(mode, str) or "b" in mode:
            continue
        yield node.lineno, ast.unparse(node)


def test_app_text_file_io_names_its_encoding():
    offenders = []
    for rel, path in _app_modules():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # e.g. the stray UTF-16 player_106bd0e.py
            continue
        offenders += [f"{rel.as_posix()}:{line}: {code[:100]}" for line, code in _unencoded_text_io(tree)]
    assert offenders == [], "pass encoding='utf-8':\n" + "\n".join(offenders)
