"""Qt compatibility shim — PySide6 with PyQt6 fallback.

``QtWebEngineWidgets`` / ``QtWebChannel`` are imported lazily, on first
attribute access (``from gui.compat import QtWebEngineWidgets``). Nothing in
the app uses them today, and importing them eagerly made the optional
QtWebEngine add-on a hard startup dependency.
"""
from __future__ import annotations

from importlib import import_module, util

QtCore = None
QtGui = None
QtWidgets = None
Signal = None
Slot = None

_BINDING = ""
_LAZY_MODULES = ("QtWebEngineWidgets", "QtWebChannel")


def _load() -> None:
    global QtCore, QtGui, QtWidgets, Signal, Slot, _BINDING

    if util.find_spec("PySide6"):
        _BINDING = "PySide6"
        QtCore = import_module("PySide6.QtCore")
        QtGui = import_module("PySide6.QtGui")
        QtWidgets = import_module("PySide6.QtWidgets")
        Signal = QtCore.Signal
        Slot = QtCore.Slot
        return

    if util.find_spec("PyQt6"):
        _BINDING = "PyQt6"
        QtCore = import_module("PyQt6.QtCore")
        QtGui = import_module("PyQt6.QtGui")
        QtWidgets = import_module("PyQt6.QtWidgets")
        Signal = QtCore.pyqtSignal
        Slot = QtCore.pyqtSlot
        return

    raise SystemExit(
        "PySide6 or PyQt6 is required. Install with:\n"
        "  pip install PySide6"
    )


def __getattr__(name: str):
    """Import the optional web modules only when someone asks for them."""
    if name in _LAZY_MODULES:
        module = import_module(f"{_BINDING}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_load()

__all__ = ["QtCore", "QtGui", "QtWidgets", "Signal", "Slot"]
