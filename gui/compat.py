"""Qt compatibility shim — PySide6 with PyQt6 fallback."""
from __future__ import annotations

from importlib import import_module, util

QtCore = None
QtGui = None
QtWidgets = None
QtWebEngineWidgets = None
QtWebChannel = None
Signal = None
Slot = None


def _load() -> None:
    global QtCore, QtGui, QtWidgets, QtWebEngineWidgets, QtWebChannel, Signal, Slot

    if util.find_spec("PySide6"):
        QtCore = import_module("PySide6.QtCore")
        QtGui = import_module("PySide6.QtGui")
        QtWidgets = import_module("PySide6.QtWidgets")
        QtWebEngineWidgets = import_module("PySide6.QtWebEngineWidgets")
        QtWebChannel = import_module("PySide6.QtWebChannel")
        Signal = QtCore.Signal
        Slot = QtCore.Slot
        return

    if util.find_spec("PyQt6"):
        QtCore = import_module("PyQt6.QtCore")
        QtGui = import_module("PyQt6.QtGui")
        QtWidgets = import_module("PyQt6.QtWidgets")
        QtWebEngineWidgets = import_module("PyQt6.QtWebEngineWidgets")
        QtWebChannel = import_module("PyQt6.QtWebChannel")
        Signal = QtCore.pyqtSignal
        Slot = QtCore.pyqtSlot
        return

    raise SystemExit(
        "PySide6 or PyQt6 is required. Install with:\n"
        "  pip install PySide6"
    )


_load()

__all__ = ["QtCore", "QtGui", "QtWidgets", "QtWebEngineWidgets", "QtWebChannel", "Signal", "Slot"]
