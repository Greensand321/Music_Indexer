"""Top bar widget — library selector, stats, and global actions."""
from __future__ import annotations

import os
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal


class TopBar(QtWidgets.QWidget):
    """Fixed-height bar at the top of the main window.

    Emits:
        library_changed(str)  — user selected a new library folder
        settings_requested()  — user clicked the ⚙ Settings button
    """

    library_changed = Signal(str)
    settings_requested = Signal()
    theme_requested = Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("topBar")
        self.setFixedHeight(64)
        self._library_path: str = ""

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 16, 0)
        layout.setSpacing(16)

        # ── Brand ────────────────────────────────────────────────────────
        brand = QtWidgets.QVBoxLayout()
        brand.setSpacing(1)
        title = QtWidgets.QLabel("AlphaDEX")
        title.setObjectName("appTitle")
        brand.addWidget(title)
        layout.addLayout(brand)

        # ── Library info ─────────────────────────────────────────────────
        lib_block = QtWidgets.QVBoxLayout()
        lib_block.setSpacing(1)

        self._path_label = QtWidgets.QLabel("No library selected")
        self._path_label.setObjectName("libraryPath")
        self._path_label.setMaximumWidth(520)

        self._stats_label = QtWidgets.QLabel("")
        self._stats_label.setObjectName("libStats")

        lib_block.addWidget(self._path_label)
        lib_block.addWidget(self._stats_label)
        layout.addLayout(lib_block, stretch=1)

        # ── Actions ──────────────────────────────────────────────────────
        change_btn = QtWidgets.QPushButton("📁  Change Library")
        change_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        change_btn.clicked.connect(self._on_change_library)
        layout.addWidget(change_btn)

        theme_btn = QtWidgets.QPushButton("🎨  Theme")
        theme_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        theme_btn.clicked.connect(self.theme_requested.emit)
        layout.addWidget(theme_btn)

        settings_btn = QtWidgets.QPushButton("⚙  Settings")
        settings_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        settings_btn.clicked.connect(self.settings_requested.emit)
        layout.addWidget(settings_btn)

    # ── Public API ────────────────────────────────────────────────────────

    def set_library(self, path: str) -> None:
        self._library_path = path
        if path:
            self._path_label.setText(path)
            self._path_label.setToolTip(path)
        else:
            self._path_label.setText("No library selected")
        self._stats_label.setText("")

    def set_stats(self, tracks: int, size_gb: float, artists: int) -> None:
        if tracks == 0:
            self._stats_label.setText("")
            return
        self._stats_label.setText(
            f"{tracks:,} tracks · {size_gb:.1f} GB · {artists:,} artists"
        )

    @property
    def library_path(self) -> str:
        return self._library_path

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_change_library(self) -> None:
        start = self._library_path or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self.window(),
            "Select Music Library Folder",
            start,
            QtWidgets.QFileDialog.Option.ShowDirsOnly
            | QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )
        if folder:
            self.library_changed.emit(folder)
