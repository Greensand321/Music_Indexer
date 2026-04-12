"""Top bar widget — library selector, stats, and global actions."""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.animations import AnimatedButton
from gui.themes.manager import get_manager


def _abbreviate_path(path: str) -> str:
    """Return the last two path segments prefixed with '…  ' when longer."""
    if not path:
        return "No library selected"
    parts = Path(path).parts
    if len(parts) <= 2:
        return path
    tail = "  /  ".join(str(p) for p in parts[-2:])
    return f"\u2026  {tail}"


class _VSep(QtWidgets.QWidget):
    """1 px vertical separator that follows the theme's card_border colour."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(1, 28)
        self._color = QtGui.QColor("#30363d")

    def set_color(self, hex_color: str) -> None:
        self._color = QtGui.QColor(hex_color)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setPen(self._color)
        p.drawLine(0, 0, 0, self.height() - 1)
        p.end()


class TopBar(QtWidgets.QWidget):
    """Fixed-height bar at the top of the main window.

    Emits:
        library_changed(str)  — user selected a new library folder
        settings_requested()  — user clicked the Settings button
        theme_requested()     — user clicked the Theme button
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
        layout.setSpacing(0)

        # ── Brand ────────────────────────────────────────────────────────────
        brand = QtWidgets.QLabel("AlphaDEX")
        brand.setObjectName("appTitle")
        layout.addWidget(brand)

        layout.addSpacing(18)

        # ── Vertical separator ────────────────────────────────────────────────
        self._sep = _VSep()
        layout.addWidget(self._sep, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        layout.addSpacing(18)

        # ── Library info ──────────────────────────────────────────────────────
        lib_col = QtWidgets.QVBoxLayout()
        lib_col.setSpacing(3)
        lib_col.setContentsMargins(0, 0, 0, 0)

        # Path row — folder glyph + abbreviated path
        path_row = QtWidgets.QHBoxLayout()
        path_row.setSpacing(6)
        path_row.setContentsMargins(0, 0, 0, 0)

        self._folder_icon = QtWidgets.QLabel("\u2302")   # ⌂ house/folder glyph
        self._folder_icon.setObjectName("mutedLabel")
        path_row.addWidget(self._folder_icon)

        self._path_label = QtWidgets.QLabel("No library selected")
        self._path_label.setObjectName("libraryPath")
        path_row.addWidget(self._path_label)
        path_row.addStretch()
        lib_col.addLayout(path_row)

        # Stats row — three (number, unit) pairs
        stats_row = QtWidgets.QHBoxLayout()
        stats_row.setSpacing(0)
        stats_row.setContentsMargins(0, 0, 0, 0)

        self._stat_widgets: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = []
        _placeholders = [("–", "tracks"), ("–", "GB"), ("–", "artists")]
        for i, (num, unit) in enumerate(_placeholders):
            if i:
                dot = QtWidgets.QLabel("  ·  ")
                dot.setObjectName("mutedLabel")
                stats_row.addWidget(dot)
            num_lbl = QtWidgets.QLabel(num)
            num_lbl.setObjectName("statNumber")
            unit_lbl = QtWidgets.QLabel(f" {unit}")
            unit_lbl.setObjectName("statUnit")
            stats_row.addWidget(num_lbl)
            stats_row.addWidget(unit_lbl)
            self._stat_widgets.append((num_lbl, unit_lbl))

        stats_row.addStretch()
        lib_col.addLayout(stats_row)

        layout.addLayout(lib_col, 1)

        layout.addSpacing(12)

        # ── Action buttons ────────────────────────────────────────────────────
        self._change_btn = AnimatedButton("Change Library")
        self._change_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._change_btn.clicked.connect(self._on_change_library)
        layout.addWidget(self._change_btn)

        layout.addSpacing(8)

        self._theme_btn = AnimatedButton("Theme")
        self._theme_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._theme_btn.clicked.connect(self.theme_requested.emit)
        layout.addWidget(self._theme_btn)

        layout.addSpacing(8)

        self._settings_btn = AnimatedButton("Settings")
        self._settings_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._settings_btn.clicked.connect(self.settings_requested.emit)
        layout.addWidget(self._settings_btn)

        # ── Theme wiring ──────────────────────────────────────────────────────
        mgr = get_manager()
        mgr.theme_changed.connect(self._on_theme_changed)
        self._apply_theme(mgr.current)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_library(self, path: str) -> None:
        self._library_path = path
        if path:
            self._path_label.setText(_abbreviate_path(path))
            self._path_label.setToolTip(path)
            self._folder_icon.setVisible(True)
        else:
            self._path_label.setText("No library selected")
            self._path_label.setToolTip("")
            self._folder_icon.setVisible(False)
        self._clear_stats()

    def set_stats(self, tracks: int, size_gb: float, artists: int) -> None:
        if tracks == 0:
            self._clear_stats()
            return
        unit_label = "artist" if artists == 1 else "artists"
        values = [f"{tracks:,}", f"{size_gb:.1f}", f"{artists:,}"]
        units  = ["tracks", "GB", unit_label]
        for (num_lbl, unit_lbl), val, unit in zip(self._stat_widgets, values, units):
            num_lbl.setText(val)
            unit_lbl.setText(f" {unit}")

    @property
    def library_path(self) -> str:
        return self._library_path

    # ── Internal ──────────────────────────────────────────────────────────────

    def _clear_stats(self) -> None:
        for num_lbl, unit_lbl in self._stat_widgets:
            num_lbl.setText("–")
            unit_lbl.setText(" tracks" if self._stat_widgets.index((num_lbl, unit_lbl)) == 0
                             else " GB" if self._stat_widgets.index((num_lbl, unit_lbl)) == 1
                             else " artists")

    def _apply_theme(self, tokens) -> None:
        t = tokens
        self._sep.set_color(t.card_border)

        path_color = t.text_primary if self._library_path else t.text_secondary
        self._path_label.setStyleSheet(f"color: {path_color};")
        self._folder_icon.setStyleSheet(f"color: {t.text_secondary};")

        for num_lbl, unit_lbl in self._stat_widgets:
            num_lbl.setStyleSheet(
                f"color: {t.text_primary}; font-weight: 600;"
            )
            unit_lbl.setStyleSheet(f"color: {t.text_secondary};")

    def _on_theme_changed(self, tokens) -> None:
        self._apply_theme(tokens)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_change_library(self) -> None:
        start = self._library_path or str(Path.home())

        dlg = QtWidgets.QFileDialog()
        dlg.setWindowTitle("Select Music Library Folder")
        dlg.setDirectory(start)
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        win = self.window()
        if win and win.isVisible():
            dlg.resize(820, 560)
            fg = win.frameGeometry()
            dlg.move(
                fg.left() + (fg.width()  - dlg.width())  // 2,
                fg.top()  + (fg.height() - dlg.height()) // 2,
            )

        if dlg.exec():
            selected = dlg.selectedFiles()
            if selected:
                self.library_changed.emit(selected[0])
