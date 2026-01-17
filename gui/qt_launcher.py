"""Standalone Qt preview window launcher."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, util
import sys


QtWidgets = None
QtCore = None
QtGui = None


def _load_qt_modules():
    """Load Qt modules preferring PySide6 and falling back to PyQt6."""
    if util.find_spec("PySide6"):
        return (
            import_module("PySide6.QtCore"),
            import_module("PySide6.QtGui"),
            import_module("PySide6.QtWidgets"),
        )
    if util.find_spec("PyQt6"):
        return (
            import_module("PyQt6.QtCore"),
            import_module("PyQt6.QtGui"),
            import_module("PyQt6.QtWidgets"),
        )
    raise SystemExit(
        "PySide6 or PyQt6 is required to launch the Qt preview window. "
        "Install one of them and try again."
    )


QtCore, QtGui, QtWidgets = _load_qt_modules()


@dataclass
class DemoRow:
    name: str
    role: str
    status: str


class QtPreviewWindow(QtWidgets.QMainWindow):
    """Simple placeholder Qt window for incremental migration."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AlphaDEX Qt Preview")
        self.resize(1100, 720)
        self.setWindowOpacity(0.97)
        self._overlay_effect = None

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        heading = QtWidgets.QLabel("Qt Preview Window")
        heading.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("font-size: 22px; font-weight: 600;")

        subheading = QtWidgets.QLabel(
            "Interact with the widgets below to get a feel for the Qt layout system."
        )
        subheading.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        subheading.setStyleSheet("color: #666;")

        layout.addWidget(heading)
        layout.addWidget(subheading)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_dashboard_tab(), "Dashboard")
        tabs.addTab(self._build_controls_tab(), "Controls")
        layout.addWidget(tabs)

        footer = QtWidgets.QLabel("Status: Ready")
        footer.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        footer.setStyleSheet("color: #666;")
        layout.addWidget(footer)
        self._status_label = footer

        self.setCentralWidget(container)
        self.statusBar().showMessage("Qt preview loaded.")

    def _build_dashboard_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        card = QtWidgets.QFrame()
        card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        card.setStyleSheet("background: #f5f6f8; border-radius: 10px;")
        card_layout = QtWidgets.QHBoxLayout(card)

        stats = QtWidgets.QVBoxLayout()
        stat_title = QtWidgets.QLabel("Library Snapshot")
        stat_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        stats.addWidget(stat_title)

        stats.addWidget(QtWidgets.QLabel("Tracks indexed: 12,430"))
        stats.addWidget(QtWidgets.QLabel("Duplicates flagged: 84"))
        stats.addWidget(QtWidgets.QLabel("Playlists generated: 18"))
        stats.addStretch(1)

        preview = QtWidgets.QLabel("Artwork Preview")
        preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(180, 180)
        preview.setStyleSheet("background: #dfe3ea; border-radius: 12px;")

        card_layout.addLayout(stats)
        card_layout.addStretch(1)
        card_layout.addWidget(preview)

        opacity_card = QtWidgets.QFrame()
        opacity_card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        opacity_card.setStyleSheet("background: #1e1f26; color: #f5f5f5; border-radius: 10px;")
        opacity_layout = QtWidgets.QVBoxLayout(opacity_card)
        opacity_layout.addWidget(QtWidgets.QLabel("Semi-transparent overlay preview"))
        opacity_layout.addWidget(QtWidgets.QLabel("Use the slider in Controls to adjust."))

        self._overlay_effect = QtWidgets.QGraphicsOpacityEffect(opacity_card)
        self._overlay_effect.setOpacity(0.85)
        opacity_card.setGraphicsEffect(self._overlay_effect)

        table = QtWidgets.QTableWidget(4, 3)
        table.setHorizontalHeaderLabels(["Track", "Artist", "Status"])
        rows = [
            DemoRow("Night Drive", "Apex City", "Ready"),
            DemoRow("Luminous", "Afterglow", "Queued"),
            DemoRow("Signal Boost", "Metroline", "Scanning"),
            DemoRow("Static Bloom", "Synth Atelier", "Ready"),
        ]
        for row_index, row in enumerate(rows):
            table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row.name))
            table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row.role))
            table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(row.status))
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)

        layout.addWidget(card)
        layout.addWidget(opacity_card)
        layout.addWidget(table)
        return panel

    def _build_controls_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(panel)
        layout.setSpacing(18)

        form_group = QtWidgets.QGroupBox("Controls")
        form_layout = QtWidgets.QFormLayout(form_group)

        self._name_input = QtWidgets.QLineEdit()
        self._name_input.setPlaceholderText("Type a track title…")

        self._genre_combo = QtWidgets.QComboBox()
        self._genre_combo.addItems(["Ambient", "Synthwave", "Lo-fi", "House"])

        self._favorite_check = QtWidgets.QCheckBox("Mark as favorite")
        self._sync_radio = QtWidgets.QRadioButton("Sync metadata")
        self._skip_radio = QtWidgets.QRadioButton("Skip metadata")
        self._sync_radio.setChecked(True)

        self._volume_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(65)

        self._opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(70, 100)
        self._opacity_slider.setValue(97)
        self._opacity_slider.valueChanged.connect(self._update_opacity)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setValue(45)

        form_layout.addRow("Track Title", self._name_input)
        form_layout.addRow("Genre", self._genre_combo)
        form_layout.addRow("", self._favorite_check)
        form_layout.addRow("Metadata", self._sync_radio)
        form_layout.addRow("", self._skip_radio)
        form_layout.addRow("Volume", self._volume_slider)
        form_layout.addRow("Window Opacity", self._opacity_slider)
        form_layout.addRow("Sync Progress", self._progress)

        actions_group = QtWidgets.QGroupBox("Actions")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)

        self._action_label = QtWidgets.QLabel("No action yet.")
        self._action_label.setStyleSheet("font-weight: 600;")

        sync_button = QtWidgets.QPushButton("Run Dummy Sync")
        sync_button.clicked.connect(self._run_dummy_sync)

        clear_button = QtWidgets.QPushButton("Reset Form")
        clear_button.clicked.connect(self._reset_form)

        actions_layout.addWidget(self._action_label)
        actions_layout.addSpacing(8)
        actions_layout.addWidget(sync_button)
        actions_layout.addWidget(clear_button)
        actions_layout.addStretch(1)

        list_group = QtWidgets.QGroupBox("Queue")
        list_layout = QtWidgets.QVBoxLayout(list_group)
        self._queue_list = QtWidgets.QListWidget()
        self._queue_list.addItems(
            ["Analyzing: Night Drive", "Queued: Luminous", "Queued: Static Bloom"]
        )
        list_layout.addWidget(self._queue_list)

        layout.addWidget(form_group, 2)
        layout.addWidget(actions_group, 1)
        layout.addWidget(list_group, 1)
        return panel

    def _run_dummy_sync(self) -> None:
        name = self._name_input.text().strip() or "Untitled Track"
        genre = self._genre_combo.currentText()
        favorite = "⭐" if self._favorite_check.isChecked() else "—"
        progress = min(100, self._progress.value() + 15)
        self._progress.setValue(progress)
        message = f"Synced {name} ({genre}) {favorite}"
        self._action_label.setText(message)
        self._status_label.setText(f"Status: {message}")
        self.statusBar().showMessage("Dummy sync completed.", 3000)

    def _reset_form(self) -> None:
        self._name_input.clear()
        self._genre_combo.setCurrentIndex(0)
        self._favorite_check.setChecked(False)
        self._sync_radio.setChecked(True)
        self._volume_slider.setValue(65)
        self._progress.setValue(0)
        self._action_label.setText("No action yet.")
        self._status_label.setText("Status: Ready")

    def _update_opacity(self, value: int) -> None:
        self.setWindowOpacity(value / 100)
        if self._overlay_effect:
            self._overlay_effect.setOpacity(max(0.4, value / 100))


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = QtPreviewWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
