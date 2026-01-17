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
    """Qt proof-of-concept window that sketches the planned workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AlphaDEX Qt Preview")
        self.resize(1280, 820)
        self.setWindowOpacity(0.97)
        self._overlay_effect = None

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        layout.addWidget(self._build_top_bar())

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_main_content(), stretch=1)
        layout.addLayout(body)

        footer = QtWidgets.QLabel("Status: Ready • Last refresh 2 min ago")
        footer.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        footer.setStyleSheet("color: #6b7280; font-size: 12px;")
        layout.addWidget(footer)
        self._status_label = footer

        self.setCentralWidget(container)
        self.statusBar().showMessage("Qt preview loaded.")

    def _build_top_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        bar.setStyleSheet("background: #101827; color: #f8fafc; border-radius: 12px;")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(16, 12, 16, 12)

        title_block = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("AlphaDEX — Proof of Concept")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        subtitle = QtWidgets.QLabel("Reimagined Qt workflow shell (no wiring yet)")
        subtitle.setStyleSheet("color: #94a3b8;")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        layout.addLayout(title_block)

        layout.addStretch(1)

        library_chip = QtWidgets.QLabel("Library: /Music/AlphaDEX")
        library_chip.setStyleSheet(
            "background: rgba(148, 163, 184, 0.2); padding: 6px 10px; border-radius: 10px;"
        )
        layout.addWidget(library_chip)

        sync_button = QtWidgets.QPushButton("Sync Now")
        sync_button.setStyleSheet(
            "background: #3b82f6; color: white; padding: 6px 14px; border-radius: 8px;"
        )
        sync_button.clicked.connect(self._run_dummy_sync)
        layout.addWidget(sync_button)
        return bar

    def _build_sidebar(self) -> QtWidgets.QWidget:
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("background: #f8fafc; border-radius: 12px;")
        layout = QtWidgets.QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        label = QtWidgets.QLabel("Workflows")
        label.setStyleSheet("font-size: 13px; font-weight: 600; color: #111827;")
        layout.addWidget(label)

        for item in (
            "Library Health",
            "Index & Organize",
            "Duplicate Review",
            "Tag Cleanup",
            "Playlist Builder",
            "Insights & Reports",
        ):
            button = QtWidgets.QPushButton(item)
            button.setStyleSheet(
                "text-align: left; padding: 8px 10px; border-radius: 8px; background: #ffffff;"
            )
            layout.addWidget(button)

        layout.addStretch(1)

        quick_actions = QtWidgets.QLabel("Quick Actions")
        quick_actions.setStyleSheet("font-size: 12px; font-weight: 600; color: #6b7280;")
        layout.addWidget(quick_actions)

        for item in ("Run Preview", "Open Reports", "Export Plan"):
            action = QtWidgets.QPushButton(item)
            action.setStyleSheet(
                "text-align: left; padding: 6px 10px; border-radius: 8px; background: #e2e8f0;"
            )
            layout.addWidget(action)

        return sidebar

    def _build_main_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(16)

        card = QtWidgets.QFrame()
        card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        card.setStyleSheet("background: #ffffff; border-radius: 12px;")
        card_layout = QtWidgets.QHBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)

        stats = QtWidgets.QVBoxLayout()
        stat_title = QtWidgets.QLabel("Library Snapshot")
        stat_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #0f172a;")
        stats.addWidget(stat_title)

        stats.addWidget(QtWidgets.QLabel("Tracks indexed: 12,430"))
        stats.addWidget(QtWidgets.QLabel("Duplicates flagged: 84"))
        stats.addWidget(QtWidgets.QLabel("Playlists generated: 18"))
        stats.addStretch(1)

        preview = QtWidgets.QLabel("Artwork Preview")
        preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(180, 180)
        preview.setStyleSheet("background: #e2e8f0; border-radius: 12px;")

        card_layout.addLayout(stats)
        card_layout.addStretch(1)
        card_layout.addWidget(preview)

        opacity_card = QtWidgets.QFrame()
        opacity_card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        opacity_card.setStyleSheet("background: #1e293b; color: #f8fafc; border-radius: 12px;")
        opacity_layout = QtWidgets.QVBoxLayout(opacity_card)
        opacity_layout.setContentsMargins(16, 16, 16, 16)
        opacity_layout.addWidget(QtWidgets.QLabel("Now Playing / Review Queue"))
        opacity_layout.addWidget(QtWidgets.QLabel("Use the controls panel to adjust preview settings."))

        self._overlay_effect = QtWidgets.QGraphicsOpacityEffect(opacity_card)
        self._overlay_effect.setOpacity(0.85)
        opacity_card.setGraphicsEffect(self._overlay_effect)

        flow_panel = QtWidgets.QFrame()
        flow_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        flow_panel.setStyleSheet("background: #ffffff; border-radius: 12px;")
        flow_layout = QtWidgets.QVBoxLayout(flow_panel)
        flow_layout.setContentsMargins(16, 16, 16, 16)

        flow_title = QtWidgets.QLabel("Workflow: Index → Review → Execute")
        flow_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #0f172a;")
        flow_layout.addWidget(flow_title)

        for step_title, step_desc, status in (
            ("1. Scan & Plan", "Analyze the library, build the routing plan.", "Ready"),
            ("2. Review Duplicates", "Approve retain/delete decisions with previews.", "Needs input"),
            ("3. Execute Changes", "Apply moves, writes, playlist updates.", "Locked"),
        ):
            step = QtWidgets.QFrame()
            step.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            step.setStyleSheet("background: #f8fafc; border-radius: 10px;")
            step_layout = QtWidgets.QHBoxLayout(step)
            step_layout.setContentsMargins(12, 10, 12, 10)

            text_layout = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel(step_title)
            title.setStyleSheet("font-weight: 600; color: #111827;")
            desc = QtWidgets.QLabel(step_desc)
            desc.setStyleSheet("color: #6b7280;")
            text_layout.addWidget(title)
            text_layout.addWidget(desc)
            step_layout.addLayout(text_layout)

            status_label = QtWidgets.QLabel(status)
            status_label.setStyleSheet(
                "padding: 4px 10px; border-radius: 999px; background: #e2e8f0; color: #0f172a;"
            )
            step_layout.addStretch(1)
            step_layout.addWidget(status_label)
            flow_layout.addWidget(step)

        queue_panel = QtWidgets.QFrame()
        queue_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        queue_panel.setStyleSheet("background: #ffffff; border-radius: 12px;")
        queue_layout = QtWidgets.QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(16, 16, 16, 16)

        queue_title = QtWidgets.QLabel("Review Queue")
        queue_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #0f172a;")
        queue_layout.addWidget(queue_title)

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
        queue_layout.addWidget(table)

        layout.addWidget(card)
        layout.addWidget(flow_panel)
        layout.addWidget(opacity_card)
        layout.addWidget(queue_panel)
        layout.addWidget(self._build_controls_panel())
        return panel

    def _build_controls_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        panel.setStyleSheet("background: #ffffff; border-radius: 12px;")
        layout = QtWidgets.QHBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
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
