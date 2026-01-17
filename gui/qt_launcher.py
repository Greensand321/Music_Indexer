"""Standalone Qt preview window launcher."""
from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
import shutil
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
Signal = QtCore.Signal if hasattr(QtCore, "Signal") else QtCore.pyqtSignal


class ClickableLabel(QtWidgets.QLabel):
    clicked = Signal()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.clicked.emit()
        super().mousePressEvent(event)


class QtPreviewWindow(QtWidgets.QMainWindow):
    """Qt proof-of-concept window that sketches the planned workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AlphaDEX Qt Preview")
        self.resize(1280, 820)
        self.setWindowOpacity(1.0)
        self._focused_tool = None
        self._fade_targets: list[QtWidgets.QWidget] = []
        self._tool_workspace = None
        self._tool_tabs = None
        self._focus_reset_button = None
        self._sidebar_content = None
        self._sidebar_toggle = None
        self._artwork_label = None
        self._artwork_path = None
        self._tool_workspace_collapsed = 320
        self._tool_workspace_expanded = 640

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

        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Tools")
        label.setStyleSheet("font-size: 13px; font-weight: 600; color: #111827;")
        header.addWidget(label)
        header.addStretch(1)

        toggle = QtWidgets.QToolButton()
        toggle.setText("Collapse")
        toggle.setStyleSheet("font-size: 11px; color: #64748b;")
        toggle.clicked.connect(self._toggle_sidebar)
        header.addWidget(toggle)
        self._sidebar_toggle = toggle
        layout.addLayout(header)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)
        self._sidebar_content = content

        for item in (
            "Indexer",
            "Duplicate Finder",
            "Player",
            "Library Compression",
            "Insights",
        ):
            button = QtWidgets.QPushButton(item)
            button.setStyleSheet(
                "text-align: left; padding: 8px 10px; border-radius: 8px; background: #ffffff;"
            )
            button.clicked.connect(lambda checked=False, name=item: self._focus_tool(name))
            content_layout.addWidget(button)

        content_layout.addStretch(1)

        debug_tools = QtWidgets.QLabel("Debug")
        debug_tools.setStyleSheet("font-size: 12px; font-weight: 600; color: #6b7280;")
        content_layout.addWidget(debug_tools)

        for item in ("Opus Tester", "M4A Tester", "Similarity Inspector"):
            action = QtWidgets.QPushButton(item)
            action.setStyleSheet(
                "text-align: left; padding: 6px 10px; border-radius: 8px; background: #e2e8f0;"
            )
            content_layout.addWidget(action)

        help_button = QtWidgets.QPushButton("Help • AI Prompter")
        help_button.setStyleSheet(
            "margin-top: 6px; text-align: center; padding: 10px 12px; border-radius: 10px; "
            "background: #1d4ed8; color: white; font-weight: 600;"
        )
        help_button.clicked.connect(self._open_ai_prompter)
        content_layout.addWidget(help_button)

        layout.addWidget(content)

        return sidebar

    def _build_main_content(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setSpacing(16)

        snapshot = self._build_snapshot_card()
        workspace = self._build_tool_workspace()
        controls = self._build_controls_panel()

        layout.addWidget(snapshot)
        layout.addWidget(workspace)
        layout.addWidget(controls)

        self._fade_targets = [snapshot, controls]
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

        self._progress = QtWidgets.QProgressBar()
        self._progress.setValue(45)

        form_layout.addRow("Track Title", self._name_input)
        form_layout.addRow("Genre", self._genre_combo)
        form_layout.addRow("", self._favorite_check)
        form_layout.addRow("Metadata", self._sync_radio)
        form_layout.addRow("", self._skip_radio)
        form_layout.addRow("Volume", self._volume_slider)
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

        layout.addWidget(form_group, 2)
        layout.addWidget(actions_group, 1)
        return panel

    def _build_snapshot_card(self) -> QtWidgets.QWidget:
        card = QtWidgets.QFrame()
        card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        card.setStyleSheet("background: #ffffff; border-radius: 12px;")
        card_layout = QtWidgets.QHBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(16)

        stats = QtWidgets.QVBoxLayout()
        stat_title = QtWidgets.QLabel("Library Snapshot")
        stat_title.setStyleSheet("font-size: 16px; font-weight: 600; color: #0f172a;")
        stats.addWidget(stat_title)

        size_label = QtWidgets.QLabel("Library size: 1.42 TB")
        size_label.setStyleSheet("color: #2563eb; font-weight: 600;")
        stats.addWidget(size_label)

        codec_title = QtWidgets.QLabel("Top audio formats")
        codec_title.setStyleSheet("margin-top: 6px; color: #0f172a; font-weight: 600;")
        stats.addWidget(codec_title)

        for codec, count, color in (
            ("FLAC", "7,420 tracks", "#0ea5e9"),
            ("M4A", "3,120 tracks", "#8b5cf6"),
            ("MP3", "1,890 tracks", "#22c55e"),
        ):
            chip = QtWidgets.QLabel(f"{codec}: {count}")
            chip.setStyleSheet(
                "padding: 4px 10px; border-radius: 999px; "
                f"background: {color}; color: white; font-weight: 600;"
            )
            stats.addWidget(chip)

        stats.addStretch(1)

        preview = ClickableLabel()
        preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        preview.setFixedSize(180, 180)
        preview.setStyleSheet(
            "background: #e2e8f0; border-radius: 12px; color: #475569; font-weight: 600;"
        )
        preview.clicked.connect(self._open_artwork_dialog)
        self._artwork_label = preview

        self._load_artwork_from_docs()

        card_layout.addLayout(stats)
        card_layout.addStretch(1)
        card_layout.addWidget(preview)
        return card

    def _build_tool_workspace(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        panel.setStyleSheet("background: #ffffff; border-radius: 12px;")
        panel.setMaximumHeight(self._tool_workspace_collapsed)

        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Tool Workspace")
        title.setStyleSheet("font-size: 15px; font-weight: 600; color: #0f172a;")
        header.addWidget(title)
        header.addStretch(1)

        back_button = QtWidgets.QPushButton("Back to Overview")
        back_button.setStyleSheet(
            "padding: 6px 12px; border-radius: 8px; background: #e2e8f0; color: #0f172a;"
        )
        back_button.clicked.connect(self._reset_focus)
        back_button.setVisible(False)
        header.addWidget(back_button)

        layout.addLayout(header)
        self._focus_reset_button = back_button

        quick_layout = QtWidgets.QHBoxLayout()
        quick_layout.setSpacing(10)
        for item in ("Indexer", "Duplicate Finder", "Player", "Library Compression"):
            button = QtWidgets.QPushButton(item)
            button.setStyleSheet(
                "text-align: left; padding: 10px 12px; border-radius: 10px; "
                "background: #f1f5f9; font-weight: 600; color: #0f172a;"
            )
            button.clicked.connect(lambda checked=False, name=item: self._focus_tool(name))
            quick_layout.addWidget(button)
        quick_layout.addStretch(1)
        layout.addLayout(quick_layout)

        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet(
            "QTabBar::tab { padding: 6px 12px; border-radius: 6px; }"
            "QTabBar::tab:selected { background: #1d4ed8; color: white; }"
        )
        self._tool_tabs = tabs

        tabs.addTab(self._build_tool_tab("Indexer", "Scan, fingerprint, and index new audio."), "Indexer")
        tabs.addTab(
            self._build_tool_tab(
                "Duplicate Finder",
                "Compare fingerprints, flag near-duplicates, and surface decisions.",
            ),
            "Duplicate Finder",
        )
        tabs.addTab(self._build_tool_tab("Player", "Preview tracks and validate metadata."), "Player")
        tabs.addTab(
            self._build_tool_tab(
                "Library Compression",
                "Transcode and archive high-bitrate collections.",
            ),
            "Library Compression",
        )
        tabs.addTab(self._build_tool_tab("Insights", "Trend analysis and library health metrics."), "Insights")

        layout.addWidget(tabs)
        self._tool_workspace = panel
        return panel

    def _build_tool_tab(self, title: str, description: str) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel(title)
        heading.setStyleSheet("font-size: 14px; font-weight: 600; color: #0f172a;")
        body = QtWidgets.QLabel(description)
        body.setWordWrap(True)
        body.setStyleSheet("color: #475569;")

        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addStretch(1)
        return widget

    def _load_artwork_from_docs(self) -> None:
        docs_dir = Path(__file__).resolve().parents[1] / "docs"
        candidates = []
        for ext in ("png", "jpg", "jpeg", "bmp", "gif", "webp"):
            candidates.append(docs_dir / f"library_artwork.{ext}")
        for candidate in candidates:
            if candidate.exists():
                self._apply_artwork(candidate)
                return
        if self._artwork_label:
            self._artwork_label.setText("Click to add artwork")

    def _apply_artwork(self, path: Path) -> None:
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            if self._artwork_label:
                self._artwork_label.setText("Click to add artwork")
            return
        if self._artwork_label:
            scaled = pixmap.scaled(
                self._artwork_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._artwork_label.setPixmap(scaled)
            self._artwork_label.setText("")
            self._artwork_path = path

    def _open_artwork_dialog(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Library Artwork",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
        )
        if not file_path:
            return
        docs_dir = Path(__file__).resolve().parents[1] / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        source = Path(file_path)
        destination = docs_dir / f"library_artwork{source.suffix.lower()}"
        shutil.copy2(source, destination)
        self._apply_artwork(destination)

    def _toggle_sidebar(self) -> None:
        if not self._sidebar_content or not self._sidebar_toggle:
            return
        collapsed = self._sidebar_content.isVisible()
        self._sidebar_content.setVisible(not collapsed)
        if collapsed:
            self._sidebar_toggle.setText("Expand")
            self._sidebar_content.parentWidget().setFixedWidth(64)
        else:
            self._sidebar_toggle.setText("Collapse")
            self._sidebar_content.parentWidget().setFixedWidth(240)

    def _open_ai_prompter(self) -> None:
        self.statusBar().showMessage("AI Prompter launched (placeholder).", 3000)

    def _focus_tool(self, name: str) -> None:
        if self._tool_tabs:
            for index in range(self._tool_tabs.count()):
                if self._tool_tabs.tabText(index) == name:
                    self._tool_tabs.setCurrentIndex(index)
                    break
        self._focused_tool = name
        self._set_focus_state(True)

    def _reset_focus(self) -> None:
        self._focused_tool = None
        self._set_focus_state(False)

    def _set_focus_state(self, focused: bool) -> None:
        if not self._tool_workspace:
            return

        self._focus_reset_button.setVisible(focused)
        animation_group = QtCore.QParallelAnimationGroup(self)

        target_height = (
            self._tool_workspace_expanded if focused else self._tool_workspace_collapsed
        )
        workspace_anim = QtCore.QPropertyAnimation(self._tool_workspace, b"maximumHeight")
        workspace_anim.setDuration(240)
        workspace_anim.setStartValue(self._tool_workspace.maximumHeight())
        workspace_anim.setEndValue(target_height)
        animation_group.addAnimation(workspace_anim)

        for widget in self._fade_targets:
            effect = widget.graphicsEffect()
            if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
                effect = QtWidgets.QGraphicsOpacityEffect(widget)
                widget.setGraphicsEffect(effect)
            fade_anim = QtCore.QPropertyAnimation(effect, b"opacity")
            fade_anim.setDuration(240)
            fade_anim.setStartValue(effect.opacity())
            fade_anim.setEndValue(0.15 if focused else 1.0)
            animation_group.addAnimation(fade_anim)
            widget.setEnabled(not focused)

        animation_group.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

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


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = QtPreviewWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
