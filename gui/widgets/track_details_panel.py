"""Track details panel showing metadata for hovered/selected tracks."""
from __future__ import annotations

from typing import Optional, Dict, List

from gui.compat import QtCore, QtGui, QtWidgets


class TrackDetailsPanel(QtWidgets.QWidget):
    """Shows track metadata when point is hovered or selected.

    Displays:
    - Album art (if available)
    - Artist, title, album
    - Genre, BPM, duration
    - Cluster assignment
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._current_metadata: Dict | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Title
        title = QtWidgets.QLabel("Track Details")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Scroll area for details
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #ddd; }")

        # Content widget
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)

        # Album art placeholder
        self._album_art = QtWidgets.QLabel()
        self._album_art.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._album_art.setStyleSheet("background-color: #eee; border: 1px solid #ddd;")
        self._album_art.setMinimumHeight(120)
        self._album_art.setMaximumHeight(120)
        content_layout.addWidget(self._album_art)

        # Metadata fields
        self._fields = {}
        field_names = [
            "artist",
            "title",
            "album",
            "genres",
            "bpm",
            "duration",
            "bitrate",
            "cluster",
        ]

        for field_name in field_names:
            field_layout = QtWidgets.QHBoxLayout()
            field_layout.setSpacing(8)

            label = QtWidgets.QLabel(f"{field_name.title()}:")
            label.setStyleSheet("font-weight: bold; width: 80px; font-size: 11px;")
            label.setMinimumWidth(80)
            field_layout.addWidget(label, 0)

            value_label = QtWidgets.QLabel("")
            value_label.setStyleSheet("font-size: 11px; color: #333;")
            value_label.setWordWrap(True)
            field_layout.addWidget(value_label, 1)

            content_layout.addLayout(field_layout)
            self._fields[field_name] = value_label

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(4)

        self.play_btn = QtWidgets.QPushButton("▶ Play")
        self.play_btn.setToolTip("Play this track")
        button_layout.addWidget(self.play_btn)

        self.details_btn = QtWidgets.QPushButton("ℹ Details")
        self.details_btn.setToolTip("Open full track details")
        button_layout.addWidget(self.details_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)
        self._clear_display()

    def set_track(self, metadata: Dict | None) -> None:
        """Set displayed track metadata.

        Parameters
        ----------
        metadata : dict or None
            Track metadata dict with keys: artist, title, album, genres, bpm, etc.
        """
        self._current_metadata = metadata

        if metadata is None:
            self._clear_display()
            return

        # Populate fields
        for field_name, value_label in self._fields.items():
            value = metadata.get(field_name, "—")

            # Format special fields
            if field_name == "genres" and isinstance(value, list):
                value = ", ".join(value)
            elif field_name == "duration" and isinstance(value, (int, float)):
                minutes = int(value // 60)
                seconds = int(value % 60)
                value = f"{minutes}:{seconds:02d}"
            elif field_name == "bpm" and isinstance(value, (int, float)):
                value = f"{value:.0f} BPM"

            value_label.setText(str(value) if value else "—")

    def _clear_display(self) -> None:
        """Clear all displayed information."""
        self._album_art.setPixmap(QtGui.QPixmap())
        self._album_art.setText("No track selected")
        self._album_art.setStyleSheet("background-color: #eee; color: #999; border: 1px solid #ddd;")

        for label in self._fields.values():
            label.setText("—")

    def get_current_metadata(self) -> Dict | None:
        """Get currently displayed metadata."""
        return self._current_metadata
