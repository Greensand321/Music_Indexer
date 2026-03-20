"""Cluster legend widget for showing/hiding clusters."""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional, List, Dict

from gui.compat import QtCore, QtGui, QtWidgets


class _ClusterClickableLabel(QtWidgets.QLabel):
    """Custom label that emits a signal when clicked."""

    cluster_clicked = QtCore.Signal(int)  # cluster_id

    def __init__(self, text: str, cluster_id: int) -> None:
        super().__init__(text)
        self.cluster_id = cluster_id

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press to emit cluster click signal."""
        self.cluster_clicked.emit(self.cluster_id)
        super().mousePressEvent(event)


class ClusterLegendWidget(QtWidgets.QWidget):
    """Widget showing cluster list with visibility toggles.

    Features:
    - Checkbox to show/hide each cluster
    - Track count per cluster
    - Cluster color indicator
    - Genre/metadata summary
    - Click to highlight cluster
    """

    cluster_toggled = QtCore.Signal(int, bool)  # cluster_id, visible
    cluster_selected = QtCore.Signal(int)  # cluster_id

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._clusters: Dict[int, dict] = {}  # cluster_id -> {size, color, genres, ...}

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Title
        title = QtWidgets.QLabel("Clusters")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Scroll area for cluster list
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #ddd; }")

        self._cluster_container = QtWidgets.QWidget()
        self._cluster_layout = QtWidgets.QVBoxLayout(self._cluster_container)
        self._cluster_layout.setContentsMargins(0, 0, 0, 0)
        self._cluster_layout.setSpacing(4)
        scroll.setWidget(self._cluster_container)

        layout.addWidget(scroll, 1)

        # Summary
        self._summary_label = QtWidgets.QLabel("No clusters")
        self._summary_label.setStyleSheet("color: #666; font-size: 11px;")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self.setLayout(layout)

    def set_clusters(
        self,
        clusters: np.ndarray,
        cluster_info: Dict[int, dict] | None = None,
        colors: np.ndarray | None = None,
    ) -> None:
        """Set cluster data.

        Parameters
        ----------
        clusters : np.ndarray
            Shape (n_samples,) - cluster ID for each sample
        cluster_info : Dict[int, dict], optional
            Additional info per cluster: {cluster_id: {genres, tempo, ...}}
        colors : np.ndarray, optional
            Shape (n_clusters, 3) - RGB colors per cluster
        """
        # Clear existing widgets completely
        # First disconnect signals and remove from layout
        while self._cluster_layout.count():
            widget = self._cluster_layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)  # Immediate removal
                widget.deleteLater()

        self._clusters = {}

        unique_clusters = sorted(np.unique(clusters))
        cluster_info = cluster_info or {}

        for i, cluster_id in enumerate(unique_clusters):
            count = np.sum(clusters == cluster_id)
            color = colors[i] if colors is not None and i < len(colors) else None
            info = cluster_info.get(cluster_id, {})

            self._clusters[cluster_id] = {
                "size": count,
                "color": color,
                "info": info,
                "visible": True,
            }

            # Create cluster item widget
            item = self._create_cluster_item(cluster_id, count, color, info)
            self._cluster_layout.addWidget(item)

        self._cluster_layout.addStretch()

        # Update summary
        self._update_summary()

    def _create_cluster_item(
        self,
        cluster_id: int,
        count: int,
        color: tuple | None = None,
        info: dict | None = None,
    ) -> QtWidgets.QWidget:
        """Create a single cluster item widget."""
        item = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(item)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Checkbox
        checkbox = QtWidgets.QCheckBox()
        checkbox.setChecked(True)
        # Use functools.partial or default argument to avoid closure issues
        checkbox.stateChanged.connect(
            lambda state, cid=cluster_id: self._on_cluster_toggled(cid, state == 2)
        )
        layout.addWidget(checkbox, 0)

        # Color indicator
        if color is not None:
            color_label = QtWidgets.QLabel("■")
            color_label.setStyleSheet(f"color: rgb({int(color[0])}, {int(color[1])}, {int(color[2])});")
            color_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(color_label, 0)

        # Cluster name and info
        info_text = f"Cluster {cluster_id}"
        info = info or {}

        if "genres" in info and info["genres"]:
            genres_str = ", ".join(info["genres"][:2])
            if len(info["genres"]) > 2:
                genres_str += ", ..."
            info_text += f"\n{genres_str}"

        if "tempo" in info:
            info_text += f"\nTempo: {info['tempo']}"

        text_label = _ClusterClickableLabel(f"{info_text}\n({count} tracks)", cluster_id)
        text_label.setStyleSheet("font-size: 11px; color: #333;")
        text_label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        text_label.cluster_clicked.connect(self._on_cluster_selected)
        layout.addWidget(text_label, 1)

        return item

    def _on_cluster_toggled(self, cluster_id: int, visible: bool) -> None:
        """Handle cluster visibility toggle."""
        if cluster_id in self._clusters:
            self._clusters[cluster_id]["visible"] = visible
            self.cluster_toggled.emit(cluster_id, visible)

    def _on_cluster_selected(self, cluster_id: int) -> None:
        """Handle cluster selection."""
        self.cluster_selected.emit(cluster_id)

    def _update_summary(self) -> None:
        """Update summary text."""
        total_clusters = len(self._clusters)
        total_tracks = sum(c["size"] for c in self._clusters.values())
        self._summary_label.setText(f"{total_clusters} clusters • {total_tracks} tracks")

    def get_visible_clusters(self) -> List[int]:
        """Get list of visible cluster IDs."""
        return [cid for cid, info in self._clusters.items() if info["visible"]]

    def set_cluster_visible(self, cluster_id: int, visible: bool) -> None:
        """Set cluster visibility programmatically."""
        if cluster_id in self._clusters:
            self._clusters[cluster_id]["visible"] = visible
