"""Three.js 3D cluster viewer using QWebEngineView and qwebchannel."""
from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List

from gui.compat import QtCore, QtWidgets, QtWebEngineWidgets, QtWebChannel, Signal, Slot

logger = logging.getLogger(__name__)


class GraphBridge(QtCore.QObject):
    """Python-JavaScript bridge for Three.js viewer communication."""

    clusterDataUpdated = Signal(str)  # JSON string

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._cluster_data: Optional[str] = None

    @Slot(result=str)
    def getClusterData(self) -> str:
        """Return current cluster data as JSON string."""
        return self._cluster_data or "{}"

    @Slot(str)
    def onSelectionChanged(self, indices_json: str) -> None:
        """Handle selection change from JavaScript."""
        try:
            indices = json.loads(indices_json)
            # Emit signal to notify parent widget
            self.selectionChanged.emit(indices)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse selection JSON: {e}")

    def updateClusterData(self, data: str) -> None:
        """Update cluster data and notify JavaScript."""
        self._cluster_data = data
        self.clusterDataUpdated.emit(data)

    # Custom signal for selection changes
    selectionChanged = Signal(list)  # list of indices


class ThreeJsViewer(QtWidgets.QWidget):
    """Three.js 3D cluster viewer embedded in QWebEngineView."""

    point_clicked = QtCore.Signal(int)  # index
    points_selected = QtCore.Signal(list)  # list of indices
    hover_changed = QtCore.Signal(int)  # index or -1

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._setup_ui()
        self._bridge = GraphBridge()
        self._bridge.selectionChanged.connect(self._on_selection_changed)
        self._cluster_data: Optional[dict] = None
        self._selected_indices: List[int] = []

    def _setup_ui(self) -> None:
        """Set up the web engine view."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create web engine view
        self._view = QtWebEngineWidgets.QWebEngineView()

        # Create web channel
        self._channel = QtWebChannel.QWebChannel()
        self._channel.registerObject("graphBridge", self._bridge)
        self._view.page().setWebChannel(self._channel)

        layout.addWidget(self._view)

        # Load HTML file
        self._load_html()

    def _load_html(self) -> None:
        """Load the Three.js viewer HTML file."""
        html_path = Path(__file__).parent.parent / "web" / "3d_viewer.html"

        if not html_path.exists():
            logger.error(f"Three.js HTML file not found: {html_path}")
            self._view.setHtml(
                "<h1>Error</h1><p>Three.js viewer HTML not found.</p>"
            )
            return

        # Load HTML with file:// URL
        try:
            self._view.load(QtCore.QUrl.fromLocalFile(str(html_path)))
            logger.info(f"Loaded Three.js viewer from {html_path}")
        except Exception as e:
            logger.error(f"Failed to load Three.js viewer: {e}")
            self._view.setHtml(f"<h1>Error</h1><p>{str(e)}</p>")

    def set_data(
        self,
        X: np.ndarray,
        clusters: np.ndarray,
        labels: List[str],
        metadata: List[dict],
    ) -> None:
        """Set cluster data for visualization.

        Args:
            X: (N, 3) array of 3D coordinates
            clusters: (N,) array of cluster IDs
            labels: (N,) list of track paths
            metadata: (N,) list of metadata dicts
        """
        if len(X) == 0:
            logger.warning("Empty cluster data provided")
            return

        try:
            # Validate shapes
            if X.ndim != 2 or X.shape[1] != 3:
                logger.error(f"Invalid X shape: {X.shape}, expected (N, 3)")
                return

            if len(clusters) != len(X):
                logger.error(f"Cluster length mismatch: {len(clusters)} != {len(X)}")
                return

            # Normalize coordinates to [0, 100] range for better visualization
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            X_range = X_max - X_min
            X_range[X_range == 0] = 1  # Avoid division by zero
            X_normalized = (X - X_min) / X_range * 100 - 50

            # Generate colors by cluster
            unique_clusters = np.unique(clusters)
            colors = []
            cluster_colors = {}

            for i, cluster_id in enumerate(unique_clusters):
                # Generate distinct colors
                hue = i / max(len(unique_clusters), 1)
                r = int(100 + 155 * (1 - hue))
                g = int(100 + 155 * hue)
                b = 200
                cluster_colors[int(cluster_id)] = (r, g, b)

            for cluster_id in clusters:
                colors.append(cluster_colors[int(cluster_id)])

            # Build JSON data for JavaScript
            data = {
                "positions": X_normalized.tolist(),
                "colors": colors,
                "clusters": clusters.astype(int).tolist(),
                "tracks": labels,
                "metadata": [
                    {
                        **meta,
                        "cluster": int(clusters[i]),
                    }
                    for i, meta in enumerate(metadata)
                ],
            }

            # Store and update
            self._cluster_data = data
            data_json = json.dumps(data)
            self._bridge.updateClusterData(data_json)

            logger.info(f"Set cluster data: {len(X)} points")

        except Exception as e:
            logger.error(f"Failed to set cluster data: {e}")
            import traceback

            traceback.print_exc()

    @Slot(list)
    def _on_selection_changed(self, indices: List[int]) -> None:
        """Handle selection change from JavaScript."""
        self._selected_indices = indices
        self.points_selected.emit(indices)

    def export_selection_paths(self) -> List[str]:
        """Export selected track paths."""
        if not self._cluster_data:
            return []

        tracks = self._cluster_data.get("tracks", [])
        return [tracks[i] for i in self._selected_indices if i < len(tracks)]

    def clear_selection(self) -> None:
        """Clear all selections."""
        self._selected_indices = []

    def highlight_cluster(self, cluster_id: int) -> None:
        """Highlight points in a cluster (for future UI enhancement)."""
        logger.info(f"Highlight cluster {cluster_id} (not yet implemented in Three.js viewer)")

    def set_cluster_visible(self, cluster_id: int, visible: bool) -> None:
        """Show/hide points in a cluster (for future UI enhancement)."""
        logger.info(
            f"Set cluster {cluster_id} visible={visible} (not yet implemented in Three.js viewer)"
        )

    def fit_view(self) -> None:
        """Fit camera to show all points (handled automatically in JavaScript)."""
        pass
