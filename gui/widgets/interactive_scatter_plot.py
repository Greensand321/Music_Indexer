"""Interactive scatter plot visualization using PyQtGraph."""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional, List, Tuple

from gui.compat import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:
    pg = None


class InteractiveScatterPlot(QtWidgets.QWidget):
    """High-performance scatter plot with interactive features.

    Features:
    - Smooth panning and zooming
    - Point selection via lasso or rectangle
    - Hover tooltips with track metadata
    - Cluster highlighting
    - Color and size customization
    """

    point_clicked = QtCore.Signal(int)  # index
    points_selected = QtCore.Signal(list)  # list of indices
    hover_changed = QtCore.Signal(int)  # index or -1

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        if pg is None:
            raise ImportError("PyQtGraph is required for scatter plot visualization")

        self._setup_ui()
        self._setup_interactions()

        # Data storage
        self._X: np.ndarray | None = None  # (n_samples, 2) array
        self._clusters: np.ndarray | None = None  # (n_samples,) cluster IDs
        self._labels: List[str] | None = None  # (n_samples,) track names/paths
        self._metadata: List[dict] | None = None  # (n_samples,) metadata dicts
        self._colors: np.ndarray | None = None  # (n_samples, 3) RGB colors
        self._sizes: np.ndarray | None = None  # (n_samples,) point sizes

        # State
        self._selected_indices: set[int] = set()
        self._hovered_index: int = -1
        self._cluster_visibility: dict[int, bool] = {}

    def _setup_ui(self) -> None:
        """Create the PyQtGraph plot widget."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "X")
        self.plot_widget.setLabel("left", "Y")
        self.plot_widget.setTitle("Cluster Visualization")
        self.plot_widget.setBackground("w")
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.enableAutoRange()

        layout.addWidget(self.plot_widget)

        # Scatter plot item
        self.scatter = pg.ScatterPlotItem()
        self.scatter.sigClicked.connect(self._on_scatter_clicked)
        self.plot_widget.addItem(self.scatter)

        # Hover tracking via ViewBox
        self.view_box = self.plot_widget.getViewBox()

        self.setLayout(layout)

    def _setup_interactions(self) -> None:
        """Setup mouse event handling."""
        self.view_box.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def set_data(
        self,
        X: np.ndarray,
        clusters: np.ndarray,
        labels: List[str] | None = None,
        metadata: List[dict] | None = None,
    ) -> None:
        """Set scatter plot data.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, 2) - 2D coordinates
        clusters : np.ndarray
            Shape (n_samples,) - cluster ID for each point
        labels : List[str], optional
            Shape (n_samples,) - track names/paths
        metadata : List[dict], optional
            Shape (n_samples,) - metadata for each track
        """
        self._X = np.asarray(X, dtype=np.float32)
        self._clusters = np.asarray(clusters, dtype=np.int32)
        self._labels = labels or [f"Track {i}" for i in range(len(X))]
        self._metadata = metadata or [{} for _ in range(len(X))]

        # Validate data shapes and dimensions
        if len(self._X) == 0:
            raise ValueError("Data is empty (0 samples)")

        if self._X.ndim != 2 or self._X.shape[1] != 2:
            raise ValueError(f"X must have shape (n_samples, 2), got {self._X.shape}")

        if len(self._X) != len(self._clusters):
            raise ValueError("X and clusters must have same length")
        if len(self._X) != len(self._labels):
            raise ValueError("X and labels must have same length")
        if len(self._X) != len(self._metadata):
            raise ValueError("X and metadata must have same length")

        # Default colors by cluster
        self._update_colors_by_cluster()
        self._update_sizes_default()

        # Initialize cluster visibility
        self._cluster_visibility = {int(c): True for c in np.unique(self._clusters)}

        self._render()

    def _update_colors_by_cluster(self) -> None:
        """Generate colors based on clusters."""
        unique_clusters = np.unique(self._clusters)
        n_clusters = len(unique_clusters)

        # Generate distinct colors using HSV space
        colors = []
        for i in range(n_clusters):
            hue = i / max(n_clusters, 1)
            color = pg.mkColor(QtGui.QColor.fromHsv(int(hue * 360), 200, 220))
            colors.append(color)

        cluster_to_color = {c: colors[i] for i, c in enumerate(unique_clusters)}
        self._colors = np.array([cluster_to_color[c] for c in self._clusters])

    def _update_sizes_default(self) -> None:
        """Set default point sizes."""
        self._sizes = np.ones(len(self._X), dtype=np.float32) * 10

    def _render(self) -> None:
        """Render scatter plot."""
        if self._X is None or len(self._X) == 0:
            return

        # Create spots for visible points
        spots = []
        for i, (x, y) in enumerate(self._X):
            cluster_id = self._clusters[i]

            # Skip hidden clusters
            if not self._cluster_visibility.get(cluster_id, True):
                continue

            color = self._colors[i] if self._colors is not None else "blue"
            size = self._sizes[i] if self._sizes is not None else 10

            # Highlight selected points
            pen = None
            if i in self._selected_indices:
                pen = pg.mkPen("red", width=2)

            spot = {
                "pos": (x, y),
                "size": size,
                "brush": pg.mkBrush(color),
                "pen": pen or pg.mkPen(color, width=0.5),
                "data": i,  # Store index
            }
            spots.append(spot)

        self.scatter.setData(spots=spots)
        self.plot_widget.enableAutoRange()

    def _on_scatter_clicked(self, plot, points) -> None:
        """Handle point clicks."""
        if len(points) > 0:
            index = points[0].data()
            self.point_clicked.emit(index)
            self.set_selection([index])

    def _on_mouse_moved(self, pos: QtCore.QPointF) -> None:
        """Handle mouse hover for tooltips."""
        if self._X is None or len(self._X) == 0:
            return

        # Validate array shape
        if self._X.ndim != 2 or self._X.shape[1] != 2:
            return

        # Get view coordinates
        view_coords = self.view_box.mapSceneToView(pos)
        x, y = view_coords.x(), view_coords.y()

        # Find nearest point
        try:
            distances = np.sqrt((self._X[:, 0] - x) ** 2 + (self._X[:, 1] - y) ** 2)
            nearest_idx = np.argmin(distances)
        except (ValueError, IndexError):
            # Handle empty or invalid array
            return

        # Only update if close enough
        hover_dist = distances[nearest_idx]

        # Calculate proximity threshold safely
        try:
            x_range = self._X[:, 0].max() - self._X[:, 0].min()
            y_range = self._X[:, 1].max() - self._X[:, 1].min()
            data_range = max(x_range, y_range, 0.001)  # Avoid division by zero
            proximity_threshold = 0.1 * data_range
        except (ValueError, IndexError):
            # If we can't compute range, use a default threshold
            proximity_threshold = 0.1

        if hover_dist < proximity_threshold:
            if self._hovered_index != nearest_idx:
                self._hovered_index = nearest_idx
                self.hover_changed.emit(nearest_idx)

                # Show tooltip
                self._show_tooltip(nearest_idx)
        else:
            if self._hovered_index != -1:
                self._hovered_index = -1
                self.hover_changed.emit(-1)

    def _show_tooltip(self, index: int) -> None:
        """Show tooltip for a point."""
        if index < 0 or index >= len(self._labels):
            return

        label = self._labels[index]
        metadata = self._metadata[index] if self._metadata else {}

        # Format tooltip
        tooltip_lines = [label]
        for key, value in list(metadata.items())[:5]:  # Show first 5 fields
            tooltip_lines.append(f"{key}: {value}")

        tooltip_text = "\n".join(tooltip_lines)
        self.scatter.setToolTip(tooltip_text)

    def set_selection(self, indices: List[int] | None = None) -> None:
        """Set selected points.

        Parameters
        ----------
        indices : List[int], optional
            List of indices to select. If None, clears selection.
        """
        self._selected_indices = set(indices or [])
        self.points_selected.emit(list(self._selected_indices))
        self._render()

    def get_selection(self) -> List[int]:
        """Get currently selected point indices."""
        return list(self._selected_indices)

    def toggle_cluster_visibility(self, cluster_id: int) -> None:
        """Show or hide a cluster."""
        if cluster_id in self._cluster_visibility:
            self._cluster_visibility[cluster_id] = not self._cluster_visibility[cluster_id]
            self._render()

    def set_cluster_visible(self, cluster_id: int, visible: bool) -> None:
        """Set cluster visibility."""
        if cluster_id in self._cluster_visibility:
            self._cluster_visibility[cluster_id] = visible
            self._render()

    def highlight_cluster(self, cluster_id: int | None) -> None:
        """Highlight all points in a cluster."""
        if cluster_id is None:
            self.set_selection([])
        else:
            indices = np.where(self._clusters == cluster_id)[0]
            self.set_selection(indices.tolist())

    def fit_view(self) -> None:
        """Fit all visible data in view."""
        self.plot_widget.enableAutoRange()

    def export_selection_paths(self) -> List[str]:
        """Get file paths of selected tracks."""
        paths = []
        for idx in self._selected_indices:
            if idx < len(self._labels):
                paths.append(self._labels[idx])
        return paths

    def export_selection_metadata(self) -> List[dict]:
        """Get metadata of selected tracks."""
        metadata = []
        for idx in self._selected_indices:
            if idx < len(self._metadata):
                metadata.append(self._metadata[idx])
        return metadata
