"""Interactive 3D scatter plot visualization using PyQtGraph with proper depth visualization."""
from __future__ import annotations

import numpy as np
from typing import Callable, Optional, List, Tuple

from gui.compat import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import GLMeshItem
except ImportError:
    pg = None
    gl = None


class Interactive3DScatterPlot(QtWidgets.QWidget):
    """True 3D scatter plot with proper depth, rotation, zoom, and perspective.

    Features:
    - Actual 3D rendering with X, Y, Z axes
    - Perspective projection showing depth
    - Interactive mouse rotation (free camera movement)
    - Scroll zoom with proper scaling
    - Point selection and cluster highlighting
    - Multiple view presets (XY plane, XZ plane, YZ plane, 3D isometric)
    - Visible depth cues and spatial reference grid
    """

    point_clicked = QtCore.Signal(int)  # index
    points_selected = QtCore.Signal(list)  # list of indices
    hover_changed = QtCore.Signal(int)  # index or -1

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        if pg is None or gl is None:
            raise ImportError("PyQtGraph with OpenGL is required for 3D scatter plot visualization")

        self._pg_available = True
        self._setup_ui()

        # Data storage
        self._X: np.ndarray | None = None  # (n_samples, 3) array - TRUE 3D coordinates
        self._clusters: np.ndarray | None = None  # (n_samples,) cluster IDs
        self._labels: List[str] | None = None  # (n_samples,) track names/paths
        self._metadata: List[dict] | None = None  # (n_samples,) metadata dicts
        self._colors: np.ndarray | None = None  # (n_samples, 4) RGBA colors
        self._sizes: np.ndarray | None = None  # (n_samples,) point sizes

        # State
        self._selected_indices: set[int] = set()
        self._hovered_index: int = -1
        self._cluster_visibility: dict[int, bool] = {}

        # Camera state
        self._data_bounds = None

    def _setup_ui(self) -> None:
        """Create the PyQtGraph OpenGL widget with proper 3D setup."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create OpenGL view widget
        self.view = gl.GLViewWidget()

        # Configure view for better 3D visualization
        self.view.opts['distance'] = 100  # Start at reasonable distance
        self.view.opts['fov'] = 60  # Field of view
        self.view.opts['azimuth'] = 45  # Rotation around vertical axis
        self.view.opts['elevation'] = 30  # Tilt angle
        self.view.setCameraPosition(distance=100, elevation=30, azimuth=45)

        # Add lighting for depth perception
        self.view.setBackgroundColor(0.1, 0.1, 0.1, 1.0)

        # Create coordinate axes (RED=X, GREEN=Y, BLUE=Z)
        self.axis = gl.GLAxisItem()
        self.axis.setSize(100, 100, 100)
        self.view.addItem(self.axis)

        # Add a reference grid at Z=0 plane for spatial orientation
        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)
        self.grid.translate(0, 0, 0)
        self.view.addItem(self.grid)

        # Add another grid at top for Z dimension reference
        self.grid_top = gl.GLGridItem()
        self.grid_top.scale(2, 2, 1)
        self.grid_top.translate(0, 0, 100)
        self.grid_top.setOpacity(0.3)
        self.view.addItem(self.grid_top)

        # Create scatter plot with initial dummy data
        self.scatter = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]], dtype=np.float32),
            size=10.0,
            color=np.array([[1, 1, 1, 1]], dtype=np.float32),
            pxMode=False
        )
        self.view.addItem(self.scatter)

        layout.addWidget(self.view, 1)

        # Control panel with instructions and presets
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(6)
        control_layout.setContentsMargins(8, 4, 8, 4)

        # Help text
        help_label = QtWidgets.QLabel(
            "🖱️ Drag to rotate | 🔍 Scroll to zoom | Middle-drag to pan | "
            "← → Axes show depth (X red, Y green, Z blue)"
        )
        help_label.setStyleSheet("font-size: 9px; color: #888;")
        control_layout.addWidget(help_label, 1)

        # View preset buttons
        control_layout.addWidget(QtWidgets.QLabel(" View:"))

        views = [
            ("XY", 90, 0, "Top-down view"),
            ("XZ", 90, 90, "Side view"),
            ("YZ", 0, 0, "Front view"),
            ("3D", 45, 45, "Isometric 3D"),
        ]

        for label, elev, azim, tooltip in views:
            btn = QtWidgets.QPushButton(label)
            btn.setMaximumWidth(45)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked, e=elev, a=azim: self._set_view_preset(e, a))
            control_layout.addWidget(btn)

        # Reset view button
        reset_btn = QtWidgets.QPushButton("↺ Reset")
        reset_btn.setMaximumWidth(60)
        reset_btn.setToolTip("Reset to default 3D isometric view")
        reset_btn.clicked.connect(self._reset_view)
        control_layout.addWidget(reset_btn)

        layout.addLayout(control_layout)

    def set_data(
        self,
        X: np.ndarray,
        clusters: np.ndarray,
        labels: List[str] | None = None,
        metadata: List[dict] | None = None,
    ) -> None:
        """Set the 3D scatter plot data.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, 3) - 3D coordinates. MUST have 3 columns!
        clusters : np.ndarray
            Shape (n_samples,) - cluster ID for each point
        labels : List[str], optional
            Track names/paths for each point
        metadata : List[dict], optional
            Track metadata for hover tooltips
        """
        print(f"[3D] Received data shape: {X.shape}")

        self._X = np.asarray(X, dtype=np.float32)
        self._clusters = np.asarray(clusters, dtype=np.int32)
        self._labels = labels or [f"Track {i}" for i in range(len(X))]
        self._metadata = metadata or [{} for _ in range(len(X))]

        # CRITICAL: Validate that we have 3D data
        if len(self._X) == 0:
            raise ValueError("Data is empty (0 samples)")

        if self._X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {self._X.ndim}D")

        if self._X.shape[1] != 3:
            raise ValueError(f"X must have 3 columns for 3D plot, got {self._X.shape[1]}")

        print(f"[3D] Data validated: {len(self._X)} points with 3 dimensions")

        if len(self._X) != len(self._clusters):
            raise ValueError(f"Length mismatch: X has {len(self._X)} rows, clusters has {len(self._clusters)}")

        # Generate colors by cluster
        self._update_colors_by_cluster()
        self._update_sizes_default()

        # Initialize cluster visibility
        self._cluster_visibility = {int(c): True for c in np.unique(self._clusters)}

        # Compute data bounds for camera positioning
        self._compute_data_bounds()

        # Render the data
        self._render()

    def _compute_data_bounds(self) -> None:
        """Compute data bounds for proper camera and grid positioning."""
        if self._X is None or len(self._X) == 0:
            self._data_bounds = None
            return

        mins = np.min(self._X, axis=0)
        maxs = np.max(self._X, axis=0)
        self._data_bounds = (mins, maxs)

        print(f"[3D] Data bounds: X[{mins[0]:.1f}, {maxs[0]:.1f}], "
              f"Y[{mins[1]:.1f}, {maxs[1]:.1f}], Z[{mins[2]:.1f}, {maxs[2]:.1f}]")

    def _update_colors_by_cluster(self) -> None:
        """Generate distinct colors for each cluster in RGBA format."""
        unique_clusters = np.unique(self._clusters)
        n_clusters = len(unique_clusters)

        # Generate colors in HSV space for better distinction
        colors = []
        for i in range(n_clusters):
            hue = i / max(n_clusters, 1)
            saturation = 0.9
            value = 0.95
            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((r, g, b, 0.8))  # Add alpha for slight transparency

        # Map cluster IDs to colors
        cluster_to_color = {int(cid): colors[i] for i, cid in enumerate(sorted(unique_clusters))}
        self._colors = np.array([cluster_to_color[int(cid)] for cid in self._clusters], dtype=np.float32)

    def _update_sizes_default(self) -> None:
        """Generate default point sizes for visualization."""
        self._sizes = np.full(len(self._X), 8.0, dtype=np.float32)

    def _render(self) -> None:
        """Render the 3D scatter plot with proper depth and perspective."""
        if self._X is None or len(self._X) == 0:
            print("[3D] No data to render")
            return

        try:
            # Filter by cluster visibility
            visible_mask = np.array([
                self._cluster_visibility.get(int(self._clusters[i]), True)
                for i in range(len(self._clusters))
            ])
            visible_indices = np.where(visible_mask)[0]

            if len(visible_indices) == 0:
                print("[3D] No visible clusters")
                self.scatter.setData(pos=np.empty((0, 3), dtype=np.float32))
                return

            # Get visible data
            visible_pos = self._X[visible_indices].astype(np.float32)
            visible_colors = self._colors[visible_indices].astype(np.float32).copy()
            visible_sizes = self._sizes[visible_indices].astype(np.float32).copy()

            # Highlight selected points (make them red and larger)
            for idx in self._selected_indices:
                if idx in visible_indices:
                    match_idx = np.where(visible_indices == idx)[0][0]
                    visible_colors[match_idx] = (1.0, 0.0, 0.0, 1.0)  # Red, fully opaque
                    visible_sizes[match_idx] *= 2.5  # Much larger

            # Ensure proper data types for OpenGL
            visible_pos = np.asarray(visible_pos, dtype=np.float32)
            visible_colors = np.asarray(visible_colors, dtype=np.float32)
            visible_sizes = np.asarray(visible_sizes, dtype=np.float32)

            print(f"[3D] Rendering {len(visible_pos)} points in 3D space")
            print(f"[3D]   Position shape: {visible_pos.shape}")
            print(f"[3D]   Color shape: {visible_colors.shape}")
            print(f"[3D]   Size shape: {visible_sizes.shape}")

            # Render all points with TRUE 3D coordinates
            self.scatter.setData(
                pos=visible_pos,  # 3D coordinates (X, Y, Z)
                color=visible_colors,
                size=visible_sizes,
                pxMode=False  # Use 3D sizes, not pixel sizes
            )

            print("[3D] ✓ Render complete - TRUE 3D visualization active")

        except Exception as e:
            print(f"[3D] ✗ Render error: {e}")
            import traceback
            traceback.print_exc()

    def set_selection(self, indices: List[int] | None = None) -> None:
        """Select specific points."""
        self._selected_indices = set(indices or [])
        self.points_selected.emit(list(self._selected_indices))
        self._render()

    def get_selection(self) -> List[int]:
        """Get currently selected point indices."""
        return list(self._selected_indices)

    def toggle_cluster_visibility(self, cluster_id: int) -> None:
        """Toggle cluster visibility."""
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

    def _set_view_preset(self, elevation: float, azimuth: float) -> None:
        """Set camera to a preset viewpoint."""
        self.view.opts['elevation'] = elevation
        self.view.opts['azimuth'] = azimuth
        self.view.update()

    def _reset_view(self) -> None:
        """Reset camera to default 3D isometric view."""
        self._set_view_preset(elevation=30, azimuth=45)

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
