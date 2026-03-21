"""Enhanced Visual Music Graph workspace — embedded interactive scatter plot."""
from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase

logger = logging.getLogger(__name__)

try:
    from gui.widgets.interactive_scatter_plot import InteractiveScatterPlot
    from gui.widgets.interactive_3d_scatter import Interactive3DScatterPlot
    from gui.widgets.cluster_legend import ClusterLegendWidget
    from gui.widgets.track_details_panel import TrackDetailsPanel
except ImportError:
    InteractiveScatterPlot = None
    Interactive3DScatterPlot = None
    ClusterLegendWidget = None
    TrackDetailsPanel = None

try:
    from gui.widgets.three_js_viewer import ThreeJsViewer
except ImportError:
    ThreeJsViewer = None


class GraphWorkspace(WorkspaceBase):
    """Embedded interactive cluster scatter-plot visualization."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)

        self._cluster_data: Dict | None = None
        self._scatter: InteractiveScatterPlot | None = None
        self._scatter_3d: Interactive3DScatterPlot | None = None
        self._scatter_3d_threejs: ThreeJsViewer | None = None
        self._use_3d: bool = False  # Track which view is active
        self._current_view: str = "2D View"  # "2D View", "3D View" (PyQtGraph), or "3D View (Three.js)"
        self._legend: ClusterLegendWidget | None = None
        self._details: TrackDetailsPanel | None = None

        self._build_ui()
        self._library_changed(library_path)

    def _build_ui(self) -> None:
        """Build the UI."""
        cl = self.content_layout

        # Title
        cl.addWidget(self._make_section_title("Visual Music Graph"))
        cl.addWidget(self._make_subtitle(
            "Explore your clustered library as an interactive scatter plot. "
            "Hover points to see metadata, select clusters to export playlists. "
            "Run Clustered Playlists first to generate cluster data."
        ))

        # Status and controls
        control_card = self._make_card()
        control_layout = QtWidgets.QHBoxLayout(control_card)

        self._status_lbl = QtWidgets.QLabel("Loading...")
        self._status_lbl.setStyleSheet("color: #666; font-size: 11px;")
        control_layout.addWidget(self._status_lbl)

        control_layout.addStretch(1)

        # View toggle (2D/3D/Three.js) - default to Three.js for immersive exploration
        view_toggle = QtWidgets.QComboBox()
        view_toggle.addItems([
            "2D View",
            "3D View (PyQtGraph)",
            "3D View (Three.js)"
        ])
        view_toggle.setCurrentText("3D View (Three.js)")  # Default to Three.js!
        view_toggle.currentTextChanged.connect(self._on_view_changed)
        control_layout.addWidget(QtWidgets.QLabel("View:"))
        control_layout.addWidget(view_toggle)

        # Axis control for 3D (X, Y, Z feature selection)
        self._axis_x_combo = QtWidgets.QComboBox()
        self._axis_y_combo = QtWidgets.QComboBox()
        self._axis_z_combo = QtWidgets.QComboBox()

        # Populate with default feature names
        axes_options = ["Energy", "Timbre", "Complexity", "MFCC-Mean", "MFCC-Std", "Tempo"]
        for combo in [self._axis_x_combo, self._axis_y_combo, self._axis_z_combo]:
            combo.addItems(axes_options)

        # Set defaults
        self._axis_x_combo.setCurrentText("Energy")
        self._axis_y_combo.setCurrentText("Timbre")
        self._axis_z_combo.setCurrentText("Complexity")

        # Connect to re-projection
        for combo in [self._axis_x_combo, self._axis_y_combo, self._axis_z_combo]:
            combo.currentTextChanged.connect(self._on_axes_changed)

        control_layout.addWidget(QtWidgets.QLabel("3D Axes:"))
        control_layout.addWidget(self._axis_x_combo)
        control_layout.addWidget(self._axis_y_combo)
        control_layout.addWidget(self._axis_z_combo)

        refresh_btn = QtWidgets.QPushButton("↺ Refresh")
        refresh_btn.clicked.connect(self._on_refresh)
        control_layout.addWidget(refresh_btn)

        cl.addWidget(control_card)

        # Main graph area (only if PyQtGraph available)
        if InteractiveScatterPlot is not None:
            self._build_graph_ui(cl)
        else:
            self._build_fallback_ui(cl)

        cl.addStretch(1)

    def _build_graph_ui(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build the interactive graph UI with 2D/3D support."""
        # Main layout: scatter plot + sidebar
        main_layout = QtWidgets.QHBoxLayout()

        # Create stacked widget for 2D/3D/Three.js switching
        self._graph_stack = QtWidgets.QStackedWidget()

        # Index 0: 2D scatter plot
        self._scatter = InteractiveScatterPlot()
        self._scatter.point_clicked.connect(self._on_point_clicked)
        self._scatter.hover_changed.connect(self._on_hover)
        self._graph_stack.addWidget(self._scatter)

        # Index 1: 3D scatter plot (PyQtGraph, if available)
        if Interactive3DScatterPlot is not None:
            try:
                self._scatter_3d = Interactive3DScatterPlot()
                self._scatter_3d.point_clicked.connect(self._on_point_clicked)
                self._scatter_3d.hover_changed.connect(self._on_hover)
                self._graph_stack.addWidget(self._scatter_3d)
            except ImportError:
                self._scatter_3d = None
                self._log("⚠ 3D scatter plot unavailable (PyQtGraph OpenGL required)", "warning")
        else:
            # Add placeholder for 3D
            self._scatter_3d = None
            placeholder = QtWidgets.QLabel("3D View not available")
            self._graph_stack.addWidget(placeholder)

        # Index 2: 3D scatter plot (Three.js WebView)
        if ThreeJsViewer is not None:
            try:
                self._scatter_3d_threejs = ThreeJsViewer()
                self._scatter_3d_threejs.points_selected.connect(self._on_threejs_selection_changed)
                self._graph_stack.addWidget(self._scatter_3d_threejs)
                # Set Three.js as default view (index 2)
                self._graph_stack.setCurrentIndex(2)
                self._use_3d = True
                self._current_view = "3D View (Three.js)"
                self._log("✓ Three.js 3D viewer loaded", "ok")
            except Exception as e:
                self._scatter_3d_threejs = None
                self._log(f"⚠ Three.js viewer unavailable: {e}", "warning")
        else:
            self._scatter_3d_threejs = None
            self._log("⚠ Three.js viewer not available (WebEngine or WebChannel missing)", "warning")

        main_layout.addWidget(self._graph_stack, 3)

        # Right sidebar: legend + details
        sidebar_layout = QtWidgets.QVBoxLayout()
        sidebar_layout.setSpacing(8)

        # Cluster legend
        self._legend = ClusterLegendWidget()
        self._legend.cluster_toggled.connect(self._on_cluster_toggled)
        self._legend.cluster_selected.connect(self._on_cluster_selected)
        sidebar_layout.addWidget(self._legend, 1)

        # Track details
        self._details = TrackDetailsPanel()
        sidebar_layout.addWidget(self._details, 1)

        main_layout.addLayout(sidebar_layout, 1)
        parent_layout.addLayout(main_layout, 1)

        # Bottom: selection actions
        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setSpacing(6)

        export_btn = QtWidgets.QPushButton("📄 Export Selection")
        export_btn.clicked.connect(self._on_export_selection)
        action_layout.addWidget(export_btn)

        playlist_btn = QtWidgets.QPushButton("🎵 Create Playlist from Selection")
        playlist_btn.clicked.connect(self._on_create_playlist)
        action_layout.addWidget(playlist_btn)

        fit_btn = QtWidgets.QPushButton("⊡ Fit View")
        fit_btn.clicked.connect(self._on_fit_view)
        action_layout.addWidget(fit_btn)

        action_layout.addStretch(1)
        parent_layout.addLayout(action_layout)

    def _build_fallback_ui(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build fallback UI when PyQtGraph not available."""
        warning_card = self._make_card()
        warning_layout = QtWidgets.QVBoxLayout(warning_card)

        warning_label = QtWidgets.QLabel(
            "⚠️ PyQtGraph not installed\n\n"
            "The interactive scatter plot requires PyQtGraph.\n"
            "However, cluster data is still being generated by the\n"
            "Clustered Playlists workspace.\n\n"
            "To enable the interactive graph:\n"
            "   1. Install PyQtGraph: pip install pyqtgraph\n"
            "   2. Restart the application\n\n"
            "You can still:\n"
            "   • Export cluster data as CSV or M3U playlists\n"
            "   • View cluster quality reports\n"
            "   • Access cluster info from the Clustered Playlists workspace"
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: #f59e0b; font-size: 12px;")
        warning_layout.addWidget(warning_label)

        # Add a button to copy the install command
        button_layout = QtWidgets.QHBoxLayout()
        copy_btn = QtWidgets.QPushButton("📋 Copy Install Command")
        copy_btn.clicked.connect(self._copy_install_command)
        button_layout.addWidget(copy_btn)
        button_layout.addStretch()
        warning_layout.addLayout(button_layout)

        parent_layout.addWidget(warning_card)

    def _copy_install_command(self) -> None:
        """Copy installation command to clipboard."""
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText("pip install pyqtgraph")
        QtWidgets.QMessageBox.information(
            self,
            "Command Copied",
            "Installation command copied to clipboard:\npip install pyqtgraph"
        )

    def _library_changed(self, path: str) -> None:
        """Called when library path changes."""
        self._library_path = path
        self._load_cluster_data()

    def _load_cluster_data(self) -> None:
        """Load cluster data from library."""
        if not self._library_path or self._scatter is None:
            self._status_lbl.setText("No library selected")
            return

        lib_path = Path(self._library_path)

        # Validate library path exists and is accessible
        if not lib_path.exists():
            self._status_lbl.setText("Library path does not exist")
            self._status_lbl.setStyleSheet("color: #ef4444;")
            logger.error(f"Library path not found: {lib_path}")
            return

        if not lib_path.is_dir():
            self._status_lbl.setText("Library path is not a directory")
            self._status_lbl.setStyleSheet("color: #ef4444;")
            logger.error(f"Library path is not a directory: {lib_path}")
            return

        # Check read permissions
        try:
            list(lib_path.iterdir())
        except PermissionError:
            self._status_lbl.setText("No permission to read library folder")
            self._status_lbl.setStyleSheet("color: #ef4444;")
            logger.error(f"Permission denied reading library: {lib_path}")
            return
        except OSError as e:
            self._status_lbl.setText(f"Cannot access library: {e}")
            self._status_lbl.setStyleSheet("color: #ef4444;")
            logger.error(f"Error accessing library: {e}")
            return

        # Look for cluster data files
        cluster_info_file = lib_path / "Docs" / "cluster_info.json"
        if not cluster_info_file.exists():
            cluster_info_file = lib_path / "cluster_info.json"

        if not cluster_info_file.exists():
            self._status_lbl.setText("No cluster data found — run Clustered Playlists first")
            if self._scatter and hasattr(self._scatter, 'scatter'):
                self._scatter.scatter.clear()
            if self._legend:
                self._legend.set_clusters(np.array([]), {})
            return

        try:
            # Load cluster info with error handling
            try:
                with open(cluster_info_file) as f:
                    cluster_info = json.load(f)
            except json.JSONDecodeError as e:
                self._status_lbl.setText(f"Corrupted cluster data file: {e}")
                self._status_lbl.setStyleSheet("color: #ef4444;")
                logger.error(f"Failed to parse JSON from {cluster_info_file}: {e}")
                return
            except OSError as e:
                self._status_lbl.setText(f"Cannot read cluster file: {e}")
                self._status_lbl.setStyleSheet("color: #ef4444;")
                logger.error(f"Failed to read {cluster_info_file}: {e}")
                return

            # Validate required keys exist
            required_keys = ["X", "labels", "tracks"]
            missing_keys = [k for k in required_keys if k not in cluster_info]
            if missing_keys:
                self._status_lbl.setText(f"Invalid cluster data: missing {', '.join(missing_keys)}")
                self._status_lbl.setStyleSheet("color: #ef4444;")
                logger.error(f"Missing required keys in cluster data: {missing_keys}")
                return

            # Extract data with type validation
            try:
                # Use 2D embedding if available, otherwise fall back to high-dimensional X
                if "X_2d" in cluster_info:
                    X = np.array(cluster_info["X_2d"], dtype=np.float32)
                    self._log("✓ Using 2D embedding for visualization", "info")
                else:
                    X = np.array(cluster_info["X"], dtype=np.float32)
                    # If X is high-dimensional, we need to compute 2D embedding
                    if X.ndim > 1 and X.shape[1] > 2:
                        self._log("⚠ Computing 2D visualization from high-dimensional features…", "warning")
                        try:
                            from clustered_playlists import compute_2d_embedding
                            X = compute_2d_embedding(X, self._log)
                        except Exception as e:
                            self._log(f"⚠ Could not compute 2D embedding: {e}", "warning")
                            # Use first 2 dimensions as fallback
                            X = X[:, :2]

                # Load 3D embedding if available
                X_3d = None
                if "X_3d" in cluster_info:
                    try:
                        X_3d = np.array(cluster_info["X_3d"], dtype=np.float32)
                        self._log("✓ Loaded 3D embedding (preserves ~80% variance)", "info")
                    except Exception as e:
                        self._log(f"⚠ Could not load 3D embedding: {e}", "warning")
                else:
                    # Compute 3D from raw features if needed
                    X_raw = np.array(cluster_info["X"], dtype=np.float32)
                    if X_raw.ndim > 1 and X_raw.shape[1] > 3:
                        self._log("→ Computing 3D visualization from high-dimensional features…", "info")
                        try:
                            from clustered_playlists import compute_3d_embedding
                            X_3d = compute_3d_embedding(X_raw, self._log)
                        except Exception as e:
                            self._log(f"⚠ Could not compute 3D embedding: {e}", "warning")

                labels = np.array(cluster_info["labels"], dtype=np.int32)
                tracks = cluster_info.get("tracks", [])
                cluster_metadata = cluster_info.get("cluster_info", {})
            except (TypeError, ValueError) as e:
                self._status_lbl.setText(f"Invalid cluster data types: {e}")
                self._status_lbl.setStyleSheet("color: #ef4444;")
                logger.error(f"Failed to parse cluster data types: {e}")
                return

            # Validate array lengths match
            if len(X) != len(labels) or len(labels) != len(tracks):
                self._status_lbl.setText(
                    f"Mismatched cluster data lengths: X={len(X)}, labels={len(labels)}, tracks={len(tracks)}"
                )
                self._status_lbl.setStyleSheet("color: #ef4444;")
                logger.error(f"Array length mismatch in cluster data")
                return

            if len(X) == 0:
                self._status_lbl.setText("No cluster data available")
                return

            # Check if X was downsampled for visualization
            x_downsampled = cluster_info.get("X_downsampled", False)
            x_total = cluster_info.get("X_total_points", len(X))
            if x_downsampled and x_total > 0:
                logger.info(f"Cluster data: X downsampled from {x_total} to {len(X)} points for visualization")

            # Create metadata dict for each point
            metadata = []
            for track_path in tracks:
                track_name = Path(track_path).name
                metadata.append({
                    "path": track_path,
                    "title": track_name,
                    "artist": "Unknown",
                    "album": "Unknown",
                    "genres": [],
                    "bpm": 0,
                    "duration": 0,
                })

            # Generate colors by cluster
            unique_clusters = np.unique(labels)
            colors = []
            for label in labels:
                # Simple color generation based on cluster
                hue = (int(label) % len(unique_clusters)) / max(len(unique_clusters), 1)
                color = (
                    int(100 + 155 * (1 - hue)),  # R
                    int(100 + 155 * hue),        # G
                    int(200),                     # B
                )
                colors.append(color)

            colors = np.array(colors)

            # Set scatter plot data
            self._scatter.set_data(
                X=X,
                clusters=labels,
                labels=tracks,
                metadata=metadata,
            )

            # Set legend
            if self._legend:
                self._legend.set_clusters(labels, cluster_metadata, colors)

            # Update status
            n_clusters = len(unique_clusters)
            n_tracks = len(labels)
            self._status_lbl.setText(
                f"✓ Loaded: {n_clusters} cluster(s), {n_tracks} track(s)"
            )
            self._status_lbl.setStyleSheet("color: #22c55e;")

            self._cluster_data = {
                "X": X,
                "X_3d": X_3d,  # 3D embedding for advanced exploration
                "labels": labels,
                "tracks": tracks,
                "metadata": metadata,
                "cluster_info": cluster_metadata,
            }

        except Exception as e:
            self._status_lbl.setText(f"Error loading cluster data: {e}")
            self._status_lbl.setStyleSheet("color: #ef4444;")
            self._log(f"Error loading cluster data: {e}", "error")

    @Slot()
    def _on_refresh(self) -> None:
        """Refresh cluster data."""
        self._load_cluster_data()
        self._log("Refreshed cluster data", "info")

    @Slot(str)
    def _on_axes_changed(self, axis_name: str) -> None:
        """Handle 3D axis selection changes."""
        if self._use_3d and self._scatter_3d is not None:
            self._log(f"3D axes updated: X={self._axis_x_combo.currentText()}, Y={self._axis_y_combo.currentText()}, Z={self._axis_z_combo.currentText()}", "info")
            # TODO: Implement custom axis mapping based on feature selection

    @Slot(str)
    def _on_view_changed(self, view_name: str) -> None:
        """Switch between 2D, 3D (PyQtGraph), and 3D (Three.js) views."""
        self._current_view = view_name

        if view_name == "3D View (PyQtGraph)":
            if self._scatter_3d is None:
                self._log("⚠ PyQtGraph 3D view not available", "warning")
                return

            # Show PyQtGraph 3D widget
            self._graph_stack.setCurrentWidget(self._scatter_3d)
            self._use_3d = True
            self._log("Switched to 3D view (PyQtGraph) — rotate with mouse, scroll to zoom", "info")

            # Load data into 3D view
            if self._cluster_data is not None:
                try:
                    # Get 3D data - might be None if computation failed
                    X_3d = self._cluster_data.get("X_3d")

                    if X_3d is None:
                        self._log("⚠ 3D data not available, computing from features…", "warning")
                        # Try to compute on-the-fly
                        X = np.array(self._cluster_data.get("X", []), dtype=np.float32)
                        if len(X) > 0 and X.ndim == 2 and X.shape[1] > 3:
                            from clustered_playlists import compute_3d_embedding
                            X_3d = compute_3d_embedding(X, self._log)
                            self._cluster_data["X_3d"] = X_3d
                        else:
                            self._log("✗ Cannot compute 3D: invalid feature data", "error")
                            return

                    # Ensure it's a numpy array
                    X_3d = np.array(X_3d, dtype=np.float32)
                    if X_3d.shape[1] != 3:
                        self._log(f"✗ Invalid 3D shape: {X_3d.shape}, expected (N, 3)", "error")
                        return

                    labels = np.array(self._cluster_data["labels"], dtype=np.int32)
                    tracks = self._cluster_data.get("tracks", [])
                    metadata = self._cluster_data.get("metadata", [])

                    self._log(f"Loading {len(X_3d)} points into PyQtGraph 3D view…", "info")
                    self._scatter_3d.set_data(X_3d, labels, tracks, metadata)
                    self._log(f"✓ PyQtGraph 3D view loaded successfully", "info")
                except Exception as e:
                    self._log(f"✗ Failed to load 3D data: {e}", "error")
                    import traceback
                    logger.exception("3D view load error")
            else:
                self._log("No cluster data available. Run Clustered Playlists first.", "warning")

        elif view_name == "3D View (Three.js)":
            if self._scatter_3d_threejs is None:
                self._log("⚠ Three.js 3D view not available", "warning")
                return

            # Show Three.js 3D widget
            self._graph_stack.setCurrentWidget(self._scatter_3d_threejs)
            self._use_3d = True
            self._log("Switched to 3D view (Three.js) — drag to rotate, scroll to zoom, click to select", "info")

            # Load data into Three.js viewer
            if self._cluster_data is not None:
                try:
                    X_3d = self._cluster_data.get("X_3d")

                    if X_3d is None:
                        self._log("⚠ 3D data not available, computing from features…", "warning")
                        X = np.array(self._cluster_data.get("X", []), dtype=np.float32)
                        if len(X) > 0 and X.ndim == 2 and X.shape[1] > 3:
                            from clustered_playlists import compute_3d_embedding
                            X_3d = compute_3d_embedding(X, self._log)
                            self._cluster_data["X_3d"] = X_3d
                        else:
                            self._log("✗ Cannot compute 3D: invalid feature data", "error")
                            return

                    X_3d = np.array(X_3d, dtype=np.float32)
                    if X_3d.shape[1] != 3:
                        self._log(f"✗ Invalid 3D shape: {X_3d.shape}, expected (N, 3)", "error")
                        return

                    labels = np.array(self._cluster_data["labels"], dtype=np.int32)
                    tracks = self._cluster_data.get("tracks", [])
                    metadata = self._cluster_data.get("metadata", [])

                    self._log(f"Loading {len(X_3d)} points into Three.js viewer…", "info")
                    self._scatter_3d_threejs.set_data(X_3d, labels, tracks, metadata)
                    self._log(f"✓ Three.js viewer loaded successfully", "info")
                except Exception as e:
                    self._log(f"✗ Failed to load Three.js viewer: {e}", "error")
                    import traceback
                    logger.exception("Three.js view load error")
            else:
                self._log("No cluster data available. Run Clustered Playlists first.", "warning")

        elif view_name == "2D View":
            # Show 2D widget
            self._graph_stack.setCurrentWidget(self._scatter)
            self._use_3d = False
            self._log("Switched to 2D view", "info")

    @Slot(list)
    def _on_threejs_selection_changed(self, indices: List[int]) -> None:
        """Handle Three.js selection change."""
        if self._cluster_data is None:
            return

        # Update first selected point details
        if indices:
            index = indices[0]
            self._on_point_clicked(index)

        self._log(f"Selected {len(indices)} point(s) in Three.js viewer", "info")

    @Slot(int)
    def _on_point_clicked(self, index: int) -> None:
        """Handle point click."""
        if self._cluster_data is None or index < 0:
            return

        # Validate index is within all arrays
        metadata_len = len(self._cluster_data.get("metadata", []))
        tracks_len = len(self._cluster_data.get("tracks", []))

        if index >= metadata_len or index >= tracks_len:
            logger.warning(f"Point index {index} out of bounds (metadata={metadata_len}, tracks={tracks_len})")
            return

        metadata = self._cluster_data["metadata"][index]
        if self._details:
            self._details.set_track(metadata)

        self._log(f"Selected: {metadata.get('title', 'Unknown')}", "info")

    @Slot(int)
    def _on_hover(self, index: int) -> None:
        """Handle point hover."""
        if index < 0 or self._cluster_data is None:
            if self._details:
                self._details._clear_display()
            return

        # Validate index is within all arrays
        metadata_len = len(self._cluster_data.get("metadata", []))
        tracks_len = len(self._cluster_data.get("tracks", []))

        if index >= metadata_len or index >= tracks_len:
            if self._details:
                self._details._clear_display()
            return

        metadata = self._cluster_data["metadata"][index]
        if self._details:
            self._details.set_track(metadata)

    @Slot(int, bool)
    def _on_cluster_toggled(self, cluster_id: int, visible: bool) -> None:
        """Handle cluster visibility toggle."""
        if self._scatter:
            self._scatter.set_cluster_visible(cluster_id, visible)
        self._log(f"Cluster {cluster_id}: {'shown' if visible else 'hidden'}", "info")

    @Slot(int)
    def _on_cluster_selected(self, cluster_id: int) -> None:
        """Handle cluster selection."""
        if self._scatter:
            self._scatter.highlight_cluster(cluster_id)
        self._log(f"Highlighted cluster {cluster_id}", "info")

    @Slot()
    def _on_fit_view(self) -> None:
        """Fit view to show all points."""
        if self._scatter:
            self._scatter.fit_view()

    @Slot()
    def _on_export_selection(self) -> None:
        """Export selected points as CSV."""
        if self._cluster_data is None:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select points first")
            return

        # Get selection from active viewer
        selected_paths = []
        if self._current_view == "3D View (Three.js)" and self._scatter_3d_threejs:
            selected_paths = self._scatter_3d_threejs.export_selection_paths()
        elif self._scatter:
            selected_paths = self._scatter.export_selection_paths()

        if not selected_paths:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select points first")
            return

        # Show save dialog
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Selection",
            f"{self._library_path}/selection.csv",
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        # Write CSV with atomic write (temp file then move)
        try:
            import tempfile
            import csv
            import os
            from pathlib import Path as PathlibPath

            # Create temp file in same directory for atomic write
            temp_fd, temp_path = tempfile.mkstemp(dir=PathlibPath(file_path).parent, text=True)

            try:
                # Close the file descriptor immediately - we'll use open() instead
                os.close(temp_fd)

                with open(temp_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["path"])
                    for path in selected_paths:
                        writer.writerow([path])

                # Atomic rename
                PathlibPath(temp_path).replace(file_path)

                self._log(f"Exported {len(selected_paths)} track(s) to {file_path}", "ok")
                QtWidgets.QMessageBox.information(
                    self, "Export Complete", f"Exported {len(selected_paths)} track(s)"
                )
            except Exception as write_err:
                # Clean up temp file
                PathlibPath(temp_path).unlink(missing_ok=True)
                raise write_err

        except OSError as e:
            self._log(f"Export failed (disk error): {e}", "error")
            QtWidgets.QMessageBox.critical(self, "Export Failed", f"Disk error: {e}")
        except Exception as e:
            self._log(f"Export failed: {e}", "error")
            logger.exception("Export selection failed")
            QtWidgets.QMessageBox.critical(self, "Export Failed", str(e))

    @Slot()
    def _on_create_playlist(self) -> None:
        """Create M3U playlist from selection."""
        if self._cluster_data is None:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select points first")
            return

        # Get selection from active viewer
        selected_paths = []
        if self._current_view == "3D View (Three.js)" and self._scatter_3d_threejs:
            selected_paths = self._scatter_3d_threejs.export_selection_paths()
        elif self._scatter:
            selected_paths = self._scatter.export_selection_paths()

        if not selected_paths:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select points first")
            return

        # Ask for playlist name
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Create Playlist",
            "Playlist name:",
            text="Selected Tracks"
        )

        if not ok or not name:
            return

        # Write M3U with proper error handling
        try:
            import tempfile
            import os

            playlists_dir = Path(self._library_path) / "Playlists"
            try:
                playlists_dir.mkdir(exist_ok=True)
            except OSError as e:
                self._log(f"Playlist creation failed (cannot create folder): {e}", "error")
                QtWidgets.QMessageBox.critical(self, "Playlist Creation Failed", f"Cannot create Playlists folder: {e}")
                return

            playlist_path = playlists_dir / f"{name}.m3u"

            # Write to temp file first
            temp_fd, temp_path = tempfile.mkstemp(dir=playlists_dir, text=True, suffix='.m3u')

            try:
                # Close the file descriptor immediately - we'll use open() instead
                os.close(temp_fd)

                with open(temp_path, 'w') as f:
                    f.write("#EXTM3U\n")
                    for path in selected_paths:
                        f.write(f"{path}\n")

                # Atomic rename
                Path(temp_path).replace(playlist_path)

                self._log(f"Created playlist: {playlist_path}", "ok")
                QtWidgets.QMessageBox.information(
                    self, "Playlist Created",
                    f"Created playlist with {len(selected_paths)} track(s)"
                )
            except Exception as write_err:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
                raise write_err

        except OSError as e:
            self._log(f"Playlist creation failed (disk error): {e}", "error")
            QtWidgets.QMessageBox.critical(self, "Playlist Creation Failed", f"Disk error: {e}")
        except Exception as e:
            self._log(f"Playlist creation failed: {e}", "error")
            logger.exception("Playlist creation failed")
            QtWidgets.QMessageBox.critical(self, "Playlist Creation Failed", str(e))
