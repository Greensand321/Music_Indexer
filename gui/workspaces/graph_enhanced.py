"""Enhanced Visual Music Graph workspace — embedded interactive scatter plot."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase

try:
    from gui.widgets.interactive_scatter_plot import InteractiveScatterPlot
    from gui.widgets.cluster_legend import ClusterLegendWidget
    from gui.widgets.track_details_panel import TrackDetailsPanel
except ImportError:
    InteractiveScatterPlot = None
    ClusterLegendWidget = None
    TrackDetailsPanel = None


class GraphWorkspace(WorkspaceBase):
    """Embedded interactive cluster scatter-plot visualization."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)

        self._cluster_data: Dict | None = None
        self._scatter: InteractiveScatterPlot | None = None
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
        """Build the interactive graph UI."""
        # Main layout: scatter plot + sidebar
        main_layout = QtWidgets.QHBoxLayout()

        # Left: scatter plot (main area)
        self._scatter = InteractiveScatterPlot()
        self._scatter.point_clicked.connect(self._on_point_clicked)
        self._scatter.hover_changed.connect(self._on_hover)
        main_layout.addWidget(self._scatter, 3)

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
            "To use the interactive graph, install pyqtgraph:\n"
            "pip install pyqtgraph\n\n"
            "Or run clustering to generate an HTML graph."
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: #f59e0b;")
        warning_layout.addWidget(warning_label)

        parent_layout.addWidget(warning_card)

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

        # Look for cluster data files
        cluster_info_file = lib_path / "Docs" / "cluster_info.json"
        if not cluster_info_file.exists():
            cluster_info_file = lib_path / "cluster_info.json"

        if not cluster_info_file.exists():
            self._status_lbl.setText("No cluster data found — run Clustered Playlists first")
            self._scatter.scatter.clear()
            if self._legend:
                self._legend.set_clusters(np.array([]), {})
            return

        try:
            # Load cluster info
            with open(cluster_info_file) as f:
                cluster_info = json.load(f)

            # Extract data
            X = np.array(cluster_info.get("X", []))
            labels = np.array(cluster_info.get("labels", []))
            tracks = cluster_info.get("tracks", [])
            cluster_metadata = cluster_info.get("cluster_info", {})

            if len(X) == 0:
                self._status_lbl.setText("No cluster data available")
                return

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

    @Slot(int)
    def _on_point_clicked(self, index: int) -> None:
        """Handle point click."""
        if self._cluster_data is None or index < 0 or index >= len(self._cluster_data["metadata"]):
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

        if index >= len(self._cluster_data["metadata"]):
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
        if self._scatter is None or self._cluster_data is None:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select points first")
            return

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

        # Write CSV
        try:
            with open(file_path, "w") as f:
                f.write("path\n")
                for path in selected_paths:
                    f.write(f"{path}\n")
            self._log(f"Exported {len(selected_paths)} track(s) to {file_path}", "ok")
            QtWidgets.QMessageBox.information(
                self, "Export Complete", f"Exported {len(selected_paths)} track(s)"
            )
        except Exception as e:
            self._log(f"Export failed: {e}", "error")
            QtWidgets.QMessageBox.critical(self, "Export Failed", str(e))

    @Slot()
    def _on_create_playlist(self) -> None:
        """Create M3U playlist from selection."""
        if self._scatter is None or self._cluster_data is None:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select points first")
            return

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

        # Write M3U
        try:
            playlists_dir = Path(self._library_path) / "Playlists"
            playlists_dir.mkdir(exist_ok=True)

            playlist_path = playlists_dir / f"{name}.m3u"
            with open(playlist_path, "w") as f:
                f.write("#EXTM3U\n")
                for path in selected_paths:
                    f.write(f"{path}\n")

            self._log(f"Created playlist: {playlist_path}", "ok")
            QtWidgets.QMessageBox.information(
                self, "Playlist Created",
                f"Created playlist with {len(selected_paths)} track(s)"
            )
        except Exception as e:
            self._log(f"Playlist creation failed: {e}", "error")
            QtWidgets.QMessageBox.critical(self, "Playlist Creation Failed", str(e))
