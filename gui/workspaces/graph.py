"""Visual Music Graph workspace — interactive 3D scatter plot for clusters."""
from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class GraphWorkspace(WorkspaceBase):
    """Embed or launch the interactive cluster scatter-plot visualisation."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Visual Music Graph"))
        cl.addWidget(self._make_subtitle(
            "Explore your clustered library as an interactive 3D scatter plot. "
            "Each point is a track; colour represents its cluster. "
            "Hover to see metadata; click to select. "
            "Run Clustered Playlists first to generate cluster data."
        ))

        # ── Launch card ────────────────────────────────────────────────────
        launch_card = self._make_card()
        launch_layout = QtWidgets.QVBoxLayout(launch_card)
        launch_layout.setContentsMargins(16, 16, 16, 16)
        launch_layout.setSpacing(10)

        launch_layout.addWidget(QtWidgets.QLabel("Launch Graph"))

        info = QtWidgets.QLabel(
            "The 3D Visual Music Graph opens in your browser with full orbit, "
            "zoom, and pan controls. Songs that are sonically similar cluster "
            "together in space. Select points to export CSV or M3U playlists."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #64748b; font-size: 12px;")
        launch_layout.addWidget(info)

        btn_row = QtWidgets.QHBoxLayout()
        self._open_graph_btn = self._make_primary_button("Open 3D Graph")
        self._open_graph_btn.clicked.connect(self._on_open_graph)
        self._generate_btn = QtWidgets.QPushButton("Regenerate HTML")
        self._generate_btn.clicked.connect(self._on_regenerate)
        btn_row.addWidget(self._open_graph_btn)
        btn_row.addWidget(self._generate_btn)
        btn_row.addStretch(1)
        launch_layout.addLayout(btn_row)

        self._status_lbl = QtWidgets.QLabel("")
        self._status_lbl.setStyleSheet("color: #64748b; font-size: 12px;")
        launch_layout.addWidget(self._status_lbl)
        cl.addWidget(launch_card)

        # ── Cluster data summary ───────────────────────────────────────────
        self._data_card = self._make_card()
        data_layout = QtWidgets.QVBoxLayout(self._data_card)
        data_layout.setContentsMargins(16, 16, 16, 16)
        data_layout.addWidget(QtWidgets.QLabel("Cluster Data Status"))

        self._data_status = QtWidgets.QLabel("No cluster data found — run Clustered Playlists first.")
        self._data_status.setStyleSheet("color: #f59e0b;")
        self._data_status.setWordWrap(True)
        data_layout.addWidget(self._data_status)

        self._refresh_btn = QtWidgets.QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._check_data)
        data_layout.addWidget(self._refresh_btn)
        cl.addWidget(self._data_card)

        cl.addStretch(1)

        # Initial data check
        self._check_data()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _cluster_info_path(self) -> Path | None:
        if not self._library_path:
            return None
        return Path(self._library_path) / "Docs" / "cluster_info.json"

    def _cluster_html_path(self) -> Path | None:
        if not self._library_path:
            return None
        return Path(self._library_path) / "Docs" / "cluster_graph.html"

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_open_graph(self) -> None:
        html = self._cluster_html_path()
        if html and html.exists():
            webbrowser.open(html.as_uri())
            return

        # Try generating on the fly if cluster_info.json exists
        info_path = self._cluster_info_path()
        if info_path and info_path.exists():
            self._do_generate()
            html = self._cluster_html_path()
            if html and html.exists():
                webbrowser.open(html.as_uri())
                return

        self._status_lbl.setText("No cluster data found. Run Clustered Playlists first.")
        self._status_lbl.setStyleSheet("color: #f59e0b; font-size: 12px;")

    @Slot()
    def _on_regenerate(self) -> None:
        if self._do_generate():
            self._status_lbl.setText("3D graph HTML regenerated successfully.")
            self._status_lbl.setStyleSheet("color: #22c55e; font-size: 12px;")
        else:
            self._status_lbl.setText("Cannot regenerate — no cluster_info.json found.")
            self._status_lbl.setStyleSheet("color: #f59e0b; font-size: 12px;")

    def _do_generate(self) -> bool:
        """Regenerate HTML from cluster_info.json.  Returns True on success."""
        if not self._library_path:
            return False
        try:
            from cluster_graph_3d import generate_cluster_graph_html
            generate_cluster_graph_html(self._library_path)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            self._status_lbl.setText(f"Error generating graph: {exc}")
            self._status_lbl.setStyleSheet("color: #ef4444; font-size: 12px;")
            return False

    @Slot()
    def _check_data(self) -> None:
        if not self._library_path:
            self._data_status.setText("No library selected.")
            self._data_status.setStyleSheet("color: #f59e0b;")
            return

        info_path = self._cluster_info_path()
        html_path = self._cluster_html_path()

        if info_path and info_path.exists():
            try:
                with open(info_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                n_tracks = len(data.get("tracks", []))
                n_clusters = len(set(l for l in data.get("labels", []) if l >= 0))
                has_3d = bool(data.get("X_3d"))
                html_exists = html_path and html_path.exists()

                parts = [f"{n_tracks} tracks", f"{n_clusters} clusters"]
                if has_3d:
                    parts.append("3D embeddings available")
                if html_exists:
                    parts.append("HTML ready")
                else:
                    parts.append("HTML not yet generated")

                self._data_status.setText(" · ".join(parts))
                self._data_status.setStyleSheet("color: #22c55e;")
            except (json.JSONDecodeError, OSError) as exc:
                self._data_status.setText(f"Error reading cluster data: {exc}")
                self._data_status.setStyleSheet("color: #ef4444;")
        else:
            self._data_status.setText("No cluster data found — run Clustered Playlists first.")
            self._data_status.setStyleSheet("color: #f59e0b;")

    def _on_library_changed(self, path: str) -> None:
        self._check_data()
