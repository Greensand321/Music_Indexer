"""Visual Music Graph — in-app 2-D cluster map, plus the 3-D browser view.

The scatter plot, legend and details panel are display-only widgets; loading
and validating ``Docs/cluster_info.json`` and turning a selection into CSV/M3U
lives in :mod:`cluster_graph_data`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase
from gui.widgets.interactive_scatter_plot import (
    MODE_LASSO,
    MODE_PAN,
    MODE_RECT,
    InteractiveScatterPlot,
)
from gui.widgets.cluster_legend import ClusterLegendWidget
from gui.widgets.track_details_panel import TrackDetailsPanel

import cluster_graph_data as cgd


class GraphWorkspace(WorkspaceBase):
    """Explore a clustered library as an interactive map."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._xy: list = []
        self._labels: list = []
        self._tracks: list = []
        self._metadata: list = []
        self._build_ui()
        self.reload_data()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Visual Music Graph"))
        cl.addWidget(self._make_subtitle(
            "Your clustered library as a map. Each dot is a track and each colour "
            "a cluster, so tracks that sound alike sit together. Hover to identify "
            "one, drag a rectangle or lasso to grab a region, then play or export "
            "the selection. Run Clustered Playlists first to produce the data."
        ))

        # ── Toolbar ────────────────────────────────────────────────────────
        toolbar = self._make_card()
        tl = QtWidgets.QVBoxLayout(toolbar)
        tl.setContentsMargins(16, 12, 16, 12)
        tl.setSpacing(8)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)

        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for mode, text, tip in (
            (MODE_PAN, "✋  Pan", "Drag to move the view"),
            (MODE_RECT, "▭  Rectangle", "Drag a box to select the points inside it"),
            (MODE_LASSO, "◌  Lasso", "Draw a freehand loop to select points"),
        ):
            btn = QtWidgets.QPushButton(text)
            btn.setCheckable(True)
            btn.setToolTip(f"{tip}  ·  hold Ctrl or Shift to add to the selection")
            btn.clicked.connect(lambda _c=False, m=mode: self._set_mode(m))
            self._mode_group.addButton(btn)
            row.addWidget(btn)
            if mode == MODE_PAN:
                btn.setChecked(True)

        row.addSpacing(12)
        self._fit_btn = QtWidgets.QPushButton("Fit view")
        self._fit_btn.clicked.connect(lambda: self._plot.fit_view())
        self._clear_btn = QtWidgets.QPushButton("Clear selection")
        self._clear_btn.clicked.connect(lambda: self._plot.clear_selection())
        self._reload_btn = QtWidgets.QPushButton("↻  Reload")
        self._reload_btn.setToolTip("Re-read cluster data from disk")
        self._reload_btn.clicked.connect(self.reload_data)
        self._browser_btn = QtWidgets.QPushButton("Open 3D view")
        self._browser_btn.setToolTip("Open the browser-based 3D graph")
        self._browser_btn.clicked.connect(self._on_open_3d)
        for b in (self._fit_btn, self._clear_btn, self._reload_btn, self._browser_btn):
            row.addWidget(b)
        row.addStretch(1)
        tl.addLayout(row)

        self._status = QtWidgets.QLabel("")
        self._status.setObjectName("statusHint")
        self._status.setWordWrap(True)
        tl.addWidget(self._status)
        cl.addWidget(toolbar)

        # ── Plot + side panels ─────────────────────────────────────────────
        graph_card = self._make_card()
        gl = QtWidgets.QVBoxLayout(graph_card)
        gl.setContentsMargins(12, 12, 12, 12)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self._plot = InteractiveScatterPlot()
        self._plot.setMinimumHeight(420)
        self._plot.point_clicked.connect(self._on_point_clicked)
        self._plot.points_selected.connect(self._on_selection_changed)
        self._plot.hover_changed.connect(self._on_hover)
        split.addWidget(self._plot)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(12)

        self._legend = ClusterLegendWidget()
        self._legend.cluster_toggled.connect(self._plot.set_cluster_visible)
        self._legend.cluster_selected.connect(self._on_cluster_selected)
        side_layout.addWidget(self._legend, 1)

        self._details = TrackDetailsPanel()
        self._details.play_requested.connect(self._on_play_one)
        self._details.reveal_requested.connect(self._on_reveal)
        side_layout.addWidget(self._details, 1)

        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        gl.addWidget(split)
        cl.addWidget(graph_card, 1)

        # ── Selection actions ──────────────────────────────────────────────
        sel_card = self._make_card()
        sl = QtWidgets.QHBoxLayout(sel_card)
        sl.setContentsMargins(16, 10, 16, 10)
        sl.setSpacing(8)

        self._sel_label = QtWidgets.QLabel("Nothing selected")
        sl.addWidget(self._sel_label)
        sl.addStretch(1)

        self._play_btn = QtWidgets.QPushButton("▶  Play selection")
        self._play_btn.clicked.connect(self._on_play_selection)
        self._csv_btn = QtWidgets.QPushButton("Export CSV…")
        self._csv_btn.clicked.connect(self._on_export_csv)
        self._m3u_btn = QtWidgets.QPushButton("Export playlist…")
        self._m3u_btn.clicked.connect(self._on_export_m3u)
        for b in (self._play_btn, self._csv_btn, self._m3u_btn):
            b.setEnabled(False)
            sl.addWidget(b)
        cl.addWidget(sel_card)

    # ── Data ──────────────────────────────────────────────────────────────

    @Slot()
    def reload_data(self) -> None:
        """Load cluster data from disk and populate the graph."""
        if not self._library_path:
            self._set_empty("No library selected.")
            return
        try:
            data = cgd.load_cluster_data(self._library_path)
            xy, labels, tracks = cgd.graph_points(data)
        except cgd.ClusterDataError as exc:
            self._set_empty(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._set_empty(f"Could not load cluster data: {exc}")
            self._log(f"Graph load failed: {exc}", "error")
            return

        self._xy, self._labels, self._tracks = xy, labels, tracks

        metadata = data.get("metadata") or []
        if len(metadata) != len(tracks):
            metadata = [{} for _ in tracks]
        else:
            metadata = [dict(m) if isinstance(m, dict) else {} for m in metadata]
        # Fall back to the file name so hover shows something useful even when
        # the payload carries no tag metadata. Selecting a point reads the real
        # tags off disk (see TrackDetailsPanel.set_track).
        for i, meta in enumerate(metadata):
            meta.setdefault("title", os.path.basename(tracks[i]))
        self._metadata = metadata

        self._plot.set_data(xy, labels, tracks, metadata)

        order = cgd.cluster_order(labels)
        self._legend.set_clusters(
            counts=cgd.cluster_counts(labels),
            order=order,
            colours={cid: self._plot.cluster_colour(cid) for cid in order},
            cluster_info=cgd.normalize_cluster_info(data.get("cluster_info")),
        )
        self._details.clear()

        summary = cgd.summarize(labels)
        if data.get("X_downsampled"):
            summary += (
                f" · showing {len(xy)} of {data.get('X_total_points', len(xy))}"
                " (downsampled for display)"
            )
        if not data.get("X_2d"):
            summary += " · using the 3D embedding's first two axes"
        self._status.setText(summary)
        self._set_controls_enabled(True)
        self._log(f"Music Graph loaded: {summary}", "ok")

    def _set_empty(self, message: str) -> None:
        self._xy, self._labels, self._tracks, self._metadata = [], [], [], []
        self._status.setText(message)
        self._set_controls_enabled(False)
        if hasattr(self, "_details"):
            self._details.clear()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for b in (self._fit_btn, self._clear_btn):
            b.setEnabled(enabled)
        for b in self._mode_group.buttons():
            b.setEnabled(enabled)
        if not enabled:
            self._sel_label.setText("Nothing selected")
            for b in (self._play_btn, self._csv_btn, self._m3u_btn):
                b.setEnabled(False)

    def _on_library_changed(self, path: str) -> None:
        self.reload_data()

    # ── Interaction ───────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        self._plot.set_mode(mode)

    @Slot(int)
    def _on_hover(self, index: int) -> None:
        if index < 0:
            return
        self._details.set_track(
            self._tracks[index],
            self._metadata_for(index),
            self._labels[index],
            load_art=False,
        )

    @Slot(int)
    def _on_point_clicked(self, index: int) -> None:
        if 0 <= index < len(self._tracks):
            self._details.set_track(
                self._tracks[index],
                self._metadata_for(index),
                self._labels[index],
                load_art=True,
            )

    def _metadata_for(self, index: int) -> dict:
        if 0 <= index < len(self._metadata):
            return self._metadata[index]
        return {}

    @Slot(int)
    def _on_cluster_selected(self, cluster_id: int) -> None:
        self._plot.select_cluster(cluster_id)

    @Slot(list)
    def _on_selection_changed(self, indices: list) -> None:
        n = len(indices)
        self._sel_label.setText(
            "Nothing selected" if n == 0
            else f"{n} track{'s' if n != 1 else ''} selected"
        )
        for b in (self._play_btn, self._csv_btn, self._m3u_btn):
            b.setEnabled(n > 0)
        if n == 1:
            self._on_point_clicked(indices[0])

    # ── Actions ───────────────────────────────────────────────────────────

    def _selection_paths(self) -> list:
        return cgd.selected_tracks(self._tracks, self._plot.get_selection())

    @Slot()
    def _on_play_selection(self) -> None:
        paths = self._selection_paths()
        if paths:
            self.play_tracks_requested.emit(paths, "Music Graph selection")

    @Slot(str)
    def _on_play_one(self, path: str) -> None:
        if path:
            self.play_tracks_requested.emit([path], Path(path).name)

    @Slot(str)
    def _on_reveal(self, path: str) -> None:
        folder = str(Path(path).parent)
        try:
            if sys.platform.startswith("win"):
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:  # noqa: BLE001
            self._log(f"Could not open {folder}: {exc}", "error")

    @Slot()
    def _on_export_csv(self) -> None:
        paths = self._selection_paths()
        if not paths:
            return
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export selection", self._docs_dir(), "CSV files (*.csv)"
        )
        if target:
            self._write(target, cgd.selection_to_csv(paths), len(paths))

    @Slot()
    def _on_export_m3u(self) -> None:
        paths = self._selection_paths()
        if not paths:
            return
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export playlist", self._playlists_dir(), "Playlists (*.m3u *.m3u8)"
        )
        if target:
            self._write(target, cgd.selection_to_m3u(paths, target), len(paths))

    def _write(self, target: str, text: str, count: int) -> None:
        try:
            Path(target).write_text(text, encoding="utf-8")
            self._log(f"Exported {count} track(s) to {target}", "ok")
            self._status.setText(f"Exported {count} track(s) to {target}")
        except OSError as exc:
            self._log(f"Export failed: {exc}", "error")

    def _docs_dir(self) -> str:
        return str(Path(self._library_path) / "Docs") if self._library_path else ""

    def _playlists_dir(self) -> str:
        return str(Path(self._library_path) / "Playlists") if self._library_path else ""

    @Slot()
    def _on_open_3d(self) -> None:
        """Open (regenerating if needed) the standalone 3-D browser view."""
        if not self._library_path:
            self._status.setText("No library selected.")
            return
        html = Path(self._library_path) / "Docs" / "cluster_graph.html"
        try:
            if not html.exists():
                from cluster_graph_3d import generate_cluster_graph_html

                generate_cluster_graph_html(self._library_path)
            if html.exists():
                webbrowser.open(html.as_uri())
                self._log(f"Opened 3D graph: {html}", "ok")
            else:
                self._status.setText("Could not generate the 3D graph.")
        except FileNotFoundError:
            self._status.setText("No cluster data found. Run Clustered Playlists first.")
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not open the 3D graph: {exc}")
            self._log(f"3D graph failed: {exc}", "error")
