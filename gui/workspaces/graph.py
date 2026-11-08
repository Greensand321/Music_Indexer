"""Sound Map → Map: the in-app 2-D cluster map, plus the 3-D browser view.

The scatter plot, legend and details panel are display-only widgets; loading
and validating ``Docs/cluster_info.json`` and turning a selection into CSV
lives in :mod:`cluster_graph_data`. A selection exported as a playlist goes
through the shared preview-first writer (``playlist_generator``), so a
replaced file is backed up and logged.

This panel also owns the Sound Map's :class:`ClusterPlaylistFlow` ("Save
cluster playlists…", docs/specs/cluster_playlists.md): it tells the flow which
clusters are hidden, which run the map shows and each cluster's colour, and
refreshes it on every (re)load and library change. ``SoundMapPage`` places
the flow's button among its page actions.
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.cluster_playlist_jobs import ClusterPlaylistFlow
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


def _page_wording(message: str) -> str:
    """The data layer still names the old "Clustered Playlists" page; on the
    Sound Map that step is the Analyse tab."""
    return (message.replace("Re-run Clustered Playlists", "Run the analysis again (Analyse tab)")
                   .replace("Run Clustered Playlists", "Run an analysis (Analyse tab)"))


class GraphWorkspace(WorkspaceBase):
    """Explore a clustered library as an interactive map."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._xy: list = []
        self._labels: list = []
        self._tracks: list = []
        self._metadata: list = []
        # The track pinned by a click: the details panel (and its Play button)
        # stays on it while the mouse wanders over other points. -1 = none.
        self._pinned: int = -1
        # (mtime_ns, size) of the cluster_info.json last loaded, so the page
        # can reload by itself when a clustering run has replaced it.
        self._loaded_stamp: tuple | None = None
        # run_id of the payload on the map: None = nothing loaded, "" = a
        # payload written before runs had ids.
        self._map_run: str | None = None
        self._build_ui()
        self._playlists_flow = ClusterPlaylistFlow(
            self,
            hidden_clusters=self._hidden_clusters,
            map_run_id=self.map_run_id,
            cluster_colour=self._cluster_colour_for,
        )
        self._playlists_flow.result_ready.connect(self._on_playlists_result)
        self.reload_data()

    @property
    def cluster_playlists_flow(self) -> ClusterPlaylistFlow:
        """The "Save cluster playlists…" flow (its button is placed by the page)."""
        return self._playlists_flow

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Explore the map"))
        cl.addWidget(self._make_subtitle(
            "Your clustered library as a map. Each dot is a track and each colour "
            "a cluster, so tracks that sound alike sit together. Hover to identify "
            "one, drag a rectangle or lasso to grab a region, then play or export "
            "the selection; Save cluster playlists… saves every cluster at once. "
            "Run an analysis first (the Analyse tab) to produce the data."
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
        # Room for several rows beside the details panel (it was squeezed to ~2).
        self._legend.setMinimumHeight(200)
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

    def _data_stamp(self) -> tuple | None:
        if not self._library_path:
            return None
        try:
            st = cgd.cluster_info_path(self._library_path).stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # noqa: N802
        """Pick up a clustering run that finished since the last load.

        The Clustered page hands its result over only through
        ``Docs/cluster_info.json``; without this the graph kept saying "No
        cluster data" until the user pressed Reload.
        """
        super().showEvent(event)
        if self._library_path and self._data_stamp() != self._loaded_stamp:
            self.reload_data()

    @Slot()
    def reload_data(self) -> None:
        """Load cluster data from disk and populate the graph."""
        try:
            self._reload_data()
        finally:
            # Contract item 2: availability follows every (failed) load.
            self._playlists_flow.refresh()

    def _reload_data(self) -> None:
        self._loaded_stamp = self._data_stamp()
        if not self._library_path:
            self._set_empty("No library selected.")
            return
        if not self._plot.available:
            self._set_empty("The interactive graph is unavailable (see the notice above).")
            return
        try:
            data = cgd.load_cluster_data(self._library_path)
            xy, labels, tracks = cgd.graph_points(data)
        except cgd.ClusterDataError as exc:
            self._set_empty(_page_wording(str(exc)))
            return
        except Exception as exc:  # noqa: BLE001
            self._set_empty(f"Could not load cluster data: {exc}")
            self._log(f"Graph load failed: {exc}", "error")
            return

        self._xy, self._labels, self._tracks = xy, labels, tracks
        run = data.get("run") if isinstance(data.get("run"), dict) else {}
        self._map_run = str(run.get("run_id") or "")

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
        self._pinned = -1

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
        self._log(f"Sound Map loaded: {summary}", "ok")

    def _set_empty(self, message: str) -> None:
        self._xy, self._labels, self._tracks, self._metadata = [], [], [], []
        self._map_run = None
        self._pinned = -1
        self._status.setText(message)
        self._set_controls_enabled(False)
        # Clear what was drawn too: a failed reload used to leave the previous
        # map and a clickable legend on screen under an error message.
        self._plot.clear_data()
        self._legend.clear()
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
        # Hidden clusters belong to the previous library's map.
        self._plot.reset_visibility()
        self._legend.reset_visibility()
        # Drops a running cluster-playlist preview; a running write finishes
        # in the library it was confirmed for.
        self._playlists_flow.on_library_changed()
        self.reload_data()

    # ── What the cluster playlist flow asks the map (contract item 4) ─────

    def map_run_id(self) -> str | None:
        """run_id of the analysis on the map (None: nothing loaded; "": a
        payload from before runs had ids)."""
        return self._map_run

    def _hidden_clusters(self) -> set:
        """Clusters the user hid in the legend (only ones the map has)."""
        shown = set(self._legend.visible_clusters())
        return {cid for cid in cgd.cluster_order(self._labels) if cid not in shown}

    def _cluster_colour_for(self, cluster_id: int):
        """The map's colour for ``cluster_id`` (None if the map lacks it)."""
        if not self._plot.available or cluster_id not in set(self._labels):
            return None
        return self._plot.cluster_colour(cluster_id)

    @Slot(str, bool)
    def _on_playlists_result(self, text: str, _ok: bool) -> None:
        self._status.setText(text)

    # ── Interaction ───────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        self._plot.set_mode(mode)

    @Slot(int)
    def _on_hover(self, index: int) -> None:
        # Hover previews only while nothing is pinned; once a point is clicked
        # the panel (and its Play button) stays on that track — the tooltip
        # still identifies whatever is under the cursor.
        if index < 0 or self._pinned >= 0 or index >= len(self._tracks):
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
            self._pinned = index
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
            if indices[0] != self._pinned:
                self._on_point_clicked(indices[0])
        elif self._pinned >= 0 and self._pinned not in indices:
            self._pinned = -1
            self._details.clear()

    # ── Actions ───────────────────────────────────────────────────────────

    def _selection_paths(self) -> list:
        return cgd.selected_tracks(self._tracks, self._plot.get_selection())

    @Slot()
    def _on_play_selection(self) -> None:
        paths = self._selection_paths()
        if paths:
            self.play_tracks_requested.emit(paths, "Sound Map selection")

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
        """Export the selection as a playlist through the shared writer
        (CP-42): the file dialog's overwrite prompt is the confirmation, and a
        replaced playlist is backed up beside it and logged to
        ``Docs/playlist_writes.log``."""
        paths = self._selection_paths()
        if not paths:
            return
        if self._library_writes_blocked("Export playlist"):
            return
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export playlist", self._playlists_dir(), "Playlists (*.m3u *.m3u8)"
        )
        if not target:
            return
        import playlist_generator as pg

        item = pg.plan_playlist(paths, target)
        item.lines = ["#EXTM3U", *item.lines]
        result = pg.write_playlist_plan(
            [item], backup=True,
            log_callback=lambda line: self._log(line, "info"),
            log_dir=self._docs_dir() or None,
        )
        if result["errors"]:
            _path, msg = result["errors"][0]
            self._log(f"Export failed: {msg}", "error")
            self._status.setText(f"Export failed: {msg}")
            return
        note = " (the previous file was backed up beside it)" if result["backups"] else ""
        self._log(f"Exported {len(paths)} track(s) to {target}{note}", "ok")
        self._status.setText(f"Exported {len(paths)} track(s) to {target}{note}")

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
            self._status.setText("No cluster data found. Run an analysis first (the Analyse tab).")
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not open the 3D graph: {exc}")
            self._log(f"3D graph failed: {exc}", "error")
