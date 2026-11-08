"""Sound Map → Map: the in-app 2-D cluster map, plus the 3-D browser view.

Layout (Liquid Glass): the map is the hero card. Its first row holds the
Pan / Box / Lasso segmented control, Fit / Clear / Reload glyph buttons and
"3-D view"; the second what is selected (with the map summary or the last
result under it) and what to do with it (CSV…, Export playlist…, Play
selection); then the plot on its glass well, or an empty state on the same
well before there is a map. The legend and the track details are glass side
cards: a column right of the map when the content is at least ``WIDE_FROM``
wide, below it otherwise (side by side, or stacked in a 900 px window), so
the page never scrolls sideways.

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

    #: Content width from which the side cards sit beside the map.
    WIDE_FROM = 900
    #: ...and from which they sit side by side below it.
    SIDE_BY_SIDE_FROM = 680
    SIDE_WIDTH = 300

    def _build_ui(self) -> None:
        from gui.themes.glass import apply_font
        from gui.widgets.glass import ElidedLabel, GlassPanel
        from gui.widgets.glass_parts import EmptyState
        from gui.widgets.glyphs import set_glyph

        cl = self.content_layout
        cl.addWidget(self._make_subtitle(
            "Each dot is a track and each colour a cluster: tracks that sound alike sit "
            "together. Select a region to play it or save it as a playlist."
        ))

        # ── The map (hero card): tools, summary, plot, selection bar ──────
        self._map_card = self._make_card()
        self._map_card.setObjectName("headerCard")
        ml = QtWidgets.QVBoxLayout(self._map_card)
        ml.setContentsMargins(16, 14, 16, 14)
        ml.setSpacing(10)

        tools = QtWidgets.QHBoxLayout()
        tools.setSpacing(6)
        # Pan / Box / Lasso: a segmented pill. The checked mode is a raised
        # pill, the others are ghost buttons on the well track.
        self._mode_bar = GlassPanel(surface="well", radius=999)
        self._mode_bar.setObjectName("soundMapModeBar")
        mode_row = QtWidgets.QHBoxLayout(self._mode_bar)
        mode_row.setContentsMargins(3, 3, 3, 3)
        mode_row.setSpacing(2)
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QtWidgets.QPushButton] = {}
        for mode, text, glyph, tip in (
            (MODE_PAN, "Pan", "move", "Drag to move the view"),
            (MODE_RECT, "Box", "marquee", "Drag a box to select the points inside it"),
            (MODE_LASSO, "Lasso", "lasso", "Draw a freehand loop to select points"),
        ):
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName("soundMapModeBtn")
            btn.setCheckable(True)
            btn.setFlat(True)
            set_glyph(btn, glyph)
            btn.setToolTip(f"{tip}  ·  hold Ctrl or Shift to add to the selection")
            btn.setAccessibleName(f"{text} mode")
            btn.setMinimumWidth(76)       # may shrink below the style's 80+ in a narrow window
            btn.toggled.connect(lambda on, b=btn: b.setFlat(not on))
            btn.clicked.connect(lambda _c=False, m=mode: self._set_mode(m))
            self._mode_group.addButton(btn)
            self._mode_buttons[mode] = btn
            mode_row.addWidget(btn)
            if mode == MODE_PAN:
                btn.setChecked(True)
        tools.addWidget(self._mode_bar)
        tools.addSpacing(6)

        def icon_button(glyph: str, tip: str, name: str) -> QtWidgets.QPushButton:
            b = QtWidgets.QPushButton()
            b.setObjectName("soundMapIconBtn")
            set_glyph(b, glyph)
            b.setToolTip(tip)
            b.setAccessibleName(name)
            b.setFixedSize(40, 36)
            return b

        self._fit_btn = icon_button("fit", "Fit view: zoom to show every point", "Fit view")
        self._fit_btn.clicked.connect(lambda: self._plot.fit_view())
        self._clear_btn = icon_button("close", "Clear selection", "Clear selection")
        self._clear_btn.clicked.connect(lambda: self._plot.clear_selection())
        self._reload_btn = icon_button("refresh", "Reload: re-read the cluster data from disk",
                                       "Reload the map")
        self._reload_btn.clicked.connect(self.reload_data)
        for b in (self._fit_btn, self._clear_btn, self._reload_btn):
            tools.addWidget(b)
        tools.addStretch(1)
        self._browser_btn = QtWidgets.QPushButton("3-D view")
        self._browser_btn.setObjectName("soundMapBrowserBtn")
        set_glyph(self._browser_btn, "external")
        self._browser_btn.setToolTip("Open the 3-D map in your web browser")
        self._browser_btn.setAccessibleName("Open the 3-D view in the browser")
        self._browser_btn.setMinimumWidth(100)
        self._browser_btn.clicked.connect(self._on_open_3d)
        tools.addWidget(self._browser_btn)
        ml.addLayout(tools)

        # Selection bar, above the plot so it stays in view: what is
        # selected (and, under it, what the map shows or the last result:
        # an export, a save, an error), then what to do with it.
        sel = QtWidgets.QHBoxLayout()
        sel.setSpacing(8)
        text_col = QtWidgets.QVBoxLayout()
        text_col.setContentsMargins(2, 0, 0, 0)
        text_col.setSpacing(0)
        self._sel_label = ElidedLabel("Nothing selected")
        apply_font(self._sel_label, "label", weight=600)
        text_col.addWidget(self._sel_label)
        self._status = ElidedLabel("", QtCore.Qt.TextElideMode.ElideMiddle)
        self._status.setObjectName("mutedLabel")
        apply_font(self._status, "caption")
        text_col.addWidget(self._status)
        sel.addLayout(text_col, 1)
        self._csv_btn = QtWidgets.QPushButton("CSV…")
        set_glyph(self._csv_btn, "download")
        self._csv_btn.setToolTip("Export the selected tracks as a CSV file")
        self._csv_btn.setAccessibleName("Export the selection as CSV")
        self._csv_btn.clicked.connect(self._on_export_csv)
        self._m3u_btn = QtWidgets.QPushButton("Export playlist…")
        set_glyph(self._m3u_btn, "save")
        self._m3u_btn.setToolTip("Save the selected tracks as a playlist (.m3u)")
        self._m3u_btn.clicked.connect(self._on_export_m3u)
        self._play_btn = QtWidgets.QPushButton("Play selection")
        self._play_btn.setObjectName("primaryBtn")
        set_glyph(self._play_btn, "play", role="inverse")
        self._play_btn.setToolTip("Play the selected tracks in the Player")
        self._play_btn.clicked.connect(self._on_play_selection)
        for b in (self._csv_btn, self._m3u_btn, self._play_btn):
            b.setEnabled(False)
            sel.addWidget(b)
        ml.addLayout(sel)

        # The plot, or an empty state on the same well before there is a map.
        self._plot_stack = QtWidgets.QStackedWidget()
        self._plot = InteractiveScatterPlot()
        self._plot.setMinimumHeight(340)
        self._plot.point_clicked.connect(self._on_point_clicked)
        self._plot.points_selected.connect(self._on_selection_changed)
        self._plot.hover_changed.connect(self._on_hover)
        self._plot_stack.addWidget(self._plot)

        self._empty_panel = GlassPanel(surface="well", radius=20)
        self._empty_panel.setMinimumHeight(340)
        ep = QtWidgets.QVBoxLayout(self._empty_panel)
        ep.setContentsMargins(24, 24, 24, 24)
        self._empty = EmptyState(
            "No map yet",
            "Run an analysis on the Analyse tab; the map of your library appears here.",
            "Go to Analyse",
        )
        self._empty.action_requested.connect(lambda: self.navigate_requested.emit("clustered"))
        ep.addWidget(self._empty)
        self._plot_stack.addWidget(self._empty_panel)
        ml.addWidget(self._plot_stack, 1)

        # ── Side cards: the legend and the track details ──────────────────
        self._legend_card = self._make_card()
        lc = QtWidgets.QVBoxLayout(self._legend_card)
        lc.setContentsMargins(16, 14, 16, 14)
        self._legend = ClusterLegendWidget()
        # Room for several rows beside the details panel (it was squeezed to ~2).
        self._legend.setMinimumHeight(200)
        self._legend.cluster_toggled.connect(self._plot.set_cluster_visible)
        self._legend.cluster_selected.connect(self._on_cluster_selected)
        lc.addWidget(self._legend)

        self._details_card = self._make_card()
        dc = QtWidgets.QVBoxLayout(self._details_card)
        dc.setContentsMargins(16, 14, 16, 14)
        self._details = TrackDetailsPanel()
        self._details.play_requested.connect(self._on_play_one)
        self._details.reveal_requested.connect(self._on_reveal)
        dc.addWidget(self._details)

        Dir = QtWidgets.QBoxLayout.Direction
        self._side = QtWidgets.QWidget()
        self._side_layout = QtWidgets.QBoxLayout(Dir.TopToBottom, self._side)
        self._side_layout.setContentsMargins(0, 0, 0, 0)
        self._side_layout.setSpacing(18)
        self._side_layout.addWidget(self._legend_card, 1)
        self._side_layout.addWidget(self._details_card, 0)

        # Wide: the side cards are a column right of the map. Medium: they
        # move below it, side by side. Narrow (the 900 px window): below it,
        # one above the other.
        self._body = QtWidgets.QWidget()
        self._body_layout = QtWidgets.QBoxLayout(Dir.LeftToRight, self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(18)
        self._body_layout.addWidget(self._map_card, 1)
        self._body_layout.addWidget(self._side, 0)
        self._tier: str | None = None
        self._apply_width(0)
        cl.addWidget(self._body, 1)

    def layout_tier(self) -> str:
        """"wide" (side cards beside the map), "medium" (below it, side by
        side) or "narrow" (below it, stacked)."""
        return self._tier or ""

    def is_wide(self) -> bool:
        """True while the legend and details sit beside the map."""
        return self._tier == "wide"

    def _apply_width(self, width: int) -> None:
        """Lay the side cards out for a content width of ``width`` px."""
        tier = ("wide" if width >= self.WIDE_FROM
                else "medium" if width >= self.SIDE_BY_SIDE_FROM else "narrow")
        if tier == self._tier:
            return
        self._tier = tier
        Dir = QtWidgets.QBoxLayout.Direction
        if tier == "wide":
            self._body_layout.setDirection(Dir.LeftToRight)
            self._side_layout.setDirection(Dir.TopToBottom)
            self._side_layout.setStretchFactor(self._details_card, 0)
            self._side.setFixedWidth(self.SIDE_WIDTH)
            return
        self._body_layout.setDirection(Dir.TopToBottom)
        self._side.setMinimumWidth(0)
        self._side.setMaximumWidth(16777215)
        if tier == "medium":
            self._side_layout.setDirection(Dir.LeftToRight)
            self._side_layout.setStretchFactor(self._details_card, 1)
        else:
            self._side_layout.setDirection(Dir.TopToBottom)
            self._side_layout.setStretchFactor(self._details_card, 0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        from gui.themes.tokens import SPACE

        # The panel's own width (the scroll viewport may not be laid out yet);
        # leave room for a vertical scroll bar.
        bar = self._scroll_area.verticalScrollBar().sizeHint().width()
        self._apply_width(event.size().width() - 2 * SPACE.gutter - bar)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)
        self._status.setToolTip(text)

    def _show_map(self, shown: bool, title: str = "", message: str = "",
                  action: bool = True) -> None:
        """Show the plot, or the empty state on the same well (never over the
        plot-unavailable notice)."""
        if shown or not self._plot.available:
            self._plot_stack.setCurrentWidget(self._plot)
            return
        self._empty.set_text(title, message)
        self._empty.set_action("Go to Analyse" if action else "")
        self._plot_stack.setCurrentWidget(self._empty_panel)

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
            self._set_empty("No library selected.", kind="no_library")
            return
        if not self._plot.available:
            self._set_empty("The interactive graph is unavailable (see the notice below).",
                            kind="unavailable")
            return
        try:
            data = cgd.load_cluster_data(self._library_path)
            xy, labels, tracks = cgd.graph_points(data)
        except cgd.ClusterDataError as exc:
            message = _page_wording(str(exc))
            self._set_empty(message, kind="no_data" if "No cluster data" in message else "error")
            return
        except Exception as exc:  # noqa: BLE001
            self._set_empty(f"Could not load cluster data: {exc}", kind="error")
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
        self._set_status(summary)
        self._show_map(True)
        self._set_controls_enabled(True)
        self._log(f"Sound Map loaded: {summary}", "ok")

    #: What the empty state says for each reason the map is empty.
    _EMPTY_TEXT = {
        "no_library": ("Choose a library first",
                       "Pick your music folder with Change library; after an analysis "
                       "its map appears here.", False),
        "no_data": ("No map yet",
                    "Run an analysis on the Analyse tab; the map of your library "
                    "appears here.", True),
        "error": ("The map could not be loaded", "", True),
    }

    def _set_empty(self, message: str, kind: str = "error") -> None:
        self._xy, self._labels, self._tracks, self._metadata = [], [], [], []
        self._map_run = None
        self._pinned = -1
        self._set_status(message)
        if kind == "unavailable":
            self._show_map(True)        # the plot shows its own notice
        else:
            title, text, action = self._EMPTY_TEXT.get(kind, self._EMPTY_TEXT["error"])
            self._show_map(False, title, text or message, action=action)
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
        self._set_status(text)

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
            self._set_status(f"Export failed: {msg}")
            return
        note = " (the previous file was backed up beside it)" if result["backups"] else ""
        self._log(f"Exported {len(paths)} track(s) to {target}{note}", "ok")
        self._set_status(f"Exported {len(paths)} track(s) to {target}{note}")

    def _write(self, target: str, text: str, count: int) -> None:
        try:
            Path(target).write_text(text, encoding="utf-8")
            self._log(f"Exported {count} track(s) to {target}", "ok")
            self._set_status(f"Exported {count} track(s) to {target}")
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
            self._set_status("No library selected.")
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
                self._set_status("Could not generate the 3D graph.")
        except FileNotFoundError:
            self._set_status("No cluster data found. Run an analysis first (the Analyse tab).")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Could not open the 3D graph: {exc}")
            self._log(f"3D graph failed: {exc}", "error")
