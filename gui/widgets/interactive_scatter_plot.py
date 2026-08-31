"""Interactive 2-D cluster scatter plot built on PyQtGraph.

Display-only: it renders points it is handed and reports what the user did with
them. Loading, validating and exporting cluster data lives in
``cluster_graph_data``.

Point counts are bounded by ``clustered_playlists.MAX_VISUALIZATION_POINTS``
(5,000), so the geometry here works in plain Python lists rather than numpy —
one less import to keep working, and well within budget at that size.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from gui.compat import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - exercised only on installs without it
    pg = None

#: Cluster id HDBSCAN assigns to points that belong to no cluster.
NOISE_LABEL = -1

#: Selection interaction modes.
MODE_PAN = "pan"
MODE_RECT = "rect"
MODE_LASSO = "lasso"


def _cluster_colour(index: int, total: int) -> QtGui.QColor:
    """Return a distinct, readable colour for the *index*-th cluster."""
    hue = int(360 * index / max(total, 1))
    return QtGui.QColor.fromHsv(hue, 190, 235)


#: Noise points are deliberately desaturated so they read as "leftovers"
#: rather than as just another cluster competing for attention.
NOISE_COLOUR = QtGui.QColor(140, 140, 150)


if pg is not None:

    class _SelectionViewBox(pg.ViewBox):
        """ViewBox that reports drags instead of panning when selecting.

        Subclassing is the supported way to intercept drags in PyQtGraph; the
        scene-level mouse signals do not distinguish a drag from a pan.
        """

        drag_started = QtCore.Signal(object)   # QPointF in view coords
        drag_moved = QtCore.Signal(object)     # QPointF in view coords
        drag_finished = QtCore.Signal(object)  # QPointF in view coords

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.selection_mode = MODE_PAN

        def mouseDragEvent(self, ev, axis=None):  # noqa: N802 - PyQtGraph API
            if self.selection_mode == MODE_PAN:
                super().mouseDragEvent(ev, axis)
                return

            ev.accept()
            point = self.mapToView(ev.pos())
            if ev.isStart():
                self.drag_started.emit(point)
            elif ev.isFinish():
                self.drag_finished.emit(point)
            else:
                self.drag_moved.emit(point)

else:  # pragma: no cover - placeholder so the module imports without PyQtGraph
    _SelectionViewBox = None  # type: ignore[assignment]


class InteractiveScatterPlot(QtWidgets.QWidget):
    """Scatter plot of a clustered library with hover, click and drag selection.

    Selection modes
    ---------------
    ``MODE_PAN``    drag pans the view (default)
    ``MODE_RECT``   drag sweeps a rectangle and selects the points inside it
    ``MODE_LASSO``  drag draws a freehand loop and selects the points inside it

    Holding Ctrl or Shift while selecting adds to the current selection instead
    of replacing it.
    """

    point_clicked = QtCore.Signal(int)      # index, -1 when cleared
    points_selected = QtCore.Signal(list)   # list[int]
    hover_changed = QtCore.Signal(int)      # index, -1 when nothing hovered

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Data — populated by set_data(). Initialised before any widget is
        # built so a mouse event arriving mid-construction cannot find a
        # half-built object.
        self._xy: List[Tuple[float, float]] = []
        self._clusters: List[int] = []
        self._labels: List[str] = []
        self._metadata: List[dict] = []
        self._colours: List[QtGui.QColor] = []

        # State
        self._selected: set[int] = set()
        self._hovered: int = -1
        self._visibility: Dict[int, bool] = {}
        self._mode: str = MODE_PAN
        self._drag_points: List[QtCore.QPointF] = []
        self._additive = False

        self._available = pg is not None
        if self._available:
            self._build_plot()
        else:
            self._build_unavailable_notice()

    # ── Construction ──────────────────────────────────────────────────────

    def _build_unavailable_notice(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            "PyQtGraph is not installed, so the interactive graph cannot be "
            "drawn.\n\nInstall it with:    pip install pyqtgraph\n\n"
            "The 3D browser view is unaffected and still works."
        )
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def _build_plot(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view_box = _SelectionViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self._view_box)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        # The embedding axes carry no interpretable units — only relative
        # position means anything — so the axes are hidden rather than
        # labelled with numbers that invite over-reading.
        self.plot_widget.setBackground(self.palette().base().color())
        layout.addWidget(self.plot_widget)

        self.scatter = pg.ScatterPlotItem(pxMode=True, hoverable=False)
        self.scatter.sigClicked.connect(self._on_points_clicked)
        self.plot_widget.addItem(self.scatter)

        # Overlay used to draw the in-progress rectangle or lasso.
        self._marquee = QtWidgets.QGraphicsPathItem()
        self._marquee.setPen(pg.mkPen(QtGui.QColor(80, 160, 255), width=1.5))
        self._marquee.setBrush(pg.mkBrush(QtGui.QColor(80, 160, 255, 40)))
        self._marquee.setZValue(50)
        self._marquee.hide()
        self._view_box.addItem(self._marquee, ignoreBounds=True)

        self._view_box.drag_started.connect(self._on_drag_started)
        self._view_box.drag_moved.connect(self._on_drag_moved)
        self._view_box.drag_finished.connect(self._on_drag_finished)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # ── Data ──────────────────────────────────────────────────────────────

    def set_data(
        self,
        xy: Sequence[Tuple[float, float]],
        clusters: Sequence[int],
        labels: Sequence[str] | None = None,
        metadata: Sequence[dict] | None = None,
    ) -> None:
        """Replace the plotted data.

        All four sequences are consumed positionally, so they must be the same
        length; mismatches raise rather than silently mapping a dot onto the
        wrong track.
        """
        xy = [(float(x), float(y)) for x, y in xy]
        clusters = [int(c) for c in clusters]
        labels = list(labels) if labels is not None else [f"Point {i}" for i in range(len(xy))]
        metadata = list(metadata) if metadata is not None else [{} for _ in xy]

        if not (len(xy) == len(clusters) == len(labels) == len(metadata)):
            raise ValueError(
                "xy, clusters, labels and metadata must be the same length "
                f"(got {len(xy)}, {len(clusters)}, {len(labels)}, {len(metadata)})"
            )

        self._xy = xy
        self._clusters = clusters
        self._labels = labels
        self._metadata = metadata
        self._selected.clear()
        self._hovered = -1

        ordered = self.cluster_ids()
        real = [c for c in ordered if c != NOISE_LABEL]
        palette = {c: _cluster_colour(i, len(real)) for i, c in enumerate(real)}
        palette[NOISE_LABEL] = NOISE_COLOUR
        self._colours = [palette[c] for c in clusters]
        self._visibility = {c: self._visibility.get(c, True) for c in ordered}

        self._render()
        if self._available:
            self.plot_widget.autoRange()

    def cluster_ids(self) -> List[int]:
        """Return cluster ids in display order — real clusters first, noise last."""
        unique = set(self._clusters)
        real = sorted(c for c in unique if c != NOISE_LABEL)
        return real + ([NOISE_LABEL] if NOISE_LABEL in unique else [])

    def cluster_colour(self, cluster_id: int) -> QtGui.QColor:
        """Return the colour used for *cluster_id* (for a matching legend)."""
        for cid, colour in zip(self._clusters, self._colours):
            if cid == cluster_id:
                return colour
        return NOISE_COLOUR

    # ── Rendering ─────────────────────────────────────────────────────────

    def _render(self) -> None:
        if not self._available or not self._xy:
            if self._available:
                self.scatter.setData(spots=[])
            return

        highlight = pg.mkPen(QtGui.QColor(255, 255, 255), width=2)
        spots = []
        for i, (x, y) in enumerate(self._xy):
            if not self._visibility.get(self._clusters[i], True):
                continue
            colour = self._colours[i]
            selected = i in self._selected
            spots.append({
                "pos": (x, y),
                "size": 13 if selected else 9,
                "brush": pg.mkBrush(colour),
                "pen": highlight if selected else pg.mkPen(colour.darker(140), width=0.5),
                "data": i,
            })
        self.scatter.setData(spots=spots)

    # ── Selection ─────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """Switch between panning, rectangle selection and lasso selection."""
        if mode not in (MODE_PAN, MODE_RECT, MODE_LASSO):
            raise ValueError(f"unknown selection mode: {mode!r}")
        self._mode = mode
        if self._available:
            self._view_box.selection_mode = mode
            self._view_box.setMouseEnabled(x=mode == MODE_PAN, y=mode == MODE_PAN)
            self.plot_widget.setCursor(
                QtCore.Qt.CursorShape.ArrowCursor if mode == MODE_PAN
                else QtCore.Qt.CursorShape.CrossCursor
            )

    def mode(self) -> str:
        return self._mode

    def set_selection(self, indices: Sequence[int] | None) -> None:
        """Replace the selection and notify listeners."""
        self._selected = {int(i) for i in (indices or []) if 0 <= int(i) < len(self._xy)}
        self._render()
        self.points_selected.emit(self.get_selection())

    def get_selection(self) -> List[int]:
        """Return selected indices in ascending order (stable for export)."""
        return sorted(self._selected)

    def clear_selection(self) -> None:
        self.set_selection([])

    def select_cluster(self, cluster_id: int, additive: bool = False) -> None:
        """Select every point belonging to *cluster_id*."""
        indices = [i for i, c in enumerate(self._clusters) if c == cluster_id]
        if additive:
            indices = sorted(self._selected.union(indices))
        self.set_selection(indices)

    def _apply_selection(self, indices: Sequence[int]) -> None:
        if self._additive:
            self.set_selection(sorted(self._selected.union(indices)))
        else:
            self.set_selection(indices)

    # ── Mouse handling ────────────────────────────────────────────────────

    @staticmethod
    def _additive_modifier() -> bool:
        mods = QtWidgets.QApplication.keyboardModifiers()
        return bool(
            mods & (QtCore.Qt.KeyboardModifier.ControlModifier
                    | QtCore.Qt.KeyboardModifier.ShiftModifier)
        )

    def _on_points_clicked(self, _scatter, points) -> None:
        if not len(points):
            return
        index = int(points[0].data())
        self._additive = self._additive_modifier()
        if self._additive:
            self._apply_selection([index])
        else:
            self.set_selection([index])
        self.point_clicked.emit(index)

    def _on_drag_started(self, point: QtCore.QPointF) -> None:
        self._additive = self._additive_modifier()
        self._drag_points = [point]
        self._marquee.show()

    def _on_drag_moved(self, point: QtCore.QPointF) -> None:
        if not self._drag_points:
            return
        if self._mode == MODE_LASSO:
            self._drag_points.append(point)
        else:
            self._drag_points = [self._drag_points[0], point]
        self._marquee.setPath(self._current_path())

    def _on_drag_finished(self, point: QtCore.QPointF) -> None:
        if not self._drag_points:
            return
        self._on_drag_moved(point)
        path = self._current_path()
        self._marquee.hide()
        self._drag_points = []

        hits = [
            i for i, (x, y) in enumerate(self._xy)
            if self._visibility.get(self._clusters[i], True)
            and path.contains(QtCore.QPointF(x, y))
        ]
        # A click-sized drag in selection mode reads as "clear", which is the
        # least surprising way to deselect without leaving the mode.
        self._apply_selection(hits)

    def _current_path(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        if len(self._drag_points) < 2:
            return path
        if self._mode == MODE_LASSO:
            path.moveTo(self._drag_points[0])
            for pt in self._drag_points[1:]:
                path.lineTo(pt)
            path.closeSubpath()
        else:
            start, end = self._drag_points[0], self._drag_points[-1]
            path.addRect(QtCore.QRectF(start, end).normalized())
        return path

    def _on_mouse_moved(self, scene_pos) -> None:
        if not self._available or not self._xy:
            return
        # Hit-testing is delegated to PyQtGraph, which tests against the drawn
        # point radii in screen space. The previous implementation compared
        # data-space distances against a threshold derived from the data range,
        # which mis-picked whenever the axes had different scales and cost a
        # full pass over every point on every mouse move.
        if not self.plot_widget.sceneBoundingRect().contains(scene_pos):
            return
        view_pos = self._view_box.mapSceneToView(scene_pos)
        hits = self.scatter.pointsAt(view_pos)
        index = int(hits[0].data()) if len(hits) else -1
        if index != self._hovered:
            self._hovered = index
            self.scatter.setToolTip(self._tooltip_for(index) if index >= 0 else "")
            self.hover_changed.emit(index)

    def _tooltip_for(self, index: int) -> str:
        if not (0 <= index < len(self._labels)):
            return ""
        lines = [self._labels[index]]
        cluster = self._clusters[index]
        lines.append("Unclustered" if cluster == NOISE_LABEL else f"Cluster {cluster}")
        for key, value in list((self._metadata[index] or {}).items())[:4]:
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    # ── Cluster visibility ────────────────────────────────────────────────

    def set_cluster_visible(self, cluster_id: int, visible: bool) -> None:
        if self._visibility.get(cluster_id) != visible:
            self._visibility[cluster_id] = visible
            self._render()

    def toggle_cluster_visibility(self, cluster_id: int) -> None:
        self.set_cluster_visible(cluster_id, not self._visibility.get(cluster_id, True))

    def visible_clusters(self) -> List[int]:
        return [c for c, vis in self._visibility.items() if vis]

    # ── View ──────────────────────────────────────────────────────────────

    def fit_view(self) -> None:
        """Zoom to fit all points."""
        if self._available and self._xy:
            self.plot_widget.autoRange()

    def selected_labels(self) -> List[str]:
        """Return the labels (track paths) of the selected points, in order."""
        return [self._labels[i] for i in self.get_selection() if i < len(self._labels)]
