"""Interactive 2-D cluster scatter plot built on PyQtGraph.

Display-only: it renders points it is handed and reports what the user did with
them. Loading, validating and exporting cluster data lives in
``cluster_graph_data``.

Rendering is split in two layers so a click does not redraw the whole map:

* ``scatter`` — every *visible* point, drawn with one cached brush/pen per
  cluster. Rebuilt only when the data or a cluster's visibility changes.
* ``selection_scatter`` — just the selected points, enlarged and outlined,
  on top. Rebuilt on every selection change (cost ∝ selection size).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager

try:
    import pyqtgraph as pg
    PG_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - only on installs without a working pyqtgraph
    # Not just ImportError: pyqtgraph can fail on a broken *dependency*
    # (e.g. 0.14 without colorama), which must not be reported as "not installed".
    pg = None
    PG_IMPORT_ERROR = _exc

#: Cluster id HDBSCAN assigns to points that belong to no cluster.
NOISE_LABEL = -1

#: Selection interaction modes.
MODE_PAN = "pan"
MODE_RECT = "rect"
MODE_LASSO = "lasso"

_POINT_SIZE = 9
_SELECTED_SIZE = 13


def _cluster_colour(index: int, total: int) -> QtGui.QColor:
    """Return a distinct, readable colour for the *index*-th cluster."""
    hue = int(360 * index / max(total, 1))
    return QtGui.QColor.fromHsv(hue, 190, 235)


#: Noise points are deliberately desaturated so they read as "leftovers"
#: rather than as just another cluster competing for attention.
NOISE_COLOUR = QtGui.QColor(140, 140, 150)


def unavailable_reason() -> str:
    """Human-readable reason the interactive plot cannot be drawn ('' if it can)."""
    exc = PG_IMPORT_ERROR
    if exc is None:
        return ""
    if isinstance(exc, ModuleNotFoundError) and (exc.name or "").split(".")[0] == "pyqtgraph":
        return (
            "PyQtGraph is not installed, so the interactive graph cannot be drawn.\n\n"
            "Install it with:    pip install pyqtgraph"
        )
    return (
        f"PyQtGraph is installed but could not be loaded:\n{type(exc).__name__}: {exc}\n\n"
        "Reinstalling it (pip install --force-reinstall pyqtgraph) usually fixes a "
        "missing dependency."
    )


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
    of replacing it. Hidden clusters are never part of the selection.
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
        self._palette: Dict[int, QtGui.QColor] = {}
        self._brushes: Dict[int, object] = {}
        self._pens: Dict[int, object] = {}

        # State
        self._selected: set[int] = set()
        self._hovered: int = -1
        self._visibility: Dict[int, bool] = {}
        self._mode: str = MODE_PAN
        self._drag_points: List[QtCore.QPointF] = []
        self._additive = False
        self._highlight_pen = None

        self._available = pg is not None
        if self._available:
            self._build_plot()
            get_manager().theme_changed.connect(self._apply_theme)
            self._apply_theme(get_manager().current)
        else:
            self._build_unavailable_notice()

    @property
    def available(self) -> bool:
        """False when PyQtGraph could not be loaded (a notice is shown instead)."""
        return self._available

    # ── Construction ──────────────────────────────────────────────────────

    def _build_unavailable_notice(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel(
            unavailable_reason() + "\n\nThe 3D browser view is unaffected and still works."
        )
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self._notice = label

    def _build_plot(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view_box = _SelectionViewBox()
        self.plot_widget = pg.PlotWidget(viewBox=self._view_box)
        self.plot_widget.setMenuEnabled(False)
        # The embedding axes carry no interpretable units — only relative
        # position means anything — so the axes are hidden rather than
        # labelled with numbers that invite over-reading.
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.hideAxis("left")
        layout.addWidget(self.plot_widget)

        self.scatter = pg.ScatterPlotItem(pxMode=True, hoverable=False)
        self.scatter.sigClicked.connect(self._on_points_clicked)
        self.plot_widget.addItem(self.scatter)

        # Selected points, drawn on top. It ignores mouse buttons so clicks
        # fall through to the base scatter, which owns hit-testing.
        self.selection_scatter = pg.ScatterPlotItem(pxMode=True, hoverable=False)
        self.selection_scatter.setZValue(10)
        self.selection_scatter.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.plot_widget.addItem(self.selection_scatter)

        # Overlay used to draw the in-progress rectangle or lasso.
        self._marquee = QtWidgets.QGraphicsPathItem()
        self._marquee.setZValue(50)
        self._marquee.hide()
        self._view_box.addItem(self._marquee, ignoreBounds=True)

        self._view_box.drag_started.connect(self._on_drag_started)
        self._view_box.drag_moved.connect(self._on_drag_moved)
        self._view_box.drag_finished.connect(self._on_drag_finished)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    @QtCore.Slot(object)
    def _apply_theme(self, tokens) -> None:
        """Follow the app theme: plot background, selection outline, marquee."""
        if not self._available:
            return
        self.plot_widget.setBackground(QtGui.QColor(tokens.card_bg))
        # The outline must contrast with the background (white vanished on
        # light themes), so it uses the primary text colour.
        self._highlight_pen = pg.mkPen(QtGui.QColor(tokens.text_primary), width=2)
        accent = QtGui.QColor(tokens.accent)
        fill = QtGui.QColor(accent)
        fill.setAlpha(40)
        self._marquee.setPen(pg.mkPen(accent, width=1.5))
        self._marquee.setBrush(pg.mkBrush(fill))
        self._render_selection()

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
        self._palette = palette
        self._colours = [palette[c] for c in clusters]
        self._visibility = {c: self._visibility.get(c, True) for c in ordered}
        if self._available:
            # One brush and pen per cluster, shared by all of its points.
            self._brushes = {c: pg.mkBrush(col) for c, col in palette.items()}
            self._pens = {c: pg.mkPen(col.darker(140), width=0.5) for c, col in palette.items()}

        self._render()
        if self._available and self._xy:
            self.plot_widget.autoRange()

    def clear_data(self) -> None:
        """Remove every point (e.g. after the cluster data failed to load)."""
        self.set_data([], [])

    def reset_visibility(self) -> None:
        """Forget hidden clusters (a new library starts with everything shown)."""
        self._visibility = {c: True for c in self._visibility}
        self._render()

    def cluster_ids(self) -> List[int]:
        """Return cluster ids in display order — real clusters first, noise last."""
        unique = set(self._clusters)
        real = sorted(c for c in unique if c != NOISE_LABEL)
        return real + ([NOISE_LABEL] if NOISE_LABEL in unique else [])

    def cluster_colour(self, cluster_id: int) -> QtGui.QColor:
        """Return the colour used for *cluster_id* (for a matching legend)."""
        return self._palette.get(cluster_id, NOISE_COLOUR)

    # ── Rendering ─────────────────────────────────────────────────────────

    def _is_visible(self, index: int) -> bool:
        return self._visibility.get(self._clusters[index], True)

    def _spot_arrays(self, indices: Sequence[int], size: int, pen=None) -> dict:
        xs = [self._xy[i][0] for i in indices]
        ys = [self._xy[i][1] for i in indices]
        brushes = [self._brushes[self._clusters[i]] for i in indices]
        pens = [pen] * len(indices) if pen is not None else [self._pens[self._clusters[i]] for i in indices]
        return dict(x=xs, y=ys, brush=brushes, pen=pens, size=size, data=list(indices))

    def _render(self) -> None:
        """Redraw the base layer (data or visibility changed) and the selection."""
        if not self._available:
            return
        visible = [i for i in range(len(self._xy)) if self._is_visible(i)]
        if visible:
            self.scatter.setData(**self._spot_arrays(visible, _POINT_SIZE))
        else:
            self.scatter.clear()
        self._render_selection()

    def _render_selection(self) -> None:
        """Redraw only the selected points (cheap: proportional to the selection)."""
        if not self._available or self._highlight_pen is None:
            return
        chosen = [i for i in sorted(self._selected) if self._is_visible(i)]
        if chosen:
            self.selection_scatter.setData(
                **self._spot_arrays(chosen, _SELECTED_SIZE, pen=self._highlight_pen)
            )
        else:
            self.selection_scatter.clear()

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
        """Replace the selection and notify listeners (hidden points are dropped)."""
        self._selected = {
            int(i) for i in (indices or [])
            if 0 <= int(i) < len(self._xy) and self._is_visible(int(i))
        }
        self._render_selection()
        self.points_selected.emit(self.get_selection())

    def get_selection(self) -> List[int]:
        """Return selected indices in ascending order (stable for export)."""
        return sorted(self._selected)

    def clear_selection(self) -> None:
        self.set_selection([])

    def select_cluster(self, cluster_id: int, additive: bool = False) -> None:
        """Select every point belonging to *cluster_id* (ignored while it is hidden)."""
        if not self._visibility.get(cluster_id, True):
            return
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
            if self._is_visible(i) and path.contains(QtCore.QPointF(x, y))
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
        # point radii in screen space.
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
        if self._visibility.get(cluster_id) == visible:
            return
        self._visibility[cluster_id] = visible
        dropped = False
        if not visible:
            # Hidden points must not stay selected: they would be played or
            # exported while nothing selected is visible.
            before = len(self._selected)
            self._selected = {i for i in self._selected if self._clusters[i] != cluster_id}
            dropped = len(self._selected) != before
        self._render()
        if dropped:
            self.points_selected.emit(self.get_selection())

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
