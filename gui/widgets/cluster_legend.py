"""Cluster legend: colour key, track counts, and per-cluster visibility.

Each cluster is a glass list row (radius 12, a faint well fill, a hover
tint): its colour dot doubles as the show/hide toggle, the name and an
optional detail line ("Ambient, Drone · 80 BPM") sit beside it, and the track
count is right-aligned. Clicking the row (not the dot) selects the cluster.
Everything paints from the active material at paint time, so a theme switch
is a repaint (no per-row stylesheet).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import apply_font, current_material, qcolor

#: Cluster id HDBSCAN assigns to points that belong to no cluster.
NOISE_LABEL = -1

_ROW_RADIUS = 12


def cluster_name(cluster_id: int) -> str:
    return "Unclustered" if cluster_id == NOISE_LABEL else f"Cluster {cluster_id}"


class _DotToggle(QtWidgets.QAbstractButton):
    """The cluster's colour dot, checkable: filled (with a soft ring) when the
    cluster is shown, a hollow ring when it is hidden."""

    SIZE = 22

    def __init__(self, colour: QtGui.QColor | None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._colour = QtGui.QColor(colour) if colour is not None else None
        self.setCheckable(True)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.TabFocus)

    def colour(self) -> QtGui.QColor | None:
        return self._colour

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return QtCore.QSize(self.SIZE, self.SIZE)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.manager import get_manager

        t = get_manager().current
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            c = QtCore.QPointF(self.width() / 2, self.height() / 2)
            colour = QtGui.QColor(self._colour) if self._colour is not None else qcolor(t.text_muted)
            if self.isChecked():
                halo = QtGui.QColor(colour)
                halo.setAlpha(46)
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(halo)
                p.drawEllipse(c, 8.0, 8.0)
                p.setBrush(colour)
                p.drawEllipse(c, 5.0, 5.0)
            else:
                ring = qcolor(t.text_muted)
                p.setPen(QtGui.QPen(ring, 1.5))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(c, 5.0, 5.0)
            if self.hasFocus():
                p.setPen(QtGui.QPen(QtGui.QColor(t.input_focus), 1.5))
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                p.drawEllipse(c, 9.5, 9.5)
        finally:
            p.end()


class _ClusterRow(QtWidgets.QWidget):
    """One legend entry: colour dot (visibility toggle), name and detail,
    track count. The row itself is the "select this cluster" target."""

    toggled = QtCore.Signal(int, bool)   # cluster_id, visible
    selected = QtCore.Signal(int)        # cluster_id

    def __init__(
        self,
        cluster_id: int,
        count: int,
        colour: QtGui.QColor | None,
        info: Mapping[str, object],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("clusterLegendRow")
        self.cluster_id = cluster_id
        self.count = count
        self._hover = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setToolTip("Click to select every track in this cluster")

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 12, 4)
        layout.setSpacing(6)

        name = cluster_name(cluster_id)
        self.checkbox = _DotToggle(colour)
        self.checkbox.setChecked(True)
        self.checkbox.setToolTip("Show or hide this cluster on the map")
        self.checkbox.setAccessibleName(f"Show {name}")
        self.checkbox.toggled.connect(self._on_toggled)
        layout.addWidget(self.checkbox, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(0)
        self.label = QtWidgets.QLabel(name)
        self.label.setObjectName("clusterLegendName")
        apply_font(self.label, "label", weight=600)
        text_col.addWidget(self.label)
        detail = self._format_detail(info)
        self.detail = QtWidgets.QLabel(detail)
        self.detail.setObjectName("mutedLabel")
        apply_font(self.detail, "caption")
        self.detail.setVisible(bool(detail))
        self.detail.setMinimumWidth(1)
        text_col.addWidget(self.detail)
        layout.addLayout(text_col, 1)

        self.count_label = QtWidgets.QLabel(f"{count:,}")
        self.count_label.setObjectName("clusterLegendCount")
        apply_font(self.count_label, "caption", weight=600, tabular=True)
        self.count_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.count_label.setToolTip(f"{count} track{'s' if count != 1 else ''}")
        layout.addWidget(self.count_label, 0)

        self.setMinimumHeight(44 if detail else 34)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    @staticmethod
    def _format_detail(info: Mapping[str, object]) -> str:
        """Summarise whatever per-cluster facts the payload carries."""
        parts: List[str] = []
        genres = info.get("genres")
        if isinstance(genres, (list, tuple)) and genres:
            shown = ", ".join(str(g) for g in list(genres)[:2])
            if len(genres) > 2:
                shown += ", …"
            parts.append(shown)
        tempo = info.get("tempo") or info.get("avg_tempo")
        if isinstance(tempo, (int, float)):
            parts.append(f"{tempo:.0f} BPM")
        return " · ".join(parts)

    def _on_toggled(self, visible: bool) -> None:
        self._sync_text_state(visible)
        self.toggled.emit(self.cluster_id, visible)

    def _sync_text_state(self, visible: bool) -> None:
        # A hidden cluster's text steps back (the disabled palette role).
        for w in (self.label, self.detail, self.count_label):
            w.setEnabled(visible)
        self.update()

    def is_shown(self) -> bool:
        return self.checkbox.isChecked()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        # The whole row is a click target for "select this cluster", except the
        # dot, which Qt delivers to the child before this ever fires.
        self.selected.emit(self.cluster_id)
        super().mousePressEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:  # noqa: D401
        kind = event.type()
        if kind == QtCore.QEvent.Type.HoverEnter:
            self._hover = True
            self.update()
        elif kind == QtCore.QEvent.Type.HoverLeave:
            self._hover = False
            self.update()
        return super().event(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        m = current_material()
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            path = QtGui.QPainterPath()
            path.addRoundedRect(rect, _ROW_RADIUS, _ROW_RADIUS)
            p.fillPath(path, qcolor(m.well.fill, 0.75))
            if self._hover:
                p.fillPath(path, qcolor(m.hover))
        finally:
            p.end()

    def set_visible_state(self, visible: bool) -> None:
        """Update the dot without re-emitting ``toggled``."""
        was = self.checkbox.blockSignals(True)
        self.checkbox.setChecked(visible)
        self.checkbox.blockSignals(was)
        self._sync_text_state(visible)


class ClusterLegendWidget(QtWidgets.QWidget):
    """Colour key for the scatter plot, doubling as visibility controls."""

    cluster_toggled = QtCore.Signal(int, bool)   # cluster_id, visible
    cluster_selected = QtCore.Signal(int)        # cluster_id

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: Dict[int, _ClusterRow] = {}
        self._visible: Dict[int, bool] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        title = QtWidgets.QLabel("Clusters")
        title.setObjectName("cardTitle")
        head.addWidget(title, 1)
        self._show_all_btn = QtWidgets.QPushButton("Show all")
        self._show_all_btn.setFlat(True)
        self._show_all_btn.setToolTip("Show every cluster on the map")
        self._show_all_btn.clicked.connect(lambda: self.set_all_visible(True))
        self._hide_all_btn = QtWidgets.QPushButton("Hide all")
        self._hide_all_btn.setFlat(True)
        self._hide_all_btn.setToolTip("Hide every cluster")
        self._hide_all_btn.clicked.connect(lambda: self.set_all_visible(False))
        for b in (self._show_all_btn, self._hide_all_btn):
            apply_font(b, "label", weight=600)
            head.addWidget(b, 0)
        layout.addLayout(head)

        self._summary = QtWidgets.QLabel("No clusters")
        self._summary.setObjectName("mutedLabel")
        apply_font(self._summary, "caption")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("clusterLegendScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)
        self._container = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)
        scroll.setWidget(self._container)
        layout.addWidget(scroll, 1)
        self._scroll = scroll

    def set_clusters(
        self,
        counts: Mapping[int, int],
        order: Sequence[int],
        colours: Mapping[int, QtGui.QColor] | None = None,
        cluster_info: Mapping[int, dict] | None = None,
    ) -> None:
        """Rebuild the legend.

        ``cluster_info`` must be keyed by **int**. Payloads read back from
        ``cluster_info.json`` arrive with string keys, so pass them through
        ``cluster_graph_data.normalize_cluster_info`` first — looking them up
        with an int key otherwise silently found nothing, which is why
        per-cluster detail never used to appear here.
        """
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self._rows.clear()
        colours = colours or {}
        cluster_info = cluster_info or {}

        for cluster_id in order:
            row = _ClusterRow(
                cluster_id,
                int(counts.get(cluster_id, 0)),
                colours.get(cluster_id),
                cluster_info.get(cluster_id, {}),
            )
            row.set_visible_state(self._visible.get(cluster_id, True))
            row.toggled.connect(self._on_row_toggled)
            row.selected.connect(self.cluster_selected)
            self._layout.addWidget(row)
            self._rows[cluster_id] = row

        self._layout.addStretch(1)
        self._visible = {cid: self._visible.get(cid, True) for cid in order}
        self._update_summary(counts, order)
        has_rows = bool(order)
        self._show_all_btn.setEnabled(has_rows)
        self._hide_all_btn.setEnabled(has_rows)

    def rows(self) -> Dict[int, _ClusterRow]:
        return dict(self._rows)

    def _on_row_toggled(self, cluster_id: int, visible: bool) -> None:
        self._visible[cluster_id] = visible
        self.cluster_toggled.emit(cluster_id, visible)

    def _update_summary(self, counts: Mapping[int, int], order: Sequence[int]) -> None:
        total = sum(int(v) for v in counts.values())
        n_clusters = len([c for c in order if c != NOISE_LABEL])
        noise = int(counts.get(NOISE_LABEL, 0))
        parts = [f"{n_clusters} clusters", f"{total:,} tracks"]
        if noise:
            parts.append(f"{noise:,} unclustered")
        self._summary.setText(" · ".join(parts))

    def set_cluster_visible(self, cluster_id: int, visible: bool) -> None:
        """Set visibility programmatically, keeping the checkbox in step.

        The checkbox used to be left untouched here, so the legend could end up
        claiming a cluster was shown while it was hidden on the plot.
        """
        self._visible[cluster_id] = visible
        row = self._rows.get(cluster_id)
        if row is not None:
            row.set_visible_state(visible)

    def set_all_visible(self, visible: bool) -> None:
        for cluster_id in list(self._rows):
            if self._visible.get(cluster_id) != visible:
                self.set_cluster_visible(cluster_id, visible)
                self.cluster_toggled.emit(cluster_id, visible)

    def visible_clusters(self) -> List[int]:
        return [cid for cid, vis in self._visible.items() if vis]

    def reset_visibility(self) -> None:
        """Forget hidden clusters (a new library starts with everything shown)."""
        self._visible = {cid: True for cid in self._visible}
        for row in self._rows.values():
            row.set_visible_state(True)

    def clear(self) -> None:
        """Remove every row (no cluster data loaded)."""
        self.set_clusters({}, [])
        self._summary.setText("No clusters")
