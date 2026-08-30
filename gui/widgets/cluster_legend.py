"""Cluster legend: colour key, track counts, and per-cluster visibility."""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from gui.compat import QtCore, QtGui, QtWidgets

#: Cluster id HDBSCAN assigns to points that belong to no cluster.
NOISE_LABEL = -1


class _ClusterRow(QtWidgets.QWidget):
    """One legend entry: visibility checkbox, colour swatch, clickable label."""

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
        self.cluster_id = cluster_id

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox.setChecked(True)
        self.checkbox.setToolTip("Show or hide this cluster")
        self.checkbox.toggled.connect(
            lambda visible: self.toggled.emit(self.cluster_id, visible)
        )
        layout.addWidget(self.checkbox, 0)

        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(12, 12)
        if colour is not None:
            swatch.setStyleSheet(
                f"background:{colour.name()};border-radius:6px;"
            )
        layout.addWidget(swatch, 0)

        name = "Unclustered" if cluster_id == NOISE_LABEL else f"Cluster {cluster_id}"
        text = f"{name}  ({count})"
        detail = self._format_detail(info)
        if detail:
            text += f"\n{detail}"

        self.label = QtWidgets.QLabel(text)
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.label.setToolTip("Click to select every track in this cluster")
        layout.addWidget(self.label, 1)

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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        # The whole row is a click target for "select this cluster", except the
        # checkbox, which Qt delivers to the child before this ever fires.
        self.selected.emit(self.cluster_id)
        super().mousePressEvent(event)

    def set_visible_state(self, visible: bool) -> None:
        """Update the checkbox without re-emitting ``toggled``."""
        was = self.checkbox.blockSignals(True)
        self.checkbox.setChecked(visible)
        self.checkbox.blockSignals(was)


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
        layout.setSpacing(6)

        title = QtWidgets.QLabel("Clusters")
        title.setStyleSheet("font-weight:600;")
        layout.addWidget(title)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._container = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        scroll.setWidget(self._container)
        layout.addWidget(scroll, 1)

        self._summary = QtWidgets.QLabel("No clusters")
        self._summary.setObjectName("statusHint")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        row = QtWidgets.QHBoxLayout()
        self._show_all_btn = QtWidgets.QPushButton("Show all")
        self._show_all_btn.clicked.connect(lambda: self.set_all_visible(True))
        self._hide_all_btn = QtWidgets.QPushButton("Hide all")
        self._hide_all_btn.clicked.connect(lambda: self.set_all_visible(False))
        row.addWidget(self._show_all_btn)
        row.addWidget(self._hide_all_btn)
        layout.addLayout(row)

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

    def _on_row_toggled(self, cluster_id: int, visible: bool) -> None:
        self._visible[cluster_id] = visible
        self.cluster_toggled.emit(cluster_id, visible)

    def _update_summary(self, counts: Mapping[int, int], order: Sequence[int]) -> None:
        total = sum(int(v) for v in counts.values())
        n_clusters = len([c for c in order if c != NOISE_LABEL])
        noise = int(counts.get(NOISE_LABEL, 0))
        parts = [f"{n_clusters} clusters", f"{total} tracks"]
        if noise:
            parts.append(f"{noise} unclustered")
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
