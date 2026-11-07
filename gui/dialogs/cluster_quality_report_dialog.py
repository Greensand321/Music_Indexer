"""Cluster quality report dialog showing metrics and suggestions."""
from __future__ import annotations

from typing import Dict, List

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.manager import get_manager


class ClusterQualityReportDialog(QtWidgets.QDialog):
    """Show clustering quality metrics and suggestions.

    Metrics:
    - Silhouette score (-1 to 1, higher is better)
    - Davies-Bouldin index (lower is better)
    - Calinski-Harabasz score (higher is better)
    - Per-cluster breakdown

    Scores are computed on clustered tracks only; HDBSCAN's unclustered
    ("noise") tracks are reported separately (``noise_fraction``).
    """

    def __init__(
        self,
        metrics: Dict,
        cluster_info: Dict,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Cluster Quality Report")
        self.setMinimumWidth(700)
        self.setMinimumHeight(600)

        self._metrics = metrics or {}
        self._cluster_info = cluster_info or {}

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI."""
        layout = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("Cluster Quality Report")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self._build_metrics_section(layout)
        self._build_cluster_details_section(layout)
        self._build_suggestions_section(layout)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    # ── Sections ──────────────────────────────────────────────────────────

    def _metric_row(self, card_layout, caption: str, value: str, colour: str, note: str) -> None:
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(caption))
        value_label = QtWidgets.QLabel(value)
        value_label.setStyleSheet(f"color: {colour};")
        row.addWidget(value_label)
        note_label = QtWidgets.QLabel(note)
        note_label.setObjectName("statusHint")
        row.addWidget(note_label)
        row.addStretch()
        card_layout.addLayout(row)

    def _build_metrics_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build overall metrics section."""
        card, card_layout = self._create_card("Overall Metrics")

        silhouette = self._metrics.get("silhouette_score", None)
        if silhouette is not None:
            self._metric_row(
                card_layout, "Silhouette Score:", f"{silhouette:.3f}",
                self._score_color(silhouette, -1, 1), self._score_label(silhouette, -1, 1),
            )

        db_index = self._metrics.get("davies_bouldin_index", None)
        if db_index is not None:
            # For DB index, lower is better, so invert the scoring
            self._metric_row(
                card_layout, "Davies-Bouldin Index:", f"{db_index:.3f}",
                self._score_color(1.0 / (1 + db_index), 0, 1), "(lower is better)",
            )

        ch_score = self._metrics.get("calinski_harabasz_score", None)
        if ch_score is not None:
            normalized_ch = min(1.0, ch_score / 500.0)
            self._metric_row(
                card_layout, "Calinski-Harabasz Score:", f"{ch_score:.1f}",
                self._score_color(normalized_ch, 0, 1), "(higher is better)",
            )

        noise = self._metrics.get("noise_fraction")
        if noise:
            note = QtWidgets.QLabel(
                f"{noise:.0%} of tracks are unclustered (noise); scores above "
                "exclude them."
            )
            note.setObjectName("statusHint")
            note.setWordWrap(True)
            card_layout.addWidget(note)

        if silhouette is None and db_index is None and ch_score is None:
            empty = QtWidgets.QLabel("Scores need at least two clusters.")
            empty.setObjectName("statusHint")
            card_layout.addWidget(empty)

        parent_layout.addWidget(card)

    def _build_cluster_details_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build per-cluster details section."""
        card, card_layout = self._create_card("Per-Cluster Breakdown")

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(6)

        def _key(item):
            try:
                return (0, int(item[0]))
            except (TypeError, ValueError):
                return (1, str(item[0]))

        for cluster_id, info in sorted(self._cluster_info.items(), key=_key):
            scroll_layout.addWidget(self._build_cluster_card(cluster_id, info or {}))

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        card_layout.addWidget(scroll)

        parent_layout.addWidget(card, 1)

    def _build_cluster_card(self, cluster_id, info: Dict) -> QtWidgets.QWidget:
        """Create a card for a single cluster."""
        card = QtWidgets.QFrame()
        card.setObjectName("workspaceCard")
        layout = QtWidgets.QFormLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title = QtWidgets.QLabel(f"Cluster {cluster_id}")
        title.setObjectName("cardTitle")
        layout.addRow(title)

        size = info.get("size", 0)
        layout.addRow("Size:", QtWidgets.QLabel(f"{size} tracks"))

        genres = info.get("genres", [])
        if genres:
            genres_str = ", ".join(str(g) for g in genres[:3])
            if len(genres) > 3:
                genres_str += ", ..."
            layout.addRow("Genres:", QtWidgets.QLabel(genres_str))

        tempo_range = info.get("tempo_range", None)
        if tempo_range:
            layout.addRow("Tempo:", QtWidgets.QLabel(f"{tempo_range[0]:.0f}-{tempo_range[1]:.0f} BPM"))

        silhouette = info.get("silhouette", None)
        if silhouette is not None:
            sil_label = QtWidgets.QLabel(f"{silhouette:.3f}")
            sil_label.setStyleSheet(f"color: {self._score_color(silhouette, -1, 1)};")
            layout.addRow("Silhouette:", sil_label)

        return card

    def _build_suggestions_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build suggestions section."""
        suggestions = self._generate_suggestions()
        if not suggestions:
            return

        card, card_layout = self._create_card("Suggestions for Improvement")
        for suggestion in suggestions:
            label = QtWidgets.QLabel(f"• {suggestion}")
            label.setWordWrap(True)
            card_layout.addWidget(label)

        parent_layout.addWidget(card)

    def _create_card(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
        """Return a themed card and the layout its content goes into.

        Each section used to call ``QVBoxLayout(card)`` a second time, which Qt
        rejects ("already has a layout"), so every row ended up in an orphan
        layout and the report rendered as three empty boxes.
        """
        card = QtWidgets.QFrame()
        card.setObjectName("workspaceCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        return card, layout

    # ── Scoring ───────────────────────────────────────────────────────────

    def _score_color(self, value: float, min_val: float, max_val: float) -> str:
        """Get a theme colour for a score (normalized to 0-1 range)."""
        tokens = get_manager().current
        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        if normalized > 0.7:
            return tokens.success
        if normalized > 0.4:
            return tokens.warning
        return tokens.danger

    def _score_label(self, value: float, min_val: float, max_val: float) -> str:
        """Get label based on score."""
        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

        if normalized > 0.7:
            return "(good)"
        elif normalized > 0.4:
            return "(fair)"
        else:
            return "(poor)"

    def _generate_suggestions(self) -> List[str]:
        """Generate improvement suggestions based on metrics."""
        suggestions = []

        silhouette = self._metrics.get("silhouette_score", None)
        if silhouette is not None:
            if silhouette < 0.3:
                suggestions.append("Low silhouette score - clusters may not be well-separated. Try different K value or more features.")
            elif silhouette < 0.5:
                suggestions.append("Moderate silhouette score - clustering is reasonable but could be improved.")

        db_index = self._metrics.get("davies_bouldin_index", 0)
        if db_index > 1.5:
            suggestions.append("High Davies-Bouldin index - try increasing K or using a different algorithm.")

        noise = self._metrics.get("noise_fraction") or 0
        if noise > 0.3:
            suggestions.append(
                f"{noise:.0%} of tracks are unclustered - try a smaller HDBSCAN min cluster size."
            )

        cluster_info = self._cluster_info or {}
        small_clusters = [c for c, info in cluster_info.items() if (info or {}).get("size", 0) < 5]
        if small_clusters:
            suggestions.append(f"Found {len(small_clusters)} very small cluster(s) - consider fewer clusters (K) or a larger min_cluster_size.")

        if not suggestions:
            suggestions.append("Clustering looks good! Feel free to proceed or try different parameters.")

        return suggestions
