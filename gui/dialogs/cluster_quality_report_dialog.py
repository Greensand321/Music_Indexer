"""Cluster quality report dialog showing metrics and suggestions."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

from gui.compat import QtCore, QtGui, QtWidgets


class ClusterQualityReportDialog(QtWidgets.QDialog):
    """Show clustering quality metrics and suggestions.

    Metrics:
    - Silhouette score (-1 to 1, higher is better)
    - Davies-Bouldin index (lower is better)
    - Calinski-Harabasz score (higher is better)
    - Per-cluster breakdown
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

        self._metrics = metrics
        self._cluster_info = cluster_info

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("Cluster Quality Report")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Overall metrics
        self._build_metrics_section(layout)

        # Per-cluster details
        self._build_cluster_details_section(layout)

        # Suggestions
        self._build_suggestions_section(layout)

        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def _build_metrics_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build overall metrics section."""
        card = self._create_card("Overall Metrics")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setSpacing(8)

        # Silhouette score
        silhouette = self._metrics.get("silhouette_score", None)
        if silhouette is not None:
            sil_layout = QtWidgets.QHBoxLayout()
            sil_layout.addWidget(QtWidgets.QLabel("Silhouette Score:"))
            sil_value = QtWidgets.QLabel(f"{silhouette:.3f}")
            sil_value.setStyleSheet(f"color: {self._score_color(silhouette, -1, 1)};")
            sil_layout.addWidget(sil_value)
            sil_label = QtWidgets.QLabel(self._score_label(silhouette, -1, 1))
            sil_label.setStyleSheet("color: #666;")
            sil_layout.addWidget(sil_label)
            sil_layout.addStretch()
            card_layout.addLayout(sil_layout)

        # Davies-Bouldin index
        db_index = self._metrics.get("davies_bouldin_index", None)
        if db_index is not None:
            db_layout = QtWidgets.QHBoxLayout()
            db_layout.addWidget(QtWidgets.QLabel("Davies-Bouldin Index:"))
            db_value = QtWidgets.QLabel(f"{db_index:.3f}")
            # For DB index, lower is better, so invert the scoring
            db_value.setStyleSheet(f"color: {self._score_color(1.0 / (1 + db_index), 0, 1)};")
            db_layout.addWidget(db_value)
            db_label = QtWidgets.QLabel("(lower is better)")
            db_label.setStyleSheet("color: #666;")
            db_layout.addWidget(db_label)
            db_layout.addStretch()
            card_layout.addLayout(db_layout)

        # Calinski-Harabasz score
        ch_score = self._metrics.get("calinski_harabasz_score", None)
        if ch_score is not None:
            ch_layout = QtWidgets.QHBoxLayout()
            ch_layout.addWidget(QtWidgets.QLabel("Calinski-Harabasz Score:"))
            ch_value = QtWidgets.QLabel(f"{ch_score:.1f}")
            # Normalize to 0-1 range for coloring (higher is better)
            normalized_ch = min(1.0, ch_score / 500.0)  # Normalize to 0-1
            ch_value.setStyleSheet(f"color: {self._score_color(normalized_ch, 0, 1)};")
            ch_layout.addWidget(ch_value)
            ch_label = QtWidgets.QLabel("(higher is better)")
            ch_label.setStyleSheet("color: #666;")
            ch_layout.addWidget(ch_label)
            ch_layout.addStretch()
            card_layout.addLayout(ch_layout)

        parent_layout.addWidget(card)

    def _build_cluster_details_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build per-cluster details section."""
        card = self._create_card("Per-Cluster Breakdown")

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(6)

        for cluster_id, info in sorted(self._cluster_info.items()):
            cluster_card = self._create_cluster_card(cluster_id, info)
            scroll_layout.addWidget(cluster_card)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)

        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.addWidget(scroll)

        parent_layout.addWidget(card, 1)

    def _build_cluster_card(self, cluster_id: int, info: Dict) -> QtWidgets.QWidget:
        """Create a card for a single cluster."""
        card = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(card)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Title
        title = QtWidgets.QLabel(f"Cluster {cluster_id}")
        title.setStyleSheet("font-weight: bold;")

        # Stats
        size = info.get("size", 0)
        layout.addRow("Size:", QtWidgets.QLabel(f"{size} tracks"))

        genres = info.get("genres", [])
        if genres:
            genres_str = ", ".join(genres[:3])
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

        card.setStyleSheet("border: 1px solid #ddd; border-radius: 4px; padding: 4px;")

        widget = QtWidgets.QWidget()
        widget_layout = QtWidgets.QVBoxLayout(widget)
        widget_layout.setContentsMargins(0, 0, 0, 0)
        widget_layout.addWidget(card)
        return widget

    def _build_suggestions_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build suggestions section."""
        suggestions = self._generate_suggestions()

        if not suggestions:
            return

        card = self._create_card("Suggestions for Improvement")
        card_layout = QtWidgets.QVBoxLayout(card)

        for suggestion in suggestions:
            suggestion_widget = QtWidgets.QLabel(f"• {suggestion}")
            suggestion_widget.setWordWrap(True)
            suggestion_widget.setStyleSheet("color: #333; font-size: 11px;")
            card_layout.addWidget(suggestion_widget)

        parent_layout.addWidget(card)

    def _create_card(self, title: str) -> QtWidgets.QWidget:
        """Create a card widget."""
        card = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.insertWidget(0, title_label)

        card.setStyleSheet("border: 1px solid #ddd; border-radius: 4px; background-color: #fafafa;")

        return card

    def _score_color(self, value: float, min_val: float, max_val: float) -> str:
        """Get color based on score (normalized to 0-1 range)."""
        normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

        if normalized > 0.7:
            return "#22c55e"  # Green
        elif normalized > 0.4:
            return "#eab308"  # Yellow
        else:
            return "#ef4444"  # Red

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

        silhouette = self._metrics.get("silhouette_score", 0)
        if silhouette < 0.3:
            suggestions.append("Low silhouette score - clusters may not be well-separated. Try different K value or more features.")
        elif silhouette < 0.5:
            suggestions.append("Moderate silhouette score - clustering is reasonable but could be improved.")

        db_index = self._metrics.get("davies_bouldin_index", 0)
        if db_index > 1.5:
            suggestions.append("High Davies-Bouldin index - try increasing K or using a different algorithm.")

        # Check for very small clusters
        cluster_info = self._cluster_info or {}
        small_clusters = [c for c, info in cluster_info.items() if info.get("size", 0) < 5]
        if small_clusters:
            suggestions.append(f"Found {len(small_clusters)} very small cluster(s) - consider merging them or increasing min_cluster_size.")

        if not suggestions:
            suggestions.append("Clustering looks good! Feel free to proceed or try different parameters.")

        return suggestions
