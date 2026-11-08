"""Cluster quality report dialog showing metrics and suggestions.

Contents in the app's glass language (the window itself is the shared dialog
chrome): the overall scores as a ``StatRow`` (toned good / fair / poor), one
glass row per cluster, and the suggestions, each group in a ``GlassCard``.
"""
from __future__ import annotations

from typing import Dict, List

from gui.compat import QtCore, QtGui, QtWidgets
from gui.themes.glass import apply_font, current_material, qcolor
from gui.themes.manager import get_manager
from gui.widgets.glass import GlassCard
from gui.widgets.glass_parts import StatRow

#: Tone → word, for the silhouette tile's label.
SILHOUETTE_QUALITY = {"success": "good", "warning": "fair", "danger": "poor"}


class _ClusterRow(QtWidgets.QWidget):
    """One cluster as a glass list row: name and size, detail, silhouette."""

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            path = QtGui.QPainterPath()
            path.addRoundedRect(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
            p.fillPath(path, qcolor(current_material().well.fill, 0.75))
        finally:
            p.end()


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
        self.setMinimumWidth(640)
        self.setMinimumHeight(560)

        self._metrics = metrics or {}
        self._cluster_info = cluster_info or {}
        self._stats: StatRow | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("Cluster Quality Report")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        intro = QtWidgets.QLabel(
            "How clearly the analysis separated your tracks. Unclustered tracks are "
            "left out of the scores."
        )
        intro.setObjectName("mutedLabel")
        apply_font(intro, "label")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._build_metrics_section(layout)
        self._build_cluster_details_section(layout)
        self._build_suggestions_section(layout)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    # ── Sections ──────────────────────────────────────────────────────────

    def stat_values(self) -> Dict[str, str]:
        """The overall scores as {key: value} (silhouette, davies_bouldin,
        calinski_harabasz, noise — whichever were computed)."""
        if self._stats is None:
            return {}
        return {key: self._stats.tile(key).value() for key in self._stats.keys()}

    def stat_tones(self) -> Dict[str, str]:
        if self._stats is None:
            return {}
        return {key: self._stats.tile(key).tone() for key in self._stats.keys()}

    def _build_metrics_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build overall metrics section: one stat tile per score."""
        card, card_layout = self._create_card("Overall")

        tiles: list[tuple[str, str]] = []
        values: dict[str, tuple[str, str]] = {}
        silhouette = self._metrics.get("silhouette_score", None)
        if silhouette is not None:
            tiles.append(("silhouette", f"Silhouette · {SILHOUETTE_QUALITY[self._silhouette_tone(silhouette)]}"))
            values["silhouette"] = (f"{silhouette:.3f}", self._silhouette_tone(silhouette))

        db_index = self._metrics.get("davies_bouldin_index", None)
        if db_index is not None:
            # For DB index, lower is better, so invert the scoring
            tiles.append(("davies_bouldin", "Davies–Bouldin"))
            values["davies_bouldin"] = (f"{db_index:.3f}", self._score_tone(1.0 / (1 + db_index), 0, 1))

        ch_score = self._metrics.get("calinski_harabasz_score", None)
        if ch_score is not None:
            normalized_ch = min(1.0, ch_score / 500.0)
            tiles.append(("calinski_harabasz", "Calinski–Harabasz"))
            values["calinski_harabasz"] = (f"{ch_score:.1f}", self._score_tone(normalized_ch, 0, 1))

        noise = self._metrics.get("noise_fraction")
        if noise:
            tiles.append(("noise", "Unclustered"))
            values["noise"] = (f"{noise:.0%}", "")

        if tiles:
            self._stats = StatRow(tiles)
            for key, (value, tone) in values.items():
                self._stats.tile(key).set_value(value, tone=tone)
            tips = {
                "silhouette": "Silhouette (−1 to 1): how clearly the clusters are apart; higher is better",
                "davies_bouldin": "Davies–Bouldin index: how much clusters overlap; lower is better",
                "calinski_harabasz": "Calinski–Harabasz score: how tight and apart the clusters are; "
                                     "higher is better",
                "noise": "Tracks HDBSCAN put in no cluster",
            }
            for key in self._stats.keys():
                self._stats.tile(key).setToolTip(tips.get(key, ""))
            card_layout.addWidget(self._stats)

        if noise:
            note = QtWidgets.QLabel(
                f"{noise:.0%} of tracks are unclustered (noise); the scores "
                "exclude them."
            )
            note.setObjectName("qualityReportNote")
            apply_font(note, "caption")
            note.setWordWrap(True)
            card_layout.addWidget(note)

        if silhouette is None and db_index is None and ch_score is None:
            empty = QtWidgets.QLabel("Scores need at least two clusters.")
            empty.setObjectName("qualityReportNote")
            apply_font(empty, "label")
            card_layout.addWidget(empty)

        parent_layout.addWidget(card)

    def _build_cluster_details_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build per-cluster details section."""
        card, card_layout = self._create_card("Per cluster")

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.viewport().setAutoFillBackground(False)

        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
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
        card_layout.addWidget(scroll, 1)

        parent_layout.addWidget(card, 1)

    def _build_cluster_card(self, cluster_id, info: Dict) -> QtWidgets.QWidget:
        """One cluster as a glass row: name and size, what it holds, its
        silhouette (toned)."""
        row = _ClusterRow()
        row.setObjectName("qualityClusterRow")
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(12)

        names = QtWidgets.QVBoxLayout()
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(0)
        title = QtWidgets.QLabel(f"Cluster {cluster_id}")
        apply_font(title, "label", weight=600)
        names.addWidget(title)

        details: List[str] = []
        genres = info.get("genres", [])
        if genres:
            genres_str = ", ".join(str(g) for g in genres[:3])
            if len(genres) > 3:
                genres_str += ", …"
            details.append(genres_str)
        tempo_range = info.get("tempo_range", None)
        if tempo_range:
            details.append(f"{tempo_range[0]:.0f}–{tempo_range[1]:.0f} BPM")
        detail = QtWidgets.QLabel(" · ".join(details))
        detail.setObjectName("mutedLabel")
        apply_font(detail, "caption")
        detail.setVisible(bool(details))
        names.addWidget(detail)
        layout.addLayout(names, 1)

        size = info.get("size", 0)
        size_label = QtWidgets.QLabel(f"{size} tracks")
        size_label.setObjectName("mutedLabel")
        apply_font(size_label, "caption", tabular=True)
        layout.addWidget(size_label, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        silhouette = info.get("silhouette", None)
        if silhouette is not None:
            sil_label = QtWidgets.QLabel(f"{silhouette:.3f}")
            apply_font(sil_label, "label", weight=600, tabular=True)
            sil_label.setToolTip("Silhouette of this cluster (−1 to 1, higher is better)")
            pal = sil_label.palette()
            pal.setColor(QtGui.QPalette.ColorRole.WindowText,
                         QtGui.QColor(getattr(get_manager().current, self._silhouette_tone(silhouette))))
            sil_label.setPalette(pal)
            layout.addWidget(sil_label, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        row.setMinimumHeight(44 if details else 34)
        return row

    def _build_suggestions_section(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build suggestions section."""
        suggestions = self._generate_suggestions()
        if not suggestions:
            return

        card, card_layout = self._create_card("Suggestions")
        for suggestion in suggestions:
            label = QtWidgets.QLabel(f"• {suggestion}")
            apply_font(label, "label")
            label.setWordWrap(True)
            card_layout.addWidget(label)

        parent_layout.addWidget(card)

    def _create_card(self, title: str) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
        """Return a glass card and the layout its content goes into.

        Each section used to call ``QVBoxLayout(card)`` a second time, which Qt
        rejects ("already has a layout"), so every row ended up in an orphan
        layout and the report rendered as three empty boxes.
        """
        card = GlassCard()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(10)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        return card, layout

    # ── Scoring ───────────────────────────────────────────────────────────

    @staticmethod
    def _silhouette_tone(value: float) -> str:
        """The silhouette's quality on the same scale as the Analyse results
        and the suggestions below: good above 0.5, fair above 0.3."""
        return "success" if value > 0.5 else "warning" if value > 0.3 else "danger"

    @staticmethod
    def _normalized(value: float, min_val: float, max_val: float) -> float:
        return (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5

    def _score_tone(self, value: float, min_val: float, max_val: float) -> str:
        """"success" / "warning" / "danger" for a score (normalized to 0-1)."""
        normalized = self._normalized(value, min_val, max_val)
        if normalized > 0.7:
            return "success"
        if normalized > 0.4:
            return "warning"
        return "danger"

    def _score_color(self, value: float, min_val: float, max_val: float) -> str:
        """Get a theme colour for a score (normalized to 0-1)."""
        tokens = get_manager().current
        return getattr(tokens, self._score_tone(value, min_val, max_val))

    def _score_label(self, value: float, min_val: float, max_val: float) -> str:
        """Get label based on score."""
        normalized = self._normalized(value, min_val, max_val)

        if normalized > 0.7:
            return "good"
        elif normalized > 0.4:
            return "fair"
        else:
            return "poor"

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
