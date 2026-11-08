"""Sound Map → Analyse: the clustering panel (Quick Start, results, settings).

Analysis only: a run writes the feature cache, ``Docs/cluster_members.json``
(the full membership), ``Docs/cluster_info.json`` (the map) and
``Docs/cluster_graph.html``. It writes no playlists itself: the Sound Map's
**Save cluster playlists…** action previews and saves one playlist per
cluster (``gui/cluster_playlist_jobs.py``), and a map selection can be
exported from the Map section (``graph.py``). After a successful run this
panel emits ``analysis_succeeded(result)``; the page uses it to offer the
preview when the run's "Offer to save a playlist per cluster" was ticked
(on the settings card, or in the wizard that fills it in).
"""
from __future__ import annotations

import copy
import logging
import os
import threading

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.cluster_playlist_jobs import ANALYSIS_JOB_NAME
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase
from gui.dialogs.clustering_wizard_dialog import (
    ClusteringWizardDialog,
    feature_toggles,
    features_text,
)
from gui.dialogs.cluster_quality_report_dialog import ClusterQualityReportDialog
from gui.themes.glass import apply_font
from gui.widgets.glass_parts import EmptyState, StatRow
from gui.widgets.glyphs import glyph_icon, set_glyph

logger = logging.getLogger(__name__)

#: Audio files an analysis looks at. ``cluster_playlist_export.AUDIO_EXTS``
#: must equal this set (a test pins the two).
_AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"}

#: Settings used by "Run Quick Start" (also the wizard's "Balanced" preset).
QUICK_START_CONFIG = {
    "features": {
        "tempo": True,
        "mfcc": True,
        "chroma": False,
        "spectral": False,
        "energy": True,
        "onset_rate": False,
    },
    "normalization": "standard",
    "algorithm": "kmeans",
    "k": 8,
    "min_cluster_size": 5,
    "min_samples": 5,
}


def compute_cluster_metrics(X, labels) -> dict:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz on the clustered points.

    HDBSCAN's noise points (label -1) are not a cluster, so they are left out
    of the scores and reported separately as ``noise_fraction``. Scores need
    at least two clusters; with fewer, only the counts are returned.
    """
    import numpy as np

    labels = np.asarray(labels)
    X = np.asarray(X)
    mask = labels >= 0
    n_clusters = int(len(np.unique(labels[mask]))) if mask.any() else 0
    metrics = {
        "n_clusters": n_clusters,
        "noise_count": int((~mask).sum()),
        "noise_fraction": float((~mask).mean()) if len(labels) else 0.0,
    }
    if n_clusters < 2 or mask.sum() <= n_clusters:
        return metrics
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    Xc, lc = X[mask], labels[mask]
    metrics["silhouette_score"] = float(silhouette_score(Xc, lc))
    metrics["davies_bouldin_index"] = float(davies_bouldin_score(Xc, lc))
    metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(Xc, lc))
    return metrics


class ClusterWorker(QtCore.QThread):
    """Background worker for clustering operation."""

    log_line = Signal(str)
    progress = Signal(int, str)  # percent, message
    # Not called ``finished``: that would shadow QThread.finished.
    done = Signal(bool, str, dict)  # success, message, results

    def __init__(self, library_path: str, config: dict) -> None:
        super().__init__()
        self.library_path = library_path
        # Deep copy config to avoid race conditions with main thread modifications
        self.config = copy.deepcopy(config)
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set() or self.isInterruptionRequested()

    def _cancelled(self) -> None:
        self.done.emit(False, "Cancelled", {})

    def run(self) -> None:
        """Run clustering with configuration."""
        try:
            import clustered_playlists as cp
            generate_clustered_playlists = cp.generate_clustered_playlists
        except ImportError as exc:
            self.done.emit(False, f"Missing dependency: {exc}", {})
            return
        except Exception as exc:  # noqa: BLE001 - e.g. a broken optional dep
            self.done.emit(False, f"Could not load the clustering engine: {exc}", {})
            return
        cancelled_exc = getattr(cp, "ClusteringCancelled", None)

        try:
            from utils.reserved_folders import walk_library

            self.progress.emit(0, "Scanning library...")
            tracks = []
            skipped = 0
            for dirpath, _dirs, files in walk_library(self.library_path):
                if self.is_cancelled():
                    self._cancelled()
                    return
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() not in _AUDIO_EXTS:
                        continue
                    full_path = os.path.join(dirpath, f)
                    try:
                        if not os.path.isfile(full_path) or not os.access(full_path, os.R_OK):
                            self.log_line.emit(f"⚠ Skipped unreadable file: {f}")
                            skipped += 1
                            continue
                    except OSError as e:
                        self.log_line.emit(f"⚠ Error checking file {f}: {e}")
                        skipped += 1
                        continue
                    tracks.append(full_path)

            if not tracks:
                if skipped > 0:
                    self.done.emit(False, f"No readable audio files found ({skipped} files skipped)", {})
                else:
                    self.done.emit(False, "No audio files found", {})
                return

            self.log_line.emit(f"Found {len(tracks)} readable tracks")
            if skipped > 0:
                self.log_line.emit(f"⚠ Skipped {skipped} files (unreadable or invalid)")
            self.progress.emit(5, f"Scanning complete: {len(tracks)} tracks")

            algorithm = str(self.config.get("algorithm", "kmeans")).lower()
            if algorithm in ("k-means", "k_means"):
                algorithm = "kmeans"
            if algorithm == "kmeans":
                k = int(self.config.get("k", 8))
                if k > len(tracks):
                    self.log_line.emit(f"⚠ K={k} exceeds track count ({len(tracks)}); reducing to {len(tracks)}")
                    k = len(tracks)
                params = {"n_clusters": k}
            elif algorithm == "hdbscan":
                min_size = int(self.config.get("min_cluster_size", 5))
                if min_size > len(tracks):
                    self.log_line.emit(
                        f"⚠ min_cluster_size={min_size} exceeds track count ({len(tracks)}); "
                        f"reducing to {len(tracks)}"
                    )
                    min_size = len(tracks)
                params = {
                    "min_cluster_size": min_size,
                    "min_samples": int(self.config.get("min_samples", 5)),
                }
            else:
                self.done.emit(False, f"Unknown clustering algorithm: {algorithm!r}", {})
                return

            selected = self.config.get("features") or {}
            if not any(selected.values()):
                # Nothing ticked: fall back to the balanced default rather than
                # failing, matching what the wizard offers as its middle preset.
                selected = dict(QUICK_START_CONFIG["features"])
            normalization = self.config.get("normalization", "standard")

            self.log_line.emit(f"Using algorithm: {algorithm.upper()}")
            self.log_line.emit(f"Using features: {', '.join(k for k, v in selected.items() if v)}")
            self.log_line.emit(f"Normalization: {normalization}")

            self.progress.emit(10, "Extracting audio features...")

            def _progress(done: int, total: int) -> None:
                pct = 10 + int(75 * done / max(total, 1))
                self.progress.emit(pct, f"Extracting audio features… {done}/{total}")

            result = generate_clustered_playlists(
                tracks,
                self.library_path,
                method=algorithm,
                params=params,
                log_callback=self.log_line.emit,
                engine="parallel",
                use_max_workers=True,
                features=selected,
                normalization=normalization,
                cancel_event=self._cancel_event,
                progress_callback=_progress,
            )

            if self.is_cancelled():
                self._cancelled()
                return

            self.progress.emit(90, "Computing cluster metrics...")
            metrics = {}
            if "labels" in result and "X" in result:
                try:
                    metrics = compute_cluster_metrics(result["X"], result["labels"])
                    if "silhouette_score" in metrics:
                        self.log_line.emit(f"Silhouette score: {metrics['silhouette_score']:.3f}")
                except Exception as e:  # noqa: BLE001
                    logger.warning("Could not compute metrics: %s", e, exc_info=True)
                    self.log_line.emit(f"Could not compute metrics: {e}")

            result["metrics"] = metrics
            result["config"] = self.config

            if self.is_cancelled():
                self._cancelled()
                return
            self.progress.emit(100, "Complete")
            self.done.emit(True, "Clustering complete", result)

        except Exception as exc:  # noqa: BLE001
            if (cancelled_exc is not None and isinstance(exc, cancelled_exc)) or self.is_cancelled():
                self._cancelled()
                return
            logger.exception("Clustering failed with exception")
            self.done.emit(False, f"Clustering failed: {exc}", {})


class EnhancedClusteredWorkspace(WorkspaceBase):
    """Sound Map → Analyse.

    Three cards, top to bottom:

    * **Quick Start** (the hero): the recommended run, one click, plus
      *Guided setup…* (the clustering wizard), which fills in the settings
      card below.
    * **Results**: an empty state before the first run, progress (bar,
      status, log, Cancel) while one runs, then the latest analysis as a
      ``StatRow`` (clusters, tracks, unclustered, silhouette, …) with *Open
      the map*, *Save cluster playlists…* and *Quality report*.
    * **Your own settings**: the features to listen for
      (``clustered_playlists.FEATURE_ORDER``) as pill toggles, the
      algorithm and its parameters, the scaling, and the offer to save a
      playlist per cluster; *Run with these settings* runs exactly them.
    """

    # Emitted whenever the latest result changes (a run finished, or the
    # results were cleared), so the Sound Map page can enable its actions.
    result_changed = Signal()
    # Emitted with the result after a *successful* run only (never on a
    # failure or cancel): the page offers "Save cluster playlists" on it.
    analysis_succeeded = Signal(dict)
    # The results card's "Save cluster playlists…": the page routes it to the
    # Map's ClusterPlaylistFlow (the same preview-first flow as its button).
    save_playlists_requested = Signal()

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: ClusterWorker | None = None
        # True while another job on the page (Save cluster playlists) runs:
        # the Run buttons wait for it (spec CP-34).
        self._external_busy = False
        # The settings card's choices (Run with these settings) are kept apart
        # from whatever ran last, so a Quick Start run cannot replace them
        # behind the user's back.
        self._last_run_config: dict | None = None
        self._current_result: dict | None = None
        self._run_buttons: list[QtWidgets.QPushButton] = []
        self._cancel_btn: QtWidgets.QPushButton | None = None
        self._save_btn: QtWidgets.QPushButton | None = None
        self._save_state: tuple[bool, str] = (False, "")
        self._stats = None
        self._build_ui()

    # Backwards-compatible alias (older code and tests read it).
    @property
    def _current_config(self) -> dict | None:
        return self._last_run_config

    @property
    def _wizard_config(self) -> dict:
        """What *Run with these settings* would run (read from the card)."""
        return self.settings_config()

    def _build_ui(self) -> None:
        """Build the UI."""
        cl = self.content_layout
        cl.addWidget(self._make_subtitle(
            "Group tracks that sound alike, then explore the groups on the Map. "
            "Analysing reads your audio; it changes nothing in your library."
        ))
        cl.addWidget(self._build_quick_start_card())
        cl.addWidget(self._build_results_card())
        cl.addWidget(self._build_settings_card())
        cl.addStretch(1)
        self._show_placeholder()

    def _card_header(self, layout: QtWidgets.QVBoxLayout, title: str, subtitle: str = "",
                     *, hero: bool = False) -> tuple[QtWidgets.QLabel, QtWidgets.QLabel]:
        if hero:
            # The hero's title is the "title" role (20/650); cardTitle's
            # stylesheet size would override an explicit font.
            head = QtWidgets.QLabel(title)
            head.setObjectName("soundMapHeroTitle")
            apply_font(head, "title")
        else:
            head = self._make_card_title(title)
        layout.addWidget(head)
        sub = QtWidgets.QLabel(subtitle)
        sub.setObjectName("mutedLabel")
        apply_font(sub, "label")
        sub.setWordWrap(True)
        sub.setVisible(bool(subtitle))
        layout.addWidget(sub)
        return head, sub

    def _build_quick_start_card(self) -> QtWidgets.QFrame:
        """The hero: the recommended run and the guided setup."""
        card = self._make_card()
        card.setObjectName("headerCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)
        q = QUICK_START_CONFIG
        self._card_header(
            layout, "Quick Start",
            f"{q['k']} clusters with K-Means, listening for {features_text(q['features'])}. "
            "The first run takes a while; later runs reuse the analysed audio.",
            hero=True,
        )
        layout.addSpacing(8)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        run_btn = self._make_primary_button("Run Quick Start")
        run_btn.setObjectName("primaryBtn")
        set_glyph(run_btn, "play", role="inverse")
        run_btn.setToolTip("Analyse the library with the recommended settings")
        run_btn.clicked.connect(self._on_run_quick_start)
        self._quick_btn = run_btn
        self._run_buttons.append(run_btn)
        row.addWidget(run_btn)
        wizard_btn = QtWidgets.QPushButton("Guided setup…")
        set_glyph(wizard_btn, "list")
        wizard_btn.setToolTip(
            "Walk through each choice step by step; your answers fill in "
            "“Your own settings” below."
        )
        wizard_btn.clicked.connect(self._on_configure_wizard)
        self._wizard_btn = wizard_btn
        row.addWidget(wizard_btn)
        row.addStretch(1)
        layout.addLayout(row)
        return card

    def _build_results_card(self) -> QtWidgets.QFrame:
        """Empty state, progress, results or an error: what the last run did."""
        card = self._make_card()
        card.setObjectName("workspaceCard")
        self._results_widget = card
        self._results_layout = QtWidgets.QVBoxLayout(card)
        self._results_layout.setContentsMargins(16, 14, 16, 14)
        self._results_layout.setSpacing(10)
        return card

    def _build_settings_card(self) -> QtWidgets.QFrame:
        """Your own settings: features, algorithm, scaling, the offer."""
        card = self._make_card()
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(8)
        self._card_header(layout, "Your own settings",
                          "Choose what the analysis listens for and how it groups tracks.")
        layout.addSpacing(4)

        layout.addWidget(self._caption("Listen for"))
        # Six pills in a row when there is room, three per row otherwise
        # (see _layout_features).
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self._feature_grid = grid
        self._feature_columns = 0
        self._feature_toggles = feature_toggles(grid)
        for toggle in self._feature_toggles.values():
            toggle.toggled.connect(self._on_settings_changed)
        self._layout_features(3)
        layout.addLayout(grid)
        layout.addSpacing(6)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(4)
        self._algo_combo = QtWidgets.QComboBox()
        self._algo_combo.addItem("K-Means", userData="kmeans")
        self._algo_combo.addItem("HDBSCAN", userData="hdbscan")
        self._algo_combo.setToolTip(
            "K-Means makes the number of groups you ask for; HDBSCAN finds them "
            "itself and leaves outliers unclustered (needs the optional 'hdbscan' package)."
        )
        self._algo_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        self._norm_combo = QtWidgets.QComboBox()
        for text, value in (("Standard", "standard"), ("MinMax", "minmax"), ("Robust", "robust")):
            self._norm_combo.addItem(text, userData=value)
        self._norm_combo.setToolTip(
            "How the features are scaled so each counts equally. Robust resists "
            "outlier tracks."
        )
        self._k_spin = QtWidgets.QSpinBox()
        self._k_spin.setRange(2, 100)
        self._k_spin.setToolTip("How many clusters K-Means makes")
        self._min_size_spin = QtWidgets.QSpinBox()
        self._min_size_spin.setRange(2, 100)
        self._min_size_spin.setToolTip("Fewest tracks an HDBSCAN cluster may have")
        self._min_samples_spin = QtWidgets.QSpinBox()
        self._min_samples_spin.setRange(1, 50)
        self._min_samples_spin.setToolTip("How dense a neighbourhood must be (HDBSCAN)")
        self._param_captions = {
            "k": self._caption("Clusters"),
            "min_size": self._caption("Min cluster size"),
            "min_samples": self._caption("Min samples"),
        }
        cells = (
            (self._caption("Algorithm"), self._algo_combo),
            (self._caption("Scaling"), self._norm_combo),
            (self._param_captions["k"], self._k_spin),
            (self._param_captions["min_size"], self._min_size_spin),
            (self._param_captions["min_samples"], self._min_samples_spin),
        )
        for col, (caption, field) in enumerate(cells):
            form.addWidget(caption, 0, col)
            form.addWidget(field, 1, col)
            form.setColumnStretch(col, 1)
        layout.addLayout(form)
        for w in (self._k_spin, self._min_size_spin, self._min_samples_spin):
            w.valueChanged.connect(self._on_settings_changed)
        self._norm_combo.currentIndexChanged.connect(self._on_settings_changed)

        self._offer_cb = QtWidgets.QCheckBox(
            "Offer to save a playlist per cluster when the analysis finishes"
        )
        self._offer_cb.setToolTip(
            "Opens the Save cluster playlists preview after a successful run. "
            "Nothing is written until you click Save."
        )
        self._offer_cb.toggled.connect(self._on_settings_changed)
        layout.addSpacing(4)
        layout.addWidget(self._offer_cb)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self._settings_summary = QtWidgets.QLabel("")
        self._settings_summary.setObjectName("mutedLabel")
        apply_font(self._settings_summary, "caption")
        self._settings_summary.setWordWrap(True)
        row.addWidget(self._settings_summary, 1)
        run_btn = QtWidgets.QPushButton("Run with these settings")
        set_glyph(run_btn, "play")
        run_btn.setToolTip("Analyse the library with the settings above")
        run_btn.clicked.connect(self._on_run_advanced)
        self._advanced_btn = run_btn
        self._run_buttons.append(run_btn)
        row.addWidget(run_btn, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        layout.addSpacing(4)
        layout.addLayout(row)

        # Start from the Quick Start settings: change what you like.
        self._display_config(copy.deepcopy(QUICK_START_CONFIG))
        return card

    @staticmethod
    def _caption(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("mutedLabel")
        apply_font(label, "caption", weight=600)
        return label

    #: Content width from which the six feature pills share one row.
    FEATURES_ONE_ROW_FROM = 900

    def _layout_features(self, columns: int) -> None:
        """Six pills at their natural width in one row, or three per row
        sharing the width (a narrow window)."""
        if columns == self._feature_columns:
            return
        self._feature_columns = columns
        grid = self._feature_grid
        for toggle in self._feature_toggles.values():
            grid.removeWidget(toggle)
        for i, toggle in enumerate(self._feature_toggles.values()):
            grid.addWidget(toggle, i // columns, i % columns)
        for c in range(7):
            grid.setColumnStretch(c, 0)
        if columns >= len(self._feature_toggles):
            grid.setColumnStretch(columns, 1)      # pills keep their width, left-aligned
        else:
            for c in range(columns):
                grid.setColumnStretch(c, 1)

    def feature_columns(self) -> int:
        return self._feature_columns

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        from gui.themes.tokens import SPACE

        bar = self._scroll_area.verticalScrollBar().sizeHint().width()
        width = event.size().width() - 2 * SPACE.gutter - bar
        self._layout_features(6 if width >= self.FEATURES_ONE_ROW_FROM else 3)

    # ── Settings card ─────────────────────────────────────────────────────

    def settings_config(self) -> dict:
        """The run config the settings card describes."""
        return {
            "features": {key: t.isChecked() for key, t in self._feature_toggles.items()},
            "normalization": self._norm_combo.currentData() or "standard",
            "algorithm": self._algo_combo.currentData() or "kmeans",
            "k": self._k_spin.value(),
            "min_cluster_size": self._min_size_spin.value(),
            "min_samples": self._min_samples_spin.value(),
            "create_playlists": self._offer_cb.isChecked(),
        }

    def _on_algorithm_changed(self, _index: int = 0) -> None:
        kmeans = self._algo_combo.currentData() == "kmeans"
        for key, widget in (("k", self._k_spin), ("min_size", self._min_size_spin),
                            ("min_samples", self._min_samples_spin)):
            shown = kmeans if key == "k" else not kmeans
            widget.setVisible(shown)
            self._param_captions[key].setVisible(shown)
        self._on_settings_changed()

    def _on_settings_changed(self, *_args) -> None:
        cfg = self.settings_config()
        if cfg["algorithm"] == "kmeans":
            how = f"{cfg['k']} clusters with K-Means"
        else:
            how = (f"HDBSCAN, clusters of at least {cfg['min_cluster_size']} tracks "
                   f"(min samples {cfg['min_samples']})")
        text = f"{how}, listening for {features_text(cfg['features'])}; {cfg['normalization']} scaling."
        if cfg.get("create_playlists"):
            text += " Offers to save them when the run finishes."
        self._settings_summary.setText(text)
        self._set_busy(self.is_busy())

    # ── Running ───────────────────────────────────────────────────────────

    def is_busy(self) -> bool:
        return self._worker is not None

    def _set_busy(self, busy: bool) -> None:
        """Enable the Run buttons (and say why not) for the current state:
        a run of our own, Save cluster playlists running on the page, or no
        feature ticked on the settings card."""
        blocked = busy or self._external_busy
        no_features = not any(t.isChecked() for t in self._feature_toggles.values())
        self._quick_btn.setEnabled(not blocked)
        self._advanced_btn.setEnabled(not blocked and not no_features)
        if self._external_busy and not busy:
            wait = "Wait for Save cluster playlists to finish."
            self._quick_btn.setToolTip(wait)
            self._advanced_btn.setToolTip(wait)
        else:
            self._quick_btn.setToolTip("Analyse the library with the recommended settings")
            self._advanced_btn.setToolTip(
                "Tick at least one feature to listen for." if no_features
                else "Analyse the library with the settings above")
        self._wizard_btn.setEnabled(not busy)

    def set_external_busy(self, busy: bool) -> None:
        """Another job on the page (Save cluster playlists) started or ended:
        a new analysis waits for it, so it cannot replace the membership the
        preview or write is working from."""
        self._external_busy = bool(busy)
        self._set_busy(self.is_busy())

    def _status(self, text: str, role: str) -> None:
        tokens = get_manager().current
        colour = {
            "busy": tokens.warning, "ok": tokens.success,
            "error": tokens.danger, "neutral": tokens.text_muted,
        }[role]
        self.status_changed.emit(text, colour)

    def _on_run_quick_start(self) -> None:
        """Run with quick start configuration."""
        self._run_clustering(copy.deepcopy(QUICK_START_CONFIG))

    def _on_configure_wizard(self) -> None:
        """Open the wizard on the current settings; its answers fill them in."""
        dialog = ClusteringWizardDialog(self)
        try:
            dialog.set_config(self.settings_config())
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self._display_config(dialog.get_config())
                self._log("Guided setup: your settings are filled in; Run with these "
                          "settings to analyse.", "info")
        finally:
            dialog.deleteLater()

    def _display_config(self, config: dict) -> None:
        """Show ``config`` in the settings card (what Run would use)."""
        features = config.get("features") or {}
        for key, toggle in self._feature_toggles.items():
            was = toggle.blockSignals(True)
            toggle.setChecked(bool(features.get(key, False)))
            toggle.blockSignals(was)
        for combo, value in ((self._algo_combo, config.get("algorithm", "kmeans")),
                             (self._norm_combo, config.get("normalization", "standard"))):
            index = combo.findData(value)
            if index >= 0:
                was = combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(was)
        for spin, key, default in ((self._k_spin, "k", 8),
                                   (self._min_size_spin, "min_cluster_size", 5),
                                   (self._min_samples_spin, "min_samples", 5)):
            was = spin.blockSignals(True)
            spin.setValue(int(config.get(key, default)))
            spin.blockSignals(was)
        was = self._offer_cb.blockSignals(True)
        self._offer_cb.setChecked(bool(config.get("create_playlists")))
        self._offer_cb.blockSignals(was)
        self._on_algorithm_changed()

    def _on_run_advanced(self) -> None:
        """Run with the settings card's configuration."""
        config = self.settings_config()
        if not any(config["features"].values()):
            QtWidgets.QMessageBox.warning(
                self, "No Features",
                "Tick at least one feature to listen for."
            )
            return
        self._run_clustering(config)

    def _run_clustering(self, config: dict) -> None:
        """Start clustering with given configuration."""
        if self.is_busy() or self._external_busy:
            return  # Run buttons are disabled; guards a queued second click
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return

        self._last_run_config = copy.deepcopy(config)
        self._set_result(None)

        self._show_progress_panel()

        worker = ClusterWorker(self._library_path, config)
        worker.log_line.connect(self._on_log_line)
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_clustering_finished)
        self._worker = worker
        self._set_busy(True)
        self._status("Clustering…", "busy")
        # The shared name lets Save cluster playlists wait for any analysis.
        self._start_worker(worker, ANALYSIS_JOB_NAME)

    # ── Results area ──────────────────────────────────────────────────────

    def _clear_results(self) -> None:
        """Remove everything from the results card, nested layouts included."""
        def clear(layout: QtWidgets.QLayout) -> None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
                elif item.layout() is not None:
                    clear(item.layout())
                    item.layout().deleteLater()
        clear(self._results_layout)
        self._cancel_btn = None
        self._save_btn = None
        self._stats = None

    def _reveal_results(self) -> None:
        from gui.themes import motion

        if self._results_widget.isVisible():
            motion.reveal(self._results_widget, duration="fast")

    def _show_placeholder(self, title: str = "", text: str = "") -> None:
        self._clear_results()
        empty = EmptyState(
            title or "No analysis yet",
            text or ("Run Quick Start, or your own settings below. The clusters, how "
                     "well they separate and a map to explore appear here."),
        )
        empty.setObjectName("soundMapEmptyState")
        self._results_layout.addWidget(empty)

    def _show_progress_panel(self) -> None:
        """Show clustering progress panel."""
        self._clear_results()
        head = QtWidgets.QHBoxLayout()
        title = self._make_card_title("Analysing your library")
        head.addWidget(title, 1)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        set_glyph(self._cancel_btn, "close")
        self._cancel_btn.setToolTip("Stop after the files being analysed now; nothing is written")
        self._cancel_btn.clicked.connect(self._on_cancel_clustering)
        head.addWidget(self._cancel_btn)
        self._results_layout.addLayout(head)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._results_layout.addWidget(self._progress_bar)

        self._progress_status = QtWidgets.QLabel("Initializing...")
        self._progress_status.setObjectName("mutedLabel")
        apply_font(self._progress_status, "label")
        self._results_layout.addWidget(self._progress_status)

        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setObjectName("logText")
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(140)
        self._log_area.setMaximumHeight(200)
        self._results_layout.addWidget(self._log_area)
        self._reveal_results()

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        if self.sender() is not self._worker:
            return
        self._progress_bar.setValue(percent)
        self._progress_status.setText(message)

    @Slot(str)
    def _on_log_line(self, line: str) -> None:
        if self.sender() is not self._worker:
            return
        self._log_area.appendPlainText(line)

    def _on_cancel_clustering(self) -> None:
        """Cancel clustering (cooperative: stops at the next file)."""
        if self._worker is not None:
            self._worker.cancel()
            self._progress_status.setText("Cancelling… (finishing the files being analysed)")
            if self._cancel_btn is not None:
                self._cancel_btn.setEnabled(False)
            self._status("Cancelling…", "busy")

    @Slot(bool, str, dict)
    def _on_clustering_finished(self, success: bool, message: str, result: dict) -> None:
        """Handle clustering completion (never blocks: no wait() here)."""
        if self.sender() is not self._worker:
            return  # a run that was already abandoned
        self._worker = None
        self._set_busy(False)
        self._clear_results()

        if success:
            self._set_result(result)
            self._show_results_panel(result)
            self._status("Clustering complete", "ok")
            self._log("Clustering complete — open the map to explore it.", "ok")
        elif message == "Cancelled":
            self._show_placeholder("Analysis cancelled",
                                   "Clustering was cancelled. No cluster data was written.")
            self._reveal_results()
            self._status("Cancelled", "neutral")
            return
        else:
            head = QtWidgets.QHBoxLayout()
            icon = QtWidgets.QLabel()
            icon.setPixmap(glyph_icon("warning", get_manager().current.danger, 20).pixmap(20, 20))
            head.addWidget(icon, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            title = self._make_card_title("The analysis did not finish")
            head.addWidget(title, 1)
            self._results_layout.addLayout(head)
            error_label = QtWidgets.QLabel(f"Error: {message}")
            error_label.setObjectName("mutedLabel")
            error_label.setWordWrap(True)
            error_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self._results_layout.addWidget(error_label)
            self._reveal_results()
            self._status("Clustering failed", "error")
            self._log(message, "error")

        if success:
            # Last, once the results are on screen: the page may offer the
            # Save cluster playlists preview (wizard option, CP-44).
            self.analysis_succeeded.emit(result)

    def _show_results_panel(self, result: dict) -> None:
        """The latest analysis: a title, a StatRow and what to do next."""
        metrics = result.get("metrics", {}) or {}
        config = result.get("config") or {}
        method = str(config.get("algorithm") or (result.get("run") or {}).get("method") or "")
        bits = [{"kmeans": "K-Means", "hdbscan": "HDBSCAN"}.get(method.lower(), method)] if method else []
        feats = config.get("features")
        if isinstance(feats, dict) and any(feats.values()):
            bits.append("listening for " + features_text(feats))
        self._card_header(self._results_layout, "Latest analysis", " · ".join(bits))

        tiles: list[tuple[str, str]] = []
        values: dict[str, tuple[str, str]] = {}
        if "labels" in result:
            import numpy as np
            labels = np.asarray(result["labels"])
            n_clusters = len(np.unique(labels[labels >= 0]))
            noise = int((labels < 0).sum())
            tiles += [("clusters", "Clusters"), ("tracks", "Tracks")]
            values["clusters"] = (str(n_clusters), "")
            values["tracks"] = (f"{len(labels):,}", "")
            if noise:
                tiles.append(("noise", "Unclustered"))
                values["noise"] = (f"{noise:,}", "")
                self._noise_note = f"{noise / max(len(labels), 1):.0%} of the tracks"
        failed = result.get("failed") or {}
        if failed:
            tiles.append(("failed", "Not analysed"))
            values["failed"] = (f"{len(failed):,}", "warning")
        if "silhouette_score" in metrics:
            sil = float(metrics["silhouette_score"])
            quality = "Good" if sil > 0.5 else "Fair" if sil > 0.3 else "Poor"
            tone = "success" if sil > 0.5 else "warning" if sil > 0.3 else "danger"
            tiles.append(("silhouette", f"Silhouette · {quality}"))
            values["silhouette"] = (f"{sil:.2f}", tone)
        if tiles:
            self._stats = StatRow(tiles)
            for key, (value, tone) in values.items():
                tile = self._stats.tile(key)
                tile.set_value(value, tone=tone)
            if "noise" in values:
                self._stats.tile("noise").setToolTip(
                    f"Tracks HDBSCAN put in no cluster ({self._noise_note})")
            if "failed" in values:
                self._stats.tile("failed").setToolTip(
                    "Tracks that could not be analysed: left out, retried next run")
            if "silhouette" in values:
                self._stats.tile("silhouette").setToolTip(
                    "Silhouette score (−1 to 1): how clearly the clusters are apart")
            self._results_layout.addSpacing(4)
            self._results_layout.addWidget(self._stats)

        hint = QtWidgets.QLabel(
            "Save a playlist per cluster with “Save cluster playlists…”, "
            "or export a lasso selection from the map."
        )
        hint.setObjectName("mutedLabel")
        apply_font(hint, "caption")
        hint.setWordWrap(True)
        self._results_layout.addWidget(hint)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setSpacing(8)
        graph_btn = QtWidgets.QPushButton("Open the map")
        graph_btn.setObjectName("primaryBtn")
        set_glyph(graph_btn, "chevron_right", role="inverse")
        graph_btn.setToolTip("Explore the clusters on the Map")
        graph_btn.clicked.connect(self._open_graph)
        buttons.addWidget(graph_btn)

        save_btn = QtWidgets.QPushButton("Save cluster playlists…")
        set_glyph(save_btn, "save")
        save_btn.clicked.connect(self.save_playlists_requested.emit)
        self._save_btn = save_btn
        self._apply_save_state()
        buttons.addWidget(save_btn)

        if "silhouette_score" in metrics:
            report_btn = QtWidgets.QPushButton("Quality report")
            report_btn.setFlat(True)
            set_glyph(report_btn, "info")
            report_btn.setToolTip("How well the analysis separated your tracks")
            report_btn.clicked.connect(lambda: self._show_quality_report(result))
            buttons.addWidget(report_btn)

        buttons.addStretch(1)
        self._results_layout.addSpacing(2)
        self._results_layout.addLayout(buttons)
        self._reveal_results()

    def stat_values(self) -> dict[str, str]:
        """The results StatRow as {key: value} (empty before a result)."""
        if self._stats is None:
            return {}
        return {key: self._stats.tile(key).value() for key in self._stats.keys()}

    # ── Save cluster playlists (mirrors the page's flow button) ───────────

    def set_save_playlists_state(self, enabled: bool, tooltip: str = "") -> None:
        """The page keeps the results card's Save button in step with the
        Map's flow button (enabled state and tooltip)."""
        self._save_state = (bool(enabled), tooltip)
        self._apply_save_state()

    def _apply_save_state(self) -> None:
        if self._save_btn is None:
            return
        enabled, tip = self._save_state
        self._save_btn.setEnabled(enabled)
        self._save_btn.setToolTip(tip)

    # ── Latest result (read by the Sound Map page's actions) ──────────────

    def _set_result(self, result: dict | None) -> None:
        self._current_result = result
        self.result_changed.emit()

    def quality_report_available(self) -> bool:
        """True when the latest run has scores to report."""
        result = self._current_result or {}
        return "silhouette_score" in (result.get("metrics") or {})

    def show_quality_report(self) -> None:
        """Show the latest run's quality report (no-op without one)."""
        if self.quality_report_available():
            self._show_quality_report(self._current_result)

    def _show_quality_report(self, result: dict) -> None:
        """Show cluster quality report dialog."""
        metrics = result.get("metrics", {})
        cluster_info = result.get("cluster_info", {})

        dialog = ClusterQualityReportDialog(metrics, cluster_info, self)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def _open_graph(self) -> None:
        """Switch to the Sound Map's Map section (the main window routes the
        old "graph" key there)."""
        self._log("Opening the map…", "info")
        self.navigate_requested.emit("graph")

    # ── Library changes ───────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        """Results belong to the library they were computed for."""
        if self._worker is not None:
            # Stop the run cooperatively; its result is ignored when it lands.
            self._worker.cancel()
            self._worker = None
            self._set_busy(False)
            self._log("Library changed — the running clustering job was cancelled.", "warn")
            self._status("Cancelled", "neutral")
        self._set_result(None)
        self._show_placeholder()
