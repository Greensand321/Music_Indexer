"""Multi-step clustering configuration wizard dialog."""
from __future__ import annotations

from typing import Dict

from gui.compat import QtCore, QtGui, QtWidgets

#: Shown on the controls the clustering engine does not implement yet. They
#: stay visible (the redesign decides their fate) but cannot be changed, so the
#: wizard no longer promises behaviour that never happens.
_NOT_AVAILABLE = "Not available yet — the clustering engine does not do this."


class ClusteringWizardDialog(QtWidgets.QDialog):
    """Multi-step wizard for clustering configuration.

    Steps:
    1. Feature Selection (which features to use)
    2. Normalization (scaling)
    3. Algorithm Selection (K-Means vs HDBSCAN)
    4. Post-Processing (not implemented by the engine yet; shown disabled)
    5. Output (what a run produces)

    Every page's controls are read when the user moves *off* the page and
    again in :meth:`get_config`, so no choice is lost whichever way the user
    navigates.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Clustering Configuration Wizard")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        # Configuration storage
        self.config = {
            "features": {
                "tempo": True,
                "mfcc": True,
                "chroma": True,
                "spectral": True,
                "energy": True,
                "onset_rate": False,
            },
            "normalization": "standard",
            "dimensionality_reduction": "none",
            "algorithm": "kmeans",
            "k": 8,
            "min_cluster_size": 5,
            "min_samples": 5,
            # Not implemented by the engine; recorded as what really happens.
            "remove_small_clusters": False,
            "min_cluster_threshold": 5,
            "merge_into_misc": False,
            "create_playlists": False,
            "generate_report": True,
            "open_graph": False,
        }

        self._current_step = 0
        self._build_ui()
        self._show_step(0)

    def _build_ui(self) -> None:
        """Build the wizard UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Title and step indicator
        title_layout = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("Step 1: Feature Selection")
        self._title.setObjectName("cardTitle")
        title_layout.addWidget(self._title)

        self._step_indicator = QtWidgets.QLabel("1 of 5")
        self._step_indicator.setObjectName("statusHint")
        title_layout.addStretch()
        title_layout.addWidget(self._step_indicator)
        layout.addLayout(title_layout)

        # Progress bar
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 5)
        self._progress_bar.setValue(1)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMaximumHeight(4)
        layout.addWidget(self._progress_bar)

        # Stack of pages
        self._stack = QtWidgets.QStackedWidget()
        self._pages = [
            self._create_page_features(),
            self._create_page_normalization(),
            self._create_page_algorithm(),
            self._create_page_postprocessing(),
            self._create_page_output(),
        ]

        for page in self._pages:
            self._stack.addWidget(page)

        layout.addWidget(self._stack, 1)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self._prev_btn = QtWidgets.QPushButton("< Back")
        self._prev_btn.clicked.connect(self._on_prev)
        button_layout.addWidget(self._prev_btn)

        self._next_btn = QtWidgets.QPushButton("Next >")
        self._next_btn.clicked.connect(self._on_next)
        button_layout.addWidget(self._next_btn)

        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)

        layout.addLayout(button_layout)

    @staticmethod
    def _hint(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("notesHint")
        label.setWordWrap(True)
        return label

    def _create_page_features(self) -> QtWidgets.QWidget:
        """Create feature selection page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Select which audio features to use for clustering.\n"
            "More features = more detailed clustering, longer computation.\n"
            "Each feature captures different aspects of the sound."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        features_info = {
            "tempo": "Beats Per Minute (BPM) - rhythm tempo",
            "mfcc": "Mel-Frequency Cepstral Coefficients - timbre/tone color",
            "chroma": "Chroma features - harmonic content / key",
            "spectral": "Spectral centroid - brightness of sound",
            "energy": "Energy - loudness characteristics",
            "onset_rate": "Onset rate - percussion/attack density (slow to compute)",
        }

        self._feature_checkboxes = {}
        for feat, info in features_info.items():
            cb = QtWidgets.QCheckBox(feat.replace("_", " ").title())
            cb.setChecked(self.config["features"][feat])
            cb.setToolTip(info)
            self._feature_checkboxes[feat] = cb
            layout.addWidget(cb)

        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Presets:"))

        for preset_name, preset_config in [
            ("Fast", {"tempo": True, "mfcc": False, "chroma": False, "spectral": False, "energy": True, "onset_rate": False}),
            ("Balanced", {"tempo": True, "mfcc": True, "chroma": False, "spectral": False, "energy": True, "onset_rate": False}),
            ("Complete", {"tempo": True, "mfcc": True, "chroma": True, "spectral": True, "energy": True, "onset_rate": False}),
        ]:
            btn = QtWidgets.QPushButton(preset_name)
            btn.clicked.connect(lambda checked, cfg=preset_config: self._apply_feature_preset(cfg))
            preset_layout.addWidget(btn)

        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        layout.addStretch()
        return page

    def _create_page_normalization(self) -> QtWidgets.QWidget:
        """Create normalization page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Choose how to normalize features.\n"
            "Normalization ensures all features contribute equally."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(QtWidgets.QLabel("Normalization Method:"))
        self._norm_combo = QtWidgets.QComboBox()
        for display_text, value in [
            ("Standard (Z-score)", "standard"),
            ("MinMax (0-1)", "minmax"),
            ("Robust (outlier-resistant)", "robust"),
        ]:
            self._norm_combo.addItem(display_text, userData=value)
        self._norm_combo.setCurrentIndex(0)
        layout.addWidget(self._norm_combo)

        layout.addWidget(self._hint(
            "• Standard: Mean 0, std dev 1 (default, good for most cases)\n"
            "• MinMax: Values between 0 and 1 (good for bounded features)\n"
            "• Robust: Less affected by outliers (good for skewed data)"
        ))

        # Dimensionality reduction before clustering is not implemented: the
        # engine always clusters on every selected feature. (The 2-D/3-D maps
        # use UMAP or t-SNE automatically, for display only.)
        layout.addWidget(QtWidgets.QLabel("Dimensionality Reduction:"))
        self._reduction_combo = QtWidgets.QComboBox()
        self._reduction_combo.addItems([
            "None (use all features)",
            "PCA (fast, linear)",
            "t-SNE (better visualization, slow)",
            "UMAP (balanced, if installed)",
        ])
        self._reduction_combo.setCurrentIndex(0)
        self._reduction_combo.setEnabled(False)
        self._reduction_combo.setToolTip(_NOT_AVAILABLE)
        layout.addWidget(self._reduction_combo)
        layout.addWidget(self._hint(
            "Clustering always uses all selected features. The map views reduce "
            "them to 2-D/3-D automatically (UMAP if installed, otherwise t-SNE), "
            "for display only."
        ))

        layout.addStretch()
        return page

    def _create_page_algorithm(self) -> QtWidgets.QWidget:
        """Create algorithm selection page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Choose a clustering algorithm.\n"
            "K-Means requires specifying K (number of clusters).\n"
            "HDBSCAN automatically finds cluster count (needs the optional "
            "'hdbscan' package)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(QtWidgets.QLabel("Algorithm:"))
        self._algo_combo = QtWidgets.QComboBox()
        # userData is the engine's method name; the text is only for display
        # ("K-Means".lower() is "k-means", which the engine did not accept).
        self._algo_combo.addItem("K-Means", userData="kmeans")
        self._algo_combo.addItem("HDBSCAN", userData="hdbscan")
        self._algo_combo.currentIndexChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self._algo_combo)

        layout.addWidget(self._hint(
            "K-Means: Fast, requires specifying cluster count. Best for genre/mood grouping.\n"
            "HDBSCAN: Density-based, finds clusters automatically. Better for complex data."
        ))

        # K-Means parameters
        self._kmeans_widget = QtWidgets.QWidget()
        kmeans_layout = QtWidgets.QFormLayout(self._kmeans_widget)

        self._k_spinbox = QtWidgets.QSpinBox()
        self._k_spinbox.setRange(2, 100)
        self._k_spinbox.setValue(self.config["k"])
        self._k_spinbox.setToolTip("Number of clusters")
        kmeans_layout.addRow("K (clusters):", self._k_spinbox)

        self._k_info = self._hint(
            "Tip: Start with 8-12 clusters and adjust based on results.\n"
            "More clusters = more specific groupings, less coherent."
        )
        kmeans_layout.addRow("", self._k_info)

        layout.addWidget(self._kmeans_widget)

        # HDBSCAN parameters
        self._hdbscan_widget = QtWidgets.QWidget()
        hdbscan_layout = QtWidgets.QFormLayout(self._hdbscan_widget)

        self._min_size_spinbox = QtWidgets.QSpinBox()
        self._min_size_spinbox.setRange(2, 100)
        self._min_size_spinbox.setValue(self.config["min_cluster_size"])
        self._min_size_spinbox.setToolTip("Minimum tracks per cluster")
        hdbscan_layout.addRow("Min cluster size:", self._min_size_spinbox)

        self._min_samples_spinbox = QtWidgets.QSpinBox()
        self._min_samples_spinbox.setRange(1, 50)
        self._min_samples_spinbox.setValue(self.config["min_samples"])
        self._min_samples_spinbox.setToolTip("Minimum samples in neighborhood")
        hdbscan_layout.addRow("Min samples:", self._min_samples_spinbox)

        self._hdbscan_info = self._hint(
            "Tip: Increase min_cluster_size to reduce outliers.\n"
            "Lower values = more clusters and outliers."
        )
        hdbscan_layout.addRow("", self._hdbscan_info)

        self._hdbscan_widget.setVisible(False)
        layout.addWidget(self._hdbscan_widget)

        layout.addStretch()
        return page

    def _create_page_postprocessing(self) -> QtWidgets.QWidget:
        """Create post-processing options page (not implemented by the engine)."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Post-processing is not available yet: every cluster the algorithm finds "
            "is kept as it is. With HDBSCAN, tracks that fit no cluster are shown as "
            "“Unclustered”."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._remove_small_cb = QtWidgets.QCheckBox("Remove clusters smaller than threshold")
        self._remove_small_cb.setChecked(self.config["remove_small_clusters"])
        layout.addWidget(self._remove_small_cb)

        self._min_threshold_spin = QtWidgets.QSpinBox()
        self._min_threshold_spin.setRange(1, 100)
        self._min_threshold_spin.setValue(self.config["min_cluster_threshold"])
        layout.addWidget(QtWidgets.QLabel("Minimum cluster size (tracks):"))
        layout.addWidget(self._min_threshold_spin)

        self._merge_cb = QtWidgets.QCheckBox("Merge small clusters into 'Miscellaneous'")
        self._merge_cb.setChecked(self.config["merge_into_misc"])
        layout.addWidget(self._merge_cb)

        for w in (self._remove_small_cb, self._min_threshold_spin, self._merge_cb):
            w.setEnabled(False)
            w.setToolTip(_NOT_AVAILABLE)

        layout.addStretch()
        return page

    def _create_page_output(self) -> QtWidgets.QWidget:
        """Create output page: what a run actually produces."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "A run saves the cluster data used by the Map and the 3D view "
            "(in the library's Docs/ folder) and, when the scores can be computed, "
            "offers a quality report. Nothing in your library is modified."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._create_playlists_cb = QtWidgets.QCheckBox("Create M3U playlists for each cluster")
        self._create_playlists_cb.setChecked(False)
        self._generate_report_cb = QtWidgets.QCheckBox("Generate cluster quality report")
        self._generate_report_cb.setChecked(True)
        self._open_graph_cb = QtWidgets.QCheckBox("Open interactive graph after clustering")
        self._open_graph_cb.setChecked(False)
        for cb in (self._create_playlists_cb, self._generate_report_cb, self._open_graph_cb):
            cb.setEnabled(False)
            layout.addWidget(cb)
        self._create_playlists_cb.setToolTip(
            "Not available yet. Export a cluster from the Map instead "
            "(click it in the legend, then “Export playlist…”)."
        )
        self._generate_report_cb.setToolTip("Always offered when the scores can be computed.")
        self._open_graph_cb.setToolTip("Use “Open the map” on the Results tab.")

        layout.addWidget(self._hint(
            "• Playlists: export any cluster or selection from the Map\n"
            "• Report: “View Quality Report” on the Results tab\n"
            "• Map: “Open the map” on the Results tab"
        ))

        layout.addStretch()
        return page

    def _on_algorithm_changed(self, _index: int = 0) -> None:
        """Show the parameters of the selected algorithm."""
        is_kmeans = self._algo_combo.currentData() == "kmeans"
        self._kmeans_widget.setVisible(is_kmeans)
        self._hdbscan_widget.setVisible(not is_kmeans)

    def _apply_feature_preset(self, preset_config: dict) -> None:
        """Apply a feature preset."""
        for feat, cb in self._feature_checkboxes.items():
            cb.setChecked(preset_config.get(feat, False))

    def _show_step(self, step: int) -> None:
        """Show a specific step (callers save the page being left first)."""
        self._current_step = step
        self._stack.setCurrentIndex(step)
        self._progress_bar.setValue(step + 1)
        self._step_indicator.setText(f"{step + 1} of 5")

        titles = [
            "Step 1: Feature Selection",
            "Step 2: Normalization",
            "Step 3: Algorithm Selection",
            "Step 4: Post-Processing",
            "Step 5: Output",
        ]
        self._title.setText(titles[step])

        self._prev_btn.setEnabled(step > 0)
        self._next_btn.setText("Finish" if step == 4 else "Next >")

    def _validate_step(self, step: int) -> bool:
        """Return False (after telling the user) if ``step`` cannot be left."""
        if step == 0 and not any(cb.isChecked() for cb in self._feature_checkboxes.values()):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Selection",
                "Select at least one audio feature to cluster on.",
            )
            return False
        return True

    def _save_step(self, step: int) -> None:
        """Copy page ``step``'s widgets into :attr:`config`."""
        if step == 0:  # Features
            for feat, cb in self._feature_checkboxes.items():
                self.config["features"][feat] = cb.isChecked()
        elif step == 1:  # Normalization
            norm_value = self._norm_combo.currentData()
            self.config["normalization"] = norm_value if norm_value else "standard"
            self.config["dimensionality_reduction"] = "none"
        elif step == 2:  # Algorithm
            self.config["algorithm"] = self._algo_combo.currentData() or "kmeans"
            self.config["k"] = self._k_spinbox.value()
            self.config["min_cluster_size"] = self._min_size_spinbox.value()
            self.config["min_samples"] = self._min_samples_spinbox.value()
        elif step == 3:  # Post-processing (disabled controls)
            self.config["remove_small_clusters"] = self._remove_small_cb.isChecked()
            self.config["min_cluster_threshold"] = self._min_threshold_spin.value()
            self.config["merge_into_misc"] = self._merge_cb.isChecked()
        elif step == 4:  # Output (disabled controls)
            self.config["create_playlists"] = self._create_playlists_cb.isChecked()
            self.config["generate_report"] = self._generate_report_cb.isChecked()
            self.config["open_graph"] = self._open_graph_cb.isChecked()

    def _save_current_step(self) -> None:
        """Backwards-compatible alias: save the page currently shown."""
        self._save_step(self._current_step)

    def _save_all(self) -> None:
        for step in range(len(self._pages)):
            self._save_step(step)

    def _on_prev(self) -> None:
        """Go to previous step (keeping what was entered on this one)."""
        if self._current_step > 0:
            self._save_step(self._current_step)
            self._show_step(self._current_step - 1)

    def _on_next(self) -> None:
        """Go to next step or accept."""
        if not self._validate_step(self._current_step):
            return
        self._save_step(self._current_step)
        if self._current_step < 4:
            self._show_step(self._current_step + 1)
        else:
            if not self._validate_step(0):
                self._show_step(0)
                return
            self._save_all()
            self.accept()

    def get_config(self) -> Dict:
        """Get the final configuration (read from every page's widgets)."""
        self._save_all()
        return self.config
