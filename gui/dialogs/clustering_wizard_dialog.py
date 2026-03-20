"""Multi-step clustering configuration wizard dialog."""
from __future__ import annotations

from typing import Dict, List

from gui.compat import QtCore, QtGui, QtWidgets


class ClusteringWizardDialog(QtWidgets.QDialog):
    """Multi-step wizard for clustering configuration.

    Steps:
    1. Feature Selection (which features to use)
    2. Normalization & Preprocessing (scaling, reduction)
    3. Algorithm Selection (K-Means vs HDBSCAN)
    4. Post-Processing (merge small clusters, remove outliers)
    5. Output Options (playlists, visualization, report)
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
            "eps": 0.5,
            "remove_small_clusters": True,
            "min_cluster_threshold": 5,
            "merge_into_misc": True,
            "create_playlists": True,
            "generate_report": True,
            "open_graph": True,
        }

        self._build_ui()
        self._current_step = 0
        self._show_step(0)

    def _build_ui(self) -> None:
        """Build the wizard UI."""
        layout = QtWidgets.QVBoxLayout(self)

        # Title and step indicator
        title_layout = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("Step 1: Feature Selection")
        self._title.setStyleSheet("font-weight: bold; font-size: 14px;")
        title_layout.addWidget(self._title)

        self._step_indicator = QtWidgets.QLabel("1 of 5")
        self._step_indicator.setStyleSheet("color: #666; font-size: 12px;")
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
        self.setLayout(layout)

    def _create_page_features(self) -> QtWidgets.QWidget:
        """Create feature selection page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        # Intro
        intro = QtWidgets.QLabel(
            "Select which audio features to use for clustering.\n"
            "More features = more detailed clustering, longer computation.\n"
            "Each feature captures different aspects of the sound."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Features as checkboxes
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

        # Preset buttons
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
        """Create normalization & preprocessing page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        # Intro
        intro = QtWidgets.QLabel(
            "Choose how to normalize features and reduce dimensions.\n"
            "Normalization ensures all features contribute equally.\n"
            "Dimensionality reduction can improve visualization and speed."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Normalization
        layout.addWidget(QtWidgets.QLabel("Normalization Method:"))
        self._norm_combo = QtWidgets.QComboBox()
        norm_options = [
            ("Standard (Z-score)", "standard"),
            ("MinMax (0-1)", "minmax"),
            ("Robust (outlier-resistant)", "robust"),
        ]
        for display_text, value in norm_options:
            self._norm_combo.addItem(display_text, userData=value)
        self._norm_combo.setCurrentIndex(0)
        layout.addWidget(self._norm_combo)

        norm_info = QtWidgets.QLabel(
            "• Standard: Mean 0, std dev 1 (default, good for most cases)\n"
            "• MinMax: Values between 0 and 1 (good for bounded features)\n"
            "• Robust: Less affected by outliers (good for skewed data)"
        )
        norm_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(norm_info)

        # Dimensionality reduction
        layout.addWidget(QtWidgets.QLabel("Dimensionality Reduction:"))
        self._reduction_combo = QtWidgets.QComboBox()
        self._reduction_combo.addItems([
            "None (use all features)",
            "PCA (fast, linear)",
            "t-SNE (better visualization, slow)",
            "UMAP (balanced, if installed)",
        ])
        self._reduction_combo.setCurrentText("None (use all features)")
        layout.addWidget(self._reduction_combo)

        reduction_info = QtWidgets.QLabel(
            "• None: Cluster using all features (most accurate)\n"
            "• PCA: Fast linear reduction, good for many features\n"
            "• t-SNE: Creates visual separation, not for clustering\n"
            "• UMAP: Preserves both local and global structure"
        )
        reduction_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(reduction_info)

        layout.addStretch()
        return page

    def _create_page_algorithm(self) -> QtWidgets.QWidget:
        """Create algorithm selection page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Choose a clustering algorithm.\n"
            "K-Means requires specifying K (number of clusters).\n"
            "HDBSCAN automatically finds cluster count."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Algorithm selection
        layout.addWidget(QtWidgets.QLabel("Algorithm:"))
        self._algo_combo = QtWidgets.QComboBox()
        self._algo_combo.addItems(["K-Means", "HDBSCAN"])
        self._algo_combo.currentTextChanged.connect(self._on_algorithm_changed)
        layout.addWidget(self._algo_combo)

        algo_info = QtWidgets.QLabel(
            "K-Means: Fast, requires specifying cluster count. Best for genre/mood grouping.\n"
            "HDBSCAN: Density-based, finds clusters automatically. Better for complex data."
        )
        algo_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(algo_info)

        # K-Means parameters
        self._kmeans_widget = QtWidgets.QWidget()
        kmeans_layout = QtWidgets.QFormLayout(self._kmeans_widget)

        self._k_spinbox = QtWidgets.QSpinBox()
        self._k_spinbox.setRange(2, 100)
        self._k_spinbox.setValue(self.config["k"])
        self._k_spinbox.setToolTip("Number of clusters")
        kmeans_layout.addRow("K (clusters):", self._k_spinbox)

        self._k_info = QtWidgets.QLabel(
            "Tip: Start with 8-12 clusters and adjust based on results.\n"
            "More clusters = more specific groupings, less coherent."
        )
        self._k_info.setStyleSheet("color: #666; font-size: 10px;")
        self._k_info.setWordWrap(True)
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

        self._hdbscan_info = QtWidgets.QLabel(
            "Tip: Increase min_cluster_size to reduce outliers.\n"
            "Lower values = more clusters and outliers."
        )
        self._hdbscan_info.setStyleSheet("color: #666; font-size: 10px;")
        self._hdbscan_info.setWordWrap(True)
        hdbscan_layout.addRow("", self._hdbscan_info)

        self._hdbscan_widget.setVisible(False)
        layout.addWidget(self._hdbscan_widget)

        layout.addStretch()
        return page

    def _create_page_postprocessing(self) -> QtWidgets.QWidget:
        """Create post-processing options page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Configure post-processing options.\n"
            "These can improve cluster quality and usability."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._remove_small_cb = QtWidgets.QCheckBox("Remove clusters smaller than threshold")
        self._remove_small_cb.setChecked(self.config["remove_small_clusters"])
        layout.addWidget(self._remove_small_cb)

        self._min_threshold_spin = QtWidgets.QSpinBox()
        self._min_threshold_spin.setRange(1, 100)
        self._min_threshold_spin.setValue(self.config["min_cluster_threshold"])
        self._min_threshold_spin.setToolTip("Minimum tracks per cluster")
        layout.addWidget(QtWidgets.QLabel("Minimum cluster size (tracks):"))
        layout.addWidget(self._min_threshold_spin)

        layout.addWidget(QtWidgets.QLabel(""))  # Spacer

        self._merge_cb = QtWidgets.QCheckBox("Merge small clusters into 'Miscellaneous'")
        self._merge_cb.setChecked(self.config["merge_into_misc"])
        layout.addWidget(self._merge_cb)

        merge_info = QtWidgets.QLabel(
            "If enabled, small clusters are merged into a catch-all cluster.\n"
            "Otherwise, small clusters are simply removed."
        )
        merge_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(merge_info)

        layout.addStretch()
        return page

    def _create_page_output(self) -> QtWidgets.QWidget:
        """Create output options page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        intro = QtWidgets.QLabel(
            "Configure what to generate after clustering.\n"
            "You can always access these later."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._create_playlists_cb = QtWidgets.QCheckBox("Create M3U playlists for each cluster")
        self._create_playlists_cb.setChecked(self.config["create_playlists"])
        layout.addWidget(self._create_playlists_cb)

        self._generate_report_cb = QtWidgets.QCheckBox("Generate cluster quality report")
        self._generate_report_cb.setChecked(self.config["generate_report"])
        layout.addWidget(self._generate_report_cb)

        self._open_graph_cb = QtWidgets.QCheckBox("Open interactive graph after clustering")
        self._open_graph_cb.setChecked(self.config["open_graph"])
        layout.addWidget(self._open_graph_cb)

        layout.addWidget(QtWidgets.QLabel(""))  # Spacer

        playlist_info = QtWidgets.QLabel(
            "• Playlists: Created in Music/Playlists/ folder\n"
            "• Report: Shows cluster quality metrics and suggestions\n"
            "• Graph: Interactive visualization to explore clusters"
        )
        playlist_info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(playlist_info)

        layout.addStretch()
        return page

    def _on_algorithm_changed(self, algo_name: str) -> None:
        """Handle algorithm change."""
        is_kmeans = algo_name == "K-Means"
        self._kmeans_widget.setVisible(is_kmeans)
        self._hdbscan_widget.setVisible(not is_kmeans)

    def _apply_feature_preset(self, preset_config: dict) -> None:
        """Apply a feature preset."""
        for feat, cb in self._feature_checkboxes.items():
            cb.setChecked(preset_config.get(feat, False))

    def _show_step(self, step: int) -> None:
        """Show a specific step."""
        self._current_step = step
        self._stack.setCurrentIndex(step)
        self._progress_bar.setValue(step + 1)
        self._step_indicator.setText(f"{step + 1} of 5")

        titles = [
            "Step 1: Feature Selection",
            "Step 2: Normalization & Preprocessing",
            "Step 3: Algorithm Selection",
            "Step 4: Post-Processing",
            "Step 5: Output Options",
        ]
        self._title.setText(titles[step])

        # Update buttons
        self._prev_btn.setEnabled(step > 0)
        self._next_btn.setText("Finish" if step == 4 else "Next >")

        # Save config from current page
        self._save_current_step()

    def _save_current_step(self) -> None:
        """Save configuration from current step."""
        if self._current_step == 0:  # Features
            for feat, cb in self._feature_checkboxes.items():
                self.config["features"][feat] = cb.isChecked()

            # Validate at least one feature is selected
            selected_features = [k for k, v in self.config["features"].items() if v]
            if not selected_features:
                # Re-check at least one feature (tempo as default)
                self.config["features"]["tempo"] = True
                self._feature_checkboxes["tempo"].setChecked(True)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Selection",
                    "At least one feature must be selected. Re-enabled tempo."
                )
        elif self._current_step == 1:  # Normalization
            # Get normalization value from combo box userData
            norm_value = self._norm_combo.currentData()
            self.config["normalization"] = norm_value if norm_value else "standard"
            self.config["dimensionality_reduction"] = self._reduction_combo.currentText().split()[0].lower()
        elif self._current_step == 2:  # Algorithm
            self.config["algorithm"] = self._algo_combo.currentText().lower()
            self.config["k"] = self._k_spinbox.value()
            self.config["min_cluster_size"] = self._min_size_spinbox.value()
            self.config["min_samples"] = self._min_samples_spinbox.value()
        elif self._current_step == 3:  # Post-processing
            self.config["remove_small_clusters"] = self._remove_small_cb.isChecked()
            self.config["min_cluster_threshold"] = self._min_threshold_spin.value()
            self.config["merge_into_misc"] = self._merge_cb.isChecked()
        elif self._current_step == 4:  # Output
            self.config["create_playlists"] = self._create_playlists_cb.isChecked()
            self.config["generate_report"] = self._generate_report_cb.isChecked()
            self.config["open_graph"] = self._open_graph_cb.isChecked()

    def _on_prev(self) -> None:
        """Go to previous step."""
        if self._current_step > 0:
            self._show_step(self._current_step - 1)

    def _on_next(self) -> None:
        """Go to next step or accept."""
        if self._current_step < 4:
            self._show_step(self._current_step + 1)
        else:
            self._save_current_step()
            self.accept()

    def get_config(self) -> Dict:
        """Get the final configuration."""
        self._save_current_step()
        return self.config
