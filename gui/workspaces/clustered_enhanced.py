"""Enhanced Clustered Playlists workspace with wizard and interactive results."""
from __future__ import annotations

import os
import logging
import threading
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase
from gui.dialogs.clustering_wizard_dialog import ClusteringWizardDialog
from gui.dialogs.cluster_quality_report_dialog import ClusterQualityReportDialog

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"}


class ClusterWorker(QtCore.QThread):
    """Background worker for clustering operation."""

    log_line = Signal(str)
    progress = Signal(int, str)  # percent, message
    finished = Signal(bool, str, dict)  # success, message, results

    def __init__(self, library_path: str, config: dict) -> None:
        super().__init__()
        self.library_path = library_path
        self.config = config
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        """Run clustering with configuration."""
        try:
            from clustered_playlists import generate_clustered_playlists
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}", {})
            return

        try:
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.log_line.emit(msg)

            # Collect audio files
            self.progress.emit(0, "Scanning library...")
            tracks = []
            for dirpath, _, files in os.walk(self.library_path):
                if self._cancelled:
                    self.finished.emit(False, "Cancelled", {})
                    return
                for f in files:
                    if os.path.splitext(f)[1].lower() in _AUDIO_EXTS:
                        tracks.append(os.path.join(dirpath, f))

            if not tracks:
                self.finished.emit(False, "No audio files found", {})
                return

            _log(f"Found {len(tracks)} tracks")
            self.progress.emit(5, f"Scanning complete: {len(tracks)} tracks")

            # Prepare parameters with validation
            algorithm = self.config.get("algorithm", "kmeans")
            if algorithm == "kmeans":
                k = self.config.get("k", 8)
                # Validate K against track count
                if k > len(tracks):
                    _log(f"⚠ K={k} exceeds track count ({len(tracks)}); reducing to {len(tracks)}")
                    k = len(tracks)
                params = {"n_clusters": k}
            else:
                min_size = self.config.get("min_cluster_size", 5)
                # Validate HDBSCAN min_cluster_size
                if min_size > len(tracks):
                    _log(f"⚠ min_cluster_size={min_size} exceeds track count ({len(tracks)}); reducing to {len(tracks)}")
                    min_size = len(tracks)
                params = {
                    "min_cluster_size": min_size,
                    "min_samples": self.config.get("min_samples", 5),
                }

            features = [k for k, v in self.config.get("features", {}).items() if v]
            if not features:
                # Use all features if none specified
                features = ["tempo", "mfcc", "chroma", "spectral", "energy"]

            _log(f"Using algorithm: {algorithm.upper()}")
            _log(f"Using features: {', '.join(features)}")

            # Run clustering
            self.progress.emit(10, "Extracting audio features...")
            result = generate_clustered_playlists(
                tracks,
                self.library_path,
                method=algorithm,
                params=params,
                log_callback=_log,
            )

            if self._cancelled:
                self.finished.emit(False, "Cancelled", {})
                return

            self.progress.emit(90, "Computing cluster metrics...")

            # Compute quality metrics
            metrics = {}
            if "labels" in result and "X" in result:
                try:
                    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
                    X = result["X"]
                    labels = result["labels"]

                    metrics["silhouette_score"] = silhouette_score(X, labels)
                    metrics["davies_bouldin_index"] = davies_bouldin_score(X, labels)
                    metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)

                    _log(f"Silhouette score: {metrics['silhouette_score']:.3f}")
                except Exception as e:
                    logger.warning(f"Could not compute metrics: {e}", exc_info=True)
                    _log(f"Could not compute metrics: {e}")

            self.progress.emit(95, "Finalizing results...")

            result["metrics"] = metrics
            result["config"] = self.config

            if not self._cancelled:
                self.progress.emit(100, "Complete")
                self.finished.emit(True, "Clustering complete", result)
            else:
                self.finished.emit(False, "Cancelled", {})

        except Exception as exc:
            # Log full traceback for debugging
            logger.exception("Clustering failed with exception")
            self.finished.emit(False, f"Clustering failed: {exc}", {})


class EnhancedClusteredWorkspace(WorkspaceBase):
    """Enhanced workspace with wizard and interactive results."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: ClusterWorker | None = None
        self._current_config: dict | None = None
        self._current_result: dict | None = None
        self._build_ui()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Clean up resources when workspace is closed."""
        self._cleanup_worker()
        super().closeEvent(event)

    def _cleanup_worker(self) -> None:
        """Stop and clean up the worker thread."""
        if self._worker is not None:
            # Signal the worker to stop
            self._worker.cancel()
            # Wait for thread to finish (with timeout)
            if not self._worker.wait(5000):  # 5 second timeout
                logger.warning("Worker thread did not finish within timeout, terminating")
                self._worker.terminate()
                self._worker.wait()
            self._worker = None

    def _build_ui(self) -> None:
        """Build the UI."""
        cl = self.content_layout

        # Title and intro
        cl.addWidget(self._make_section_title("Clustered Playlists"))
        cl.addWidget(self._make_subtitle(
            "Extract audio features (tempo, timbre, harmony) and group similar tracks using "
            "K-Means or HDBSCAN. Create cluster-based playlists and explore results visually."
        ))

        # ── Config Tabs ────────────────────────────────────────────────────────
        self._tab_widget = QtWidgets.QTabWidget()

        # Quick start tab
        self._tab_widget.addTab(self._build_quick_start_tab(), "🚀 Quick Start")

        # Advanced tab
        self._tab_widget.addTab(self._build_advanced_tab(), "⚙ Advanced")

        # Results tab
        self._results_widget = QtWidgets.QWidget()
        self._results_layout = QtWidgets.QVBoxLayout(self._results_widget)
        self._tab_widget.addTab(self._results_widget, "📊 Results")

        cl.addWidget(self._tab_widget)

        cl.addStretch(1)

    def _build_quick_start_tab(self) -> QtWidgets.QWidget:
        """Build quick start tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Info box
        info_card = self._make_card()
        info_layout = QtWidgets.QVBoxLayout(info_card)
        info_text = QtWidgets.QLabel(
            "Run clustering with recommended settings:\n"
            "• 8 clusters using K-Means\n"
            "• All features (tempo, timbre, harmony, energy)\n"
            "• Standard normalization\n\n"
            "Want more control? Use the Advanced tab."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_card)

        # Button
        run_btn = self._make_primary_button("📊  Run Quick Start")
        run_btn.clicked.connect(self._on_run_quick_start)
        layout.addWidget(run_btn)

        layout.addStretch(1)
        return tab

    def _build_advanced_tab(self) -> QtWidgets.QWidget:
        """Build advanced configuration tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Info
        info_label = QtWidgets.QLabel(
            "Configure every aspect of the clustering process:"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Configuration display
        self._config_display = QtWidgets.QPlainTextEdit()
        self._config_display.setReadOnly(True)
        self._config_display.setPlaceholderText("Configuration will appear here")
        self._config_display.setMinimumHeight(150)
        layout.addWidget(self._config_display)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        configure_btn = self._make_primary_button("⚙ Configure with Wizard...")
        configure_btn.clicked.connect(self._on_configure_wizard)
        button_layout.addWidget(configure_btn)

        run_btn = QtWidgets.QPushButton("▶ Run")
        run_btn.clicked.connect(self._on_run_advanced)
        button_layout.addWidget(run_btn)

        button_layout.addStretch(1)
        layout.addLayout(button_layout)

        layout.addStretch(1)
        return tab

    def _on_run_quick_start(self) -> None:
        """Run with quick start configuration."""
        config = {
            "features": {
                "tempo": True,
                "mfcc": True,
                "chroma": False,
                "spectral": False,
                "energy": True,
                "onset_rate": False,
            },
            "normalization": "standard",
            "dimensionality_reduction": "none",
            "algorithm": "kmeans",
            "k": 8,
            "min_cluster_size": 5,
            "remove_small_clusters": True,
            "min_cluster_threshold": 5,
            "merge_into_misc": True,
            "create_playlists": True,
            "generate_report": True,
            "open_graph": True,
        }
        self._run_clustering(config)

    def _on_configure_wizard(self) -> None:
        """Open configuration wizard."""
        dialog = ClusteringWizardDialog(self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            self._current_config = config
            self._display_config(config)

    def _display_config(self, config: dict) -> None:
        """Display configuration in text area."""
        lines = [
            "Configuration:",
            f"  Algorithm: {config.get('algorithm', 'unknown').upper()}",
            f"  Normalization: {config.get('normalization', 'unknown')}",
            f"  Dimensionality reduction: {config.get('dimensionality_reduction', 'none')}",
            "",
            "Features:",
        ]

        for feat, enabled in config.get("features", {}).items():
            status = "✓" if enabled else "✗"
            lines.append(f"  {status} {feat.replace('_', ' ').title()}")

        if config.get("algorithm") == "kmeans":
            lines.extend([
                "",
                f"K-Means Parameters:",
                f"  K (clusters): {config.get('k', 8)}",
            ])
        else:
            lines.extend([
                "",
                f"HDBSCAN Parameters:",
                f"  Min cluster size: {config.get('min_cluster_size', 5)}",
                f"  Min samples: {config.get('min_samples', 5)}",
            ])

        lines.extend([
            "",
            "Output:",
            f"  {'✓' if config.get('create_playlists') else '✗'} Create playlists",
            f"  {'✓' if config.get('generate_report') else '✗'} Generate quality report",
            f"  {'✓' if config.get('open_graph') else '✗'} Open interactive graph",
        ])

        self._config_display.setPlainText("\n".join(lines))

    def _on_run_advanced(self) -> None:
        """Run with advanced configuration."""
        if self._current_config is None:
            QtWidgets.QMessageBox.warning(
                self, "No Configuration",
                "Please configure with the wizard first."
            )
            return
        self._run_clustering(self._current_config)

    def _run_clustering(self, config: dict) -> None:
        """Start clustering with given configuration."""
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return

        self._current_config = config
        self._current_result = None

        # Clean up any existing worker
        self._cleanup_worker()

        # Clear results tab
        while self._results_layout.count():
            self._results_layout.takeAt(0).widget().deleteLater()

        # Show progress
        self._show_progress_panel()

        # Start worker
        self._worker = ClusterWorker(self._library_path, config)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_clustering_finished)
        self._worker.start()

    def _show_progress_panel(self) -> None:
        """Show clustering progress panel."""
        # Clear results
        while self._results_layout.count():
            self._results_layout.takeAt(0).widget().deleteLater()

        # Progress bar
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._results_layout.addWidget(self._progress_bar)

        # Status text
        self._progress_status = QtWidgets.QLabel("Initializing...")
        self._progress_status.setStyleSheet("color: #666; font-size: 11px;")
        self._results_layout.addWidget(self._progress_status)

        # Log area
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(200)
        self._log_area.setStyleSheet("font-family: monospace; font-size: 10px;")
        self._results_layout.addWidget(self._log_area)

        # Cancel button
        cancel_btn = QtWidgets.QPushButton("✕ Cancel")
        cancel_btn.clicked.connect(self._on_cancel_clustering)
        self._results_layout.addWidget(cancel_btn)

        self._results_layout.addStretch(1)

        # Switch to results tab
        self._tab_widget.setCurrentIndex(2)

    def _on_progress(self, percent: int, message: str) -> None:
        """Update progress."""
        self._progress_bar.setValue(percent)
        self._progress_status.setText(message)

    def _on_log_line(self, line: str) -> None:
        """Add log line."""
        self._log_area.appendPlainText(line)

    def _on_cancel_clustering(self) -> None:
        """Cancel clustering."""
        if self._worker:
            self._worker.cancel()

    def _on_clustering_finished(self, success: bool, message: str, result: dict) -> None:
        """Handle clustering completion."""
        self._current_result = result

        # Clean up worker after it finishes
        if self._worker is not None:
            self._worker.wait()
            self._worker = None

        # Clear progress panel
        while self._results_layout.count():
            self._results_layout.takeAt(0).widget().deleteLater()

        if success:
            self._show_results_panel(result)
        else:
            error_card = self._make_card()
            error_layout = QtWidgets.QVBoxLayout(error_card)
            error_label = QtWidgets.QLabel(f"❌ Error: {message}")
            error_label.setStyleSheet("color: #ef4444;")
            error_label.setWordWrap(True)
            error_layout.addWidget(error_label)
            self._results_layout.addWidget(error_card)

        self._results_layout.addStretch(1)

    def _show_results_panel(self, result: dict) -> None:
        """Show clustering results panel."""
        # Success message
        success_card = self._make_card()
        success_layout = QtWidgets.QVBoxLayout(success_card)
        success_label = QtWidgets.QLabel("✓ Clustering complete!")
        success_label.setStyleSheet("color: #22c55e; font-weight: bold;")
        success_layout.addWidget(success_label)
        self._results_layout.addWidget(success_card)

        # Summary
        summary_card = self._make_card()
        summary_layout = QtWidgets.QFormLayout(summary_card)

        if "labels" in result:
            import numpy as np
            labels = result["labels"]
            n_clusters = len(np.unique(labels))
            summary_layout.addRow("Clusters created:", QtWidgets.QLabel(str(n_clusters)))
            summary_layout.addRow("Total tracks:", QtWidgets.QLabel(str(len(labels))))

        metrics = result.get("metrics", {})
        if metrics:
            if "silhouette_score" in metrics:
                sil = metrics["silhouette_score"]
                quality = "Good" if sil > 0.5 else "Fair" if sil > 0.3 else "Poor"
                sil_label = QtWidgets.QLabel(f"{sil:.3f} ({quality})")
                summary_layout.addRow("Silhouette score:", sil_label)

        self._results_layout.addWidget(summary_card)

        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()

        if metrics:
            report_btn = QtWidgets.QPushButton("📄 View Quality Report")
            report_btn.clicked.connect(lambda: self._show_quality_report(result))
            button_layout.addWidget(report_btn)

        playlists_btn = QtWidgets.QPushButton("🎵 View Playlists")
        playlists_btn.clicked.connect(lambda: self._open_playlists_folder())
        button_layout.addWidget(playlists_btn)

        graph_btn = QtWidgets.QPushButton("📊 Open Visual Graph")
        graph_btn.clicked.connect(lambda: self._open_graph())
        button_layout.addWidget(graph_btn)

        button_layout.addStretch(1)
        self._results_layout.addLayout(button_layout)

    def _show_quality_report(self, result: dict) -> None:
        """Show cluster quality report dialog."""
        metrics = result.get("metrics", {})
        cluster_info = result.get("cluster_info", {})

        dialog = ClusterQualityReportDialog(metrics, cluster_info, self)
        dialog.exec()

    def _open_playlists_folder(self) -> None:
        """Open playlists folder in file manager."""
        playlists_dir = Path(self._library_path) / "Playlists"
        if playlists_dir.exists():
            import subprocess
            if hasattr(subprocess, "Popen"):
                subprocess.Popen(["xdg-open", str(playlists_dir)])  # Linux
        else:
            QtWidgets.QMessageBox.warning(self, "Not Found", "Playlists folder not found")

    def _open_graph(self) -> None:
        """Open Visual Music Graph workspace."""
        self._log("Opening Visual Music Graph...", "info")
        # TODO: Signal to switch workspace
        QtWidgets.QMessageBox.information(
            self, "Open Graph",
            "Switch to the Visual Music Graph workspace in the sidebar."
        )
