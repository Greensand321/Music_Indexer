"""Sound Map → Analyse: the clustering panel (wizard and interactive results).

Analysis only: a run writes the feature cache, ``Docs/cluster_members.json``
(the full membership), ``Docs/cluster_info.json`` (the map) and
``Docs/cluster_graph.html``. It writes no playlists itself: the Sound Map's
**Save cluster playlists…** action previews and saves one playlist per
cluster (``gui/cluster_playlist_jobs.py``), and a map selection can be
exported from the Map section (``graph.py``). After a successful run this
panel emits ``analysis_succeeded(result)``; the page uses it to offer the
preview when the wizard's "Offer to save a playlist per cluster" is ticked.
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
from gui.dialogs.clustering_wizard_dialog import ClusteringWizardDialog
from gui.dialogs.cluster_quality_report_dialog import ClusterQualityReportDialog

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
    """Enhanced workspace with wizard and interactive results."""

    # Emitted whenever the latest result changes (a run finished, or the
    # results were cleared), so the Sound Map page can enable its actions.
    result_changed = Signal()
    # Emitted with the result after a *successful* run only (never on a
    # failure or cancel): the page offers "Save cluster playlists" on it.
    analysis_succeeded = Signal(dict)

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: ClusterWorker | None = None
        # True while another job on the page (Save cluster playlists) runs:
        # the Run buttons wait for it (spec CP-34).
        self._external_busy = False
        # The wizard's settings (Advanced → Run) are kept apart from whatever
        # ran last, so a Quick Start run cannot replace them behind the user's
        # back while the Advanced tab still displays the wizard's choices.
        self._wizard_config: dict | None = None
        self._last_run_config: dict | None = None
        self._current_result: dict | None = None
        self._run_buttons: list[QtWidgets.QPushButton] = []
        self._cancel_btn: QtWidgets.QPushButton | None = None
        self._build_ui()

    # Backwards-compatible alias (older code and tests read it).
    @property
    def _current_config(self) -> dict | None:
        return self._last_run_config

    def _build_ui(self) -> None:
        """Build the UI."""
        cl = self.content_layout

        # Title and intro
        cl.addWidget(self._make_section_title("Analyse your music"))
        cl.addWidget(self._make_subtitle(
            "Analyse your tracks' audio (tempo, timbre, energy and, if you choose, harmony, "
            "brightness and rhythm density) and group similar tracks with K-Means or "
            "HDBSCAN. Then explore the groups on the Map, play a cluster, or save a "
            "playlist per cluster with Save cluster playlists… (previewed first). "
            "The analysis itself modifies nothing in your library."
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
        self._show_placeholder("Run Quick Start or the Advanced configuration to see results here.")

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
            "• Tempo, timbre (MFCC) and energy features\n"
            "• Standard normalization\n\n"
            "Want more control (harmony, brightness, HDBSCAN…)? Use the Advanced tab."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_card)

        # Button
        run_btn = self._make_primary_button("📊  Run Quick Start")
        run_btn.clicked.connect(self._on_run_quick_start)
        self._run_buttons.append(run_btn)
        layout.addWidget(run_btn)

        layout.addStretch(1)
        return tab

    def _build_advanced_tab(self) -> QtWidgets.QWidget:
        """Build advanced configuration tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Info
        info_label = QtWidgets.QLabel(
            "Choose the features, normalization and algorithm with the wizard, then Run:"
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
        self._run_buttons.append(run_btn)
        button_layout.addWidget(run_btn)

        button_layout.addStretch(1)
        layout.addLayout(button_layout)

        layout.addStretch(1)
        return tab

    # ── Running ───────────────────────────────────────────────────────────

    def is_busy(self) -> bool:
        return self._worker is not None

    def _set_busy(self, busy: bool) -> None:
        for btn in self._run_buttons:
            btn.setEnabled(not busy and not self._external_busy)
            btn.setToolTip(
                "Wait for Save cluster playlists to finish." if self._external_busy and not busy else ""
            )

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
        """Open configuration wizard."""
        dialog = ClusteringWizardDialog(self)
        try:
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                config = dialog.get_config()
                self._wizard_config = config
                self._display_config(config)
        finally:
            dialog.deleteLater()

    def _display_config(self, config: dict) -> None:
        """Display configuration in text area."""
        lines = [
            "Configuration:",
            f"  Algorithm: {'K-Means' if config.get('algorithm') == 'kmeans' else 'HDBSCAN'}",
            f"  Normalization: {config.get('normalization', 'unknown')}",
            "",
            "Features:",
        ]

        for feat, enabled in config.get("features", {}).items():
            status = "✓" if enabled else "✗"
            lines.append(f"  {status} {feat.replace('_', ' ').title()}")

        if config.get("algorithm") == "kmeans":
            lines.extend([
                "",
                "K-Means Parameters:",
                f"  K (clusters): {config.get('k', 8)}",
            ])
        else:
            lines.extend([
                "",
                "HDBSCAN Parameters:",
                f"  Min cluster size: {config.get('min_cluster_size', 5)}",
                f"  Min samples: {config.get('min_samples', 5)}",
            ])

        lines.extend([
            "",
            "Output:",
            "  Cluster data for the Map and the 3D view, plus a quality report.",
            "  Save cluster playlists… writes one playlist per cluster",
        ])
        if config.get("create_playlists"):
            lines.append("  Offers to save them when the run finishes")

        self._config_display.setPlainText("\n".join(lines))

    def _on_run_advanced(self) -> None:
        """Run with advanced configuration."""
        if self._wizard_config is None:
            QtWidgets.QMessageBox.warning(
                self, "No Configuration",
                "Please configure with the wizard first."
            )
            return
        self._run_clustering(self._wizard_config)

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
        """Remove everything from the results tab, nested layouts included."""
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

    def _show_placeholder(self, text: str) -> None:
        self._clear_results()
        label = QtWidgets.QLabel(text)
        label.setObjectName("statusHint")
        label.setWordWrap(True)
        self._results_layout.addWidget(label)
        self._results_layout.addStretch(1)

    def _show_progress_panel(self) -> None:
        """Show clustering progress panel."""
        self._clear_results()

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._results_layout.addWidget(self._progress_bar)

        self._progress_status = QtWidgets.QLabel("Initializing...")
        self._progress_status.setObjectName("statusHint")
        self._results_layout.addWidget(self._progress_status)

        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setObjectName("logText")
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(200)
        self._results_layout.addWidget(self._log_area)

        self._cancel_btn = QtWidgets.QPushButton("✕ Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel_clustering)
        self._results_layout.addWidget(self._cancel_btn)

        self._results_layout.addStretch(1)

        # Switch to results tab
        self._tab_widget.setCurrentIndex(2)

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
            self._show_placeholder("Clustering was cancelled. No cluster data was written.")
            self._status("Cancelled", "neutral")
            return
        else:
            error_card = self._make_card()
            error_layout = QtWidgets.QVBoxLayout(error_card)
            error_label = QtWidgets.QLabel(f"❌ Error: {message}")
            error_label.setWordWrap(True)
            error_layout.addWidget(error_label)
            self._results_layout.addWidget(error_card)
            self._status("Clustering failed", "error")
            self._log(message, "error")

        self._results_layout.addStretch(1)
        if success:
            # Last, once the results are on screen: the page may offer the
            # Save cluster playlists preview (wizard option, CP-44).
            self.analysis_succeeded.emit(result)

    def _show_results_panel(self, result: dict) -> None:
        """Show clustering results panel."""
        success_card = self._make_card()
        success_layout = QtWidgets.QVBoxLayout(success_card)
        success_label = QtWidgets.QLabel("✓ Clustering complete!")
        success_label.setObjectName("cardTitle")
        success_layout.addWidget(success_label)
        self._results_layout.addWidget(success_card)

        summary_card = self._make_card()
        summary_layout = QtWidgets.QFormLayout(summary_card)

        metrics = result.get("metrics", {}) or {}
        if "labels" in result:
            import numpy as np
            labels = np.asarray(result["labels"])
            n_clusters = len(np.unique(labels[labels >= 0]))
            noise = int((labels < 0).sum())
            summary_layout.addRow("Clusters created:", QtWidgets.QLabel(str(n_clusters)))
            summary_layout.addRow("Total tracks:", QtWidgets.QLabel(str(len(labels))))
            if noise:
                summary_layout.addRow(
                    "Unclustered (noise):",
                    QtWidgets.QLabel(f"{noise} ({noise / max(len(labels), 1):.0%})"),
                )
        failed = result.get("failed") or {}
        if failed:
            summary_layout.addRow(
                "Could not analyse:",
                QtWidgets.QLabel(f"{len(failed)} track(s) (left out; retried next run)"),
            )

        if "silhouette_score" in metrics:
            sil = metrics["silhouette_score"]
            quality = "Good" if sil > 0.5 else "Fair" if sil > 0.3 else "Poor"
            summary_layout.addRow("Silhouette score:", QtWidgets.QLabel(f"{sil:.3f} ({quality})"))

        self._results_layout.addWidget(summary_card)

        hint = QtWidgets.QLabel(
            "Save a playlist per cluster with “Save cluster playlists…” (at the top "
            "of the Sound Map), or export a lasso selection from the map."
        )
        hint.setObjectName("statusHint")
        hint.setWordWrap(True)
        self._results_layout.addWidget(hint)

        button_layout = QtWidgets.QHBoxLayout()

        if "silhouette_score" in metrics:
            report_btn = QtWidgets.QPushButton("📄 View Quality Report")
            report_btn.clicked.connect(lambda: self._show_quality_report(result))
            button_layout.addWidget(report_btn)

        graph_btn = QtWidgets.QPushButton("📊 Open the map")
        graph_btn.clicked.connect(self._open_graph)
        button_layout.addWidget(graph_btn)

        button_layout.addStretch(1)
        self._results_layout.addLayout(button_layout)

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
        self._show_placeholder("Run Quick Start or the Advanced configuration to see results here.")
