"""Clustered Playlists workspace — K-Means / HDBSCAN clustering."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class ClusterWorker(QtCore.QThread):
    log_line = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, library_path: str, method: str, n_clusters: int,
                 min_cluster_size: int, output_dir: str) -> None:
        super().__init__()
        self.library_path = library_path
        self.method = method
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.output_dir = output_dir
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import clustered_playlists
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.log_line.emit(msg)

            clustered_playlists.run(
                self.library_path,
                method=self.method,
                n_clusters=self.n_clusters,
                min_cluster_size=self.min_cluster_size,
                output_dir=self.output_dir,
                log_callback=_log,
            )
            if not self._cancelled:
                self.finished.emit(True, "Clustering complete.")
            else:
                self.finished.emit(False, "Cancelled.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class ClusteredWorkspace(WorkspaceBase):
    """K-Means / HDBSCAN clustering and cluster playlist generation."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: ClusterWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Clustered Playlists"))
        cl.addWidget(self._make_subtitle(
            "Extract audio features (tempo, MFCC, chroma) and cluster tracks "
            "into sonically similar groups using K-Means or HDBSCAN. "
            "Each cluster becomes a separate playlist. "
            "Progress is logged to <method>_log.txt inside your library."
        ))

        # ── Config card ────────────────────────────────────────────────────
        cfg_card = self._make_card()
        cfg_layout = QtWidgets.QFormLayout(cfg_card)
        cfg_layout.setContentsMargins(16, 16, 16, 16)
        cfg_layout.setSpacing(10)
        cfg_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self._method_combo = QtWidgets.QComboBox()
        self._method_combo.addItems(["K-Means", "HDBSCAN"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        cfg_layout.addRow("Clustering method:", self._method_combo)

        self._k_spin = QtWidgets.QSpinBox()
        self._k_spin.setRange(2, 100)
        self._k_spin.setValue(8)
        self._k_spin.setToolTip("Number of clusters for K-Means.")
        cfg_layout.addRow("K (clusters):", self._k_spin)

        self._min_size_spin = QtWidgets.QSpinBox()
        self._min_size_spin.setRange(2, 200)
        self._min_size_spin.setValue(5)
        self._min_size_spin.setEnabled(False)
        self._min_size_spin.setToolTip("Minimum cluster size for HDBSCAN.")
        cfg_layout.addRow("Min cluster size:", self._min_size_spin)

        self._out_entry = QtWidgets.QLineEdit()
        self._out_entry.setPlaceholderText("Leave blank to use Playlists/ inside library")
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self._out_entry, 1)
        out_browse = QtWidgets.QPushButton("Browse…")
        out_browse.clicked.connect(
            lambda: self._browse_dir(self._out_entry)
        )
        out_row.addWidget(out_browse)
        cfg_layout.addRow("Output folder:", out_row)
        cl.addWidget(cfg_card)

        # ── Feature extraction options ─────────────────────────────────────
        feat_card = self._make_card()
        feat_layout = QtWidgets.QVBoxLayout(feat_card)
        feat_layout.setContentsMargins(16, 16, 16, 16)
        feat_layout.setSpacing(6)
        feat_layout.addWidget(QtWidgets.QLabel("Features"))

        feat_row = QtWidgets.QHBoxLayout()
        for feat in ("Tempo", "MFCC", "Chroma", "Spectral centroid", "Energy"):
            cb = QtWidgets.QCheckBox(feat)
            cb.setChecked(True)
            feat_row.addWidget(cb)
        feat_row.addStretch(1)
        feat_layout.addLayout(feat_row)

        engine_row = QtWidgets.QHBoxLayout()
        engine_row.addWidget(QtWidgets.QLabel("Audio engine:"))
        self._engine_combo = QtWidgets.QComboBox()
        self._engine_combo.addItems(["librosa", "essentia (if installed)"])
        engine_row.addWidget(self._engine_combo)
        engine_row.addStretch(1)
        feat_layout.addLayout(engine_row)
        cl.addWidget(feat_card)

        # ── Buttons ────────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = self._make_primary_button("📊  Run Clustering")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QtWidgets.QPushButton("✕  Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._graph_btn = QtWidgets.QPushButton("🌐  Open Visual Graph")
        self._graph_btn.setEnabled(False)
        self._graph_btn.clicked.connect(self._on_open_graph)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._graph_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        # ── Progress ───────────────────────────────────────────────────────
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setRange(0, 0)
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setVisible(False)
        self._prog_status = QtWidgets.QLabel("Idle")
        self._prog_status.setStyleSheet("color: #64748b; font-size: 12px;")
        cl.addWidget(self._prog_bar)
        cl.addWidget(self._prog_status)

        # ── Log ────────────────────────────────────────────────────────────
        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(QtWidgets.QLabel("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(180)
        self._log_area.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return

        self._log_area.clear()
        self._prog_bar.setVisible(True)
        self._prog_status.setText("Clustering…")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._graph_btn.setEnabled(False)
        self._log("Starting clustering…", "info")
        self.status_changed.emit("Clustering…", "#f59e0b")

        method_map = {"K-Means": "kmeans", "HDBSCAN": "hdbscan"}
        method = method_map.get(self._method_combo.currentText(), "kmeans")

        self._worker = ClusterWorker(
            library_path=self._library_path,
            method=method,
            n_clusters=self._k_spin.value(),
            min_cluster_size=self._min_size_spin.value(),
            output_dir=self._out_entry.text(),
        )
        self._worker.log_line.connect(self._log_area.appendPlainText)
        self._worker.log_line.connect(lambda m: self._log(m))
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    @Slot()
    def _on_open_graph(self) -> None:
        """Switch to the Visual Graph workspace."""
        self.log_message.emit("Opening Visual Music Graph…", "info")
        # The main window listens for this to switch the workspace
        # We emit a custom signal here; the parent will handle navigation
        # For now, just log
        self._log("Switch to the Visual Music Graph workspace in the sidebar.", "info")

    @Slot(str)
    def _on_method_changed(self, method: str) -> None:
        is_hdbscan = method == "HDBSCAN"
        self._k_spin.setEnabled(not is_hdbscan)
        self._min_size_spin.setEnabled(is_hdbscan)

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str) -> None:
        self._prog_bar.setVisible(False)
        self._prog_status.setText(message)
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        if success:
            self._log(message, "ok")
            self.status_changed.emit("Done", "#22c55e")
            self._graph_btn.setEnabled(True)
        else:
            self._log(message, "error" if "Cancelled" not in message else "warn")
            self.status_changed.emit("Error", "#ef4444")
        self._worker = None

    def _browse_dir(self, entry: QtWidgets.QLineEdit) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            entry.setText(folder)
