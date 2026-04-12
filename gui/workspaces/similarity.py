"""Similarity Inspector workspace — compare two tracks side by side."""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class SimilarityWorker(QtCore.QThread):
    finished = Signal(bool, str)

    def __init__(self, path_a: str, path_b: str, params: dict) -> None:
        super().__init__()
        self.path_a = path_a
        self.path_b = path_b
        self.params = params

    def run(self) -> None:
        try:
            from near_duplicate_detector import fingerprint_distance, _parse_fp
            from fingerprint_cache import get_cached_fingerprint

            # Try to locate a fingerprint DB relative to the files' library root
            docs_path = Path(self.path_a).parent.parent / "Docs"
            db_path = str(docs_path / ".soundvault.db") if docs_path.exists() else ""

            def _get_fp(path: str) -> str | None:
                if db_path:
                    try:
                        fp = get_cached_fingerprint(path, db_path)
                        if fp:
                            return fp
                    except Exception:
                        pass
                try:
                    import acoustid
                    _dur, fp = acoustid.fingerprint_file(path)
                    return fp
                except Exception:
                    return None

            fp_a = _get_fp(self.path_a)
            fp_b = _get_fp(self.path_b)

            if fp_a is None or fp_b is None:
                self.finished.emit(False, "Could not compute fingerprints for one or both files.")
                return

            distance = fingerprint_distance(fp_a, fp_b)
            near_thresh = self.params.get("near_threshold", 0.1)
            exact_thresh = self.params.get("exact_threshold", 0.02)
            codec_boost = self.params.get("codec_boost", 0.03)
            ext_a = Path(self.path_a).suffix.lower()
            ext_b = Path(self.path_b).suffix.lower()
            lossless = {".flac", ".wav"}
            mixed = (ext_a in lossless) != (ext_b in lossless)
            effective = near_thresh + (codec_boost if mixed else 0.0)

            if distance <= exact_thresh:
                verdict = "EXACT DUPLICATE"
            elif distance <= effective:
                verdict = "NEAR DUPLICATE"
            else:
                verdict = "NOT A DUPLICATE"

            report_lines = [
                f"File A: {self.path_a}",
                f"File B: {self.path_b}",
                f"",
                f"Codec A: {ext_a}  |  Codec B: {ext_b}",
                f"Mixed-codec: {'Yes' if mixed else 'No'}",
                f"",
                f"Raw fingerprint distance: {distance:.6f}",
                f"Exact threshold:          {exact_thresh:.4f}",
                f"Near threshold:           {near_thresh:.4f}",
                f"Codec boost applied:      {codec_boost if mixed else 0.0:.4f}",
                f"Effective threshold:      {effective:.4f}",
                f"",
                f"VERDICT: {verdict}",
            ]
            self.finished.emit(True, "\n".join(report_lines))
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class SimilarityWorkspace(WorkspaceBase):
    """Compare two tracks to understand why they do or do not match."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: SimilarityWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Similarity Inspector"))
        cl.addWidget(self._make_subtitle(
            "Select two audio files to understand why they match (or do not match) "
            "in the duplicate detection pipeline. Reports codec, duration, raw fingerprint "
            "distance, and the verdict with threshold breakdown."
        ))

        # ── File selection card ────────────────────────────────────────────
        files_card = self._make_card()
        files_layout = QtWidgets.QFormLayout(files_card)
        files_layout.setContentsMargins(16, 16, 16, 16)
        files_layout.setSpacing(10)
        files_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        for attr, label in (("_file_a", "Song A:"), ("_file_b", "Song B:")):
            entry = QtWidgets.QLineEdit()
            entry.setPlaceholderText("Click Browse… to select a file")
            browse = QtWidgets.QPushButton("Browse…")
            browse.setFixedWidth(80)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(entry, 1)
            row.addWidget(browse)
            browse.clicked.connect(
                lambda checked=False, e=entry: self._browse_file(e)
            )
            files_layout.addRow(label, row)
            setattr(self, attr, entry)
        cl.addWidget(files_card)

        # ── Advanced overrides (collapsible) ───────────────────────────────
        adv_group = QtWidgets.QGroupBox("Advanced Overrides")
        adv_group.setCheckable(True)
        adv_group.setChecked(False)
        adv_form = QtWidgets.QFormLayout(adv_group)
        adv_form.setContentsMargins(8, 4, 8, 8)

        self._trim_cb = QtWidgets.QCheckBox("Trim leading/trailing silence")
        adv_form.addRow(self._trim_cb)

        self._fp_offset = QtWidgets.QLineEdit("0")
        self._fp_dur = QtWidgets.QLineEdit("120000")
        self._silence_thresh = QtWidgets.QLineEdit("-60")
        self._silence_min = QtWidgets.QLineEdit("200")
        self._exact_t = QtWidgets.QLineEdit("0.02")
        self._near_t = QtWidgets.QLineEdit("0.1")
        self._codec_boost = QtWidgets.QLineEdit("0.03")

        for label, widget in [
            ("Fingerprint offset (ms):", self._fp_offset),
            ("Fingerprint duration (ms):", self._fp_dur),
            ("Silence threshold (dB):", self._silence_thresh),
            ("Silence min length (ms):", self._silence_min),
            ("Exact threshold:", self._exact_t),
            ("Near threshold:", self._near_t),
            ("Mixed-codec boost:", self._codec_boost),
        ]:
            adv_form.addRow(label, widget)

        cl.addWidget(adv_group)

        # ── Action buttons ─────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = self._make_primary_button("⚖  Run Comparison")
        self._run_btn.clicked.connect(self._on_run)
        self._close_btn = QtWidgets.QPushButton("Clear")
        self._close_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._close_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        # ── Results ────────────────────────────────────────────────────────
        results_card = self._make_card()
        results_layout = QtWidgets.QVBoxLayout(results_card)
        results_layout.setContentsMargins(16, 16, 16, 16)
        results_layout.addWidget(QtWidgets.QLabel("Comparison Report"))

        self._results_text = QtWidgets.QPlainTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMinimumHeight(260)
        self._results_text.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        self._results_text.setPlaceholderText("Results will appear here after running the comparison…")
        results_layout.addWidget(self._results_text)
        cl.addWidget(results_card)

        cl.addStretch(1)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self) -> None:
        a = self._file_a.text().strip()
        b = self._file_b.text().strip()
        if not a or not b:
            QtWidgets.QMessageBox.warning(self, "Missing Files", "Please select both files.")
            return

        self._run_btn.setEnabled(False)
        self._results_text.setPlainText("Computing fingerprints…")
        self._log("Running similarity comparison…", "info")
        self.status_changed.emit("Comparing…", "#f59e0b")

        try:
            params = {
                "exact_threshold": float(self._exact_t.text() or "0.02"),
                "near_threshold": float(self._near_t.text() or "0.1"),
                "codec_boost": float(self._codec_boost.text() or "0.03"),
            }
        except ValueError:
            params = {}

        self._worker = SimilarityWorker(a, b, params)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _on_clear(self) -> None:
        self._file_a.clear()
        self._file_b.clear()
        self._results_text.clear()

    @Slot(bool, str)
    def _on_finished(self, success: bool, report: str) -> None:
        self._run_btn.setEnabled(True)
        self._results_text.setPlainText(report)
        if success:
            self._log("Comparison complete.", "ok")
            self.status_changed.emit("Done", "#22c55e")
        else:
            self._log(f"Comparison failed: {report}", "error")
            self.status_changed.emit("Error", "#ef4444")
        self._worker = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _browse_file(self, entry: QtWidgets.QLineEdit) -> None:
        start = str(Path(entry.text()).parent) if entry.text() else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Audio File", start,
            "Audio files (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All files (*)"
        )
        if path:
            entry.setText(path)
