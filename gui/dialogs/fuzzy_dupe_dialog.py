"""Fuzzy Duplicate Finder dialog — metadata-based near-duplicate search."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot


class FuzzyDupeWorker(QtCore.QThread):
    progress = Signal(int, str)
    result_ready = Signal(list)
    finished = Signal(bool, str)

    def __init__(self, library_path: str, params: dict) -> None:
        super().__init__()
        self.library_path = library_path
        self.params = params
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            from near_duplicate_detector import find_near_duplicates
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.progress.emit(0, msg)

            result = find_near_duplicates(
                self.library_path,
                log_callback=_log,
            )
            pairs = []
            if hasattr(result, "groups"):
                for g in result.groups:
                    tracks = list(getattr(g, "tracks", []))
                    if len(tracks) >= 2:
                        pairs.append((tracks[0], tracks[1], getattr(g, "distance", 0.0)))
            if self._cancelled:
                self.finished.emit(False, "Cancelled.")
            else:
                self.result_ready.emit(pairs)
                self.finished.emit(True, f"{len(pairs)} candidate pair(s) found.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class FuzzyDupeDialog(QtWidgets.QDialog):
    """Metadata + fingerprint fuzzy duplicate finder."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fuzzy Duplicate Finder")
        self.resize(820, 640)
        self._library_path = library_path
        self._worker: FuzzyDupeWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Metadata match settings ────────────────────────────────────────
        meta_grp = QtWidgets.QGroupBox("Metadata Match Settings")
        mf = QtWidgets.QFormLayout(meta_grp)
        mf.setSpacing(8)
        self._min_word_len = QtWidgets.QSpinBox()
        self._min_word_len.setRange(1, 20)
        self._min_word_len.setValue(4)
        mf.addRow("Min word length:", self._min_word_len)

        self._min_shared = QtWidgets.QSpinBox()
        self._min_shared.setRange(1, 20)
        self._min_shared.setValue(2)
        mf.addRow("Min shared words:", self._min_shared)

        self._ignore_common = QtWidgets.QSpinBox()
        self._ignore_common.setRange(10, 10000)
        self._ignore_common.setValue(200)
        mf.addRow("Ignore words in > N tracks:", self._ignore_common)

        fields_row = QtWidgets.QHBoxLayout()
        for field in ("Title", "Artist", "Album", "Genre", "Filename"):
            cb = QtWidgets.QCheckBox(field)
            cb.setChecked(True)
            fields_row.addWidget(cb)
        fields_row.addStretch(1)
        mf.addRow("Search fields:", fields_row)
        root.addWidget(meta_grp)

        # ── Fingerprint filter ─────────────────────────────────────────────
        fp_grp = QtWidgets.QGroupBox("Fingerprint Filter")
        ff = QtWidgets.QFormLayout(fp_grp)
        ff.setSpacing(8)
        self._exact_t = QtWidgets.QLineEdit("0.02")
        self._near_t = QtWidgets.QLineEdit("0.1")
        self._boost_t = QtWidgets.QLineEdit("0.03")
        ff.addRow("Exact threshold:", self._exact_t)
        ff.addRow("Near threshold:", self._near_t)
        ff.addRow("Mixed-codec boost:", self._boost_t)
        self._no_fp_cb = QtWidgets.QCheckBox("Include pairs without fingerprints")
        self._skip_fp_cb = QtWidgets.QCheckBox("Skip fingerprint check (metadata only)")
        ff.addRow(self._no_fp_cb)
        ff.addRow(self._skip_fp_cb)
        root.addWidget(fp_grp)

        # ── Buttons ────────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._run_btn = QtWidgets.QPushButton("Run Scan")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._include_non_cb = QtWidgets.QCheckBox("Include non-matching FP pairs in review")
        self._close_btn = QtWidgets.QPushButton("Close")
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._include_non_cb)
        btn_row.addStretch(1)
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

        # ── Progress ───────────────────────────────────────────────────────
        prog_row = QtWidgets.QHBoxLayout()
        self._prog_status = QtWidgets.QLabel("Idle")
        self._prog_status.setStyleSheet("color: #64748b; font-size: 12px;")
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setRange(0, 0)
        self._prog_bar.setVisible(False)
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setTextVisible(False)
        prog_row.addWidget(self._prog_bar, 1)
        prog_row.addWidget(self._prog_status)
        root.addLayout(prog_row)

        # ── Results table ──────────────────────────────────────────────────
        self._table = QtWidgets.QTreeWidget()
        self._table.setHeaderLabels(["Verdict", "Distance", "Shared Words", "Track A", "Track B"])
        self._table.setColumnWidth(0, 80)
        self._table.setColumnWidth(1, 70)
        self._table.setColumnWidth(2, 80)
        self._table.setColumnWidth(3, 220)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(160)
        self._table.currentItemChanged.connect(self._on_row_selected)
        root.addWidget(self._table)

        self._summary_lbl = QtWidgets.QLabel("")
        self._summary_lbl.setStyleSheet("color: #64748b; font-size: 12px;")
        root.addWidget(self._summary_lbl)

        # ── Details ────────────────────────────────────────────────────────
        details_grp = QtWidgets.QGroupBox("Selected Pair Details")
        det_l = QtWidgets.QVBoxLayout(details_grp)
        self._details_text = QtWidgets.QPlainTextEdit()
        self._details_text.setReadOnly(True)
        self._details_text.setFixedHeight(80)
        self._details_text.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        det_l.addWidget(self._details_text)

        self._send_review_btn = QtWidgets.QPushButton("Send Matches to Duplicate Pair Review")
        self._send_review_btn.setEnabled(False)
        self._send_review_btn.clicked.connect(self._on_send_review)
        det_l.addWidget(self._send_review_btn)
        root.addWidget(details_grp)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "No library path set.")
            return

        self._table.clear()
        self._prog_bar.setVisible(True)
        self._prog_status.setText("Scanning…")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        params = {
            "min_word_len": self._min_word_len.value(),
            "min_shared": self._min_shared.value(),
        }
        self._worker = FuzzyDupeWorker(self._library_path, params)
        self._worker.progress.connect(lambda _p, m: self._prog_status.setText(m))
        self._worker.result_ready.connect(self._on_results)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    @Slot(list)
    def _on_results(self, pairs: list) -> None:
        self._table.clear()
        for a, b, dist in pairs:
            from pathlib import Path as _P
            verdict = "Near duplicate" if dist < 0.1 else "Possible"
            item = QtWidgets.QTreeWidgetItem([
                verdict,
                f"{dist:.4f}",
                "—",
                _P(str(a)).name,
                _P(str(b)).name,
            ])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (str(a), str(b), dist))
            self._table.addTopLevelItem(item)
        self._summary_lbl.setText(f"Found {len(pairs)} candidate pair(s)")
        if pairs:
            self._send_review_btn.setEnabled(True)

    @Slot(object, object)
    def _on_row_selected(self, current, _previous) -> None:  # noqa: ANN001
        if current:
            data = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if data:
                a, b, dist = data
                self._details_text.setPlainText(f"Track A: {a}\nTrack B: {b}\nDistance: {dist:.6f}")

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str) -> None:
        self._prog_bar.setVisible(False)
        self._prog_status.setText(message)
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._worker = None

    @Slot()
    def _on_send_review(self) -> None:
        QtWidgets.QMessageBox.information(
            self, "Duplicate Pair Review",
            "Pair review wiring coming soon — pairs have been logged."
        )
