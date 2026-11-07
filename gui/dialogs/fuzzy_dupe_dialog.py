"""Fuzzy Duplicate Finder dialog — near-duplicate candidates from cached fingerprints.

A proof of concept. It used to call ``find_near_duplicates`` with the wrong
arguments (it always failed) and read result attributes that do not exist
(it would always have found 0 pairs). It now runs the real
``near_duplicate_detector.find_near_duplicates`` over the fingerprints the
Duplicate Finder has already cached, and only shows controls that do
something; the metadata word-overlap search it was designed around has no
backend, so those controls are disabled and say so.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.workers import start_worker

_AUDIO_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
_NOT_IMPLEMENTED = "Not implemented in this proof of concept (no backend for it)."


class FuzzyDupeWorker(QtCore.QThread):
    progress = Signal(int, str)
    result_ready = Signal(list)
    done = Signal(bool, str)

    def __init__(self, library_path: str, params: dict) -> None:
        super().__init__()
        self.library_path = library_path
        self.params = params
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            from fingerprint_cache import get_cached_fingerprint_metadata
            from near_duplicate_detector import find_near_duplicates
            from simple_duplicate_finder import EXT_PRIORITY
            from utils.reserved_folders import iter_library_files
        except ImportError as exc:
            self.done.emit(False, f"Import error: {exc}")
            return
        try:
            db_path = os.path.join(self.library_path, "Docs", ".duplicate_fingerprints.db")
            if not os.path.isfile(db_path):
                self.done.emit(False, "No fingerprints cached yet — run a scan in Duplicates first.")
                return

            file_infos: dict[str, dict] = {}
            uncached = 0
            paths = sorted(str(p) for p in iter_library_files(self.library_path, _AUDIO_EXTS))
            for i, path in enumerate(paths, 1):
                if self._cancel.is_set():
                    self.done.emit(False, "Cancelled.")
                    return
                fp, meta = get_cached_fingerprint_metadata(path, db_path)
                if not fp:
                    uncached += 1
                    continue
                meta = meta or {}
                tags = meta.get("tags") if isinstance(meta.get("tags"), dict) else {}
                file_infos[path] = {
                    "fp": fp,
                    "title": tags.get("title"),
                    "artist": tags.get("artist"),
                    "album": tags.get("album"),
                    "primary": tags.get("albumartist") or tags.get("artist"),
                    "ext": meta.get("ext") or os.path.splitext(path)[1].lower(),
                    "meta_count": sum(1 for v in tags.values() if v),
                }
                if i % 200 == 0:
                    self.progress.emit(i, f"Reading cached fingerprints… {i}/{len(paths)}")

            if self._cancel.is_set():
                self.done.emit(False, "Cancelled.")
                return
            self.progress.emit(len(paths), f"Comparing {len(file_infos)} fingerprinted tracks…")
            result = find_near_duplicates(
                file_infos,
                EXT_PRIORITY,
                self.params["near_threshold"],
                mixed_codec_boost=self.params["codec_boost"],
            )
            pairs = [
                (group.winner, loser, group.max_distance)
                for group in result.review_groups
                for loser in group.losers
            ]
            if self._cancel.is_set():
                self.done.emit(False, "Cancelled.")
                return
            self.result_ready.emit(pairs)
            note = f"; {uncached} track(s) not fingerprinted yet were skipped" if uncached else ""
            self.done.emit(True, f"{len(pairs)} candidate pair(s) found{note}.")
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


class FuzzyDupeDialog(QtWidgets.QDialog):
    """Near-duplicate candidates among already-fingerprinted tracks."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fuzzy Duplicate Finder")
        # The Tools launcher creates one per click; don't keep each one alive.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(820, 640)
        self._library_path = library_path
        self._worker: FuzzyDupeWorker | None = None
        self._closing = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        intro = QtWidgets.QLabel(
            "Lists near-duplicate candidates within each album, using the fingerprints the "
            "Duplicate Finder has already cached (run a scan there first). Read-only."
        )
        intro.setWordWrap(True)
        intro.setObjectName("sectionSubtitle")
        root.addWidget(intro)

        # ── Metadata match settings (no backend: shown disabled) ──────────
        meta_grp = QtWidgets.QGroupBox("Metadata Match Settings (not implemented)")
        meta_grp.setToolTip(_NOT_IMPLEMENTED)
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
        self._field_cbs: list[QtWidgets.QCheckBox] = []
        for field in ("Title", "Artist", "Album", "Genre", "Filename"):
            cb = QtWidgets.QCheckBox(field)
            cb.setChecked(True)
            self._field_cbs.append(cb)
            fields_row.addWidget(cb)
        fields_row.addStretch(1)
        mf.addRow("Search fields:", fields_row)
        meta_grp.setEnabled(False)
        root.addWidget(meta_grp)

        # ── Fingerprint filter ─────────────────────────────────────────────
        fp_grp = QtWidgets.QGroupBox("Fingerprint Filter")
        ff = QtWidgets.QFormLayout(fp_grp)
        ff.setSpacing(8)
        from gui.workspaces.duplicates import _config_thresholds

        exact, near, boost = _config_thresholds()
        self._exact_t = QtWidgets.QLineEdit(f"{exact:g}")
        self._exact_t.setToolTip("Pairs at or below this distance are labelled 'Exact'.")
        self._near_t = QtWidgets.QLineEdit(f"{near:g}")
        self._near_t.setToolTip("Maximum fingerprint distance for a candidate pair.")
        self._boost_t = QtWidgets.QLineEdit(f"{boost:g}")
        self._boost_t.setToolTip("Extra distance allowed between lossless and lossy files.")
        ff.addRow("Exact threshold:", self._exact_t)
        ff.addRow("Near threshold:", self._near_t)
        ff.addRow("Mixed-codec boost:", self._boost_t)
        self._no_fp_cb = QtWidgets.QCheckBox("Include pairs without fingerprints")
        self._skip_fp_cb = QtWidgets.QCheckBox("Skip fingerprint check (metadata only)")
        for cb in (self._no_fp_cb, self._skip_fp_cb):
            cb.setEnabled(False)
            cb.setToolTip(_NOT_IMPLEMENTED)
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
        self._include_non_cb.setEnabled(False)
        self._include_non_cb.setToolTip(_NOT_IMPLEMENTED)
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
        self._prog_status.setObjectName("statusHint")
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
        self._table.setHeaderLabels(["Verdict", "Distance", "Track A (kept)", "Track B"])
        self._table.setColumnWidth(0, 110)
        self._table.setColumnWidth(1, 70)
        self._table.setColumnWidth(2, 260)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(160)
        self._table.currentItemChanged.connect(self._on_row_selected)
        root.addWidget(self._table)

        self._summary_lbl = QtWidgets.QLabel("")
        self._summary_lbl.setObjectName("statusHint")
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
        self._send_review_btn.setToolTip(
            "Not implemented. Tools → Pair Review reviews the Duplicate Finder's plan instead."
        )
        det_l.addWidget(self._send_review_btn)
        root.addWidget(details_grp)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _thresholds(self) -> dict | None:
        values = {}
        for key, label, widget in (
            ("exact_threshold", "Exact threshold", self._exact_t),
            ("near_threshold", "Near threshold", self._near_t),
            ("codec_boost", "Mixed-codec boost", self._boost_t),
        ):
            try:
                value = float(widget.text().strip())
            except ValueError:
                value = -1.0
            if not 0.0 <= value <= 1.0:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid value", f"{label}: {widget.text()!r} is not a number between 0 and 1."
                )
                return None
            values[key] = value
        return values

    @Slot()
    def _on_run(self) -> None:
        if self._worker is not None:
            return
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "No library path set.")
            return
        params = self._thresholds()
        if params is None:
            return

        self._table.clear()
        self._summary_lbl.setText("")
        self._prog_bar.setVisible(True)
        self._prog_status.setText("Scanning…")
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)

        self._worker = FuzzyDupeWorker(self._library_path, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_results)
        self._worker.done.connect(self._on_finished)
        start_worker(self._worker, owner=self, name="Fuzzy duplicate finder")

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)

    def reject(self) -> None:  # noqa: D401 - Esc, Close and the title bar all end here
        """Closing mid-run cancels the scan."""
        if self._worker is not None:
            self._closing = True
            self._worker.cancel()
        super().reject()

    @Slot(int, str)
    def _on_progress(self, _value: int, message: str) -> None:
        self._prog_status.setText(message)

    @Slot(list)
    def _on_results(self, pairs: list) -> None:
        self._table.clear()
        exact = float(self._exact_t.text() or 0)
        for a, b, dist in pairs:
            verdict = "Exact duplicate" if dist <= exact else "Near duplicate"
            item = QtWidgets.QTreeWidgetItem([
                verdict,
                f"{dist:.4f}",
                Path(str(a)).name,
                Path(str(b)).name,
            ])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (str(a), str(b), dist))
            self._table.addTopLevelItem(item)
        self._summary_lbl.setText(f"Found {len(pairs)} candidate pair(s)")

    @Slot(object, object)
    def _on_row_selected(self, current, _previous) -> None:  # noqa: ANN001
        if current:
            data = current.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if data:
                a, b, dist = data
                self._details_text.setPlainText(f"Track A: {a}\nTrack B: {b}\nDistance: {dist:.6f}")

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str) -> None:
        self._worker = None
        if self._closing:
            return
        self._prog_bar.setVisible(False)
        self._prog_status.setText(message)
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
