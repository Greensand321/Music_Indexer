"""Similarity Inspector workspace — compare two tracks side by side."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from gui.compat import QtCore, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

# What the Duplicate Finder passes to chromaprint_utils.fingerprint_fpcalc
# (its defaults); used unless "Advanced Overrides" is on.
_DEFAULT_FP_PARAMS = {
    "trim": True,
    "start_sec": 0.0,
    "duration_sec": 120.0,
    "threshold_db": -50.0,
    "min_silence_duration": 0.5,
}


def _candidate_cache_dbs(path: str, library_path: str) -> list[str]:
    """Existing Duplicate Finder caches that could hold ``path``'s fingerprint."""
    seen: list[str] = []
    roots = [library_path] if library_path else []
    roots += [str(p) for p in list(Path(path).resolve().parents)[:8]]
    for root in roots:
        db = os.path.join(root, "Docs", ".duplicate_fingerprints.db")
        if db not in seen and os.path.isfile(db):
            seen.append(db)
    return seen


class SimilarityWorker(QtCore.QThread):
    done = Signal(bool, str)

    def __init__(self, path_a: str, path_b: str, params: dict, library_path: str = "") -> None:
        super().__init__()
        self.path_a = path_a
        self.path_b = path_b
        self.params = params
        self.library_path = library_path
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def _fingerprint(self, path: str) -> tuple[str | None, str]:
        """(fingerprint, where it came from) — the same pipeline as the Duplicate Finder."""
        from near_duplicate_detector import _fp_text

        overrides = self.params.get("fingerprint")
        if not overrides:
            from fingerprint_cache import get_cached_fingerprint

            for db in _candidate_cache_dbs(path, self.library_path):
                try:
                    fp = get_cached_fingerprint(path, db)
                except Exception:
                    fp = None
                if fp:
                    return _fp_text(fp), f"Duplicate Finder cache ({db})"

        fp_params = dict(_DEFAULT_FP_PARAMS)
        fp_params.update(overrides or {})
        import chromaprint_utils

        if not chromaprint_utils.missing_tools():
            try:
                fp = chromaprint_utils.fingerprint_fpcalc(path, **fp_params)
            except chromaprint_utils.FingerprintError as exc:
                return None, f"fpcalc failed: {exc}"
            if not fp:
                return None, "fpcalc returned no fingerprint (clip too short or silent?)"
            note = " with your overrides" if overrides else ""
            return _fp_text(fp), f"fpcalc{note}"
        try:
            import acoustid

            _dur, fp = acoustid.fingerprint_file(path)
        except Exception as exc:  # noqa: BLE001
            return None, f"no fingerprinter available (fpcalc/ffmpeg not found; pyacoustid: {exc})"
        note = " — fpcalc not found, overrides not applied" if overrides else " (fpcalc not found)"
        return _fp_text(fp), f"libchromaprint via pyacoustid{note}"

    def run(self) -> None:
        try:
            from near_duplicate_detector import LOSSLESS_EXTS, decode_chromaprint, fingerprint_distance

            fp_a, source_a = self._fingerprint(self.path_a)
            if self._cancel.is_set():
                self.done.emit(False, "Cancelled.")
                return
            fp_b, source_b = self._fingerprint(self.path_b)
            if self._cancel.is_set():
                self.done.emit(False, "Cancelled.")
                return

            if fp_a is None or fp_b is None:
                lines = ["Could not compute fingerprints for one or both files."]
                if fp_a is None:
                    lines.append(f"  A: {source_a}")
                if fp_b is None:
                    lines.append(f"  B: {source_b}")
                self.done.emit(False, "\n".join(lines))
                return

            distance = fingerprint_distance(fp_a, fp_b)
            near_thresh = self.params["near_threshold"]
            exact_thresh = self.params["exact_threshold"]
            codec_boost = self.params["codec_boost"]
            ext_a = Path(self.path_a).suffix.lower()
            ext_b = Path(self.path_b).suffix.lower()
            mixed = (ext_a in LOSSLESS_EXTS) != (ext_b in LOSSLESS_EXTS)
            effective = near_thresh + (codec_boost if mixed else 0.0)

            if distance <= exact_thresh:
                verdict = "EXACT DUPLICATE"
            elif distance <= effective:
                verdict = "NEAR DUPLICATE"
            else:
                verdict = "NOT A DUPLICATE"

            def _items(fp: str) -> str:
                decoded = decode_chromaprint(fp)
                return f"{len(decoded[1])} sub-fingerprints" if decoded else "not a Chromaprint fingerprint"

            def _kind(ext: str) -> str:
                return "lossless" if ext in LOSSLESS_EXTS else "lossy"

            report_lines = [
                f"File A: {self.path_a}",
                f"File B: {self.path_b}",
                "",
                f"Fingerprint A: {source_a}; {_items(fp_a)}",
                f"Fingerprint B: {source_b}; {_items(fp_b)}",
                f"Codec A: {ext_a} ({_kind(ext_a)})  |  Codec B: {ext_b} ({_kind(ext_b)})",
                f"Mixed-codec: {'Yes' if mixed else 'No'}",
                "",
                f"Fingerprint distance:     {distance:.6f}"
                "   (bit error rate: 0 = identical, about 0.5 = unrelated)",
                f"Exact threshold:          {exact_thresh:.4f}",
                f"Near threshold:           {near_thresh:.4f}",
                f"Codec boost applied:      {codec_boost if mixed else 0.0:.4f}",
                f"Effective threshold:      {effective:.4f}",
                f"Thresholds from:          {self.params.get('threshold_source', 'Settings')}",
                "",
                f"VERDICT: {verdict}",
            ]
            self.done.emit(True, "\n".join(report_lines))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


class SimilarityWorkspace(WorkspaceBase):
    """Compare two tracks to understand why they do or do not match."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: SimilarityWorker | None = None
        self._build_ui()
        self._show_config_thresholds()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Similarity Inspector"))
        cl.addWidget(self._make_subtitle(
            "Select two audio files to understand why they match (or do not match) "
            "in the duplicate detection pipeline. Reports codec, raw fingerprint "
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
        self._adv_group = adv_group = QtWidgets.QGroupBox("Advanced Overrides")
        adv_group.setCheckable(True)
        adv_group.setChecked(False)
        adv_group.setToolTip(
            "Off: fingerprint exactly like the Duplicate Finder (and reuse its cache) and use "
            "the thresholds saved in Settings. On: use the values below (needs fpcalc)."
        )
        adv_form = QtWidgets.QFormLayout(adv_group)
        adv_form.setContentsMargins(8, 4, 8, 8)

        self._trim_cb = QtWidgets.QCheckBox("Trim leading/trailing silence")
        self._trim_cb.setChecked(_DEFAULT_FP_PARAMS["trim"])
        adv_form.addRow(self._trim_cb)

        self._fp_offset = QtWidgets.QLineEdit("0")
        self._fp_dur = QtWidgets.QLineEdit("120000")
        self._silence_thresh = QtWidgets.QLineEdit("-50")
        self._silence_min = QtWidgets.QLineEdit("500")
        self._exact_t = QtWidgets.QLineEdit()
        self._near_t = QtWidgets.QLineEdit()
        self._codec_boost = QtWidgets.QLineEdit()

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

    # ── Parameters ────────────────────────────────────────────────────────

    def _show_config_thresholds(self) -> None:
        """Fill the (disabled) threshold fields with the Settings values."""
        if self._adv_group.isChecked():
            return
        from gui.workspaces.duplicates import _config_thresholds

        exact, near, boost = _config_thresholds()
        self._exact_t.setText(f"{exact:g}")
        self._near_t.setText(f"{near:g}")
        self._codec_boost.setText(f"{boost:g}")

    def showEvent(self, event) -> None:  # noqa: N802
        self._show_config_thresholds()
        super().showEvent(event)

    def _collect_params(self) -> dict | None:
        """Parameters for the worker, or None after telling the user what's wrong."""
        if not self._adv_group.isChecked():
            from gui.workspaces.duplicates import _config_thresholds

            exact, near, boost = _config_thresholds()
            return {
                "exact_threshold": exact,
                "near_threshold": near,
                "codec_boost": boost,
                "threshold_source": "Settings",
                "fingerprint": None,
            }

        fields = [
            ("Exact threshold", self._exact_t, 0.0, 1.0),
            ("Near threshold", self._near_t, 0.0, 1.0),
            ("Mixed-codec boost", self._codec_boost, 0.0, 1.0),
            ("Fingerprint offset (ms)", self._fp_offset, 0.0, None),
            ("Fingerprint duration (ms)", self._fp_dur, 1.0, None),
            ("Silence threshold (dB)", self._silence_thresh, None, 0.0),
            ("Silence min length (ms)", self._silence_min, 0.0, None),
        ]
        values: dict[str, float] = {}
        for label, widget, low, high in fields:
            try:
                value = float(widget.text().strip())
            except ValueError:
                value = None
            if value is None or (low is not None and value < low) or (high is not None and value > high):
                QtWidgets.QMessageBox.warning(
                    self, "Invalid value",
                    f"{label}: {widget.text()!r} is not a valid number"
                    + (f" (minimum {low:g})" if low is not None else "")
                    + (f" (maximum {high:g})" if high is not None else "")
                    + ".",
                )
                widget.setFocus()
                return None
            values[label] = value
        return {
            "exact_threshold": values["Exact threshold"],
            "near_threshold": values["Near threshold"],
            "codec_boost": values["Mixed-codec boost"],
            "threshold_source": "Advanced Overrides",
            "fingerprint": {
                "trim": self._trim_cb.isChecked(),
                "start_sec": values["Fingerprint offset (ms)"] / 1000.0,
                "duration_sec": values["Fingerprint duration (ms)"] / 1000.0,
                "threshold_db": values["Silence threshold (dB)"],
                "min_silence_duration": values["Silence min length (ms)"] / 1000.0,
            },
        }

    # ── Slots ─────────────────────────────────────────────────────────────

    def _status(self, text: str, kind: str) -> None:
        tokens = get_manager().current
        colour = {"busy": tokens.warning, "ok": tokens.success, "error": tokens.danger}.get(
            kind, tokens.text_secondary
        )
        self.status_changed.emit(text, colour)

    @Slot()
    def _on_run(self) -> None:
        if self._worker is not None:
            return
        a = self._file_a.text().strip()
        b = self._file_b.text().strip()
        if not a or not b:
            QtWidgets.QMessageBox.warning(self, "Missing Files", "Please select both files.")
            return
        missing = [p for p in (a, b) if not os.path.isfile(p)]
        if missing:
            QtWidgets.QMessageBox.warning(self, "File not found", "Not found:\n" + "\n".join(missing))
            return
        params = self._collect_params()
        if params is None:
            return

        self._run_btn.setEnabled(False)
        self._results_text.setPlainText("Computing fingerprints…")
        self._log("Running similarity comparison…", "info")
        self._status("Comparing…", "busy")

        self._worker = SimilarityWorker(a, b, params, self._library_path)
        self._worker.done.connect(self._on_finished)
        self._start_worker(self._worker, "Similarity comparison")

    @Slot()
    def _on_clear(self) -> None:
        if self._worker is not None:
            # Its late result must not overwrite the cleared view.
            self._worker.cancel()
            self._worker = None
            self._run_btn.setEnabled(True)
        self._file_a.clear()
        self._file_b.clear()
        self._results_text.clear()

    @Slot(bool, str)
    def _on_finished(self, success: bool, report: str) -> None:
        if self.sender() is not None and self.sender() is not self._worker:
            return  # cleared meanwhile
        self._worker = None
        self._run_btn.setEnabled(True)
        self._results_text.setPlainText(report)
        if success:
            self._log("Comparison complete.", "ok")
            self._status("Done", "ok")
        else:
            self._log(f"Comparison failed: {report}", "error")
            self._status("Error", "error")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _browse_file(self, entry: QtWidgets.QLineEdit) -> None:
        start = str(Path(entry.text()).parent) if entry.text() else (self._library_path or str(Path.home()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Audio File", start,
            "Audio files (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All files (*)"
        )
        if path:
            entry.setText(path)
