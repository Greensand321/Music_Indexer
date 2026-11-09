"""Duplicates → Compare two files: why two tracks do (or do not) match."""
from __future__ import annotations

import os
import threading
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes import motion
from gui.themes.glass import apply_font
from gui.themes.manager import get_manager
from gui.themes.tokens import RADIUS, SPACE
from gui.widgets.glass import GlassPanel, path_label
from gui.widgets.glass_parts import EmptyState, TileGrid, hero_title
from gui.widgets.glyphs import set_glyph
from gui.widgets.layouts import fit_stack_to_current
from gui.widgets.pill import ToneBadge
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
        # Set before ``done(True, …)`` is emitted; read by the workspace's slot.
        self.details: dict | None = None

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
            from near_duplicate_detector import (
                LOSSLESS_EXTS,
                decode_chromaprint,
                duplicate_fingerprint_distance,
                fingerprint_distance,
            )

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

            # The verdict uses the Duplicate Finder's own distance, which counts
            # the length the two prints don't share: a 30-second clip is not a
            # duplicate of its full song even though the overlap is identical.
            distance = duplicate_fingerprint_distance(fp_a, fp_b)
            overlap = fingerprint_distance(fp_a, fp_b)
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
                f"Overlap-only distance:    {overlap:.6f}"
                "   (ignores length the two don't share; a clip of a song scores near 0 here)",
                f"Exact threshold:          {exact_thresh:.4f}",
                f"Near threshold:           {near_thresh:.4f}",
                f"Codec boost applied:      {codec_boost if mixed else 0.0:.4f}",
                f"Effective threshold:      {effective:.4f}",
                f"Thresholds from:          {self.params.get('threshold_source', 'Settings')}",
                "",
                f"VERDICT: {verdict}",
            ]
            # The numbers behind the report, for the page's verdict and tiles.
            self.details = {
                "verdict": verdict,
                "distance": distance,
                "overlap": overlap,
                "exact": exact_thresh,
                "near": near_thresh,
                "boost": codec_boost if mixed else 0.0,
                "effective": effective,
                "mixed": mixed,
                "ext_a": ext_a,
                "ext_b": ext_b,
                "kind_a": _kind(ext_a),
                "kind_b": _kind(ext_b),
                "source_a": source_a,
                "source_b": source_b,
                "threshold_source": self.params.get("threshold_source", "Settings"),
            }
            self.done.emit(True, "\n".join(report_lines))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


_AUDIO_SUFFIXES = (".flac", ".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus")


def _explain(details: dict) -> str:
    """The verdict in words: which threshold decided it, and why."""
    distance, effective, exact = details["distance"], details["effective"], details["exact"]
    verdict = details["verdict"]
    if verdict == "EXACT DUPLICATE":
        lines = [f"Distance {distance:.4f} is within the exact threshold ({exact:.4f}): "
                 "Find duplicates groups these as exact duplicates."]
    elif verdict == "NEAR DUPLICATE":
        lines = [f"Distance {distance:.4f} is above the exact threshold ({exact:.4f}) but within "
                 f"the effective threshold ({effective:.4f}): Find duplicates groups these as near "
                 "duplicates."]
    else:
        lines = [f"Distance {distance:.4f} is above the effective threshold ({effective:.4f}): "
                 "Find duplicates keeps these apart."]
        if details["overlap"] <= effective:
            lines.append(f"Where the two overlap they match ({details['overlap']:.4f}), so one is "
                         "probably a clip or an edit of the other.")
    kinds = (f"A is {details['ext_a'] or '?'} ({details['kind_a']}), "
             f"B is {details['ext_b'] or '?'} ({details['kind_b']})")
    if details["mixed"]:
        lines.append(f"{kinds}: mixed codecs add {details['boost']:.4f} to the near threshold.")
    else:
        lines.append(f"{kinds}.")
    return " ".join(lines)


class _FileRow(GlassPanel):
    """One file to compare: a letter badge, the elided path and Browse….
    Dropping an audio file on the row picks it too."""

    browse_requested = Signal(object)   # the row's PathLabel

    def __init__(self, letter: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("well", radius=RADIUS.sm, parent=parent)
        self.setObjectName(f"similarityFile{letter}")
        self.setAcceptDrops(True)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 6, 6)
        lay.setSpacing(SPACE.sm)
        badge = ToneBadge(letter, "accent", weight=700, pad_x=9)
        lay.addWidget(badge, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.path_label = path_label("", "No file chosen — Browse, or drop an audio file here")
        self.path_label.setAccessibleName(f"File {letter}")
        lay.addWidget(self.path_label, 1)
        browse = QtWidgets.QPushButton("Browse…")
        set_glyph(browse, "folder")
        browse.setToolTip(f"Choose file {letter}")
        browse.setAccessibleName(f"Browse for file {letter}")
        browse.clicked.connect(lambda _checked=False: self.browse_requested.emit(self.path_label))
        self.browse_button = browse
        lay.addWidget(browse)

    @staticmethod
    def _dropped_file(event) -> str:
        mime = event.mimeData()
        if not mime.hasUrls():
            return ""
        for url in mime.urls():
            path = url.toLocalFile()
            if path and os.path.isfile(path) and path.lower().endswith(_AUDIO_SUFFIXES):
                return path
        return ""

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        if self._dropped_file(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        path = self._dropped_file(event)
        if path:
            self.path_label.setText(path)
            event.acceptProposedAction()


class SimilarityWorkspace(WorkspaceBase):
    """Compare two tracks to understand why they do or do not match."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: SimilarityWorker | None = None
        self._build_ui()
        self._show_config_thresholds()

    def _build_ui(self) -> None:
        cl = self.content_layout

        # ── Hero: the two files and Compare ───────────────────────────────
        hero = self._make_card()
        hero.setObjectName("headerCard")
        hl = QtWidgets.QVBoxLayout(hero)
        hl.setContentsMargins(16, 16, 16, 16)
        hl.setSpacing(SPACE.sm)
        title = hero_title("Pick two files")
        hl.addWidget(title)
        hl.addWidget(self._make_subtitle(
            "Uses the same fingerprints and thresholds as Find duplicates, to show why "
            "they match or not."
        ))
        hl.addSpacing(SPACE.xxs)
        self._file_rows: list[_FileRow] = []
        for attr, letter in (("_file_a", "A"), ("_file_b", "B")):
            row = _FileRow(letter, self)
            row.browse_requested.connect(self._browse_file)
            self._file_rows.append(row)
            setattr(self, attr, row.path_label)
            hl.addWidget(row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(SPACE.sm)
        btn_row.addStretch(1)
        self._close_btn = QtWidgets.QPushButton("Clear")
        set_glyph(self._close_btn, "close")
        self._close_btn.setToolTip("Forget both files and the result (a running comparison is discarded).")
        self._close_btn.clicked.connect(self._on_clear)
        self._run_btn = self._make_primary_button("Compare")
        set_glyph(self._run_btn, "search", role="inverse")
        self._run_btn.setToolTip("Fingerprint both files and give Find duplicates' verdict (read-only).")
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._close_btn)
        btn_row.addWidget(self._run_btn)
        hl.addLayout(btn_row)
        cl.addWidget(hero)

        # ── Result: verdict, distances, explanation ──────────────────────
        result_card = self._make_card()
        rl = QtWidgets.QVBoxLayout(result_card)
        rl.setContentsMargins(16, 16, 16, 16)
        rl.setSpacing(SPACE.md)
        rl.addWidget(self._make_card_title("Verdict"))
        self._result_stack = QtWidgets.QStackedWidget()
        self._empty = EmptyState(
            "No comparison yet",
            "Choose two files above, then Compare. The verdict and the distances appear here.",
        )
        self._result_stack.addWidget(self._empty)

        page = QtWidgets.QWidget()
        self._result_page = page
        pl = QtWidgets.QVBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(SPACE.md)
        verdict_row = QtWidgets.QHBoxLayout()
        verdict_row.setSpacing(SPACE.sm)
        self._verdict = ToneBadge("", "", role="title", weight=700, pad_x=18, pad_y=6)
        self._verdict.setObjectName("similarityVerdict")
        verdict_row.addWidget(self._verdict, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        verdict_row.addStretch(1)
        self._source_lbl = QtWidgets.QLabel("")
        self._source_lbl.setObjectName("mutedLabel")
        apply_font(self._source_lbl, "caption")
        verdict_row.addWidget(self._source_lbl, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        pl.addLayout(verdict_row)

        self._stats = TileGrid([
            ("distance", "Fingerprint distance"),
            ("overlap", "Overlap-only distance"),
            ("effective", "Effective threshold"),
            ("exact", "Exact threshold"),
        ], min_tile_width=140)
        self._stats.tile("distance").setToolTip(
            "Bit error rate over the whole length of both prints: 0 = identical, about 0.5 = unrelated. "
            "The verdict uses this."
        )
        self._stats.tile("overlap").setToolTip(
            "Ignores length the two don't share: a clip of a song scores near 0 here."
        )
        self._stats.tile("effective").setToolTip(
            "The near threshold, plus the mixed-codec boost when one file is lossless and the other lossy."
        )
        pl.addWidget(self._stats)

        self._explain_lbl = QtWidgets.QLabel("")
        self._explain_lbl.setObjectName("mutedLabel")
        self._explain_lbl.setWordWrap(True)
        self._explain_lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        apply_font(self._explain_lbl, "label")
        pl.addWidget(self._explain_lbl)

        self._report_toggle = QtWidgets.QPushButton("Full report")
        self._report_toggle.setFlat(True)
        self._report_toggle.setCheckable(True)
        self._report_toggle.setToolTip("Show or hide the complete comparison report.")
        set_glyph(self._report_toggle, "chevron_right", role="muted")
        self._report_toggle.toggled.connect(self._on_report_toggled)
        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.addWidget(self._report_toggle)
        toggle_row.addStretch(1)
        pl.addLayout(toggle_row)

        self._results_text = QtWidgets.QPlainTextEdit()
        self._results_text.setObjectName("logText")
        self._results_text.setReadOnly(True)
        self._results_text.setMinimumHeight(240)
        self._results_text.setPlaceholderText("Results will appear here after running the comparison…")
        self._results_text.setAccessibleName("Comparison report")
        self._results_text.setVisible(False)
        pl.addWidget(self._results_text)
        self._result_stack.addWidget(page)
        fit_stack_to_current(self._result_stack)
        rl.addWidget(self._result_stack)
        cl.addWidget(result_card)

        # ── Advanced overrides (off: exactly the Duplicate Finder's settings) ─
        adv_card = self._make_card()
        al = QtWidgets.QVBoxLayout(adv_card)
        al.setContentsMargins(16, 14, 16, 14)
        al.setSpacing(SPACE.sm)
        self._adv_group = QtWidgets.QCheckBox("Advanced overrides")
        apply_font(self._adv_group, "heading")
        self._adv_group.setToolTip(
            "Off: fingerprint exactly like Find duplicates (and reuse its cache) and use "
            "the thresholds saved in Settings. On: use the values below (needs fpcalc)."
        )
        al.addWidget(self._adv_group)
        self._adv_summary = QtWidgets.QLabel("")
        self._adv_summary.setObjectName("mutedLabel")
        self._adv_summary.setWordWrap(True)
        apply_font(self._adv_summary, "label")
        al.addWidget(self._adv_summary)

        self._adv_box = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(self._adv_box)
        grid.setContentsMargins(0, SPACE.xxs, 0, 0)
        grid.setHorizontalSpacing(SPACE.lg)
        grid.setVerticalSpacing(SPACE.xxs)

        self._trim_cb = QtWidgets.QCheckBox("Trim leading/trailing silence")
        self._trim_cb.setChecked(_DEFAULT_FP_PARAMS["trim"])
        grid.addWidget(self._trim_cb, 0, 0, 1, 2)

        self._fp_offset = QtWidgets.QLineEdit("0")
        self._fp_dur = QtWidgets.QLineEdit("120000")
        self._silence_thresh = QtWidgets.QLineEdit("-50")
        self._silence_min = QtWidgets.QLineEdit("500")
        self._exact_t = QtWidgets.QLineEdit()
        self._near_t = QtWidgets.QLineEdit()
        self._codec_boost = QtWidgets.QLineEdit()

        fields = [
            ("Fingerprint offset (ms)", self._fp_offset),
            ("Fingerprint duration (ms)", self._fp_dur),
            ("Silence threshold (dB)", self._silence_thresh),
            ("Silence min length (ms)", self._silence_min),
            ("Exact threshold", self._exact_t),
            ("Near threshold", self._near_t),
            ("Mixed-codec boost", self._codec_boost),
        ]
        for i, (label, widget) in enumerate(fields):
            row, col = 1 + (i // 2) * 2, i % 2
            caption = QtWidgets.QLabel(label)
            caption.setObjectName("mutedLabel")
            apply_font(caption, "caption")
            caption.setBuddy(widget)
            widget.setAccessibleName(label)
            grid.addWidget(caption, row, col)
            grid.addWidget(widget, row + 1, col)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        al.addWidget(self._adv_box)
        self._adv_group.toggled.connect(self._on_advanced_toggled)
        self._on_advanced_toggled(False)
        cl.addWidget(adv_card)

        cl.addStretch(1)

    @Slot(bool)
    def _on_advanced_toggled(self, on: bool) -> None:
        """Off: the fields show the Settings values, disabled and folded away
        (the summary line names them). On: they are editable."""
        self._adv_box.setEnabled(on)
        self._adv_box.setVisible(on)
        self._adv_summary.setVisible(not on)
        if not on:
            self._show_config_thresholds()

    @Slot(bool)
    def _on_report_toggled(self, shown: bool) -> None:
        self._results_text.setVisible(shown)
        set_glyph(self._report_toggle, "chevron_down" if shown else "chevron_right", role="muted")

    def _show_verdict(self, details: dict | None, report: str, *, running: bool = False) -> None:
        """Fill the result card: verdict pill, distance tiles, the explanation."""
        tiles = self._stats
        if self._result_stack.currentWidget() is not self._result_page:
            self._result_stack.setCurrentWidget(self._result_page)
            motion.reveal(self._result_page)
        if details is None:
            for key in tiles.keys():
                tiles.tile(key).set_value("–", tone="")
            if running:
                self._verdict.set_text("Comparing…", "")
                self._explain_lbl.setText("Computing fingerprints…")
                self._source_lbl.setText("")
            else:
                self._verdict.set_text("Could not compare", "danger")
                self._explain_lbl.setText(report)
                self._source_lbl.setText("")
            return
        verdict = details["verdict"]
        tone = {"EXACT DUPLICATE": "success", "NEAR DUPLICATE": "accent"}.get(verdict, "")
        self._verdict.set_text(verdict.capitalize(), tone)
        tiles.tile("distance").set_value(f"{details['distance']:.4f}", tone=tone)
        tiles.tile("overlap").set_value(f"{details['overlap']:.4f}")
        tiles.tile("effective").set_value(f"{details['effective']:.4f}")
        tiles.tile("exact").set_value(f"{details['exact']:.4f}")
        self._source_lbl.setText(f"Thresholds from {details['threshold_source']}")
        self._explain_lbl.setText(_explain(details))

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
        self._adv_summary.setText(
            "Off: the fingerprints Find duplicates uses and the thresholds saved in Settings "
            f"(exact {exact:g} · near {near:g} · mixed-codec boost {boost:g})."
        )

    def on_settings_saved(self) -> None:
        self._show_config_thresholds()

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
        self._show_verdict(None, "", running=True)
        self._log("Comparing two files…", "info")
        self._status("Comparing…", "busy")

        self._worker = SimilarityWorker(a, b, params, self._library_path)
        self._worker.done.connect(self._on_finished)
        self._start_worker(self._worker, "Compare two files")

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
        self._result_stack.setCurrentWidget(self._empty)

    @Slot(bool, str)
    def _on_finished(self, success: bool, report: str) -> None:
        worker = self.sender()
        if worker is not None and worker is not self._worker:
            return  # cleared meanwhile
        worker = worker or self._worker
        self._worker = None
        self._run_btn.setEnabled(True)
        self._results_text.setPlainText(report)
        details = getattr(worker, "details", None) if success else None
        self._show_verdict(details, report)
        if success:
            self._log("Comparison complete.", "ok")
            self._status("Done", "ok")
        else:
            self._log(f"Comparison failed: {report}", "error")
            self._status("Error", "error")

    # ── Helpers ───────────────────────────────────────────────────────────

    @Slot(object)
    def _browse_file(self, entry: PathLabel) -> None:
        start = str(Path(entry.text()).parent) if entry.text() else (self._library_path or str(Path.home()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Audio File", start,
            "Audio files (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All files (*)"
        )
        if path:
            entry.setText(path)
