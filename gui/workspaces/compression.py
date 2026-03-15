"""Library Compression workspace — transcode and archive high-bitrate files."""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class CompressionWorkspace(WorkspaceBase):
    """Transcode and archive settings (currently a planning/configuration panel)."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Library Compression"))
        cl.addWidget(self._make_subtitle(
            "Transcode and archive high-bitrate tracks to save disk space. "
            "Configure codec targets, bitrate caps, and which formats to re-encode. "
            "Outputs land in a separate archive folder so originals are preserved."
        ))

        # ── Format targets card ────────────────────────────────────────────
        fmt_card = self._make_card()
        fmt_layout = QtWidgets.QVBoxLayout(fmt_card)
        fmt_layout.setContentsMargins(16, 16, 16, 16)
        fmt_layout.setSpacing(10)
        fmt_layout.addWidget(QtWidgets.QLabel("Transcoding Targets"))

        target_form = QtWidgets.QFormLayout()
        target_form.setSpacing(8)

        self._flac_target = QtWidgets.QComboBox()
        self._flac_target.addItems(["Keep FLAC (no change)", "→ OPUS", "→ MP3", "→ AAC"])
        target_form.addRow("FLAC:", self._flac_target)

        self._wav_target = QtWidgets.QComboBox()
        self._wav_target.addItems(["→ FLAC (lossless)", "→ OPUS", "→ MP3", "Keep WAV"])
        target_form.addRow("WAV:", self._wav_target)

        self._mp3_target = QtWidgets.QComboBox()
        self._mp3_target.addItems(["Keep MP3 (no change)", "→ OPUS (smaller)", "→ AAC"])
        target_form.addRow("MP3:", self._mp3_target)

        fmt_layout.addLayout(target_form)

        bitrate_row = QtWidgets.QHBoxLayout()
        bitrate_row.addWidget(QtWidgets.QLabel("Target bitrate (kbps):"))
        self._bitrate_spin = QtWidgets.QSpinBox()
        self._bitrate_spin.setRange(64, 512)
        self._bitrate_spin.setValue(192)
        bitrate_row.addWidget(self._bitrate_spin)
        bitrate_row.addStretch(1)
        fmt_layout.addLayout(bitrate_row)
        cl.addWidget(fmt_card)

        # ── Archive options card ───────────────────────────────────────────
        arch_card = self._make_card()
        arch_layout = QtWidgets.QVBoxLayout(arch_card)
        arch_layout.setContentsMargins(16, 16, 16, 16)
        arch_layout.setSpacing(8)
        arch_layout.addWidget(QtWidgets.QLabel("Archive Options"))

        self._keep_originals_cb = QtWidgets.QCheckBox("Keep originals in Archive/ subfolder")
        self._keep_originals_cb.setChecked(True)
        self._keep_originals_cb.setToolTip(
            "When checked, original files are moved to Archive/ rather than deleted."
        )
        arch_layout.addWidget(self._keep_originals_cb)

        self._update_playlists_cb = QtWidgets.QCheckBox("Update playlists after transcoding")
        self._update_playlists_cb.setChecked(True)
        arch_layout.addWidget(self._update_playlists_cb)

        self._dry_run_cb = QtWidgets.QCheckBox("Dry run (estimate space savings only)")
        self._dry_run_cb.setChecked(True)
        arch_layout.addWidget(self._dry_run_cb)
        cl.addWidget(arch_card)

        # ── Action buttons ─────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._estimate_btn = self._make_primary_button("📦  Estimate Space Savings")
        self._estimate_btn.clicked.connect(self._on_estimate)
        self._run_btn = QtWidgets.QPushButton("▶  Start Compression")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._estimate_btn)
        btn_row.addWidget(self._run_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        # ── Estimate results ───────────────────────────────────────────────
        est_card = self._make_card()
        est_layout = QtWidgets.QVBoxLayout(est_card)
        est_layout.setContentsMargins(16, 16, 16, 16)
        est_layout.addWidget(QtWidgets.QLabel("Estimate / Progress"))
        self._result_lbl = QtWidgets.QLabel(
            "Click 'Estimate Space Savings' to see a breakdown before committing."
        )
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setStyleSheet("color: #64748b;")
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setValue(0)
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setVisible(False)
        est_layout.addWidget(self._result_lbl)
        est_layout.addWidget(self._prog_bar)
        cl.addWidget(est_card)

        note = QtWidgets.QLabel(
            "Note: FFmpeg must be installed and on your PATH for transcoding."
        )
        note.setStyleSheet("color: #94a3b8; font-size: 11px;")
        cl.addWidget(note)

        cl.addStretch(1)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_estimate(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        self._log("Compression estimation not yet wired to backend.", "warn")
        self._result_lbl.setText(
            "Estimation requires FFmpeg and the compression backend.\n"
            "Not yet wired — feature coming soon."
        )
        self._run_btn.setEnabled(False)

    @Slot()
    def _on_run(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm",
            "This will transcode files in your library.\n\nProceed?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._log("Compression run not yet wired to backend.", "warn")
