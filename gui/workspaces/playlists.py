"""Playlist Generator workspace — m3u playlists, Auto-DJ, folder playlists."""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class PlaylistWorker(QtCore.QThread):
    log_line = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, library_path: str, mode: str, output_dir: str,
                 tempo_range: tuple, energy_range: tuple, autodj_count: int) -> None:
        super().__init__()
        self.library_path = library_path
        self.mode = mode
        self.output_dir = output_dir
        self.tempo_range = tempo_range
        self.energy_range = energy_range
        self.autodj_count = autodj_count
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import playlist_generator
        except ImportError as exc:
            self.finished.emit(False, f"Import error: {exc}")
            return
        try:
            def _log(msg: str) -> None:
                if not self._cancelled:
                    self.log_line.emit(msg)

            if self.mode == "folder":
                playlist_generator.create_folder_playlists(
                    self.library_path, output_dir=self.output_dir, log_callback=_log
                )
            elif self.mode == "tempo":
                playlist_generator.create_tempo_playlists(
                    self.library_path,
                    tempo_range=self.tempo_range,
                    output_dir=self.output_dir,
                    log_callback=_log,
                )
            elif self.mode == "autodj":
                playlist_generator.create_autodj_playlist(
                    self.library_path,
                    count=self.autodj_count,
                    output_dir=self.output_dir,
                    log_callback=_log,
                )
            if not self._cancelled:
                self.finished.emit(True, "Playlist generation complete.")
            else:
                self.finished.emit(False, "Cancelled.")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, str(exc))


class PlaylistsWorkspace(WorkspaceBase):
    """Generate m3u playlists by folder, tempo/energy, or Auto-DJ."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: PlaylistWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Playlist Generator"))
        cl.addWidget(self._make_subtitle(
            "Build .m3u playlists from your library by folder structure, "
            "tempo/energy buckets, or Auto-DJ flow. "
            "Generated playlists are saved under Playlists/ in your library."
        ))

        # ── Tabs for playlist modes ────────────────────────────────────────
        tabs = QtWidgets.QTabWidget()

        # ── Folder Playlists tab ───────────────────────────────────────────
        folder_widget = QtWidgets.QWidget()
        folder_layout = QtWidgets.QVBoxLayout(folder_widget)
        folder_layout.setContentsMargins(16, 16, 16, 16)
        folder_layout.setSpacing(10)

        folder_layout.addWidget(self._make_subtitle(
            "Create one playlist per top-level artist folder. "
            "Each playlist contains all tracks in that artist's subfolders."
        ))

        self._folder_out_entry = QtWidgets.QLineEdit()
        self._folder_out_entry.setPlaceholderText("Leave blank to use Playlists/ inside library")
        folder_out_row = QtWidgets.QHBoxLayout()
        folder_out_row.addWidget(QtWidgets.QLabel("Output folder:"))
        folder_out_row.addWidget(self._folder_out_entry, 1)
        browse_folder = QtWidgets.QPushButton("Browse…")
        browse_folder.clicked.connect(lambda: self._browse_dir(self._folder_out_entry))
        folder_out_row.addWidget(browse_folder)
        folder_layout.addLayout(folder_out_row)

        folder_btn = self._make_primary_button("🎵  Generate Folder Playlists")
        folder_btn.clicked.connect(lambda: self._on_run("folder"))
        folder_layout.addWidget(folder_btn)
        folder_layout.addStretch(1)
        tabs.addTab(folder_widget, "Folder Playlists")

        # ── Tempo/Energy Playlists tab ─────────────────────────────────────
        tempo_widget = QtWidgets.QWidget()
        tempo_layout = QtWidgets.QVBoxLayout(tempo_widget)
        tempo_layout.setContentsMargins(16, 16, 16, 16)
        tempo_layout.setSpacing(10)

        tempo_layout.addWidget(self._make_subtitle(
            "Group tracks into playlists by BPM and energy level. "
            "Uses librosa for feature extraction — may take a while on large libraries."
        ))

        tform = QtWidgets.QFormLayout()
        tform.setSpacing(8)

        tempo_range_row = QtWidgets.QHBoxLayout()
        self._tempo_min = QtWidgets.QSpinBox()
        self._tempo_min.setRange(0, 300)
        self._tempo_min.setValue(60)
        self._tempo_max = QtWidgets.QSpinBox()
        self._tempo_max.setRange(0, 300)
        self._tempo_max.setValue(180)
        tempo_range_row.addWidget(self._tempo_min)
        tempo_range_row.addWidget(QtWidgets.QLabel("–"))
        tempo_range_row.addWidget(self._tempo_max)
        tempo_range_row.addWidget(QtWidgets.QLabel("BPM"))
        tempo_range_row.addStretch(1)
        tform.addRow("Tempo range:", tempo_range_row)

        energy_range_row = QtWidgets.QHBoxLayout()
        self._energy_min = QtWidgets.QDoubleSpinBox()
        self._energy_min.setRange(0.0, 1.0)
        self._energy_min.setSingleStep(0.05)
        self._energy_min.setValue(0.0)
        self._energy_max = QtWidgets.QDoubleSpinBox()
        self._energy_max.setRange(0.0, 1.0)
        self._energy_max.setSingleStep(0.05)
        self._energy_max.setValue(1.0)
        energy_range_row.addWidget(self._energy_min)
        energy_range_row.addWidget(QtWidgets.QLabel("–"))
        energy_range_row.addWidget(self._energy_max)
        energy_range_row.addStretch(1)
        tform.addRow("Energy range:", energy_range_row)
        tempo_layout.addLayout(tform)

        tempo_btn = self._make_primary_button("🎵  Generate Tempo Playlists")
        tempo_btn.clicked.connect(lambda: self._on_run("tempo"))
        tempo_layout.addWidget(tempo_btn)
        tempo_layout.addStretch(1)
        tabs.addTab(tempo_widget, "Tempo / Energy")

        # ── Auto-DJ tab ────────────────────────────────────────────────────
        autodj_widget = QtWidgets.QWidget()
        autodj_layout = QtWidgets.QVBoxLayout(autodj_widget)
        autodj_layout.setContentsMargins(16, 16, 16, 16)
        autodj_layout.setSpacing(10)

        autodj_layout.addWidget(self._make_subtitle(
            "Build a smooth transition playlist using similarity-based track ordering. "
            "Tracks are chained so each song is sonically close to the next."
        ))

        count_row = QtWidgets.QHBoxLayout()
        count_row.addWidget(QtWidgets.QLabel("Track count:"))
        self._autodj_count = QtWidgets.QSpinBox()
        self._autodj_count.setRange(5, 500)
        self._autodj_count.setValue(20)
        count_row.addWidget(self._autodj_count)
        count_row.addStretch(1)
        autodj_layout.addLayout(count_row)

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(QtWidgets.QLabel("Seed track (optional):"))
        self._autodj_seed = QtWidgets.QLineEdit()
        self._autodj_seed.setPlaceholderText("Path to starting track…")
        seed_browse = QtWidgets.QPushButton("Browse…")
        seed_browse.clicked.connect(lambda: self._browse_file(self._autodj_seed))
        seed_row.addWidget(self._autodj_seed, 1)
        seed_row.addWidget(seed_browse)
        autodj_layout.addLayout(seed_row)

        autodj_btn = self._make_primary_button("🎵  Generate Auto-DJ Playlist")
        autodj_btn.clicked.connect(lambda: self._on_run("autodj"))
        autodj_layout.addWidget(autodj_btn)
        autodj_layout.addStretch(1)
        tabs.addTab(autodj_widget, "Auto-DJ")

        # ── Playlist Repair tab ────────────────────────────────────────────
        repair_widget = QtWidgets.QWidget()
        repair_layout = QtWidgets.QVBoxLayout(repair_widget)
        repair_layout.setContentsMargins(16, 16, 16, 16)
        repair_layout.setSpacing(10)

        repair_layout.addWidget(self._make_subtitle(
            "Re-scan playlists for broken paths after you've moved or renamed files. "
            "Each .m3u / .m3u8 file is checked and missing entries are updated."
        ))

        self._repair_list = QtWidgets.QListWidget()
        self._repair_list.setMinimumHeight(120)
        repair_layout.addWidget(self._repair_list)

        repair_add_row = QtWidgets.QHBoxLayout()
        repair_add_btn = QtWidgets.QPushButton("Add Playlists…")
        repair_add_btn.clicked.connect(self._on_add_playlists)
        repair_rem_btn = QtWidgets.QPushButton("Remove Selected")
        repair_rem_btn.clicked.connect(
            lambda: self._repair_list.takeItem(self._repair_list.currentRow())
        )
        repair_clear_btn = QtWidgets.QPushButton("Clear")
        repair_clear_btn.clicked.connect(self._repair_list.clear)
        self._prefer_opus_cb = QtWidgets.QCheckBox("Prefer Opus when searching for missing FLAC")
        repair_add_row.addWidget(repair_add_btn)
        repair_add_row.addWidget(repair_rem_btn)
        repair_add_row.addWidget(repair_clear_btn)
        repair_add_row.addWidget(self._prefer_opus_cb)
        repair_add_row.addStretch(1)
        repair_layout.addLayout(repair_add_row)

        repair_run_btn = self._make_primary_button("🔧  Run Repair")
        repair_run_btn.clicked.connect(lambda: self._on_run("repair"))
        repair_layout.addWidget(repair_run_btn)
        repair_layout.addStretch(1)
        tabs.addTab(repair_widget, "Playlist Repair")

        cl.addWidget(tabs)

        # ── Progress + Log ─────────────────────────────────────────────────
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setRange(0, 0)  # indeterminate
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setVisible(False)
        cl.addWidget(self._prog_bar)

        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(QtWidgets.QLabel("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(130)
        self._log_area.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        self._log_area.setPlaceholderText(
            "The app switches to the Activity Log automatically during playlist generation."
        )
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_run(self, mode: str) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return

        self._log_area.clear()
        self._prog_bar.setVisible(True)
        self._log(f"Starting {mode} playlist generation…", "info")
        self.status_changed.emit("Generating playlists…", "#f59e0b")

        self._worker = PlaylistWorker(
            library_path=self._library_path,
            mode=mode,
            output_dir=self._folder_out_entry.text() if hasattr(self, "_folder_out_entry") else "",
            tempo_range=(self._tempo_min.value(), self._tempo_max.value()),
            energy_range=(self._energy_min.value(), self._energy_max.value()),
            autodj_count=self._autodj_count.value(),
        )
        self._worker.log_line.connect(self._log_area.appendPlainText)
        self._worker.log_line.connect(lambda m: self._log(m))
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot()
    def _on_add_playlists(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Playlists", str(Path.home()),
            "Playlists (*.m3u *.m3u8);;All files (*)"
        )
        for p in paths:
            self._repair_list.addItem(p)

    @Slot(bool, str)
    def _on_finished(self, success: bool, message: str) -> None:
        self._prog_bar.setVisible(False)
        if success:
            self._log(message, "ok")
            self.status_changed.emit("Done", "#22c55e")
        else:
            self._log(message, "error" if "Cancelled" not in message else "warn")
            self.status_changed.emit("Error", "#ef4444")
        self._worker = None

    def _browse_dir(self, entry: QtWidgets.QLineEdit) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            entry.setText(folder)

    def _browse_file(self, entry: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All (*)"
        )
        if path:
            entry.setText(path)
