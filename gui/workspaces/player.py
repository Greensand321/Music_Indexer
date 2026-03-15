"""Player workspace — in-app audio preview using libVLC."""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.workspaces.base import WorkspaceBase


class PlayerWorkspace(WorkspaceBase):
    """Audio preview player with metadata display."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._vlc_instance = None
        self._media_player = None
        self._current_file: str = ""
        self._timer = QtCore.QTimer()
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._update_position)
        self._build_ui()
        self._init_vlc()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Player"))
        cl.addWidget(self._make_subtitle(
            "Preview audio files directly in the app. "
            "Requires VLC / libVLC installed on your system. "
            "Used for quick validation during duplicate review and similarity inspection."
        ))

        # ── File selector card ─────────────────────────────────────────────
        file_card = self._make_card()
        file_layout = QtWidgets.QVBoxLayout(file_card)
        file_layout.setContentsMargins(16, 16, 16, 16)
        file_layout.setSpacing(10)

        file_row = QtWidgets.QHBoxLayout()
        self._file_entry = QtWidgets.QLineEdit()
        self._file_entry.setPlaceholderText("Select an audio file to preview…")
        self._file_entry.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        file_row.addWidget(self._file_entry, 1)
        file_row.addWidget(browse_btn)
        file_layout.addLayout(file_row)

        # ── Artwork + metadata ─────────────────────────────────────────────
        meta_row = QtWidgets.QHBoxLayout()

        self._artwork_lbl = QtWidgets.QLabel()
        self._artwork_lbl.setFixedSize(96, 96)
        self._artwork_lbl.setStyleSheet(
            "background: #e2e8f0; border-radius: 6px; color: #94a3b8;"
        )
        self._artwork_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._artwork_lbl.setText("🎵")
        meta_row.addWidget(self._artwork_lbl)

        meta_col = QtWidgets.QVBoxLayout()
        self._title_lbl = QtWidgets.QLabel("—")
        self._title_lbl.setStyleSheet("font-size: 15px; font-weight: 600;")
        self._artist_lbl = QtWidgets.QLabel("—")
        self._artist_lbl.setStyleSheet("color: #64748b;")
        self._album_lbl = QtWidgets.QLabel("—")
        self._album_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self._codec_lbl = QtWidgets.QLabel("")
        self._codec_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        meta_col.addWidget(self._title_lbl)
        meta_col.addWidget(self._artist_lbl)
        meta_col.addWidget(self._album_lbl)
        meta_col.addWidget(self._codec_lbl)
        meta_col.addStretch(1)
        meta_row.addLayout(meta_col, 1)
        file_layout.addLayout(meta_row)
        cl.addWidget(file_card)

        # ── Transport controls ─────────────────────────────────────────────
        transport_card = self._make_card()
        transport_layout = QtWidgets.QVBoxLayout(transport_card)
        transport_layout.setContentsMargins(16, 16, 16, 16)
        transport_layout.setSpacing(10)

        self._seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setValue(0)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        transport_layout.addWidget(self._seek_slider)

        time_row = QtWidgets.QHBoxLayout()
        self._pos_lbl = QtWidgets.QLabel("0:00")
        self._pos_lbl.setStyleSheet("color: #64748b; font-size: 12px;")
        self._dur_lbl = QtWidgets.QLabel("0:00")
        self._dur_lbl.setStyleSheet("color: #64748b; font-size: 12px;")
        time_row.addWidget(self._pos_lbl)
        time_row.addStretch(1)
        time_row.addWidget(self._dur_lbl)
        transport_layout.addLayout(time_row)

        ctrl_row = QtWidgets.QHBoxLayout()
        self._play_btn = self._make_primary_button("▶  Play")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._on_play_pause)

        self._stop_btn = QtWidgets.QPushButton("■  Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)

        vol_lbl = QtWidgets.QLabel("Volume:")
        vol_lbl.setStyleSheet("color: #64748b; font-size: 12px;")
        self._vol_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setFixedWidth(100)
        self._vol_slider.valueChanged.connect(self._on_volume)

        ctrl_row.addWidget(self._play_btn)
        ctrl_row.addWidget(self._stop_btn)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(vol_lbl)
        ctrl_row.addWidget(self._vol_slider)
        transport_layout.addLayout(ctrl_row)

        self._vlc_status_lbl = QtWidgets.QLabel("VLC: checking…")
        self._vlc_status_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        transport_layout.addWidget(self._vlc_status_lbl)
        cl.addWidget(transport_card)

        cl.addStretch(1)

    def _init_vlc(self) -> None:
        try:
            import vlc  # type: ignore
            self._vlc_instance = vlc.Instance()
            self._media_player = self._vlc_instance.media_player_new()
            self._vlc_status_lbl.setText("VLC: ready")
        except ImportError:
            self._vlc_status_lbl.setText("VLC not installed — playback unavailable")
            self._log("libVLC not found. Install VLC for in-app playback.", "warn")

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_browse(self) -> None:
        start = self._library_path or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Audio File", start,
            "Audio (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All (*)"
        )
        if path:
            self._load_file(path)

    @Slot()
    def _on_play_pause(self) -> None:
        if not self._media_player:
            return
        if self._media_player.is_playing():
            self._media_player.pause()
            self._play_btn.setText("▶  Play")
            self._timer.stop()
        else:
            self._media_player.play()
            self._play_btn.setText("⏸  Pause")
            self._timer.start()

    @Slot()
    def _on_stop(self) -> None:
        if self._media_player:
            self._media_player.stop()
            self._play_btn.setText("▶  Play")
            self._timer.stop()
            self._seek_slider.setValue(0)
            self._pos_lbl.setText("0:00")

    @Slot(int)
    def _on_seek(self, value: int) -> None:
        if self._media_player and self._media_player.get_length() > 0:
            pos = value / 1000.0
            self._media_player.set_position(pos)

    @Slot(int)
    def _on_volume(self, value: int) -> None:
        if self._media_player:
            self._media_player.audio_set_volume(value)

    @Slot()
    def _update_position(self) -> None:
        if not self._media_player:
            return
        dur = self._media_player.get_length()
        pos = self._media_player.get_time()
        if dur > 0:
            self._seek_slider.setValue(int((pos / dur) * 1000))
            self._pos_lbl.setText(self._fmt_time(pos // 1000))
            self._dur_lbl.setText(self._fmt_time(dur // 1000))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_file(self, path: str) -> None:
        self._current_file = path
        self._file_entry.setText(path)
        if self._media_player:
            if self._vlc_instance:
                media = self._vlc_instance.media_new(path)
                self._media_player.set_media(media)
                self._media_player.audio_set_volume(self._vol_slider.value())
            self._play_btn.setEnabled(True)
            self._stop_btn.setEnabled(True)

        # Load metadata
        try:
            from mutagen import File as MutagenFile
            tags = MutagenFile(path, easy=True)
            if tags:
                self._title_lbl.setText(
                    (tags.get("title") or ["—"])[0]
                )
                self._artist_lbl.setText(
                    (tags.get("artist") or ["—"])[0]
                )
                self._album_lbl.setText(
                    (tags.get("album") or ["—"])[0]
                )
        except Exception:
            pass

        ext = Path(path).suffix.upper()
        self._codec_lbl.setText(f"{ext}  ·  {Path(path).name}")

    @staticmethod
    def _fmt_time(seconds: int) -> str:
        m, s = divmod(max(0, seconds), 60)
        return f"{m}:{s:02d}"
