"""Persistent Now Playing bar — always visible at the bottom of the main window.

Shows the current track's art thumbnail, title/artist, a seek slider, and
transport controls so the user never needs to navigate to the Player tab
just to skip a track or adjust volume.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal


class NowPlayingBar(QtWidgets.QWidget):
    """Slim transport bar pinned to the bottom of the main window.

    Signals emitted (routed back to PlayerWorkspace by AlphaDEXWindow):
        play_pause_requested
        next_requested
        prev_requested
        seek_requested(int)   — target position in ms
    """

    play_pause_requested = Signal()
    next_requested       = Signal()
    prev_requested       = Signal()
    seek_requested       = Signal(int)   # ms
    volume_changed       = Signal(int)   # 0-100

    _ART_SIZE = 44   # px, small square thumbnail

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setObjectName("nowPlayingBar")

        self._is_playing  = False
        self._duration_ms = 0
        self._user_seeking = False

        self._build_ui()
        self.setVisible(False)   # hidden until first track is loaded

    # ── Construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 6)
        root.setSpacing(10)

        # Art thumbnail
        self._art_lbl = QtWidgets.QLabel()
        self._art_lbl.setFixedSize(self._ART_SIZE, self._ART_SIZE)
        self._art_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._art_lbl.setStyleSheet(
            "background: #1e2130; border-radius: 6px; color: #475569; font-size: 18px;"
        )
        self._art_lbl.setText("♪")
        root.addWidget(self._art_lbl)

        # Title + artist column
        meta_col = QtWidgets.QVBoxLayout()
        meta_col.setSpacing(1)
        self._title_lbl = QtWidgets.QLabel("—")
        self._title_lbl.setStyleSheet("font-size: 12px; font-weight: 600;")
        self._title_lbl.setMaximumWidth(260)
        self._title_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.NoTextInteraction
        )
        self._artist_lbl = QtWidgets.QLabel("")
        self._artist_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        self._artist_lbl.setMaximumWidth(260)
        meta_col.addWidget(self._title_lbl)
        meta_col.addWidget(self._artist_lbl)
        root.addLayout(meta_col)

        root.addStretch(1)

        # Transport controls
        def _tbtn(icon: str, tip: str = "") -> QtWidgets.QPushButton:
            b = QtWidgets.QPushButton(icon)
            b.setFixedSize(30, 28)
            b.setFlat(True)
            if tip:
                b.setToolTip(tip)
            return b

        self._prev_btn = _tbtn("⏮", "Previous track")
        self._prev_btn.clicked.connect(self.prev_requested)

        self._play_btn = _tbtn("▶", "Play / Pause  [Space]")
        self._play_btn.setFixedSize(36, 30)
        self._play_btn.clicked.connect(self.play_pause_requested)

        self._next_btn = _tbtn("⏭", "Next track")
        self._next_btn.clicked.connect(self.next_requested)

        root.addWidget(self._prev_btn)
        root.addWidget(self._play_btn)
        root.addWidget(self._next_btn)
        root.addSpacing(8)

        # Seek area: time + slider + duration
        self._pos_lbl = QtWidgets.QLabel("0:00")
        self._pos_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        self._pos_lbl.setFixedWidth(34)

        self._seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setFixedWidth(180)
        self._seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self._seek_slider.sliderReleased.connect(self._on_slider_released)

        self._dur_lbl = QtWidgets.QLabel("0:00")
        self._dur_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        self._dur_lbl.setFixedWidth(34)
        self._dur_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        root.addWidget(self._pos_lbl)
        root.addWidget(self._seek_slider)
        root.addWidget(self._dur_lbl)

        root.addSpacing(8)

        # Volume
        vol_lbl = QtWidgets.QLabel("🔊")
        vol_lbl.setStyleSheet("font-size: 11px;")
        self._vol_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setFixedWidth(70)
        self._vol_slider.setToolTip("Volume")
        # Use sliderMoved so programmatic set_volume() doesn't echo back
        self._vol_slider.sliderMoved.connect(self.volume_changed)
        self._vol_slider.sliderReleased.connect(
            lambda: self.volume_changed.emit(self._vol_slider.value())
        )
        root.addWidget(vol_lbl)
        root.addWidget(self._vol_slider)

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            try:
                from gui.themes.manager import get_manager
                t = get_manager().current
                bg = t.sidebar_bg
                border = t.card_border
            except Exception:
                bg, border = "#0d1117", "#30363d"

            r = QtCore.QRectF(self.rect())
            # Background
            p.fillRect(r.toRect(), QtGui.QColor(bg))
            # Top border line
            p.setPen(QtGui.QPen(QtGui.QColor(border), 1.0))
            p.drawLine(
                QtCore.QPointF(r.left(), r.top()),
                QtCore.QPointF(r.right(), r.top()),
            )
        finally:
            p.end()

    # ── Public slots (called by AlphaDEXWindow) ───────────────────────────

    @QtCore.Slot(str, str, str, object)
    def update_now_playing(
        self, title: str, artist: str, album: str, pm: object
    ) -> None:
        """Update track info and optionally the art thumbnail."""
        self._title_lbl.setText(title or "—")
        sub = "  ·  ".join(p for p in [artist, album] if p)
        self._artist_lbl.setText(sub)

        if pm and not pm.isNull():
            scaled = pm.scaled(
                self._ART_SIZE, self._ART_SIZE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._art_lbl.setText("")
            self._art_lbl.setPixmap(scaled)
        # else: keep previous art until new art is ready

        self.setVisible(True)

    @QtCore.Slot(bool)
    def set_playing(self, is_playing: bool) -> None:
        self._is_playing = is_playing
        self._play_btn.setText("⏸" if is_playing else "▶")

    @QtCore.Slot(int, int)
    def update_position(self, pos_ms: int, dur_ms: int) -> None:
        self._duration_ms = dur_ms
        if not self._user_seeking:
            if dur_ms > 0:
                self._seek_slider.setValue(int((pos_ms / dur_ms) * 1000))
            self._pos_lbl.setText(self._fmt(pos_ms))
        self._dur_lbl.setText(self._fmt(dur_ms))

    @QtCore.Slot(int)
    def set_volume(self, value: int) -> None:
        """Keep the bar's volume knob in sync with the player workspace slider."""
        self._vol_slider.setValue(max(0, min(100, value)))

    # ── Seek interaction ──────────────────────────────────────────────────

    def _on_slider_pressed(self) -> None:
        self._user_seeking = True

    def _on_slider_released(self) -> None:
        self._user_seeking = False
        if self._duration_ms > 0:
            target_ms = int((self._seek_slider.value() / 1000.0) * self._duration_ms)
            self.seek_requested.emit(target_ms)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(ms: int) -> str:
        ms = max(0, ms)
        m, s = divmod(ms // 1000, 60)
        return f"{m}:{s:02d}"
