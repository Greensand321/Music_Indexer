"""Persistent Now Playing bar — always visible at the bottom of the main window.

Shows the current track's art thumbnail, title/artist, a seek slider, and
transport controls so the user never needs to navigate to the Player tab
just to skip a track or adjust volume.
"""
from __future__ import annotations

from gui.compat import QtCore, QtGui, QtWidgets, Signal
from gui.themes.manager import get_manager


class NowPlayingBar(QtWidgets.QWidget):
    """Slim transport bar pinned to the bottom of the main window.

    Signals emitted (routed back to PlayerWorkspace by AlphaDEXWindow):
        play_pause_requested
        next_requested
        prev_requested
        seek_requested(int)   — target position in ms (drag release, wheel,
                                keys, groove click)
        volume_changed(int)   — 0-100, from any user interaction
        keyboard_mode_toggled(bool)  — arrow-key navigation mode

    ``update_now_playing(title, artist, album, pm)``: ``pm`` None keeps the
    current thumbnail (art still loading); a *null* QPixmap means "no art"
    (♪ placeholder); an empty title with a null pixmap means nothing is
    loaded any more — the bar resets and hides.
    """

    play_pause_requested = Signal()
    next_requested       = Signal()
    prev_requested       = Signal()
    seek_requested       = Signal(int)   # ms
    volume_changed       = Signal(int)   # 0-100
    keyboard_mode_toggled = Signal(bool)

    _ART_SIZE = 44   # px, small square thumbnail
    _KB_PULSE_MS = 120

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setObjectName("nowPlayingBar")

        self._is_playing  = False
        self._duration_ms = 0
        self._user_seeking = False

        self._build_ui()
        get_manager().theme_changed.connect(self._apply_theme)
        self._apply_theme(get_manager().current)
        self.setVisible(False)   # hidden until first track is loaded

    # ── Construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(12, 6, 12, 6)
        root.setSpacing(10)

        # Art thumbnail
        self._art_lbl = QtWidgets.QLabel()
        self._art_lbl.setObjectName("npbArt")
        self._art_lbl.setFixedSize(self._ART_SIZE, self._ART_SIZE)
        self._art_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._art_lbl.setText("♪")
        root.addWidget(self._art_lbl)

        # Title + artist column
        meta_col = QtWidgets.QVBoxLayout()
        meta_col.setSpacing(1)
        self._title_lbl = QtWidgets.QLabel("—")
        self._title_lbl.setObjectName("npbTitle")
        self._title_lbl.setMaximumWidth(260)
        self._title_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.NoTextInteraction
        )
        self._artist_lbl = QtWidgets.QLabel("")
        self._artist_lbl.setObjectName("npbMuted")
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
        self._pos_lbl.setObjectName("npbMuted")
        self._pos_lbl.setFixedWidth(34)

        self._seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setFixedWidth(180)
        self._seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self._seek_slider.sliderReleased.connect(self._on_slider_released)
        # Wheel / keys / groove clicks change the value without a press.
        self._seek_slider.valueChanged.connect(self._on_seek_value_changed)

        self._dur_lbl = QtWidgets.QLabel("0:00")
        self._dur_lbl.setObjectName("npbMuted")
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
        # Any user change (drag, wheel, keys, groove click) is emitted;
        # set_volume() blocks signals, so syncing from the page can't echo.
        self._vol_slider.valueChanged.connect(self.volume_changed)
        root.addWidget(vol_lbl)
        root.addWidget(self._vol_slider)

        root.addSpacing(8)

        # ── Keyboard mode toggle (pill button) ────────────────────────────
        self._kb_btn = QtWidgets.QPushButton("⌨")
        self._kb_btn.setCheckable(True)
        self._kb_btn.setFlat(True)
        self._kb_btn.setFixedSize(36, 28)
        self._kb_btn.setToolTip(
            "Keyboard Mode: ↓ preview next  → next  ← prev  ↑ restart"
        )
        self._kb_btn.setObjectName("kbModeBtn")
        self._kb_btn.toggled.connect(self._on_kb_toggled)
        root.addWidget(self._kb_btn)

        # Press pulse: animate an opacity effect, never the geometry of a
        # layout-managed widget (that drifted when toggled quickly). One
        # reusable, parented animation that is restarted, not stacked.
        self._kb_fx = QtWidgets.QGraphicsOpacityEffect(self._kb_btn)
        self._kb_fx.setOpacity(1.0)
        self._kb_fx.setEnabled(False)
        self._kb_btn.setGraphicsEffect(self._kb_fx)
        self._kb_anim = QtCore.QPropertyAnimation(self._kb_fx, b"opacity", self)
        self._kb_anim.setDuration(self._KB_PULSE_MS)
        self._kb_anim.setKeyValueAt(0.0, 1.0)
        self._kb_anim.setKeyValueAt(0.5, 0.55)
        self._kb_anim.setKeyValueAt(1.0, 1.0)
        self._kb_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._kb_anim.finished.connect(self._on_kb_pulse_finished)

    # ── Theme ─────────────────────────────────────────────────────────────

    def _apply_theme(self, tokens=None) -> None:
        t = tokens if tokens is not None else get_manager().current
        self.setStyleSheet(
            f"#npbArt {{ background: {t.card_bg}; border-radius: 6px; color: {t.text_muted};"
            f" font-size: 18px; }}"
            "#npbTitle { font-size: 12px; font-weight: 600; }"
            f"#npbMuted {{ color: {t.text_secondary}; font-size: 11px; }}"
            f"#kbModeBtn {{ background: transparent; color: {t.text_secondary};"
            f" border: 1px solid {t.card_border}; border-radius: 12px; font-size: 14px; }}"
            f"#kbModeBtn:hover {{ color: {t.accent}; border-color: {t.accent}; }}"
            f"#kbModeBtn:checked {{ background: {t.accent}; color: {t.text_inverse};"
            f" border: 2px solid {t.accent_hover}; font-weight: bold; }}"
            f"#kbModeBtn:checked:hover {{ background: {t.accent_hover}; }}"
        )
        self.update()

    # ── Keyboard mode ─────────────────────────────────────────────────────

    def _on_kb_toggled(self, checked: bool) -> None:
        # Styling follows ``:checked`` in the theme QSS (_apply_theme).
        self._animate_kb_press(checked)
        self.keyboard_mode_toggled.emit(checked)

    def _animate_kb_press(self, active: bool) -> None:
        """Short opacity pulse when toggling keyboard mode."""
        self._kb_anim.stop()
        self._kb_fx.setEnabled(True)
        self._kb_fx.setOpacity(1.0)
        self._kb_anim.start()

    def _on_kb_pulse_finished(self) -> None:
        self._kb_fx.setOpacity(1.0)
        self._kb_fx.setEnabled(False)

    # ── Painting ──────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            t = get_manager().current
            r = QtCore.QRectF(self.rect())
            # Background
            p.fillRect(r.toRect(), QtGui.QColor(t.sidebar_bg))
            # Top border line
            p.setPen(QtGui.QPen(QtGui.QColor(t.card_border), 1.0))
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
        """Update track info and the art thumbnail (see the class docstring)."""
        is_null_pm = isinstance(pm, QtGui.QPixmap) and pm.isNull()
        if not title and is_null_pm:
            self.clear()
            return

        self._title_lbl.setText(title or "—")
        sub = "  ·  ".join(p for p in [artist, album] if p)
        self._artist_lbl.setText(sub)

        if isinstance(pm, QtGui.QPixmap) and not pm.isNull():
            scaled = pm.scaled(
                self._ART_SIZE, self._ART_SIZE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._art_lbl.setText("")
            self._art_lbl.setPixmap(scaled)
        elif is_null_pm:
            self._reset_art()
        # else (None): art still loading — keep what is shown

        self.setVisible(True)

    @QtCore.Slot()
    def clear(self) -> None:
        """Nothing is loaded any more: reset to the initial state and hide."""
        self._title_lbl.setText("—")
        self._artist_lbl.setText("")
        self._reset_art()
        self.set_playing(False)
        self._duration_ms = 0
        self._set_seek_value(0)
        self._pos_lbl.setText("0:00")
        self._dur_lbl.setText("0:00")
        self.setVisible(False)

    def _reset_art(self) -> None:
        self._art_lbl.setPixmap(QtGui.QPixmap())
        self._art_lbl.setText("♪")

    @QtCore.Slot(bool)
    def set_playing(self, is_playing: bool) -> None:
        self._is_playing = is_playing
        self._play_btn.setText("⏸" if is_playing else "▶")

    @QtCore.Slot(int, int)
    def update_position(self, pos_ms: int, dur_ms: int) -> None:
        self._duration_ms = dur_ms
        if not self._user_seeking and not self._seek_slider.isSliderDown():
            self._set_seek_value(int((pos_ms / dur_ms) * 1000) if dur_ms > 0 else 0)
            self._pos_lbl.setText(self._fmt(pos_ms))
        self._dur_lbl.setText(self._fmt(dur_ms))

    @QtCore.Slot(int)
    def set_volume(self, value: int) -> None:
        """Keep the bar's volume knob in sync with the player (no echo)."""
        was = self._vol_slider.blockSignals(True)
        self._vol_slider.setValue(max(0, min(100, int(value))))
        self._vol_slider.blockSignals(was)

    # ── Seek interaction ──────────────────────────────────────────────────

    def _set_seek_value(self, value: int) -> None:
        was = self._seek_slider.blockSignals(True)
        self._seek_slider.setValue(value)
        self._seek_slider.blockSignals(was)

    def _on_slider_pressed(self) -> None:
        self._user_seeking = True

    def _on_slider_released(self) -> None:
        self._user_seeking = False
        self._emit_seek(self._seek_slider.value())

    def _on_seek_value_changed(self, value: int) -> None:
        if self._user_seeking or self._seek_slider.isSliderDown():
            return  # the release emits the final position
        self._emit_seek(value)

    def _emit_seek(self, value: int) -> None:
        if self._duration_ms > 0:
            self.seek_requested.emit(int((value / 1000.0) * self._duration_ms))

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _fmt(ms: int) -> str:
        ms = max(0, ms)
        m, s = divmod(ms // 1000, 60)
        return f"{m}:{s:02d}"
