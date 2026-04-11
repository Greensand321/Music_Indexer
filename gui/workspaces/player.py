"""Player workspace — full-featured in-app audio player using libVLC."""
from __future__ import annotations

import os
import random
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.widgets.gradient_bg import GradientWidget
from gui.workspaces.base import WorkspaceBase

_AUDIO_EXTS = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
_ART_SIZE = 200

# Repeat modes
_REPEAT_OFF   = 0
_REPEAT_TRACK = 1
_REPEAT_ALL   = 2

# 30-second highlight: seek to this position, auto-advance after duration
_HIGHLIGHT_START_MS    = 30_000
_HIGHLIGHT_DURATION_MS = 15_000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_ms(ms: int) -> str:
    """Format milliseconds as M:SS."""
    ms = max(0, ms)
    m, s = divmod(ms // 1000, 60)
    return f"{m}:{s:02d}"


def _load_row(path: str) -> dict:
    """Read tags for one audio file and return a row dict.  Never raises.

    Opens the file once with Mutagen, passes the pre-loaded object to
    read_metadata_from_mutagen for tags, and reads .info.length from the
    same object for duration — avoiding the previous double-open pattern.
    """
    title = artist = album = dur_str = track_num = ""
    try:
        from mutagen import File as _MF  # type: ignore
        audio = _MF(path)
    except Exception:
        audio = None

    try:
        from utils.audio_metadata_reader import read_metadata_from_mutagen
        tags, _, _, _ = read_metadata_from_mutagen(audio, path, include_cover=False)
        title  = str(tags.get("title")  or "")
        artist = str(tags.get("artist") or "")
        album  = str(tags.get("album")  or "")
        tn = tags.get("track") or tags.get("tracknumber")
        if tn is not None:
            try:
                track_num = str(int(str(tn).split("/")[0]))
            except Exception:
                pass
    except Exception:
        pass

    if not title:
        title = Path(path).stem

    if audio and getattr(getattr(audio, "info", None), "length", None):
        m, s = divmod(int(audio.info.length), 60)
        dur_str = f"{m}:{s:02d}"

    return {
        "track_num": track_num,
        "title":     title,
        "artist":    artist,
        "album":     album,
        "dur":       dur_str,
        "path":      path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────────────────────

class _LibraryScanner(QtCore.QThread):
    """Walk a library folder and emit batches of metadata rows with auto-throttling."""

    tracks_ready  = Signal(list)   # list[dict]
    scan_complete = Signal(int)    # total count

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self._path = library_path
        self._ui_latency_ms = 0
        self._current_batch_size = 50
        self._sleep_ms = 0

    def report_ui_latency(self, elapsed_ms: int) -> None:
        self._ui_latency_ms = elapsed_ms

    def run(self) -> None:
        batch: list[dict] = []
        total = 0
        try:
            for root, _dirs, files in os.walk(self._path):
                if self.isInterruptionRequested():
                    break
                for fname in sorted(files):
                    if self.isInterruptionRequested():
                        break
                    if os.path.splitext(fname)[1].lower() not in _AUDIO_EXTS:
                        continue
                    batch.append(_load_row(os.path.join(root, fname)))
                    total += 1
                    
                    if len(batch) >= self._current_batch_size:
                        self.tracks_ready.emit(list(batch))
                        batch.clear()
                        
                        latency = self._ui_latency_ms
                        if latency > 15:
                            self._current_batch_size = max(10, self._current_batch_size - 10)
                            self._sleep_ms = min(50, self._sleep_ms + 5)
                        elif latency < 5:
                            self._current_batch_size = min(500, self._current_batch_size + 20)
                            self._sleep_ms = max(0, self._sleep_ms - 2)
                            
                        if self._sleep_ms > 0:
                            self.msleep(self._sleep_ms)
        finally:
            if batch:
                self.tracks_ready.emit(batch)
            self.scan_complete.emit(total)


class _ArtLoader(QtCore.QThread):
    """Load and scale album art for one track off the GUI thread."""

    art_ready = Signal(object, str)   # QPixmap | None,  path

    def __init__(self, path: str, size: int = _ART_SIZE) -> None:
        super().__init__()
        self._path = path
        self._size = size

    def run(self) -> None:
        try:
            from utils.audio_metadata_reader import read_metadata
            _, covers, _, _ = read_metadata(self._path, include_cover=True)
            if covers:
                img = QtGui.QImage()
                if img.loadFromData(covers[0]):
                    s = self._size
                    img = img.scaled(
                        s, s,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                    if img.width() > s:
                        img = img.copy((img.width() - s) // 2, 0, s, s)
                    elif img.height() > s:
                        img = img.copy(0, (img.height() - s) // 2, s, s)
                    pm = QtGui.QPixmap.fromImage(img)
                    if not pm.isNull():
                        self.art_ready.emit(pm, self._path)
                        return
        except Exception:
            pass
        self.art_ready.emit(None, self._path)


# ─────────────────────────────────────────────────────────────────────────────
# _BgArtWidget — blurred album-art background behind the library table
# ─────────────────────────────────────────────────────────────────────────────

class _BgArtWidget(QtWidgets.QWidget):
    """Wraps the library table with a slowly cross-fading blurred album-art background.

    The table is laid out as a transparent child so the blurred art bleeds
    through each row at low opacity, giving a subtle ambient-art effect without
    hurting readability.
    """

    _BLUR_DIV   = 10    # scale-down factor for cheap Gaussian approximation
    _ART_OPACITY = 0.30  # max opacity of the art layer
    _FADE_MS    = 3500   # cross-fade duration in ms

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._pm_cur: QtGui.QPixmap | None = None
        self._pm_nxt: QtGui.QPixmap | None = None
        self._fade_val: float = 1.0

        self._anim = QtCore.QPropertyAnimation(self, b"crossfade")
        self._anim.setDuration(self._FADE_MS)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutSine)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    # Qt property for the animation target
    def _get_cf(self) -> float:
        return self._fade_val

    def _set_cf(self, v: float) -> None:
        self._fade_val = v
        self.update()
        if v >= 1.0 and self._pm_nxt is not None:
            self._pm_cur = self._pm_nxt
            self._pm_nxt = None

    crossfade = QtCore.Property(float, _get_cf, _set_cf)

    def set_art(self, pm: QtGui.QPixmap) -> None:
        if pm is None or pm.isNull():
            return
        blurred = self._blur(pm)
        if self._pm_cur is None:
            self._pm_cur = blurred
            self.update()
            return
        self._pm_nxt = blurred
        self._anim.stop()
        self._fade_val = 0.0
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _blur(self, pm: QtGui.QPixmap) -> QtGui.QPixmap:
        img = pm.toImage()
        d   = self._BLUR_DIV
        small = img.scaled(
            max(1, img.width() // d), max(1, img.height() // d),
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        big = small.scaled(
            img.width(), img.height(),
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        return QtGui.QPixmap.fromImage(big)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        r = self.rect()

        p.fillRect(r, QtGui.QColor(13, 17, 23))

        def _blit(pm: QtGui.QPixmap, op: float) -> None:
            if pm is None or pm.isNull():
                return
            s = pm.scaled(
                r.width(), r.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.FastTransformation,
            )
            p.setOpacity(op)
            p.drawPixmap((r.width() - s.width()) // 2,
                         (r.height() - s.height()) // 2, s)

        op = self._ART_OPACITY
        if self._pm_nxt is not None:
            _blit(self._pm_cur, op * (1.0 - self._fade_val))
            _blit(self._pm_nxt, op * self._fade_val)
        else:
            _blit(self._pm_cur, op)

        # Dark vignette so table text stays readable
        p.setOpacity(0.70)
        p.fillRect(r, QtGui.QColor(10, 13, 20))
        p.setOpacity(1.0)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# CoverFlowWidget — 3-D Cover Flow display
# ─────────────────────────────────────────────────────────────────────────────

class CoverFlowWidget(QtWidgets.QGraphicsView):
    """3-D Cover Flow panel (Optimized 2.5D).

    * Front cover (currently playing): centered, full-size, facing the viewer.
    * Back cover (single-clicked track): slides in from one side.
    """

    _FRONT      = 160     # front cover square size (px)
    _BACK_W     = 152     # back cover source width
    _BACK_H     = 145     # back cover source height
    _DRAW_RATIO = 0.44    # fraction of _BACK_W visible after perspective
    _V_INNER    = 0.455   # inner-edge half-height fraction of _BACK_H
    _V_OUTER    = 0.370   # outer-edge half-height fraction (perspective taper)
    _GAP        = 10      # px gap between front and back inner edges
    _SLIDE_MS   = 430

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHints(
            QtGui.QPainter.RenderHint.Antialiasing |
            QtGui.QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setStyleSheet("background: transparent;")
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        # Create Items
        self._back_item = QtWidgets.QGraphicsPixmapItem()
        self._front_item = QtWidgets.QGraphicsPixmapItem()
        
        # Add to scene
        self._scene.addItem(self._back_item)
        self._scene.addItem(self._front_item)

        self._front_pm: QtGui.QPixmap | None = None
        self._back_pm:  QtGui.QPixmap | None = None
        self._back_right: bool = True
        self._slide_val: float = 0.0

        self._anim = QtCore.QPropertyAnimation(self, b"slide_progress")
        self._anim.setDuration(self._SLIDE_MS)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        side_vis = int(self._BACK_W * self._DRAW_RATIO)
        self.setMinimumSize(self._FRONT + 2 * side_vis + 24, self._FRONT + 20)
        
        self._bake_front()
        self._back_item.hide()

    def sizeHint(self) -> QtCore.QSize:
        side_vis = int(self._BACK_W * self._DRAW_RATIO)
        return QtCore.QSize(self._FRONT + 2 * side_vis + 24, self._FRONT + 20)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._scene.setSceneRect(0, 0, self.width(), self.height())
        self._update_layout()

    def _get_sp(self) -> float:
        return self._slide_val

    def _set_sp(self, v: float) -> None:
        self._slide_val = v
        self._update_layout()

    slide_progress = QtCore.Property(float, _get_sp, _set_sp)

    def set_front(self, pm: QtGui.QPixmap | None) -> None:
        self._front_pm = pm
        self._bake_front()
        self._update_layout()

    def set_back(self, pm: QtGui.QPixmap | None, right_side: bool = True) -> None:
        side_changed = (right_side != self._back_right)
        self._back_pm    = pm
        self._back_right = right_side
        if pm:
            self._bake_back()
            start = 0.0 if (side_changed or self._slide_val < 0.05) else self._slide_val
            self._anim.stop()
            self._slide_val = start
            self._anim.setStartValue(start)
            self._anim.setEndValue(1.0)
            self._anim.start()
        else:
            self._anim.stop()
            self._slide_val = 0.0
            self._back_item.hide()
        self._update_layout()

    def clear_back(self) -> None:
        self._anim.stop()
        self._back_pm   = None
        self._slide_val = 0.0
        self._back_item.hide()
        self._update_layout()

    def _bake_front(self) -> None:
        sz = self._FRONT
        pm = QtGui.QPixmap(sz + 20, sz + 20)
        pm.fill(QtCore.Qt.GlobalColor.transparent)
        
        p = QtGui.QPainter(pm)
        p.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        
        # Drop shadow
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 95))
        p.drawRoundedRect(15, 17, sz, sz, 10, 10)

        if self._front_pm:
            scaled = self._front_pm.scaled(
                sz, sz,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            clip = QtGui.QPainterPath()
            clip.addRoundedRect(QtCore.QRectF(10, 10, scaled.width(), scaled.height()), 8, 8)
            p.setClipPath(clip)
            p.drawPixmap(10, 10, scaled)
        else:
            p.setBrush(QtGui.QColor(30, 33, 48))
            p.drawRoundedRect(10, 10, sz, sz, 8, 8)
            p.setPen(QtGui.QColor(71, 85, 105))
            fnt = self.font()
            fnt.setPointSize(30)
            p.setFont(fnt)
            p.drawText(QtCore.QRect(10, 10, sz, sz), QtCore.Qt.AlignmentFlag.AlignCenter, "♪")
        
        p.end()
        self._front_item.setPixmap(pm)

    def _bake_back(self) -> None:
        # Pre-compute the exact 3D distorted shape and shadow ONCE
        bw, bh  = self._BACK_W, self._BACK_H
        drawn_w = int(bw * self._DRAW_RATIO)
        inner_hh = int(bh * self._V_INNER)
        outer_hh = int(bh * self._V_OUTER)
        
        pm = self._back_pm.scaled(
            bw, bh,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        
        out_pm = QtGui.QPixmap(drawn_w, inner_hh * 2)
        out_pm.fill(QtCore.Qt.GlobalColor.transparent)
        
        p = QtGui.QPainter(out_pm)
        p.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        
        cy_local = inner_hh
        
        src = QtGui.QPolygonF([
            QtCore.QPointF(0,  0),
            QtCore.QPointF(bw, 0),
            QtCore.QPointF(bw, bh),
            QtCore.QPointF(0,  bh),
        ])
        
        if self._back_right:
            inner_x, outer_x = 0, drawn_w
            dst = QtGui.QPolygonF([
                QtCore.QPointF(inner_x, cy_local - inner_hh),
                QtCore.QPointF(outer_x, cy_local - outer_hh),
                QtCore.QPointF(outer_x, cy_local + outer_hh),
                QtCore.QPointF(inner_x, cy_local + inner_hh),
            ])
        else:
            inner_x, outer_x = drawn_w, 0
            dst = QtGui.QPolygonF([
                QtCore.QPointF(outer_x, cy_local - outer_hh),
                QtCore.QPointF(inner_x, cy_local - inner_hh),
                QtCore.QPointF(inner_x, cy_local + inner_hh),
                QtCore.QPointF(outer_x, cy_local + outer_hh),
            ])
            
        t = QtGui.QTransform()
        ok = False
        try:
            result = QtGui.QTransform.quadToQuad(src, dst)
            if isinstance(result, tuple) and len(result) == 2:
                ok, t = result
            elif isinstance(result, bool):
                ok = result
            elif result is not None:
                ok = True
                t = result
        except Exception:
            try:
                result = QtGui.QTransform.quadToQuad(src, dst, t)
                ok = result if isinstance(result, bool) else (result[0] if result else False)
                if isinstance(result, tuple) and len(result) == 2:
                    ok, t = result
            except Exception:
                pass
                
        if not ok:
            t = QtGui.QTransform()
            if self._back_right:
                t.translate(0, float(cy_local - bh // 2))
            else:
                t.translate(0, float(cy_local - bh // 2))
            t.scale(self._DRAW_RATIO, 1.0)
            
        p.setTransform(t)
        p.drawPixmap(0, 0, pm)
        p.resetTransform()
        
        # Apply shadow
        clip_path = QtGui.QPainterPath()
        clip_path.addPolygon(dst)
        clip_path.closeSubpath()
        p.setClipPath(clip_path)
        
        grad = QtGui.QLinearGradient(QtCore.QPointF(inner_x, cy_local), QtCore.QPointF(outer_x, cy_local))
        grad.setColorAt(0.0, QtGui.QColor(0, 0, 0, 0))
        grad.setColorAt(0.55, QtGui.QColor(0, 0, 0, 70))
        grad.setColorAt(1.0, QtGui.QColor(0, 0, 0, 190))
        
        p.fillRect(QtCore.QRectF(0, 0, float(drawn_w), float(inner_hh * 2)), grad)
        p.end()
        
        self._back_item.setPixmap(out_pm)
        self._back_item.show()

    def _update_layout(self) -> None:
        cx = self.width() // 2
        cy = self.height() // 2
        
        self._front_item.setPos(cx - self._FRONT // 2 - 10, cy - self._FRONT // 2 - 10)
        self._front_item.setZValue(10)
        
        if self._back_pm and self._slide_val > 0.001:
            a = self._slide_val
            drawn_w = int(self._BACK_W * self._DRAW_RATIO)
            slide_extra = int((1.0 - a) * (drawn_w + 28))
            inner_hh = int(self._BACK_H * self._V_INNER)
            
            if self._back_right:
                x_pos = cx + self._FRONT // 2 + self._GAP + slide_extra
            else:
                x_pos = cx - self._FRONT // 2 - self._GAP - drawn_w - slide_extra
                
            self._back_item.setPos(x_pos, cy - inner_hh)
            self._back_item.setOpacity(0.88 * a)
            self._back_item.setZValue(5)
            self._back_item.show()
        else:
            self._back_item.hide()


# ─────────────────────────────────────────────────────────────────────────────
# PlayerWorkspace
# ─────────────────────────────────────────────────────────────────────────────

class PlayerWorkspace(WorkspaceBase):
    """Full-featured audio player: library browser, queue, playlist management."""

    # ── Signals for NowPlayingBar ──────────────────────────────────────────
    now_playing_changed    = Signal(str, str, str, object)  # title, artist, album, QPixmap|None
    playback_state_changed = Signal(bool)                   # is_playing
    position_changed       = Signal(int, int)               # pos_ms, dur_ms

    # ── Setup ──────────────────────────────────────────────────────────────

    def _setup_scroll(self) -> None:
        """Override: no scroll area — player fills the viewport with a fixed layout."""
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._inner = GradientWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._inner)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        outer.addWidget(self._inner)

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)

        # VLC handles
        self._vlc_instance  = None
        self._media_player  = None
        self._vlc_state_ended = None   # cached vlc.State.Ended after init

        # Queue / playback state
        self._queue:        list[dict] = []
        self._queue_index:  int        = -1
        self._shuffle:      bool       = False
        self._repeat:       int        = _REPEAT_OFF
        self._recent_plays: list[dict] = []   # most-recent first, capped at 30

        # Library cache
        self._all_rows: list[dict] = []      # full scan results

        # Progressive render state
        self._scan_finished: bool       = False

        # Workers
        self._lib_scanner:    _LibraryScanner | None = None
        self._art_loader:     _ArtLoader | None      = None
        self._sel_art_loader: _ArtLoader | None      = None  # art for single-clicked row

        # Position / highlight timers
        self._pos_timer = QtCore.QTimer()
        self._pos_timer.setInterval(400)
        self._pos_timer.timeout.connect(self._tick_position)

        self._highlight_timer = QtCore.QTimer()
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self._highlight_advance)

        self._build_ui()
        self._init_vlc()
        self._install_shortcuts()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self._content_layout

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QtWidgets.QWidget()
        toolbar.setFixedHeight(50)
        tb = QtWidgets.QHBoxLayout(toolbar)
        tb.setContentsMargins(16, 6, 16, 6)
        tb.setSpacing(8)

        title_lbl = self._make_section_title("Player")
        title_lbl.setFixedWidth(68)
        tb.addWidget(title_lbl)

        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText("Search tracks…")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filter)
        tb.addWidget(self._search, 2)

        self._scan_progress = QtWidgets.QProgressBar()
        self._scan_progress.setRange(0, 0)
        self._scan_progress.setFixedHeight(6)
        self._scan_progress.setTextVisible(False)
        self._scan_progress.setVisible(False)
        self._scan_progress.setStyleSheet(
            "QProgressBar { background: transparent; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: #6366f1; border-radius: 3px; }"
        )
        tb.addWidget(self._scan_progress, 1)

        self._reload_btn = QtWidgets.QPushButton("⟳  Reload")
        self._reload_btn.setToolTip("Re-scan the library folder")
        self._reload_btn.clicked.connect(self._reload_library)
        tb.addWidget(self._reload_btn)

        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.setToolTip("Open and play a single audio file")
        browse_btn.clicked.connect(self._on_browse_file)
        tb.addWidget(browse_btn)

        load_pl_btn = QtWidgets.QPushButton("Load M3U")
        load_pl_btn.setToolTip("Load an M3U playlist into the queue")
        load_pl_btn.clicked.connect(self._load_m3u)
        tb.addWidget(load_pl_btn)

        self._highlight_chk = QtWidgets.QCheckBox("30s Preview")
        self._highlight_chk.setToolTip(
            "Skip to the 30s mark and auto-advance after 15s — good for quick audits"
        )
        self._highlight_chk.toggled.connect(self._on_highlight_toggled)
        tb.addWidget(self._highlight_chk)

        cl.addWidget(toolbar)

        # Thin separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("background: #30363d; border: none; max-height: 1px;")
        cl.addWidget(sep)

        # ── Main splitter: library table (left) + info/queue (right) ──────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── Library table (inside a background-art container) ────────────
        self._lib_table = QtWidgets.QTableWidget(0, 5)
        self._lib_table.setHorizontalHeaderLabels(["#", "Title", "Artist", "Album", "Time"])
        self._lib_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._lib_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._lib_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._lib_table.setAlternatingRowColors(True)
        self._lib_table.verticalHeader().setVisible(False)
        self._lib_table.verticalHeader().setDefaultSectionSize(24)
        self._lib_table.setShowGrid(False)

        hh = self._lib_table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self._lib_table.setColumnWidth(0, 36)
        self._lib_table.setColumnWidth(4, 52)
        self._lib_table.setSortingEnabled(True)

        # Semi-transparent items so the background art bleeds through subtly
        self._lib_table.setAutoFillBackground(False)
        self._lib_table.viewport().setAutoFillBackground(False)
        self._lib_table.viewport().setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True
        )
        self._lib_table.setStyleSheet("""
            QTableWidget            { background: transparent; border: none; }
            QTableWidget::item      { background: rgba(13,17,23, 210); }
            QTableWidget::item:alternate { background: rgba(18,22,32, 210); }
            QTableWidget::item:selected  { background: rgba(99,102,241, 100); }
            QHeaderView::section    { background: rgba(20,24,35, 240);
                                      color: #94a3b8; border: none;
                                      padding: 2px 4px; }
        """)

        self._lib_table.doubleClicked.connect(self._on_table_double_click)
        self._lib_table.clicked.connect(self._on_table_single_click)
        self._lib_table.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._lib_table.customContextMenuRequested.connect(self._on_table_context_menu)

        # Wrap the table in the background-art container
        self._bg_art = _BgArtWidget()
        self._bg_art.layout().addWidget(self._lib_table)
        splitter.addWidget(self._bg_art)

        # ── Right panel ───────────────────────────────────────────────────
        right = QtWidgets.QWidget()
        right.setMinimumWidth(240)
        right.setMaximumWidth(340)
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(12, 10, 12, 10)
        rl.setSpacing(8)

        # Cover Flow widget (replaces static art label)
        self._cflow = CoverFlowWidget()
        rl.addWidget(self._cflow)

        # Now-playing metadata
        self._np_title = QtWidgets.QLabel("—")
        self._np_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._np_title.setWordWrap(True)
        self._np_title.setStyleSheet("font-size: 13px; font-weight: 600; padding: 0 4px;")

        self._np_artist = QtWidgets.QLabel("")
        self._np_artist.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._np_artist.setStyleSheet("color: #64748b; font-size: 11px;")

        rl.addWidget(self._np_title)
        rl.addWidget(self._np_artist)

        # Queue header + list
        q_hdr_row = QtWidgets.QHBoxLayout()
        q_hdr = QtWidgets.QLabel("Queue")
        q_hdr.setObjectName("cardTitle")
        q_count_lbl = QtWidgets.QLabel("")
        q_count_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        self._q_count_lbl = q_count_lbl
        q_hdr_row.addWidget(q_hdr)
        q_hdr_row.addStretch()
        q_hdr_row.addWidget(q_count_lbl)
        rl.addLayout(q_hdr_row)

        self._queue_list = QtWidgets.QListWidget()
        self._queue_list.setAlternatingRowColors(True)
        self._queue_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._queue_list.doubleClicked.connect(self._on_queue_double_click)
        self._queue_list.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self._queue_list.customContextMenuRequested.connect(self._on_queue_context_menu)
        rl.addWidget(self._queue_list, 1)

        # Queue action buttons
        q_btns1 = QtWidgets.QHBoxLayout()
        self._add_queue_btn = QtWidgets.QPushButton("+ Add Selected")
        self._add_queue_btn.setToolTip("Add selected library tracks to the queue")
        self._add_queue_btn.clicked.connect(self._add_selected_to_queue)
        self._clear_queue_btn = QtWidgets.QPushButton("Clear")
        self._clear_queue_btn.clicked.connect(self._clear_queue)
        q_btns1.addWidget(self._add_queue_btn, 1)
        q_btns1.addWidget(self._clear_queue_btn)
        rl.addLayout(q_btns1)

        save_m3u_btn = QtWidgets.QPushButton("💾  Save Queue as M3U…")
        save_m3u_btn.clicked.connect(self._save_m3u)
        rl.addWidget(save_m3u_btn)

        # Recently played (collapsed by default — expands in-place)
        rp_hdr_row = QtWidgets.QHBoxLayout()
        rp_hdr = QtWidgets.QLabel("Recently Played")
        rp_hdr.setObjectName("cardTitle")
        self._rp_toggle_btn = QtWidgets.QPushButton("▶")
        self._rp_toggle_btn.setFlat(True)
        self._rp_toggle_btn.setFixedSize(22, 18)
        self._rp_toggle_btn.setToolTip("Show / hide recently played")
        self._rp_toggle_btn.clicked.connect(self._toggle_recent)
        rp_hdr_row.addWidget(rp_hdr)
        rp_hdr_row.addStretch()
        rp_hdr_row.addWidget(self._rp_toggle_btn)
        rl.addLayout(rp_hdr_row)

        self._recent_list = QtWidgets.QListWidget()
        self._recent_list.setFixedHeight(0)   # collapsed initially
        self._recent_list.setAlternatingRowColors(True)
        self._recent_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._recent_list.doubleClicked.connect(self._on_recent_double_click)
        rl.addWidget(self._recent_list)

        splitter.addWidget(right)
        splitter.setSizes([720, 280])
        cl.addWidget(splitter, 1)

        # ── Transport bar ─────────────────────────────────────────────────
        transport = QtWidgets.QWidget()
        transport.setFixedHeight(84)
        transport.setObjectName("transportBar")
        tl = QtWidgets.QVBoxLayout(transport)
        tl.setContentsMargins(16, 6, 16, 4)
        tl.setSpacing(4)

        # Seek row
        seek_row = QtWidgets.QHBoxLayout()
        self._pos_lbl = QtWidgets.QLabel("0:00")
        self._pos_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        self._pos_lbl.setFixedWidth(36)
        self._seek_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        self._dur_lbl = QtWidgets.QLabel("0:00")
        self._dur_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        self._dur_lbl.setFixedWidth(36)
        self._dur_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        seek_row.addWidget(self._pos_lbl)
        seek_row.addWidget(self._seek_slider, 1)
        seek_row.addWidget(self._dur_lbl)
        tl.addLayout(seek_row)

        # Buttons row
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(4)

        def _tbtn(icon: str, size: tuple = (34, 30)) -> QtWidgets.QPushButton:
            b = QtWidgets.QPushButton(icon)
            b.setFixedSize(*size)
            return b

        self._prev_btn = _tbtn("⏮")
        self._prev_btn.setToolTip("Previous track  [P]")
        self._prev_btn.clicked.connect(self.play_prev)

        self._play_btn = self._make_primary_button("▶")
        self._play_btn.setFixedSize(46, 34)
        self._play_btn.setEnabled(False)
        self._play_btn.setToolTip("Play / Pause  [Space]")
        self._play_btn.clicked.connect(self._on_play_pause)

        self._next_btn = _tbtn("⏭")
        self._next_btn.setToolTip("Next track  [N]")
        self._next_btn.clicked.connect(self.play_next)

        self._stop_btn = _tbtn("■")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip("Stop")
        self._stop_btn.clicked.connect(self._on_stop)

        self._shuffle_btn = _tbtn("🔀")
        self._shuffle_btn.setCheckable(True)
        self._shuffle_btn.setToolTip("Shuffle")
        self._shuffle_btn.toggled.connect(self._on_shuffle_toggled)

        self._repeat_btn = _tbtn("🔁")
        self._repeat_btn.setToolTip("Repeat: Off")
        self._repeat_btn.clicked.connect(self._cycle_repeat)

        ctrl_row.addWidget(self._prev_btn)
        ctrl_row.addWidget(self._play_btn)
        ctrl_row.addWidget(self._next_btn)
        ctrl_row.addWidget(self._stop_btn)
        ctrl_row.addSpacing(8)
        ctrl_row.addWidget(self._shuffle_btn)
        ctrl_row.addWidget(self._repeat_btn)
        ctrl_row.addStretch(1)

        vol_lbl = QtWidgets.QLabel("🔊")
        vol_lbl.setStyleSheet("font-size: 12px;")
        self._vol_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setFixedWidth(90)
        self._vol_slider.setToolTip("Volume")
        # sliderMoved only fires on user interaction, preventing feedback loops
        # when set_volume() programmatically repositions the slider.
        self._vol_slider.sliderMoved.connect(self._on_volume_drag)
        self._vol_slider.sliderReleased.connect(self._on_volume_released)

        self._vlc_status_lbl = QtWidgets.QLabel("VLC: checking…")
        self._vlc_status_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        ctrl_row.addWidget(vol_lbl)
        ctrl_row.addWidget(self._vol_slider)
        ctrl_row.addSpacing(10)
        ctrl_row.addWidget(self._vlc_status_lbl)
        tl.addLayout(ctrl_row)

        cl.addWidget(transport)

        # Status strip
        self._status_lbl = QtWidgets.QLabel("Load a library to begin.")
        self._status_lbl.setStyleSheet(
            "color: #64748b; font-size: 11px; padding: 2px 18px 4px;"
        )
        cl.addWidget(self._status_lbl)

    def _init_vlc(self) -> None:
        try:
            import vlc  # type: ignore
            self._vlc_instance  = vlc.Instance()
            self._media_player  = self._vlc_instance.media_player_new()
            self._vlc_state_ended = vlc.State.Ended
            self._vlc_status_lbl.setText("VLC: ready")
        except ImportError:
            self._vlc_status_lbl.setText("VLC not installed — playback unavailable")
            self._log("libVLC not found. Install VLC for in-app playback.", "warn")

    def _install_shortcuts(self) -> None:
        def _sc(key: str, fn) -> None:
            QtGui.QShortcut(QtGui.QKeySequence(key), self).activated.connect(fn)

        _sc("Space",       self._on_play_pause)
        _sc("N",           self.play_next)
        _sc("P",           self.play_prev)
        _sc("Right",       lambda: self._seek_relative(+5000))
        _sc("Left",        lambda: self._seek_relative(-5000))
        _sc("Shift+Right", lambda: self._seek_relative(+30000))
        _sc("Shift+Left",  lambda: self._seek_relative(-30000))

    # ── Library scanning ───────────────────────────────────────────────────

    def _on_library_changed(self, path: str) -> None:
        """Auto-reload when the library changes."""
        self._all_rows.clear()
        self._lib_table.setRowCount(0)
        self._status_lbl.setText("Library changed — click Reload to scan.")
        if path:
            self._reload_library()

    def _reload_library(self) -> None:
        if not self._library_path:
            self._status_lbl.setText("No library selected.")
            return
        # Stop any running scanner
        if self._lib_scanner and self._lib_scanner.isRunning():
            self._lib_scanner.requestInterruption()
            self._lib_scanner.wait(1000)

        self._all_rows.clear()
        self._scan_finished = False

        self._lib_table.setSortingEnabled(False)
        self._lib_table.setRowCount(0)
        self._reload_btn.setEnabled(False)
        self._status_lbl.setText("Scanning library…")

        # Indeterminate pulsing progress bar during the directory walk
        self._scan_progress.setRange(0, 0)
        self._scan_progress.setValue(0)
        self._scan_progress.setVisible(True)

        scanner = _LibraryScanner(self._library_path)
        scanner.tracks_ready.connect(self._on_tracks_ready)
        scanner.scan_complete.connect(self._on_scan_complete)
        self._lib_scanner = scanner
        scanner.start()
        scanner.setPriority(QtCore.QThread.Priority.LowPriority)

    @Slot(list)
    def _on_tracks_ready(self, batch: list) -> None:
        """Add incoming rows to the table immediately and report latency for auto-throttling."""
        t0 = QtCore.QElapsedTimer()
        t0.start()
        
        self._all_rows.extend(batch)
        
        # Turn off sorting during bulk insert to prevent lag
        self._lib_table.setSortingEnabled(False)
        
        start_row = self._lib_table.rowCount()
        self._lib_table.setRowCount(start_row + len(batch))
        
        for i, row in enumerate(batch):
            self._set_table_row(start_row + i, row)
            
        # Update progress labels
        loaded = self._lib_table.rowCount()
        self._status_lbl.setText(f"Loading… {loaded} tracks found")
        
        if self._scan_progress.maximum() > 0:
            self._scan_progress.setValue(loaded)
            
        # Report latency back to scanner for dynamic backpressure
        if self._lib_scanner:
            self._lib_scanner.report_ui_latency(t0.elapsed())

    def _set_table_row(self, r: int, row: dict) -> None:
        def _item(text: str) -> QtWidgets.QTableWidgetItem:
            item = QtWidgets.QTableWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row["path"])
            return item

        self._lib_table.setItem(r, 0, _item(row["track_num"]))
        self._lib_table.setItem(r, 1, _item(row["title"]))
        self._lib_table.setItem(r, 2, _item(row["artist"]))
        self._lib_table.setItem(r, 3, _item(row["album"]))
        self._lib_table.setItem(r, 4, _item(row["dur"]))

    @Slot(int)
    def _on_scan_complete(self, total: int) -> None:
        """Scanner has finished walking the directory."""
        self._scan_finished = True
        self._reload_btn.setEnabled(True)
        
        # Final cleanup and sort
        self._lib_table.setSortingEnabled(True)
        self._apply_filter(self._search.text())
        
        n = len(self._all_rows)
        suffix = "track" if n == 1 else "tracks"
        self._status_lbl.setText(f"Loaded {n} {suffix}.")
        
        self._scan_progress.setVisible(False)
        self._scan_progress.setRange(0, 0)

    def _apply_filter(self, text: str) -> None:
        query = text.strip().lower()
        for r in range(self._lib_table.rowCount()):
            if not query:
                self._lib_table.setRowHidden(r, False)
                continue
            row_text = " ".join(
                (self._lib_table.item(r, c) or QtWidgets.QTableWidgetItem()).text().lower()
                for c in range(1, 4)  # Title, Artist, Album
            )
            self._lib_table.setRowHidden(r, query not in row_text)

    # ── Table interaction ──────────────────────────────────────────────────

    @Slot(QtCore.QModelIndex)
    def _on_table_single_click(self, index: QtCore.QModelIndex) -> None:
        """Single-click: load art for the selected track and show it as the
        Cover Flow back cover (without starting playback)."""
        item = self._lib_table.item(index.row(), 1)
        if item is None:
            return
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not path:
            return
        # Don't show a back cover when the user clicks the currently playing track
        if (self._queue_index >= 0 and self._queue_index < len(self._queue)
                and self._queue[self._queue_index]["path"] == path):
            return
        # Cancel any previous selection loader
        if self._sel_art_loader and self._sel_art_loader.isRunning():
            self._sel_art_loader.requestInterruption()
        loader = _ArtLoader(path, _ART_SIZE)
        loader.art_ready.connect(self._on_sel_art_ready)
        self._sel_art_loader = loader
        loader.start()

    @Slot(QtCore.QModelIndex)
    def _on_table_double_click(self, index: QtCore.QModelIndex) -> None:
        row = self._lib_table.item(index.row(), 0)
        if row is None:
            return
        path = row.data(QtCore.Qt.ItemDataRole.UserRole)
        if not path:
            path = (self._lib_table.item(index.row(), 1) or row).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
        self._play_path_now(path)

    def _on_table_context_menu(self, pos: QtCore.QPoint) -> None:
        rows = self._get_selected_paths()
        if not rows:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("▶  Play now",         lambda: self._play_path_now(rows[0]))
        menu.addAction("+ Add to queue",      lambda: self._add_paths_to_queue(rows))
        menu.addSeparator()
        menu.addAction("📂  Open folder",      lambda: self._open_in_explorer(rows[0]))
        menu.addAction("💾  Save selection as M3U…", lambda: self._save_paths_as_m3u(rows))
        menu.exec(self._lib_table.viewport().mapToGlobal(pos))

    def _get_selected_paths(self) -> list[str]:
        seen: set[int] = set()
        paths: list[str] = []
        for idx in self._lib_table.selectedIndexes():
            r = idx.row()
            if r in seen:
                continue
            seen.add(r)
            item = self._lib_table.item(r, 1)
            if item:
                p = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if p:
                    paths.append(p)
        return paths

    # ── Playback ───────────────────────────────────────────────────────────

    def _play_path_now(self, path: str) -> None:
        """Immediately play a single path, inserting it as the next queue item."""
        row = _load_row(path)
        self._queue.insert(self._queue_index + 1 if self._queue_index >= 0 else 0, row)
        self._queue_index = self._queue_index + 1 if self._queue_index >= 0 else 0
        self._play_current()

    def _play_current(self) -> None:
        if not (0 <= self._queue_index < len(self._queue)):
            return
        row = self._queue[self._queue_index]
        path = row["path"]

        if self._media_player and self._vlc_instance:
            try:
                media = self._vlc_instance.media_new(path)
                self._media_player.set_media(media)
                self._media_player.play()
                # Apply volume after VLC has loaded the audio stream.
                # Setting it before play() is silently discarded because the
                # audio output hasn't been created yet.
                vol = self._vol_slider.value()
                QtCore.QTimer.singleShot(
                    250, lambda v=vol: self._apply_volume(v)
                )
            except Exception as exc:
                self._log(f"VLC error: {exc}", "error")
                return

        self._play_btn.setEnabled(True)
        self._play_btn.setText("⏸")
        self._stop_btn.setEnabled(True)
        self._seek_slider.setValue(0)
        self._pos_timer.start()

        # Highlight mode: seek to 30s, set auto-advance timer
        if self._highlight_chk.isChecked():
            QtCore.QTimer.singleShot(300, self._start_highlight_timer)

        # Update metadata display
        self._np_title.setText(row.get("title") or Path(path).stem)
        artist = row.get("artist", "")
        album  = row.get("album", "")
        sub = "  ·  ".join(p for p in [artist, album] if p)
        self._np_artist.setText(sub)

        # Highlight the playing row in queue list
        self._refresh_queue_list()

        # Record in recently played
        self._push_recent(row)

        # Load art async
        self._load_art(path)

        # Emit for NowPlayingBar
        self.now_playing_changed.emit(
            row.get("title") or Path(path).stem,
            artist, album, None,
        )
        self.playback_state_changed.emit(True)
        self._log(f"Playing: {row.get('title') or Path(path).name}", "ok")

    def _start_highlight_timer(self) -> None:
        if self._media_player:
            try:
                self._media_player.set_time(_HIGHLIGHT_START_MS)
            except Exception:
                pass
        self._highlight_timer.start(_HIGHLIGHT_DURATION_MS)

    def _highlight_advance(self) -> None:
        self.play_next()

    @Slot()
    def _on_play_pause(self) -> None:
        if not self._media_player:
            return
        if self._media_player.is_playing():
            self._media_player.pause()
            self._play_btn.setText("▶")
            self._pos_timer.stop()
            self.playback_state_changed.emit(False)
        else:
            if self._queue_index < 0 and self._queue:
                self._queue_index = 0
                self._play_current()
            else:
                self._media_player.play()
                self._play_btn.setText("⏸")
                self._pos_timer.start()
                self.playback_state_changed.emit(True)

    @Slot()
    def _on_stop(self) -> None:
        if self._media_player:
            self._media_player.stop()
        self._play_btn.setText("▶")
        self._play_btn.setEnabled(bool(self._queue))
        self._stop_btn.setEnabled(False)
        self._pos_timer.stop()
        self._highlight_timer.stop()
        self._seek_slider.setValue(0)
        self._pos_lbl.setText("0:00")
        self.playback_state_changed.emit(False)

    @Slot(int)
    def _on_seek(self, value: int) -> None:
        if self._media_player and self._media_player.get_length() > 0:
            self._media_player.set_position(value / 1000.0)

    def _seek_relative(self, delta_ms: int) -> None:
        if self._media_player:
            t = self._media_player.get_time()
            if t >= 0:
                self._media_player.set_time(max(0, t + delta_ms))

    # ── Volume ────────────────────────────────────────────────────────────

    @Slot(int)
    def _on_volume_drag(self, value: int) -> None:
        """Called only while the user is dragging the knob (not on programmatic
        moves), so set_volume() can reposition the slider without feedback."""
        self._apply_volume(value)

    @Slot()
    def _on_volume_released(self) -> None:
        """Ensure the final resting value is applied when the user lets go."""
        self._apply_volume(self._vol_slider.value())

    def _apply_volume(self, value: int) -> None:
        """Send *value* (0-100) to VLC, guarded against uninitialised state."""
        if self._media_player:
            try:
                self._media_player.audio_set_volume(value)
            except Exception:
                pass

    def set_volume(self, value: int) -> None:
        """Public API — update the slider position and apply to VLC.

        Safe to call from outside (e.g. NowPlayingBar volume knob) because the
        slider uses ``sliderMoved`` not ``valueChanged``, so repositioning it
        programmatically does not trigger a second VLC call.
        """
        value = max(0, min(100, value))
        self._vol_slider.setValue(value)
        self._apply_volume(value)

    def seek_to_ms(self, ms: int) -> None:
        """Public API — seek to an absolute position in milliseconds."""
        if self._media_player:
            try:
                self._media_player.set_time(max(0, ms))
            except Exception:
                pass

    @Slot()
    def _tick_position(self) -> None:
        if not self._media_player:
            return
        dur = self._media_player.get_length()
        pos = self._media_player.get_time()

        if dur > 0:
            self._seek_slider.setValue(int((pos / dur) * 1000))
            self._pos_lbl.setText(_fmt_ms(pos))
            self._dur_lbl.setText(_fmt_ms(dur))
            self.position_changed.emit(pos, dur)

        # Auto-advance when track ends
        if (
            self._vlc_state_ended is not None
            and self._media_player.get_state() == self._vlc_state_ended
        ):
            self._pos_timer.stop()
            if self._repeat == _REPEAT_TRACK:
                QtCore.QTimer.singleShot(50, self._play_current)
            else:
                QtCore.QTimer.singleShot(50, self.play_next)

    # ── Next / Previous ────────────────────────────────────────────────────

    def play_next(self) -> None:
        """Advance to the next queue item."""
        if not self._queue:
            return
        self._highlight_timer.stop()
        n = len(self._queue)
        if self._repeat == _REPEAT_TRACK:
            pass  # stay on same index
        elif self._shuffle:
            self._queue_index = random.randrange(n)
        else:
            self._queue_index = (self._queue_index + 1) % n
            if self._queue_index == 0 and self._repeat == _REPEAT_OFF:
                self._on_stop()
                return
        self._play_current()

    def play_prev(self) -> None:
        """Go back to the previous queue item (or restart current if >3s in)."""
        if not self._queue:
            return
        self._highlight_timer.stop()
        if self._media_player and self._media_player.get_time() > 3000:
            # Restart current track
            self._media_player.set_time(0)
            return
        n = len(self._queue)
        self._queue_index = (self._queue_index - 1) % n
        self._play_current()

    # ── Shuffle / Repeat ───────────────────────────────────────────────────

    @Slot(bool)
    def _on_shuffle_toggled(self, checked: bool) -> None:
        self._shuffle = checked

    def _cycle_repeat(self) -> None:
        self._repeat = (self._repeat + 1) % 3
        labels = {_REPEAT_OFF: ("🔁", "Repeat: Off"), _REPEAT_TRACK: ("🔂", "Repeat: Track"), _REPEAT_ALL: ("🔁", "Repeat: All")}
        icon, tip = labels[self._repeat]
        self._repeat_btn.setText(icon)
        self._repeat_btn.setToolTip(tip)

    def _on_highlight_toggled(self, checked: bool) -> None:
        if not checked:
            self._highlight_timer.stop()

    # ── Queue management ───────────────────────────────────────────────────

    def _add_paths_to_queue(self, paths: list[str]) -> None:
        for p in paths:
            self._queue.append(_load_row(p))
        self._refresh_queue_list()
        if self._queue_index < 0:
            self._queue_index = 0
            self._play_btn.setEnabled(True)

    def _add_selected_to_queue(self) -> None:
        paths = self._get_selected_paths()
        if not paths:
            self._status_lbl.setText("Select tracks in the library to add to queue.")
            return
        self._add_paths_to_queue(paths)
        self._status_lbl.setText(f"Added {len(paths)} track(s) to queue.")

    def _clear_queue(self) -> None:
        self._on_stop()
        self._queue.clear()
        self._queue_index = -1
        self._refresh_queue_list()
        self._np_title.setText("—")
        self._np_artist.setText("")
        self._cflow.set_front(None)
        self._cflow.clear_back()

    def _refresh_queue_list(self) -> None:
        self._queue_list.clear()
        for i, row in enumerate(self._queue):
            title = row.get("title") or Path(row["path"]).stem
            artist = row.get("artist", "")
            label = f"{i + 1}. {title}"
            if artist:
                label += f"  –  {artist}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
            if i == self._queue_index:
                item.setForeground(QtGui.QColor("#6366f1"))
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self._queue_list.addItem(item)
        # Scroll to current
        if 0 <= self._queue_index < self._queue_list.count():
            self._queue_list.scrollToItem(self._queue_list.item(self._queue_index))
        self._q_count_lbl.setText(
            f"{len(self._queue)} track{'s' if len(self._queue) != 1 else ''}"
        )

    @Slot(QtCore.QModelIndex)
    def _on_queue_double_click(self, index: QtCore.QModelIndex) -> None:
        item = self._queue_list.item(index.row())
        if item:
            qi = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(qi, int) and 0 <= qi < len(self._queue):
                self._queue_index = qi
                self._play_current()

    def _on_queue_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self._queue_list.itemAt(pos)
        if not item:
            return
        qi = item.data(QtCore.Qt.ItemDataRole.UserRole)
        menu = QtWidgets.QMenu(self)
        menu.addAction("▶  Play now",         lambda: self._jump_queue(qi))
        menu.addAction("✕  Remove from queue", lambda: self._remove_from_queue(qi))
        menu.exec(self._queue_list.mapToGlobal(pos))

    def _jump_queue(self, qi: int) -> None:
        if 0 <= qi < len(self._queue):
            self._queue_index = qi
            self._play_current()

    def _remove_from_queue(self, qi: int) -> None:
        if 0 <= qi < len(self._queue):
            self._queue.pop(qi)
            if self._queue_index >= qi and self._queue_index > 0:
                self._queue_index -= 1
            self._refresh_queue_list()

    # ── Recently played ────────────────────────────────────────────────────

    _MAX_RECENT = 30

    def _push_recent(self, row: dict) -> None:
        """Add *row* to the front of the recently-played list (deduped)."""
        path = row.get("path", "")
        self._recent_plays = [r for r in self._recent_plays if r.get("path") != path]
        self._recent_plays.insert(0, row)
        self._recent_plays = self._recent_plays[: self._MAX_RECENT]
        self._refresh_recent_list()

    def _refresh_recent_list(self) -> None:
        self._recent_list.clear()
        for row in self._recent_plays:
            title  = row.get("title") or Path(row["path"]).stem
            artist = row.get("artist", "")
            label  = f"{title}  –  {artist}" if artist else title
            item   = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, row["path"])
            self._recent_list.addItem(item)

    def _toggle_recent(self) -> None:
        collapsed = self._recent_list.height() == 0
        self._recent_list.setFixedHeight(120 if collapsed else 0)
        self._rp_toggle_btn.setText("▼" if collapsed else "▶")

    @Slot(QtCore.QModelIndex)
    def _on_recent_double_click(self, index: QtCore.QModelIndex) -> None:
        item = self._recent_list.item(index.row())
        if item:
            path = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if path:
                self._play_path_now(path)

    # ── File browser ───────────────────────────────────────────────────────

    @Slot()
    def _on_browse_file(self) -> None:
        start = self._library_path or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Audio File", start,
            "Audio (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All Files (*)"
        )
        if path:
            self._play_path_now(path)

    # ── Open in Explorer ───────────────────────────────────────────────────

    def _open_in_explorer(self, path: str) -> None:
        import sys, subprocess
        folder = str(Path(path).parent)
        try:
            if sys.platform == "win32":
                # Select the file in Explorer, not just open the folder
                subprocess.Popen(["explorer", "/select,", path])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            self._log(f"Could not open folder: {exc}", "warn")

    # ── Album art ──────────────────────────────────────────────────────────

    def _load_art(self, path: str) -> None:
        # Cancel any running loader
        if self._art_loader and self._art_loader.isRunning():
            self._art_loader.requestInterruption()
        loader = _ArtLoader(path, _ART_SIZE)
        loader.art_ready.connect(self._on_art_ready)
        self._art_loader = loader
        loader.start()

    @Slot(object, str)
    def _on_art_ready(self, pm, path: str) -> None:
        # Only apply if this is still the current track
        if self._queue_index < 0 or self._queue_index >= len(self._queue):
            return
        if self._queue[self._queue_index]["path"] != path:
            return
        if pm and not pm.isNull():
            self._cflow.set_front(pm)
            self._cflow.clear_back()        # playing track is now the front — dismiss back
            self._bg_art.set_art(pm)
            # Re-emit now_playing with art
            row = self._queue[self._queue_index]
            self.now_playing_changed.emit(
                row.get("title") or Path(path).stem,
                row.get("artist", ""),
                row.get("album", ""),
                pm,
            )
        else:
            self._cflow.set_front(None)

    @Slot(object, str)
    def _on_sel_art_ready(self, pm, path: str) -> None:
        """Art loaded for a single-clicked (selected but not playing) track."""
        if pm is None or pm.isNull():
            return
        # Determine which side: compare selected row to the currently playing
        # visible row index in the table (right if selection is below playing).
        sel_row = -1
        play_row = -1
        for r in range(self._lib_table.rowCount()):
            item = self._lib_table.item(r, 1)
            if item is None:
                continue
            p = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if p == path:
                sel_row = r
            if (self._queue_index >= 0 and self._queue_index < len(self._queue)
                    and p == self._queue[self._queue_index]["path"]):
                play_row = r
        right_side = sel_row >= play_row  # default True if play_row unknown
        self._cflow.set_back(pm, right_side=right_side)

    # ── M3U playlist ───────────────────────────────────────────────────────

    def _save_m3u(self) -> None:
        self._save_paths_as_m3u([r["path"] for r in self._queue])

    def _save_paths_as_m3u(self, paths: list[str]) -> None:
        if not paths:
            QtWidgets.QMessageBox.information(self, "Save M3U", "Queue is empty.")
            return
        playlists_dir = (
            os.path.join(self._library_path, "Playlists")
            if self._library_path else str(Path.home())
        )
        os.makedirs(playlists_dir, exist_ok=True)
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Playlist", playlists_dir, "M3U Playlist (*.m3u)"
        )
        if not dest:
            return
        try:
            with open(dest, "w", encoding="utf-8") as fh:
                fh.write("#EXTM3U\n")
                for p in paths:
                    fh.write(f"{p}\n")
            self._status_lbl.setText(f"Saved {len(paths)} tracks to {Path(dest).name}")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Save M3U", f"Could not save: {exc}")

    def _load_m3u(self) -> None:
        start = (
            os.path.join(self._library_path, "Playlists")
            if self._library_path and os.path.isdir(
                os.path.join(self._library_path, "Playlists")
            ) else (self._library_path or str(Path.home()))
        )
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Playlist", start, "M3U Playlist (*.m3u);;All Files (*)"
        )
        if not chosen:
            return
        try:
            with open(chosen, "r", encoding="utf-8") as fh:
                lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Load M3U", f"Could not read file: {exc}")
            return
        base = os.path.dirname(chosen)
        paths = []
        for line in lines:
            p = line if os.path.isabs(line) else os.path.normpath(os.path.join(base, line))
            if os.path.isfile(p):
                paths.append(p)
        if not paths:
            QtWidgets.QMessageBox.information(self, "Load M3U", "No valid tracks found.")
            return
        self.load_tracks_and_play(paths, Path(chosen).stem)

    # ── Public API ─────────────────────────────────────────────────────────

    def load_directory_and_play(self, dirpath: str) -> None:
        """Load all audio files in *dirpath* as the queue and start playing."""
        try:
            paths = sorted(
                str(p) for p in Path(dirpath).iterdir()
                if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
            )
        except OSError:
            return
        if not paths:
            return
        self.load_tracks_and_play(paths, Path(dirpath).name)

    def load_tracks_and_play(self, paths: list[str], label: str = "") -> None:
        """Replace the queue with *paths* and start playing from the first track.

        Called by other workspaces (duplicate finder, similarity) to send
        tracks to the player.
        """
        self._on_stop()
        self._queue = [_load_row(p) for p in paths]
        self._queue_index = 0
        self._refresh_queue_list()
        if label:
            self._status_lbl.setText(f"Now playing: {label}  ({len(paths)} tracks)")
        self._play_current()

    def toggle_shuffle(self) -> None:
        self._shuffle_btn.setChecked(not self._shuffle)

    def toggle_repeat(self) -> None:
        self._cycle_repeat()
