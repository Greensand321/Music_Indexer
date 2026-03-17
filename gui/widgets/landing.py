"""AlphaDEX — Mosaic Reveal landing page.

Shown once after the splash screen fades.  A 7 × 5 grid of colourful
album-art placeholder tiles flies in from off-screen with a staggered delay,
assembling into a full-window mosaic.  A frosted-glass CTA card floats in the
reserved centre block with the app name and an "Open Library" button.

Sequence
--------
1. ``show_animated()`` — fade the window in, then start tile fly-in.
2. User clicks "Open Library" → ``QFileDialog`` → ``_accept(path)``.
   (Or "Continue" if a saved library path is available.)
3. Any still-running fly-in is stopped; all tiles snap to their resting
   positions, then scatter to random off-screen targets.
4. ``library_selected(path)`` is emitted at the *start* of the fade-out so
   the caller can cross-fade the main window in simultaneously.
5. ``finished`` is emitted after the landing window is hidden.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal

# ── Grid constants ─────────────────────────────────────────────────────────────
_COLS      = 7
_ROWS      = 5
_TILE_SZ   = 110   # px, square
_GAP       = 10    # px between tiles

# Centre block reserved for the CTA card (0-based indices)
_SKIP_COLS = frozenset({2, 3, 4})
_SKIP_ROWS = frozenset({1, 2, 3})

# ── Timing ────────────────────────────────────────────────────────────────────
_STAGGER_MS  = 38    # delay increment per tile during fly-in
_FLY_IN_MS   = 500   # each tile's fly-in duration
_FLY_OUT_MS  = 360   # each tile's scatter duration
_SCATTER_MAX = 160   # max random pre-scatter delay (ms)
_FADE_IN_MS  = 320   # landing window fade-in

# Exported so alpha_dex_gui.py can match the cross-fade duration exactly.
FADE_OUT_MS  = 420   # landing window fade-out / main-window fade-in

# ── Album-art sidecar detection ───────────────────────────────────────────────
_ART_NAMES = ("cover", "folder", "front", "albumart", "artwork", "album", "thumb")
_ART_EXTS  = (".jpg", ".jpeg", ".png", ".webp")

# ── Colour pool — diagonal gradient pairs for placeholder tiles ───────────────
_GRADS: list[tuple[str, str]] = [
    ("#6366f1", "#a78bfa"),
    ("#0ea5e9", "#38bdf8"),
    ("#10b981", "#34d399"),
    ("#f59e0b", "#fbbf24"),
    ("#ef4444", "#f87171"),
    ("#8b5cf6", "#c4b5fd"),
    ("#06b6d4", "#67e8f9"),
    ("#ec4899", "#f9a8d4"),
    ("#84cc16", "#a3e635"),
    ("#f97316", "#fb923c"),
    ("#14b8a6", "#2dd4bf"),
    ("#a855f7", "#d8b4fe"),
    ("#3b82f6", "#93c5fd"),
    ("#d946ef", "#f0abfc"),
    ("#22c55e", "#86efac"),
    ("#eab308", "#fde047"),
    ("#64748b", "#94a3b8"),
    ("#1e40af", "#60a5fa"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _darken(hex_color: str, pct: int) -> str:
    """Return a version of *hex_color* darkened by *pct* percent (HSV value)."""
    c = QtGui.QColor(hex_color)
    h, s, v, a = c.getHsvF()
    c.setHsvF(h, s, max(0.0, v - pct / 100.0), a)
    return c.name()


# ─────────────────────────────────────────────────────────────────────────────
# _Tile
# ─────────────────────────────────────────────────────────────────────────────

class _Tile(QtWidgets.QWidget):
    """Single album-art placeholder tile — a colourful diagonal gradient square."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        grad: tuple[str, str],
        target: QtCore.QPoint,
    ) -> None:
        super().__init__(parent)
        self._grad   = grad
        self._target = target
        self._pixmap: QtGui.QPixmap | None = None
        self.setFixedSize(_TILE_SZ, _TILE_SZ)
        # Prevent Qt from pre-filling the background so rounded corners show
        # the parent's gradient through them.
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)

    def set_pixmap(self, pm: QtGui.QPixmap) -> None:
        """Set album-art pixmap; tile repaints itself automatically."""
        self._pixmap = pm
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(self.rect())

            path = QtGui.QPainterPath()
            path.addRoundedRect(r, 10, 10)

            if self._pixmap is not None:
                # Clip to rounded rect and draw scaled pixmap
                p.setClipPath(path)
                scaled = self._pixmap.scaled(
                    self.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                ox = (scaled.width()  - self.width())  // 2
                oy = (scaled.height() - self.height()) // 2
                p.drawPixmap(QtCore.QPoint(-ox, -oy), scaled)
                p.setClipping(False)

                # Vignette overlay for depth
                vignette = QtGui.QRadialGradient(
                    r.center(), max(r.width(), r.height()) * 0.75
                )
                vignette.setColorAt(0.5, QtGui.QColor(0, 0, 0, 0))
                vignette.setColorAt(1.0, QtGui.QColor(0, 0, 0, 80))
                p.fillPath(path, QtGui.QBrush(vignette))
            else:
                grad = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
                grad.setColorAt(0.0, QtGui.QColor(self._grad[0]))
                grad.setColorAt(1.0, QtGui.QColor(self._grad[1]))
                p.fillPath(path, QtGui.QBrush(grad))

                # Subtle top-edge sheen — gives a soft "depth" impression
                p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 42), 1.0))
                p.drawLine(
                    QtCore.QPointF(r.left() + 12, r.top() + 1.0),
                    QtCore.QPointF(r.right() - 12, r.top() + 1.0),
                )

                # Bottom-right dark overlay to hint at depth / shadow
                shadow = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
                shadow.setColorAt(0.55, QtGui.QColor(0, 0, 0, 0))
                shadow.setColorAt(1.0,  QtGui.QColor(0, 0, 0, 55))
                p.fillPath(path, QtGui.QBrush(shadow))
        finally:
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# _CTACard
# ─────────────────────────────────────────────────────────────────────────────

class _CTACard(QtWidgets.QFrame):
    """Frosted-glass centre card: app name, tagline, and library buttons."""

    open_clicked  = Signal()
    reuse_clicked = Signal()

    def __init__(self, parent: QtWidgets.QWidget, saved_path: str = "") -> None:
        super().__init__(parent)
        self._saved = saved_path
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self._build()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build(self) -> None:
        try:
            from gui.themes.manager import get_manager
            t            = get_manager().current
            accent       = t.accent
            text_primary = t.text_primary
            text_muted   = t.text_secondary
        except Exception:
            accent, text_primary, text_muted = "#6366f1", "#f8fafc", "#94a3b8"

        try:
            from gui.fonts.loader import UI_FAMILY
        except ImportError:
            UI_FAMILY = "Arial"

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(44, 52, 44, 52)
        lay.setSpacing(0)
        lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # ── Decorative accent bar ──────────────────────────────────────────
        bar = QtWidgets.QFrame()
        bar.setFixedSize(36, 4)
        bar.setStyleSheet(
            f"background: {accent}; border-radius: 2px; border: none;"
        )
        bar_wrap = QtWidgets.QHBoxLayout()
        bar_wrap.setContentsMargins(0, 0, 0, 0)
        bar_wrap.addStretch()
        bar_wrap.addWidget(bar)
        bar_wrap.addStretch()
        lay.addLayout(bar_wrap)
        lay.addSpacing(20)

        # ── App name ──────────────────────────────────────────────────────
        name_lbl = QtWidgets.QLabel("AlphaDEX")
        nf = QtGui.QFont(UI_FAMILY, 34)
        nf.setWeight(QtGui.QFont.Weight.Bold)
        nf.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
        name_lbl.setFont(nf)
        name_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet(
            f"color: {text_primary}; background: transparent; letter-spacing: -1px;"
        )

        # ── Tagline ───────────────────────────────────────────────────────
        tag_lbl = QtWidgets.QLabel("Your library, organized.")
        tf = QtGui.QFont(UI_FAMILY, 12)
        tag_lbl.setFont(tf)
        tag_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        tag_lbl.setStyleSheet(
            f"color: {text_muted}; background: transparent; letter-spacing: 0.3px;"
        )

        lay.addWidget(name_lbl)
        lay.addSpacing(6)
        lay.addWidget(tag_lbl)
        lay.addSpacing(36)

        # ── "Continue" button (existing library) ──────────────────────────
        if self._saved:
            short = Path(self._saved).name or self._saved
            reuse_btn = QtWidgets.QPushButton(f"Continue  ·  {short}")
            reuse_btn.setFixedHeight(38)
            reuse_btn.setCursor(
                QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            )
            reuse_btn.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(255,255,255,0.07);
                    color: {text_muted};
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 8px;
                    font-family: '{UI_FAMILY}';
                    font-size: 12px;
                    padding: 0 18px;
                }}
                QPushButton:hover {{
                    background: rgba(255,255,255,0.13);
                    color: {text_primary};
                    border-color: rgba(255,255,255,0.18);
                }}
            """)
            reuse_btn.clicked.connect(self.reuse_clicked.emit)
            lay.addWidget(reuse_btn)
            lay.addSpacing(8)

        # ── "Open Library" button (primary CTA) ───────────────────────────
        open_btn = QtWidgets.QPushButton("Open Library  →")
        open_btn.setFixedHeight(46)
        open_btn.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        open_btn.setStyleSheet(f"""
            QPushButton {{
                background: {accent};
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-family: '{UI_FAMILY}';
                font-size: 14px;
                font-weight: 600;
                padding: 0 28px;
                letter-spacing: 0.3px;
            }}
            QPushButton:hover  {{ background: {_darken(accent, 12)}; }}
            QPushButton:pressed {{ background: {_darken(accent, 26)}; }}
        """)
        open_btn.clicked.connect(self.open_clicked.emit)
        lay.addWidget(open_btn)

    # ── Paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = QtCore.QRectF(self.rect())

            path = QtGui.QPainterPath()
            path.addRoundedRect(r, 20, 20)

            # Dark semi-transparent fill — frosted glass look
            p.fillPath(path, QtGui.QBrush(QtGui.QColor(10, 13, 20, 218)))

            # Hairline border
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 22), 1.0))
            p.drawPath(path)

            # Subtle top-edge inner glow
            glow = QtGui.QLinearGradient(
                QtCore.QPointF(r.left(), r.top()),
                QtCore.QPointF(r.left(), r.top() + 60),
            )
            glow.setColorAt(0.0, QtGui.QColor(255, 255, 255, 14))
            glow.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
            p.fillPath(path, QtGui.QBrush(glow))
        finally:
            p.end()


# ─────────────────────────────────────────────────────────────────────────────
# _ArtScanner — background QThread that feeds album art to the mosaic tiles
# ─────────────────────────────────────────────────────────────────────────────

class _ArtScanner(QtCore.QThread):
    """Scan *library_path* for album art and emit one QPixmap per tile slot.

    Strategy
    --------
    1. Walk up to two directory levels (artist / album) and collect sidecar
       image files whose stem matches ``_ART_NAMES`` (fast, no file I/O for
       audio content).
    2. If fewer images are found than tiles needed, fall back to reading
       embedded cover art from audio files via
       ``utils.audio_metadata_reader.read_metadata``.
    3. Shuffle the collected paths so the mosaic looks varied on each run,
       then emit ``art_found(tile_index, pixmap)`` for every tile slot.
       Tiles cycle through the found images when there are fewer images
       than tiles.
    """

    art_found = Signal(int, QtGui.QImage)  # (tile_index, image) — QImage is thread-safe

    def __init__(self, library_path: str, tile_count: int) -> None:
        super().__init__()
        self._library  = library_path
        self._n        = tile_count
        self.setObjectName("ArtScanner")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_sidecar(name: str) -> bool:
        stem, ext = os.path.splitext(name.lower())
        return stem in _ART_NAMES and ext in _ART_EXTS

    @staticmethod
    def _image_from_bytes(data: bytes) -> QtGui.QImage | None:
        img = QtGui.QImage()
        if img.loadFromData(data) and not img.isNull():
            return img
        return None

    # ── Main scan loop ────────────────────────────────────────────────────

    def run(self) -> None:  # noqa: N802
        art_paths: list[str] = []

        # Phase 1 — sidecar image files (very fast)
        try:
            root = self._library
            with os.scandir(root) as lvl1:
                for artist_entry in lvl1:
                    if self.isInterruptionRequested():
                        return
                    if not artist_entry.is_dir(follow_symlinks=False):
                        continue
                    try:
                        with os.scandir(artist_entry.path) as lvl2:
                            for album_entry in lvl2:
                                if self.isInterruptionRequested():
                                    return
                                if not album_entry.is_dir(follow_symlinks=False):
                                    continue
                                try:
                                    with os.scandir(album_entry.path) as lvl3:
                                        for f in lvl3:
                                            if (
                                                f.is_file(follow_symlinks=False)
                                                and self._is_sidecar(f.name)
                                            ):
                                                art_paths.append(f.path)
                                except OSError:
                                    pass
                    except OSError:
                        pass
        except OSError:
            return

        random.shuffle(art_paths)

        # Phase 2 — embedded tags fallback (only if we need more art)
        embedded_bytes: list[bytes] = []
        if len(art_paths) < self._n:
            audio_exts = {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
            try:
                from utils.audio_metadata_reader import read_metadata
                root = self._library
                for dirpath, _dirs, files in os.walk(root):
                    if self.isInterruptionRequested():
                        return
                    for fname in files:
                        if self.isInterruptionRequested():
                            return
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in audio_exts:
                            continue
                        try:
                            _tags, covers, _err, _hint = read_metadata(
                                os.path.join(dirpath, fname),
                                include_cover=True,
                            )
                            if covers:
                                embedded_bytes.append(covers[0])
                                if len(art_paths) + len(embedded_bytes) >= self._n:
                                    break
                        except Exception:
                            pass
                    if len(art_paths) + len(embedded_bytes) >= self._n:
                        break
            except ImportError:
                pass

            random.shuffle(embedded_bytes)

        if not art_paths and not embedded_bytes:
            return

        # Emit one QImage per tile slot, cycling through available art.
        # QImage (unlike QPixmap) is safe to create in a non-GUI thread.
        tile_index = 0
        total_path = len(art_paths)
        total_emb  = len(embedded_bytes)

        for i in range(self._n):
            if self.isInterruptionRequested():
                return

            img: QtGui.QImage | None = None

            # Try sidecar path first (cycling)
            if total_path > 0:
                candidate = QtGui.QImage(art_paths[i % total_path])
                if not candidate.isNull():
                    img = candidate

            # Fall back to embedded bytes (cycling)
            if img is None and total_emb > 0:
                img = self._image_from_bytes(embedded_bytes[i % total_emb])

            if img is not None and not img.isNull():
                self.art_found.emit(tile_index, img)
                tile_index += 1


# ─────────────────────────────────────────────────────────────────────────────
# MosaicLanding
# ─────────────────────────────────────────────────────────────────────────────

class MosaicLanding(QtWidgets.QWidget):
    """Full-window Mosaic Reveal landing shown between the splash and the main UI.

    Usage::

        landing = MosaicLanding(geometry, saved_path)
        landing.library_selected.connect(_on_library_selected)
        landing.finished.connect(_on_done)
        landing.show_animated()
    """

    library_selected = Signal(str)   # emitted at start of fade-out; path is ready
    finished         = Signal()      # emitted after window is hidden

    def __init__(
        self,
        geometry: QtCore.QRect,
        saved_path: str = "",
    ) -> None:
        super().__init__(None, QtCore.Qt.WindowType.FramelessWindowHint)
        self.setGeometry(geometry)
        # Prevent the OS from maximizing or resizing a frameless window,
        # which would cause concurrent paintEvents at unexpected sizes.
        self.setFixedSize(geometry.width(), geometry.height())
        self._w       = geometry.width()
        self._h       = geometry.height()
        self._saved   = saved_path
        self._pending = ""

        self._tiles: list[_Tile] = []
        self._fly_in_grp:  QtCore.QParallelAnimationGroup | None = None
        self._fly_out_grp: QtCore.QParallelAnimationGroup | None = None
        # Keep animation references alive
        self._fade_in_anim:  object = None
        self._fade_out_anim: object = None
        self._scanner: object = None   # _ArtScanner | None

        self._compute_grid()
        self._build_tiles()
        self._build_cta()

    # ── Grid geometry ──────────────────────────────────────────────────────

    def _compute_grid(self) -> None:
        gw = _COLS * _TILE_SZ + (_COLS - 1) * _GAP
        gh = _ROWS * _TILE_SZ + (_ROWS - 1) * _GAP
        ox = (self._w - gw) // 2
        oy = (self._h - gh) // 2
        self._origin = QtCore.QPoint(ox, oy)

        sc0, sc1 = min(_SKIP_COLS), max(_SKIP_COLS)
        sr0, sr1 = min(_SKIP_ROWS), max(_SKIP_ROWS)
        cx = ox + sc0 * (_TILE_SZ + _GAP)
        cy = oy + sr0 * (_TILE_SZ + _GAP)
        cw = (sc1 - sc0 + 1) * _TILE_SZ + (sc1 - sc0) * _GAP
        ch = (sr1 - sr0 + 1) * _TILE_SZ + (sr1 - sr0) * _GAP
        self._center_rect = QtCore.QRect(cx, cy, cw, ch)

    def _target_for(self, col: int, row: int) -> QtCore.QPoint:
        return QtCore.QPoint(
            self._origin.x() + col * (_TILE_SZ + _GAP),
            self._origin.y() + row * (_TILE_SZ + _GAP),
        )

    def _off_screen(self, target: QtCore.QPoint) -> QtCore.QPoint:
        """Return the off-screen start position for a tile (radially outward)."""
        tcx = target.x() + _TILE_SZ * 0.5
        tcy = target.y() + _TILE_SZ * 0.5
        wcx, wcy = self._w * 0.5, self._h * 0.5
        dx, dy = tcx - wcx, tcy - wcy
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            dx, dy = 1.0, 0.0
        sx = (wcx + _TILE_SZ) / (abs(dx) + 0.001)
        sy = (wcy + _TILE_SZ) / (abs(dy) + 0.001)
        scale = min(sx, sy) * 1.3
        return QtCore.QPoint(
            int(wcx + dx * scale - _TILE_SZ * 0.5),
            int(wcy + dy * scale - _TILE_SZ * 0.5),
        )

    def _scatter_target(self, target: QtCore.QPoint) -> QtCore.QPoint:
        """Return a randomised scatter destination beyond the window edges."""
        tcx = target.x() + _TILE_SZ * 0.5
        tcy = target.y() + _TILE_SZ * 0.5
        wcx, wcy = self._w * 0.5, self._h * 0.5
        dx = (tcx - wcx) + random.uniform(-90, 90)
        dy = (tcy - wcy) + random.uniform(-90, 90)
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            dx, dy = 60.0, 60.0
        sx = (wcx + _TILE_SZ * 2.5) / (abs(dx) + 0.001)
        sy = (wcy + _TILE_SZ * 2.5) / (abs(dy) + 0.001)
        scale = min(sx, sy) * 1.45
        return QtCore.QPoint(
            int(wcx + dx * scale - _TILE_SZ * 0.5),
            int(wcy + dy * scale - _TILE_SZ * 0.5),
        )

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_tiles(self) -> None:
        pool = (_GRADS * 4)[: _COLS * _ROWS]
        random.shuffle(pool)
        i = 0
        for row in range(_ROWS):
            for col in range(_COLS):
                if col in _SKIP_COLS and row in _SKIP_ROWS:
                    continue
                target = self._target_for(col, row)
                tile   = _Tile(self, pool[i % len(pool)], target)
                tile.move(self._off_screen(target))
                tile.show()
                self._tiles.append(tile)
                i += 1

        if self._saved:
            self._start_art_scanner()

    def _build_cta(self) -> None:
        self._cta = _CTACard(self, self._saved)
        self._cta.setGeometry(self._center_rect)
        self._cta.open_clicked.connect(self._on_open_clicked)
        self._cta.reuse_clicked.connect(
            lambda: self._accept(self._saved)
        )
        self._cta.raise_()
        self._cta.show()

    # ── Public API ────────────────────────────────────────────────────────

    def show_animated(self) -> None:
        """Fade the window in, then start the tile mosaic fly-in animation."""
        self.setWindowOpacity(0.0)
        self.show()
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(_FADE_IN_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        anim.finished.connect(self._fly_in)
        self._fade_in_anim = anim
        anim.start()

    # ── Fly-in ────────────────────────────────────────────────────────────

    def _fly_in(self) -> None:
        """Staggered fly-in: tiles arrive in random order from off-screen edges."""
        order = list(range(len(self._tiles)))
        random.shuffle(order)

        grp = QtCore.QParallelAnimationGroup(self)
        for seq_i, tile_i in enumerate(order):
            tile = self._tiles[tile_i]
            seq  = QtCore.QSequentialAnimationGroup(grp)
            seq.addAnimation(QtCore.QPauseAnimation(seq_i * _STAGGER_MS))

            anim = QtCore.QPropertyAnimation(tile, b"pos")
            anim.setStartValue(self._off_screen(tile._target))
            anim.setEndValue(tile._target)
            anim.setDuration(_FLY_IN_MS)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutBack)
            seq.addAnimation(anim)

            grp.addAnimation(seq)

        self._fly_in_grp = grp
        grp.start()

    # ── User action handlers ───────────────────────────────────────────────

    def _on_open_clicked(self) -> None:
        start = self._saved or str(Path.home())

        # Standalone top-level dialog (no parent) to avoid compositor
        # blank-outs on Linux; manually centred over this landing window.
        dlg = QtWidgets.QFileDialog()
        dlg.setWindowTitle("Select Music Library Folder")
        dlg.setDirectory(start)
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        dlg.resize(820, 560)
        if self.isVisible():
            fg = self.frameGeometry()
            dlg.move(
                fg.left() + (fg.width()  - dlg.width())  // 2,
                fg.top()  + (fg.height() - dlg.height()) // 2,
            )

        if dlg.exec():
            selected = dlg.selectedFiles()
            if selected:
                self._accept(selected[0])

    def _accept(self, path: str) -> None:
        """Validate the chosen path and start the exit sequence."""
        if not path:
            return
        self._pending = path

        # Stop any running fly-in and snap all tiles to their resting positions
        if (
            self._fly_in_grp is not None
            and self._fly_in_grp.state()
            == QtCore.QAbstractAnimation.State.Running
        ):
            self._fly_in_grp.stop()
        for tile in self._tiles:
            tile.move(tile._target)

        self._do_scatter()

    # ── Scatter / exit ────────────────────────────────────────────────────

    def _do_scatter(self) -> None:
        """Scatter tiles to random off-screen positions, then fade out."""
        order = list(range(len(self._tiles)))
        random.shuffle(order)

        grp = QtCore.QParallelAnimationGroup(self)
        for tile_i in order:
            tile = self._tiles[tile_i]
            seq  = QtCore.QSequentialAnimationGroup(grp)
            seq.addAnimation(
                QtCore.QPauseAnimation(random.randint(0, _SCATTER_MAX))
            )
            anim = QtCore.QPropertyAnimation(tile, b"pos")
            anim.setStartValue(tile.pos())
            anim.setEndValue(self._scatter_target(tile._target))
            anim.setDuration(_FLY_OUT_MS)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.InBack)
            seq.addAnimation(anim)
            grp.addAnimation(seq)

        self._fly_out_grp = grp
        grp.finished.connect(self._fade_out)
        grp.start()

    def _fade_out(self) -> None:
        # Emit path now so the caller can start fading the main window in,
        # creating a simultaneous cross-dissolve while we fade out.
        self.library_selected.emit(self._pending)

        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setDuration(FADE_OUT_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        anim.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        anim.finished.connect(self._done)
        self._fade_out_anim = anim
        anim.start()

    def _done(self) -> None:
        if self._scanner is not None:
            self._scanner.requestInterruption()
            self._scanner.quit()
            self._scanner.wait(800)
            self._scanner = None
        self.hide()
        self.finished.emit()

    # ── Art scanner ───────────────────────────────────────────────────────

    def _start_art_scanner(self) -> None:
        scanner = _ArtScanner(self._saved, len(self._tiles))
        scanner.art_found.connect(self._on_art_found)
        self._scanner = scanner
        scanner.start()

    def _on_art_found(self, tile_index: int, image: QtGui.QImage) -> None:
        # We are back on the main thread here; QPixmap conversion is safe.
        if 0 <= tile_index < len(self._tiles):
            pm = QtGui.QPixmap.fromImage(image)
            if not pm.isNull():
                self._tiles[tile_index].set_pixmap(pm)

    # ── Background painting ────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            from gui.themes.manager import get_manager
            from gui.widgets.gradient_bg import (
                _gradient_enabled,
                paint_window_gradient,
            )
            t = get_manager().current
            # Solid base (same as sidebar_bg so it blends with the main window)
            p.fillRect(self.rect(), QtGui.QColor(t.sidebar_bg))
            if _gradient_enabled():
                paint_window_gradient(
                    p, self, t,
                    radius_scale=0.85,
                    alpha_tr=55,
                    alpha_bl=70,
                )
        except Exception:
            p.fillRect(self.rect(), QtGui.QColor("#0d1117"))
        finally:
            p.end()
