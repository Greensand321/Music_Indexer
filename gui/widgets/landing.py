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

import json
import os
import random
import subprocess
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

_SCAN_DEPTH = 7   # max folder depth searched for art (root = 0)

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
# _ArtHistory — persistent "do-not-repeat" log for the mosaic scanner
# ─────────────────────────────────────────────────────────────────────────────

# Audio extensions understood by read_metadata (including the opus reader).
_AUDIO_EXTS: frozenset[str] = frozenset(
    {".flac", ".m4a", ".aac", ".mp3", ".wav", ".ogg", ".opus"}
)


class _ArtHistory:
    """FIFO log of directory paths recently used for the landing mosaic.

    Persisted as a JSON array at ``~/.soundvault_art_history.json``.  Capped
    at ``MAX_ENTRIES`` paths; when full the oldest entries are evicted so the
    scanner keeps cycling through fresh parts of the library indefinitely.

    The scanner passes this object to ``_ordered_dirs()`` which puts unseen
    directories first (shuffled) and recently-seen directories last (also
    shuffled).  After emitting, the directories that contributed to the on-
    screen tiles are appended here and saved.
    """

    MAX_ENTRIES = 10_000
    _PATH = os.path.expanduser("~/.soundvault_art_history.json")

    def __init__(self) -> None:
        self._log: list[str] = []   # ordered oldest → newest
        self._set: set[str] = set() # fast membership test
        self._load()

    def _load(self) -> None:
        try:
            with open(self._PATH, encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                self._log = [p for p in raw if isinstance(p, str)]
                self._set = set(self._log)
        except Exception:
            pass

    def add_and_save(self, paths: list[str]) -> None:
        """Append *paths* (deduped), trim to MAX_ENTRIES, and persist."""
        for p in paths:
            if p not in self._set:
                self._log.append(p)
                self._set.add(p)
        if len(self._log) > self.MAX_ENTRIES:
            evicted    = self._log[: len(self._log) - self.MAX_ENTRIES]
            self._log  = self._log[-self.MAX_ENTRIES :]
            self._set -= set(evicted)
        try:
            with open(self._PATH, "w", encoding="utf-8") as fh:
                json.dump(self._log, fh, separators=(",", ":"))
        except Exception:
            pass

    def __contains__(self, path: str) -> bool:
        return path in self._set

    def __len__(self) -> int:
        return len(self._log)


# ─────────────────────────────────────────────────────────────────────────────
# _ArtScanner — background QThread that feeds album art to the mosaic tiles
# ─────────────────────────────────────────────────────────────────────────────

class _ArtScanner(QtCore.QThread):
    """Scan *library_path* for embedded album art and emit one image per tile.

    Cover extraction
    ----------------
    ``_cover_from_file()`` reads embedded covers directly from audio tags
    without relying on sidecar image files.  It tries mutagen first (pure
    Python, fast, no subprocess) and falls back to an ``ffmpeg`` subprocess
    for any format mutagen cannot handle.  Supported formats include:

    * **FLAC** — ``mutagen.File().pictures``
    * **MP3 / WAV / AIFF** — ID3 ``APIC`` frames via ``mutagen``
    * **M4A / AAC** — ``covr`` atom via ``mutagen``
    * **OGG Vorbis / OGG Opus** — ``metadata_block_picture`` (base64 + FLAC
      Picture) via ``mutagen``
    * **Everything else** — ``ffmpeg -an -vframes 1 -vcodec copy -f image2pipe``

    Directory ordering — fresh-first with history
    ---------------------------------------------
    ``_ordered_dirs()`` collects all directories up to ``_SCAN_DEPTH`` levels
    and splits them into two independently-shuffled buckets:

    * **Fresh** — not in ``_ArtHistory`` (never shown, or log was reset).
    * **Used**  — appeared in a previous run's selection.

    Fresh directories come first.  Within each directory the file list is also
    shuffled, so every launch reads a different track from each folder — giving
    variety even when multiple albums share a directory.

    Pool + selection
    ----------------
    ``_collect_covers()`` walks the ordered list, reading one audio file per
    directory until it finds a cover.  Unique covers (deduplicated by first
    64 bytes) are kept; duplicates are skipped and the next file tried.
    Scanning stops when ``_pool_cap`` (≥ ``tile_count × 5``, min 128) unique
    covers are gathered, or when the library is exhausted.

    ``random.sample(pool, tile_count)`` picks a fresh random subset each
    launch.  After emitting, the source directories are saved to
    ``_ArtHistory`` so the next run explores different folders first.
    """

    art_found = Signal(int, QtGui.QImage)  # (tile_index, image) — QImage is thread-safe

    _POOL_CAP_MULTIPLIER = 5   # pool = at least this many × tile_count

    def __init__(self, library_path: str, tile_count: int) -> None:
        super().__init__()
        self._library  = library_path
        self._n        = tile_count
        self._pool_cap = max(tile_count * self._POOL_CAP_MULTIPLIER, 128)
        self.setObjectName("ArtScanner")

    # ── Low-level helpers ─────────────────────────────────────────────────

    @staticmethod
    def _image_from_bytes(data: bytes) -> QtGui.QImage | None:
        img = QtGui.QImage()
        if img.loadFromData(data) and not img.isNull():
            return img
        return None

    @staticmethod
    def _scandir_files(dirpath: str) -> list[str]:
        """Shuffled list of filenames in *dirpath*; returns [] on OSError."""
        try:
            names = [e.name for e in os.scandir(dirpath) if e.is_file(follow_symlinks=False)]
        except OSError:
            return []
        random.shuffle(names)
        return names

    @staticmethod
    def _cover_from_file(path: str) -> bytes | None:
        """Return the first embedded cover image from *path*, or None.

        Each extraction attempt is wrapped in its own try/except so a failure
        in one path does not prevent the others from running.

        Order tried:
          1. mutagen .pictures  — FLAC, OGG Vorbis, OGG Opus
          2. ID3 APIC frames    — MP3, WAV, AIFF (mutagen)
          3. MP4 covr atom      — M4A, AAC (mutagen)
          4. metadata_block_picture — OGG raw dict access + base64 decode
          5. coverart field     — OGG simpler base64 (no Picture wrapper)
          6. ffmpeg temp-file   — universal fallback for anything mutagen misses
        """
        audio = None
        try:
            import mutagen
            audio = mutagen.File(path, easy=False)
        except Exception:
            pass

        if audio is not None:

            # 1. FLAC / OGG Vorbis / OGG Opus — .pictures property
            try:
                for pic in audio.pictures:
                    if pic.data:
                        return pic.data
            except Exception:
                pass

            # 2. ID3 APIC frames (MP3, WAV, AIFF …)
            try:
                tags = audio.tags
                if tags is not None and hasattr(tags, "getall"):
                    for frame in tags.getall("APIC:") or tags.getall("APIC"):
                        if getattr(frame, "data", None):
                            return frame.data
            except Exception:
                pass

            # 3. MP4 / M4A / AAC — covr atom
            try:
                tags = audio.tags
                if tags is not None:
                    covr = tags.get("covr")
                    if covr:
                        data = bytes(covr[0])
                        if data:
                            return data
            except Exception:
                pass

            # 4. OGG metadata_block_picture (base64-encoded FLAC Picture)
            #    OGG files are dict-like; access via audio.get(), not audio.tags.get()
            try:
                import base64
                from mutagen.flac import Picture
                mbp = audio.get("metadata_block_picture") or []
                if not isinstance(mbp, list):
                    mbp = [mbp]
                for item in mbp:
                    pic = Picture(base64.b64decode(str(item)))
                    if pic.data:
                        return pic.data
            except Exception:
                pass

            # 5. OGG coverart (raw base64 image bytes, no Picture wrapper)
            try:
                import base64
                ca = audio.get("coverart") or []
                if not isinstance(ca, list):
                    ca = [ca]
                for item in ca:
                    data = base64.b64decode(str(item))
                    if data:
                        return data
            except Exception:
                pass

        # 6. ffmpeg fallback — write to a temp file to avoid pipe format issues
        tmppath = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmppath = tmp.name
            result = subprocess.run(
                [
                    "ffmpeg", "-v", "quiet", "-y", "-i", path,
                    "-an", "-map", "0:v:0", "-vframes", "1", tmppath,
                ],
                capture_output=True,
                timeout=8,
            )
            if result.returncode == 0:
                with open(tmppath, "rb") as fh:
                    data = fh.read()
                if data:
                    return data
        except Exception:
            pass
        finally:
            if tmppath:
                try:
                    os.unlink(tmppath)
                except OSError:
                    pass

        return None

    # ── Directory ordering ────────────────────────────────────────────────

    def _ordered_dirs(self, history: _ArtHistory) -> list[str]:
        """All library dirs up to _SCAN_DEPTH, fresh dirs first.

        One lightweight walk — no file content reads.  Each bucket is
        independently shuffled so traversal order differs on every launch.
        """
        fresh: list[str] = []
        used:  list[str] = []
        for dirpath, dirs, _files in os.walk(self._library):
            rel   = os.path.relpath(dirpath, self._library)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth >= _SCAN_DEPTH:
                dirs.clear()
            (used if dirpath in history else fresh).append(dirpath)
        random.shuffle(fresh)
        random.shuffle(used)
        return fresh + used

    # ── Cover collection ──────────────────────────────────────────────────

    def _collect_covers(
        self, dirpaths: list[str]
    ) -> list[tuple[str, bytes]]:
        """Return ``(dirpath, cover_bytes)`` up to ``_pool_cap`` unique covers.

        For each directory, audio files are scanned in shuffled order.  The
        first file that yields a cover terminates the per-directory scan
        (whether unique or a duplicate already in the pool).  Unique covers
        are kept; duplicates are skipped and the next file is tried instead.
        Scanning continues through the full ordered list — if fresh dirs run
        dry, used dirs fill any remaining gap (backtrack semantics).
        """
        results:   list[tuple[str, bytes]] = []
        seen_sigs: set[bytes] = set()

        for dirpath in dirpaths:
            if self.isInterruptionRequested():
                return results
            for name in self._scandir_files(dirpath):
                if os.path.splitext(name)[1].lower() not in _AUDIO_EXTS:
                    continue
                data = self._cover_from_file(os.path.join(dirpath, name))
                if not data:
                    continue
                # Found a cover in this directory — record if unique, then move on.
                sig = data[:64]
                if sig not in seen_sigs:
                    seen_sigs.add(sig)
                    results.append((dirpath, data))
                    if len(results) >= self._pool_cap:
                        return results
                break   # one cover attempt per directory (unique or duplicate)

        return results

    # ── Orchestration + emit ──────────────────────────────────────────────

    def run(self) -> None:  # noqa: N802
        history  = _ArtHistory()
        dirpaths = self._ordered_dirs(history)
        if self.isInterruptionRequested():
            return

        pool = self._collect_covers(dirpaths)
        if self.isInterruptionRequested() or not pool:
            return

        selected = random.sample(pool, min(len(pool), self._n))

        for tile_index, (_, cover) in enumerate(selected):
            if self.isInterruptionRequested():
                return
            img = self._image_from_bytes(cover)
            if img is not None:
                self.art_found.emit(tile_index, img)

        # Deprioritise the shown directories on the next run.
        history.add_and_save([dirpath for dirpath, _ in selected])


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
