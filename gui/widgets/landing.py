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

import concurrent.futures
import hashlib
import json
import os
import random
import sys
import time
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

# ── Art-scanner diagnostics ────────────────────────────────────────────────────
# Set True to print per-file extraction results to the terminal on startup.
# Flip to False once you have confirmed which extraction path your library uses.
_ART_SCAN_DEBUG: bool = True

# Pause between the window finishing its fade-in and the tiles starting to fly.
# During this window the CTA card ("AlphaDEX") is the only visible element —
# this is the merged "logo moment" that replaces the old SplashScreen.
_LOGO_PAUSE_MS: int = 600

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

    clicked = Signal(str)   # emitted on click; payload is the directory path

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        grad: tuple[str, str],
        target: QtCore.QPoint,
    ) -> None:
        super().__init__(parent)
        self._grad        = grad          # kept for lazy placeholder bake
        self._target      = target
        self._dirpath:    str             = ""   # set when art scanner assigns this tile
        self._ready_pm:   QtGui.QPixmap | None = None
        self._placeholder: QtGui.QPixmap | None = None  # baked on first paintEvent
        self.setFixedSize(_TILE_SZ, _TILE_SZ)
        # Prevent Qt from pre-filling the background so rounded corners show
        # the parent's gradient through them.
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

    def set_pixmap(self, pm: QtGui.QPixmap) -> None:
        """Store the pre-baked pixmap (baking done off-thread); tile repaints."""
        self._ready_pm = pm
        self.update()

    def set_dirpath(self, path: str) -> None:
        """Record which library directory this tile's cover came from."""
        self._dirpath = path

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._dirpath:
            self.clicked.emit(self._dirpath)
        super().mousePressEvent(event)

    # ── Bake helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _bake_placeholder(grad: tuple[str, str], size: int, radius: int = 10) -> QtGui.QPixmap:
        """Render gradient + sheen + shadow into a QPixmap once.
        Must be called on the main (GUI) thread."""
        out = QtGui.QPixmap(size, size)
        out.fill(QtGui.QColor(0, 0, 0, 0))
        r = QtCore.QRectF(0, 0, size, size)
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, radius, radius)
        p = QtGui.QPainter(out)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            lg = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
            lg.setColorAt(0.0, QtGui.QColor(grad[0]))
            lg.setColorAt(1.0, QtGui.QColor(grad[1]))
            p.fillPath(path, QtGui.QBrush(lg))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 42), 1.0))
            p.drawLine(
                QtCore.QPointF(12, 1.0),
                QtCore.QPointF(size - 12, 1.0),
            )
            shadow = QtGui.QLinearGradient(r.topLeft(), r.bottomRight())
            shadow.setColorAt(0.55, QtGui.QColor(0, 0, 0, 0))
            shadow.setColorAt(1.0,  QtGui.QColor(0, 0, 0, 55))
            p.fillPath(path, QtGui.QBrush(shadow))
        finally:
            p.end()
        return out

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        # Lazy-bake the placeholder on first paint so __init__ stays fast.
        if self._placeholder is None and self._ready_pm is None:
            self._placeholder = self._bake_placeholder(self._grad, _TILE_SZ)
        p = QtGui.QPainter(self)
        try:
            src = self._ready_pm if self._ready_pm is not None else self._placeholder
            p.drawPixmap(0, 0, src)
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
        nf = QtGui.QFont(UI_FAMILY, 42)
        nf.setWeight(QtGui.QFont.Weight.Bold)
        nf.setHintingPreference(QtGui.QFont.HintingPreference.PreferNoHinting)
        name_lbl.setFont(nf)
        name_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet(
            "color: #ffffff; background: transparent; letter-spacing: -1px;"
        )

        # ── Tagline ───────────────────────────────────────────────────────
        tag_lbl = QtWidgets.QLabel("Your library, organized.")
        tf = QtGui.QFont(UI_FAMILY, 15)
        tag_lbl.setFont(tf)
        tag_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        tag_lbl.setStyleSheet(
            f"color: {text_muted}; background: transparent; letter-spacing: 0.3px;"
        )

        lay.addWidget(name_lbl)
        lay.addSpacing(6)
        lay.addWidget(tag_lbl)
        lay.addSpacing(36)

        # ── Single smart button ────────────────────────────────────────────
        # Button text and behavior depends on whether a library is saved:
        # - Saved library: "Go" → uses saved library
        # - No saved library: "Choose Library Folder" → opens file dialog
        if self._saved:
            btn_text = "Initialize"
            btn_tooltip = f"Open saved library: {self._saved}"
            btn_action = self.reuse_clicked.emit
        else:
            btn_text = "Choose Library Folder"
            btn_tooltip = "Select a music library folder"
            btn_action = self.open_clicked.emit

        main_btn = QtWidgets.QPushButton(btn_text)
        main_btn.setFixedHeight(46)
        main_btn.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        main_btn.setToolTip(btn_tooltip)
        main_btn.setStyleSheet(f"""
            QPushButton {{
                background: {accent};
                color: #ffffff;
                border: none;
                border-radius: 10px;
                font-family: '{UI_FAMILY}';
                font-size: 15px;
                font-weight: 600;
                padding: 0 28px;
                letter-spacing: 0.3px;
            }}
            QPushButton:hover  {{ background: {_darken(accent, 12)}; }}
            QPushButton:pressed {{ background: {_darken(accent, 26)}; }}
        """)
        main_btn.clicked.connect(btn_action)
        lay.addWidget(main_btn)

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
# Lowercase variants for fast extension checking in hot path
_AUDIO_EXTS_LOWER: frozenset[str] = frozenset(
    ext.lower() for ext in _AUDIO_EXTS
)


def _is_audio_file(filename: str) -> bool:
    """Fast check if filename has an audio extension (case-insensitive).

    Optimized for the hot path: checks the filename suffix directly
    without calling os.path.splitext().
    """
    # Find the last dot; if none or at position 0, not a file with extension
    dot_pos = filename.rfind('.')
    if dot_pos <= 0:
        return False
    # Extract extension (includes the dot) and check case-insensitively
    ext = filename[dot_pos:].lower()
    return ext in _AUDIO_EXTS_LOWER


class _ArtHistory:
    """FIFO log of directory paths recently used for the landing mosaic.

    Persisted as a JSON array at ``~/.soundvault_art_history.json``.  Capped
    at ``MAX_ENTRIES`` paths; when full the oldest entries are evicted so the
    scanner keeps cycling through fresh parts of the library indefinitely.

    The scanner passes this object to ``_ordered_dirs()`` which puts unseen
    directories first (shuffled) and recently-seen directories last (also
    shuffled).  After emitting, the directories that contributed to the on-
    screen tiles are appended here and marked dirty. Saves are deferred to
    avoid blocking I/O on the scanner thread.
    """

    MAX_ENTRIES = 10_000
    _PATH = os.path.expanduser("~/.soundvault_art_history.json")

    def __init__(self) -> None:
        self._log: list[str] = []   # ordered oldest → newest
        self._set: set[str] = set() # fast membership test
        self._dirty: bool = False   # whether changes need to be persisted
        self._load()

    def _load(self) -> None:
        try:
            with open(self._PATH, encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                self._log = [p for p in raw if isinstance(p, str)]
                self._set = set(self._log)
        except FileNotFoundError:
            # First startup; no history file yet. This is normal.
            pass
        except Exception as e:
            # Unexpected error reading history; log but continue
            print(f"[Warning] Failed to load art history: {e}", file=sys.stderr)

    def add_and_save(self, paths: list[str]) -> None:
        """Append *paths* (deduped), trim to MAX_ENTRIES, and mark dirty.

        Actual file I/O is deferred via the _dirty flag. Call save_if_dirty()
        when convenient (e.g., on next idle cycle) to persist changes.
        """
        for p in paths:
            if p not in self._set:
                self._log.append(p)
                self._set.add(p)
        if len(self._log) > self.MAX_ENTRIES:
            evicted    = self._log[: len(self._log) - self.MAX_ENTRIES]
            self._log  = self._log[-self.MAX_ENTRIES :]
            self._set -= set(evicted)
        # Mark changes for deferred persistence; don't block on I/O
        self._dirty = True

    def save_if_dirty(self) -> None:
        """Persist changes to disk if any have been made since last save."""
        if not self._dirty:
            return
        try:
            # Write to temp file first, then rename (atomic on most filesystems)
            temp_path = self._PATH + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as fh:
                json.dump(self._log, fh, separators=(",", ":"))
            os.replace(temp_path, self._PATH)
            self._dirty = False
        except Exception as e:
            print(f"[Warning] Failed to save art history: {e}", file=sys.stderr)
            # Don't clear _dirty; retry on next opportunity

    def __contains__(self, path: str) -> bool:
        return path in self._set

    def __len__(self) -> int:
        return len(self._log)


# ─────────────────────────────────────────────────────────────────────────────
# _ArtScanner — background QThread that feeds album art to the mosaic tiles
# ─────────────────────────────────────────────────────────────────────────────

class _ArtScanner(QtCore.QThread):
    """Scan *library_path* for album art and emit one image per tile.

    Cover extraction
    ----------------
    ``_first_cover_in_dir()`` checks each directory in two stages:

    1. **Sidecar images** (fast path) — looks for ``cover.jpg``, ``folder.jpg``,
       etc. before touching any audio file.  Zero mutagen overhead for
       well-organised libraries.
    2. **Embedded art** — delegates to ``utils.audio_metadata_reader.read_metadata``
       (the canonical extractor for the whole app) for up to ``_MAX_FILES_PER_DIR``
       audio files.  Handles FLAC, MP3, OGG Vorbis, OGG Opus, M4A/AAC, WAV,
       AIFF, and WMA; applies ``ensure_long_path`` for Windows paths > 260 chars.

    Directory ordering — fresh-first with history
    ---------------------------------------------
    ``_ordered_dirs()`` collects all directories up to ``_SCAN_DEPTH`` levels
    (including the library root itself for flat libraries) and splits them into
    two independently-shuffled buckets:

    * **Fresh** — not in ``_ArtHistory`` (never shown, or log was reset).
    * **Used**  — appeared in a previous run's selection.

    Fresh directories come first.  Within each directory the file list is also
    shuffled, so every launch reads a different track from each folder.

    After emitting, the source directories are saved to ``_ArtHistory`` so the
    next run explores different folders first.
    """

    art_found = Signal(int, QtGui.QImage, str)  # (tile_index, image, dirpath) — QImage is thread-safe

    _MAX_SCAN_WORKERS   = 6  # parallel cover-extraction threads
    _IN_FLIGHT          = _MAX_SCAN_WORKERS * 3  # futures kept active at once

    def __init__(self, library_path: str, tile_count: int) -> None:
        super().__init__()
        self._library = library_path
        self._n       = tile_count
        self.setObjectName("ArtScanner")

    # ── Low-level helpers ─────────────────────────────────────────────────

    @staticmethod
    def _bake_image(raw: bytes, size: int = _TILE_SZ, radius: int = 10) -> QtGui.QImage | None:
        """Decode, scale, round-clip, and vignette *raw* cover bytes into a QImage.

        Called from pool worker threads.  QPainter on QImage is reentrant and
        safe to use from any thread; QPixmap is not (GUI-thread only).
        Returning a QImage lets the main thread do a single fast
        QPixmap.fromImage() with no further drawing work.
        """
        src = QtGui.QImage()
        if not src.loadFromData(raw) or src.isNull():
            return None
        src = src.scaled(
            QtCore.QSize(size, size),
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        out = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(QtGui.QColor(0, 0, 0, 0))
        r    = QtCore.QRectF(0, 0, size, size)
        path = QtGui.QPainterPath()
        path.addRoundedRect(r, radius, radius)
        p = QtGui.QPainter(out)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setClipPath(path)
            ox = (src.width()  - size) // 2
            oy = (src.height() - size) // 2
            p.drawImage(QtCore.QPoint(-ox, -oy), src)
            p.setClipping(False)
            vignette = QtGui.QRadialGradient(r.center(), max(r.width(), r.height()) * 0.75)
            vignette.setColorAt(0.5, QtGui.QColor(0, 0, 0, 0))
            vignette.setColorAt(1.0, QtGui.QColor(0, 0, 0, 80))
            p.fillPath(path, QtGui.QBrush(vignette))
        finally:
            p.end()
        return out

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
    def _cover_from_file(path: str) -> tuple[str, bytes] | None:
        """Return ``(method, image_bytes)`` for the first embedded cover found,
        or ``None`` if the file has no cover art.

        Delegates to ``utils.audio_metadata_reader.read_metadata`` which is the
        canonical cover extractor for the entire application.  It handles FLAC,
        MP3, OGG Vorbis, OGG Opus, M4A/AAC, WAV, AIFF, and WMA; applies
        ``ensure_long_path`` for Windows paths > 260 chars; and falls back to
        per-track sidecar files (``<track>.jpg``, ``<track>.artwork``, etc.).
        """
        try:
            from utils.audio_metadata_reader import read_metadata
            _tags, covers, _err, _hint = read_metadata(path, include_cover=True)
            if covers:
                return ("read_metadata", covers[0])
        except Exception:
            pass
        return None

    # ── Directory ordering ────────────────────────────────────────────────

    def _ordered_dirs(self, history: _ArtHistory) -> list[str]:
        """All library dirs up to _SCAN_DEPTH, fresh dirs first.

        One lightweight walk — no file content reads.  Each bucket is
        independently shuffled so traversal order differs on every launch.
        Depth computed incrementally during traversal to avoid O(n) path operations.
        """
        fresh: list[str] = []
        used:  list[str] = []

        def _walk(dirpath: str, depth: int) -> None:
            """Recursively walk directories, tracking depth incrementally."""
            if depth >= _SCAN_DEPTH:
                return

            try:
                entries = os.scandir(dirpath)
                subdirs = [e.name for e in entries if e.is_dir(follow_symlinks=False)]
            except OSError:
                return

            random.shuffle(subdirs)
            for name in subdirs:
                subdirpath = os.path.join(dirpath, name)
                bucket = used if subdirpath in history else fresh
                bucket.append(subdirpath)
                _walk(subdirpath, depth + 1)

        # Include the library root itself so flat libraries (all files in one
        # folder) are not silently skipped.
        root_bucket = used if self._library in history else fresh
        root_bucket.append(self._library)

        _walk(self._library, 0)
        random.shuffle(fresh)
        random.shuffle(used)
        return fresh + used

    # ── Per-directory cover helper ────────────────────────────────────────

    def _covers_in_dir(
        self, dirpath: str
    ) -> list[tuple[str, str, bytes]]:
        """Return every unique cover found in *dirpath* as a list of
        ``(method, dirpath, raw_bytes)`` tuples.  Returns ``[]`` if none found.

        Returning a list (rather than stopping at the first hit) is what allows
        flat libraries — where every song lives in one folder — to populate
        multiple tiles with different album covers.  For a normal album folder
        the list will contain exactly one entry (the shared cover art); for a
        flat folder it will contain one entry per uniquely-covered track.

        Deduplication within the directory is done here via a local md5 set so
        that tracks sharing the same embedded art (the typical album case) only
        contribute one result.  Cross-directory deduplication is handled by
        ``seen_sigs`` in ``run()``.

        Workers do I/O only — no Qt calls.  Baking happens in the scanner
        QThread (``run()``) so only one background thread ever touches Qt APIs.
        """
        results:     list[tuple[str, str, bytes]] = []
        local_sigs:  set[bytes]                   = set()

        _COVER_NAMES = (
            "cover.jpg", "folder.jpg", "album.jpg", "artwork.jpg",
            "front.jpg", "cover.png", "folder.png", "album.png",
        )

        # ── 1. Sidecar image ───────────────────────────────────────────────
        for img_name in _COVER_NAMES:
            img_path = os.path.join(dirpath, img_name)
            try:
                if os.path.isfile(img_path):
                    with open(img_path, "rb") as fh:
                        raw = fh.read()
                    if raw:
                        sig = hashlib.md5(raw).digest()
                        local_sigs.add(sig)
                        results.append(("folder_image", dirpath, raw))
                    break  # at most one sidecar per directory
            except OSError:
                pass

        # ── 2. Embedded art — every audio file in the directory ────────────
        for name in self._scandir_files(dirpath):
            if not _is_audio_file(name):
                continue
            result = self._cover_from_file(os.path.join(dirpath, name))
            if result:
                method, raw = result
                sig = hashlib.md5(raw).digest()
                if sig not in local_sigs:
                    local_sigs.add(sig)
                    results.append((method, dirpath, raw))

        return results

    # ── Orchestration + emit ──────────────────────────────────────────────

    def run(self) -> None:  # noqa: N802
        t0 = time.monotonic()

        history  = _ArtHistory()
        dirpaths = self._ordered_dirs(history)

        if self.isInterruptionRequested():
            return

        selected:       list[str]              = []
        seen_sigs:      set[tuple[int, ...]]   = set()   # visual fingerprints — catches same cover re-encoded
        found_images:   list[QtGui.QImage]    = []      # baked images for fill-cycle fallback
        found_dirpaths: list[str]             = []      # dirpath paired with each found_image entry
        tile_index:     int                   = 0
        method_counts:  dict[str, int]        = {}
        no_cover_dirs:  int                   = 0

        # Rolling pool: keep _IN_FLIGHT futures active at all times.
        # As each completes we immediately submit the next directory.
        # When we have enough tiles the executor is shut down and any
        # queued-but-not-started futures are cancelled (Python 3.9+).
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._MAX_SCAN_WORKERS,
            thread_name_prefix="ArtScan",
        )
        pending:  dict[concurrent.futures.Future, None] = {}
        dir_iter  = iter(dirpaths)

        def _fill() -> None:
            while len(pending) < self._IN_FLIGHT:
                try:
                    d = next(dir_iter)
                    pending[executor.submit(self._covers_in_dir, d)] = None
                except StopIteration:
                    break

        try:
            _fill()
            while pending and tile_index < self._n and not self.isInterruptionRequested():
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                    timeout=1.0,
                )
                for f in done:
                    del pending[f]
                    try:
                        hits = f.result()
                    except Exception:
                        hits = []
                    if not hits:
                        no_cover_dirs += 1
                    else:
                        for method, dirpath, raw in hits:
                            if tile_index >= self._n:
                                break
                            # Bake first — normalises to 110×110 ARGB32 regardless
                            # of source encoding.  The visual sig is then derived
                            # from the baked pixels, so the same cover re-saved at
                            # a different JPEG quality or by a different tool still
                            # produces the same fingerprint and gets deduplicated.
                            img = self._bake_image(raw)
                            if img is None:
                                continue
                            # 9-point visual fingerprint: sample a 3×3 grid at
                            # ~30 / 50 / 70 % of the tile width (inset from edges
                            # to avoid crop differences between different source
                            # aspect ratios).  Mask the 6 low bits of each channel
                            # (tolerance ±63) so interpolation differences caused
                            # by the same image existing at different source
                            # resolutions collapse to the same value.
                            _S = (33, 55, 77)
                            sig = tuple(
                                img.pixel(x, y) & 0xC0C0C0C0
                                for x in _S for y in _S
                            )
                            if sig not in seen_sigs:
                                seen_sigs.add(sig)
                                self.art_found.emit(tile_index, img, dirpath)
                                found_images.append(img)
                                found_dirpaths.append(dirpath)
                                selected.append(dirpath)
                                method_counts[method] = method_counts.get(method, 0) + 1
                                tile_index += 1
                _fill()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        # ── Fill-cycle: fewer unique covers than tiles ─────────────────────
        # If the library has fewer unique albums than tile slots, cycle through
        # the covers we did find (shuffled so repeats are distributed randomly)
        # rather than leaving tiles as colour placeholders.
        fill_count = 0
        if tile_index < self._n and found_images and not self.isInterruptionRequested():
            fill_pairs = list(zip(found_images, found_dirpaths))
            random.shuffle(fill_pairs)
            pool_len = len(fill_pairs)
            while tile_index < self._n and not self.isInterruptionRequested():
                fill_img, fill_dir = fill_pairs[fill_count % pool_len]
                self.art_found.emit(tile_index, fill_img, fill_dir)
                tile_index += 1
                fill_count += 1

        if _ART_SCAN_DEBUG:
            unique = tile_index - fill_count
            parts = [f"tiles={tile_index}/{self._n}"]
            if fill_count:
                parts.append(f"unique={unique}, fill={fill_count}")
            parts += [f"{m}={c}" for m, c in sorted(method_counts.items())]
            if no_cover_dirs:
                parts.append(f"no-cover={no_cover_dirs}")
            parts.append(f"dirs={len(dirpaths)}")
            parts.append(f"elapsed={time.monotonic() - t0:.2f}s")
            print(f"[ArtScan] {', '.join(parts)}", file=sys.stderr, flush=True)

        # Mark history as dirty (deferred write), then persist it
        history.add_and_save(selected)
        # Persist the history file now that scanning is complete
        history.save_if_dirty()


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
    tile_clicked     = Signal(str)   # emitted when user clicks a tile; payload is dirpath

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
        self._art_history: _ArtHistory | None = None  # Track for deferred save

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

        for tile in self._tiles:
            tile.clicked.connect(self._on_tile_clicked)

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
        """Fade the window in, pause on the logo, then start the tile fly-in.

        Sequence
        --------
        1. Window fades in over ``_FADE_IN_MS`` — only the CTA card ("AlphaDEX")
           is visible; all tiles are still off-screen.  This is the logo moment.
        2. ``_LOGO_PAUSE_MS`` pause — user reads the brand name.
        3. Tile fly-in begins (``_fly_in``).
        """
        self.setWindowOpacity(0.0)
        self.show()
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(_FADE_IN_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(lambda v: self.setWindowOpacity(float(v)))
        anim.finished.connect(
            lambda: QtCore.QTimer.singleShot(_LOGO_PAUSE_MS, self._fly_in)
        )
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
            # Disconnect signals before cleanup to prevent race conditions
            self._scanner.art_found.disconnect()
            self._scanner.requestInterruption()
            if not self._scanner.wait(2000):
                # Thread did not exit cleanly; force termination
                self._scanner.terminate()
                self._scanner.wait()
            self._scanner = None
        self.hide()
        self.finished.emit()

    # ── Tile click handler ────────────────────────────────────────────────

    def _on_tile_clicked(self, dirpath: str) -> None:
        """User clicked an album-art tile — begin the exit sequence and signal
        the caller to navigate to the player and start playing *dirpath*."""
        if not dirpath:
            return
        self.tile_clicked.emit(dirpath)
        # Reuse the same exit flow as clicking Initialize.
        # If no saved library is set we have no library to open, so guard.
        if self._saved:
            self._accept(self._saved)

    # ── Art scanner ───────────────────────────────────────────────────────

    def _start_art_scanner(self) -> None:
        scanner = _ArtScanner(self._saved, len(self._tiles))
        scanner.art_found.connect(self._on_art_found)
        self._scanner = scanner
        scanner.start()

    def _on_art_found(self, tile_index: int, image: QtGui.QImage, dirpath: str) -> None:
        # We are back on the main thread here; QPixmap conversion is safe.
        if 0 <= tile_index < len(self._tiles):
            pm = QtGui.QPixmap.fromImage(image)
            if not pm.isNull():
                tile = self._tiles[tile_index]
                tile.set_pixmap(pm)
                tile.set_dirpath(dirpath)


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
