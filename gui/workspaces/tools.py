"""Tools workspace — export utilities, diagnostics, and debug tools."""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

_SUPPORTED_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour + alpha float to a CSS rgba() string.

    Qt stylesheets do not reliably support 8-digit hex (#rrggbbaa), so
    this helper is used everywhere an alpha-tinted colour is needed.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


# ── Background workers ─────────────────────────────────────────────────────────

class FileCleanupWorker(QtCore.QThread):
    log_line = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, library_path: str) -> None:
        super().__init__()
        self.library_path = library_path

    def run(self) -> None:
        numeric_suffix = re.compile(r"\s*\(\d+\)$")
        copy_suffix = re.compile(r"\s*(?:-\s*)?copy(?:\s*\(\d+\))?$", re.IGNORECASE)
        rename_map: dict[str, str] = {}
        renamed = skipped = conflicts = errors = 0

        for root, _dirs, files in os.walk(self.library_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in _SUPPORTED_EXTS:
                    continue
                stem = os.path.splitext(filename)[0]
                new_stem = copy_suffix.sub("", stem)
                new_stem = numeric_suffix.sub("", new_stem)
                if new_stem == stem:
                    skipped += 1
                    continue
                new_name = f"{new_stem}{ext}"
                src = os.path.join(root, filename)
                dst = os.path.join(root, new_name)
                if os.path.exists(dst):
                    self.log_line.emit(f"! Conflict: {filename} \u2192 {new_name} already exists")
                    conflicts += 1
                    continue
                try:
                    os.rename(src, dst)
                    self.log_line.emit(f"\u2192 {filename} \u2192 {new_name}")
                    rename_map[src] = dst
                    renamed += 1
                except OSError as exc:
                    self.log_line.emit(f"! Error renaming {filename}: {exc}")
                    errors += 1

        if rename_map:
            try:
                from playlist_generator import update_playlists
                update_playlists(rename_map)
                self.log_line.emit("\u2713 Updated playlists")
            except Exception as exc:  # noqa: BLE001
                self.log_line.emit(f"! Playlist update failed: {exc}")

        self.finished.emit(
            True,
            f"Done: {renamed} renamed, {skipped} unchanged, {conflicts} conflicts, {errors} errors."
        )


class ArtistTitleWorker(QtCore.QThread):
    """Background thread for the Artist/Title export."""

    progress = Signal(int, int)       # (done, total)
    log_line = Signal(str)
    finished = Signal(str, int, int)  # (output_path, entry_count, error_count)
    error = Signal(str)

    def __init__(
        self,
        library_path: str,
        exclude_flac: bool,
        add_album_duplicates: bool,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.library_path = library_path
        self.exclude_flac = exclude_flac
        self.add_album_duplicates = add_album_duplicates

    @staticmethod
    def _clean(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    return value.decode(enc).strip() or None
                except UnicodeDecodeError:
                    continue
            return value.decode("utf-8", errors="replace").strip() or None
        if isinstance(value, (list, tuple)):
            for item in value:
                cleaned = ArtistTitleWorker._clean(item)
                if cleaned:
                    return cleaned
            return None
        cleaned = str(value).strip()
        return cleaned or None

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def _run(self) -> None:
        from utils.audio_metadata_reader import read_tags
        from utils.opus_metadata_reader import read_opus_metadata

        audio_files: list[str] = []
        for dirpath, _, files in os.walk(self.library_path):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if self.exclude_flac and ext == ".flac":
                    continue
                if ext in _SUPPORTED_EXTS:
                    audio_files.append(os.path.join(dirpath, filename))

        total = len(audio_files)
        self.log_line.emit(f"Found {total} audio files to process.")

        entry_data: list[tuple[str, str, str | None, str | None]] = []
        error_count = 0

        for idx, full_path in enumerate(audio_files, start=1):
            ext = os.path.splitext(full_path)[1].lower()
            filename = os.path.basename(full_path)

            if ext == ".opus":
                tags, _covers, read_error = read_opus_metadata(full_path)
                if read_error:
                    error_count += 1
                    self.log_line.emit(f"Skipped unreadable OPUS file: {filename}")
            else:
                tags = read_tags(full_path)

            artist = self._clean(tags.get("artist") or tags.get("albumartist"))
            title  = self._clean(tags.get("title"))
            album  = self._clean(tags.get("album"))
            track_raw = self._clean(tags.get("tracknumber") or tags.get("track"))
            if track_raw and "/" in track_raw:
                track_raw = track_raw.split("/", 1)[0].strip() or None

            if not title:
                title = os.path.splitext(filename)[0]
            if not artist:
                artist = "Unknown Artist"

            entry_data.append((artist, title, album, track_raw))

            if idx == 1 or idx % 50 == 0 or idx == total:
                self.progress.emit(idx, total)

        entries: list[str] = []
        duplicate_counts = Counter(
            (artist, title) for artist, title, _album, _track in entry_data
        )
        album_dup_counts = Counter(
            (artist, title, album) for artist, title, album, _track in entry_data
        )

        for artist, title, album, track in entry_data:
            if self.add_album_duplicates and duplicate_counts[(artist, title)] > 1:
                album_label = album or "Unknown Album"
                if album_dup_counts[(artist, title, album)] > 1:
                    track_label = track or "Unknown Track"
                    entries.append(f"{artist} - {title} - {album_label} - {track_label}")
                else:
                    entries.append(f"{artist} - {title} - {album_label}")
            else:
                entries.append(f"{artist} - {title}")

        entries = sorted(set(entries), key=str.lower)

        out = Path(self.library_path) / "Docs" / "artist_title_list.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(entries), encoding="utf-8")

        self.finished.emit(str(out), len(entries), error_count)


class _GlassBadge(QtWidgets.QWidget):
    def __init__(self, icon: str, size: int = 40, parent=None):
        super().__init__(parent)
        self._icon = icon
        self.setFixedSize(size, size)
        from gui.themes.manager import get_manager
        get_manager().theme_changed.connect(lambda _: self.update())

    def set_size(self, size: int) -> None:
        self.setFixedSize(size, size)
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        sz  = float(self.width())
        r   = QtCore.QRectF(0, 0, sz, sz)
        rad = sz * 0.28
        ac  = QtGui.QColor(t.accent)

        fill = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 55)
        p.setBrush(fill); p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, rad, rad)

        spec = QtGui.QLinearGradient(0, 0, 0, sz * 0.55)
        s1 = QtGui.QColor(255, 255, 255); s1.setAlphaF(0.38)
        s2 = QtGui.QColor(255, 255, 255); s2.setAlphaF(0.0)
        spec.setColorAt(0.0, s1); spec.setColorAt(1.0, s2)
        p.setBrush(spec)
        p.drawRoundedRect(r, rad, rad)

        tint = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 70)
        p.setBrush(tint)
        p.drawRoundedRect(r, rad, rad)

        rim = QtGui.QLinearGradient(0, 0, 0, sz)
        r1 = QtGui.QColor(255, 255, 255); r1.setAlphaF(0.65)
        r2 = QtGui.QColor(ac.red(), ac.green(), ac.blue()); r2.setAlphaF(0.35)
        rim.setColorAt(0.0, r1); rim.setColorAt(1.0, r2)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QBrush(rim), 1.1))
        p.drawRoundedRect(r.adjusted(0.55, 0.55, -0.55, -0.55), rad, rad)

        p.setPen(QtGui.QColor(t.text_inverse))
        font = QtGui.QFont(self.font().family(), int(sz * 0.42))
        font.setBold(True)
        p.setFont(font)
        p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._icon)
        p.end()


class _GlassResultChip(QtWidgets.QWidget):
    NEUTRAL = 0
    SUCCESS = 1
    ERROR   = 2

    _COLOR = {
        NEUTRAL: "#94a3b8",   
        SUCCESS: "#22c55e",   
        ERROR:   "#ef4444",   
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text  = "\u29d6  Idle"
        self._state = self.NEUTRAL
        self._opacity = 1.0
        self._anim: QtCore.QVariantAnimation | None = None
        self.setFixedHeight(26)
        self._resize()
        from gui.themes.manager import get_manager
        get_manager().theme_changed.connect(lambda _: self.update())

    def _resize(self):
        fm = QtGui.QFontMetrics(self.font())
        self.setFixedWidth(fm.horizontalAdvance(self._text) + 34)

    def set_text_fast(self, text: str, state: int | None = None):
        self._text = text
        if state is not None:
            self._state = state
        self._resize()
        self.update()

    def show_neutral(self, text: str = "\u29d6  Idle"):
        self._transition(text, self.NEUTRAL)

    def show_success(self, text: str = "\u2713  Done"):
        self._transition(text, self.SUCCESS)

    def show_error(self, text: str = "\u2717  Error"):
        self._transition(text, self.ERROR)

    def _transition(self, text: str, state: int):
        if self._anim:
            self._anim.stop()
        a = QtCore.QVariantAnimation(self)
        a.setStartValue(self._opacity)
        a.setEndValue(0.0)
        a.setDuration(160)
        a.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        a.valueChanged.connect(lambda v: self._set_op(float(v)))

        def _swap():
            self._text  = text
            self._state = state
            self._resize()
            b = QtCore.QVariantAnimation(self)
            b.setStartValue(0.0)
            b.setEndValue(1.0)
            b.setDuration(260)
            b.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            b.valueChanged.connect(lambda v: self._set_op(float(v)))
            b.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
            self._anim = b

        a.finished.connect(_swap)
        a.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        self._anim = a

    def _set_op(self, v: float):
        self._opacity = v
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setOpacity(self._opacity)

        w, h = float(self.width()), float(self.height())
        r    = QtCore.QRectF(0, 0, w, h)
        rad  = h / 2.0
        ac   = QtGui.QColor(self._COLOR[self._state])

        fill = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 38)
        p.setBrush(fill); p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, rad, rad)

        spec = QtGui.QLinearGradient(0, 0, 0, h * 0.58)
        s1 = QtGui.QColor(255, 255, 255); s1.setAlphaF(0.28)
        s2 = QtGui.QColor(255, 255, 255); s2.setAlphaF(0.0)
        spec.setColorAt(0.0, s1); spec.setColorAt(1.0, s2)
        p.setBrush(spec)
        p.drawRoundedRect(r, rad, rad)

        bdr = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 170)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(bdr, 1.0))
        p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), rad, rad)

        p.setPen(ac)
        f = p.font(); f.setPointSize(10); f.setBold(True); p.setFont(f)
        p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()


# ── ToolTile ───────────────────────────────────────────────────────────────────

class ToolTile(QtWidgets.QFrame):
    """Self-contained tool card: header ▸ options ▸ footer ▸ animated drawer.

    Usage
    -----
    tile = ToolTile("↗", "My Tool", "Short description.")
    tile.add_option(some_checkbox)
    tile.set_run_button("Run", self._my_slot)
    secondary = tile.add_secondary_button("Open File")
    tile.finish_footer()

    When an operation starts call tile.open_drawer() to reveal the progress
    area and tile.set_running(True/False) to lock/unlock footer buttons.
    """

    def __init__(
        self,
        icon: str,
        title: str,
        description: str,
        *,
        log_height: int = 110,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("liquidGlassTile")
        self.setStyleSheet("QFrame#liquidGlassTile { background: transparent; border: none; }")
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self._hover_t = 0.0
        self._time = 0.0
        self._target_pos = QtCore.QPointF(200, 100)
        self._glow_pos = QtCore.QPointF(200, 100)
        self._is_hovered = False

        self._footer_buttons: list[QtWidgets.QPushButton] = []
        self._drawer_open = False

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        header = QtWidgets.QWidget()
        header_l = QtWidgets.QVBoxLayout(header)
        header_l.setContentsMargins(18, 16, 18, 12)
        header_l.setSpacing(4)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(12)
        
        self._badge = _GlassBadge(icon, size=40)
        title_row.addWidget(self._badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        
        info = QtWidgets.QVBoxLayout()
        info.setSpacing(6)
        
        self._title_lbl = QtWidgets.QLabel(title)
        self._title_lbl.setObjectName("cardTitle")
        info.addWidget(self._title_lbl)

        self._desc_lbl: QtWidgets.QLabel | None = None
        if description:
            self._desc_lbl = QtWidgets.QLabel(description)
            self._desc_lbl.setObjectName("sectionSubtitle")
            self._desc_lbl.setWordWrap(True)
            self._desc_lbl.setContentsMargins(0, 2, 0, 2)
            info.addWidget(self._desc_lbl)
            
        title_row.addLayout(info, 1)

        self.status_chip = _GlassResultChip()
        self.status_chip.show_neutral("Ready.")
        title_row.addWidget(self.status_chip, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        header_l.addLayout(title_row)
        outer.addWidget(header)

        # ── Options area ──────────────────────────────────────────────────
        self._options = QtWidgets.QWidget()
        self._opts_l = QtWidgets.QVBoxLayout(self._options)
        self._opts_l.setContentsMargins(18, 0, 18, 12)
        self._opts_l.setSpacing(8)
        outer.addWidget(self._options)

        # ── Footer ────────────────────────────────────────────────────────
        self._footer = QtWidgets.QWidget()
        self._footer_l = QtWidgets.QHBoxLayout(self._footer)
        self._footer_l.setContentsMargins(18, 8, 18, 16)
        self._footer_l.setSpacing(8)
        outer.addWidget(self._footer)

        # ── Separator (hidden until drawer opens) ─────────────────────────
        self._sep = QtWidgets.QFrame()
        self._sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self._sep.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self._sep.setVisible(False)
        outer.addWidget(self._sep)

        # ── Drawer ────────────────────────────────────────────────────────
        self._drawer = QtWidgets.QWidget()
        drawer_l = QtWidgets.QVBoxLayout(self._drawer)
        drawer_l.setContentsMargins(18, 10, 18, 16)
        drawer_l.setSpacing(6)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)

        # Hidden but retained so existing tool logic doesn't crash on .setText()
        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setVisible(False)

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(log_height)
        self.log_box.setObjectName("logBox")
        self.log_box.setStyleSheet("QPlainTextEdit { background: rgba(0, 0, 0, 0.1); border-radius: 6px; }")

        drawer_l.addWidget(self.progress_bar)
        drawer_l.addWidget(self.status_label)
        drawer_l.addWidget(self.log_box)

        self._drawer.setMinimumHeight(0)
        self._drawer.setMaximumHeight(0)
        outer.addWidget(self._drawer)

        # ── Theme ─────────────────────────────────────────────────────────
        self._apply_sep_color(get_manager().current)
        get_manager().theme_changed.connect(self._on_theme_changed)

    def mouseMoveEvent(self, e):
        self._target_pos = e.position()
        super().mouseMoveEvent(e)

    def enterEvent(self, e):
        self._is_hovered = True
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._is_hovered = False
        super().leaveEvent(e)

    def _tick(self):
        target_h = 1.0 if self._is_hovered else 0.0
        hover_delta = (target_h - self._hover_t) * 0.15
        self._hover_t += hover_delta

        if not self._is_hovered:
            self._target_pos = QtCore.QPointF(self.width() / 2, self.height() / 2)

        dx = self._target_pos.x() - self._glow_pos.x()
        dy = self._target_pos.y() - self._glow_pos.y()
        self._glow_pos.setX(self._glow_pos.x() + dx * 0.15)
        self._glow_pos.setY(self._glow_pos.y() + dy * 0.15)

        # Skip repaint when nothing is animating
        if abs(hover_delta) < 0.001 and abs(dx) < 0.5 and abs(dy) < 0.5:
            return
        self.update()

    def paintEvent(self, e):  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        rect = QtCore.QRectF(self.rect())
        inner = rect.adjusted(2, 2, -2, -2)
        radius = 16.0
        ht = self._hover_t
        ac = QtGui.QColor(t.accent)

        path = QtGui.QPainterPath()
        path.addRoundedRect(inner, radius, radius)
        p.save()
        p.setClipPath(path)

        p.setPen(QtCore.Qt.PenStyle.NoPen)
        bg_alpha = 0.05 + ht * 0.03
        p.setBrush(QtGui.QColor(255, 255, 255, int(bg_alpha * 255)))
        p.drawRect(inner)

        t_alpha = 0.1 + ht * 0.05
        is_dark = getattr(t, "is_dark", True)
        if is_dark:
            tint_color = QtGui.QColor(0, 0, 0, int(t_alpha * 255))
        else:
            tint_color = QtGui.QColor(255, 255, 255, int(t_alpha * 255))
        p.setBrush(tint_color)
        p.drawRect(inner)

        glow_rad = inner.width() * 0.8
        glow = QtGui.QRadialGradient(self._glow_pos, glow_rad)
        c1 = QtGui.QColor(ac); c1.setAlphaF(0.15 + ht * 0.15)
        c2 = QtGui.QColor(ac); c2.setAlphaF(0.0)
        glow.setColorAt(0.0, c1)
        glow.setColorAt(1.0, c2)
        p.setBrush(glow)
        p.drawRect(inner)

        spec = QtGui.QLinearGradient(0, inner.top(), 0, inner.bottom())
        spec.setColorAt(0.0, QtGui.QColor(255, 255, 255, int((0.15 + ht * 0.05) * 255)))
        spec.setColorAt(0.4, QtGui.QColor(255, 255, 255, 0))
        p.setBrush(spec)
        p.drawRect(inner)
        p.restore()
        
        p.end()

    # ── Public API ────────────────────────────────────────────────────────

    def add_option(self, widget: QtWidgets.QWidget) -> None:
        """Add a widget to the options section."""
        self._opts_l.addWidget(widget)

    def add_option_layout(self, layout: QtWidgets.QLayout) -> None:
        """Add a sub-layout to the options section."""
        self._opts_l.addLayout(layout)

    def set_run_button(self, label: str, slot) -> QtWidgets.QPushButton:
        """Add the primary (accent-coloured) action button and return it."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(label)
        btn.setObjectName("primaryBtn")
        btn.setMinimumHeight(34)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.clicked.connect(slot)
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def add_secondary_button(self, label: str) -> QtWidgets.QPushButton:
        """Add a secondary (default-style) button and return it."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(label)
        btn.setMinimumHeight(34)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def add_icon_button(self, icon: str, tooltip: str = "") -> QtWidgets.QPushButton:
        """Add a small square icon-only button and return it."""
        from gui.themes.animations import AnimatedButton
        btn = AnimatedButton(icon)
        btn.setFixedSize(34, 34)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        if tooltip:
            btn.setToolTip(tooltip)
        self._footer_l.addWidget(btn)
        self._footer_buttons.append(btn)
        return btn

    def finish_footer(self) -> None:
        """Add trailing stretch after all buttons. Call last."""
        self._footer_l.addStretch(1)

    def hide_footer(self) -> None:
        """Hide the footer row entirely (e.g. for tiles with no run action)."""
        self._footer.setVisible(False)

    def open_drawer(self) -> None:
        """Animate the log drawer from 0 → its natural height (once)."""
        if self._drawer_open:
            return
        self._drawer_open = True
        self._sep.setVisible(True)
        target = max(self._drawer.sizeHint().height(), 100)
        anim = QtCore.QPropertyAnimation(self._drawer, b"maximumHeight", self)
        anim.setDuration(250)
        anim.setStartValue(0)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: self._drawer.setMaximumHeight(16777215))
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def set_running(self, running: bool) -> None:
        """Enable or disable all footer buttons (call during long operations)."""
        for btn in self._footer_buttons:
            btn.setEnabled(not running)

    def flash_status(self, from_hex: str, duration: int = 1600) -> None:
        """Animate status_label colour from from_hex → theme text_secondary."""
        t = get_manager().current
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(QtGui.QColor(from_hex))
        anim.setEndValue(QtGui.QColor(t.text_secondary))
        anim.setDuration(duration)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(
            lambda c: self.status_label.setStyleSheet(f"color: {c.name()};")
        )
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    # ── Text scale ────────────────────────────────────────────────────────

    _TITLE_PT = {1: 10, 2: 11, 3: 12, 4: 14, 5: 16}
    _DESC_PT  = {1:  8, 2:  9, 3: 10, 4: 11, 5: 13}
    _BADGE_SZ = {1: 30, 2: 34, 3: 40, 4: 46, 5: 52}

    def set_text_scale(self, level: int) -> None:
        """Adjust font sizes and badge icon to match the given 1–5 level."""
        if level not in self._TITLE_PT:
            return
        f = self._title_lbl.font()
        f.setPointSize(self._TITLE_PT[level])
        self._title_lbl.setFont(f)
        if self._desc_lbl is not None:
            f2 = self._desc_lbl.font()
            f2.setPointSize(self._DESC_PT[level])
            self._desc_lbl.setFont(f2)
        self._badge.set_size(self._BADGE_SZ[level])

    # ── Internal ──────────────────────────────────────────────────────────

    def _apply_sep_color(self, tokens) -> None:
        self._sep.setStyleSheet(f"QFrame {{ color: {tokens.card_border}; }}")

    def _on_theme_changed(self, tokens) -> None:
        self._apply_sep_color(tokens)



# ── TileGrid ───────────────────────────────────────────────────────────────────

class TileGrid(QtWidgets.QWidget):
    """Responsive grid container for ToolTile instances.

    Column count is computed from available width divided by _TILE_MIN_WIDTH.
    Full-width tiles span all columns and always start a fresh row.

    Adding a new tool tile:
        self._tile_grid.add_tile(self._build_new_tile(), full_width=False)
    """

    _TILE_MIN_WIDTH = 500   # px — 2 columns from ~1050 px wide, 1 column on narrow windows

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._tiles: list[tuple[ToolTile, bool]] = []   # (tile, is_full_width)
        self._current_cols: int = 2                      # default; updated on resize

        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setSpacing(16)
        self._grid.setContentsMargins(0, 0, 0, 0)

    def add_tile(self, tile: ToolTile, full_width: bool = False) -> None:
        """Register a tile and add it to the current grid layout."""
        self._tiles.append((tile, full_width))
        self._reflow()

    def set_size_level(self, level: int) -> None:
        for tile, _ in self._tiles:
            tile.set_text_scale(level)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        new_cols = max(1, event.size().width() // self._TILE_MIN_WIDTH)
        if new_cols != self._current_cols:
            self._current_cols = new_cols
            self._reflow()
        super().resizeEvent(event)

    def _reflow(self) -> None:
        """Re-place all tiles in the grid with the current column count."""
        cols = self._current_cols

        # Detach every tile from the layout without destroying it
        for tile, _ in self._tiles:
            self._grid.removeWidget(tile)

        col = row = 0
        for tile, full_width in self._tiles:
            if full_width:
                if col > 0:          # push to a new row first
                    row += 1
                    col = 0
                self._grid.addWidget(tile, row, 0, 1, cols)
                row += 1
            else:
                self._grid.addWidget(tile, row, col)
                col += 1
                if col >= cols:
                    col = 0
                    row += 1

        for c in range(cols):
            self._grid.setColumnStretch(c, 1)


# ── _PillToggle ────────────────────────────────────────────────────────────────

class _PillToggle(QtWidgets.QAbstractButton):
    """Animated iOS-style pill toggle switch.

    Drop-in replacement for QCheckBox for single on/off options.
    isChecked() / toggled signal / setChecked() all work identically.
    """

    _TRACK_W = 38
    _TRACK_H = 22
    _KNOB_D  = 16
    _GAP     = 10   # px between track right edge and label text

    def __init__(self, label: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(label)
        self._knob_x: float = 0.0          # 0.0 = off, 1.0 = on
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(lambda v: self._set_knob(float(v)))
        self.toggled.connect(self._start_anim)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        get_manager().theme_changed.connect(lambda _: self.update())

    # ── Animation ──────────────────────────────────────────────────────────

    def _start_anim(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._knob_x)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _set_knob(self, v: float) -> None:
        self._knob_x = v
        self.update()

    # ── Sizing ─────────────────────────────────────────────────────────────

    def sizeHint(self) -> QtCore.QSize:
        label = self.text()
        w = self._TRACK_W
        h = self._TRACK_H
        if label:
            fm = self.fontMetrics()
            w += self._GAP + fm.horizontalAdvance(label)
            h = max(h, fm.height() + 4)
        return QtCore.QSize(w, h)

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()

    # ── Paint ──────────────────────────────────────────────────────────────

    def paintEvent(self, e) -> None:  # noqa: N802
        t = get_manager().current
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        tw = float(self._TRACK_W)
        th = float(self._TRACK_H)
        kd = float(self._KNOB_D)
        cy = self.height() / 2.0

        # Track
        tr = QtCore.QRectF(0, cy - th / 2, tw, th)
        if self.isChecked():
            ac = QtGui.QColor(t.accent)
            track_col = QtGui.QColor(ac.red(), ac.green(), ac.blue(), 220)
        else:
            track_col = QtGui.QColor(t.card_border)
        p.setBrush(track_col)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(tr, th / 2, th / 2)

        # Knob (white circle)
        margin = (th - kd) / 2.0
        travel = tw - kd - 2 * margin
        kx = margin + self._knob_x * travel
        ky = cy - kd / 2.0
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawEllipse(QtCore.QRectF(kx, ky, kd, kd))

        # Label text
        label = self.text()
        if label:
            p.setPen(QtGui.QColor(t.text_primary))
            f = p.font()
            f.setPointSize(10)
            p.setFont(f)
            lx = int(tw) + self._GAP
            p.drawText(
                lx, 0, self.width() - lx, self.height(),
                int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft),
                label,
            )
        p.end()


# ── ToolsWorkspace ─────────────────────────────────────────────────────────────

class ToolsWorkspace(WorkspaceBase):
    """All export, diagnostic, and utility tools in one place."""

    def __init__(
        self, library_path: str = "", parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(library_path, parent)
        self._cleanup_worker: FileCleanupWorker | None = None
        self._at_worker: ArtistTitleWorker | None = None
        self._at_open_connected: bool = False
        self._at_folder_connected: bool = False
        self._codec_open_connected: bool = False
        self._codec_folder_connected: bool = False
        # Tile references (assigned by _build_*_tile helpers)
        self._at_tile: ToolTile | None = None
        self._cleanup_tile: ToolTile | None = None
        # Codec chip checkboxes for theme-refresh
        self._codec_chips: dict[str, QtWidgets.QCheckBox] = {}
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        cl = self.content_layout
        cl.addWidget(self._make_section_title("Export & Utilities"))
        cl.addWidget(self._make_subtitle(
            "Export reports, run diagnostics, and validate your library layout. "
            "All export files land in Docs/ inside your library folder."
        ))

        grid = TileGrid()
        grid.add_tile(self._build_at_tile())
        grid.add_tile(self._build_codec_tile())
        grid.add_tile(self._build_cleanup_tile())
        grid.add_tile(self._build_diag_tile(), full_width=True)
        grid.add_tile(self._build_validator_tile())
        cl.addWidget(grid, 1)
        cl.addStretch(1)

    # ── Tile builders ──────────────────────────────────────────────────────

    def _build_at_tile(self) -> ToolTile:
        tile = ToolTile(
            "\u2197", "Artist / Title Export",
            "Scan the library and write Docs/artist_title_list.txt "
            "with every Artist \u2013 Title pair.",
        )
        self._at_tile = tile

        self._exclude_flac_cb = _PillToggle("Exclude FLAC files")
        self._dupe_tracks_cb  = _PillToggle("Include per-album duplicate titles")
        tile.add_option(self._exclude_flac_cb)
        tile.add_option(self._dupe_tracks_cb)

        # Wire drawer widgets to the same attribute names used by the slots
        tile.status_label.setVisible(True)
        self._at_status = tile.status_label
        self._at_log    = tile.log_box
        self._at_prog   = tile.progress_bar

        tile.set_run_button("Export", self._on_export_at)
        self._at_open = tile.add_secondary_button("Open File")
        self._at_open.setEnabled(False)
        self._at_folder = tile.add_icon_button("\U0001f4c1", "Open folder")
        self._at_folder.setEnabled(False)
        tile.finish_footer()
        return tile

    def _build_codec_tile(self) -> ToolTile:
        tile = ToolTile(
            "\u2261", "Codec List Export",
            "Export a grouped list of all tracks by codec format.",
        )

        chips_row = QtWidgets.QHBoxLayout()
        chips_row.setSpacing(6)
        self._codec_ext_cbs: dict[str, QtWidgets.QCheckBox] = {}
        for ext in (".flac", ".mp3", ".m4a", ".aac", ".wav", ".opus", ".ogg"):
            cb = self._make_chip(ext)
            self._codec_ext_cbs[ext] = cb
            chips_row.addWidget(cb)
        chips_row.addStretch(1)
        tile.add_option_layout(chips_row)

        self._omit_paths_cb = _PillToggle("Filenames only (no full paths)")
        tile.add_option(self._omit_paths_cb)

        # Status label always visible in options area (export is synchronous)
        self._codec_status = QtWidgets.QLabel("Ready.")
        self._codec_status.setObjectName("statusLabel")
        tile.add_option(self._codec_status)
        self._codec_prog = tile.progress_bar   # available but drawer stays closed

        tile.set_run_button("Export", self._on_export_codec)
        self._codec_open = tile.add_secondary_button("Open File")
        self._codec_open.setEnabled(False)
        self._codec_folder = tile.add_icon_button("\U0001f4c1", "Open folder")
        self._codec_folder.setEnabled(False)
        tile.finish_footer()
        return tile

    def _build_cleanup_tile(self) -> ToolTile:
        tile = ToolTile(
            "\u2736", "File Cleanup",
            "Remove trailing \u2018 (1)\u2019, \u2018 (2)\u2019, \u2018 copy\u2019 suffixes "
            "from audio filenames left by macOS Finder and Windows Explorer. "
            "Playlists referencing renamed files are updated automatically.",
        )
        self._cleanup_tile = tile

        tile.status_label.setVisible(True)
        self._cleanup_status = tile.status_label
        self._cleanup_log    = tile.log_box

        self._cleanup_run_btn = tile.set_run_button("Run Cleanup", self._on_file_cleanup)
        tile.finish_footer()
        return tile

    @staticmethod
    def _make_diag_button(
        label: str,
        icon_sp: "QtWidgets.QStyle.StandardPixmap",
        slot,
    ) -> QtWidgets.QPushButton:
        """Return a square diagnostic button: label on top, Qt standard icon below."""
        from gui.themes.animations import AnimatedButton

        SP = QtWidgets.QStyle.StandardPixmap
        btn = AnimatedButton("", None)
        btn.setFixedHeight(72)
        btn.setMinimumWidth(80)
        btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.clicked.connect(slot)

        # Inner layout: text label on top, icon label on bottom
        inner = QtWidgets.QVBoxLayout(btn)
        inner.setContentsMargins(4, 8, 4, 8)
        inner.setSpacing(4)

        text_lbl = QtWidgets.QLabel(label)
        text_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop
        )
        text_lbl.setWordWrap(True)
        text_lbl.setStyleSheet("font-size: 10px; background: transparent;")
        text_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        icon_lbl = QtWidgets.QLabel()
        icon_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        icon_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        icon_lbl.setStyleSheet("background: transparent;")
        px = QtWidgets.QApplication.style().standardIcon(icon_sp).pixmap(22, 22)
        icon_lbl.setPixmap(px)

        inner.addWidget(text_lbl, 1)
        inner.addWidget(icon_lbl, 0)
        return btn

    def _build_diag_tile(self) -> ToolTile:
        tile = ToolTile(
            "\u2699", "Diagnostics",
            "Media testing, duplicate analysis tools, and system utilities.",
        )

        SP = QtWidgets.QStyle.StandardPixmap
        _entries = [
            ("M4A Tester\u2026",     SP.SP_MediaPlay,              self._on_m4a_tester),
            ("Opus Tester\u2026",    SP.SP_MediaVolume,            self._on_opus_tester),
            ("Bucketing POC\u2026",  SP.SP_FileDialogContentsView, self._on_bucketing_poc),
            ("Scan Engine\u2026",    SP.SP_BrowserReload,          self._on_scan_engine),
            ("Fuzzy Finder\u2026",   SP.SP_FileDialogListView,     self._on_fuzzy_dupes),
            ("Pair Review\u2026",    SP.SP_FileDialogDetailedView, self._on_pair_review),
            ("View Crash Log\u2026", SP.SP_MessageBoxWarning,      self._on_crash_log),
        ]

        _COLS = 4
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        for col in range(_COLS):
            grid.setColumnStretch(col, 1)

        for idx, (label, icon_sp, slot) in enumerate(_entries):
            row, col = divmod(idx, _COLS)
            btn = self._make_diag_button(label, icon_sp, slot)
            grid.addWidget(btn, row, col)

        # Fill remaining cells in last row with spacers so buttons don't stretch oddly
        last_count = len(_entries) % _COLS
        if last_count:
            last_row = len(_entries) // _COLS
            for col in range(last_count, _COLS):
                grid.addItem(
                    QtWidgets.QSpacerItem(
                        0, 0,
                        QtWidgets.QSizePolicy.Policy.Expanding,
                        QtWidgets.QSizePolicy.Policy.Minimum,
                    ),
                    last_row, col,
                )

        tile.add_option_layout(grid)
        tile.hide_footer()
        return tile

    def _build_validator_tile(self) -> ToolTile:
        tile = ToolTile(
            "\u2713", "Library Validator",
            "Verify that your library folder layout matches AlphaDEX conventions.",
        )

        self._val_log = QtWidgets.QPlainTextEdit()
        self._val_log.setReadOnly(True)
        self._val_log.setFixedHeight(160)
        self._val_log.setObjectName("logBox")
        self._val_log.setStyleSheet(
            "QPlainTextEdit { background: rgba(0,0,0,0.1); border-radius: 6px; }"
        )
        tile.add_option(self._val_log)

        tile.set_run_button("Run Validator", self._on_validate)
        tile.finish_footer()
        return tile

    # ── Codec chip helpers ─────────────────────────────────────────────────

    def _make_chip(self, label: str) -> QtWidgets.QCheckBox:
        cb = QtWidgets.QCheckBox(label)
        cb.setChecked(True)
        self._codec_chips[label] = cb
        self._refresh_chip(cb)
        cb.stateChanged.connect(lambda _s, c=cb: self._refresh_chip(c))
        return cb

    def _refresh_chip(self, cb: QtWidgets.QCheckBox) -> None:
        t = get_manager().current
        if cb.isChecked():
            style = (
                f"QCheckBox {{"
                f" background: {_rgba(t.accent, 0.16)};"
                f" border: 1px solid {t.accent};"
                f" border-radius: 11px;"
                f" padding: 4px 12px;"
                f" color: {t.accent};"
                f" font-size: 13px;"
                f"}}"
                f"QCheckBox::indicator {{ width: 0; height: 0; }}"
            )
        else:
            style = (
                f"QCheckBox {{"
                f" background: transparent;"
                f" border: 1px solid {t.card_border};"
                f" border-radius: 11px;"
                f" padding: 4px 12px;"
                f" color: {t.text_secondary};"
                f" font-size: 13px;"
                f"}}"
                f"QCheckBox::indicator {{ width: 0; height: 0; }}"
            )
        cb.setStyleSheet(style)

    def _reveal_card(self, card: QtWidgets.QWidget) -> None:
        if card.maximumHeight() > 0:
            return
        target = max(card.sizeHint().height(), 80)
        anim = QtCore.QPropertyAnimation(card, b"maximumHeight", self)
        anim.setDuration(220)
        anim.setStartValue(0)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: card.setMaximumHeight(16777215))
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _flash_label(self, label: QtWidgets.QLabel, from_hex: str, duration: int = 1600) -> None:
        t = get_manager().current
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(QtGui.QColor(from_hex))
        anim.setEndValue(QtGui.QColor(t.text_secondary))
        anim.setDuration(duration)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        anim.valueChanged.connect(
            lambda c: label.setStyleSheet(f"color: {c.name()}; font-size: 12px;")
        )
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    # ── Theme refresh ──────────────────────────────────────────────────────

    def _on_theme_changed_base(self, tokens: object) -> None:
        super()._on_theme_changed_base(tokens)
        for cb in self._codec_chips.values():
            self._refresh_chip(cb)

    # ── Slots: Artist / Title ──────────────────────────────────────────────

    @Slot()
    def _on_export_at(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return
        if self._at_worker and self._at_worker.isRunning():
            return

        self._at_log.clear()
        self._at_status.setText("Scanning\u2026")
        self._at_status.setStyleSheet("color: inherit; font-size: 12px;")
        self._at_prog.setValue(0)
        self._at_open.setEnabled(False)
        self._at_folder.setEnabled(False)
        if self._at_tile is not None:
            self._at_tile.open_drawer()

        self._at_worker = ArtistTitleWorker(
            self._library_path,
            self._exclude_flac_cb.isChecked(),
            self._dupe_tracks_cb.isChecked(),
            parent=self,
        )
        self._at_worker.progress.connect(self._on_at_progress)
        self._at_worker.log_line.connect(self._at_log.appendPlainText)
        self._at_worker.finished.connect(self._on_at_finished)
        self._at_worker.error.connect(self._on_at_error)
        self._at_worker.start()
        self._log("Starting artist/title export\u2026", "info")

    @Slot(int, int)
    def _on_at_progress(self, done: int, total: int) -> None:
        self._at_prog.setMaximum(max(total, 1))
        self._at_prog.setValue(done)
        self._at_status.setText(f"{done} / {total} processed")

    @Slot(str, int, int)
    def _on_at_finished(self, out_path: str, entry_count: int, error_count: int) -> None:
        self._at_prog.setValue(self._at_prog.maximum())
        note = f"  ({error_count} files skipped)" if error_count else ""
        self._at_status.setText(f"Done \u2014 {entry_count} entries written{note}")
        self._at_log.appendPlainText(f"Export complete: {out_path}")
        if error_count:
            self._at_log.appendPlainText(f"Skipped {error_count} files due to read errors.")
        t = get_manager().current
        self._flash_label(self._at_status, t.success)

        if self._at_open_connected:
            try:
                self._at_open.clicked.disconnect()
            except RuntimeError:
                pass
        self._at_open.clicked.connect(lambda: self._open_file(out_path))
        self._at_open_connected = True
        self._at_open.setEnabled(True)

        if self._at_folder_connected:
            try:
                self._at_folder.clicked.disconnect()
            except RuntimeError:
                pass
        self._at_folder.clicked.connect(lambda: self._open_folder(out_path))
        self._at_folder_connected = True
        self._at_folder.setEnabled(True)
        self._log(f"Artist/title export complete: {entry_count} entries \u2192 {out_path}", "ok")
        self._at_worker = None

    @Slot(str)
    def _on_at_error(self, message: str) -> None:
        t = get_manager().current
        self._at_status.setText(f"Error: {message}")
        self._flash_label(self._at_status, t.danger)
        self._at_log.appendPlainText(f"Export failed: {message}")
        self._log(f"Artist/title export failed: {message}", "error")
        self._at_worker = None

    # ── Slots: Codec Export ────────────────────────────────────────────────

    @Slot()
    def _on_export_codec(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return
        selected_exts = {ext for ext, cb in self._codec_ext_cbs.items() if cb.isChecked()}
        if not selected_exts:
            QtWidgets.QMessageBox.warning(self, "No Codecs", "Select at least one codec.")
            return

        omit_paths = self._omit_paths_cb.isChecked()
        self._codec_prog.setValue(0)
        self._codec_status.setText("Scanning\u2026")
        self._log("Starting codec list export\u2026", "info")

        try:
            by_ext: dict[str, list[str]] = {e: [] for e in sorted(selected_exts)}
            total = 0
            for dirpath, _, files in os.walk(self._library_path):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in by_ext:
                        path = f if omit_paths else os.path.join(dirpath, f)
                        by_ext[ext].append(path)
                        total += 1

            out = Path(self._library_path) / "Docs" / "codec_file_list.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            for ext in sorted(by_ext):
                lines.append(f"=== {ext} ({len(by_ext[ext])} files) ===")
                lines.extend(sorted(by_ext[ext]))
                lines.append("")
            out.write_text("\n".join(lines), encoding="utf-8")

            self._codec_prog.setValue(100)
            self._codec_status.setText(f"Written: {out}  ({total} files)")
            t = get_manager().current
            self._flash_label(self._codec_status, t.success)

            if self._codec_open_connected:
                try:
                    self._codec_open.clicked.disconnect()
                except RuntimeError:
                    pass
            self._codec_open.clicked.connect(lambda: self._open_file(str(out)))
            self._codec_open_connected = True
            self._codec_open.setEnabled(True)

            if self._codec_folder_connected:
                try:
                    self._codec_folder.clicked.disconnect()
                except RuntimeError:
                    pass
            self._codec_folder.clicked.connect(lambda: self._open_folder(str(out)))
            self._codec_folder_connected = True
            self._codec_folder.setEnabled(True)
            self._log(f"Codec list exported: {total} files \u2192 {out}", "ok")

        except Exception as exc:  # noqa: BLE001
            t = get_manager().current
            self._codec_status.setText(str(exc))
            self._flash_label(self._codec_status, t.danger)
            self._log(str(exc), "error")

    # ── Slots: File Cleanup ────────────────────────────────────────────────

    @Slot()
    def _on_file_cleanup(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Run File Cleanup",
            "This will rename files by removing trailing copy/numeric suffixes.\n\nProceed?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self._cleanup_log.clear()
        self._cleanup_status.setText("Running\u2026")
        self._cleanup_status.setStyleSheet("color: inherit; font-size: 12px;")
        self._cleanup_run_btn.setEnabled(False)
        if self._cleanup_tile is not None:
            self._cleanup_tile.open_drawer()
        self._log("Starting file cleanup\u2026", "info")

        self._cleanup_worker = FileCleanupWorker(self._library_path)
        self._cleanup_worker.log_line.connect(self._cleanup_log.appendPlainText)
        self._cleanup_worker.finished.connect(self._on_cleanup_finished)
        self._cleanup_worker.start()

    @Slot(bool, str)
    def _on_cleanup_finished(self, success: bool, message: str) -> None:
        self._cleanup_status.setText(message)
        t = get_manager().current
        self._flash_label(self._cleanup_status, t.success if success else t.danger)
        self._cleanup_run_btn.setEnabled(True)
        self._log(message, "ok" if success else "error")
        self._cleanup_worker = None

    # ── Slots: Validator ───────────────────────────────────────────────────

    @Slot()
    def _on_validate(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return
        try:
            import validator
            valid, errors = validator.validate_soundvault_structure(self._library_path)
            if valid:
                self._val_log.setPlainText("\u2713 Library structure is valid.")
                self._log("Validation complete \u2014 structure OK.", "ok")
            else:
                self._val_log.setPlainText("\n".join(errors))
                self._log("Validation found issues.", "error")
        except Exception as exc:  # noqa: BLE001
            self._val_log.setPlainText(str(exc))
            self._log(str(exc), "error")

    # ── Slots: Diagnostics ─────────────────────────────────────────────────

    @Slot()
    def _on_m4a_tester(self) -> None:
        from gui.dialogs.media_tester_dialog import MediaTesterDialog
        dlg = MediaTesterDialog(self, codec="m4a")
        dlg.exec()

    @Slot()
    def _on_opus_tester(self) -> None:
        from gui.dialogs.media_tester_dialog import MediaTesterDialog
        dlg = MediaTesterDialog(self, codec="opus")
        dlg.exec()

    @Slot()
    def _on_bucketing_poc(self) -> None:
        from gui.dialogs.bucketing_poc_dialog import BucketingPocDialog
        dlg = BucketingPocDialog(self._library_path, self)
        dlg.exec()

    @Slot()
    def _on_scan_engine(self) -> None:
        from gui.dialogs.scan_engine_dialog import ScanEngineDialog
        dlg = ScanEngineDialog(self._library_path, self)
        dlg.exec()

    @Slot()
    def _on_crash_log(self) -> None:
        from gui.dialogs.crash_log_dialog import CrashLogDialog
        dlg = CrashLogDialog(self)
        dlg.exec()

    @Slot()
    def _on_fuzzy_dupes(self) -> None:
        from gui.dialogs.fuzzy_dupe_dialog import FuzzyDupeDialog
        dlg = FuzzyDupeDialog(self._library_path, self)
        dlg.exec()

    @Slot()
    def _on_pair_review(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return
        from gui.dialogs.pair_review_dialog import PairReviewDialog, load_pairs
        if not load_pairs(self._library_path):
            QtWidgets.QMessageBox.information(
                self, "Duplicate Pair Review",
                "No duplicate preview found.\n\n"
                "Run Duplicate Finder first to generate paired results.",
            )
            return
        dlg = PairReviewDialog(self._library_path, self)
        dlg.exec()

    # ── Utility ───────────────────────────────────────────────────────────

    def _open_file(self, path: str) -> None:
        import subprocess
        import sys
        if sys.platform == "win32":
            import os as _os
            _os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _open_folder(self, path: str) -> None:
        """Open the folder containing *path* in the system file manager."""
        import subprocess
        import sys
        folder = str(Path(path).parent)
        if sys.platform == "win32":
            # /select highlights the file inside the folder
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", folder])
