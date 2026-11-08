"""M4A / Opus tester — validate album art and metadata parsing.

``MediaTesterPanel`` is embedded in Mirror → "Verify the mirror" (it used to
be a dialog opened from the Utilities page). The panel reads files through ``utils.audio_metadata_reader.read_metadata``,
the same reader the rest of the app uses, so what it shows is what the
Indexer, Tag Fixer and Player will see. (It used to probe tags itself with a
guessed ``\\xa9`` + first-three-letters key, which raised on Opus files,
showed the album as "Albumartist" for M4A and never found Opus cover art —
TOOL-08/09/10.)

Layout (Liquid Glass): the opened file as an elided path row with a folder
glyph button; until a file is open, an empty state; then a stat row (tags
found · cover art · problems), one glass row per problem (or one "all good"
row), the cover on a well and the tags as text.
"""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Slot
from gui.themes.manager import get_manager

# (label, key in read_metadata's tag dict), in display order.
_DISPLAY_FIELDS = (
    ("Title", "title"),
    ("Artist", "artist"),
    ("Album", "album"),
    ("Album artist", "albumartist"),
    ("Track", "tracknumber"),
    ("Disc", "discnumber"),
    ("Date", "date"),
    ("Year", "year"),
    ("Genre", "genre"),
    ("Compilation", "compilation"),
)
# Tags a mirrored file should never lose; a missing one is a problem row.
_KEY_FIELDS = ("Title", "Artist", "Album")
_NOTE_PREFIX = "Note: "


def describe_media(path: str) -> tuple[list[str], bytes | None, str | None]:
    """Return ``(metadata lines, first cover image bytes or None, error or None)``."""
    from utils.audio_metadata_reader import read_metadata

    try:
        tags, covers, error, _hint = read_metadata(path)
    except Exception as exc:  # noqa: BLE001 - report, never raise into the GUI
        return [], None, str(exc)
    lines = []
    for label, key in _DISPLAY_FIELDS:
        value = tags.get(key)
        if value in (None, "", []):
            continue
        if key == "year" and value == "Unknown":
            continue  # normalize_year's placeholder for "no date tag"
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        lines.append(f"{label}: {value}")
    art = next((c for c in covers if c), None)
    if art is None and Path(path).suffix.lower() in (".m4a", ".mp4"):
        art = _direct_mp4_cover(path)
        if art is not None:
            lines.append(
                "Note: cover art found only by reading the MP4 'covr' atom directly; "
                "the app's shared metadata reader did not return it."
            )
    return lines, art, error


def _direct_mp4_cover(path: str) -> bytes | None:
    try:
        from mutagen.mp4 import MP4
        covers = (MP4(path).tags or {}).get("covr") or []
    except Exception:  # noqa: BLE001
        return None
    return bytes(covers[0]) if covers else None


def find_problems(lines: list[str], has_art: bool, art_readable: bool,
                  error: str | None) -> list[tuple[str, str]]:
    """What went wrong with a file, as ``(tone, text)`` rows ("danger" ·
    "warning"); empty when the tags and the cover came through."""
    problems: list[tuple[str, str]] = []
    tags = [line for line in lines if not line.startswith(_NOTE_PREFIX)]
    if error and not tags:
        return [("danger", f"Could not read tags: {error}")]
    if error:
        problems.append(("warning", f"The reader warned: {error}"))
    present = {line.split(":", 1)[0] for line in tags}
    for label in _KEY_FIELDS:
        if label not in present:
            problems.append(("warning", f"No {label.lower()} tag"))
    if not has_art:
        problems.append(("warning", "No cover art"))
    elif not art_readable:
        problems.append(("danger", "Embedded cover art is not a readable image"))
    for line in lines:
        if line.startswith(_NOTE_PREFIX):
            problems.append(("warning", line[len(_NOTE_PREFIX):].rstrip(".").capitalize()))
    return problems


# ── Small painted parts ───────────────────────────────────────────────────────

def _tone_colour(tone: str) -> QtGui.QColor:
    t = get_manager().current
    return QtGui.QColor({"danger": t.danger, "warning": t.warning,
                         "success": t.success}.get(tone, t.text_secondary))


class _ToneGlyph(QtWidgets.QWidget):
    """A 16 px line glyph in a tone colour, painted at paint time."""

    def __init__(self, name: str, tone: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._name, self._tone = name, tone
        self.setFixedSize(18, 18)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.widgets.glyphs import paint_glyph

        p = QtGui.QPainter(self)
        try:
            paint_glyph(p, QtCore.QRectF(1, 1, 16, 16), self._name, _tone_colour(self._tone))
        finally:
            p.end()


class ProblemRow(QtWidgets.QWidget):
    """One glass list row (radius 12, faint fill): a tone glyph and a line of
    text that wraps when it must."""

    _GLYPHS = {"danger": "close", "warning": "warning", "success": "check"}

    def __init__(self, tone: str, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mediaProblemRow")
        self._tone = tone
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)
        lay.addWidget(_ToneGlyph(self._GLYPHS.get(tone, "info"), tone), 0,
                      QtCore.Qt.AlignmentFlag.AlignTop)
        self.label = QtWidgets.QLabel(text)
        self.label.setWordWrap(True)
        lay.addWidget(self.label, 1)
        self.setMinimumHeight(34)
        self.setAccessibleName(text)

    def tone(self) -> str:
        return self._tone

    def text(self) -> str:
        return self.label.text()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, paint_surface
        from gui.themes.tokens import RADIUS, Surface

        well = current_material().well
        p = QtGui.QPainter(self)
        try:
            paint_surface(p, QtCore.QRectF(self.rect()), Surface(well.fill, radius=RADIUS.sm))
        finally:
            p.end()


class _ArtWell(QtWidgets.QLabel):
    """The cover on a well (rounded); its placeholder text while there is none.
    ``pixmap()`` is the shown cover, as on any QLabel."""

    SIDE = 148

    def __init__(self, text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("mediaArtWell")
        self.setFixedSize(self.SIDE, self.SIDE)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        from gui.themes.glass import current_material, font, paint_surface

        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
            rect = QtCore.QRectF(self.rect())
            well = current_material().well
            paint_surface(p, rect, well)
            pix = self.pixmap()
            if pix is not None and not pix.isNull():
                inner = rect.adjusted(6, 6, -6, -6)
                clip = QtGui.QPainterPath()
                clip.addRoundedRect(inner, well.radius - 4, well.radius - 4)
                p.setClipPath(clip)
                size = QtCore.QSizeF(pix.size()) / pix.devicePixelRatio()
                size.scale(inner.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                target = QtCore.QRectF(QtCore.QPointF(0, 0), size)
                target.moveCenter(inner.center())
                p.drawPixmap(target, pix, QtCore.QRectF(pix.rect()))
            elif self.text():
                p.setFont(font("caption"))
                p.setPen(self.palette().color(self.foregroundRole()))
                p.drawText(rect.adjusted(12, 12, -12, -12),
                           int(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap),
                           self.text())
        finally:
            p.end()


# ── The panel ─────────────────────────────────────────────────────────────────

class MediaTesterPanel(QtWidgets.QWidget):
    """Open an .m4a or .opus file and display its album art and metadata.

    Embedded in Mirror → "Verify the mirror"; ``MediaTesterDialog`` wraps it
    in a window.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None, codec: str = "m4a") -> None:
        super().__init__(parent)
        self._codec = codec.lower()
        self._problem_rows: list[ProblemRow] = []
        self._build_ui()

    @property
    def codec(self) -> str:
        return self._codec

    def _build_ui(self) -> None:
        from gui.themes.tokens import SPACE
        from gui.widgets.glass_parts import EmptyState, StatRow
        from gui.widgets.path_field import PathField

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(SPACE.sm)

        ext = self._codec
        self._file_entry = PathField(f"No .{ext} file opened", button_tip=f"Open a .{ext} file")
        self._file_entry.setAccessibleName(f"{ext.upper()} file")
        self._file_entry.button.setObjectName(f"browse_{ext}")
        self._file_entry.browse_requested.connect(self._on_browse)
        root.addWidget(self._file_entry)

        self._empty = EmptyState(
            "No file opened",
            f"Open a .{ext} file from your mirror to see the tags and cover art "
            "AlphaDEX reads from it.",
        )
        root.addWidget(self._empty)

        self._results = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(self._results)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(SPACE.sm)
        self._stats = StatRow([("tags", "Tags found"), ("art", "Cover art"),
                               ("problems", "Problems")])
        rl.addWidget(self._stats)
        self._problems_box = QtWidgets.QVBoxLayout()
        self._problems_box.setSpacing(SPACE.xs)
        rl.addLayout(self._problems_box)

        detail = QtWidgets.QHBoxLayout()
        detail.setSpacing(SPACE.sm)
        self._art_lbl = _ArtWell("No album art found")
        detail.addWidget(self._art_lbl, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        from gui.widgets.glass import GlassPanel

        tags_well = GlassPanel("well")
        tags_well.setFixedHeight(_ArtWell.SIDE)
        tw = QtWidgets.QVBoxLayout(tags_well)
        tw.setContentsMargins(12, 8, 4, 4)
        self._meta_text = QtWidgets.QPlainTextEdit()
        self._meta_text.setReadOnly(True)
        self._meta_text.setObjectName("wellText")   # transparent on the well
        self._meta_text.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._meta_text.setAccessibleName("Tags")
        tw.addWidget(self._meta_text)
        detail.addWidget(tags_well, 1)
        rl.addLayout(detail)
        self._results.setVisible(False)
        root.addWidget(self._results)

    @Slot()
    def _on_browse(self) -> None:
        ext = self._codec
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Select .{ext} file", str(Path.home()),
            f"{ext.upper()} files (*.{ext});;All files (*)"
        )
        if not path:
            return
        self._file_entry.setText(path)
        self._load_file(path)

    def _clear_art(self, text: str = "No album art found") -> None:
        self._art_lbl.clear()
        self._art_lbl.setText(text)

    def problems(self) -> list[tuple[str, str]]:
        """The problem rows on screen now, as ``(tone, text)`` (not the
        all-clear row)."""
        return [(row.tone(), row.text()) for row in self._problem_rows if row.tone() != "success"]

    def _show(self, lines: list[str], has_art: bool, art_readable: bool,
              error: str | None) -> None:
        """Fill the stat row and the problem rows, and show the results."""
        while self._problem_rows:
            row = self._problem_rows.pop()
            self._problems_box.removeWidget(row)
            row.deleteLater()
        problems = find_problems(lines, has_art, art_readable, error)
        rows = problems or [("success", "Tags and cover art came through.")]
        for tone, text in rows:
            row = ProblemRow(tone, text)
            self._problems_box.addWidget(row)
            self._problem_rows.append(row)
        tags = [line for line in lines if not line.startswith(_NOTE_PREFIX)]
        self._stats.tile("tags").set_value(f"{len(tags)}")
        self._stats.tile("tags").set_label(f"Tags found (of {len(_DISPLAY_FIELDS)})")
        art_ok = has_art and art_readable
        self._stats.tile("art").set_value("Yes" if art_ok else "No",
                                          tone="success" if art_ok else "warning")
        worst = "danger" if any(t == "danger" for t, _ in problems) else "warning"
        self._stats.tile("problems").set_value(
            str(len(problems)), tone=worst if problems else "success")
        was_hidden = not self._results.isVisible()
        self._empty.setVisible(False)
        self._results.setVisible(True)
        if was_hidden and self.isVisible():
            from gui.themes import motion
            motion.reveal(self._results, duration="fast")

    def _load_file(self, path: str) -> None:
        if self._file_entry.text() != path:
            self._file_entry.setText(path)
        lines, art, error = describe_media(path)
        if error and not lines:
            self._meta_text.setPlainText(f"Could not read tags: {error}")
            self._clear_art("No album art (file could not be read)")
            self._show(lines, False, False, error)
            return
        tags = [line for line in lines if not line.startswith(_NOTE_PREFIX)]
        text = "\n".join(tags) if tags else "No tags found."
        if error:
            text += f"\n\nWarning: {error}"
        self._meta_text.setPlainText(text)

        art_readable = False
        if art:
            pix = QtGui.QPixmap()
            pix.loadFromData(art)
            if not pix.isNull():
                side = _ArtWell.SIDE * 2   # crisp on high-DPI screens
                self._art_lbl.setPixmap(pix.scaled(
                    side, side,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                ))
                art_readable = True
            else:
                self._clear_art("Embedded art found but it is not a readable image")
        else:
            self._clear_art()
        self._show(lines, bool(art), art_readable, error)


class MediaTesterDialog(QtWidgets.QDialog):
    """``MediaTesterPanel`` in its own window (the app embeds the panel)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, codec: str = "m4a") -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{codec.upper()} Tester")
        self.setMinimumWidth(480)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 16)
        root.setSpacing(16)
        self.panel = MediaTesterPanel(self, codec=codec)
        root.addWidget(self.panel)
        root.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        root.addWidget(buttons)
        # The panel's widgets, under the names callers used before the split.
        self._art_lbl = self.panel._art_lbl
        self._meta_text = self.panel._meta_text
        self._file_entry = self.panel._file_entry

    def _load_file(self, path: str) -> None:
        self.panel._load_file(path)
