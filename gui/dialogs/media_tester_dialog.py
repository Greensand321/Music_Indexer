"""M4A / Opus tester dialog — validate album art and metadata parsing.

The dialog reads files through ``utils.audio_metadata_reader.read_metadata``,
the same reader the rest of the app uses, so what it shows is what the
Indexer, Tag Fixer and Player will see. (It used to probe tags itself with a
guessed ``\\xa9`` + first-three-letters key, which raised on Opus files,
showed the album as "Albumartist" for M4A and never found Opus cover art —
TOOL-08/09/10.)
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


class MediaTesterDialog(QtWidgets.QDialog):
    """Open an .m4a or .opus file and display its album art and metadata."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, codec: str = "m4a") -> None:
        super().__init__(parent)
        self._codec = codec.lower()
        self.setWindowTitle(f"{codec.upper()} Tester")
        self.setMinimumWidth(480)
        self._build_ui()
        get_manager().theme_changed.connect(self._apply_theme)

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        root.addWidget(QtWidgets.QLabel(
            f"Select a .{self._codec} file to validate album art and metadata parsing."
        ))

        # ── File section ───────────────────────────────────────────────────
        file_grp = QtWidgets.QGroupBox(f"{self._codec.upper()} File")
        file_l = QtWidgets.QHBoxLayout(file_grp)
        self._file_entry = QtWidgets.QLineEdit()
        self._file_entry.setReadOnly(True)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        file_l.addWidget(self._file_entry, 1)
        file_l.addWidget(browse)
        root.addWidget(file_grp)

        # ── Album art ──────────────────────────────────────────────────────
        art_grp = QtWidgets.QGroupBox("Album Art")
        art_l = QtWidgets.QVBoxLayout(art_grp)
        self._art_lbl = QtWidgets.QLabel("No album art found")
        self._art_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._art_lbl.setFixedHeight(180)
        # Themed via the palette (it was a hard-coded light-grey box).
        self._art_lbl.setAutoFillBackground(True)
        art_l.addWidget(self._art_lbl)
        root.addWidget(art_grp)

        # ── Metadata ───────────────────────────────────────────────────────
        meta_grp = QtWidgets.QGroupBox("Metadata")
        meta_l = QtWidgets.QVBoxLayout(meta_grp)
        self._meta_text = QtWidgets.QPlainTextEdit()
        self._meta_text.setReadOnly(True)
        self._meta_text.setMinimumHeight(160)
        self._meta_text.setObjectName("logText")  # monospace, themed
        meta_l.addWidget(self._meta_text)
        root.addWidget(meta_grp)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)
        self._apply_theme(get_manager().current)

    def _apply_theme(self, tokens) -> None:
        pal = self._art_lbl.palette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(tokens.input_bg))
        pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(tokens.text_muted))
        self._art_lbl.setPalette(pal)

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

    def _load_file(self, path: str) -> None:
        lines, art, error = describe_media(path)
        if error and not lines:
            self._meta_text.setPlainText(f"Could not read tags: {error}")
            self._clear_art("No album art (file could not be read)")
            return
        text = "\n".join(lines) if lines else "No tags found."
        if error:
            text += f"\n\nWarning: {error}"
        self._meta_text.setPlainText(text)

        if art:
            pix = QtGui.QPixmap()
            pix.loadFromData(art)
            if not pix.isNull():
                scaled = pix.scaled(
                    180, 180,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                self._art_lbl.setPixmap(scaled)
                return
            self._clear_art("Embedded art found but it is not a readable image")
            return
        self._clear_art()
