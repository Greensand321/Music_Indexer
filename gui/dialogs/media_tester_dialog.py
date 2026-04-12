"""M4A / Opus tester dialog — validate album art and metadata parsing."""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot


class MediaTesterDialog(QtWidgets.QDialog):
    """Open an .m4a or .opus file and display its album art and metadata."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, codec: str = "m4a") -> None:
        super().__init__(parent)
        self._codec = codec.lower()
        self.setWindowTitle(f"{codec.upper()} Tester")
        self.setMinimumWidth(480)
        self._build_ui()

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
        self._art_lbl.setStyleSheet("background: #e2e8f0; border-radius: 6px; color: #94a3b8;")
        art_l.addWidget(self._art_lbl)
        root.addWidget(art_grp)

        # ── Metadata ───────────────────────────────────────────────────────
        meta_grp = QtWidgets.QGroupBox("Metadata")
        meta_l = QtWidgets.QVBoxLayout(meta_grp)
        self._meta_text = QtWidgets.QPlainTextEdit()
        self._meta_text.setReadOnly(True)
        self._meta_text.setMinimumHeight(160)
        self._meta_text.setStyleSheet("font-family: 'Consolas', monospace; font-size: 12px;")
        meta_l.addWidget(self._meta_text)
        root.addWidget(meta_grp)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)

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

    def _load_file(self, path: str) -> None:
        try:
            from mutagen import File as MutagenFile
            f = MutagenFile(path)
            if f is None:
                self._meta_text.setPlainText("Could not parse file.")
                return

            lines = []
            for key in ("title", "artist", "album", "albumartist", "tracknumber",
                        "discnumber", "date", "year", "genre", "compilation"):
                val = f.tags.get(key) or f.tags.get(f"\xa9{key[:3].lower()}")
                if val:
                    lines.append(f"{key.title()}: {val[0] if isinstance(val, list) else val}")
            self._meta_text.setPlainText("\n".join(lines) if lines else "No tags found.")

            # Extract artwork
            artwork_data: bytes | None = None
            if hasattr(f, "tags"):
                tags = f.tags
                if tags:
                    for key in list(tags.keys()):
                        if "covr" in key.lower() or "apic" in key.lower():
                            try:
                                tag_val = tags[key]
                                if isinstance(tag_val, list):
                                    artwork_data = bytes(tag_val[0])
                                else:
                                    artwork_data = bytes(tag_val)
                            except Exception:
                                pass

            if artwork_data:
                pix = QtGui.QPixmap()
                pix.loadFromData(artwork_data)
                if not pix.isNull():
                    scaled = pix.scaled(
                        180, 180,
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                    self._art_lbl.setPixmap(scaled)
                    self._art_lbl.setText("")
                    return
            self._art_lbl.clear()
            self._art_lbl.setText("No album art found")
        except Exception as exc:  # noqa: BLE001
            self._meta_text.setPlainText(f"Error: {exc}")
