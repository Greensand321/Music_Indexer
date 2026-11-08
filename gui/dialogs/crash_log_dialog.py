"""Crash log viewer dialog.

Shows the tail of the crash log the Qt app writes (see
``crash_logger.install_for_qt``, installed by ``alpha_dex_gui.main``):
uncaught Python exceptions in ``~/.alphadex/crash.log`` and native crash
tracebacks (faulthandler) in ``~/.alphadex/faults.log``.

Glass dialog chrome: a title, the two log files as one middle-elided muted
line (full paths in the tooltip), the log on a monospaced well and Close on
the footer band.
"""
from __future__ import annotations

from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets

_TAIL_LINES = 200


def crash_log_paths() -> list[Path]:
    """The files this viewer reads, in display order."""
    import crash_logger

    return [Path(crash_logger.default_log_path()), Path(crash_logger.default_fault_log_path())]


def _tail(path: Path, lines: int = _TAIL_LINES) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:]).strip()


class CrashLogDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        from gui.themes.glass import apply_font
        from gui.widgets.glass import ElidedLabel

        self.setWindowTitle("Crash Log")
        self.resize(720, 480)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 16)
        root.setSpacing(10)

        title = QtWidgets.QLabel("Crash log")
        title.setObjectName("dialogTitle")
        apply_font(title, "title")
        root.addWidget(title)

        self._path_label = ElidedLabel("", QtCore.Qt.TextElideMode.ElideMiddle)
        self._path_label.setObjectName("mutedLabel")
        self._path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        root.addWidget(self._path_label)

        from gui.widgets.glass import GlassPanel

        well = GlassPanel("well", radius=14)
        well_lay = QtWidgets.QVBoxLayout(well)
        well_lay.setContentsMargins(12, 10, 6, 6)
        self._text = QtWidgets.QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setObjectName("wellMono")   # monospace, transparent on the well
        self._text.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._text.setAccessibleName("Crash log")
        well_lay.addWidget(self._text)
        root.addWidget(well, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        root.addWidget(buttons)

        self._load_log()

    def _load_log(self) -> None:
        paths = crash_log_paths()
        self._path_label.setText("  ·  ".join(str(p) for p in paths))
        self._path_label.setToolTip("\n".join(str(p) for p in paths))
        sections = []
        for path in paths:
            body = _tail(path)
            if body:
                sections.append(f"── {path.name} ──\n{body}")
        if sections:
            self._text.setPlainText("\n\n".join(sections))
            self._text.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        else:
            self._text.setPlainText(f"No crash log found.\n\nAlphaDEX records crashes in {paths[0]}.")
