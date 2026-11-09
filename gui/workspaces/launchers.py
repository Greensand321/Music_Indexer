"""Duplicates-page sections that open a dialog: Pair review and the three
prototypes (Fuzzy finder, Scan engine, Bucketing).

These tools still live in their own dialogs (``gui/dialogs/``); each section
says what the tool is for and opens it. The prototypes carry an "Experimental"
pill and a one-line "not ready to test" note: they are kept so the owner can
see where they would live, not as part of the real duplicate workflow.

One live dialog per tool (TOOL-21 / DUP-20): a dialog is deleted after it
closes unless its worker is still running, in which case the next "Open"
re-opens that same dialog (showing the run in progress) instead of starting a
second, concurrent run.
"""
from __future__ import annotations

from gui.compat import QtCore, QtWidgets, Slot
from gui.themes.glass import apply_font
from gui.themes.tokens import RADIUS, SPACE
from gui.widgets.glass import GlassPanel
from gui.widgets.glass_parts import hero_title
from gui.widgets.glyphs import GlyphLabel, set_glyph
from gui.widgets.pill import ExperimentalPill
from gui.workspaces.base import WorkspaceBase

NOT_READY_NOTE = (
    "Experimental — not ready to test. A prototype of another way to find "
    "duplicates: its results are not used by Find duplicates and have not been "
    "checked. Use Find duplicates for real work."
)


def dialog_busy(dlg: QtWidgets.QDialog) -> bool:
    """True while ``dlg``'s worker (or any job it started) is still running."""
    try:
        worker = getattr(dlg, "_worker", None)
        if worker is not None and worker.isRunning():
            return True
    except RuntimeError:  # C++ object already gone
        return False
    from gui.workers import job_registry
    try:
        name = type(dlg).__name__
    except RuntimeError:
        return False
    return any(job.owner_name == name for job in job_registry().jobs())


class LauncherWorkspace(WorkspaceBase):
    """A section that explains a dialog-based tool and opens it.

    Subclasses set the class attributes and implement ``_make_dialog``.
    """

    TITLE = ""
    HEADLINE = ""          # the card title: what the tool is for (the tab names the tool)
    SUMMARY = ""           # one line under it
    DETAILS: tuple[str, ...] = ()   # bullet points in the card
    OPEN_LABEL = "Open…"
    EXPERIMENTAL = False
    NEEDS_LIBRARY = False

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._dialog: QtWidgets.QDialog | None = None
        self.pill: ExperimentalPill | None = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """One glass card: what the tool is for, what it does (a row per
        point), the Experimental marker where flagged, and Open. The section
        tab already names the tool, so the card's title says what it is for."""
        cl = self.content_layout
        card = self._make_card()
        card.setObjectName("headerCard")
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(SPACE.md)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(SPACE.sm)
        title = hero_title(self.HEADLINE or self.TITLE)
        title.setWordWrap(True)
        title_row.addWidget(title)
        if self.EXPERIMENTAL:
            self.pill = ExperimentalPill()
            title_row.addWidget(self.pill, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch(1)
        lay.addLayout(title_row)
        lay.addWidget(self._make_subtitle(self.SUMMARY))

        if self.EXPERIMENTAL:
            well = GlassPanel("well", radius=RADIUS.sm)
            well.setObjectName("experimentalWell")
            wl = QtWidgets.QHBoxLayout(well)
            wl.setContentsMargins(12, 10, 12, 10)
            wl.setSpacing(SPACE.sm)
            wl.addWidget(GlyphLabel("warning", role="warning"), 0, QtCore.Qt.AlignmentFlag.AlignTop)
            note = QtWidgets.QLabel(NOT_READY_NOTE)
            note.setObjectName("experimentalNote")
            note.setWordWrap(True)
            apply_font(note, "label")
            wl.addWidget(note, 1)
            lay.addWidget(well)
            self._note = note

        if self.DETAILS:
            points = QtWidgets.QVBoxLayout()
            points.setSpacing(SPACE.xs)
            for line in self.DETAILS:
                row = QtWidgets.QHBoxLayout()
                row.setSpacing(SPACE.sm)
                row.addWidget(GlyphLabel("chevron_right", role="accent"), 0, QtCore.Qt.AlignmentFlag.AlignTop)
                text = QtWidgets.QLabel(line)
                text.setWordWrap(True)
                apply_font(text, "body")
                row.addWidget(text, 1)
                points.addLayout(row)
            lay.addLayout(points)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(SPACE.sm)
        self._hint = QtWidgets.QLabel("The tool opens in its own window.")
        self._hint.setObjectName("statusHint")
        self._hint.setWordWrap(True)
        row.addWidget(self._hint, 1)
        self._open_btn = self._make_primary_button(self.OPEN_LABEL)
        self._open_btn.setObjectName("launcherOpenBtn")
        set_glyph(self._open_btn, "external")
        self._open_btn.setToolTip(f"Open {self.TITLE} in its own window.")
        self._open_btn.clicked.connect(self.open_tool)
        row.addWidget(self._open_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        lay.addLayout(row)
        cl.addWidget(card)
        cl.addStretch(1)

    # ── Opening ───────────────────────────────────────────────────────────

    def _make_dialog(self) -> QtWidgets.QDialog:
        raise NotImplementedError

    def _can_open(self) -> bool:
        """Checks before opening (may tell the user why not)."""
        if self.NEEDS_LIBRARY and not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return False
        return True

    @Slot()
    def open_tool(self) -> QtWidgets.QDialog | None:
        dlg = self._dialog
        self._dialog = None
        if dlg is not None:
            try:
                busy = dialog_busy(dlg)
            except RuntimeError:
                busy = False
            if not busy:
                dlg.deleteLater()
                dlg = None
        if dlg is None:
            if not self._can_open():
                return None
            dlg = self._make_dialog()
        self._dialog = dlg
        try:
            dlg.exec()
        finally:
            if dialog_busy(dlg):
                self._log(
                    f"{dlg.windowTitle() or self.TITLE}: still running in the background; "
                    "reopen it to see progress.", "info",
                )
                self._hint.setText("Still running in the background — open it to see progress.")
            else:
                self._dialog = None
                dlg.deleteLater()
                self._hint.setText("The tool opens in its own window.")
        return dlg

    def shutdown(self) -> None:
        if self._dialog is not None:
            try:
                self._dialog.close()
            except RuntimeError:
                pass
        super().shutdown()


class PairReviewWorkspace(LauncherWorkspace):
    TITLE = "Pair review"
    HEADLINE = "Decide pair by pair"
    SUMMARY = (
        "Step through duplicate pairs side by side and choose which copy to keep."
    )
    DETAILS = (
        "Reads the pairs from the duplicate preview in Docs/ (run Find duplicates first).",
        "Shows both files' tags, format and cover art next to each other.",
        "A file you reject is moved to Quarantine/, never deleted; every decision is "
        "logged to Docs/.",
    )
    OPEN_LABEL = "Open Pair Review…"
    NEEDS_LIBRARY = True

    def _can_open(self) -> bool:
        if not super()._can_open():
            return False
        from gui.dialogs.pair_review_dialog import load_pairs
        if not load_pairs(self._library_path):
            QtWidgets.QMessageBox.information(
                self, "Duplicate Pair Review",
                "No duplicate preview found.\n\n"
                "Run Find duplicates first to generate paired results.",
            )
            return False
        return True

    def _make_dialog(self) -> QtWidgets.QDialog:
        from gui.dialogs.pair_review_dialog import PairReviewDialog
        return PairReviewDialog(self._library_path, self)


class FuzzyFinderWorkspace(LauncherWorkspace):
    TITLE = "Fuzzy finder"
    HEADLINE = "Near-duplicate candidates"
    EXPERIMENTAL = True
    SUMMARY = "Lists likely duplicates using only the fingerprints Find duplicates has already cached."
    DETAILS = (
        "Runs the near-duplicate detector over cached fingerprints only; it does not "
        "fingerprint anything new.",
        "Read-only: lists candidate pairs, changes no file.",
        "The metadata word-overlap search it was designed around has no backend yet.",
    )
    OPEN_LABEL = "Open Fuzzy Finder…"

    def _make_dialog(self) -> QtWidgets.QDialog:
        from gui.dialogs.fuzzy_dupe_dialog import FuzzyDupeDialog
        return FuzzyDupeDialog(self._library_path, self)


class ScanEngineWorkspace(LauncherWorkspace):
    TITLE = "Scan engine"
    HEADLINE = "A staged duplicate scan"
    EXPERIMENTAL = True
    SUMMARY = "Audio headers first, then fingerprint buckets, then a verification pass."
    DETAILS = (
        "Writes its findings to a separate SQLite database; nothing in AlphaDEX reads "
        "that database yet, so its summary is the only output.",
        "Read-only on audio.",
    )
    OPEN_LABEL = "Open Scan Engine…"

    def _make_dialog(self) -> QtWidgets.QDialog:
        from gui.dialogs.scan_engine_dialog import ScanEngineDialog
        return ScanEngineDialog(self._library_path, self)


class BucketingWorkspace(LauncherWorkspace):
    TITLE = "Bucketing"
    HEADLINE = "Duplicate buckets report"
    EXPERIMENTAL = True
    SUMMARY = "Groups a folder's tracks into duplicate buckets and writes an HTML report."
    DETAILS = (
        "The proof of concept for the bucketing that Find duplicates' planner now does "
        "internally.",
        "Writes its report to Docs/; read-only on audio.",
    )
    OPEN_LABEL = "Open Bucketing…"

    def _make_dialog(self) -> QtWidgets.QDialog:
        from gui.dialogs.bucketing_poc_dialog import BucketingPocDialog
        return BucketingPocDialog(self._library_path, self)
