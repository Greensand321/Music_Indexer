"""Tools workspace — export utilities, diagnostics, and debug tools."""
from __future__ import annotations

import os
import re
import webbrowser
from collections import Counter
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

_SUPPORTED_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"}


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
                    self.log_line.emit(f"! Conflict: {filename} → {new_name} already exists")
                    conflicts += 1
                    continue
                try:
                    os.rename(src, dst)
                    self.log_line.emit(f"→ {filename} → {new_name}")
                    rename_map[src] = dst
                    renamed += 1
                except OSError as exc:
                    self.log_line.emit(f"! Error renaming {filename}: {exc}")
                    errors += 1

        if rename_map:
            try:
                from playlist_generator import update_playlists
                update_playlists(rename_map)
                self.log_line.emit("✓ Updated playlists")
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

    # ── Tag cleaning (mirrors _clean_tag_text from the legacy GUI) ────────

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
            # mutagen often returns lists — take the first non-empty item
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

        # ── Collect audio files ───────────────────────────────────────────
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

        # ── Read tags and build raw entry data ────────────────────────────
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
            title = self._clean(tags.get("title"))
            album = self._clean(tags.get("album"))
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

        # ── Build entries with optional album-duplicate disambiguation ────
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

        # ── Write output ──────────────────────────────────────────────────
        out = Path(self.library_path) / "Docs" / "artist_title_list.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(entries), encoding="utf-8")

        self.finished.emit(str(out), len(entries), error_count)


class ToolsWorkspace(WorkspaceBase):
    """All export, diagnostic, and utility tools in one place."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._cleanup_worker: FileCleanupWorker | None = None
        self._at_worker: ArtistTitleWorker | None = None
        self._at_open_connected = False
        self._codec_open_connected = False
        # Tab bar state
        self._tab_btns: list[QtWidgets.QPushButton] = []
        self._stacked: QtWidgets.QStackedWidget | None = None
        # Collapsible progress cards (start at maxHeight=0)
        self._at_prog_card: QtWidgets.QWidget | None = None
        self._cleanup_prog_card: QtWidgets.QWidget | None = None
        # Codec chip checkboxes for theme-refresh
        self._codec_chips: dict[str, QtWidgets.QCheckBox] = {}
        self._build_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        from gui.themes.animations import AnimatedTabButton

        cl = self.content_layout
        cl.addWidget(self._make_section_title("Export & Utilities"))
        cl.addWidget(self._make_subtitle(
            "Export reports, run diagnostics, and validate your library layout. "
            "All export files land in Docs/ inside your library folder."
        ))

        # ── Animated tab bar ──────────────────────────────────────────────
        tab_card = self._make_card()
        tab_row = QtWidgets.QHBoxLayout(tab_card)
        tab_row.setContentsMargins(10, 8, 10, 8)
        tab_row.setSpacing(4)

        for i, label in enumerate([
            "Artist / Title", "Codec Export", "File Cleanup", "Diagnostics", "Validator",
        ]):
            btn = AnimatedTabButton(label)
            btn.setAutoExclusive(False)   # we manage exclusive state manually
            btn.clicked.connect(lambda _checked=False, idx=i: self._switch_tab(idx))
            tab_row.addWidget(btn)
            self._tab_btns.append(btn)
        tab_row.addStretch(1)
        cl.addWidget(tab_card)

        # ── Stacked pages ─────────────────────────────────────────────────
        self._stacked = QtWidgets.QStackedWidget()
        self._stacked.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._stacked.addWidget(self._build_at_page())
        self._stacked.addWidget(self._build_codec_page())
        self._stacked.addWidget(self._build_cleanup_page())
        self._stacked.addWidget(self._build_diag_page())
        self._stacked.addWidget(self._build_validator_page())
        cl.addWidget(self._stacked, 1)

        cl.addStretch(1)

        # Activate first tab without animation
        self._switch_tab(0, animated=False)

    # ── Tab navigation ────────────────────────────────────────────────────

    def _switch_tab(self, index: int, animated: bool = True) -> None:
        """Switch the visible page; crossfade when animated=True."""
        # Keep checked state in sync
        for i, btn in enumerate(self._tab_btns):
            btn.setChecked(i == index)

        if self._stacked is None:
            return

        if not animated:
            self._stacked.setCurrentIndex(index)
            return

        if index == self._stacked.currentIndex():
            return

        outgoing = self._stacked.currentWidget()

        # Fade out the outgoing page
        fx_out = QtWidgets.QGraphicsOpacityEffect(outgoing)
        outgoing.setGraphicsEffect(fx_out)
        anim_out = QtCore.QPropertyAnimation(fx_out, b"opacity", self)
        anim_out.setDuration(100)
        anim_out.setStartValue(1.0)
        anim_out.setEndValue(0.0)

        def _show_incoming() -> None:
            outgoing.setGraphicsEffect(None)
            self._stacked.setCurrentIndex(index)
            incoming = self._stacked.currentWidget()
            fx_in = QtWidgets.QGraphicsOpacityEffect(incoming)
            incoming.setGraphicsEffect(fx_in)
            fx_in.setOpacity(0.0)
            anim_in = QtCore.QPropertyAnimation(fx_in, b"opacity", self)
            anim_in.setDuration(180)
            anim_in.setStartValue(0.0)
            anim_in.setEndValue(1.0)
            anim_in.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            anim_in.finished.connect(lambda: incoming.setGraphicsEffect(None))
            anim_in.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

        anim_out.finished.connect(_show_incoming)
        anim_out.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    # ── Page builders ─────────────────────────────────────────────────────

    def _build_at_page(self) -> QtWidgets.QWidget:
        from gui.themes.animations import AnimatedButton

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        # Options card
        opts_card = self._make_card()
        opts_l = QtWidgets.QVBoxLayout(opts_card)
        opts_l.setContentsMargins(16, 14, 16, 14)
        opts_l.setSpacing(10)
        opts_l.addWidget(self._make_card_title("Options"))
        opts_l.addWidget(self._make_subtitle(
            "Scan the library and write Docs/artist_title_list.txt "
            "with every Artist – Title pair."
        ))
        opts_row = QtWidgets.QHBoxLayout()
        opts_row.setSpacing(20)
        self._exclude_flac_cb = QtWidgets.QCheckBox("Exclude FLAC files")
        self._dupe_tracks_cb = QtWidgets.QCheckBox("Include per-album duplicate titles")
        opts_row.addWidget(self._exclude_flac_cb)
        opts_row.addWidget(self._dupe_tracks_cb)
        opts_row.addStretch(1)
        opts_l.addLayout(opts_row)
        layout.addWidget(opts_card)

        # Action card
        action_card = self._make_card()
        action_l = QtWidgets.QHBoxLayout(action_card)
        action_l.setContentsMargins(16, 12, 16, 12)
        action_l.setSpacing(8)
        at_run = AnimatedButton("Export Artist / Title List")
        at_run.setObjectName("primaryBtn")
        at_run.setMinimumHeight(34)
        at_run.clicked.connect(self._on_export_at)
        self._at_open = AnimatedButton("Open File")
        self._at_open.setMinimumHeight(34)
        self._at_open.setEnabled(False)
        action_l.addWidget(at_run)
        action_l.addWidget(self._at_open)
        action_l.addStretch(1)
        layout.addWidget(action_card)

        # Progress card — starts collapsed (maxHeight=0)
        prog_card = self._make_card()
        prog_l = QtWidgets.QVBoxLayout(prog_card)
        prog_l.setContentsMargins(16, 12, 16, 12)
        prog_l.setSpacing(6)
        self._at_prog = QtWidgets.QProgressBar()
        self._at_prog.setFixedHeight(8)
        self._at_prog.setTextVisible(False)
        self._at_status = QtWidgets.QLabel("Ready.")
        self._at_status.setObjectName("statusLabel")
        self._at_log = QtWidgets.QPlainTextEdit()
        self._at_log.setReadOnly(True)
        self._at_log.setFixedHeight(110)
        self._at_log.setObjectName("logBox")
        prog_l.addWidget(self._at_prog)
        prog_l.addWidget(self._at_status)
        prog_l.addWidget(self._at_log)
        self._at_prog_card = prog_card
        prog_card.setMinimumHeight(0)
        prog_card.setMaximumHeight(0)
        layout.addWidget(prog_card)

        layout.addStretch(1)
        return page

    def _build_codec_page(self) -> QtWidgets.QWidget:
        from gui.themes.animations import AnimatedButton

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        # Codec chips card
        chips_card = self._make_card()
        chips_l = QtWidgets.QVBoxLayout(chips_card)
        chips_l.setContentsMargins(16, 14, 16, 14)
        chips_l.setSpacing(10)
        chips_l.addWidget(self._make_card_title("Include Codecs"))
        chips_l.addWidget(self._make_subtitle(
            "Export a grouped list of all tracks by codec format."
        ))
        chips_row = QtWidgets.QHBoxLayout()
        chips_row.setSpacing(6)
        self._codec_ext_cbs: dict[str, QtWidgets.QCheckBox] = {}
        for ext in (".flac", ".mp3", ".m4a", ".aac", ".wav", ".opus", ".ogg"):
            cb = self._make_chip(ext)
            self._codec_ext_cbs[ext] = cb
            chips_row.addWidget(cb)
        chips_row.addStretch(1)
        chips_l.addLayout(chips_row)
        self._omit_paths_cb = QtWidgets.QCheckBox("Filenames only (no full paths)")
        chips_l.addWidget(self._omit_paths_cb)
        layout.addWidget(chips_card)

        # Action card
        action_card = self._make_card()
        action_l = QtWidgets.QHBoxLayout(action_card)
        action_l.setContentsMargins(16, 12, 16, 12)
        action_l.setSpacing(8)
        codec_run = AnimatedButton("Export Codec List")
        codec_run.setObjectName("primaryBtn")
        codec_run.setMinimumHeight(34)
        codec_run.clicked.connect(self._on_export_codec)
        self._codec_open = AnimatedButton("Open File")
        self._codec_open.setMinimumHeight(34)
        self._codec_open.setEnabled(False)
        action_l.addWidget(codec_run)
        action_l.addWidget(self._codec_open)
        action_l.addStretch(1)
        layout.addWidget(action_card)

        # Status card (always visible — codec export runs synchronously)
        status_card = self._make_card()
        status_l = QtWidgets.QVBoxLayout(status_card)
        status_l.setContentsMargins(16, 10, 16, 10)
        status_l.setSpacing(6)
        self._codec_prog = QtWidgets.QProgressBar()
        self._codec_prog.setFixedHeight(8)
        self._codec_prog.setTextVisible(False)
        self._codec_prog.setValue(0)
        self._codec_status = QtWidgets.QLabel("Ready.")
        self._codec_status.setObjectName("statusLabel")
        status_l.addWidget(self._codec_prog)
        status_l.addWidget(self._codec_status)
        layout.addWidget(status_card)

        layout.addStretch(1)
        return page

    def _build_cleanup_page(self) -> QtWidgets.QWidget:
        from gui.themes.animations import AnimatedButton

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        # Info card
        info_card = self._make_card()
        info_l = QtWidgets.QVBoxLayout(info_card)
        info_l.setContentsMargins(16, 14, 16, 14)
        info_l.setSpacing(6)
        info_l.addWidget(self._make_card_title("File Cleanup"))
        info_l.addWidget(self._make_subtitle(
            "Remove trailing \u2018 (1)\u2019, \u2018 (2)\u2019, \u2018 copy\u2019 suffixes "
            "from audio filenames left by macOS Finder and Windows Explorer. "
            "Playlists referencing renamed files are updated automatically."
        ))
        layout.addWidget(info_card)

        # Action card
        action_card = self._make_card()
        action_l = QtWidgets.QHBoxLayout(action_card)
        action_l.setContentsMargins(16, 12, 16, 12)
        self._cleanup_run_btn = AnimatedButton("Run File Cleanup")
        self._cleanup_run_btn.setObjectName("primaryBtn")
        self._cleanup_run_btn.setMinimumHeight(34)
        self._cleanup_run_btn.clicked.connect(self._on_file_cleanup)
        action_l.addWidget(self._cleanup_run_btn)
        action_l.addStretch(1)
        layout.addWidget(action_card)

        # Progress card — starts collapsed
        prog_card = self._make_card()
        prog_l = QtWidgets.QVBoxLayout(prog_card)
        prog_l.setContentsMargins(16, 12, 16, 12)
        prog_l.setSpacing(6)
        self._cleanup_status = QtWidgets.QLabel("Ready.")
        self._cleanup_status.setObjectName("statusLabel")
        self._cleanup_log = QtWidgets.QPlainTextEdit()
        self._cleanup_log.setReadOnly(True)
        self._cleanup_log.setFixedHeight(130)
        self._cleanup_log.setObjectName("logBox")
        prog_l.addWidget(self._cleanup_status)
        prog_l.addWidget(self._cleanup_log)
        self._cleanup_prog_card = prog_card
        prog_card.setMinimumHeight(0)
        prog_card.setMaximumHeight(0)
        layout.addWidget(prog_card)

        layout.addStretch(1)
        return page

    def _build_diag_page(self) -> QtWidgets.QWidget:
        from gui.themes.animations import AnimatedButton

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        # Media testing card
        media_card = self._make_card()
        media_l = QtWidgets.QVBoxLayout(media_card)
        media_l.setContentsMargins(16, 14, 16, 14)
        media_l.setSpacing(10)
        media_l.addWidget(self._make_card_title("Media Testing"))
        media_l.addWidget(self._make_subtitle(
            "Open codec-specific test dialogs to verify playback and metadata reading."
        ))
        media_btn_row = QtWidgets.QHBoxLayout()
        media_btn_row.setSpacing(8)
        for label, slot in (
            ("M4A Tester\u2026", self._on_m4a_tester),
            ("Opus Tester\u2026", self._on_opus_tester),
        ):
            btn = AnimatedButton(label)
            btn.setMinimumHeight(34)
            btn.clicked.connect(slot)
            media_btn_row.addWidget(btn)
        media_btn_row.addStretch(1)
        media_l.addLayout(media_btn_row)
        layout.addWidget(media_card)

        # Duplicate tools card
        dupe_card = self._make_card()
        dupe_l = QtWidgets.QVBoxLayout(dupe_card)
        dupe_l.setContentsMargins(16, 14, 16, 14)
        dupe_l.setSpacing(10)
        dupe_l.addWidget(self._make_card_title("Duplicate Tools"))
        dupe_l.addWidget(self._make_subtitle(
            "Advanced tools for inspecting and resolving duplicate detection results."
        ))
        dupe_btn_row = QtWidgets.QHBoxLayout()
        dupe_btn_row.setSpacing(8)
        for label, slot in (
            ("Bucketing POC\u2026", self._on_bucketing_poc),
            ("Scan Engine\u2026", self._on_scan_engine),
            ("Fuzzy Finder\u2026", self._on_fuzzy_dupes),
            ("Pair Review\u2026", self._on_pair_review),
        ):
            btn = AnimatedButton(label)
            btn.setMinimumHeight(34)
            btn.clicked.connect(slot)
            dupe_btn_row.addWidget(btn)
        dupe_btn_row.addStretch(1)
        dupe_l.addLayout(dupe_btn_row)
        layout.addWidget(dupe_card)

        # System card
        sys_card = self._make_card()
        sys_l = QtWidgets.QVBoxLayout(sys_card)
        sys_l.setContentsMargins(16, 14, 16, 14)
        sys_l.setSpacing(10)
        sys_l.addWidget(self._make_card_title("System"))
        sys_btn_row = QtWidgets.QHBoxLayout()
        crash_btn = AnimatedButton("View Crash Log\u2026")
        crash_btn.setMinimumHeight(34)
        crash_btn.clicked.connect(self._on_crash_log)
        sys_btn_row.addWidget(crash_btn)
        sys_btn_row.addStretch(1)
        sys_l.addLayout(sys_btn_row)
        layout.addWidget(sys_card)

        layout.addStretch(1)
        return page

    def _build_validator_page(self) -> QtWidgets.QWidget:
        from gui.themes.animations import AnimatedButton

        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)

        # Action card
        action_card = self._make_card()
        action_l = QtWidgets.QVBoxLayout(action_card)
        action_l.setContentsMargins(16, 14, 16, 14)
        action_l.setSpacing(10)
        action_l.addWidget(self._make_card_title("Library Validator"))
        action_l.addWidget(self._make_subtitle(
            "Verify that your library folder layout matches AlphaDEX conventions."
        ))
        val_run = AnimatedButton("Run Validator")
        val_run.setObjectName("primaryBtn")
        val_run.setMinimumHeight(34)
        val_run.clicked.connect(self._on_validate)
        action_l.addWidget(val_run)
        layout.addWidget(action_card)

        # Results card
        results_card = self._make_card()
        results_l = QtWidgets.QVBoxLayout(results_card)
        results_l.setContentsMargins(16, 12, 16, 12)
        results_l.setSpacing(6)
        results_l.addWidget(self._make_card_title("Results"))
        self._val_log = QtWidgets.QPlainTextEdit()
        self._val_log.setReadOnly(True)
        self._val_log.setMinimumHeight(180)
        self._val_log.setObjectName("logBox")
        results_l.addWidget(self._val_log)
        layout.addWidget(results_card)

        layout.addStretch(1)
        return page

    # ── UI helpers ────────────────────────────────────────────────────────

    def _make_chip(self, label: str) -> QtWidgets.QCheckBox:
        """Create a toggleable codec chip that re-styles itself on state change."""
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
                f" background: {t.accent}28;"
                f" border: 1px solid {t.accent};"
                f" border-radius: 10px;"
                f" padding: 3px 10px;"
                f" color: {t.accent};"
                f" font-size: 12px;"
                f"}}"
                f"QCheckBox::indicator {{ width: 0; height: 0; }}"
            )
        else:
            style = (
                f"QCheckBox {{"
                f" background: transparent;"
                f" border: 1px solid {t.card_border};"
                f" border-radius: 10px;"
                f" padding: 3px 10px;"
                f" color: {t.text_secondary};"
                f" font-size: 12px;"
                f"}}"
                f"QCheckBox::indicator {{ width: 0; height: 0; }}"
            )
        cb.setStyleSheet(style)

    def _reveal_card(self, card: QtWidgets.QWidget) -> None:
        """Animate a collapsed card from maxHeight=0 to its natural size."""
        if card.maximumHeight() > 0:
            return  # already open
        target = max(card.sizeHint().height(), 80)
        anim = QtCore.QPropertyAnimation(card, b"maximumHeight", self)
        anim.setDuration(220)
        anim.setStartValue(0)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        # After animation ends, lift the constraint so the card can resize freely
        anim.finished.connect(lambda: card.setMaximumHeight(16777215))
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _flash_label(
        self,
        label: QtWidgets.QLabel,
        from_hex: str,
        duration: int = 1600,
    ) -> None:
        """Animate label color from from_hex → theme text_secondary."""
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

    # ── Theme refresh ─────────────────────────────────────────────────────

    def _on_theme_changed_base(self, tokens: object) -> None:
        super()._on_theme_changed_base(tokens)
        for cb in self._codec_chips.values():
            self._refresh_chip(cb)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_export_at(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return
        if self._at_worker and self._at_worker.isRunning():
            return

        exclude_flac = self._exclude_flac_cb.isChecked()
        add_dupes = self._dupe_tracks_cb.isChecked()

        self._at_log.clear()
        self._at_status.setText("Scanning\u2026")
        self._at_status.setStyleSheet("color: inherit; font-size: 12px;")
        self._at_prog.setValue(0)
        self._at_open.setEnabled(False)
        if self._at_prog_card is not None:
            self._reveal_card(self._at_prog_card)

        self._at_worker = ArtistTitleWorker(
            self._library_path, exclude_flac, add_dupes, parent=self
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

        # Flash status green → neutral
        t = get_manager().current
        self._flash_label(self._at_status, t.success)

        # Reconnect Open button (disconnect old first)
        if self._at_open_connected:
            try:
                self._at_open.clicked.disconnect()
            except RuntimeError:
                pass
        self._at_open.clicked.connect(lambda: self._open_file(out_path))
        self._at_open_connected = True
        self._at_open.setEnabled(True)

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
            # Flash status green → neutral
            t = get_manager().current
            self._flash_label(self._codec_status, t.success)
            # Wire Open File (disconnect previous connection first)
            if self._codec_open_connected:
                try:
                    self._codec_open.clicked.disconnect()
                except RuntimeError:
                    pass
            self._codec_open.clicked.connect(lambda: self._open_file(str(out)))
            self._codec_open_connected = True
            self._codec_open.setEnabled(True)
            self._log(f"Codec list exported: {total} files \u2192 {out}", "ok")
        except Exception as exc:  # noqa: BLE001
            t = get_manager().current
            self._codec_status.setText(str(exc))
            self._flash_label(self._codec_status, t.danger)
            self._log(str(exc), "error")

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
        if self._cleanup_prog_card is not None:
            self._reveal_card(self._cleanup_prog_card)
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
        self._log("Duplicate Bucketing POC — open from Tools menu or run directly.", "info")

    @Slot()
    def _on_scan_engine(self) -> None:
        self._log("Duplicate Scan Engine — not yet wired to Qt dialog.", "info")

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
        self._log("Duplicate Pair Review — launch from Fuzzy Duplicate Finder results.", "info")

    @Slot()
    def _on_validate(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return
        try:
            import validator
            result = validator.validate(self._library_path)
            self._val_log.setPlainText(str(result))
            self._log("Validation complete.", "ok")
        except Exception as exc:  # noqa: BLE001
            self._val_log.setPlainText(str(exc))
            self._log(str(exc), "error")

    def _open_file(self, path: str) -> None:
        import subprocess
        import sys
        if sys.platform == "win32":
            subprocess.Popen(["explorer", path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
