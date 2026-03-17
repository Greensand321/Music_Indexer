"""Tools workspace — export utilities, diagnostics, and debug tools."""
from __future__ import annotations

import os
import re
import webbrowser
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
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


class ToolsWorkspace(WorkspaceBase):
    """All export, diagnostic, and utility tools in one place."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._cleanup_worker: FileCleanupWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Export & Utilities"))
        cl.addWidget(self._make_subtitle(
            "Export reports, run diagnostics, and access utility tools. "
            "All exports land in Docs/ inside your library."
        ))

        tabs = QtWidgets.QTabWidget()

        # ── Export: Artist/Title List ──────────────────────────────────────
        export_at_w = QtWidgets.QWidget()
        export_at_l = QtWidgets.QVBoxLayout(export_at_w)
        export_at_l.setContentsMargins(16, 16, 16, 16)
        export_at_l.setSpacing(10)
        export_at_l.addWidget(self._make_subtitle(
            "Scan the library and write Docs/artist_title_list.txt "
            "with every artist + title pair."
        ))
        at_opts = QtWidgets.QHBoxLayout()
        self._exclude_flac_cb = QtWidgets.QCheckBox("Exclude FLAC files")
        self._dupe_tracks_cb = QtWidgets.QCheckBox("Include per-album duplicate titles")
        at_opts.addWidget(self._exclude_flac_cb)
        at_opts.addWidget(self._dupe_tracks_cb)
        at_opts.addStretch(1)
        export_at_l.addLayout(at_opts)
        self._at_prog = QtWidgets.QProgressBar()
        self._at_prog.setFixedHeight(6)
        self._at_prog.setTextVisible(False)
        self._at_status = QtWidgets.QLabel("")
        self._at_status.setStyleSheet("color: #64748b; font-size: 12px;")
        at_btn_row = QtWidgets.QHBoxLayout()
        at_run = self._make_primary_button("Export Artist/Title List")
        at_run.clicked.connect(self._on_export_at)
        self._at_open = QtWidgets.QPushButton("Open File")
        self._at_open.setEnabled(False)
        at_btn_row.addWidget(at_run)
        at_btn_row.addWidget(self._at_open)
        at_btn_row.addStretch(1)
        export_at_l.addLayout(at_btn_row)
        export_at_l.addWidget(self._at_prog)
        export_at_l.addWidget(self._at_status)
        self._at_log = QtWidgets.QPlainTextEdit()
        self._at_log.setReadOnly(True)
        self._at_log.setFixedHeight(120)
        self._at_log.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        export_at_l.addWidget(self._at_log)
        export_at_l.addStretch(1)
        tabs.addTab(export_at_w, "Artist/Title Export")

        # ── Export: Codec List ─────────────────────────────────────────────
        export_codec_w = QtWidgets.QWidget()
        export_codec_l = QtWidgets.QVBoxLayout(export_codec_w)
        export_codec_l.setContentsMargins(16, 16, 16, 16)
        export_codec_l.setSpacing(10)
        export_codec_l.addWidget(self._make_subtitle(
            "Export a list of all tracks grouped by codec."
        ))
        codec_opts = QtWidgets.QHBoxLayout()
        codec_opts.addWidget(QtWidgets.QLabel("Include codecs:"))
        self._codec_ext_cbs: dict[str, QtWidgets.QCheckBox] = {}
        for ext in (".flac", ".mp3", ".m4a", ".aac", ".wav", ".opus", ".ogg"):
            cb = QtWidgets.QCheckBox(ext)
            cb.setChecked(True)
            codec_opts.addWidget(cb)
            self._codec_ext_cbs[ext] = cb
        codec_opts.addStretch(1)
        export_codec_l.addLayout(codec_opts)
        self._omit_paths_cb = QtWidgets.QCheckBox("Filenames only (no full paths)")
        export_codec_l.addWidget(self._omit_paths_cb)
        self._codec_prog = QtWidgets.QProgressBar()
        self._codec_prog.setFixedHeight(6)
        self._codec_prog.setTextVisible(False)
        self._codec_status = QtWidgets.QLabel("")
        self._codec_status.setStyleSheet("color: #64748b; font-size: 12px;")
        codec_btn_row = QtWidgets.QHBoxLayout()
        codec_run = self._make_primary_button("Export Codec List")
        codec_run.clicked.connect(self._on_export_codec)
        self._codec_open = QtWidgets.QPushButton("Open File")
        self._codec_open.setEnabled(False)
        codec_btn_row.addWidget(codec_run)
        codec_btn_row.addWidget(self._codec_open)
        codec_btn_row.addStretch(1)
        export_codec_l.addLayout(codec_btn_row)
        export_codec_l.addWidget(self._codec_prog)
        export_codec_l.addWidget(self._codec_status)
        export_codec_l.addStretch(1)
        tabs.addTab(export_codec_w, "Codec List Export")

        # ── File Cleanup ───────────────────────────────────────────────────
        cleanup_w = QtWidgets.QWidget()
        cleanup_l = QtWidgets.QVBoxLayout(cleanup_w)
        cleanup_l.setContentsMargins(16, 16, 16, 16)
        cleanup_l.setSpacing(10)
        cleanup_l.addWidget(self._make_subtitle(
            "Remove trailing ' (1)', ' (2)', ' copy' suffixes from filenames "
            "left by macOS/Windows duplicate handling."
        ))
        self._cleanup_status = QtWidgets.QLabel("Ready to scan.")
        self._cleanup_status.setStyleSheet("color: #64748b;")
        cleanup_l.addWidget(self._cleanup_status)
        cleanup_btn_row = QtWidgets.QHBoxLayout()
        self._cleanup_run_btn = self._make_primary_button("Run File Cleanup")
        self._cleanup_run_btn.clicked.connect(self._on_file_cleanup)
        cleanup_btn_row.addWidget(self._cleanup_run_btn)
        cleanup_btn_row.addStretch(1)
        cleanup_l.addLayout(cleanup_btn_row)
        self._cleanup_log = QtWidgets.QPlainTextEdit()
        self._cleanup_log.setReadOnly(True)
        self._cleanup_log.setFixedHeight(140)
        self._cleanup_log.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        cleanup_l.addWidget(self._cleanup_log)
        cleanup_l.addStretch(1)
        tabs.addTab(cleanup_w, "File Cleanup")

        # ── Diagnostics ────────────────────────────────────────────────────
        diag_w = QtWidgets.QWidget()
        diag_l = QtWidgets.QVBoxLayout(diag_w)
        diag_l.setContentsMargins(16, 16, 16, 16)
        diag_l.setSpacing(12)
        diag_l.addWidget(self._make_subtitle("Diagnostic and testing utilities."))

        for label, slot in (
            ("M4A Tester…", self._on_m4a_tester),
            ("Opus Tester…", self._on_opus_tester),
            ("Duplicate Bucketing POC…", self._on_bucketing_poc),
            ("Duplicate Scan Engine…", self._on_scan_engine),
            ("View Crash Log…", self._on_crash_log),
            ("Fuzzy Duplicate Finder…", self._on_fuzzy_dupes),
            ("Duplicate Pair Review…", self._on_pair_review),
        ):
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(slot)
            diag_l.addWidget(btn)

        diag_l.addStretch(1)
        tabs.addTab(diag_w, "Diagnostics")

        # ── Validator ──────────────────────────────────────────────────────
        val_w = QtWidgets.QWidget()
        val_l = QtWidgets.QVBoxLayout(val_w)
        val_l.setContentsMargins(16, 16, 16, 16)
        val_l.setSpacing(10)
        val_l.addWidget(self._make_subtitle(
            "Verify that the library folder layout matches AlphaDEX conventions."
        ))
        val_run = self._make_primary_button("Run Validator")
        val_run.clicked.connect(self._on_validate)
        val_l.addWidget(val_run)
        self._val_log = QtWidgets.QPlainTextEdit()
        self._val_log.setReadOnly(True)
        self._val_log.setMinimumHeight(180)
        self._val_log.setStyleSheet("font-family: 'Consolas', monospace; font-size: 11px;")
        val_l.addWidget(self._val_log)
        val_l.addStretch(1)
        tabs.addTab(val_w, "Validator")

        cl.addWidget(tabs)
        cl.addStretch(1)

    # ── Slots ─────────────────────────────────────────────────────────────

    @Slot()
    def _on_export_at(self) -> None:
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Select a library folder first.")
            return
        self._log("Starting artist/title export…", "info")
        try:
            from music_indexer_api import build_primary_counts
            counts = build_primary_counts(self._library_path)
            out = Path(self._library_path) / "Docs" / "artist_title_list.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"{a}\t{t}" for a, titles in counts.items() for t in titles]
            out.write_text("\n".join(lines))
            self._at_status.setText(f"Written: {out}")
            self._at_log.appendPlainText(f"Exported {len(lines)} tracks to {out}")
            self._at_prog.setValue(100)
            self._at_open.setEnabled(True)
            self._at_open.clicked.connect(lambda: self._open_file(str(out)))
            self._log("Artist/title export complete.", "ok")
        except Exception as exc:  # noqa: BLE001
            self._at_status.setText(str(exc))
            self._log(str(exc), "error")

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
        self._codec_status.setText("Scanning…")
        self._log("Starting codec list export…", "info")
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
            self._codec_open.setEnabled(True)
            self._codec_open.clicked.connect(lambda: self._open_file(str(out)))
            self._log(f"Codec list exported: {total} files → {out}", "ok")
        except Exception as exc:  # noqa: BLE001
            self._codec_status.setText(str(exc))
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
        self._cleanup_status.setText("Running…")
        self._cleanup_run_btn.setEnabled(False)
        self._log("Starting file cleanup…", "info")
        self._cleanup_worker = FileCleanupWorker(self._library_path)
        self._cleanup_worker.log_line.connect(self._cleanup_log.appendPlainText)
        self._cleanup_worker.finished.connect(self._on_cleanup_finished)
        self._cleanup_worker.start()

    @Slot(bool, str)
    def _on_cleanup_finished(self, success: bool, message: str) -> None:
        self._cleanup_status.setText(message)
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
        import subprocess, sys
        if sys.platform == "win32":
            subprocess.Popen(["explorer", path])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
