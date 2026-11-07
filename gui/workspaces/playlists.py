"""Playlist Generator workspace — m3u playlists, Auto-DJ, folder playlists.

Every mode is preview-first: a background job works out exactly which
``.m3u`` files would be written (and which existing ones overwritten), the
user confirms that list, and only then is exactly that plan written, with a
timestamped ``.bak`` of every file it overwrites (see
``playlist_generator.write_playlist_plan``).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from gui.compat import QtCore, QtGui, QtWidgets, Signal, Slot
from gui.themes.manager import get_manager
from gui.workspaces.base import WorkspaceBase

_AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"}

#: How many planned files the confirmation lists before summarising the rest.
_CONFIRM_LIST_LIMIT = 12


def _find_audio_files(root: str) -> list[str]:
    """Library tracks, skipping the reserved folders (Docs, Quarantine, Trash…)."""
    from utils.reserved_folders import iter_library_files

    return sorted(str(p) for p in iter_library_files(root, _AUDIO_EXTS))


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class PlaylistWorker(QtCore.QThread):
    """Plans (``mode`` = folder/tempo/autodj/repair) or writes (``mode`` = write)."""

    log_line = Signal(str)
    # (success, message, payload). payload is the list of PlannedPlaylist for
    # the planning modes and the write_playlist_plan() summary for "write".
    # Deliberately not called ``finished``: that would shadow QThread.finished.
    done = Signal(bool, str, object)

    def __init__(
        self,
        library_path: str,
        mode: str,
        output_dir: str = "",
        tempo_range: tuple | None = None,
        energy_range: tuple | None = None,
        autodj_count: int = 20,
        seed_track: str = "",
        repair_paths: list | None = None,
        prefer_opus: bool = False,
        plan: list | None = None,
    ) -> None:
        super().__init__()
        self.library_path = library_path
        self.mode = mode
        self.output_dir = output_dir
        self.tempo_range = tempo_range
        self.energy_range = energy_range
        self.autodj_count = autodj_count
        self.seed_track = seed_track
        self.repair_paths = repair_paths or []
        self.prefer_opus = prefer_opus
        self.plan = plan or []
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set() or self.isInterruptionRequested()

    def _log(self, msg: str) -> None:
        self.log_line.emit(msg)

    def run(self) -> None:
        try:
            import playlist_generator as pg
        except ImportError as exc:
            self.done.emit(False, f"Import error: {exc}", [])
            return
        try:
            if self.mode == "write":
                self._run_write(pg)
                return
            plan = self._run_plan(pg)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "IndexCancelled" or self.is_cancelled():
                self.done.emit(False, "Cancelled.", [])
            else:
                self.done.emit(False, str(exc) or type(exc).__name__, [])
            return
        if plan is None:
            return  # done already emitted
        if self.is_cancelled():
            self.done.emit(False, "Cancelled.", [])
        else:
            self.done.emit(True, "Preview ready.", plan)

    # ── planning (no file is written) ────────────────────────────────────

    def _run_plan(self, pg):
        if self.mode == "folder":
            plan = pg.plan_folder_playlists(self.library_path, output_dir=self.output_dir or None)
            self._log(f"Found {len(plan)} album folder(s) with audio files.")
            return plan

        if self.mode == "tempo":
            from playlist_engine import bucket_by_tempo_energy

            tracks = _find_audio_files(self.library_path)
            self._log(f"Found {len(tracks)} tracks — analyzing tempo/energy…")
            result = bucket_by_tempo_energy(
                tracks,
                self.library_path,
                log_callback=self._log,
                cancel_event=self._cancel_event,
                tempo_range=self.tempo_range,
                energy_range=self.energy_range,
            )
            if result.get("excluded"):
                self._log(f"{result['excluded']} track(s) fell outside the tempo/energy range.")
            paths = result.get("playlist_paths", {})
            return [
                pg.plan_playlist(items, paths[key])
                for key, items in sorted(result.get("buckets", {}).items())
                if items and key in paths
            ]

        if self.mode == "autodj":
            from playlist_engine import autodj_playlist

            tracks = _find_audio_files(self.library_path)
            if not tracks:
                self.done.emit(False, "No audio files found in the library.", [])
                return None
            seed = next((t for t in tracks if _same_path(t, self.seed_track)), None)
            if seed is None:
                self.done.emit(
                    False,
                    f"Seed track is not in the library (or is in a reserved folder): {self.seed_track}",
                    [],
                )
                return None
            self._log(f"Building Auto-DJ playlist from {Path(seed).name}…")
            ordered = autodj_playlist(
                seed,
                tracks,
                n=self.autodj_count,
                log_callback=self._log,
                cancel_event=self._cancel_event,
            )
            out_path = os.path.join(self.library_path, "Playlists", "autodj.m3u")
            return [pg.plan_playlist(ordered, out_path)]

        if self.mode == "repair":
            paths = list(self.repair_paths)
            if not paths:
                pl_dir = Path(self.library_path) / "Playlists"
                paths = sorted(
                    str(p) for p in pl_dir.rglob("*")
                    if p.suffix.lower() in (".m3u", ".m3u8") and p.is_file()
                ) if pl_dir.is_dir() else []
                self._log(f"Checking the {len(paths)} playlist(s) in {pl_dir}…")
            plan = pg.plan_playlist_repair(
                paths, self.library_path, prefer_opus=self.prefer_opus, log_callback=self._log
            )
            for item in plan:
                name = Path(item.path).name
                if item.error:
                    self._log(f"! {name}: could not be read ({item.error})")
                    continue
                self._log(
                    f"{name}: {item.entries} entries, {len(item.changes)} fixable, "
                    f"{len(item.unresolved)} still missing"
                )
                for old, new in item.changes:
                    self._log(f"   {old}  →  {new}")
                for entry in item.unresolved:
                    self._log(f"   ! not found: {entry}")
            return plan

        raise ValueError(f"Unknown playlist mode: {self.mode}")

    # ── writing exactly the confirmed plan ───────────────────────────────

    def _run_write(self, pg) -> None:
        result = pg.write_playlist_plan(
            self.plan,
            backup=True,
            log_callback=self._log,
            log_dir=os.path.join(self.library_path, "Docs"),
        )
        n, errors = len(result["written"]), result["errors"]
        msg = f"Wrote {n} playlist(s)"
        if result["backups"]:
            msg += f", backed up {len(result['backups'])} overwritten file(s) as .bak"
        if errors:
            msg += f"; {len(errors)} failed"
        self.done.emit(not errors, msg + ".", result)


class PlaylistsWorkspace(WorkspaceBase):
    """Generate m3u playlists by folder, tempo/energy, or Auto-DJ."""

    def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(library_path, parent)
        self._worker: PlaylistWorker | None = None
        self._run_buttons: list[QtWidgets.QPushButton] = []
        self._build_ui()
        self._set_busy(False)

    def _build_ui(self) -> None:
        cl = self.content_layout

        cl.addWidget(self._make_section_title("Playlist Generator"))
        cl.addWidget(self._make_subtitle(
            "Build .m3u playlists from your library by folder structure, "
            "tempo/energy buckets, or Auto-DJ flow. "
            "You see the list of playlists that would be written (and which "
            "existing ones would be replaced) before anything is saved; replaced "
            "playlists are backed up as .bak files first."
        ))

        # ── Tabs for playlist modes ────────────────────────────────────────
        tabs = QtWidgets.QTabWidget()

        # ── Folder Playlists tab ───────────────────────────────────────────
        folder_widget = QtWidgets.QWidget()
        folder_layout = QtWidgets.QVBoxLayout(folder_widget)
        folder_layout.setContentsMargins(16, 16, 16, 16)
        folder_layout.setSpacing(10)

        folder_layout.addWidget(self._make_subtitle(
            "Create one playlist per album folder (every folder that directly "
            "contains audio files), named after its path, e.g. Artist_Album.m3u. "
            "The library's own folders (Docs, Playlists, Trash, Quarantine, "
            "Not Sorted, Manual Review) are skipped."
        ))

        self._folder_out_entry = QtWidgets.QLineEdit()
        self._folder_out_entry.setPlaceholderText("Leave blank to use Playlists/ inside library")
        folder_out_row = QtWidgets.QHBoxLayout()
        folder_out_row.addWidget(QtWidgets.QLabel("Output folder:"))
        folder_out_row.addWidget(self._folder_out_entry, 1)
        browse_folder = QtWidgets.QPushButton("Browse…")
        browse_folder.clicked.connect(lambda: self._browse_dir(self._folder_out_entry))
        folder_out_row.addWidget(browse_folder)
        folder_layout.addLayout(folder_out_row)

        folder_btn = self._make_primary_button("🎵  Generate Folder Playlists")
        folder_btn.clicked.connect(lambda: self._on_run("folder"))
        self._run_buttons.append(folder_btn)
        folder_layout.addWidget(folder_btn)
        folder_layout.addStretch(1)
        tabs.addTab(folder_widget, "Folder Playlists")

        # ── Tempo/Energy Playlists tab ─────────────────────────────────────
        tempo_widget = QtWidgets.QWidget()
        tempo_layout = QtWidgets.QVBoxLayout(tempo_widget)
        tempo_layout.setContentsMargins(16, 16, 16, 16)
        tempo_layout.setSpacing(10)

        tempo_layout.addWidget(self._make_subtitle(
            "Group tracks into playlists by BPM and energy level "
            "(Playlists/slow_low.m3u … fast_high.m3u). Tracks outside the ranges "
            "below are left out. Uses librosa for feature extraction — may take a "
            "while on large libraries."
        ))

        tform = QtWidgets.QFormLayout()
        tform.setSpacing(8)

        tempo_range_row = QtWidgets.QHBoxLayout()
        self._tempo_min = QtWidgets.QSpinBox()
        self._tempo_min.setRange(0, 300)
        self._tempo_min.setValue(0)
        self._tempo_max = QtWidgets.QSpinBox()
        self._tempo_max.setRange(0, 300)
        self._tempo_max.setValue(300)
        tempo_range_row.addWidget(self._tempo_min)
        tempo_range_row.addWidget(QtWidgets.QLabel("–"))
        tempo_range_row.addWidget(self._tempo_max)
        tempo_range_row.addWidget(QtWidgets.QLabel("BPM"))
        tempo_range_row.addStretch(1)
        tform.addRow("Tempo range:", tempo_range_row)

        energy_range_row = QtWidgets.QHBoxLayout()
        self._energy_min = QtWidgets.QDoubleSpinBox()
        self._energy_min.setRange(0.0, 1.0)
        self._energy_min.setSingleStep(0.05)
        self._energy_min.setValue(0.0)
        self._energy_max = QtWidgets.QDoubleSpinBox()
        self._energy_max.setRange(0.0, 1.0)
        self._energy_max.setSingleStep(0.05)
        self._energy_max.setValue(1.0)
        energy_range_row.addWidget(self._energy_min)
        energy_range_row.addWidget(QtWidgets.QLabel("–"))
        energy_range_row.addWidget(self._energy_max)
        energy_range_row.addStretch(1)
        tform.addRow("Energy range:", energy_range_row)
        tempo_layout.addLayout(tform)

        tempo_btn = self._make_primary_button("🎵  Generate Tempo Playlists")
        tempo_btn.clicked.connect(lambda: self._on_run("tempo"))
        self._run_buttons.append(tempo_btn)
        tempo_layout.addWidget(tempo_btn)
        tempo_layout.addStretch(1)
        tabs.addTab(tempo_widget, "Tempo / Energy")

        # ── Auto-DJ tab ────────────────────────────────────────────────────
        autodj_widget = QtWidgets.QWidget()
        autodj_layout = QtWidgets.QVBoxLayout(autodj_widget)
        autodj_layout.setContentsMargins(16, 16, 16, 16)
        autodj_layout.setSpacing(10)

        autodj_layout.addWidget(self._make_subtitle(
            "Build a smooth transition playlist using similarity-based track ordering. "
            "Tracks are chained so each song is sonically close to the next. "
            "Saved as Playlists/autodj.m3u."
        ))

        count_row = QtWidgets.QHBoxLayout()
        count_row.addWidget(QtWidgets.QLabel("Track count:"))
        self._autodj_count = QtWidgets.QSpinBox()
        self._autodj_count.setRange(5, 500)
        self._autodj_count.setValue(20)
        count_row.addWidget(self._autodj_count)
        count_row.addStretch(1)
        autodj_layout.addLayout(count_row)

        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(QtWidgets.QLabel("Seed track:"))
        self._autodj_seed = QtWidgets.QLineEdit()
        self._autodj_seed.setPlaceholderText("Path to the starting track (in your library)…")
        seed_browse = QtWidgets.QPushButton("Browse…")
        seed_browse.clicked.connect(lambda: self._browse_file(self._autodj_seed))
        seed_row.addWidget(self._autodj_seed, 1)
        seed_row.addWidget(seed_browse)
        autodj_layout.addLayout(seed_row)

        autodj_btn = self._make_primary_button("🎵  Generate Auto-DJ Playlist")
        autodj_btn.clicked.connect(lambda: self._on_run("autodj"))
        self._run_buttons.append(autodj_btn)
        autodj_layout.addWidget(autodj_btn)
        autodj_layout.addStretch(1)
        tabs.addTab(autodj_widget, "Auto-DJ")

        # ── Playlist Repair tab ────────────────────────────────────────────
        repair_widget = QtWidgets.QWidget()
        repair_layout = QtWidgets.QVBoxLayout(repair_widget)
        repair_layout.setContentsMargins(16, 16, 16, 16)
        repair_layout.setSpacing(10)

        repair_layout.addWidget(self._make_subtitle(
            "Re-scan playlists for broken paths after you've moved or renamed files. "
            "Missing entries are matched to library tracks by name (or to the same "
            "track in the other codec). Add playlists below, or leave the list empty "
            "to check every playlist in the library's Playlists/ folder. You review "
            "the fixes before anything is saved."
        ))

        self._repair_list = QtWidgets.QListWidget()
        self._repair_list.setMinimumHeight(120)
        repair_layout.addWidget(self._repair_list)

        repair_add_row = QtWidgets.QHBoxLayout()
        repair_add_btn = QtWidgets.QPushButton("Add Playlists…")
        repair_add_btn.clicked.connect(self._on_add_playlists)
        repair_rem_btn = QtWidgets.QPushButton("Remove Selected")
        repair_rem_btn.clicked.connect(
            lambda: self._repair_list.takeItem(self._repair_list.currentRow())
        )
        repair_clear_btn = QtWidgets.QPushButton("Clear")
        repair_clear_btn.clicked.connect(self._repair_list.clear)
        self._prefer_opus_cb = QtWidgets.QCheckBox("Prefer Opus when searching for missing FLAC")
        self._prefer_opus_cb.setToolTip(
            "Checked: a missing .flac entry is pointed at the same track's .opus.\n"
            "Unchecked: a missing .opus entry is pointed at the same track's .flac."
        )
        repair_add_row.addWidget(repair_add_btn)
        repair_add_row.addWidget(repair_rem_btn)
        repair_add_row.addWidget(repair_clear_btn)
        repair_add_row.addWidget(self._prefer_opus_cb)
        repair_add_row.addStretch(1)
        repair_layout.addLayout(repair_add_row)

        repair_run_btn = self._make_primary_button("🔧  Run Repair")
        repair_run_btn.clicked.connect(lambda: self._on_run("repair"))
        self._run_buttons.append(repair_run_btn)
        repair_layout.addWidget(repair_run_btn)
        repair_layout.addStretch(1)
        tabs.addTab(repair_widget, "Playlist Repair")

        cl.addWidget(tabs)

        # ── Progress + Cancel + Log ────────────────────────────────────────
        prog_row = QtWidgets.QHBoxLayout()
        self._prog_bar = QtWidgets.QProgressBar()
        self._prog_bar.setRange(0, 0)  # indeterminate
        self._prog_bar.setFixedHeight(6)
        self._prog_bar.setTextVisible(False)
        prog_row.addWidget(self._prog_bar, 1)
        self._cancel_btn = QtWidgets.QPushButton("✕ Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        prog_row.addWidget(self._cancel_btn)
        cl.addLayout(prog_row)

        log_card = self._make_card()
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(QtWidgets.QLabel("Log"))
        self._log_area = QtWidgets.QPlainTextEdit()
        self._log_area.setObjectName("logText")
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(130)
        self._log_area.setPlaceholderText(
            "The app switches to the Activity Log automatically during playlist generation."
        )
        log_layout.addWidget(self._log_area)
        cl.addWidget(log_card)

        cl.addStretch(1)

    # ── State ─────────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool) -> None:
        for btn in self._run_buttons:
            btn.setEnabled(not busy)
        self._prog_bar.setVisible(busy)
        self._cancel_btn.setVisible(busy)
        self._cancel_btn.setEnabled(busy)

    def is_busy(self) -> bool:
        return self._worker is not None

    def _status(self, text: str, role: str) -> None:
        tokens = get_manager().current
        colour = {
            "busy": tokens.warning, "ok": tokens.success,
            "error": tokens.danger, "neutral": tokens.text_muted,
        }[role]
        self.status_changed.emit(text, colour)

    def _start(self, worker: PlaylistWorker, name: str) -> None:
        self._worker = worker
        worker.log_line.connect(self._on_worker_log)
        worker.done.connect(self._on_worker_done)
        self._set_busy(True)
        self._start_worker(worker, name)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_run(self, mode: str) -> None:
        if self.is_busy():
            return  # buttons are disabled; guards a queued double-click
        if not self._library_path:
            QtWidgets.QMessageBox.warning(self, "No Library", "Please select a library folder first.")
            return
        seed = self._autodj_seed.text().strip()
        if mode == "autodj" and not seed:
            QtWidgets.QMessageBox.warning(
                self, "Seed Track Needed",
                "Choose the track the Auto-DJ playlist should start from.",
            )
            return

        self._log_area.clear()
        self._log(f"Previewing {mode} playlists…", "info")
        self._status("Preparing playlists…", "busy")

        repair_paths = [
            self._repair_list.item(i).text()
            for i in range(self._repair_list.count())
        ] if mode == "repair" else []
        tempo_range = (self._tempo_min.value(), self._tempo_max.value())
        energy_range = (self._energy_min.value(), self._energy_max.value())
        worker = PlaylistWorker(
            library_path=self._library_path,
            mode=mode,
            output_dir=self._folder_out_entry.text().strip(),
            tempo_range=(min(tempo_range), max(tempo_range)),
            energy_range=(min(energy_range), max(energy_range)),
            autodj_count=self._autodj_count.value(),
            seed_track=seed,
            repair_paths=repair_paths,
            prefer_opus=self._prefer_opus_cb.isChecked(),
        )
        self._start(worker, "Playlist preview")

    @Slot()
    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._log("Cancelling…", "warn")

    @Slot(str)
    def _on_worker_log(self, line: str) -> None:
        if self.sender() is not self._worker:
            return
        self._log_area.appendPlainText(line)
        self._log(line)

    @Slot(bool, str, object)
    def _on_worker_done(self, success: bool, message: str, payload: object) -> None:
        worker = self.sender()
        if worker is not self._worker:
            return  # a job we already abandoned (e.g. library changed)
        self._worker = None
        self._set_busy(False)

        if not success:
            if message.startswith("Cancelled"):
                self._log("Cancelled — nothing was written.", "warn")
                self._status("Cancelled", "neutral")
            else:
                self._log(message, "error")
                self._status("Error", "error")
            return

        if worker.mode == "write":
            self._log(message, "ok")
            self._log_area.appendPlainText(message)
            self._status("Done", "ok")
            return

        plan = list(payload or [])
        if worker.mode == "repair":
            plan = [item for item in plan if item.changes]
        if not plan:
            text = (
                "Nothing to repair — no missing entry could be matched."
                if worker.mode == "repair" else "Nothing to write."
            )
            self._log(text, "info")
            self._log_area.appendPlainText(text)
            self._status("Nothing to write", "neutral")
            return

        if not self._confirm_plan(worker.mode, plan):
            self._log("Nothing was written.", "info")
            self._log_area.appendPlainText("Nothing was written.")
            self._status("Not saved", "neutral")
            return

        self._status("Writing playlists…", "busy")
        self._start(
            PlaylistWorker(library_path=self._library_path, mode="write", plan=plan),
            "Write playlists",
        )

    def _confirm_plan(self, mode: str, plan: list) -> bool:
        overwrites = sum(1 for item in plan if item.exists)
        if mode == "repair":
            fixes = sum(len(item.changes) for item in plan)
            missing = sum(len(item.unresolved) for item in plan)
            head = (
                f"Repair {len(plan)} playlist(s): {fixes} entr{'y' if fixes == 1 else 'ies'} "
                f"will be re-pointed"
                + (f", {missing} still missing (left unchanged)." if missing else ".")
            )
            rows = [
                f"• {Path(i.path).name}: {len(i.changes)} fixed"
                + (f", {len(i.unresolved)} missing" if i.unresolved else "")
                for i in plan
            ]
        else:
            head = (
                f"Write {len(plan)} playlist(s)"
                + (f" — {overwrites} existing file(s) will be replaced." if overwrites else ".")
            )
            rows = [
                f"• {'replace' if i.exists else 'new'}: {self._display_path(i.path)} "
                f"({i.entries} track{'s' if i.entries != 1 else ''})"
                for i in plan
            ]
        if len(rows) > _CONFIRM_LIST_LIMIT:
            rows = rows[:_CONFIRM_LIST_LIMIT] + [f"… and {len(rows) - _CONFIRM_LIST_LIMIT} more"]
        tail = (
            "\n\nReplaced files are backed up first (name.m3u.<time>.bak)."
            if overwrites or mode == "repair" else ""
        )
        for row in rows:
            self._log_area.appendPlainText(row)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Save playlists?",
            head + "\n\n" + "\n".join(rows) + tail,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _display_path(self, path: str) -> str:
        if self._library_path:
            try:
                return os.path.relpath(path, self._library_path)
            except ValueError:
                pass
        return path

    def _on_library_changed(self, path: str) -> None:
        # A plan is only valid for the library it was computed against.
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
            self._set_busy(False)
            self._log("Library changed — the running playlist job was cancelled.", "warn")
            self._status("Cancelled", "neutral")

    @Slot()
    def _on_add_playlists(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Playlists", self._library_path or str(Path.home()),
            "Playlists (*.m3u *.m3u8);;All files (*)"
        )
        for p in paths:
            self._repair_list.addItem(p)

    def _browse_dir(self, entry: QtWidgets.QLineEdit) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            entry.setText(folder)

    def _browse_file(self, entry: QtWidgets.QLineEdit) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Audio File", self._library_path or "",
            "Audio (*.flac *.mp3 *.m4a *.aac *.wav *.ogg *.opus);;All (*)"
        )
        if path:
            entry.setText(path)
