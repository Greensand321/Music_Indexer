"""Save cluster playlists: the jobs, the lock and the button (spec §8.3).

:class:`ClusterPlaylistFlow` is page-agnostic. A page owns one, puts its
``button`` wherever it likes, and talks to it through a five-item contract
(docs/specs/cluster_playlists.md §8.4):

1. create it and place ``flow.button``;
2. call ``flow.refresh()`` after (failing to) load map data, and on show;
3. call ``flow.on_library_changed()`` from ``_on_library_changed``;
4. the map page passes ``hidden_clusters``, ``map_run_id`` and
   ``cluster_colour`` callables;
5. the analysis page calls ``flow.offer_after_run(result)`` after a
   *successful* run.

On the Sound Map the Map panel (``GraphWorkspace``) owns the flow and does
2–4; ``SoundMapPage`` places the button among its page actions, makes the
flow watch the page (not the panel) for visibility, and does 5.

The preview-first sequence::

    [Save cluster playlists…] → lock check → summary job (read-only)
      → dialog (the preview) → [Save N playlists] → lock check
      → (any Replace) question → write job (library_write=True)
      → write_playlist_plan(verify=True) → .bak + Docs/playlist_writes.log

Jobs start through the host page's ``_start_worker`` (so the JobRegistry
keeps them alive and ``job_registry().jobs(owner=page)`` lists them) and log
through ``page._log``.
"""
from __future__ import annotations

import threading
from typing import Callable

from gui.compat import QtCore, QtWidgets, Signal
from gui.workers import job_registry

#: Name the analysis job is started under (``EnhancedClusteredWorkspace``);
#: while a job of this name runs anywhere, Save cluster playlists waits.
ANALYSIS_JOB_NAME = "Clustering"
PREVIEW_JOB_NAME = "Cluster playlist preview"
WRITE_JOB_NAME = "Save cluster playlists"
ACTION_NAME = "Save cluster playlists"

BUTTON_TEXT = "Save cluster playlists…"
CANCEL_TEXT = "Cancel preview"
TIP_READY = (
    "Preview one playlist per cluster of the latest analysis, then save them "
    "to Playlists/Clusters/. Nothing is written until you click Save."
)
TIP_NO_LIBRARY = "Choose a library first."
TIP_NO_MEMBERS = (
    "Analyse your library first. (Analyses made before this feature need to be re-run.)"
)
TIP_ANALYSIS = "Wait for the analysis to finish."
TIP_PREVIEW = "Stop preparing the preview (nothing has been written)."
TIP_WRITING = "Saving the cluster playlists…"
OLDER_MAP_WARNING = "The map shows an older analysis; press Reload to see this one"
NOTHING_WRITTEN = "Nothing was written."


def _wording(message: str) -> str:
    """The data layer still names the old "Clustered Playlists" page; on the
    Sound Map that step is the Analyse tab."""
    return (message.replace("Re-run Clustered Playlists", "Run the analysis again (Analyse tab)")
                   .replace("Run Clustered Playlists", "Run an analysis (Analyse tab)"))


class ClusterPlaylistSummaryWorker(QtCore.QThread):
    """Builds the read-only :class:`cluster_playlist_export.RunSummary`."""

    # Not called ``finished``: that would shadow QThread.finished.
    done = Signal(object, str)      # summary or None, "" | "cancelled" | error text
    progress = Signal(int, int)
    log_line = Signal(str)

    def __init__(self, library_root: str) -> None:
        super().__init__()
        self.library_root = library_root
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            import cluster_playlist_export as cpe
        except Exception as exc:  # noqa: BLE001
            self.done.emit(None, f"could not load the export module: {exc}")
            return
        try:
            summary = cpe.summarize_cluster_run(
                self.library_root,
                cancel_event=self._cancel,
                progress_callback=self.progress.emit,
                log_callback=self.log_line.emit,
            )
        except cpe.ClusterPlaylistsCancelled:
            self.done.emit(None, "cancelled")
            return
        except Exception as exc:  # noqa: BLE001 - report, never crash the app
            self.done.emit(None, str(exc) or type(exc).__name__)
            return
        if self._cancel.is_set():
            self.done.emit(None, "cancelled")
            return
        self.done.emit(summary, "")


class ClusterPlaylistWriteWorker(QtCore.QThread):
    """Writes exactly the confirmed plan (``write_cluster_playlists``)."""

    done = Signal(object)           # the result dict
    log_line = Signal(str)

    def __init__(self, plan) -> None:
        super().__init__()
        self.plan = plan
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Stop between files (used by the main window's quit path)."""
        self._cancel.set()

    def run(self) -> None:
        import cluster_playlist_export as cpe
        try:
            # Looked up at call time so a test can hold the write.
            result = cpe.write_cluster_playlists(
                self.plan, cancel_event=self._cancel, log_callback=self.log_line.emit,
            )
        except Exception as exc:  # noqa: BLE001
            result = {"written": [], "backups": [], "errors": [(self.plan.run_folder, str(exc))],
                      "cancelled": False, "refused": None}
        self.done.emit(result)


class ClusterPlaylistFlow(QtCore.QObject):
    """Owns the Save cluster playlists button, its two jobs and the dialog.

    Signals:
        busy_changed(bool) — a preview or write job started / ended.
        progress(int, int) — preview progress (done, total).
        result_ready(str, bool) — the result line of a write (text, ok).
    """

    busy_changed = Signal(bool)
    progress = Signal(int, int)
    result_ready = Signal(str, bool)

    def __init__(
        self,
        page,
        *,
        hidden_clusters: Callable[[], set] | None = None,
        map_run_id: Callable[[], str | None] | None = None,
        cluster_colour: Callable[[int], object] | None = None,
        show_target: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(page)
        self._page = page
        self._hidden_clusters = hidden_clusters
        self._map_run_id = map_run_id
        self._cluster_colour = cluster_colour
        self._summary_worker: ClusterPlaylistSummaryWorker | None = None
        self._write_worker: ClusterPlaylistWriteWorker | None = None
        self._only: set | None = None
        self._cancelling = False
        self._pending_offer = False
        self._pending_summary = None      # (summary, only) waiting to be shown
        self._show_target: QtWidgets.QWidget | None = None

        self.button = QtWidgets.QPushButton(BUTTON_TEXT)
        self.button.setObjectName("clusterPlaylistsBtn")
        self.button.clicked.connect(self._on_button_clicked)

        job_registry().jobs_changed.connect(self._on_jobs_changed)
        self.set_show_target(show_target or page)
        self.refresh()

    # ── Public interface ──────────────────────────────────────────────────

    def set_show_target(self, widget: QtWidgets.QWidget) -> None:
        """Watch ``widget`` for visibility: a held offer or preview opens the
        next time it is shown, so no modal appears over another page."""
        if self._show_target is not None:
            self._show_target.removeEventFilter(self)
        self._show_target = widget
        widget.installEventFilter(self)

    def is_busy(self) -> bool:
        return self._summary_worker is not None or self._write_worker is not None

    def refresh(self) -> None:
        """Re-evaluate the button (CP-37). Cheap: one ``stat``, no JSON parsing."""
        btn = self.button
        if self._summary_worker is not None:
            btn.setText(CANCEL_TEXT)
            btn.setEnabled(not self._cancelling)
            btn.setToolTip("Cancelling…" if self._cancelling else TIP_PREVIEW)
            return
        btn.setText(BUTTON_TEXT)
        if self._write_worker is not None:
            enabled, tip = False, TIP_WRITING
        elif self._analysis_running():
            enabled, tip = False, TIP_ANALYSIS
        elif not self._library():
            enabled, tip = False, TIP_NO_LIBRARY
        elif not self._members_file_exists():
            enabled, tip = False, TIP_NO_MEMBERS
        else:
            enabled, tip = True, TIP_READY
        btn.setEnabled(enabled)
        btn.setToolTip(tip)

    def start(self, only: set | None = None) -> bool:
        """Start the preview: lock check, then the read-only summary job.
        ``only=None`` preselects every cluster except the map's hidden ones.
        True when the job started."""
        if self.is_busy():
            return False
        library = self._library()
        if not library:
            self._log(f"{ACTION_NAME}: choose a library first. {NOTHING_WRITTEN}", "warn")
            return False
        if self._analysis_running():
            self._log(f"{ACTION_NAME}: {TIP_ANALYSIS} {NOTHING_WRITTEN}", "warn")
            return False
        # While another page is moving files, a presence check is meaningless.
        if self._page._library_writes_blocked(ACTION_NAME):
            self._log(f"{ACTION_NAME}: the library is busy. {NOTHING_WRITTEN}", "info")
            return False
        worker = ClusterPlaylistSummaryWorker(library)
        worker.done.connect(self._on_summary_done)
        worker.progress.connect(self._on_summary_progress)
        worker.log_line.connect(self._on_worker_log)
        self._summary_worker = worker
        self._cancelling = False
        self._only = set(only) if only is not None else None
        self._pending_summary = None
        self._log("Preparing the cluster playlist preview…", "info")
        self._set_busy()
        self._page._start_worker(worker, PREVIEW_JOB_NAME, ask_before_quit=False)
        return True

    def cancel(self) -> None:
        """Cancel whichever job runs (a write stops between files)."""
        if self._summary_worker is not None:
            self._summary_worker.cancel()
            self._cancelling = True
            self.refresh()
        if self._write_worker is not None:
            self._write_worker.cancel()

    def offer_after_run(self, result: dict) -> None:
        """CP-44: after a successful analysis whose config asked for it, open
        the preview — now, or the next time the page is shown."""
        config = (result or {}).get("config") or {}
        if not config.get("create_playlists") or not isinstance((result or {}).get("run"), dict):
            return
        self._pending_offer = True
        # Deferred: the analysis job is still finishing (its thread has not
        # returned yet), and this may run inside the analysis' done slot.
        QtCore.QTimer.singleShot(0, self, self._maybe_offer)

    def on_library_changed(self) -> None:
        """CP-36: drop a running preview and any held offer; a running write
        finishes in the old library (it only ever touches ``plan.library_root``)."""
        if self._summary_worker is not None:
            self._summary_worker.cancel()
            self._summary_worker = None   # its result is ignored when it lands
            self._cancelling = False
            self._set_busy()
        self._pending_offer = False
        self._pending_summary = None
        self.refresh()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _library(self) -> str:
        return str(getattr(self._page, "_library_path", "") or "")

    def _members_file_exists(self) -> bool:
        import cluster_graph_data as cgd
        try:
            return cgd.cluster_members_path(self._library()).is_file()
        except OSError:
            return False

    @staticmethod
    def _analysis_running() -> bool:
        return any(j.name == ANALYSIS_JOB_NAME for j in job_registry().jobs())

    def _log(self, message: str, level: str = "info") -> None:
        self._page._log(message, level)

    def _target_visible(self) -> bool:
        target = self._show_target or self._page
        try:
            return bool(target.isVisible())
        except RuntimeError:
            return False

    def _set_busy(self) -> None:
        self.refresh()
        self.busy_changed.emit(self.is_busy())

    # ── Events ────────────────────────────────────────────────────────────

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self._show_target and event.type() == QtCore.QEvent.Type.Show:
            QtCore.QTimer.singleShot(0, self, self._on_target_shown)
        return False

    def _on_target_shown(self) -> None:
        self.refresh()
        if not self._target_visible():
            return
        if self._pending_summary is not None:
            summary, only = self._pending_summary
            self._pending_summary = None
            self._open_dialog(summary, only)
        else:
            self._maybe_offer()

    def _on_jobs_changed(self, _count: int = 0) -> None:
        self.refresh()
        if self._pending_offer:
            self._maybe_offer()

    def _maybe_offer(self) -> None:
        if not self._pending_offer or self.is_busy():
            return
        if self._analysis_running() or not self._target_visible():
            return   # retried on jobs_changed / the next Show
        self._pending_offer = False
        self.start(None)

    def _on_button_clicked(self) -> None:
        if self._summary_worker is not None:
            self.cancel()          # "Cancel preview" (CP-35)
            return
        self.start(None)

    # ── Summary job ───────────────────────────────────────────────────────

    def _on_summary_progress(self, done: int, total: int) -> None:
        if self.sender() is not self._summary_worker:
            return
        if total > 0 and not self._cancelling:
            self.button.setToolTip(f"{TIP_PREVIEW}  ·  {int(100 * done / total)}% checked")
        self.progress.emit(done, total)

    def _on_worker_log(self, line: str) -> None:
        if self.sender() not in (self._summary_worker, self._write_worker):
            return
        self._log(line, "info")

    def _on_summary_done(self, summary, error: str) -> None:
        if self.sender() is not self._summary_worker:
            return   # a preview that was dropped (library change)
        self._summary_worker = None
        self._cancelling = False
        self._set_busy()
        if error == "cancelled":
            self._log(f"Cancelled. {NOTHING_WRITTEN}", "info")
            return
        if error or summary is None:
            text = f"Could not prepare the cluster playlists: {error}"
            QtWidgets.QMessageBox.warning(self._page, ACTION_NAME, text)
            self._log(f"{text}. {NOTHING_WRITTEN}", "error")
            return
        if summary.blocking_error:
            text = _wording(summary.blocking_error)
            QtWidgets.QMessageBox.information(self._page, ACTION_NAME, text)
            self._log(f"{ACTION_NAME}: {text} {NOTHING_WRITTEN}", "warn")
            return
        if not self._target_visible():
            self._pending_summary = (summary, self._only)
            self._log("The cluster playlist preview is ready; it opens when you return to the Sound Map.", "info")
            return
        self._open_dialog(summary, self._only)

    def _preselection(self, summary, only: set | None) -> set:
        """CP-37: every cluster of the *run* (not of the legend), minus the
        clusters hidden on the map — but only when the map shows this run."""
        clusters = set(summary.clusters)
        if only is not None:
            return clusters & {int(c) for c in only}
        map_run = None
        if self._map_run_id is not None:
            try:
                map_run = self._map_run_id()
            except Exception:  # noqa: BLE001
                map_run = None
        if map_run is not None and map_run != summary.run_id:
            summary.warnings.append(OLDER_MAP_WARNING)
            return clusters
        if map_run == summary.run_id and self._hidden_clusters is not None:
            try:
                hidden = {int(c) for c in (self._hidden_clusters() or ())}
            except Exception:  # noqa: BLE001
                hidden = set()
            return clusters - hidden
        return clusters

    def _open_dialog(self, summary, only: set | None) -> None:
        from gui.dialogs.cluster_playlists_dialog import ClusterPlaylistsDialog

        preselect = self._preselection(summary, only)
        dialog = ClusterPlaylistsDialog(
            summary, preselect, colour_for=self._cluster_colour, parent=self._page,
        )
        try:
            accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted.value
            plan = dialog.plan()
        finally:
            dialog.deleteLater()
        if not accepted:
            self._log(f"{ACTION_NAME}: cancelled. {NOTHING_WRITTEN}", "info")
            self.refresh()
            return
        if not plan.ok or not plan.items:
            self._log(f"{ACTION_NAME}: no playlist to save. {NOTHING_WRITTEN}", "info")
            return
        if self._page._library_writes_blocked(ACTION_NAME):
            self._log(f"{ACTION_NAME}: the library is busy. {NOTHING_WRITTEN}", "info")
            return
        replace = plan.replace_count
        if replace:
            answer = QtWidgets.QMessageBox.question(
                self._page,
                "Replace cluster playlists",
                f"{replace} existing file{'s' if replace != 1 else ''} in "
                f"{summary.run_folder_rel} will be replaced (backed up first, beside "
                "each file).\n\nSave the cluster playlists?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self._log(f"{ACTION_NAME}: replace declined. {NOTHING_WRITTEN}", "info")
                return
        self._start_write(plan)

    # ── Write job ─────────────────────────────────────────────────────────

    def _start_write(self, plan) -> None:
        worker = ClusterPlaylistWriteWorker(plan)
        worker.done.connect(self._on_write_done)
        worker.log_line.connect(self._on_worker_log)
        self._write_worker = worker
        self._set_busy()
        self._page._start_worker(worker, WRITE_JOB_NAME, library_write=True)

    def _on_write_done(self, result: dict) -> None:
        if self.sender() is not self._write_worker:
            return
        import cluster_playlist_export as cpe

        plan = self._write_worker.plan
        self._write_worker = None
        self._set_busy()
        text, ok = cpe.result_message(plan, result)
        if result.get("refused"):
            self._log(text, "error")
        else:
            self._log(text, "ok" if not result.get("errors") else "error")
            for path, msg in result.get("errors") or []:
                self._log(f"  {path}: {msg}", "error")
        self.result_ready.emit(text, ok)
        self.refresh()
