"""Save cluster playlists — the flow, its jobs, the dialog and the Sound Map wiring.

docs/specs/cluster_playlists.md §8.3–§8.4 (CP-29…CP-42). Flow and dialog
tests host the flow on a bare workspace (``make_workspace("help")`` +
``ClusterPlaylistFlow(ws)``) so they don't depend on the page layout; map
integration tests use the Map panel (``make_workspace("graph")``) and need
pyqtgraph for the legend. Tags are stubbed (``read_tags`` → no tags) unless a
test gates them to hold the preview job.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from gui.compat import QtCore, QtWidgets
from gui.cluster_playlist_jobs import (
    ANALYSIS_JOB_NAME,
    CANCEL_TEXT,
    OLDER_MAP_WARNING,
    PREVIEW_JOB_NAME,
    TIP_ANALYSIS,
    TIP_NO_MEMBERS,
    WRITE_JOB_NAME,
    ClusterPlaylistFlow,
)
from gui.dialogs.cluster_playlists_dialog import ClusterPlaylistsDialog
from gui.workers import LIBRARY_WRITES, job_registry
import cluster_graph_data as cgd
import cluster_playlist_export as cpe
from _helpers.cluster_payloads import make_cluster_run, make_member_files, write_cluster_run
from playlists_helpers import FakeClustering, make_library, write_cluster_payload
from ui_harness import process_events

ACCEPT = QtWidgets.QDialog.DialogCode.Accepted
REJECT = QtWidgets.QDialog.DialogCode.Rejected


@pytest.fixture(autouse=True)
def _no_disk_tags(monkeypatch):
    import utils.audio_metadata_reader as amr
    monkeypatch.setattr(amr, "read_metadata", lambda path, include_cover=False: ({}, [], None, None))


_GATES: list = []


@pytest.fixture(autouse=True)
def _release_gates():
    """A failing test must not leave a worker parked on a gate (the harness
    would then find a running QThread)."""
    yield
    while _GATES:
        _GATES.pop().release.set()


class Gate:
    """Hold a function call (in a worker thread) until released."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        _GATES.append(self)

    def wait(self, cancel_event=None) -> None:
        self.entered.set()
        while not self.release.wait(0.02):
            if cancel_event is not None and cancel_event.is_set():
                return

    def wait_entered(self, qtbot, timeout=8000) -> None:
        """Wait (spinning the event loop: the job may first need the GUI
        thread, e.g. to open the dialog) until a worker reached the gate."""
        qtbot.waitUntil(self.entered.is_set, timeout=timeout)


def _gate_tags(monkeypatch) -> Gate:
    import utils.audio_metadata_reader as amr

    gate = Gate()

    def read_tags(path):
        gate.wait()
        return {}

    monkeypatch.setattr(amr, "read_tags", read_tags)
    return gate


def _gate_write(monkeypatch) -> Gate:
    gate = Gate()
    real = cpe.write_cluster_playlists

    def write(plan, *, cancel_event=None, log_callback=None):
        gate.wait(cancel_event)
        return real(plan, cancel_event=cancel_event, log_callback=log_callback)

    monkeypatch.setattr(cpe, "write_cluster_playlists", write)
    return gate


def _library(tmp_path, name="lib", **kw) -> Path:
    lib = tmp_path / name
    lib.mkdir(parents=True, exist_ok=True)
    write_cluster_run(lib, files=True, **kw)
    return lib


def _host(make_workspace, lib, **kw):
    ws = make_workspace("help", library_path=lib)
    logs: list = []
    ws.log_message.connect(lambda m, level: logs.append((m, level)))
    flow = ClusterPlaylistFlow(ws, **kw)
    return ws, flow, logs


def _idle(flow, qtbot, timeout=8000) -> None:
    qtbot.waitUntil(lambda: not flow.is_busy(), timeout=timeout)
    process_events(20)


def _run_dialog(flow, qtbot, dialogs, handler=REJECT) -> None:
    """Start the preview, answer its dialog with ``handler``, wait for the end."""
    dialogs.on_exec(handler)
    assert flow.start()
    qtbot.waitUntil(lambda: bool(dialogs.calls_of("exec")), timeout=8000)
    _idle(flow, qtbot)


def _texts(logs) -> list:
    return [m for m, _level in logs]


def _run_folder(lib: Path) -> Path:
    members = cgd.load_cluster_members(lib)
    return lib / "Playlists" / "Clusters" / cpe.run_folder_name(members["run"])


# ── CP-37: the button ─────────────────────────────────────────────────────


def test_button_disabled_without_members_file_with_tooltip(make_workspace, tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    write_cluster_payload(lib, with_members=False)          # an analysis from before this feature
    _ws, flow, _logs = _host(make_workspace, lib)
    assert flow.button.text() == "Save cluster playlists…"
    assert not flow.button.isEnabled()
    assert flow.button.toolTip() == TIP_NO_MEMBERS
    write_cluster_run(lib, files=True)
    flow.refresh()
    assert flow.button.isEnabled()

    _ws2, no_lib, _ = _host(make_workspace, "")
    assert not no_lib.button.isEnabled() and "library" in no_lib.button.toolTip()


def test_button_enabled_without_the_map_loaded(make_workspace, tmp_path, monkeypatch):
    import gui.widgets.interactive_scatter_plot as isp

    monkeypatch.setattr(isp, "pg", None)
    monkeypatch.setattr(isp, "PG_IMPORT_ERROR", ImportError("no pyqtgraph here"))
    lib = _library(tmp_path)
    graph = make_workspace("graph", library_path=lib)
    assert not graph._plot.available and graph.map_run_id() is None
    flow = graph.cluster_playlists_flow
    assert flow.button.isEnabled()


# ── CP-12 / CP-29: blocking data ──────────────────────────────────────────


@pytest.mark.parametrize("kind, expected", [
    ("mismatched_run", "out of step with the map"),
    ("legacy_with_stale_members", "made before cluster playlists existed"),
])
def test_legacy_or_mismatched_data_shows_the_blocking_message(make_workspace, tmp_path, qtbot, dialogs, kind, expected):
    lib = tmp_path / "lib"
    lib.mkdir()
    if kind == "legacy_with_stale_members":
        write_cluster_run(lib, files=True)
        write_cluster_run(lib, kind="legacy")       # the map is rewritten; members stay
    else:
        write_cluster_run(lib, kind=kind, files=True)
    _ws, flow, logs = _host(make_workspace, lib)
    assert flow.button.isEnabled()
    assert flow.start()
    _idle(flow, qtbot)
    (info,) = dialogs.calls_of("information")
    assert expected in info.text
    assert not dialogs.calls_of("exec")
    assert not (lib / "Playlists").exists()
    assert any("Nothing was written." in m for m in _texts(logs))


# ── CP-32: the preview job is read-only ───────────────────────────────────


def test_preview_job_is_read_only(make_workspace, tmp_path, qtbot, dialogs, monkeypatch):
    gate = _gate_tags(monkeypatch)
    lib = _library(tmp_path)
    ws, flow, _logs = _host(make_workspace, lib)
    busy = []
    flow.busy_changed.connect(busy.append)
    dialogs.on_exec(REJECT)
    assert flow.start()
    gate.wait_entered(qtbot)
    jobs = job_registry().jobs(owner=ws)
    assert [(j.name, j.ask_before_quit, j.group) for j in jobs] == [(PREVIEW_JOB_NAME, False, None)]
    assert flow.is_busy() and busy == [True]
    gate.release.set()
    _idle(flow, qtbot)
    assert busy == [True, False]


# ── CP-37: preselection ───────────────────────────────────────────────────


class _Seen:
    """What a dialog showed (read inside exec: the dialog is deleted after)."""

    selected: set
    header: str
    warnings: str
    warnings_visible: bool
    items: list
    rows: list
    table: list


def _record(seen: list, code=REJECT):
    def handler(dlg):
        assert isinstance(dlg, ClusterPlaylistsDialog)
        s = _Seen()
        s.selected = dlg.selected_clusters()
        s.header = dlg._header.text()
        s.warnings = dlg._warnings.text()
        s.warnings_visible = dlg._warnings.isVisibleTo(dlg)
        s.items = list(dlg.plan().items)
        s.rows = list(dlg.plan().rows)
        s.table = [
            [(dlg.table.item(r, c).text(), dlg.table.item(r, c).flags()) for c in range(dlg.table.columnCount())]
            for r in range(dlg.table.rowCount())
        ]
        seen.append(s)
        return code
    return handler


def test_dialog_preticks_all_but_hidden_clusters(make_workspace, tmp_path, qtbot, dialogs):
    pytest.importorskip("pyqtgraph")
    lib = _library(tmp_path)
    graph = make_workspace("graph", library_path=lib)
    assert graph.map_run_id() == cgd.load_cluster_members(lib)["run"]["run_id"]
    graph._legend._rows[1].checkbox.setChecked(False)
    seen: list = []
    _run_dialog(graph.cluster_playlists_flow, qtbot, dialogs, _record(seen))
    (shown,) = seen
    assert shown.selected == {0, 2}
    assert [i.lines[2].rsplit("=", 1)[1] for i in shown.items] == ["0", "2"]
    assert not shown.warnings_visible


def test_cluster_absent_from_downsampled_map_is_preticked(make_workspace, tmp_path, qtbot, dialogs):
    pytest.importorskip("pyqtgraph")
    lib = tmp_path / "lib"
    labels = [i % 3 for i in range(40)]
    for i in (1, 2, 3, 5, 6):            # cluster 3 has no point on a 10-point map
        labels[i] = 3
    payload, members = make_cluster_run(labels=labels, n_clusters=4, root=str(lib), downsample_to=10)
    assert 3 not in payload["labels"]
    docs = lib / "Docs"
    docs.mkdir(parents=True)
    (docs / "cluster_info.json").write_text(json.dumps(payload), encoding="utf-8")
    (docs / "cluster_members.json").write_text(json.dumps(members), encoding="utf-8")
    make_member_files(members["tracks"])
    graph = make_workspace("graph", library_path=lib)
    assert 3 not in graph._legend._rows
    graph._legend._rows[1].checkbox.setChecked(False)
    seen: list = []
    _run_dialog(graph.cluster_playlists_flow, qtbot, dialogs, _record(seen))
    (shown,) = seen
    assert shown.selected == {0, 2, 3}
    assert "The map shows 10 of 40 tracks; the playlists include all of them." in shown.header


def test_hidden_clusters_ignored_when_map_is_older(make_workspace, tmp_path, qtbot, dialogs):
    pytest.importorskip("pyqtgraph")
    lib = _library(tmp_path)
    graph = make_workspace("graph", library_path=lib)
    graph._legend._rows[1].checkbox.setChecked(False)
    write_cluster_run(lib, files=True, run_id="20261108-090000-0000ff")   # a newer analysis, not reloaded
    seen: list = []
    _run_dialog(graph.cluster_playlists_flow, qtbot, dialogs, _record(seen))
    (shown,) = seen
    assert shown.selected == {0, 1, 2}
    assert OLDER_MAP_WARNING in shown.warnings and shown.warnings_visible


# ── CP-19 / CP-24 / CP-25 / CP-39: the dialog ─────────────────────────────


def test_noise_row_is_not_checkable(make_workspace, tmp_path, qtbot, dialogs):
    lib = _library(tmp_path, n=24, noise=4)
    _ws, flow, _logs = _host(make_workspace, lib)
    seen: list = []
    _run_dialog(flow, qtbot, dialogs, _record(seen))
    (shown,) = seen
    (_save_text, save_flags), _colour, (name_text, name_flags), *_rest = shown.table[-1]
    assert not save_flags & QtCore.Qt.ItemFlag.ItemIsUserCheckable
    assert not save_flags & QtCore.Qt.ItemFlag.ItemIsEnabled
    assert name_text == "Unclustered (4) — not saved"
    assert not name_flags & QtCore.Qt.ItemFlag.ItemIsEditable
    assert len(shown.items) == 3
    assert shown.rows[-1].cluster_id == -1


def test_options_replan_rows(make_workspace, tmp_path, qtbot, dialogs):
    lib = _library(tmp_path)
    _ws, flow, _logs = _host(make_workspace, lib)
    out: dict = {}

    def handler(dlg):
        out["first"] = dlg.plan().items[0].lines[3:]
        assert dlg.save_btn.text() == "Save 3 playlists" and dlg.save_btn.isEnabled()
        dlg._min_spin.setValue(7)                          # cluster 2 has 6 tracks
        rows = {r.cluster_id: r for r in dlg.plan().rows}
        out["skip"] = (rows[2].action, rows[2].reason)
        out["locked"] = not dlg.table.item(2, 0).flags() & QtCore.Qt.ItemFlag.ItemIsEnabled
        out["label"] = dlg.save_btn.text()
        dlg._order_combo.setCurrentIndex(dlg._order_combo.findData("library"))
        out["second"] = dlg.plan().items[0].lines[3:]
        dlg.table.item(0, 2).setText("Late night/chill")   # a committed name edit
        out["path"] = os.path.basename(dlg.plan().items[0].path)
        out["cell"] = dlg.table.item(0, 2).text()
        dlg.table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        out["after_untick"] = len(dlg.plan().items)
        dlg._select_none()
        out["none"] = (len(dlg.plan().items), dlg.save_btn.isEnabled())
        dlg._select_all()
        out["all"] = len(dlg.plan().items)
        return REJECT

    _run_dialog(flow, qtbot, dialogs, handler)
    assert out["skip"] == (cpe.ACTION_SKIP, "too small (6 tracks)") and out["locked"]
    assert out["label"] == "Save 2 playlists"
    assert out["first"] != out["second"] and sorted(out["first"]) == sorted(out["second"])
    assert out["path"] == "Late night-chill.m3u" and out["cell"] == "Late night-chill"
    assert out["after_untick"] == 1
    assert out["none"] == (0, False)
    assert out["all"] == 2                                 # cluster 2 is still too small


def test_cancelling_the_dialog_writes_nothing(make_workspace, tmp_path, qtbot, dialogs):
    lib = _library(tmp_path)
    _ws, flow, logs = _host(make_workspace, lib)
    _run_dialog(flow, qtbot, dialogs, REJECT)
    assert not (lib / "Playlists" / "Clusters").exists()
    assert not (lib / "Docs" / "playlist_writes.log").exists()
    assert any("Nothing was written." in m for m in _texts(logs))


# ── CP-28 / CP-33: Save ───────────────────────────────────────────────────


def test_save_writes_exactly_the_shown_plan(make_workspace, tmp_path, qtbot, dialogs):
    lib = _library(tmp_path)
    _ws, flow, _logs = _host(make_workspace, lib)
    shown: list = []

    def handler(dlg):
        dlg.table.item(1, 0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        dlg.table.item(0, 2).setText("Morning")
        dlg._order_combo.setCurrentIndex(dlg._order_combo.findData("library"))
        shown.append(dlg.plan())
        return ACCEPT

    _run_dialog(flow, qtbot, dialogs, handler)
    (plan,) = shown
    folder = _run_folder(lib)
    assert sorted(os.listdir(folder)) == sorted(os.path.basename(i.path) for i in plan.items)
    assert sorted(os.listdir(folder)) == ["Cluster 02 - ~110 BPM - high energy.m3u", "Morning.m3u"]
    for item in plan.items:
        assert Path(item.path).read_bytes() == "".join(line + "\n" for line in item.lines).encode("utf-8")


def test_write_job_takes_the_library_write_lock(make_workspace, tmp_path, qtbot, dialogs, monkeypatch):
    gate = _gate_write(monkeypatch)
    lib = _library(tmp_path)
    ws, flow, _logs = _host(make_workspace, lib)
    dialogs.on_exec(ACCEPT)
    assert flow.start()
    gate.wait_entered(qtbot)
    jobs = job_registry().jobs(owner=ws)
    assert [(j.name, j.ask_before_quit, j.group) for j in jobs] == [(WRITE_JOB_NAME, True, LIBRARY_WRITES)]
    assert not flow.button.isEnabled()
    gate.release.set()
    _idle(flow, qtbot)
    assert len(os.listdir(_run_folder(lib))) == 3


def test_replace_asks_and_backs_up(make_workspace, tmp_path, qtbot, dialogs):
    lib = _library(tmp_path)
    _ws, flow, logs = _host(make_workspace, lib)
    _run_dialog(flow, qtbot, dialogs, ACCEPT)
    assert not dialogs.calls_of("question")         # nothing to replace the first time
    dialogs.answer_yes()
    _run_dialog(flow, qtbot, dialogs, ACCEPT)
    (question,) = dialogs.calls_of("question")
    assert "3 existing files" in question.text and "backed up" in question.text
    folder = _run_folder(lib)
    assert len(list(folder.glob("*.m3u.*.bak"))) == 3
    log = (lib / "Docs" / "playlist_writes.log").read_text(encoding="utf-8")
    assert log.count(" BACKUP ") == 3
    assert any("(3 replaced, backed up as .bak; 0 failed)" in m for m in _texts(logs))


def test_replace_declined_writes_nothing(make_workspace, tmp_path, qtbot, dialogs):
    lib = _library(tmp_path)
    _ws, flow, logs = _host(make_workspace, lib)
    _run_dialog(flow, qtbot, dialogs, ACCEPT)
    folder = _run_folder(lib)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    log_before = (lib / "Docs" / "playlist_writes.log").read_text(encoding="utf-8")
    dialogs.answer_no()
    _run_dialog(flow, qtbot, dialogs, ACCEPT)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert (lib / "Docs" / "playlist_writes.log").read_text(encoding="utf-8") == log_before
    assert "Nothing was written." in _texts(logs)[-1]


def test_busy_library_blocks_preview_and_save(make_workspace, tmp_path, qtbot, dialogs):
    from controllers import tag_write_guard

    lib = _library(tmp_path)
    _ws, flow, logs = _host(make_workspace, lib)
    assert tag_write_guard.acquire("Tag Fixer apply")
    try:
        assert not flow.start()
        assert [c.title for c in dialogs.calls_of("information")] == ["Library Busy"]
        assert not flow.is_busy()
    finally:
        tag_write_guard.release("Tag Fixer apply")
    assert "Nothing was written." in _texts(logs)[-1]

    def take_lock_then_save(dlg):
        assert tag_write_guard.acquire("Tag Fixer apply")
        return ACCEPT

    try:
        _run_dialog(flow, qtbot, dialogs, take_lock_then_save)
    finally:
        tag_write_guard.release("Tag Fixer apply")
    assert [c.title for c in dialogs.calls_of("information")] == ["Library Busy", "Library Busy"]
    assert not (lib / "Playlists").exists()
    assert "Nothing was written." in _texts(logs)[-1]


# ── CP-34: busy rules ─────────────────────────────────────────────────────


def test_button_disabled_while_an_analysis_runs_on_another_page(make_workspace, tmp_path, qtbot, monkeypatch):
    fake = FakeClustering(hold=True).install(monkeypatch)
    lib = make_library(tmp_path / "lib", [f"Artist {i % 3}/Album/t{i:02d}.mp3" for i in range(15)])
    analysis = make_workspace("clustered", library_path=lib)
    graph = make_workspace("graph", library_path=lib)
    flow = graph.cluster_playlists_flow
    assert not flow.button.isEnabled() and flow.button.toolTip() == TIP_NO_MEMBERS
    analysis._on_run_quick_start()
    assert fake.started.wait(3)
    process_events(20)
    assert not flow.button.isEnabled() and flow.button.toolTip() == TIP_ANALYSIS
    assert not flow.start()
    fake.release.set()
    # The worker then scores the clusters with scikit-learn, whose first
    # import in the process can take seconds on a loaded machine (this
    # timed out once in four R3 runs at 5 s while other jobs ran).
    qtbot.waitUntil(lambda: not analysis.is_busy(), timeout=30_000)
    qtbot.waitUntil(lambda: flow.button.isEnabled(), timeout=5000)


def test_analysis_page_uses_the_shared_job_name(make_workspace, tmp_path, qtbot, monkeypatch):
    fake = FakeClustering(hold=True).install(monkeypatch)
    lib = make_library(tmp_path / "lib", [f"A/t{i}.mp3" for i in range(6)])
    analysis = make_workspace("clustered", library_path=lib)
    analysis._on_run_quick_start()
    assert fake.started.wait(3)
    assert [j.name for j in job_registry().jobs(owner=analysis)] == [ANALYSIS_JOB_NAME] == ["Clustering"]
    fake.release.set()
    qtbot.waitUntil(lambda: not analysis.is_busy(), timeout=30_000)


# ── CP-35 / CP-36: cancel and library change ──────────────────────────────


def test_cancel_preview_opens_no_dialog(make_workspace, tmp_path, qtbot, dialogs, monkeypatch):
    gate = _gate_tags(monkeypatch)
    lib = _library(tmp_path)
    _ws, flow, logs = _host(make_workspace, lib)
    flow.button.click()
    gate.wait_entered(qtbot)
    assert flow.button.text() == CANCEL_TEXT and flow.button.isEnabled()
    flow.button.click()                              # "Cancel preview"
    assert not flow.button.isEnabled()
    gate.release.set()
    _idle(flow, qtbot)
    assert not dialogs.calls_of("exec")
    assert "Cancelled. Nothing was written." in _texts(logs)
    assert flow.button.text() == "Save cluster playlists…" and flow.button.isEnabled()


def test_library_change_cancels_preview_but_lets_write_finish(make_workspace, tmp_path, qtbot, dialogs, monkeypatch):
    lib = _library(tmp_path)
    other = _library(tmp_path, name="other", run_id="20261108-090000-0000ee")
    graph = make_workspace("graph", library_path=lib)
    flow = graph.cluster_playlists_flow

    gate = _gate_tags(monkeypatch)
    assert flow.start()
    gate.wait_entered(qtbot)
    graph.set_library_path(str(other))               # drops the preview
    assert not flow.is_busy()
    gate.release.set()
    process_events(300)
    assert not dialogs.calls_of("exec"), "a dropped preview must not open its dialog"

    import utils.audio_metadata_reader as amr
    monkeypatch.setattr(amr, "read_tags", lambda path: {})
    graph.set_library_path(str(lib))
    write_gate = _gate_write(monkeypatch)
    dialogs.on_exec(ACCEPT)
    assert flow.start()
    write_gate.wait_entered(qtbot)
    graph.set_library_path(str(other))               # mid-write
    assert flow.is_busy()
    write_gate.release.set()
    _idle(flow, qtbot)
    assert len(os.listdir(_run_folder(lib))) == 3     # written in the confirmed library
    assert (lib / "Docs" / "playlist_writes.log").exists()
    assert not (other / "Playlists").exists() and not (other / "Docs" / "playlist_writes.log").exists()


# ── CP-41: result ─────────────────────────────────────────────────────────


def test_result_is_logged_and_emitted(make_workspace, tmp_path, qtbot, dialogs):
    lib = _library(tmp_path)
    _ws, flow, logs = _host(make_workspace, lib)
    dialogs.on_exec(ACCEPT)
    with qtbot.waitSignal(flow.result_ready, timeout=8000) as sig:
        assert flow.start()
    text, ok = sig.args
    assert ok is True
    assert text == ("Saved 3 cluster playlists to Playlists/Clusters/2026-11-07 15.30.12 K-Means (k=3)/ "
                    "(0 replaced, backed up as .bak; 0 failed).")
    assert (text, "ok") in logs
    _idle(flow, qtbot)
    assert flow.button.isEnabled()


def test_refused_write_is_reported(make_workspace, tmp_path, qtbot, dialogs):
    lib = _library(tmp_path)
    _ws, flow, logs = _host(make_workspace, lib)

    def new_analysis_lands(dlg):
        write_cluster_run(lib, files=True, run_id="20261108-090000-0000dd")
        return ACCEPT

    dialogs.on_exec(new_analysis_lands)
    with qtbot.waitSignal(flow.result_ready, timeout=8000) as sig:
        assert flow.start()
    assert sig.args == [cpe.RUN_CHANGED_MESSAGE, False]
    assert (cpe.RUN_CHANGED_MESSAGE, "error") in logs
    assert not (lib / "Playlists").exists()


def test_closing_the_window_mid_write_asks(main_window, tmp_path, qtbot, dialogs, monkeypatch):
    lib = _library(tmp_path)
    main_window.set_library(str(lib))
    flow = main_window._workspaces["graph"].cluster_playlists_flow
    main_window._on_nav_changed("sound_map/map")
    gate = _gate_write(monkeypatch)
    dialogs.on_exec(ACCEPT)
    assert flow.start()
    gate.wait_entered(qtbot)
    dialogs.answer_yes()
    main_window.close()
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=8000)
    asked = [c for c in dialogs.calls_of("question") if c.title == "Background tasks running"]
    assert asked and WRITE_JOB_NAME in asked[0].text
    # Quitting cancelled the write before its first file.
    assert not _run_folder(lib).exists()


# ── Sound Map page wiring ─────────────────────────────────────────────────


def test_sound_map_places_the_button_and_watches_the_page(main_window, tmp_path, qtbot, dialogs):
    page = main_window._pages["sound_map"]
    flow = page.cluster_playlists_flow
    assert flow is main_window._workspaces["graph"].cluster_playlists_flow
    assert flow.button.parentWidget() is page._bar
    assert flow._show_target is page
    lib = _library(tmp_path)
    main_window.set_library(str(lib))
    main_window._on_nav_changed("sound_map/analyse")       # visible from both sections
    assert flow.button.isVisible() and flow.button.isEnabled()
    _run_dialog(flow, qtbot, dialogs, REJECT)
    (call,) = dialogs.calls_of("exec")
    assert isinstance(call.widget, ClusterPlaylistsDialog)


# ── CP-42: lasso export through the shared writer ─────────────────────────


def test_lasso_export_backs_up_logs_and_respects_the_lock(make_workspace, tmp_path, dialogs):
    pytest.importorskip("pyqtgraph")
    from controllers import tag_write_guard

    lib = tmp_path / "lib"
    lib.mkdir()
    write_cluster_payload(lib, n=9, tracks_root=str(lib / "Music"))
    graph = make_workspace("graph", library_path=lib)
    graph._on_cluster_selected(1)
    assert graph._m3u_btn.isEnabled()
    target = lib / "Playlists" / "picked.m3u"
    target.parent.mkdir()
    target.write_text("#EXTM3U\nold.mp3\n", encoding="utf-8")

    assert tag_write_guard.acquire("Tag Fixer apply")
    try:
        graph._on_export_m3u()
    finally:
        tag_write_guard.release("Tag Fixer apply")
    assert [c.title for c in dialogs.calls_of("information")] == ["Library Busy"]
    assert not dialogs.calls_of("getSaveFileName")
    assert target.read_text(encoding="utf-8") == "#EXTM3U\nold.mp3\n"

    dialogs.choose_save_file(target)
    graph._on_export_m3u()
    lines = target.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "#EXTM3U" and len(lines) == 4
    assert lines[1] == os.path.join("..", "Music", "track_00001.mp3")
    (bak,) = target.parent.glob("picked.m3u.*.bak")
    assert bak.read_text(encoding="utf-8") == "#EXTM3U\nold.mp3\n"
    log = (lib / "Docs" / "playlist_writes.log").read_text(encoding="utf-8")
    assert " BACKUP " in log and f"WRITE {target}" in log
    assert "backed up" in graph._status.text()
