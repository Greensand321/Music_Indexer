"""Compression (Opus mirror) workspace regressions.

TOOL-04/05 worker lifetime + real cancel (ffmpeg terminated, partial output
removed, close-while-running) · TOOL-13 no folders created inside the source ·
TOOL-19 estimate off the GUI thread · TOOL-23 stale estimate / bitrate spin ·
DS-14/TOOL-16 readable result text on light themes.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from PySide6 import QtGui

from _helpers.fake_ffmpeg import install_fake_ffmpeg, pid_alive

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _install_hanging_ffmpeg(bin_dir: Path, monkeypatch) -> Path:
    """A fake ffmpeg that writes partial output, records its pid and hangs.

    A Python script run through ``sys.executable`` (see
    ``tests/_helpers/fake_ffmpeg.py``), so it is the one that runs on Windows
    as well."""
    pid_file = bin_dir / "ffmpeg.pid"
    install_fake_ffmpeg(
        bin_dir, monkeypatch,
        "import os, sys, time\n"
        "open(sys.argv[-1], 'wb').write(b'OggS-partial')\n"
        f"open({str(pid_file)!r}, 'a').write(str(os.getpid()) + chr(10))\n"
        "time.sleep(60)\n",
    )
    return pid_file


_pid_alive = pid_alive


def _estimate(ws, qtbot):
    ws._estimate_btn.click()
    qtbot.waitUntil(lambda: ws._estimate_worker is None, timeout=10000)


def test_estimate_with_destination_inside_source_creates_nothing(make_workspace, library, dialogs, qtbot):
    ws = make_workspace("compression", library_path=library.root)
    inside = library.root / "Mirror" / "Phone"
    ws._dest_edit.setText(str(inside))
    ws._estimate_btn.click()
    assert [c.title for c in dialogs.calls_of("critical")] == ["Invalid Destination"]
    assert not (library.root / "Mirror").exists(), "no folders may be created in the library"
    assert not ws._run_btn.isEnabled()


def test_estimate_runs_in_a_worker_and_is_invalidated_by_changes(
    make_workspace, library, threads, qtbot, tmp_path
):
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(tmp_path / "Mirror"))
    _estimate(ws, qtbot)
    assert "EstimateWorker" in threads.started, "the walk must not block the GUI thread"
    assert "FLAC files to convert:    2" in ws._result_lbl.text()
    assert not (tmp_path / "Mirror").exists(), "estimating must not create the destination"
    assert ws._run_btn.isEnabled()

    ws._bitrate_spin.setValue(128)
    assert not ws._run_btn.isEnabled(), "a changed bitrate makes the estimate stale"
    assert "run the estimate again" in ws._result_lbl.text()
    _estimate(ws, qtbot)
    assert ws._run_btn.isEnabled()
    ws.set_library_path(str(tmp_path))
    assert not ws._run_btn.isEnabled(), "a different library makes the estimate stale"


def test_result_text_is_readable_on_light_themes(make_workspace, library, qtbot, tmp_path):
    from gui.themes.manager import get_manager

    get_manager().apply("pearl")
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(tmp_path / "Mirror"))
    _estimate(ws, qtbot)
    color = ws._result_lbl.palette().color(QtGui.QPalette.ColorRole.WindowText)
    assert color == QtGui.QColor(get_manager().current.text_primary)
    assert color != QtGui.QColor("#e2e8f0")
    assert ws._result_lbl.styleSheet() == ""
    get_manager().apply("midnight")
    color = ws._result_lbl.palette().color(QtGui.QPalette.ColorRole.WindowText)
    assert color == QtGui.QColor(get_manager().current.text_primary), "follows theme switches"


@needs_ffmpeg
def test_real_run_mirrors_and_skips_reserved_folders(make_workspace, library, dialogs, threads, qtbot, tmp_path):
    from library_factory import TrackSpec, build_library

    build_library(library.root, [TrackSpec("G", "Loser", "X", 1, relpath="Quarantine/loser.mp3")])
    dest = tmp_path / "Mirror"
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(dest))
    _estimate(ws, qtbot)
    dialogs.answer_yes()
    ws._run_btn.click()
    confirm = dialogs.calls_of("question")[0].text
    assert "kept (skipped)" in confirm and "Quarantine" in confirm
    assert not ws._bitrate_spin.isEnabled(), "bitrate edits during a run would have no effect"
    qtbot.waitUntil(lambda: ws._worker is None, timeout=60000)

    assert ws._bitrate_spin.isEnabled()
    assert (dest / "Basalt" / "Granite" / "01 - Fault Line.opus").exists()
    assert not (dest / "Quarantine").exists()
    assert "Done." in ws._result_lbl.text()
    assert ws._report_btn.isEnabled()


def test_cancel_stops_ffmpeg_and_leaves_no_partial_files(
    make_workspace, library, dialogs, threads, qtbot, tmp_path, monkeypatch
):
    pid_file = _install_hanging_ffmpeg(tmp_path / "bin", monkeypatch)
    dest = tmp_path / "Mirror"
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(dest))
    _estimate(ws, qtbot)
    dialogs.answer_yes()
    ws._run_btn.click()
    qtbot.waitUntil(pid_file.exists, timeout=20000)

    # A library change mid-run must not rewrite the running job's Source.
    other = tmp_path / "Other"
    other.mkdir()
    ws.set_library_path(str(other))
    assert ws._source_edit.text() == str(library.root)

    assert ws._cancel_btn.isEnabled()
    ws._cancel_btn.click()
    qtbot.waitUntil(lambda: ws._worker is None, timeout=20000)
    assert threads.wait_all(10000) == []

    pids = [int(p) for p in pid_file.read_text().split()]
    assert pids and not any(_pid_alive(p) for p in pids), "ffmpeg must be terminated, not orphaned"
    # (The library's own Opus track is copied; the FLAC album must have no output.)
    assert not list((dest / "Basalt").rglob("*.opus")), "no partial or untagged .opus may remain"
    assert not [p for p in dest.rglob("*") if ".alphadex-partial" in p.name]
    assert ws._result_lbl.text().startswith("Cancelled.")
    assert ws._source_edit.text() == str(other), "the pending library is applied after the run"


def test_closing_the_window_mid_compression_stops_cleanly(
    main_window, library, dialogs, threads, qtbot, tmp_path, monkeypatch
):
    """TOOL-05: closing mid-run aborted the process and orphaned ffmpeg."""
    pid_file = _install_hanging_ffmpeg(tmp_path / "bin", monkeypatch)
    ws = main_window._workspaces["compression"]
    ws.set_library_path(str(library.root))
    ws._dest_edit.setText(str(tmp_path / "Mirror"))
    _estimate(ws, qtbot)
    dialogs.answer_yes()
    ws._run_btn.click()
    qtbot.waitUntil(pid_file.exists, timeout=20000)

    dialogs.answer_yes()          # "Background tasks running — stop them and quit?"
    main_window.close()
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=20000)
    assert threads.wait_all(10000) == []
    pids = [int(p) for p in pid_file.read_text().split()]
    deadline = time.monotonic() + 5
    while any(_pid_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(_pid_alive(p) for p in pids)
    out = tmp_path / "Mirror" / "Basalt"
    assert not list(out.rglob("*.opus"))
    assert not [p for p in out.rglob("*") if ".alphadex-partial" in p.name]


# ── Folder checks, out-of-date files and orphans (mirror backend audit) ─────


def test_source_inside_the_destination_is_refused(make_workspace, library, dialogs, qtbot):
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(library.root.parent))
    ws._estimate_btn.click()
    (call,) = dialogs.calls_of("critical")
    assert call.title == "Invalid Destination" and "inside the destination" in call.text
    assert not ws._run_btn.isEnabled()


def _fake_mirror(monkeypatch, summary_extra, seen):
    import utils.opus_library_mirror as mirror

    def fake(source, destination, overwrite, **kwargs):
        seen.append(kwargs)
        summary = {"total": 1, "converted": 0, "copied": 1, "skipped": 0, "errors": 0,
                   "stale": 0, "updated": 0, "cancelled": False, "error_files": [],
                   "skipped_files": [], "orphan_files": [], "partials_removed": 0}
        summary.update(summary_extra(destination))
        return summary

    monkeypatch.setattr(mirror, "mirror_library", fake)
    monkeypatch.setattr(shutil, "which", lambda name, *a, **k: "/usr/bin/" + name)


def _run(ws, dialogs, qtbot):
    _estimate(ws, qtbot)
    dialogs.answer_yes()
    ws._run_btn.click()
    qtbot.waitUntil(lambda: ws._worker is None, timeout=20000)


def test_update_changed_is_on_by_default_and_reaches_the_backend(
        make_workspace, library, dialogs, qtbot, tmp_path, monkeypatch):
    seen = []
    _fake_mirror(monkeypatch, lambda _d: {"stale": 3, "updated": 3}, seen)
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(tmp_path / "Mirror"))
    assert ws._update_cb.isChecked()
    _run(ws, dialogs, qtbot)
    assert seen[-1]["update_changed"] is True
    assert "unless their source changed" in dialogs.calls_of("question")[-1].text
    assert "3 out-of-date file(s) redone" in ws._result_lbl.text()

    _fake_mirror(monkeypatch, lambda _d: {"stale": 2}, seen)
    ws._update_cb.setChecked(False)
    _run(ws, dialogs, qtbot)
    assert seen[-1]["update_changed"] is False
    assert "2 file(s) older than their changed source were kept" in ws._result_lbl.text()


def test_orphans_are_listed_then_removed_only_after_confirmation(
        make_workspace, library, dialogs, qtbot, threads, tmp_path, monkeypatch):
    dest = tmp_path / "Mirror"
    orphan = dest / "Old Artist" / "gone.opus"
    keep = dest / "keep.txt"

    def extra(destination):
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"opus")
        keep.write_text("not listed")
        return {"orphan_files": [str(orphan)]}

    _fake_mirror(monkeypatch, extra, [])
    before = sorted(p.relative_to(library.root) for p in library.root.rglob("*"))
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(dest))
    _run(ws, dialogs, qtbot)
    assert "1 file(s) in the mirror no longer have a source" in ws._result_lbl.text()
    assert ws._orphans_btn.isVisibleTo(ws) and "1 orphaned" in ws._orphans_btn.text()

    dialogs.answer_no()
    ws._orphans_btn.click()
    assert orphan.exists(), "declining deletes nothing"
    assert os.path.join("Old Artist", "gone.opus") in dialogs.calls_of("question")[-1].text

    dialogs.answer_yes()
    ws._orphans_btn.click()
    threads.wait_all()
    qtbot.waitUntil(lambda: ws._orphan_worker is None, timeout=20000)
    assert not orphan.exists()
    assert keep.exists(), "only the listed files are deleted"
    assert sorted(p.relative_to(library.root) for p in library.root.rglob("*")) == before
    assert not ws._orphans_btn.isVisibleTo(ws)
    assert "Removed 1 orphaned file(s)" in ws._result_lbl.text()
    assert list((dest / "Docs").rglob("*orphan*")), "every deletion is logged in the mirror's Docs/"


def test_changing_the_folders_drops_the_orphan_list(
        make_workspace, library, dialogs, qtbot, tmp_path, monkeypatch):
    dest = tmp_path / "Mirror"
    _fake_mirror(monkeypatch, lambda d: {"orphan_files": [os.path.join(d, "x.opus")]}, [])
    ws = make_workspace("compression", library_path=library.root)
    ws._dest_edit.setText(str(dest))
    _run(ws, dialogs, qtbot)
    assert ws._orphans_btn.isVisibleTo(ws)
    ws._dest_edit.setText(str(tmp_path / "Elsewhere"))
    assert not ws._orphans_btn.isVisibleTo(ws)
