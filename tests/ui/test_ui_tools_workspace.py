"""File Cleanup and Exports pages (the old Utilities page), and the dialog
launchers that replaced its Diagnostics tile.

TOOL-01/02 File Cleanup preview-first · TOOL-04 worker lifetime · TOOL-14
exports skip reserved folders · TOOL-17 status chip · TOOL-19 codec export off
the GUI thread · TOOL-21/DUP-20 one live dialog per tool (now the Duplicates
page's Pair review / prototype sections) · DS-06 chip transitions · DS-08
light-theme tint · DS-09 hover timers · DS-20 colour flashes.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from library_factory import ExtraFile, TrackSpec, build_library


def _cleanup_library(root: Path):
    return build_library(root, [
        TrackSpec("Band", "Song", "LP", 1, relpath="Band/LP/Song (1).mp3"),
        TrackSpec("Band", "Carbon Copy", "LP", 2, relpath="Band/LP/Carbon Copy.mp3"),
        TrackSpec("Band", "Live at Wembley (1986)", "LP", 3,
                  relpath="Band/LP/Live at Wembley (1986).mp3"),
        TrackSpec("Band", "Love Song (2)", "LP", 4, relpath="Band/LP/Love Song (2).mp3"),
        TrackSpec("Band", "Loser", "LP", 5, relpath="Quarantine/Loser (1).mp3"),
        TrackSpec("Band", "Tune", "EP", 1, relpath="Band/EP/Tune.mp3"),
        TrackSpec("Band", "Tune", "EP", 1, relpath="Band/EP/Tune - Copy.mp3"),
    ], [ExtraFile("Playlists/mix.m3u", b"#EXTM3U\n../Band/LP/Song (1).mp3\n")])


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _cleanup_idle(ws) -> bool:
    return (ws._cleanup_plan_worker is None and ws._cleanup_worker is None
            and ws._cleanup_run_btn.isEnabled())


# ── File Cleanup ──────────────────────────────────────────────────────────────

def test_file_cleanup_previews_then_renames_exactly_the_plan(
    make_workspace, dialogs, threads, qtbot, tmp_path
):
    lib = _cleanup_library(tmp_path / "Music")
    ws = make_workspace("file_cleanup", library_path=lib.root)
    seen: dict = {}

    def review(dlg):
        seen["rows"] = dlg.rows()
        seen["untouched_during_preview"] = (lib.root / "Band/LP/Song (1).mp3").exists()
        return QtWidgets.QDialog.DialogCode.Accepted

    dialogs.on_exec(review)
    ws._cleanup_run_btn.click()
    qtbot.waitUntil(lambda: "rows" in seen and _cleanup_idle(ws), timeout=15000)

    assert seen["untouched_during_preview"], "nothing may be renamed before the user confirms"
    rows = {(r[1], r[2]): r[3] for r in seen["rows"]}
    assert rows[("Song (1).mp3", "Song.mp3")] == "Rename"
    assert rows[("Tune - Copy.mp3", "Tune.mp3")].startswith("Skip")   # collision flagged
    assert len(rows) == 2, "real titles and reserved folders must not appear in the plan"

    files = _files(lib.root)
    assert "Band/LP/Song.mp3" in files and "Band/LP/Song (1).mp3" not in files
    for kept in ("Band/LP/Carbon Copy.mp3", "Band/LP/Live at Wembley (1986).mp3",
                 "Band/LP/Love Song (2).mp3", "Quarantine/Loser (1).mp3",
                 "Band/EP/Tune.mp3", "Band/EP/Tune - Copy.mp3"):
        assert kept in files, kept

    logs = list((lib.root / "Docs").glob("file_cleanup_*.jsonl"))
    assert len(logs) == 1
    renames = [json.loads(l) for l in logs[0].read_text(encoding="utf-8").splitlines()]
    assert [Path(r["new"]).name for r in renames if r["type"] == "rename"] == ["Song.mp3"]
    assert ws._cleanup_tile.status_chip.state == ws._cleanup_tile.status_chip.SUCCESS
    assert "Undo log" in ws._cleanup_status.text()


def test_file_cleanup_cancelled_preview_changes_nothing(
    make_workspace, dialogs, threads, qtbot, tmp_path
):
    lib = _cleanup_library(tmp_path / "Music")
    before = _files(lib.root)
    ws = make_workspace("file_cleanup", library_path=lib.root)
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Rejected)
    ws._cleanup_run_btn.click()
    qtbot.waitUntil(lambda: bool(dialogs.calls_of("exec")) and _cleanup_idle(ws), timeout=15000)
    assert _files(lib.root) == before
    assert not (lib.root / "Docs").exists()


def test_file_cleanup_passes_library_root_to_playlist_update(
    make_workspace, dialogs, threads, qtbot, tmp_path, monkeypatch
):
    """TOOL-03 call site: renames confined to one album must still reach
    <library>/Playlists, so the root is passed explicitly."""
    import playlist_generator

    calls = []

    def fake_update(changes, library_root=None):
        calls.append((dict(changes), library_root))
        return 1

    monkeypatch.setattr(playlist_generator, "update_playlists", fake_update)
    lib = _cleanup_library(tmp_path / "Music")
    ws = make_workspace("file_cleanup", library_path=lib.root)
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Accepted)
    ws._cleanup_run_btn.click()
    qtbot.waitUntil(lambda: bool(dialogs.calls_of("exec")) and _cleanup_idle(ws), timeout=15000)
    assert calls and Path(calls[0][1]) == lib.root
    assert "1 playlist(s) updated" in ws._cleanup_status.text()


def test_file_cleanup_repeated_runs_do_not_crash(make_workspace, threads, qtbot, tmp_path):
    """TOOL-04: the old unparented worker aborted the process within ~20 runs."""
    lib = tmp_path / "Empty"
    lib.mkdir()
    ws = make_workspace("file_cleanup", library_path=lib)
    for _ in range(15):
        ws._cleanup_run_btn.click()
        qtbot.waitUntil(lambda: _cleanup_idle(ws), timeout=5000)
    assert threads.wait_all(5000) == []
    assert "Nothing to clean up" in ws._cleanup_status.text()


def test_cleanup_button_is_disabled_while_running(make_workspace, qtbot, tmp_path, monkeypatch):
    import file_cleanup

    release = threading.Event()
    real = file_cleanup.build_cleanup_plan

    def slow_plan(*a, **k):
        release.wait(5)
        return real(*a, **k)

    monkeypatch.setattr(file_cleanup, "build_cleanup_plan", slow_plan)
    lib = tmp_path / "Empty"
    lib.mkdir()
    ws = make_workspace("file_cleanup", library_path=lib)
    ws._cleanup_run_btn.click()
    assert not ws._cleanup_run_btn.isEnabled()
    ws._on_file_cleanup()          # a second trigger is ignored
    assert ws._cleanup_cancel_btn.isEnabled()
    release.set()
    qtbot.waitUntil(lambda: _cleanup_idle(ws), timeout=5000)


# ── Exports ───────────────────────────────────────────────────────────────────

def test_artist_title_export_skips_reserved_folders(make_workspace, library, qtbot, threads):
    build_library(library.root, [
        TrackSpec("Ghost", "Quarantined Loser", "X", 1, relpath="Quarantine/loser.mp3"),
        TrackSpec("Ghost", "Trashed", "X", 1, relpath="Trash/old.mp3"),
    ])
    ws = make_workspace("exports", library_path=library.root)
    ws._at_run_btn.click()
    qtbot.waitUntil(lambda: ws._at_worker is None and ws._at_run_btn.isEnabled(), timeout=15000)
    text = (library.root / "Docs" / "artist_title_list.txt").read_text(encoding="utf-8")
    assert "Aurora Lines - First Light" in text
    assert "Quarantined Loser" not in text and "Trashed" not in text
    chip = ws._at_tile.status_chip
    assert chip.state == chip.SUCCESS and "Done" in chip.text
    assert ws._at_open.isEnabled()


def test_artist_title_export_can_be_cancelled(make_workspace, library, qtbot, threads, monkeypatch):
    import utils.audio_metadata_reader as reader

    gate = threading.Event()
    real = reader.read_tags

    def slow(path):
        gate.wait(5)
        return real(path)

    monkeypatch.setattr(reader, "read_tags", slow)
    ws = make_workspace("exports", library_path=library.root)
    ws._at_run_btn.click()
    assert ws._at_cancel_btn.isEnabled() and not ws._at_run_btn.isEnabled()
    ws._at_cancel_btn.click()
    gate.set()
    qtbot.waitUntil(lambda: ws._at_worker is None, timeout=10000)
    assert ws._at_tile.status_chip.text == "Cancelled"
    assert not (library.root / "Docs" / "artist_title_list.txt").exists()


def test_codec_export_runs_in_a_worker_and_skips_reserved(make_workspace, library, qtbot, threads):
    build_library(library.root, [
        TrackSpec("Ghost", "Loser", "X", 1, relpath="Quarantine/loser.mp3"),
    ])
    ws = make_workspace("exports", library_path=library.root)
    ws._codec_run_btn.click()
    assert "CodecExportWorker" in threads.started, "the walk must not run on the GUI thread"
    qtbot.waitUntil(lambda: ws._codec_worker is None, timeout=15000)
    text = (library.root / "Docs" / "codec_file_list.txt").read_text(encoding="utf-8")
    assert "First Light" in text
    assert "Quarantine" not in text
    assert ws._codec_open.isEnabled()


def test_library_change_resets_stale_results(make_workspace, library, qtbot, threads, tmp_path):
    ws = make_workspace("exports", library_path=library.root)
    ws._codec_run_btn.click()
    qtbot.waitUntil(lambda: ws._codec_worker is None, timeout=15000)
    assert ws._codec_open.isEnabled()
    other = tmp_path / "Other"
    other.mkdir()
    ws.set_library_path(str(other))
    assert not ws._codec_open.isEnabled()
    assert ws._codec_out_path is None


# ── Dialog launchers: Duplicates → Pair review / prototypes (TOOL-21 / DUP-20) ─

def _children(ws, cls_name: str) -> list:
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete.value)
    return [c for c in ws.findChildren(QtWidgets.QDialog) if type(c).__name__ == cls_name]


@pytest.mark.parametrize("key, dialog", [
    ("fuzzy_finder", "FuzzyDupeDialog"),
    ("scan_engine", "ScanEngineDialog"),
    ("bucketing", "BucketingPocDialog"),
])
def test_launcher_dialogs_do_not_leak(make_workspace, dialogs, qtbot, tmp_path, key, dialog):
    ws = make_workspace(key, library_path=tmp_path)
    btn = ws.findChild(QtWidgets.QPushButton, "launcherOpenBtn")
    for _ in range(3):
        dialogs.on_exec(QtWidgets.QDialog.DialogCode.Rejected)
        btn.click()
    assert [c.widget for c in dialogs.calls_of("exec")] and len(dialogs.calls_of("exec")) == 3
    assert _children(ws, dialog) == []


def test_pair_review_without_a_preview_says_so(make_workspace, dialogs, tmp_path):
    ws = make_workspace("pair_review", library_path=tmp_path)
    ws.findChild(QtWidgets.QPushButton, "launcherOpenBtn").click()
    (info,) = dialogs.calls_of("information")
    assert "No duplicate preview found" in info.text
    assert dialogs.calls_of("exec") == []


class _BusyWorker(QtCore.QThread):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def run(self):
        self.gate.wait(10)


def test_reopening_a_busy_scan_reuses_its_dialog(make_workspace, dialogs, qtbot, monkeypatch):
    """Closing Scan Engine mid-run and reopening must not start a second scan."""
    import gui.dialogs.scan_engine_dialog as mod

    gate = threading.Event()
    created = []

    class FakeScanDialog(QtWidgets.QDialog):
        def __init__(self, library_path, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Scan Engine")
            self._worker = None
            created.append(self)

        def start_scan(self):
            self._worker = _BusyWorker(gate)
            self._worker.start()

    monkeypatch.setattr(mod, "ScanEngineDialog", FakeScanDialog)
    ws = make_workspace("scan_engine")
    btn = ws.findChild(QtWidgets.QPushButton, "launcherOpenBtn")

    def start_then_close(dlg):
        dlg.start_scan()
        return QtWidgets.QDialog.DialogCode.Rejected

    dialogs.on_exec(start_then_close)
    btn.click()
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Rejected)
    btn.click()
    assert len(created) == 1, "the busy dialog is reopened, not duplicated"
    assert dialogs.calls_of("exec")[1].widget is created[0]

    gate.set()
    created[0]._worker.wait(5000)
    dialogs.on_exec(QtWidgets.QDialog.DialogCode.Rejected)
    btn.click()
    assert len(created) == 2, "once idle, the next open gets a fresh dialog"
    assert len(_children(ws, "FakeScanDialog")) == 0


# ── Tiles: chip, timers, theme, flashes ───────────────────────────────────────

def test_status_chip_survives_repeated_transitions(make_workspace, qtbot, motion):
    """DS-06: the second transition used to raise 'Internal C++ object already deleted'."""
    ws = make_workspace("exports")
    chip = ws._at_tile.status_chip
    for i in range(4):
        chip.show_success(f"ok {i}")
        motion.settle(50)
        chip.show_error(f"bad {i}")
        motion.settle(50)
    chip.show_neutral("x")
    chip.show_success("final")     # overlapping transitions: newest wins
    motion.settle(100)
    assert chip.text == "final" and chip.state == chip.SUCCESS


def test_tiles_are_static_glass_with_no_hover_timers(make_workspace, qtbot):
    """DS-09: the old tiles ran a 60 Hz hover-glow timer (forever, until it was
    made to stop). The glass tiles paint a static card surface: hovering
    starts no timer and no animation, and their shadow is painted by the
    page like every other card's."""
    from gui.widgets.glass import GlassCard
    from gui.workspaces.tools import ToolTile

    ws = make_workspace("exports")
    tiles = ws.findChildren(ToolTile)
    assert len(tiles) == 2      # Artist/Title export, Codec list export
    assert all(isinstance(t, GlassCard) for t in tiles)
    assert set(map(id, tiles)) <= set(map(id, ws._inner.elevated_widgets()))

    def busy(tile):
        timers = [t for t in tile.findChildren(QtCore.QTimer) if t.isActive()]
        anims = [a for a in tile.findChildren(QtCore.QAbstractAnimation)
                 if a.state() == QtCore.QAbstractAnimation.State.Running]
        return timers + anims

    tile = tiles[0]
    QtWidgets.QApplication.sendEvent(tile, QtGui.QEnterEvent(
        QtCore.QPointF(10, 10), QtCore.QPointF(10, 10), QtCore.QPointF(10, 10)))
    assert busy(tile) == []
    QtWidgets.QApplication.sendEvent(tile, QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert busy(tile) == []


def _mean_lightness(widget) -> float:
    img = widget.grab().toImage()
    total = n = 0
    for x in range(20, img.width() - 20, 17):
        for y in range(10, min(img.height() - 10, 60), 7):
            total += QtGui.QColor(img.pixel(x, y)).lightness()
            n += 1
    return total / max(n, 1)


def test_light_theme_tiles_paint_the_light_card_surface(make_workspace, qtbot):
    """DS-08: tiles read the nonexistent tokens.is_dark, so light themes got
    the dark (black) tint and looked muddy grey. They now paint the theme's
    own card surface, so a light theme gives a light tile."""
    from ui_harness import process_events

    from gui.themes.manager import get_manager
    from gui.workspaces.tools import ToolTile

    get_manager().apply("pearl")
    ws = make_workspace("exports")
    tile = ws.findChildren(ToolTile)[0]
    process_events(10)
    light = _mean_lightness(tile)
    get_manager().apply("midnight")
    process_events(10)
    dark = _mean_lightness(tile)
    assert light > 200 and light > dark + 80, (light, dark)


def test_colour_flash_reuses_one_animation_and_never_restyles(make_workspace, qtbot, motion):
    """DS-20: one animation per label, palette only (no setStyleSheet per frame)."""
    from gui.workspaces.label_tone import tone_color, tone_of

    ws = make_workspace("exports")
    label = ws._codec_status
    sheet = label.styleSheet()
    ws._flash_label(label, "success")
    ws._flash_label(label, "danger")
    ws._flash_label(label, "success")
    anims = label.findChildren(QtCore.QVariantAnimation)
    assert len(anims) == 1
    motion.settle(100)
    assert label.styleSheet() == sheet
    assert tone_of(label) == "secondary"
    assert label.palette().color(QtGui.QPalette.ColorRole.WindowText) == tone_color("secondary")


def test_status_tones_follow_theme_changes(make_workspace, qtbot, motion):
    from gui.themes.manager import get_manager
    from gui.workspaces.label_tone import tone_color

    ws = make_workspace("exports")
    ws._flash_label(ws._codec_status, "success")
    motion.settle(100)
    get_manager().apply("pearl")
    expected = QtGui.QColor(get_manager().current.text_secondary)
    assert ws._codec_status.palette().color(QtGui.QPalette.ColorRole.WindowText) == expected
    assert tone_color("secondary") == expected


def test_a_full_width_tile_spans_a_one_column_grid(make_workspace, qtbot):
    """Going from two columns to one left the second column stretching, so a
    lone (or full-width) tile took half the page."""
    from gui.workspaces.tools import TileGrid, ToolTile

    ws = make_workspace("file_cleanup")
    ws.resize(1300, 760)
    process = QtWidgets.QApplication.processEvents
    process()
    ws.resize(900, 760)          # < 2 × TileGrid._TILE_MIN_WIDTH: one column
    for _ in range(5):
        process()
    grid = ws.findChild(TileGrid)
    (tile,) = ws.findChildren(ToolTile)
    assert grid._current_cols == 1
    assert tile.width() >= grid.width() - 2
