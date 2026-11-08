"""Music Graph workspace and its widgets.

Covers PL-06 (reload after a clustering run), PL-19 (hover vs pinned track),
PL-20 (hidden clusters leave the selection), PL-21 (plot follows the theme),
PL-22 (failed reload clears the map), PL-25 (a selection change does not
redraw every point), PL-29 (real pyqtgraph import error) and PL-30.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("pyqtgraph")

from gui.compat import QtGui
from gui.themes.manager import get_manager
import gui.widgets.interactive_scatter_plot as isp
from playlists_helpers import write_cluster_payload
from ui_harness import process_events


@pytest.fixture(autouse=True)
def _no_disk_tags(monkeypatch):
    import utils.audio_metadata_reader as amr
    monkeypatch.setattr(amr, "read_metadata", lambda path, include_cover=False: ({}, [], None, None))


def _graph(make_workspace, tmp_path, **payload):
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    if payload.pop("write", True):
        write_cluster_payload(lib, **payload)
    return make_workspace("graph", library_path=lib), lib


def _rendered(ws) -> int:
    return len(ws._plot.scatter.data)


# ── PL-06 ────────────────────────────────────────────────────────────────


def test_graph_reloads_when_shown_after_a_clustering_run(main_window, tmp_path, qtbot):
    lib = tmp_path / "lib"
    lib.mkdir()
    main_window.set_library(str(lib))
    graph = main_window._workspaces["graph"]
    assert "No cluster data" in graph._status.text()

    main_window._on_nav_changed("clustered")
    write_cluster_payload(lib, n=9)          # what a clustering run leaves behind
    main_window._on_nav_changed("graph")
    process_events(20)
    assert "No cluster data" not in graph._status.text()
    assert _rendered(graph) == 9


def test_graph_does_not_reload_when_nothing_changed(make_workspace, tmp_path, monkeypatch):
    ws, _lib = _graph(make_workspace, tmp_path, n=6)
    calls = []
    monkeypatch.setattr(ws, "reload_data", lambda: calls.append(1))
    ws.hide()
    ws.show()
    assert calls == []


# ── PL-22 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["malformed", "misaligned"])
def test_failed_reload_clears_the_previous_map(make_workspace, tmp_path, kind):
    ws, lib = _graph(make_workspace, tmp_path, n=8)
    assert _rendered(ws) == 8 and ws._legend._rows
    write_cluster_payload(lib, kind=kind, n=8)
    ws.reload_data()
    assert _rendered(ws) == 0
    assert not ws._legend._rows
    assert not ws._fit_btn.isEnabled()
    assert ws._status.text()


# ── PL-20 ────────────────────────────────────────────────────────────────


def test_hiding_a_cluster_drops_it_from_the_selection(make_workspace, tmp_path):
    ws, _lib = _graph(make_workspace, tmp_path, n=9, n_clusters=3)
    ws._on_cluster_selected(1)
    assert ws._sel_label.text() == "3 tracks selected"
    ws._legend._rows[1].checkbox.setChecked(False)
    assert ws._plot.get_selection() == []
    assert ws._sel_label.text() == "Nothing selected"
    assert not ws._play_btn.isEnabled()
    # Clicking a hidden cluster's row selects nothing invisible.
    ws._on_cluster_selected(1)
    assert ws._plot.get_selection() == []


def test_lasso_ignores_hidden_points(make_workspace, tmp_path):
    from gui.compat import QtCore

    ws, _lib = _graph(make_workspace, tmp_path, n=9, n_clusters=3)
    ws._plot.set_cluster_visible(0, False)
    ws._plot.set_mode(isp.MODE_RECT)
    ws._plot._on_drag_started(QtCore.QPointF(-1, -1))
    ws._plot._on_drag_finished(QtCore.QPointF(100, 100))
    sel = ws._plot.get_selection()
    assert sel and all(ws._labels[i] != 0 for i in sel)


# ── PL-19 ────────────────────────────────────────────────────────────────


def test_hover_does_not_replace_the_clicked_track(make_workspace, tmp_path, qtbot):
    ws, _lib = _graph(make_workspace, tmp_path, n=6)
    ws._plot.set_selection([0])
    ws._on_point_clicked(0)
    ws._on_hover(3)            # mouse crosses another point on its way to ▶ Play
    assert ws._details.current_path() == ws._tracks[0]
    with qtbot.waitSignal(ws.play_tracks_requested) as sig:
        ws._details.play_btn.click()
    assert sig.args[0] == [ws._tracks[0]]


def test_hover_previews_when_nothing_is_pinned(make_workspace, tmp_path):
    ws, _lib = _graph(make_workspace, tmp_path, n=6)
    ws._on_hover(2)
    assert ws._details.current_path() == ws._tracks[2]
    ws._plot.set_selection([4])           # single selection pins
    ws._on_hover(1)
    assert ws._details.current_path() == ws._tracks[4]
    ws._plot.clear_selection()            # unpins
    ws._on_hover(1)
    assert ws._details.current_path() == ws._tracks[1]


# ── PL-21 ────────────────────────────────────────────────────────────────


def test_plot_background_follows_theme_switches(make_workspace, tmp_path):
    ws, _lib = _graph(make_workspace, tmp_path, n=4)
    mgr = get_manager()
    for key in ("pearl", "midnight"):
        mgr.apply(key)
        expected = QtGui.QColor(mgr.current.card_bg).name()
        assert ws._plot.plot_widget.backgroundBrush().color().name() == expected
        assert ws._plot._highlight_pen.color().name() == QtGui.QColor(mgr.current.text_primary).name()


# ── PL-25 ────────────────────────────────────────────────────────────────


def test_selection_change_does_not_redraw_every_point(make_workspace, tmp_path, monkeypatch):
    ws, _lib = _graph(make_workspace, tmp_path, n=5000, n_clusters=8)
    assert _rendered(ws) == 5000
    calls = []
    real = ws._plot.scatter.setData
    monkeypatch.setattr(ws._plot.scatter, "setData", lambda *a, **k: (calls.append(1), real(*a, **k)))
    t0 = time.perf_counter()
    for i in range(20):
        ws._plot.set_selection([i])
    per_click = (time.perf_counter() - t0) / 20
    assert calls == [], "a selection change rebuilt the 5,000-point base layer"
    assert per_click < 0.05
    assert len(ws._plot.selection_scatter.data) == 1


def test_large_payload_loads_aligned(make_workspace, tmp_path):
    ws, _lib = _graph(make_workspace, tmp_path, n=5000, n_clusters=6, noise=40)
    assert len(ws._tracks) == len(ws._labels) == len(ws._xy) == 5000
    assert sum(r for r in [len(ws._legend._rows)]) == 7  # 6 clusters + Unclustered


# ── PL-29 / PL-30 ────────────────────────────────────────────────────────


def test_unavailable_reason_reports_the_real_import_error(monkeypatch):
    monkeypatch.setattr(isp, "PG_IMPORT_ERROR", ModuleNotFoundError("No module named 'colorama'", name="colorama"))
    text = isp.unavailable_reason()
    assert "could not be loaded" in text and "colorama" in text
    monkeypatch.setattr(isp, "PG_IMPORT_ERROR", ModuleNotFoundError("No module named 'pyqtgraph'", name="pyqtgraph"))
    assert "not installed" in isp.unavailable_reason()


def test_controls_stay_disabled_without_a_plot(make_workspace, tmp_path, monkeypatch):
    monkeypatch.setattr(isp, "pg", None)
    monkeypatch.setattr(isp, "PG_IMPORT_ERROR", ImportError("broken"))
    ws, _lib = _graph(make_workspace, tmp_path, n=4)
    assert not ws._plot.available
    assert not ws._fit_btn.isEnabled()
    assert not any(b.isEnabled() for b in ws._mode_group.buttons())


def test_legend_has_room_for_several_rows(make_workspace, tmp_path):
    ws, _lib = _graph(make_workspace, tmp_path, n=12, n_clusters=6)
    assert ws._legend.minimumHeight() >= 200


def test_empty_map_points_at_the_analyse_tab(make_workspace, tmp_path):
    """The data layer still says "Run Clustered Playlists"; on the Sound Map
    that step is the Analyse tab."""
    lib = tmp_path / "Lib"
    lib.mkdir()
    graph = make_workspace("graph", library_path=lib)
    text = graph._status.text()
    assert "No cluster data" in text and "Analyse tab" in text
    assert "Clustered Playlists" not in text
