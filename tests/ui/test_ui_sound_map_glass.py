"""Sound Map in the Liquid Glass look: Analyse, Map and the three dialogs.

Covers the restyle's new state handling and layout, not pixels:

* Analyse: the empty state before the first run, the hero card, the
  features as pill toggles in the engine's order (and what Run sends), the
  results StatRow, the results card's Save cluster playlists mirroring the
  page flow, no horizontal scroll at 900 px.
* Map: glyph buttons with tooltips and accessible names, the segmented
  mode control, the empty state routing to Analyse, the side cards moving
  below the map in a narrow window, glass legend rows, the selection dimming
  the base layer, one cached point sprite per cluster, the hover ring, and
  the frame-gap check on pan, zoom and hover over 5,000 points.
* Dialogs: the wizard's step pills and pre-fill, the cluster playlists
  table (34 px rows, no grid, round colour dots, warnings well).
"""
from __future__ import annotations

import pytest

from gui.compat import QtCore, QtWidgets
from playlists_helpers import FakeClustering, make_library, write_cluster_payload
from ui_harness import FrameGapMeter, process_events


def _analyse(make_workspace, tmp_path, n_files: int = 9):
    files = [f"Artist {i % 3}/Album/t{i:02d}.mp3" for i in range(n_files)]
    lib = make_library(tmp_path / "lib", files)
    return make_workspace("clustered", library_path=lib), lib


def _wait_idle(ws, qtbot, timeout=5000):
    qtbot.waitUntil(lambda: not ws.is_busy(), timeout=timeout)
    process_events(20)


@pytest.fixture(autouse=True)
def _no_disk_tags(monkeypatch):
    import utils.audio_metadata_reader as amr
    monkeypatch.setattr(amr, "read_metadata", lambda path, include_cover=False: ({}, [], None, None))
    monkeypatch.setattr(amr, "read_tags", lambda path: {})


# ── Analyse ───────────────────────────────────────────────────────────────


def test_analyse_starts_with_an_empty_state_and_a_hero(make_workspace, tmp_path):
    from gui.widgets.glass_parts import EmptyState

    ws, _lib = _analyse(make_workspace, tmp_path)
    empties = [e for e in ws._results_widget.findChildren(EmptyState) if e.isVisibleTo(ws)]
    assert len(empties) == 1 and empties[0].title() == "No analysis yet"
    assert ws.stat_values() == {}
    # The page header and the section tab already name the page: no repeat.
    assert not [lb for lb in ws.findChildren(QtWidgets.QLabel) if lb.objectName() == "sectionTitle"]
    hero = ws._quick_btn.parentWidget()
    assert hero.objectName() == "headerCard"
    assert ws._quick_btn.objectName() == "primaryBtn"
    assert ws._quick_btn.property("glyphName") == "play"
    assert ws._wizard_btn.parentWidget() is hero


def test_feature_pills_follow_the_engine_order():
    import clustered_playlists as cp
    from gui.dialogs.clustering_wizard_dialog import FEATURE_CHOICES

    assert [key for key, _label, _tip in FEATURE_CHOICES] == list(cp.FEATURE_ORDER)


def test_settings_card_runs_exactly_what_it_shows(make_workspace, tmp_path, qtbot, monkeypatch, dialogs):
    from gui.dialogs.clustering_wizard_dialog import PillToggle

    fake = FakeClustering().install(monkeypatch)
    ws, _lib = _analyse(make_workspace, tmp_path)
    toggles = ws._feature_toggles
    assert all(isinstance(t, PillToggle) for t in toggles.values())
    assert list(toggles) == ["mfcc", "tempo", "chroma", "spectral", "energy", "onset_rate"]
    # Starts from the Quick Start settings.
    assert {k for k, t in toggles.items() if t.isChecked()} == {"mfcc", "tempo", "energy"}
    toggles["mfcc"].click()
    toggles["chroma"].click()
    ws._k_spin.setValue(5)
    ws._norm_combo.setCurrentIndex(ws._norm_combo.findData("robust"))
    assert "harmony" in ws._settings_summary.text() and "5 clusters" in ws._settings_summary.text()
    ws._advanced_btn.click()
    _wait_idle(ws, qtbot)
    call = fake.calls[0]
    assert call["features"] == {"mfcc": False, "tempo": True, "chroma": True, "spectral": False,
                                "energy": True, "onset_rate": False}
    assert call["params"] == {"n_clusters": 5} and call["normalization"] == "robust"
    # Nothing ticked: Run waits for a feature.
    for t in toggles.values():
        t.setChecked(False)
    assert not ws._advanced_btn.isEnabled()


def test_settings_card_hdbscan_parameters(make_workspace, tmp_path, qtbot, monkeypatch):
    fake = FakeClustering().install(monkeypatch)
    ws, _lib = _analyse(make_workspace, tmp_path)
    ws._algo_combo.setCurrentIndex(ws._algo_combo.findData("hdbscan"))
    assert not ws._k_spin.isVisibleTo(ws) and ws._min_size_spin.isVisibleTo(ws)
    ws._min_size_spin.setValue(4)
    ws._min_samples_spin.setValue(2)
    ws._advanced_btn.click()
    _wait_idle(ws, qtbot)
    assert fake.calls[0]["method"] == "hdbscan"
    assert fake.calls[0]["params"] == {"min_cluster_size": 4, "min_samples": 2}


def test_guided_setup_opens_on_the_card_and_fills_it_back(make_workspace, tmp_path, dialogs):
    ws, _lib = _analyse(make_workspace, tmp_path)
    ws._feature_toggles["onset_rate"].setChecked(True)
    seen = {}

    def answer(dlg):
        seen["prefilled"] = dlg._feature_checkboxes["onset_rate"].isChecked()
        dlg._feature_checkboxes["chroma"].setChecked(True)
        dlg._on_next(); dlg._on_next()
        dlg._algo_combo.setCurrentIndex(1)
        dlg._min_size_spinbox.setValue(7)
        dlg._on_next(); dlg._on_next(); dlg._on_next()
        return QtWidgets.QDialog.DialogCode.Accepted

    dialogs.on_exec(answer)
    ws._wizard_btn.click()
    assert seen["prefilled"] is True
    cfg = ws.settings_config()
    assert cfg["algorithm"] == "hdbscan" and cfg["min_cluster_size"] == 7
    assert cfg["features"]["chroma"] and cfg["features"]["onset_rate"]


def test_results_are_a_stat_row_with_next_steps(make_workspace, tmp_path, qtbot, monkeypatch):
    FakeClustering(n_clusters=3, noise=2).install(monkeypatch)
    ws, _lib = _analyse(make_workspace, tmp_path, n_files=12)
    ws._on_run_quick_start()
    _wait_idle(ws, qtbot)
    stats = ws.stat_values()
    assert stats["clusters"] == "3" and stats["tracks"] == "12" and stats["noise"] == "2"
    assert "silhouette" in stats
    tone = ws._stats.tile("silhouette").tone()
    assert tone in ("success", "warning", "danger")
    texts = [b.text() for b in ws._results_widget.findChildren(QtWidgets.QPushButton)
             if b.isVisibleTo(ws)]
    assert "Open the map" in texts and "Save cluster playlists…" in texts


def _sound_map(qtbot, tmp_path, n_files: int = 12):
    from gui import navigation
    from gui.pages import SoundMapPage
    from gui.workspaces.clustered_enhanced import EnhancedClusteredWorkspace
    from gui.workspaces.graph import GraphWorkspace

    files = [f"Artist {i % 3}/Album/t{i:02d}.mp3" for i in range(n_files)]
    lib = make_library(tmp_path / "lib", files)
    analysis = EnhancedClusteredWorkspace(library_path=str(lib))
    graph = GraphWorkspace(library_path=str(lib))
    page = SoundMapPage(navigation.PAGE_BY_KEY["sound_map"], {"clustered": analysis, "graph": graph})
    qtbot.addWidget(page)
    page.resize(1100, 760)
    page.show()
    return page, analysis, graph


def test_results_save_button_runs_the_page_flow(qtbot, tmp_path, monkeypatch, dialogs):
    from gui.dialogs.cluster_playlists_dialog import ClusterPlaylistsDialog

    FakeClustering(write_members=True).install(monkeypatch)
    page, ws, _graph = _sound_map(qtbot, tmp_path)
    flow = page.cluster_playlists_flow
    assert flow.button.property("glyphName") == "save"
    assert page._report_btn.property("glyphName") == "info"
    ws._on_run_quick_start()
    _wait_idle(ws, qtbot)
    qtbot.waitUntil(flow.button.isEnabled, timeout=5000)
    assert ws._save_btn.isEnabled() and ws._save_btn.toolTip() == flow.button.toolTip()
    shown = []
    dialogs.on_exec(lambda dlg: (shown.append(type(dlg)), QtWidgets.QDialog.DialogCode.Rejected)[1])
    ws._save_btn.click()
    assert not ws._save_btn.isEnabled(), "the results button waits while the preview runs"
    qtbot.waitUntil(lambda: bool(shown), timeout=8000)
    assert shown == [ClusterPlaylistsDialog]
    qtbot.waitUntil(lambda: not flow.is_busy(), timeout=5000)
    qtbot.waitUntil(ws._save_btn.isEnabled, timeout=5000)


# ── Narrow window: no horizontal scroll ───────────────────────────────────


@pytest.mark.parametrize("size", [(900, 600), (1300, 860), (1600, 1000)])
def test_sound_map_never_scrolls_sideways(main_window, tmp_path, size):
    lib = tmp_path / "lib"
    lib.mkdir()
    write_cluster_payload(lib, n=40, n_clusters=6, noise=3)
    main_window.resize(*size)
    main_window.set_library(str(lib))
    analysis = main_window._workspaces["clustered"]
    result = {"labels": [i % 6 for i in range(40)], "failed": {"x": "y"},
              "metrics": {"silhouette_score": 0.44}, "config": {"algorithm": "kmeans"}}
    analysis._clear_results()
    analysis._set_result(result)
    analysis._show_results_panel(result)
    for key in ("sound_map/analyse", "sound_map/map"):
        main_window._on_nav_changed(key)
        process_events(80)
        panel = main_window.current_panel()
        bar = panel._scroll_area.horizontalScrollBar()
        assert bar.maximum() == 0, f"{key} scrolls sideways at {size}"
    graph = main_window._workspaces["graph"]
    assert graph.layout_tier() == ("narrow" if size[0] == 900 else "wide")
    assert analysis.feature_columns() == (3 if size[0] == 900 else 6)


# ── Map ───────────────────────────────────────────────────────────────────


def _graph(make_workspace, tmp_path, **payload):
    pytest.importorskip("pyqtgraph")
    lib = tmp_path / "lib"
    lib.mkdir(exist_ok=True)
    if payload.pop("write", True):
        write_cluster_payload(lib, **payload)
    return make_workspace("graph", library_path=lib), lib


def test_map_tools_are_glyph_buttons(make_workspace, tmp_path):
    ws, _lib = _graph(make_workspace, tmp_path, n=9)
    expected = {
        ws._fit_btn: "fit", ws._clear_btn: "close", ws._reload_btn: "refresh",
        ws._browser_btn: "external", ws._play_btn: "play", ws._m3u_btn: "save",
        ws._csv_btn: "download",
    }
    for button, glyph in expected.items():
        assert button.property("glyphName") == glyph
        assert button.toolTip()
    for button in (ws._fit_btn, ws._clear_btn, ws._reload_btn):
        assert not button.text() and button.accessibleName()
    assert ws._play_btn.objectName() == "primaryBtn"
    assert [b.property("glyphName") for b in ws._mode_group.buttons()] == ["move", "marquee", "lasso"]


def test_mode_control_raises_the_chosen_mode(make_workspace, tmp_path):
    from gui.widgets.interactive_scatter_plot import MODE_LASSO, MODE_PAN

    ws, _lib = _graph(make_workspace, tmp_path, n=9)
    pan, box, lasso = ws._mode_group.buttons()
    assert pan.isChecked() and not pan.isFlat() and box.isFlat() and lasso.isFlat()
    lasso.click()
    assert ws._plot.mode() == MODE_LASSO
    assert not lasso.isFlat() and pan.isFlat()
    pan.click()
    assert ws._plot.mode() == MODE_PAN


def test_empty_map_offers_the_analyse_tab(make_workspace, tmp_path, qtbot):
    ws, lib = _graph(make_workspace, tmp_path, write=False)
    assert ws._plot_stack.currentWidget() is ws._empty_panel
    assert ws._empty.title() == "No map yet"
    button = ws._empty.action_button()
    assert button.isVisibleTo(ws) and button.text() == "Go to Analyse"
    with qtbot.waitSignal(ws.navigate_requested) as sig:
        button.click()
    assert sig.args == ["clustered"]
    write_cluster_payload(lib, n=6)
    ws.reload_data()
    assert ws._plot_stack.currentWidget() is ws._plot
    write_cluster_payload(lib, kind="malformed")
    ws.reload_data()
    assert ws._plot_stack.currentWidget() is ws._empty_panel
    assert ws._empty.title() == "The map could not be loaded"


def test_map_without_a_library_says_so(make_workspace):
    pytest.importorskip("pyqtgraph")
    ws = make_workspace("graph")
    assert ws._empty.title() == "Choose a library first"
    assert not ws._empty.action_button().isVisibleTo(ws)


def test_legend_rows_are_painted_glass_rows(make_workspace, tmp_path):
    from gui.widgets.cluster_legend import _DotToggle

    ws, _lib = _graph(make_workspace, tmp_path, n=12, n_clusters=3)
    rows = ws._legend.rows()
    assert len(rows) == 3
    for row in rows.values():
        assert isinstance(row.checkbox, _DotToggle) and row.checkbox.colour() is not None
        assert not row.styleSheet() and not row.checkbox.styleSheet()
        assert row.count_label.text() == "4"
    row = rows[1]
    row.checkbox.click()                       # the dot hides the cluster
    assert 1 not in ws._plot.visible_clusters()
    assert not row.label.isEnabled() and not row.is_shown()
    ws._legend.set_all_visible(True)
    assert row.is_shown() and row.label.isEnabled()


def test_selection_dims_the_rest_without_redrawing_it(make_workspace, tmp_path, monkeypatch):
    ws, _lib = _graph(make_workspace, tmp_path, n=30, n_clusters=3)
    plot = ws._plot
    calls = []
    real = plot.scatter.setData
    monkeypatch.setattr(plot.scatter, "setData", lambda *a, **k: (calls.append(1), real(*a, **k)))
    assert plot.scatter.opacity() == 1.0
    ws._on_cluster_selected(2)
    assert plot.scatter.opacity() < 1.0
    assert len(plot.selection_scatter.data) == 10
    plot.clear_selection()
    assert plot.scatter.opacity() == 1.0
    assert calls == []


def test_one_cached_point_sprite_per_cluster(make_workspace, tmp_path):
    ws, _lib = _graph(make_workspace, tmp_path, n=600, n_clusters=5, noise=12)
    scatter = ws._plot.scatter
    brushes = {id(b) for b in scatter.data["brush"]}
    assert len(brushes) == 6                   # 5 clusters + Unclustered
    ws.grab()                                  # paint once: the atlas renders the sprites
    assert 0 < len(scatter.fragmentAtlas) <= 6


def test_hover_ring_follows_the_cursor(make_workspace, tmp_path, qtbot):
    ws, _lib = _graph(make_workspace, tmp_path, n=9)
    plot = ws._plot
    with qtbot.waitSignal(plot.hover_changed) as sig:
        plot._set_hovered(4)
    assert sig.args == [4] and len(plot.hover_scatter.data) == 1
    plot._set_hovered(-1)
    assert len(plot.hover_scatter.data) == 0


def test_map_pan_zoom_and_hover_stay_smooth(make_workspace, tmp_path, qtbot):
    """FrameGapMeter over panning, zooming and hovering a 5,000-point map:
    no frame-sized freezes (the sprites are cached pixmaps, no per-point
    effects, and hover moves a one-point layer)."""
    ws, _lib = _graph(make_workspace, tmp_path, n=5000, n_clusters=8, noise=40)
    qtbot.waitExposed(ws)
    plot = ws._plot
    view = plot._view_box
    process_events(100)
    xy = ws._xy
    with FrameGapMeter() as meter:
        for i in range(24):
            view.translateBy(x=0.15 if i % 2 else -0.15, y=0.1)
            view.scaleBy((0.96, 0.96) if i < 12 else (1.04, 1.04))
            x, y = xy[(i * 197) % len(xy)]
            plot._on_mouse_moved(view.mapViewToScene(QtCore.QPointF(x, y)))
            plot.plot_widget.viewport().repaint()
            process_events(4)
    print(f"\nSound Map pan/zoom/hover, 5,000 points: {meter.summary()}")
    assert meter.max_gap_ms < 100, meter.summary()
    assert plot.hovered() >= -1


# ── Dialogs ───────────────────────────────────────────────────────────────


def test_wizard_step_pills_show_the_real_step(qtbot):
    from gui.dialogs.clustering_wizard_dialog import ClusteringWizardDialog

    d = ClusteringWizardDialog()
    qtbot.addWidget(d)
    assert d._steps.current() == 0
    d._on_next()
    d._on_next()
    assert d._steps.current() == 2 and d._step_indicator.text() == "3 of 5"
    d._on_prev()
    assert d._steps.current() == 1
    assert d._next_btn.objectName() == "primaryBtn"


def test_wizard_prefill_round_trips(qtbot):
    from gui.dialogs.clustering_wizard_dialog import ClusteringWizardDialog

    d = ClusteringWizardDialog()
    qtbot.addWidget(d)
    cfg = {"features": {"mfcc": False, "tempo": True, "chroma": False, "spectral": True,
                        "energy": False, "onset_rate": True},
           "normalization": "minmax", "algorithm": "hdbscan", "k": 4,
           "min_cluster_size": 9, "min_samples": 3, "create_playlists": True}
    d.set_config(cfg)
    out = d.get_config()
    for key in ("features", "normalization", "algorithm", "k", "min_cluster_size",
                "min_samples", "create_playlists"):
        assert out[key] == cfg[key], key


def test_cluster_playlists_table_is_glass_styled(qtbot, tmp_path):
    pytest.importorskip("pyqtgraph")
    import cluster_playlist_export as cpe
    from _helpers.cluster_payloads import write_cluster_run
    from gui.dialogs.cluster_playlists_dialog import ClusterPlaylistsDialog
    from gui.widgets.glass import GlassCard
    from gui.widgets.interactive_scatter_plot import _cluster_colour

    lib = tmp_path / "lib"
    lib.mkdir()
    write_cluster_run(lib, files=True, n=30, n_clusters=3)
    summary = cpe.summarize_cluster_run(str(lib))
    d = ClusterPlaylistsDialog(summary, {0, 1, 2}, colour_for=lambda c: _cluster_colour(c, 3))
    qtbot.addWidget(d)
    d.show()
    table = d.table
    assert not table.showGrid()
    assert table.verticalHeader().defaultSectionSize() == 34
    assert isinstance(table.parentWidget().parentWidget(), GlassCard) or any(
        isinstance(w, GlassCard) for w in d.findChildren(GlassCard))
    assert not table.item(0, 1).icon().isNull()
    assert not d._warnings_panel.isVisibleTo(d)
    assert d.save_btn.property("glyphName") == "save"
    assert d.cancel_btn.isDefault()
