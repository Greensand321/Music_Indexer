"""Clustered workspace, clustering wizard and quality report.

Covers PL-02/07/08/09 (worker lifecycle and cooperative cancel), PL-03/04
(wizard keeps choices; "kmeans" not "k-means"), PL-05 (report renders),
PL-15 (no promise of cluster playlists), PL-18 (noise is not a cluster),
PL-27 (wizard config kept apart from Quick Start) and PL-31 (library change).
"""
from __future__ import annotations

import numpy as np
import pytest

from gui.compat import QtWidgets
from gui.dialogs.cluster_quality_report_dialog import ClusterQualityReportDialog
from gui.dialogs.clustering_wizard_dialog import ClusteringWizardDialog
from gui.workspaces.clustered_enhanced import compute_cluster_metrics
from playlists_helpers import FakeClustering, make_library
from ui_harness import FrameGapMeter, process_events, visible_texts


def _results_text(ws) -> list[str]:
    return visible_texts(ws._results_widget)


def _ws(make_workspace, tmp_path, n_files: int = 0):
    files = [f"Artist {i % 3}/Album/t{i:02d}.mp3" for i in range(n_files)] or None
    lib = make_library(tmp_path / "lib", files) if files else make_library(tmp_path / "lib")
    return make_workspace("clustered", library_path=lib), lib


def _wait_idle(ws, qtbot, timeout=5000):
    qtbot.waitUntil(lambda: not ws.is_busy(), timeout=timeout)
    process_events(20)


# ── Wizard (PL-03 / PL-04) ───────────────────────────────────────────────


def test_wizard_keeps_every_choice_across_pages(qtbot):
    d = ClusteringWizardDialog()
    qtbot.addWidget(d)
    d._feature_checkboxes["chroma"].setChecked(False)
    d._feature_checkboxes["spectral"].setChecked(False)
    d._feature_checkboxes["onset_rate"].setChecked(True)
    d._on_next()
    d._norm_combo.setCurrentIndex(2)  # robust
    d._on_next()
    d._algo_combo.setCurrentIndex(0)  # K-Means
    d._k_spinbox.setValue(3)
    d._on_next()
    d._on_prev()  # going back must not lose page 3 either
    d._on_next()
    d._on_next()
    cfg = d.get_config()
    assert cfg["features"]["chroma"] is False
    assert cfg["features"]["spectral"] is False
    assert cfg["features"]["onset_rate"] is True
    assert cfg["normalization"] == "robust"
    assert cfg["algorithm"] == "kmeans"
    assert cfg["k"] == 3


def test_wizard_hdbscan_choice_survives(qtbot):
    d = ClusteringWizardDialog()
    qtbot.addWidget(d)
    d._on_next()
    d._on_next()
    d._algo_combo.setCurrentIndex(1)
    d._min_size_spinbox.setValue(9)
    d._min_samples_spinbox.setValue(2)
    d._on_next()
    cfg = d.get_config()
    assert (cfg["algorithm"], cfg["min_cluster_size"], cfg["min_samples"]) == ("hdbscan", 9, 2)


def test_untouched_wizard_defaults_to_kmeans(qtbot):
    d = ClusteringWizardDialog()
    qtbot.addWidget(d)
    for _ in range(5):
        d._on_next()
    assert d.result() == QtWidgets.QDialog.DialogCode.Accepted.value
    assert d.get_config()["algorithm"] == "kmeans"


def test_wizard_blocks_leaving_page_one_without_features(qtbot, dialogs):
    d = ClusteringWizardDialog()
    qtbot.addWidget(d)
    for cb in d._feature_checkboxes.values():
        cb.setChecked(False)
    d._on_next()
    assert d._current_step == 0
    assert dialogs.calls_of("warning")


def test_wizard_does_not_offer_unbuilt_options(qtbot):
    d = ClusteringWizardDialog()
    qtbot.addWidget(d)
    for w in (d._reduction_combo, d._remove_small_cb, d._min_threshold_spin, d._merge_cb,
              d._create_playlists_cb, d._open_graph_cb):
        assert not w.isEnabled()
    cfg = d.get_config()
    assert cfg["create_playlists"] is False
    assert cfg["dimensionality_reduction"] == "none"


# ── Quality report (PL-05) ───────────────────────────────────────────────


def test_quality_report_rows_are_visible(qtbot):
    metrics = {"silhouette_score": 0.42, "davies_bouldin_index": 1.1,
               "calinski_harabasz_score": 230.0, "noise_fraction": 0.1}
    info = {0: {"size": 12}, 1: {"size": 3}}
    d = ClusterQualityReportDialog(metrics, info)
    qtbot.addWidget(d)
    d.show()
    texts = visible_texts(d)
    assert "0.420" in texts and "1.100" in texts and "230.0" in texts
    assert "Cluster 0" in texts and "12 tracks" in texts and "3 tracks" in texts
    assert any(t.startswith("• ") for t in texts), "suggestions are missing"


# ── Worker lifecycle (PL-07 / PL-08 / PL-09 / PL-02) ─────────────────────


def test_run_disables_buttons_and_never_blocks_or_double_runs(make_workspace, tmp_path, qtbot, monkeypatch, threads):
    fake = FakeClustering(hold=True).install(monkeypatch)
    ws, _lib = _ws(make_workspace, tmp_path, 9)
    ws._on_run_quick_start()
    assert fake.started.wait(3)
    assert all(not b.isEnabled() for b in ws._run_buttons)
    with FrameGapMeter() as meter:
        ws._on_run_quick_start()     # a second click must neither block nor restart
        process_events(100)
    assert meter.max_gap_ms < 500
    assert threads.started.count("ClusterWorker") == 1
    fake.release.set()
    _wait_idle(ws, qtbot)
    assert all(b.isEnabled() for b in ws._run_buttons)
    assert "Clusters created:" in _results_text(ws)


def test_cancel_reaches_the_extraction_loop(make_workspace, tmp_path, qtbot, monkeypatch):
    fake = FakeClustering(hold=True).install(monkeypatch)
    ws, lib = _ws(make_workspace, tmp_path, 9)
    statuses = []
    ws.status_changed.connect(lambda text, colour: statuses.append(text))
    ws._on_run_quick_start()
    assert fake.started.wait(3)
    ws._cancel_btn.click()
    _wait_idle(ws, qtbot)
    assert fake.saw_cancel, "cancel_event was not passed to / honoured by the backend"
    assert statuses[-1] == "Cancelled"
    assert not (lib / "Docs" / "cluster_info.json").exists()
    assert any("cancelled" in t.lower() for t in _results_text(ws))


def test_stale_result_from_an_abandoned_run_is_ignored(make_workspace, tmp_path, qtbot, monkeypatch, threads):
    fake = FakeClustering(hold=True).install(monkeypatch)
    ws, _lib = _ws(make_workspace, tmp_path, 6)
    ws._on_run_quick_start()
    assert fake.started.wait(3)
    other = make_library(tmp_path / "other", [f"B/x{i}.mp3" for i in range(4)])
    ws.set_library_path(str(other))  # PL-31: cancels and resets the results tab
    assert not ws.is_busy()
    assert threads.wait_all(5000) == []
    process_events(50)
    assert "Clusters created:" not in _results_text(ws)
    assert ws._current_result is None


def test_closing_the_window_mid_clustering(main_window, tmp_path, qtbot, monkeypatch, dialogs):
    fake = FakeClustering(hold=True).install(monkeypatch)
    lib = make_library(tmp_path / "lib", [f"A/t{i}.mp3" for i in range(5)])
    main_window.set_library(str(lib))
    ws = main_window._workspaces["clustered"]
    ws._on_run_quick_start()
    assert fake.started.wait(3)
    dialogs.answer_yes()
    main_window.close()
    qtbot.waitUntil(lambda: not main_window.isVisible(), timeout=5000)
    assert fake.saw_cancel


# ── Wizard → worker (PL-04 / PL-27) ──────────────────────────────────────


def test_wizard_kmeans_reaches_the_backend_and_survives_quick_start(make_workspace, tmp_path, qtbot, monkeypatch, dialogs):
    fake = FakeClustering().install(monkeypatch)
    ws, _lib = _ws(make_workspace, tmp_path, 9)

    def configure(dlg):
        dlg._on_next(); dlg._on_next()
        dlg._k_spinbox.setValue(3)
        dlg._on_next(); dlg._on_next()
        return QtWidgets.QDialog.DialogCode.Accepted

    dialogs.on_exec(configure)
    ws._on_configure_wizard()
    assert "K-Means Parameters:" in ws._config_display.toPlainText()

    ws._on_run_quick_start()
    _wait_idle(ws, qtbot)
    ws._on_run_advanced()
    _wait_idle(ws, qtbot)
    assert fake.calls[0]["method"] == "kmeans" and fake.calls[0]["params"] == {"n_clusters": 8}
    assert fake.calls[1]["method"] == "kmeans" and fake.calls[1]["params"] == {"n_clusters": 3}


# ── Results (PL-15 / PL-18) ──────────────────────────────────────────────


def test_noise_is_not_counted_as_a_cluster(make_workspace, tmp_path, qtbot, monkeypatch):
    FakeClustering(n_clusters=2, noise=3).install(monkeypatch)
    ws, _lib = _ws(make_workspace, tmp_path, 10)
    ws._on_run_quick_start()
    _wait_idle(ws, qtbot)
    texts = _results_text(ws)
    assert texts[texts.index("Clusters created:") + 1] == "2"
    assert "Unclustered (noise):" in texts


def test_metrics_exclude_noise():
    X = np.array([[0, 0], [0, 0.1], [5, 5], [5, 5.1], [100, -100]], dtype=float)
    labels = np.array([0, 0, 1, 1, -1])
    m = compute_cluster_metrics(X, labels)
    clean = compute_cluster_metrics(X[:4], labels[:4])
    assert m["n_clusters"] == 2
    assert m["noise_count"] == 1
    assert m["silhouette_score"] == pytest.approx(clean["silhouette_score"])


def test_metrics_need_two_clusters():
    m = compute_cluster_metrics(np.zeros((4, 2)), np.array([0, 0, -1, -1]))
    assert "silhouette_score" not in m and m["n_clusters"] == 1


def test_results_do_not_promise_cluster_playlists(make_workspace, tmp_path, qtbot, monkeypatch):
    FakeClustering().install(monkeypatch)
    ws, _lib = _ws(make_workspace, tmp_path, 9)
    ws._on_run_quick_start()
    _wait_idle(ws, qtbot)
    texts = _results_text(ws)
    assert not any("View Playlists" in t for t in texts)
    assert any("Playlists are not created automatically" in t for t in texts)
    assert any("Open Visual Graph" in t for t in texts)


def test_reserved_folders_are_not_clustered(make_workspace, tmp_path, qtbot, monkeypatch):
    fake = FakeClustering().install(monkeypatch)
    ws, _lib = _ws(make_workspace, tmp_path)  # standard files incl. Docs/Quarantine/Trash
    ws._on_run_quick_start()
    _wait_idle(ws, qtbot)
    tracks = fake.calls[0]["tracks"]
    assert tracks and not any(part in t for t in tracks for part in ("Quarantine", "Trash", "Docs", "Not Sorted"))
