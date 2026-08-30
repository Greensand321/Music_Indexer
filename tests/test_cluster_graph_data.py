"""Tests for cluster_graph_data — the Music Graph's data layer.

Deliberately Qt-free: these cover the rules that decide which dot maps to which
track, which is where a mistake silently sends the wrong song to a playlist.
"""
import json
import os

import pytest

import cluster_graph_data as cgd


# ── Fixtures ──────────────────────────────────────────────────────────────


def _payload(n=6, n_clusters=2, with_noise=True, downsampled=False):
    labels = [i % n_clusters for i in range(n)]
    if with_noise:
        labels[-1] = -1
    return {
        "X_2d": [[float(i), float(i * 2)] for i in range(n)],
        "X_3d": [[float(i), float(i * 2), float(i * 3)] for i in range(n)],
        "labels": labels,
        "tracks": [f"/music/track_{i}.mp3" for i in range(n)],
        "cluster_info": {str(c): {"genres": ["Rock"], "tempo": 120} for c in range(n_clusters)},
        "X_downsampled": downsampled,
        "X_total_points": n,
    }


@pytest.fixture
def library(tmp_path):
    docs = tmp_path / "Docs"
    docs.mkdir()
    (docs / "cluster_info.json").write_text(json.dumps(_payload()), encoding="utf-8")
    return tmp_path


# ── Loading & validation ──────────────────────────────────────────────────


def test_load_cluster_data_reads_payload(library):
    data = cgd.load_cluster_data(library)
    assert len(data["tracks"]) == 6


def test_load_missing_file_is_actionable(tmp_path):
    with pytest.raises(cgd.ClusterDataError, match="Run Clustered Playlists"):
        cgd.load_cluster_data(tmp_path)


def test_load_invalid_json_raises(tmp_path):
    docs = tmp_path / "Docs"
    docs.mkdir()
    (docs / "cluster_info.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(cgd.ClusterDataError):
        cgd.load_cluster_data(tmp_path)


def test_validate_rejects_misaligned_payload():
    """The bug that made large libraries unusable: coords subset, labels not."""
    data = _payload(n=6)
    data["X_2d"] = data["X_2d"][:3]  # downsampled coordinates
    with pytest.raises(cgd.ClusterDataError, match="misaligned"):
        cgd.validate_cluster_payload(data)


def test_validate_rejects_empty_points():
    data = _payload()
    data["X_2d"] = []
    data["X_3d"] = []
    data["X"] = []
    data["labels"] = []
    data["tracks"] = []
    with pytest.raises(cgd.ClusterDataError, match="no points"):
        cgd.validate_cluster_payload(data)


def test_validate_reports_missing_keys():
    with pytest.raises(cgd.ClusterDataError, match="labels"):
        cgd.validate_cluster_payload({"X_2d": [[0, 0]], "tracks": ["a"]})


# ── Coordinate selection ──────────────────────────────────────────────────


def test_graph_points_prefers_2d_embedding():
    xy, labels, tracks = cgd.graph_points(_payload(n=3, with_noise=False))
    assert xy == [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]
    assert len(labels) == len(tracks) == 3


def test_graph_points_falls_back_to_3d_when_2d_absent():
    """Older payloads predate X_2d; the first two 3D axes are a usable map."""
    data = _payload(n=3, with_noise=False)
    del data["X_2d"]
    xy, _labels, _tracks = cgd.graph_points(data)
    assert xy == [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]


# ── Cluster bookkeeping ───────────────────────────────────────────────────


def test_normalize_cluster_info_coerces_json_string_keys():
    """JSON keys are strings; the legend looks them up by int."""
    raw = {"0": {"genres": ["Jazz"]}, "1": {"genres": ["Rock"]}}
    info = cgd.normalize_cluster_info(raw)
    assert info[0]["genres"] == ["Jazz"]
    assert set(info) == {0, 1}


def test_normalize_cluster_info_drops_non_numeric_keys():
    assert cgd.normalize_cluster_info({"nope": {}, "2": {}}) == {2: {}}


def test_normalize_cluster_info_handles_none():
    assert cgd.normalize_cluster_info(None) == {}


def test_cluster_counts_includes_noise():
    counts = cgd.cluster_counts([0, 0, 1, -1, -1])
    assert counts == {0: 2, 1: 1, -1: 2}


def test_cluster_order_puts_noise_last():
    assert cgd.cluster_order([2, -1, 0, 1]) == [0, 1, 2, -1]


def test_cluster_order_without_noise():
    assert cgd.cluster_order([1, 0]) == [0, 1]


def test_summarize_mentions_unclustered_only_when_present():
    assert "unclustered" in cgd.summarize([0, 1, -1])
    assert "unclustered" not in cgd.summarize([0, 1])


# ── Selection & export ────────────────────────────────────────────────────


def test_selected_tracks_is_sorted_and_deduplicated():
    tracks = ["a", "b", "c", "d"]
    assert cgd.selected_tracks(tracks, [3, 1, 1]) == ["b", "d"]


def test_selected_tracks_ignores_out_of_range_indices():
    assert cgd.selected_tracks(["a", "b"], [0, 5, -1]) == ["a"]


def test_selection_to_csv_has_header_and_rows():
    csv_text = cgd.selection_to_csv(["/music/a.mp3", "/music/b.mp3"])
    lines = csv_text.strip().splitlines()
    assert lines[0] == "track_path"
    assert lines[1] == "/music/a.mp3"
    assert len(lines) == 3


def test_selection_to_csv_quotes_commas():
    csv_text = cgd.selection_to_csv(["/music/Hello, World.mp3"])
    assert '"/music/Hello, World.mp3"' in csv_text


def test_selection_to_m3u_is_relative_to_the_playlist():
    m3u = cgd.selection_to_m3u(
        ["/lib/Music/a.mp3", "/lib/Music/sub/b.mp3"],
        "/lib/Playlists/sel.m3u",
    )
    lines = m3u.strip().splitlines()
    assert lines[0] == "#EXTM3U"
    assert lines[1] == os.path.join("..", "Music", "a.mp3")
    assert lines[2] == os.path.join("..", "Music", "sub", "b.mp3")


def test_selection_to_m3u_ends_with_newline():
    assert cgd.selection_to_m3u(["/a/b.mp3"], "/a/p.m3u").endswith("\n")
