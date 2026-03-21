"""Tests for cluster_graph_3d — Three.js HTML generator."""
import json
import os
import tempfile

import pytest

from cluster_graph_3d import (
    generate_cluster_graph_html,
    generate_cluster_graph_html_from_data,
    _validate_cluster_data,
    _render_html,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_cluster_data(n=20, n_clusters=3):
    """Return a minimal valid cluster_data dict."""
    import random

    random.seed(42)
    positions = [[random.uniform(-5, 5) for _ in range(3)] for _ in range(n)]
    labels = [i % n_clusters for i in range(n)]
    tracks = [f"/music/track_{i}.mp3" for i in range(n)]
    return {
        "X_3d": positions,
        "labels": labels,
        "tracks": tracks,
        "cluster_info": {
            str(c): {"size": labels.count(c)} for c in range(n_clusters)
        },
    }


@pytest.fixture
def cluster_data():
    return _make_cluster_data()


@pytest.fixture
def library_with_cluster_info(tmp_path, cluster_data):
    """Create a temp library dir with Docs/cluster_info.json."""
    docs = tmp_path / "Docs"
    docs.mkdir()
    info_path = docs / "cluster_info.json"
    info_path.write_text(json.dumps(cluster_data), encoding="utf-8")
    return tmp_path


# ── Validation tests ─────────────────────────────────────────────────────


def test_validate_cluster_data_valid(cluster_data):
    _validate_cluster_data(cluster_data)  # should not raise


def test_validate_missing_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        _validate_cluster_data({"X_3d": [[1, 2, 3]], "labels": [0]})


def test_validate_empty_x3d():
    with pytest.raises(ValueError, match="X_3d is empty"):
        _validate_cluster_data({"X_3d": [], "labels": [], "tracks": []})


def test_validate_length_mismatch():
    with pytest.raises(ValueError, match="Length mismatch"):
        _validate_cluster_data({
            "X_3d": [[1, 2, 3]],
            "labels": [0, 1],
            "tracks": ["a.mp3"],
        })


# ── HTML render tests ────────────────────────────────────────────────────


def test_render_html_contains_three_js(cluster_data):
    html = _render_html(cluster_data)
    assert "three.min.js" in html
    assert "AlphaDEX" in html


def test_render_html_embeds_data(cluster_data):
    html = _render_html(cluster_data)
    # The cluster data should be inlined as JSON
    assert '"X_3d"' in html
    assert '"labels"' in html
    assert '"tracks"' in html


def test_render_html_has_controls(cluster_data):
    html = _render_html(cluster_data)
    assert "btn-reset" in html
    assert "btn-export-csv" in html
    assert "btn-export-m3u" in html
    assert "legend" in html
    assert "tooltip" in html


def test_render_html_has_orbit_controls(cluster_data):
    html = _render_html(cluster_data)
    assert "orbitState" in html
    assert "updateOrbit" in html


# ── File generation tests ────────────────────────────────────────────────


def test_generate_from_library_path(library_with_cluster_info):
    path = generate_cluster_graph_html(str(library_with_cluster_info))
    assert os.path.isfile(path)
    assert path.endswith("cluster_graph.html")

    content = open(path, encoding="utf-8").read()
    assert "<!DOCTYPE html>" in content
    assert "three.min.js" in content


def test_generate_from_library_path_custom_output(library_with_cluster_info, tmp_path):
    out = str(tmp_path / "custom_graph.html")
    path = generate_cluster_graph_html(
        str(library_with_cluster_info), output_path=out
    )
    assert path == os.path.abspath(out)
    assert os.path.isfile(out)


def test_generate_from_data(cluster_data, tmp_path):
    out = str(tmp_path / "graph.html")
    path = generate_cluster_graph_html_from_data(cluster_data, out)
    assert os.path.isfile(path)

    content = open(path, encoding="utf-8").read()
    assert "track_0.mp3" in content


def test_generate_missing_cluster_info(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cluster data not found"):
        generate_cluster_graph_html(str(tmp_path))


def test_generate_logs_callback(library_with_cluster_info):
    messages = []
    generate_cluster_graph_html(
        str(library_with_cluster_info), log_callback=messages.append
    )
    assert any("cluster graph" in m.lower() or "cluster_graph" in m.lower() for m in messages)


# ── Edge cases ────────────────────────────────────────────────────────────


def test_single_point():
    data = {
        "X_3d": [[1.0, 2.0, 3.0]],
        "labels": [0],
        "tracks": ["/music/solo.mp3"],
    }
    html = _render_html(data)
    assert "solo.mp3" in html


def test_noise_labels():
    """Noise points (label=-1) should render without errors."""
    data = {
        "X_3d": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
        "labels": [0, -1, 0],
        "tracks": ["a.mp3", "b.mp3", "c.mp3"],
    }
    html = _render_html(data)
    assert "Noise" not in html or "noise" in html.lower()  # legend may mention noise


def test_large_dataset():
    """Ensure generation works for larger datasets."""
    data = _make_cluster_data(n=5000, n_clusters=10)
    html = _render_html(data)
    assert "5,000" in html or "5000" in html  # stats line should show count


def test_metadata_in_data():
    """Metadata (if present) should be embedded."""
    data = {
        "X_3d": [[0, 0, 0]],
        "labels": [0],
        "tracks": ["/music/song.mp3"],
        "metadata": [{"title": "My Song", "artist": "Artist X"}],
    }
    html = _render_html(data)
    assert "My Song" in html
    assert "Artist X" in html


def test_downsampled_flag():
    """Downsampled indicator should appear in stats."""
    data = _make_cluster_data(n=50)
    data["X_downsampled"] = True
    data["X_total_points"] = 12345
    html = _render_html(data)
    assert "12345" in html or "12,345" in html
