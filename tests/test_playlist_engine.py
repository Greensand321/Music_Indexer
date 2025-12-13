import os

from playlist_engine import (
    bucket_by_tempo_energy,
    categorize_tempo,
    categorize_energy,
    more_like_this,
    autodj_playlist,
)


def test_categorize_helpers():
    assert categorize_tempo(80) == "slow"
    assert categorize_tempo(110) == "medium"
    assert categorize_tempo(130) == "fast"

    assert categorize_energy(0.05) == "low"
    assert categorize_energy(0.2) == "medium"
    assert categorize_energy(0.4) == "high"


def test_more_like_this_and_autodj():
    tracks = ["a", "b", "c"]
    feats = {
        "a": [0.0, 0.0],
        "b": [1.0, 1.0],
        "c": [2.0, 2.0],
    }

    sim = more_like_this("a", tracks, n=2, feature_cache=feats)
    assert sim == ["b", "c"]

    dj = autodj_playlist("a", tracks, n=3, feature_cache=feats)
    assert dj == ["a", "b", "c"]


def test_bucket_by_tempo_energy(tmp_path):
    music = tmp_path / "Music"
    music.mkdir()

    track_a = music / "slow_low.mp3"
    track_b = music / "mid_high.mp3"
    track_c = music / "fast_mid.mp3"
    for p in (track_a, track_b, track_c):
        p.write_text("x")

    features = {
        str(track_a): (85.0, 0.05),
        str(track_b): (115.0, 0.65),
        str(track_c): (145.0, 0.25),
    }

    out_dir = tmp_path / "Playlists"
    _ = bucket_by_tempo_energy(
        str(tmp_path),
        tempo_ranges=[(0, 100), (100, 140), (140, None)],
        energy_thresholds=[0.2, 0.6],
        output_dir=str(out_dir),
        feature_provider=lambda path: features[path],
    )

    tempo_low = out_dir / "tempo_0-100.m3u"
    tempo_mid = out_dir / "tempo_100-140.m3u"
    tempo_high = out_dir / "tempo_140+.m3u"
    assert tempo_low.exists()
    assert tempo_mid.exists()
    assert tempo_high.exists()

    assert os.path.relpath(track_a, out_dir) in tempo_low.read_text().splitlines()
    assert os.path.relpath(track_b, out_dir) in tempo_mid.read_text().splitlines()
    assert os.path.relpath(track_c, out_dir) in tempo_high.read_text().splitlines()

    energy_low = out_dir / "energy_0-0.2.m3u"
    energy_mid = out_dir / "energy_0.2-0.6.m3u"
    energy_high = out_dir / "energy_0.6+.m3u"
    assert energy_low.exists()
    assert energy_mid.exists()
    assert energy_high.exists()

    assert os.path.relpath(track_a, out_dir) in energy_low.read_text().splitlines()
    assert os.path.relpath(track_b, out_dir) in energy_high.read_text().splitlines()
    assert os.path.relpath(track_c, out_dir) in energy_mid.read_text().splitlines()
