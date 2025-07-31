from playlist_engine import categorize_tempo, categorize_energy, more_like_this, autodj_playlist


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
