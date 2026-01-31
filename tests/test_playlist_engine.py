import os

from playlist_engine import (
    categorize_tempo,
    categorize_energy,
    more_like_this,
    autodj_playlist,
    export_genre_playlists,
    sort_tracks_by_genre,
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


def test_sort_tracks_by_genre(tmp_path):
    rock = tmp_path / "rock" / "song1.mp3"
    jazz = tmp_path / "jazz" / "song2.flac"
    mixed = tmp_path / "var" / "song3.wav"
    for p in [rock, jazz, mixed]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("test")

    genre_map = {
        str(rock): ["Rock"],
        str(jazz): ["Jazz"],
        str(mixed): ["Jazz", "Rock"],
    }

    logs: list[str] = []
    progress: list[int] = []
    result = sort_tracks_by_genre(
        [str(rock), str(jazz), str(mixed)],
        str(tmp_path),
        log_callback=lambda m: logs.append(m),
        progress_callback=lambda c: progress.append(c),
        genre_reader=lambda p: genre_map[p],
    )

    playlists_dir = tmp_path / "Playlists" / "Genres"
    rock_playlist = (playlists_dir / "Rock.m3u").read_text().splitlines()
    jazz_playlist = (playlists_dir / "Jazz.m3u").read_text().splitlines()

    assert result["genres"]["Rock"]["count"] == 2
    assert result["genres"]["Jazz"]["count"] == 2
    rock_rel = os.path.relpath(rock, playlists_dir)
    jazz_rel = os.path.relpath(jazz, playlists_dir)
    mixed_rel = os.path.relpath(mixed, playlists_dir)

    assert rock_rel in rock_playlist
    assert mixed_rel in rock_playlist
    assert jazz_rel in jazz_playlist
    assert mixed_rel in jazz_playlist
    assert progress == [1, 2, 3]
    assert len(logs) == 5
    assert any("song1" in log for log in logs)


def test_manual_genre_export(tmp_path):
    rock = tmp_path / "rock" / "song1.mp3"
    jazz = tmp_path / "jazz" / "song2.flac"
    for p in [rock, jazz]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("test")

    genre_map = {str(rock): ["Rock"], str(jazz): ["Jazz"]}

    result = sort_tracks_by_genre(
        [str(rock), str(jazz)],
        str(tmp_path),
        log_callback=lambda _m: None,
        genre_reader=lambda p: genre_map[p],
        export=False,
    )

    playlists_dir = tmp_path / "Playlists" / "Genres"
    assert not playlists_dir.exists()
    assert "buckets" in result and "playlist_paths" in result

    logs: list[str] = []
    exported = export_genre_playlists(
        result["buckets"],
        str(tmp_path),
        selected_genres={"Jazz"},
        log_callback=lambda m: logs.append(m),
        planned_paths=result["playlist_paths"],
    )

    assert (playlists_dir / "Jazz.m3u").exists()
    assert not (playlists_dir / "Rock.m3u").exists()
    assert exported["genres"]["Jazz"]["exported"] is True
    assert exported["genres"]["Rock"]["exported"] is False
    assert any("Jazz.m3u" in log for log in logs)
