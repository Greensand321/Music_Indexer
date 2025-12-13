import importlib
import os
import sys
import types


def _load_controller(monkeypatch):
    mutagen_stub = types.ModuleType("mutagen")
    mutagen_stub.File = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mutagen", mutagen_stub)

    if "controllers.genre_playlist_controller" in sys.modules:
        del sys.modules["controllers.genre_playlist_controller"]

    return importlib.import_module("controllers.genre_playlist_controller")


def test_group_tracks_by_genre_applies_mapping(monkeypatch, tmp_path):
    gpc = _load_controller(monkeypatch)
    track_a = tmp_path / "song_a.mp3"
    track_b = tmp_path / "song_b.mp3"
    track_a.write_text("a")
    track_b.write_text("b")

    genre_tags = {
        str(track_a): ["Alt;Rock"],
        str(track_b): [],
    }

    class DummyAudio:
        def __init__(self, path):
            tags = genre_tags.get(path, [])
            self.tags = {"genre": tags} if tags is not None else None

    monkeypatch.setattr(gpc, "MutagenFile", lambda path, easy=True: DummyAudio(path))

    grouped = gpc.group_tracks_by_genre(
        [str(track_a), str(track_b)],
        mapping={"Alt": "Alternative"},
        include_unknown=True,
    )

    assert grouped["Alternative"] == [str(track_a)]
    assert grouped["Rock"] == [str(track_a)]
    assert grouped["Unknown"] == [str(track_b)]


def test_write_genre_playlists_creates_files(monkeypatch, tmp_path):
    gpc = _load_controller(monkeypatch)
    track_a = tmp_path / "song_a.mp3"
    track_b = tmp_path / "song_b.mp3"
    track_a.write_text("a")
    track_b.write_text("b")

    grouped = {
        "Indie Rock": [str(track_a)],
        "Chill/Lo-fi": [str(track_b)],
    }

    playlists_dir = tmp_path / "Playlists"
    paths = gpc.write_genre_playlists(grouped, str(playlists_dir))

    indie = playlists_dir / "Indie Rock.m3u"
    chill = playlists_dir / "Chill_Lo-fi.m3u"

    assert indie.exists()
    assert chill.exists()
    assert paths["Indie Rock"] == str(indie)
    assert paths["Chill/Lo-fi"] == str(chill)

    with open(indie, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    with open(chill, "r", encoding="utf-8") as f:
        chill_lines = [line.strip() for line in f.readlines()]

    assert lines == [os.path.relpath(track_a, playlists_dir)]
    assert chill_lines == [os.path.relpath(track_b, playlists_dir)]
