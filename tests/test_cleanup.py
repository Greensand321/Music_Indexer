import types
import sys

import pytest

from _helpers.module_isolation import isolated_modules, make_mutagen_stub


@pytest.fixture
def apply_indexer_moves():
    # Stub mutagen so the indexer imports without the real library; the stub
    # and the stub-bound music_indexer_api are removed again after the test.
    with isolated_modules(make_mutagen_stub(), fresh=["music_indexer_api"]):
        import music_indexer_api

        yield music_indexer_api.apply_indexer_moves


def test_cleanup_removes_empty_dirs(tmp_path, monkeypatch, apply_indexer_moves):
    src = tmp_path / "Old"
    src.mkdir(parents=True)
    (src / "song.mp3").write_text("x")
    dst = tmp_path / "Music" / "Artist" / "Album" / "song.mp3"

    moves = {str(src / "song.mp3"): str(dst)}

    fake_fg = types.ModuleType('fingerprint_generator')
    fake_fg.compute_fingerprints_parallel = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'fingerprint_generator', fake_fg)
    def fake_compute(*a, **k):
        return moves, {}, []
    monkeypatch.setattr('music_indexer_api.compute_moves_and_tag_index', fake_compute)

    summary = apply_indexer_moves(str(tmp_path), log_callback=lambda m: None, create_playlists=False)

    assert summary["moved"] == 1
    assert not src.exists()


def test_cleanup_removes_nested_empty_dirs(tmp_path, monkeypatch, apply_indexer_moves):
    src = tmp_path / "Old" / "A" / "B"
    src.mkdir(parents=True)
    (src / "song.mp3").write_text("x")
    dst = tmp_path / "Music" / "Artist" / "Album" / "song.mp3"

    moves = {str(src / "song.mp3"): str(dst)}

    fake_fg = types.ModuleType('fingerprint_generator')
    fake_fg.compute_fingerprints_parallel = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'fingerprint_generator', fake_fg)
    def fake_compute(*a, **k):
        return moves, {}, []
    monkeypatch.setattr('music_indexer_api.compute_moves_and_tag_index', fake_compute)

    summary = apply_indexer_moves(str(tmp_path), log_callback=lambda m: None, create_playlists=False)

    assert summary["moved"] == 1
    assert not (tmp_path / "Old").exists()
