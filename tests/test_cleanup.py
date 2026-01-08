import os
import types
import sys

# Provide simple stubs for mutagen so the module imports
mutagen_stub = types.ModuleType('mutagen')
class DummyAudio:
    def __init__(self):
        self.tags = None
        self.pictures = []
def File(*a, **k):
    return DummyAudio()
mutagen_stub.File = File
id3_stub = types.ModuleType('id3')
id3_stub.ID3NoHeaderError = Exception
mutagen_stub.id3 = id3_stub
sys.modules['mutagen'] = mutagen_stub
sys.modules['mutagen.id3'] = id3_stub

from music_indexer_api import apply_indexer_moves

def test_cleanup_removes_empty_dirs(tmp_path, monkeypatch):
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


def test_cleanup_removes_nested_empty_dirs(tmp_path, monkeypatch):
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
