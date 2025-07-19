import os
import types
import sys
import importlib

from plugins.base import MetadataPlugin


class DummyPlugin(MetadataPlugin):
    def identify(self, file_path: str) -> dict:
        return {"artist": "NewA", "title": "NewT", "score": 1.0}


def test_fix_tags_long_paths(tmp_path, monkeypatch):
    # Stub dependencies
    mutagen_stub = types.ModuleType('mutagen')

    class DummyAudio:
        def __init__(self):
            self.tags = {}
            self.saved = False

        def __setitem__(self, key, value):
            self.tags[key] = value

        def save(self):
            self.saved = True

    def File(*_a, **_k):
        return DummyAudio()

    mutagen_stub.File = File
    id3_stub = types.ModuleType('id3')
    id3_stub.ID3NoHeaderError = Exception
    mutagen_stub.id3 = id3_stub
    monkeypatch.setitem(sys.modules, 'mutagen', mutagen_stub)
    monkeypatch.setitem(sys.modules, 'mutagen.id3', id3_stub)

    acoustid_stub = types.ModuleType('acoustid')

    def match(_key, _path):
        yield (1.0, None, 'Song', 'Artist')

    acoustid_stub.match = match
    acoustid_stub.NoBackendError = Exception
    acoustid_stub.FingerprintGenerationError = Exception
    acoustid_stub.WebServiceError = Exception
    monkeypatch.setitem(sys.modules, 'acoustid', acoustid_stub)

    musicbrainzngs_stub = types.ModuleType('musicbrainzngs')
    musicbrainzngs_stub.set_useragent = lambda *a, **k: None
    musicbrainzngs_stub.get_recording_by_id = lambda *a, **k: {"recording": {}}
    monkeypatch.setitem(sys.modules, 'musicbrainzngs', musicbrainzngs_stub)

    requests_stub = types.ModuleType('requests')
    requests_stub.get = lambda *a, **k: types.SimpleNamespace(json=lambda: {})
    monkeypatch.setitem(sys.modules, 'requests', requests_stub)

    import tag_fixer

    importlib.reload(tag_fixer)
    tag_fixer.PLUGINS = [DummyPlugin()]

    base = tmp_path
    for i in range(6):
        base = base / ("d" * 50 + str(i))
    base.mkdir(parents=True)
    path = base / "dummy_song.mp3"
    path.write_text("x")

    monkeypatch.setattr(tag_fixer, "MutagenFile", File)

    summary = tag_fixer.fix_tags(str(base), log_callback=lambda m: None)
    assert summary["processed"] == 1
    assert summary["updated"] == 1


def test_corrupted_files_are_skipped(tmp_path, monkeypatch):
    mutagen_stub = types.ModuleType('mutagen')

    def File(*_a, **_k):
        raise Exception('bad file')

    mutagen_stub.File = File
    id3_stub = types.ModuleType('id3')
    id3_stub.ID3NoHeaderError = Exception
    mutagen_stub.id3 = id3_stub
    monkeypatch.setitem(sys.modules, 'mutagen', mutagen_stub)
    monkeypatch.setitem(sys.modules, 'mutagen.id3', id3_stub)

    import tag_fixer
    import importlib
    importlib.reload(tag_fixer)
    tag_fixer.PLUGINS = []

    path = tmp_path / 'bad.mp3'
    path.write_text('x')

    monkeypatch.setattr(tag_fixer, 'MutagenFile', File)

    # is_remix should not raise and should return False
    assert tag_fixer.is_remix(str(path)) is False

    import sqlite3
    db_path = tmp_path / 'db.sqlite'
    tag_fixer.init_db(str(db_path))
    db = sqlite3.connect(str(db_path))
    records = tag_fixer.build_file_records(
        str(tmp_path), db_conn=db, log_callback=lambda m: None
    )
    assert records == []

    summary = tag_fixer.fix_tags(str(tmp_path), log_callback=lambda m: None)
    assert summary['processed'] == 1
    assert summary['updated'] == 0

