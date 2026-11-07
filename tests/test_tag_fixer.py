import contextlib
import sqlite3
import types

import pytest

from _helpers.module_isolation import isolated_modules, make_mutagen_stub
from plugins.base import MetadataPlugin


class DummyPlugin(MetadataPlugin):
    def identify(self, file_path: str) -> dict:
        return {"artist": "NewA", "title": "NewT", "score": 1.0}


def _service_stubs():
    acoustid_stub = types.ModuleType("acoustid")

    def match(_key, _path):
        yield (1.0, None, "Song", "Artist")

    acoustid_stub.match = match
    acoustid_stub.NoBackendError = Exception
    acoustid_stub.FingerprintGenerationError = Exception
    acoustid_stub.WebServiceError = Exception

    musicbrainzngs_stub = types.ModuleType("musicbrainzngs")
    musicbrainzngs_stub.set_useragent = lambda *a, **k: None
    musicbrainzngs_stub.get_recording_by_id = lambda *a, **k: {"recording": {}}

    requests_stub = types.ModuleType("requests")
    requests_stub.get = lambda *a, **k: types.SimpleNamespace(json=lambda: {})

    return {
        "acoustid": acoustid_stub,
        "musicbrainzngs": musicbrainzngs_stub,
        "requests": requests_stub,
    }


@pytest.fixture
def load_tag_fixer():
    """Import a fresh ``tag_fixer`` bound to stubbed mutagen/service modules.

    The stubs and the stub-bound ``tag_fixer`` are removed again afterwards,
    so later tests importing ``tag_fixer`` get the real module.
    """
    with contextlib.ExitStack() as stack:

        def _load(file_func):
            mods = stack.enter_context(
                isolated_modules(
                    {**make_mutagen_stub(file_func), **_service_stubs()},
                    fresh=["tag_fixer"],
                )
            )
            return mods["tag_fixer"]

        yield _load


def test_fix_tags_long_paths(tmp_path, monkeypatch, load_tag_fixer):
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

    tag_fixer = load_tag_fixer(File)
    monkeypatch.setattr(tag_fixer, "PLUGINS", [DummyPlugin()])

    base = tmp_path
    for i in range(6):
        base = base / ("d" * 50 + str(i))
    base.mkdir(parents=True)
    path = base / "dummy_song.mp3"
    path.write_text("x")

    # Writes use tag_fixer.MutagenFile; reads use the shared reader.
    import utils.audio_metadata_reader as reader

    monkeypatch.setattr(tag_fixer, "MutagenFile", File)
    monkeypatch.setattr(reader, "MutagenFile", File)

    summary = tag_fixer.fix_tags(str(base), log_callback=lambda m: None)
    assert summary["processed"] == 1
    assert summary["updated"] == 1


def test_corrupted_files_are_skipped(tmp_path, monkeypatch, load_tag_fixer):
    def File(*_a, **_k):
        raise Exception("bad file")

    tag_fixer = load_tag_fixer(File)
    monkeypatch.setattr(tag_fixer, "PLUGINS", [])

    path = tmp_path / "bad.mp3"
    path.write_text("x")

    # Tag writes go through tag_fixer's own MutagenFile; tag *reads* go
    # through the shared reader in utils.audio_metadata_reader. Make both see
    # the corrupt file.
    import utils.audio_metadata_reader as reader

    monkeypatch.setattr(tag_fixer, "MutagenFile", File)
    monkeypatch.setattr(reader, "MutagenFile", File)

    # is_remix should not raise and should return False
    assert tag_fixer.is_remix(str(path)) is False

    db_path = tmp_path / "db.sqlite"
    tag_fixer.init_db(str(db_path))
    logs = []
    db = sqlite3.connect(str(db_path))
    try:
        records = tag_fixer.build_file_records(
            str(tmp_path), db_conn=db, log_callback=logs.append
        )
    finally:
        db.close()
    assert records == []
    assert any("Failed to read" in line and "bad.mp3" in line for line in logs)

    summary = tag_fixer.fix_tags(str(tmp_path), log_callback=lambda m: None)
    assert summary["processed"] == 1
    assert summary["updated"] == 0


def test_untagged_but_readable_files_are_still_listed(tmp_path, monkeypatch, load_tag_fixer):
    # The corrupt-file skip must not swallow files that are merely untagged.
    tag_fixer = load_tag_fixer(None)
    monkeypatch.setattr(tag_fixer, "PLUGINS", [])
    import utils.audio_metadata_reader as reader

    monkeypatch.setattr(reader, "MutagenFile", lambda *_a, **_k: None)
    (tmp_path / "untagged.mp3").write_text("x")

    db_path = tmp_path / "db.sqlite"
    tag_fixer.init_db(str(db_path))
    db = sqlite3.connect(str(db_path))
    try:
        records = tag_fixer.build_file_records(
            str(tmp_path), db_conn=db, log_callback=lambda m: None
        )
    finally:
        db.close()
    assert [r.path.name for r in records] == ["untagged.mp3"]
    assert records[0].status == "unmatched"
