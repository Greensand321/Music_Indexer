import io
import sys
import types
import importlib

from tests.test_fingerprint_norm import DummyLogger


def load_ts(monkeypatch):
    mutagen_stub = types.ModuleType('mutagen')
    mutagen_stub.File = lambda *a, **k: None
    id3_stub = types.ModuleType('id3')
    class DummyID3:
        pass
    id3_stub.ID3 = DummyID3
    id3_stub.ID3NoHeaderError = Exception
    mutagen_stub.id3 = id3_stub
    mp3_stub = types.ModuleType('mp3')
    class MP3:
        def __init__(self, *a, **k):
            self.tags = None
    mp3_stub.MP3 = MP3
    mutagen_stub.mp3 = mp3_stub
    monkeypatch.setitem(sys.modules, 'mutagen', mutagen_stub)
    monkeypatch.setitem(sys.modules, 'mutagen.id3', id3_stub)
    monkeypatch.setitem(sys.modules, 'mutagen.mp3', mp3_stub)

    acoustid_stub = types.ModuleType('acoustid')
    monkeypatch.setitem(sys.modules, 'acoustid', acoustid_stub)

    import tidal_sync
    importlib.reload(tidal_sync)
    return tidal_sync


def test_fingerprint_logs_exception(tmp_path, monkeypatch):
    ts = load_ts(monkeypatch)

    import config
    monkeypatch.setattr(config, 'load_config', lambda: {})

    def fake_norm(path, *a, **k):
        return io.BytesIO(b'data')
    import audio_norm
    monkeypatch.setattr(audio_norm, 'normalize_for_fp', fake_norm)

    def bad_fp(*a, **k):
        raise RuntimeError('boom')
    monkeypatch.setattr(sys.modules['acoustid'], 'fingerprint_file', bad_fp, raising=False)

    p = tmp_path / 'a.mp3'
    p.write_text('x')

    log = DummyLogger()
    res = ts._fingerprint(str(p), log_callback=log)
    assert res is None
    assert any('Fingerprint failed' in m for m in log.msgs)
