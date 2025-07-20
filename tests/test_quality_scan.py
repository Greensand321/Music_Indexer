import sys
import types
import importlib

class DummyFrame:
    def __init__(self, text, broken=False):
        self._text = text
        self.broken = broken
    @property
    def text(self):
        if self.broken:
            raise UnicodeDecodeError("id3", b"x", 0, 1, "bad")
        return [self._text]

def prepare_modules(monkeypatch, id3_data=None, mp3_tags=None, raise_error=False):
    mutagen_stub = types.ModuleType('mutagen')
    mp3_stub = types.ModuleType('mutagen.mp3')
    id3_stub = types.ModuleType('mutagen.id3')
    mutagen_stub.mp3 = mp3_stub
    mutagen_stub.id3 = id3_stub
    mutagen_stub.File = lambda *a, **k: types.SimpleNamespace(tags=None)

    mp3_stub.MP3 = lambda p: types.SimpleNamespace(tags=mp3_tags or {})

    class ID3Error(Exception):
        pass

    id3_stub.ID3NoHeaderError = ID3Error

    class DummyID3:
        def __init__(self, path):
            if raise_error:
                raise ID3Error()
            self._data = id3_data or {}
        def get(self, key):
            return self._data.get(key)
        def __getitem__(self, key):
            return self._data[key]
        def keys(self):
            return self._data.keys()
    id3_stub.ID3 = DummyID3

    monkeypatch.setitem(sys.modules, 'mutagen', mutagen_stub)
    monkeypatch.setitem(sys.modules, 'mutagen.mp3', mp3_stub)
    monkeypatch.setitem(sys.modules, 'mutagen.id3', id3_stub)

    import tidal_sync
    importlib.reload(tidal_sync)
    return tidal_sync


def test_fallback_to_tpe2(monkeypatch):
    ts = prepare_modules(monkeypatch, id3_data={
        'TPE2': DummyFrame('B Artist'),
        'TIT2': DummyFrame('Song'),
    })
    artist, title = ts._read_artist_title('x.mp3')
    assert artist == 'B Artist'
    assert title == 'Song'


def test_unicode_error_fallback(monkeypatch):
    ts = prepare_modules(monkeypatch, id3_data={
        'TPE1': DummyFrame('bad', True),
        'TIT2': DummyFrame('bad', True),
    }, mp3_tags={'artist': ['v1'], 'title': ['t1']})
    artist, title = ts._read_artist_title('x.mp3')
    assert artist == 'v1'
    assert title == 't1'


def test_id3v1_only(monkeypatch):
    ts = prepare_modules(monkeypatch, id3_data={}, mp3_tags={'artist': ['A'], 'title': ['T']}, raise_error=True)
    artist, title = ts._read_artist_title('x.mp3')
    assert artist == 'A'
    assert title == 'T'
