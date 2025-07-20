import sys
import types
import importlib


def test_musicbrainz_service_success(monkeypatch):
    mb = types.ModuleType('musicbrainzngs')
    mb.set_useragent = lambda *a, **k: None
    mb.search_artists = lambda **kw: {'artist-list': [1]}
    mb.MusicBrainzError = Exception
    monkeypatch.setitem(sys.modules, 'musicbrainzngs', mb)
    import plugins.acoustid_plugin as ap
    importlib.reload(ap)
    svc = ap.MusicBrainzService()
    ok, msg = svc.test_connection()
    assert ok is True
    assert 'artist' in msg


def test_musicbrainz_service_error(monkeypatch):
    class Err(Exception):
        pass
    mb = types.ModuleType('musicbrainzngs')
    mb.set_useragent = lambda *a, **k: None
    def fail(**kw):
        raise Err('fail')
    mb.search_artists = fail
    mb.MusicBrainzError = Err
    monkeypatch.setitem(sys.modules, 'musicbrainzngs', mb)
    import plugins.acoustid_plugin as ap
    importlib.reload(ap)
    svc = ap.MusicBrainzService()
    ok, msg = svc.test_connection()
    assert ok is False
    assert 'fail' in msg
