import types

from _helpers.module_isolation import isolated_modules


def test_musicbrainz_service_success(monkeypatch):
    mb = types.ModuleType('musicbrainzngs')
    mb.set_useragent = lambda *a, **k: None
    mb.search_artists = lambda **kw: {'artist-list': [1]}
    mb.MusicBrainzError = Exception
    # Fresh plugin module bound to the stub; both are removed afterwards.
    with isolated_modules({'musicbrainzngs': mb}, fresh=['plugins.acoustid_plugin']) as mods:
        svc = mods['plugins.acoustid_plugin'].MusicBrainzService()
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
    # Fresh plugin module bound to the stub; both are removed afterwards.
    with isolated_modules({'musicbrainzngs': mb}, fresh=['plugins.acoustid_plugin']) as mods:
        svc = mods['plugins.acoustid_plugin'].MusicBrainzService()
        ok, msg = svc.test_connection()
    assert ok is False
    assert 'fail' in msg
