import metadata_service


def test_query_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(metadata_service, "query_acoustid", lambda k, f: calls.append("a"))
    monkeypatch.setattr(metadata_service, "query_lastfm", lambda k, f: calls.append("l"))
    monkeypatch.setattr(metadata_service, "query_spotify", lambda k, f: calls.append("s"))
    metadata_service.query_metadata("AcoustID", "key", "file")
    metadata_service.query_metadata("Last.fm", "key", "file")
    metadata_service.query_metadata("Spotify", "key", "file")
    assert calls == ["a", "l", "s"]
