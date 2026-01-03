"""Dispatch metadata queries to supported services."""

from typing import Dict
from utils.audio_metadata_reader import read_tags
from utils.path_helpers import ensure_long_path


def query_acoustid(api_key: str, audio_file: str) -> Dict:
    """Query AcoustID for ``audio_file`` using ``api_key``.

    If ``audio_file`` is an empty string a lightweight request is issued to
    verify connectivity and API key validity. An empty dictionary is
    returned in this case.
    """
    import acoustid
    import musicbrainzngs
    from itertools import islice
    import requests

    if not audio_file:
        url = "https://api.acoustid.org/v2/user/status"
        resp = requests.get(url, params={"client": api_key}, timeout=5)
        resp.raise_for_status()
        return {}

    match_gen = acoustid.match(api_key, ensure_long_path(audio_file))
    peek = list(islice(match_gen, 5))
    if not peek:
        return {}
    best_score, best_rid, best_title, best_artist = peek[0]

    album = None
    genres = []
    if best_rid:
        try:
            rec = musicbrainzngs.get_recording_by_id(
                best_rid, includes=["releases", "tags"]
            )["recording"]
            rels = rec.get("releases", [])
            if rels:
                album = rels[0].get("title")
            mb_tags = rec.get("tag-list", [])
            genres = [t["name"] for t in mb_tags if "name" in t]
        except Exception:
            pass

    return {
        "artist": best_artist,
        "title": best_title,
        "album": album,
        "genres": genres,
        "score": best_score,
    }


def query_lastfm(api_key: str, audio_file: str) -> Dict:
    """Query Last.fm for ``audio_file`` using ``api_key``."""
    import requests
    tags = read_tags(ensure_long_path(audio_file))
    artist = tags.get("artist")
    title = tags.get("title")
    if not artist or not title:
        return {}

    params = {
        "method": "track.getInfo",
        "api_key": api_key,
        "artist": artist,
        "track": title,
        "format": "json",
    }
    resp = requests.get("http://ws.audioscrobbler.com/2.0/", params=params, timeout=5)
    data = resp.json().get("track", {})
    tags = data.get("toptags", {}).get("tag", [])
    genres = [
        t.get("name", "").title()
        for t in tags
        if t.get("name") and int(t.get("count", 0)) > 10
    ]
    if genres:
        return {
            "artist": artist,
            "title": title,
            "genres": genres,
            "score": min(1.0, len(genres) / 10),
        }
    return {}


def query_spotify(api_key: str, audio_file: str) -> Dict:
    """Placeholder for Spotify metadata lookup."""
    return {}


def query_musicbrainz(api_key: str, audio_file: str) -> Dict:
    """Query MusicBrainz for ``audio_file`` using ``musicbrainzngs``.

    The ``api_key`` argument is treated as the contact string passed to
    ``musicbrainzngs.set_useragent``.  If ``audio_file`` is an empty string the
    function simply performs a trivial request to verify connectivity.
    """
    import musicbrainzngs
    contact = api_key or ""
    musicbrainzngs.set_useragent(
        "SoundVaultTagFixer",
        "1.0",
        contact,
    )

    # When called with no file we just issue a simple search request to check
    # that the API is reachable.
    if not audio_file:
        musicbrainzngs.search_recordings(recording="test", limit=1)
        return {}

    tags = read_tags(ensure_long_path(audio_file))
    artist = tags.get("artist")
    title = tags.get("title")
    if not artist or not title:
        return {}

    result = musicbrainzngs.search_recordings(
        recording=title,
        artist=artist,
        includes=["releases", "tags"],
        limit=1,
        strict=True,
    )
    recs = result.get("recording-list", [])
    if not recs:
        return {}

    rec = recs[0]
    album = None
    rels = rec.get("releases", [])
    if rels:
        album = rels[0].get("title")
    mb_tags = rec.get("tag-list", [])
    genres = [t["name"] for t in mb_tags if "name" in t]
    score = float(rec.get("ext:score", 100)) / 100.0

    return {
        "artist": artist,
        "title": title,
        "album": album,
        "genres": genres,
        "score": score,
    }


def query_gracenote(api_key: str, audio_file: str) -> Dict:
    """Placeholder for Gracenote metadata lookup."""
    return {}


def query_metadata(service: str, api_key: str, audio_file: str) -> Dict:
    """Dispatch to the appropriate metadata query function."""
    if service == "AcoustID":
        return query_acoustid(api_key, audio_file)
    if service == "Last.fm":
        return query_lastfm(api_key, audio_file)
    if service == "Spotify":
        return query_spotify(api_key, audio_file)
    if service == "MusicBrainz":
        return query_musicbrainz(api_key, audio_file)
    if service == "Gracenote":
        return query_gracenote(api_key, audio_file)
    raise ValueError(f"Unsupported service: {service}")
