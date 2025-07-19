"""Dispatch metadata queries to supported services."""

from typing import Dict
from utils.path_helpers import ensure_long_path


def query_acoustid(api_key: str, audio_file: str) -> Dict:
    """Query AcoustID for ``audio_file`` using ``api_key``.

    Returns a metadata dict or raises on failure.
    """
    import acoustid
    import musicbrainzngs
    from itertools import islice

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
    from mutagen import File as MutagenFile

    audio = MutagenFile(ensure_long_path(audio_file), easy=True)
    artist = (audio.tags.get("artist") or [None])[0] if audio and audio.tags else None
    title = (audio.tags.get("title") or [None])[0] if audio and audio.tags else None
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
    """Placeholder for MusicBrainz metadata lookup using ``musicbrainzngs``."""
    return {}


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
