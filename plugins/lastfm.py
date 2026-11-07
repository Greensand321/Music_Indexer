import os
from typing import List

import requests
from utils.audio_metadata_reader import read_tags
from utils.path_helpers import ensure_long_path

from plugins.base import MetadataPlugin, SourceStatus

# Read at import for backward compatibility; ``_api_key()`` re-reads the
# environment at scan time so setting it later (or in tests) takes effect.
API_KEY = os.getenv("LASTFM_API_KEY")


def _api_key() -> str:
    return (os.getenv("LASTFM_API_KEY") or API_KEY or "").strip()


def _selected_service() -> str:
    try:
        from config import load_config

        return str(load_config().get("metadata_service") or "")
    except Exception:  # noqa: BLE001
        return ""


class LastfmPlugin(MetadataPlugin):
    name = "Last.fm"

    def preflight(self) -> SourceStatus:
        if _api_key():
            return SourceStatus(self.name, True, "key from LASTFM_API_KEY")
        problem = ""
        if _selected_service() == "Last.fm":
            # Settings has no Last.fm key field; say where the key goes.
            problem = (
                "Last.fm is selected in Settings, but no Last.fm API key is set "
                "(set the LASTFM_API_KEY environment variable)"
            )
        return SourceStatus(self.name, False, "", problem)

    def identify(self, file_path: str) -> dict:
        api_key = _api_key()
        if not api_key:
            return {}
        tags = read_tags(ensure_long_path(file_path))
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
        # Network errors propagate so the scan can report them.
        resp = requests.get(
            "http://ws.audioscrobbler.com/2.0/", params=params, timeout=5
        )
        data = resp.json().get("track", {}) or {}

        tags = (data.get("toptags", {}) or {}).get("tag", []) or []
        genres: List[str] = []
        for t in tags:
            try:
                count = int(t.get("count", 0))
            except (TypeError, ValueError):
                count = 0
            if t.get("name") and count > 10:
                genres.append(t.get("name", "").title())
        if genres:
            return {
                "artist": artist,
                "title": title,
                "genres": genres,
                "score": min(1.0, len(genres) / 10),
            }
        return {}
