import os
from typing import List

import requests
from utils.audio_metadata_reader import read_tags
from utils.path_helpers import ensure_long_path

from plugins.base import MetadataPlugin

API_KEY = os.getenv("LASTFM_API_KEY")

class LastfmPlugin(MetadataPlugin):
    def identify(self, file_path: str) -> dict:
        if not API_KEY:
            return {}
        tags = read_tags(ensure_long_path(file_path))
        artist = tags.get("artist")
        title = tags.get("title")
        if not artist or not title:
            return {}

        params = {
            "method": "track.getInfo",
            "api_key": API_KEY,
            "artist": artist,
            "track": title,
            "format": "json",
        }
        try:
            resp = requests.get(
                "http://ws.audioscrobbler.com/2.0/", params=params, timeout=5
            )
            data = resp.json().get("track", {})
        except Exception:
            return {}

        tags = data.get("toptags", {}).get("tag", [])
        genres: List[str] = [
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
