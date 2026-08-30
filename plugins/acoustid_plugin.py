"""Metadata lookup services for AcoustID and MusicBrainz.

Backend only — deliberately free of any GUI toolkit import so this module stays
importable headlessly (see the "GUI <-> backend separation" rule in CLAUDE.md).
The legacy Tkinter configuration UI that used to live here now sits in
``metadata_config_frame.py`` at the repository root.
"""
import musicbrainzngs

from plugins.base import MetadataPlugin
from plugins.api_service import ApiService
from metadata_service import query_metadata
from config import load_config, CONFIG_PATH
import tag_fixer


class AcoustIDService(MetadataPlugin, ApiService):
    """Metadata lookup via the AcoustID web service."""

    def __init__(self):
        ApiService.__init__(self, CONFIG_PATH)

    def test_connection(self):
        import requests
        try:
            requests.get("https://api.acoustid.org/v2/", timeout=5)
        except Exception as e:
            return False, str(e)
        return True, "OK"

    def query(self, fingerprint: str):
        cfg = load_config()
        api_key = cfg.get("metadata_api_key", tag_fixer.ACOUSTID_API_KEY)
        return query_metadata("AcoustID", api_key, fingerprint)

    def identify(self, file_path: str) -> dict:  # for MetadataPlugin
        try:
            return self.query(file_path)
        except Exception:
            return {}

    @staticmethod
    def check_connection() -> bool:
        ok, _ = AcoustIDService().test_connection()
        return ok


class MusicBrainzService(ApiService):
    """MusicBrainz integration using ``musicbrainzngs``."""

    def __init__(self):
        ApiService.__init__(self, CONFIG_PATH)
        cfg = load_config()
        ua = cfg.get("musicbrainz_useragent", {})
        self.app = ua.get("app", "SoundVault")
        self.version = ua.get("version", "1.0")
        self.contact = ua.get("contact", "")

    def test_connection(self):
        from musicbrainzngs import MusicBrainzError

        try:
            musicbrainzngs.set_useragent(self.app, self.version, self.contact)
            res = musicbrainzngs.search_artists(query="Beatles", limit=1)
            n = len(res.get("artist-list", []))
            return True, f"OK – found {n} artist(s)"
        except MusicBrainzError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def query(self, fingerprint: str):
        musicbrainzngs.set_useragent(self.app, self.version, self.contact)
        return query_metadata("MusicBrainz", self.contact, fingerprint)
