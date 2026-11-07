"""Metadata lookup services for AcoustID and MusicBrainz.

Backend only — deliberately free of any GUI toolkit import so this module stays
importable headlessly (see the "GUI <-> backend separation" rule in CLAUDE.md).
The legacy Tkinter configuration UI that used to live here now sits in
``metadata_config_frame.py`` at the repository root.
"""
import os
import shutil

import musicbrainzngs

from plugins.base import MetadataPlugin, SourceStatus, describe_lookup_error
from plugins.api_service import ApiService
from metadata_service import query_metadata, musicbrainz_useragent
from config import load_config, CONFIG_PATH
import tag_fixer


def acoustid_api_key() -> tuple[str, str]:
    """Return ``(key, where_it_came_from)`` for AcoustID lookups.

    Settings -> Metadata Services saves the AcoustID key as
    ``metadata_api_key``. Clearing that field saves ``""``, which used to beat
    the built-in default and make every lookup fail with "invalid API key";
    an empty or blank value now means "use the built-in key".
    """
    try:
        key = str(load_config().get("metadata_api_key") or "").strip()
    except Exception:  # noqa: BLE001
        key = ""
    if key:
        return key, "key from Settings"
    return tag_fixer.ACOUSTID_API_KEY, "built-in key"


class AcoustIDService(MetadataPlugin, ApiService):
    """Metadata lookup via the AcoustID web service."""

    name = "AcoustID"

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
        api_key, _source = acoustid_api_key()
        return query_metadata("AcoustID", api_key, fingerprint)

    def identify(self, file_path: str) -> dict:  # for MetadataPlugin
        # Errors propagate (as readable messages) so the scan can count and
        # report *why* files went unmatched instead of showing nothing.
        try:
            return self.query(file_path) or {}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(describe_lookup_error(exc)) from exc

    @staticmethod
    def fingerprinting_available() -> bool:
        """True if pyacoustid can fingerprint: Chromaprint library or fpcalc."""
        try:
            import acoustid
        except Exception:  # noqa: BLE001
            return False
        if getattr(acoustid, "have_audioread", False) and getattr(acoustid, "have_chromaprint", False):
            return True
        command = os.environ.get(
            getattr(acoustid, "FPCALC_ENVVAR", "FPCALC"),
            getattr(acoustid, "FPCALC_COMMAND", "fpcalc"),
        )
        return shutil.which(command) is not None

    def preflight(self) -> SourceStatus:
        _key, key_source = acoustid_api_key()
        if not self.fingerprinting_available():
            return SourceStatus(
                self.name, False, key_source,
                "fpcalc (Chromaprint) was not found, so tracks can't be identified by sound",
            )
        try:
            online = self.check_connection()
        except Exception:  # noqa: BLE001
            online = False
        if not online:
            return SourceStatus(
                self.name, False, key_source,
                "api.acoustid.org can't be reached (offline?) — AcoustID lookups were skipped",
            )
        return SourceStatus(self.name, True, key_source)

    @staticmethod
    def check_connection() -> bool:
        ok, _ = AcoustIDService().test_connection()
        return ok


class MusicBrainzService(ApiService):
    """MusicBrainz integration using ``musicbrainzngs``."""

    def __init__(self):
        ApiService.__init__(self, CONFIG_PATH)
        # Blank Settings fields fall back to defaults; MusicBrainz rejects an
        # empty app name or version.
        self.app, self.version, self.contact = musicbrainz_useragent()

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
