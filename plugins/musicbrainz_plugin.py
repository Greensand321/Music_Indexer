"""MusicBrainz text search as a Tag Fixer lookup source.

Settings -> Metadata Services lets the user pick "MusicBrainz", and saves that
choice as ``metadata_service``. Nothing read it, so the choice did nothing.
When it is selected, the Tag Fixer now also looks each track up on
MusicBrainz by its existing artist and title (``metadata_service.
query_musicbrainz``) — useful when fingerprinting is unavailable, and the
source of album and genre proposals. AcoustID still runs when it can.
"""
from __future__ import annotations

from plugins.base import MetadataPlugin, SourceStatus, describe_lookup_error


def _selected_service() -> str:
    try:
        from config import load_config

        return str(load_config().get("metadata_service") or "")
    except Exception:  # noqa: BLE001
        return ""


class MusicBrainzSearchPlugin(MetadataPlugin):
    name = "MusicBrainz search"

    def preflight(self) -> SourceStatus:
        if _selected_service() != "MusicBrainz":
            return SourceStatus(self.name, False)
        from metadata_service import musicbrainz_useragent

        _app, _version, contact = musicbrainz_useragent()
        detail = "selected in Settings" + ("" if contact else " (no contact set)")
        return SourceStatus(self.name, True, detail)

    def identify(self, file_path: str) -> dict:
        from metadata_service import query_musicbrainz

        try:
            return query_musicbrainz("", file_path) or {}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(describe_lookup_error(exc)) from None
