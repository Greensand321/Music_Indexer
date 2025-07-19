import acoustid
import musicbrainzngs
from itertools import islice
import tkinter as tk
from tkinter import messagebox

from plugins.base import MetadataPlugin
from utils.path_helpers import ensure_long_path
from tag_fixer import (
    ACOUSTID_API_KEY,
    ACOUSTID_APP_NAME,
    ACOUSTID_APP_VERSION,
)

musicbrainzngs.set_useragent(
    ACOUSTID_APP_NAME,
    ACOUSTID_APP_VERSION,
    "youremail@example.com",
)

class AcoustIDPlugin(MetadataPlugin):
    @staticmethod
    def _prompt_reconnect() -> bool:
        """Show a retry dialog when the AcoustID service cannot be reached."""
        root = tk.Tk()
        root.withdraw()
        try:
            return messagebox.askretrycancel(
                "AcoustID Connection Failed",
                (
                    "Unable to reach the AcoustID service.\n"
                    "Check your network connection or API key, then click Retry."
                ),
            )
        finally:
            root.destroy()

    def identify(self, file_path: str) -> dict:
        while True:
            try:
                match_gen = acoustid.match(ACOUSTID_API_KEY, ensure_long_path(file_path))
                peek = list(islice(match_gen, 5))
                if not peek:
                    return {}

                best_score, best_rid, best_title, best_artist = peek[0]
                break
            except acoustid.NoBackendError:
                return {}
            except acoustid.FingerprintGenerationError:
                return {}
            except acoustid.WebServiceError:
                if not self._prompt_reconnect():
                    return {}
                continue

        album = None
        genres = []
        if best_rid:
            try:
                rec = musicbrainzngs.get_recording_by_id(
                    best_rid,
                    includes=["releases", "tags"],
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

    @staticmethod
    def check_connection() -> bool:
        """Return True if AcoustID web service is reachable."""
        try:
            acoustid.match(ACOUSTID_API_KEY, b"")
        except acoustid.WebServiceError:
            return AcoustIDPlugin._prompt_reconnect()
        except Exception:
            return AcoustIDPlugin._prompt_reconnect()
        return True
