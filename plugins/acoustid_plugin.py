import acoustid
import musicbrainzngs
from itertools import islice
import tkinter as tk
from tkinter import messagebox, simpledialog

from plugins.base import MetadataPlugin
from utils.path_helpers import ensure_long_path
import tag_fixer
from tag_fixer import (
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
        """Prompt for API key update and test the AcoustID connection."""
        root = tk.Tk()
        root.withdraw()
        try:
            api_key = simpledialog.askstring(
                "AcoustID Connection Failed",
                (
                    "Unable to reach the AcoustID service.\n"
                    "Update the API key and press OK to retry."
                ),
                initialvalue=tag_fixer.ACOUSTID_API_KEY or "",
                parent=root,
            )
            if api_key is None:
                return False
            tag_fixer.ACOUSTID_API_KEY = api_key
            try:
                acoustid.match(tag_fixer.ACOUSTID_API_KEY, b"")
            except Exception:
                messagebox.showerror(
                    "AcoustID Connection",
                    "Connection failed",
                    parent=root,
                )
                return False
            messagebox.showinfo(
                "AcoustID Connection",
                "Connection success",
                parent=root,
            )
            return True
        finally:
            root.destroy()

    def identify(self, file_path: str) -> dict:
        while True:
            try:
                match_gen = acoustid.match(tag_fixer.ACOUSTID_API_KEY, ensure_long_path(file_path))
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
            acoustid.match(tag_fixer.ACOUSTID_API_KEY, b"")
        except acoustid.WebServiceError:
            return AcoustIDPlugin._prompt_reconnect()
        except Exception:
            return AcoustIDPlugin._prompt_reconnect()
        return True
