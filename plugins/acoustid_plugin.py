import acoustid
import musicbrainzngs
from itertools import islice

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
    def identify(self, file_path: str) -> dict:
        try:
            match_gen = acoustid.match(ACOUSTID_API_KEY, ensure_long_path(file_path))
            peek = list(islice(match_gen, 5))
            if not peek:
                return {}

            best_score, best_rid, best_title, best_artist = peek[0]
        except acoustid.NoBackendError:
            return {}
        except acoustid.FingerprintGenerationError:
            return {}
        except acoustid.WebServiceError:
            return {}

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
            return False
        except Exception:
            return False
        return True
