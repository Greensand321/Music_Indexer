import os
from plugins.base import MetadataPlugin

class TestPlugin(MetadataPlugin):
    def identify(self, file_path: str) -> dict:
        if os.path.basename(file_path).lower().startswith("dummy_"):
            return {
                "genres": ["TestGenre"],
                "score": 1.0,
            }
        return {}
