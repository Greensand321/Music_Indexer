from plugins.base import MetadataPlugin

class SpotifyPlugin(MetadataPlugin):
    def identify(self, file_path: str) -> dict:
        return {}
