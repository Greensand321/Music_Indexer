from plugins.base import MetadataPlugin

class DiscogsPlugin(MetadataPlugin):
    def identify(self, file_path: str) -> dict:
        return {}
