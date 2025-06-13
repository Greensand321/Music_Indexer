class MetadataPlugin:
    def identify(self, file_path: str) -> dict:
        """Given a filepath, return a dict with any of:
          - 'artist': str
          - 'title': str
          - 'album': str
          - 'genres': list[str]
          - 'score': float
        or return {} or None if no data."""
        raise NotImplementedError
