class FLAC:
    """Fallback FLAC implementation used when the real mutagen package is not installed."""
    def __init__(self, *a, **k):
        self.tags = {}
        self.pictures = []

    def get(self, key, default=None):
        return self.tags.get(key, default)

