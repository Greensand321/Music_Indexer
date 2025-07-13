class EasyID3(dict):
    """Fallback EasyID3 implementation used when the real mutagen package is not installed."""
    def __init__(self, *a, **k):
        super().__init__()



