from . import id3


class DummyAudio:
    def __init__(self):
        self.tags = None
        self.pictures = []


File = lambda *a, **k: DummyAudio()

