from utils.audio_metadata_reader import read_metadata_from_mutagen


class DummyApic:
    def __init__(self, payload: bytes) -> None:
        self.data = payload


class DummyTags(dict):
    def getall(self, key):
        return self.get(key, [])


class DummyAudio:
    def __init__(self, tags):
        self.tags = tags


def test_read_metadata_from_mutagen_mp4_atoms() -> None:
    tags = DummyTags(
        {
            "\xa9ART": ["Artist"],
            "\xa9nam": ["Title"],
            "\xa9alb": ["Album"],
            "aART": ["Album Artist"],
            "\xa9day": ["2020"],
            "\xa9gen": ["Rock"],
            "cpil": [1],
            "trkn": [(1, 10)],
            "disk": [(2, 2)],
            "covr": [b"cover-bytes"],
        }
    )
    audio = DummyAudio(tags)

    metadata, covers, error, reader_hint = read_metadata_from_mutagen(
        audio, "song.m4a", include_cover=True, reader_hint_suffix="test"
    )

    assert error is None
    assert metadata["artist"] == "Artist"
    assert metadata["albumartist"] == "Album Artist"
    assert metadata["title"] == "Title"
    assert metadata["album"] == "Album"
    assert metadata["genre"] == "Rock"
    assert metadata["tracknumber"] == 1
    assert metadata["track"] == 1
    assert metadata["discnumber"] == 2
    assert metadata["disc"] == 2
    assert metadata["year"] == "2020"
    assert covers == [b"cover-bytes"]
    assert reader_hint == "mp4 atoms (test)"


def test_read_metadata_from_mutagen_tags_and_cover() -> None:
    tags = DummyTags(
        {
            "artist": ["Artist"],
            "albumartist": ["Album Artist"],
            "title": ["Title"],
            "album": ["Album"],
            "date": ["1999"],
            "genre": ["Pop"],
            "tracknumber": ["3/12"],
            "discnumber": ["1/2"],
            "APIC": [DummyApic(b"apic-bytes")],
        }
    )
    audio = DummyAudio(tags)

    metadata, covers, error, reader_hint = read_metadata_from_mutagen(
        audio, "song.mp3", include_cover=True, reader_hint_suffix="test"
    )

    assert error is None
    assert metadata["artist"] == "Artist"
    assert metadata["albumartist"] == "Album Artist"
    assert metadata["title"] == "Title"
    assert metadata["album"] == "Album"
    assert metadata["genre"] == "Pop"
    assert metadata["tracknumber"] == "3/12"
    assert metadata["track"] == 3
    assert metadata["discnumber"] == "1/2"
    assert metadata["disc"] == 1
    assert metadata["year"] == "1999"
    assert covers == [b"apic-bytes"]
    assert reader_hint == "mutagen tags (test)"
