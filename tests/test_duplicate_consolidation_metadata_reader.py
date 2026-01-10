import utils.audio_metadata_reader as audio_metadata_reader

import duplicate_consolidation


class DummyTags(dict):
    def getall(self, key):
        return self.get(key, [])


class DummyInfo:
    bitrate = 320000
    sample_rate = 44100
    bits_per_sample = 16
    channels = 2


class DummyAudio:
    def __init__(self):
        self.tags = DummyTags(
            {
                "artist": ["Artist"],
                "title": ["Title"],
                "album": ["Album"],
                "tracknumber": ["4/10"],
            }
        )
        self.info = DummyInfo()


def test_read_tags_and_artwork_uses_preloaded_mutagen(monkeypatch) -> None:
    audio = DummyAudio()
    monkeypatch.setattr(duplicate_consolidation, "_read_audio_file", lambda _p: (audio, None))
    monkeypatch.setattr(
        audio_metadata_reader,
        "MutagenFile",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("MutagenFile should not be used")),
    )

    tags, _art, error, art_error, audio_props, trace = duplicate_consolidation._read_tags_and_artwork(
        "song.mp3", {}
    )

    assert error is None
    assert art_error == "deferred"
    assert tags["artist"] == "Artist"
    assert tags["album"] == "Album"
    assert tags["track"] == 4
    assert audio_props["bitrate"] == 320000
    assert audio_props["sample_rate"] == 44100
    assert audio_props["bit_depth"] == 16
    assert audio_props["channels"] == 2
    assert trace["reader_hint"] and "preloaded" in trace["reader_hint"]
