import os
import types

import pytest

from _helpers.module_isolation import make_acoustid_stub, stub_if_unavailable

# library_sync's vendored engine imports ``acoustid`` at module level; stub it
# only if it isn't installed.
stub_if_unavailable({"acoustid": make_acoustid_stub()})

import library_sync
from library_sync import compare_libraries, compute_quality_score
from library_sync_types import MatchStatus
from fingerprint_cache import flush_cache

fp_map = {
    'a.flac': '1 2',
    'b.mp3': '2 3',
    'b.flac': '2 3',
    'new.mp3': '9 9',
}


class DummyAudio:
    def __init__(self, bitrate=128000):
        self.tags = {'artist': ['A'], 'title': ['T'], 'album': ['AL']}
        self.info = types.SimpleNamespace(bitrate=bitrate)


def File(path, easy=False):
    return DummyAudio()


def fake_compute(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kwargs):
    for p in paths:
        yield p, 0, fp_map.get(os.path.basename(p), 'x')


def _patch_fingerprints(monkeypatch, compute):
    # Patch the module object library_sync actually bound (its vendored
    # engine's fingerprint_generator), not whatever sits in sys.modules.
    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", compute)


@pytest.fixture(autouse=True)
def _media_stubs(monkeypatch):
    """Every file reads as tagged A/T/AL at 128 kbps; fingerprints come from fp_map.

    Library Sync reads tags through the vendored engine's ``get_tags`` (which
    goes through the shared reader in ``utils.audio_metadata_reader``) and
    bitrate through the vendored ``MutagenFile``; patch both bindings.
    """
    import utils.audio_metadata_reader as reader

    monkeypatch.setattr(reader, "MutagenFile", File)
    monkeypatch.setattr(library_sync.idx, "MutagenFile", File)
    _patch_fingerprints(monkeypatch, fake_compute)


def test_quality_score_simple():
    info = {'ext': '.flac', 'bitrate': 1000}
    score = compute_quality_score(info, {'.flac': 3, '.mp3': 1})
    assert score == 3000
    fallback = compute_quality_score({"ext": ".mp3", "bitrate": None}, {".mp3": 2})
    assert fallback == 2


def test_compare_libraries(tmp_path):
    lib = tmp_path / 'lib'
    inc = tmp_path / 'inc'
    lib.mkdir()
    inc.mkdir()
    (lib / 'a.flac').write_text('x')
    (lib / 'b.mp3').write_text('x')
    (inc / 'a.flac').write_text('x')
    (inc / 'b.flac').write_text('x')
    (inc / 'new.mp3').write_text('x')

    db = tmp_path / 'fp.db'
    res = compare_libraries(str(lib), str(inc), str(db))

    existing_paths = set(res["existing"])
    assert existing_paths == {str(lib / 'a.flac'), str(lib / 'b.mp3')}

    incoming_paths = set(res["new_tracks"])
    assert incoming_paths == {
        str(inc / 'a.flac'),
        str(inc / 'b.flac'),
        str(inc / 'new.mp3'),
    }

    new_incoming = set(res["new"])
    assert str(inc / 'new.mp3') in new_incoming

    existing_pairs = set(res["existing_matches"])
    assert (str(inc / 'a.flac'), str(lib / 'a.flac')) in existing_pairs

    upgrades = set(res["improved"])
    assert (str(inc / 'b.flac'), str(lib / 'b.mp3')) in upgrades

    matches = {m["incoming"]["path"]: m for m in res["matches"]}
    assert matches[str(inc / "a.flac")]["status"] == MatchStatus.EXACT_MATCH.value
    assert matches[str(inc / "b.flac")]["quality_label"] == "Potential Upgrade"
    assert matches[str(inc / "new.mp3")]["status"] == MatchStatus.NEW.value

    # cleanup cache between tests
    flush_cache(str(db))


def test_compare_libraries_thresholds(tmp_path, monkeypatch):
    lib = tmp_path / 'lib'
    inc = tmp_path / 'inc'
    lib.mkdir()
    inc.mkdir()
    (lib / 'loose.wav').write_text('x')
    (inc / 'loose.wav').write_text('x')

    def fake_fp(path, **kw):
        if "inc" in path:
            return "1 1 1 1 1 1 1 2"
        return "1 1 1 1 1 1 1 1"

    def fake_compute(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kwargs):
        for p in paths:
            fp = fake_fp(p)
            yield p, 0, fp

    _patch_fingerprints(monkeypatch, fake_compute)

    db = tmp_path / 'fp.db'
    res = compare_libraries(
        str(lib),
        str(inc),
        str(db),
        thresholds={".wav": 0.1, "default": 0.3},
    )

    match = res["matches"][0]
    assert match["threshold_used"] == 0.1
    assert match["status"] == MatchStatus.LOW_CONFIDENCE.value
    flush_cache(str(db))


def test_copy_and_replace(tmp_path):
    with pytest.raises(RuntimeError):
        library_sync.copy_new_tracks(["/tmp/a.flac"], "/tmp/inc", "/tmp/lib")
    with pytest.raises(RuntimeError):
        library_sync.replace_tracks([("/tmp/inc.flac", "/tmp/lib.mp3")])


def test_format_threshold_match(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    inc = tmp_path / "inc"
    lib.mkdir()
    inc.mkdir()

    def make_wav(path):
        import wave
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\x00\x00" * 44100)

    mp3_file = lib / "song.mp3"
    wav_file = inc / "song.wav"
    make_wav(mp3_file)
    make_wav(wav_file)

    def fake_fp(path, **kw):
        if path.endswith(".wav"):
            return "1 2 3 4"
        return "1 2 3 5"

    def fake_compute(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kwargs):
        for p in paths:
            fp = fake_fp(p)
            yield p, 0, fp

    _patch_fingerprints(monkeypatch, fake_compute)

    db = tmp_path / "fp.db"
    res = compare_libraries(
        str(lib),
        str(inc),
        str(db),
        thresholds={".wav": 0.5, ".mp3": 0.5},
    )

    assert (str(wav_file), str(mp3_file)) in set(res["improved"])
    flush_cache(str(db))


def test_compare_libraries_relaxed_threshold(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    inc = tmp_path / "inc"
    lib.mkdir()
    inc.mkdir()

    def make_wav(path):
        import wave

        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(b"\x00\x00" * 44100)

    mp3_file = lib / "same.mp3"
    wav_file = inc / "same.wav"
    make_wav(mp3_file)
    make_wav(wav_file)

    def fake_fp(path, **kw):
        if path.endswith(".wav"):
            return "10 20 30 40 50"
        return "10 20 30 40 51"

    def fake_compute(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kwargs):
        for p in paths:
            fp = fake_fp(p)
            yield p, 0, fp

    _patch_fingerprints(monkeypatch, fake_compute)

    db = tmp_path / "fp.db"
    res = compare_libraries(
        str(lib),
        str(inc),
        str(db),
        threshold=0.5,
    )

    assert (str(wav_file), str(mp3_file)) in set(res["improved"])
    flush_cache(str(db))
