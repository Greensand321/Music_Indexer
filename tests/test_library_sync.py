import types
import sys
import os

# Stub mutagen and chromaprint_utils
mutagen_stub = types.ModuleType('mutagen')
class DummyAudio:
    def __init__(self, bitrate=128000):
        self.tags = {'artist': ['A'], 'title': ['T'], 'album': ['AL']}
        self.info = types.SimpleNamespace(bitrate=bitrate)

def File(path, easy=False):
    return DummyAudio()
mutagen_stub.File = File
id3_stub = types.ModuleType('id3')
id3_stub.ID3NoHeaderError = Exception
mutagen_stub.id3 = id3_stub
sys.modules['mutagen'] = mutagen_stub
sys.modules['mutagen.id3'] = id3_stub

chroma_stub = types.ModuleType('chromaprint_utils')
fp_map = {
    'a.flac': '1 2',
    'b.mp3': '2 3',
    'b.flac': '2 3',
    'new.mp3': '9 9',
}
chroma_stub.fingerprint_fpcalc = lambda p: fp_map.get(os.path.basename(p), 'x')
sys.modules['chromaprint_utils'] = chroma_stub

import importlib
import music_indexer_api
importlib.reload(music_indexer_api)
import library_sync
importlib.reload(library_sync)
from library_sync import compare_libraries, compute_quality_score
from fingerprint_cache import flush_cache


def test_quality_score_simple():
    info = {'ext': '.flac', 'bitrate': 1000}
    score = compute_quality_score(info, {'.flac': 3, '.mp3': 1})
    assert score == 3000


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

    new_set = set(res['new'])
    assert str(inc / 'new.mp3') in new_set

    ex_pairs = set(res['existing'])
    assert (str(inc / 'a.flac'), str(lib / 'a.flac')) in ex_pairs

    imp_pairs = set(res['improved'])
    assert (str(inc / 'b.flac'), str(lib / 'b.mp3')) in imp_pairs

    # cleanup cache between tests
    flush_cache(str(db))


def test_compare_libraries_thresholds(tmp_path):
    lib = tmp_path / 'lib'
    inc = tmp_path / 'inc'
    lib.mkdir()
    inc.mkdir()
    (lib / 'a.flac').write_text('x')
    (inc / 'a.flac').write_text('x')

    db = tmp_path / 'fp.db'
    res = compare_libraries(
        str(lib),
        str(inc),
        str(db),
        thresholds={"default": 0.0},
    )

    assert str(inc / 'a.flac') in set(res['new'])
    flush_cache(str(db))


def test_copy_and_replace(tmp_path):
    lib = tmp_path / 'lib'
    inc = tmp_path / 'inc'
    lib.mkdir()
    inc.mkdir()
    f_new = inc / 'new.flac'
    f_new.write_text('x')
    copied = library_sync.copy_new_tracks([str(f_new)], str(inc), str(lib))
    assert len(copied) == 1
    assert os.path.exists(copied[0])

    f_old = lib / 'old.mp3'
    f_old.write_text('old')
    f_better = inc / 'old.flac'
    f_better.write_text('better')
    replaced = library_sync.replace_tracks([(str(f_better), str(f_old))])
    assert replaced == [str(f_old)]
    backup = lib / '__backup__' / 'old.mp3'
    assert backup.exists()
    assert f_old.read_text() == 'better'


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

    def fake_fp(path):
        if path.endswith(".wav"):
            return "1 2 3 4"
        return "1 2 3 5"

    monkeypatch.setattr(sys.modules["chromaprint_utils"], "fingerprint_fpcalc", fake_fp)

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

    def fake_fp(path):
        if path.endswith(".wav"):
            return "10 20 30 40 50"
        return "10 20 30 40 51"

    monkeypatch.setattr(sys.modules["chromaprint_utils"], "fingerprint_fpcalc", fake_fp)

    db = tmp_path / "fp.db"
    res = compare_libraries(
        str(lib),
        str(inc),
        str(db),
        threshold=0.5,
    )

    assert (str(wav_file), str(mp3_file)) in set(res["improved"])
    flush_cache(str(db))
