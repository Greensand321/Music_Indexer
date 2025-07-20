import types
import sys
import os

# Stub mutagen and acoustid
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

acoustid_stub = types.ModuleType('acoustid')
fp_map = {
    'a.flac': '1 2',
    'b.mp3': '2 3',
    'b.flac': '2 3',
    'new.mp3': '9 9',
}
acoustid_stub.fingerprint_file = lambda p: (0, fp_map.get(os.path.basename(p), 'x'))
sys.modules['acoustid'] = acoustid_stub

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
