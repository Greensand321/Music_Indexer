import logging
import types
import sys

mutagen_stub = types.ModuleType('mutagen')
mutagen_stub.File = lambda *a, **k: None
id3_stub = types.ModuleType('id3')
id3_stub.ID3NoHeaderError = Exception
mutagen_stub.id3 = id3_stub
sys.modules['mutagen'] = mutagen_stub
sys.modules['mutagen.id3'] = id3_stub

from near_duplicate_detector import find_near_duplicates

def test_phase_b_logging(caplog):
    infos = {
        'a1': {'fp': '1 2 3', 'album': 'A', 'title': 't'},
        'a2': {'fp': '1 2 3', 'album': 'A', 'title': 't'},
        'b1': {'fp': '1 2 3', 'album': 'B', 'title': 't'},
    }
    with caplog.at_level(logging.DEBUG):
        find_near_duplicates(infos, {'.mp3':0}, 1.0, enable_cross_album=False)
    msgs = [r.message for r in caplog.records]
    assert any('Scanning album' in m and "'A'" in m for m in msgs)
    assert any('Phase B summary' in m for m in msgs)

