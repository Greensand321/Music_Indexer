import types
import sys

mutagen_stub = types.ModuleType('mutagen')
def File(*a, **k):
    return None
mutagen_stub.File = File
id3_stub = types.ModuleType('id3')
id3_stub.ID3NoHeaderError = Exception
mutagen_stub.id3 = id3_stub
sys.modules['mutagen'] = mutagen_stub
sys.modules['mutagen.id3'] = id3_stub

from dry_run_coordinator import DryRunCoordinator
from near_duplicate_detector import find_near_duplicates


def test_phase_b_clusters_and_html():
    infos = {
        'a1': {'fp': '1 2', 'album': 'A', 'title': 't', 'meta_count': 1},
        'a2': {'fp': '1 2', 'album': 'A', 'title': 't', 'meta_count': 1},
        'b1': {'fp': '3 4', 'album': 'B', 'title': 't', 'meta_count': 1},
        'b2': {'fp': '3 4', 'album': 'B', 'title': 't', 'meta_count': 1},
    }
    coord = DryRunCoordinator()
    find_near_duplicates(infos, {'.mp3': 0}, 1.0, coord=coord)
    clusters = [set(c) for c in coord.near_dupe_clusters]
    assert {frozenset({'a1', 'a2'}), frozenset({'b1', 'b2'})} == {frozenset(c) for c in clusters}
    assert 'Phase B – Album Near-Duplicates' in coord._sections.get('B', '')


def test_phase_b_error(monkeypatch):
    infos = {
        'x1': {'fp': '1', 'album': 'X', 'title': 't', 'meta_count': 1},
        'x2': {'fp': '2', 'album': 'X', 'title': 't', 'meta_count': 1},
    }

    def boom(*a, **k):
        raise RuntimeError('boom')

    monkeypatch.setattr('near_duplicate_detector.fingerprint_distance', boom)
    coord = DryRunCoordinator()
    find_near_duplicates(infos, {'.mp3': 0}, 1.0, coord=coord)
    assert 'Error scanning album' in coord._sections.get('B', '')
