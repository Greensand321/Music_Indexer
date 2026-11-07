from dry_run_coordinator import DryRunCoordinator
from near_duplicate_detector import find_near_duplicates


def test_cross_album_clusters_and_html():
    infos = {
        'a1': {'fp': '1 2', 'album': 'A', 'title': 'Song', 'primary': 'a', 'meta_count': 1},
        'b1': {'fp': '1 2', 'album': 'B', 'title': 'Song', 'primary': 'a', 'meta_count': 1},
        'c1': {'fp': '3 4', 'album': 'C', 'title': 'Song', 'primary': 'a', 'meta_count': 1},
    }
    coord = DryRunCoordinator()
    find_near_duplicates(infos, {'.mp3': 0}, 1.0, enable_cross_album=True, coord=coord)
    clusters = [set(c) for c in coord.near_dupe_clusters]
    assert any({'a1', 'b1'} <= c for c in clusters)
    assert 'Phase C – Cross-Album Near-Duplicates' in coord._sections.get('C', '')
