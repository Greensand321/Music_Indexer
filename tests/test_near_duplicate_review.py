from dry_run_coordinator import DryRunCoordinator
from near_duplicate_detector import find_near_duplicates


def test_near_duplicates_marked_for_review():
    infos = {
        "a1": {"fp": "1 2 3", "album": "A", "title": "Song", "primary": "Artist", "meta_count": 2},
        "a2": {"fp": "1 2 3", "album": "A", "title": "Song", "primary": "Artist", "meta_count": 1},
    }
    result = find_near_duplicates(infos, {".mp3": 0}, 0.5)
    assert result.review_required
    assert dict(result.items()) == {}
    assert result.review_groups
    group = result.review_groups[0]
    assert set(group.losers) == {"a2"} or set(group.losers) == {"a1"}
    assert "review required" in group.reason.lower()


def test_metadata_gate_blocks_conflicting_titles():
    infos = {
        "x1": {"fp": "1 2 3", "album": "Mix", "title": "Song Alpha", "primary": "Artist", "meta_count": 1},
        "x2": {"fp": "1 2 3", "album": "Mix", "title": "Song Beta", "primary": "Artist", "meta_count": 1},
    }
    coord = DryRunCoordinator()
    result = find_near_duplicates(infos, {".mp3": 0}, 0.5, coord=coord)
    assert not result.review_required
    assert not coord.near_dupe_clusters
