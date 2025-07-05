import threading
from dry_run_coordinator import DryRunCoordinator


def test_concurrent_additions():
    coord = DryRunCoordinator()
    def worker():
        for _ in range(50):
            coord.add_exact_dupes([1])
            coord.add_metadata_groups([2])
            coord.add_near_dupe_clusters([[3]])
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(coord.exact_dupes) == 200
    assert len(coord.metadata_groups) == 200
    assert len(coord.near_dupe_clusters) == 200


def test_html_assembly_order():
    coord = DryRunCoordinator()
    coord.set_html_section('B', 'b')
    coord.set_html_section('A', 'a')
    coord.set_html_section('C', 'c')
    result = coord.assemble_final_report()
    assert result == 'a\nb\nc'

