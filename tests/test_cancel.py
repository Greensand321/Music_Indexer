import pytest
from music_indexer_api import run_full_indexer
from indexer_control import cancel_event, IndexCancelled


def test_run_full_indexer_cancel(tmp_path):
    cancel_event.set()
    with pytest.raises(IndexCancelled):
        run_full_indexer(str(tmp_path), str(tmp_path / "out.html"))
    cancel_event.clear()
