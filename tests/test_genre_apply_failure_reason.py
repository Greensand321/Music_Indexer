"""A failed genre write records *why* in the change log (locked, read-only, ...)."""
from __future__ import annotations

from controllers.normalize_controller import GenreChange, apply_genre_changes
from utils.tag_change_log import TagChangeLog, read_change_log


def test_failed_write_reason_reaches_the_change_log(tmp_path, monkeypatch):
    import update_genres
    from utils.tag_writer import TagWriteError

    def refuse(path, genres):
        raise TagWriteError("file is open in another program")

    monkeypatch.setattr(update_genres, "write_genre_tag", refuse)
    target = tmp_path / "a.flac"
    target.write_bytes(b"")
    with TagChangeLog(str(tmp_path), "genre_normalize") as log:
        # read_tags=None with the default writer would verify via the real reader;
        # pass a reader that reports the previewed state so the write is attempted.
        result = apply_genre_changes(
            [GenreChange(str(target), ["Rock"], ["Pop"])],
            change_log=log,
            read_tags=lambda _p: {"genre": ["Rock"]},
        )
    assert result.failed == [str(target)] and not result.applied
    (entry,) = read_change_log(log.path)
    assert entry["status"] == "failed"
    assert "open in another program" in str(entry.get("error") or entry.get("reason") or entry)
