"""Regression tests for the playlist writers (writers review, findings 1–3).

``update_playlists`` runs as a side effect of File Cleanup, the Indexer,
Library Sync and Pair Review. It used to (a) treat a non-UTF-8 playlist as
empty and rewrite it holding only the renamed track, (b) scramble every
folder playlist into a sorted set with ``#EXTINF`` lines detached, and (c)
silently fuzzy-"repair" unrelated missing entries. The Repair preview crashed
on one non-UTF-8 playlist.
"""
from __future__ import annotations

import os
from pathlib import Path

import playlist_generator as pg


def _touch(root: Path, rel: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\0")
    return str(p)


def _rename(old: str, new: str) -> None:
    os.rename(old, new)


def _backups(root: Path) -> list[Path]:
    d = root / "Docs" / pg.PLAYLIST_BACKUP_DIR
    return sorted(d.iterdir()) if d.is_dir() else []


# ── Finding 1: non-UTF-8 playlists are edited in place, not emptied ──────


def test_latin1_playlist_keeps_every_entry_order_and_encoding(tmp_path):
    alb = tmp_path / "Artist" / "Album"
    for n in ("01 - Café.mp3", "02 - Zebra.mp3", "03 - Song (1).mp3"):
        _touch(alb, n)
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    pl = pl_dir / "Artist_Album.m3u"
    original = (
        "#EXTM3U\n../Artist/Album/02 - Zebra.mp3\n../Artist/Album/01 - Café.mp3\n"
        "../Artist/Album/03 - Song (1).mp3\n"
    ).encode("latin-1")
    pl.write_bytes(original)

    old = str(alb / "03 - Song (1).mp3")
    new = str(alb / "03 - Song.mp3")
    _rename(old, new)
    assert pg.update_playlists({old: new}, library_root=str(tmp_path)) == 1

    assert pl.read_bytes() == original.replace(
        b"../Artist/Album/03 - Song (1).mp3",
        b"../Artist/Album/03 - Song.mp3",   # keeps the entry's "/" style on every OS
    )
    # Still latin-1 (the é is one byte), still in the user's order.
    assert b"Caf\xe9" in pl.read_bytes()
    lines = pl.read_bytes().decode("latin-1").splitlines()
    assert lines[0] == "#EXTM3U"
    assert lines[1].endswith("02 - Zebra.mp3") and lines[2].endswith("01 - Café.mp3")
    # The overwritten file was backed up first, and the write was logged.
    baks = _backups(tmp_path)
    assert len(baks) == 1 and baks[0].read_bytes() == original
    log = (tmp_path / "Docs" / "playlist_writes.log").read_text(encoding="utf-8")
    assert "BACKUP" in log and "UPDATE" in log


def test_extinf_pairing_and_order_survive_a_rename(tmp_path):
    alb = tmp_path / "Artist" / "Album"
    for n in ("01 - A (1).mp3", "02 - B.mp3", "03 - C.mp3"):
        _touch(alb, n)
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    pl = pl_dir / "Artist_Album.m3u"
    text = (
        "#EXTM3U\n#EXTINF:1,Three\n../Artist/Album/03 - C.mp3\n"
        "#EXTINF:1,One\n../Artist/Album/01 - A (1).mp3\n"
        "#EXTINF:1,Two\n../Artist/Album/02 - B.mp3\n"
    )
    pl.write_text(text, encoding="utf-8")
    old, new = str(alb / "01 - A (1).mp3"), str(alb / "01 - A.mp3")
    _rename(old, new)

    assert pg.update_playlists({old: new}, library_root=str(tmp_path)) == 1
    expected = text.replace(
        "../Artist/Album/01 - A (1).mp3", "../Artist/Album/01 - A.mp3"
    )
    assert pl.read_text(encoding="utf-8") == expected


def test_untouched_playlists_are_not_rewritten_or_backed_up(tmp_path):
    a = _touch(tmp_path, "X/a (1).mp3")
    _touch(tmp_path, "Y/b.mp3")
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    other = pl_dir / "Y.m3u"
    other.write_text("../Y/b.mp3\n../Y/zz.mp3\n", encoding="utf-8")
    before = other.read_bytes()
    new = str(tmp_path / "X" / "a.mp3")
    _rename(a, new)
    assert pg.update_playlists({a: new}, library_root=str(tmp_path)) == 0
    assert other.read_bytes() == before
    assert _backups(tmp_path) == []


def test_crlf_endings_bom_and_absolute_entries_are_kept(tmp_path):
    old = _touch(tmp_path, "A/t (1).flac")
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    pl = pl_dir / "mix.m3u8"
    content = f"#EXTM3U\r\n#EXTINF:5,T\r\n{old}\r\n# my note\r\n"
    pl.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
    new = str(tmp_path / "A" / "t.flac")
    _rename(old, new)

    assert pg.update_playlists({old: new}, library_root=str(tmp_path)) == 1
    data = pl.read_bytes()
    assert data == b"\xef\xbb\xbf" + content.replace(old, new).encode("utf-8")


def test_deleted_entry_takes_its_extinf_line_with_it(tmp_path):
    gone = _touch(tmp_path, "A/gone.mp3")
    _touch(tmp_path, "A/kept.mp3")
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    pl = pl_dir / "p.m3u"
    pl.write_text(
        "#EXTM3U\n#EXTINF:1,Gone\n../A/gone.mp3\n#EXTINF:1,Kept\n../A/kept.mp3\n",
        encoding="utf-8",
    )
    os.remove(gone)
    assert pg.update_playlists({gone: None}, library_root=str(tmp_path)) == 1
    assert pl.read_text(encoding="utf-8") == "#EXTM3U\n#EXTINF:1,Kept\n../A/kept.mp3\n"


def test_name_that_does_not_fit_the_playlist_encoding_leaves_it_untouched(tmp_path):
    old = _touch(tmp_path, "A/Cafe (1).mp3")
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    pl = pl_dir / "p.m3u"
    original = "../A/Café.mp3\n../A/Cafe (1).mp3\n".encode("cp1252")
    pl.write_bytes(original)
    new = str(tmp_path / "A" / "日本.mp3")
    _rename(old, new)
    logs: list[str] = []
    assert pg.update_playlists({old: new}, library_root=str(tmp_path), log_callback=logs.append) == 0
    assert pl.read_bytes() == original
    assert any("left unchanged" in m for m in logs)


def test_unreadable_playlist_is_never_written(tmp_path):
    old = _touch(tmp_path, "A/x (1).mp3")
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    broken = pl_dir / "broken.m3u"
    broken.symlink_to(tmp_path / "does-not-exist.m3u")
    new = str(tmp_path / "A" / "x.mp3")
    _rename(old, new)
    logs: list[str] = []
    assert pg.update_playlists({old: new}, library_root=str(tmp_path), log_callback=logs.append) == 0
    assert not (tmp_path / "does-not-exist.m3u").exists()
    assert any("Could not read" in m for m in logs)


def test_a_rename_never_merges_tracks_into_folder_playlists(tmp_path):
    # The old merge pass appended the renamed track to the folder playlist of
    # its folder, sorted, with the user's order lost.
    alb = tmp_path / "Art" / "Alb"
    _touch(alb, "b.mp3")
    old = _touch(alb, "a (1).mp3")
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    folder_pl = pl_dir / "Art_Alb.m3u"
    folder_pl.write_text("#EXTM3U\n../Art/Alb/b.mp3\n", encoding="utf-8")
    before = folder_pl.read_bytes()
    new = str(alb / "a.mp3")
    _rename(old, new)
    assert pg.update_playlists({old: new}, library_root=str(tmp_path)) == 0
    assert folder_pl.read_bytes() == before


def test_iterable_of_paths_changes_nothing(tmp_path):
    a = _touch(tmp_path, "A/a.mp3")
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    pl = pl_dir / "A.m3u"
    pl.write_text("#EXTM3U\n../A/zz.mp3\n", encoding="utf-8")
    before = pl.read_bytes()
    assert pg.update_playlists([a], library_root=str(tmp_path)) == 0
    assert pl.read_bytes() == before


# ── Finding 2: no silent fuzzy repair of unrelated entries ───────────────


def test_unrelated_missing_entries_are_not_fuzzy_repaired(tmp_path):
    other = tmp_path / "Other" / "Best"
    _touch(other, "05 - Love Me Tender.mp3")
    _touch(other, "06 - Love Me Do.mp3")
    old = _touch(tmp_path, "Artist/Album/03 - Song (1).mp3")
    pl_dir = tmp_path / "Playlists"
    pl_dir.mkdir()
    mix = pl_dir / "Mix.m3u8"
    text = (
        "#EXTM3U\n#EXTINF:120,Elvis - Love Me Tender (Live)\n"
        "../Other/Best/05 - Love Me Tender (Live).mp3\n"
        "../Artist/Album/03 - Song (1).mp3\n"
    )
    mix.write_text(text, encoding="utf-8")
    new = str(tmp_path / "Artist" / "Album" / "03 - Song.mp3")
    _rename(old, new)

    assert pg.update_playlists({old: new}, library_root=str(tmp_path)) == 1
    out = mix.read_text(encoding="utf-8")
    assert "05 - Love Me Tender (Live).mp3" in out, "an unrelated entry was rewritten"
    assert out == text.replace(
        "../Artist/Album/03 - Song (1).mp3", "../Artist/Album/03 - Song.mp3"
    )


# ── Finding 3: Repair preview survives a non-UTF-8 playlist ──────────────


def test_repair_preview_reads_latin1_playlists_and_keeps_their_encoding(tmp_path):
    lib = tmp_path
    _touch(lib, "A/Café.mp3")
    pl_dir = lib / "Playlists"
    pl_dir.mkdir()
    good = pl_dir / "good.m3u"
    good.write_text("../A/missing.mp3\n", encoding="utf-8")
    latin = pl_dir / "latin.m3u"
    latin.write_bytes("#EXTM3U\n../Old/Café.mp3\n".encode("latin-1"))

    plan = pg.plan_playlist_repair([str(good), str(latin)], str(lib))

    assert [p.error for p in plan] == ["", ""]
    item = plan[1]
    assert item.changes == [("../Old/Café.mp3", "../A/Café.mp3")]   # entry's "/" style kept
    result = pg.write_playlist_plan([item], log_dir=str(lib / "Docs"))
    assert result["errors"] == []
    data = latin.read_bytes()
    assert b"Caf\xe9" in data and "Café".encode("utf-8") not in data


def test_repair_preview_marks_only_the_unreadable_playlist(tmp_path):
    lib = tmp_path
    _touch(lib, "A/x.mp3")
    pl_dir = lib / "Playlists"
    pl_dir.mkdir()
    good = pl_dir / "good.m3u"
    good.write_text("../B/x.mp3\n", encoding="utf-8")
    missing = pl_dir / "gone.m3u"
    plan = pg.plan_playlist_repair([str(missing), str(good)], str(lib))
    assert plan[0].error and not plan[1].error
    assert plan[1].changes


def test_repair_write_refuses_to_reencode(tmp_path):
    lib = tmp_path
    pl = lib / "p.m3u"
    pl.write_bytes("../A/Café.mp3\n".encode("cp1252"))
    item = pg.PlannedPlaylist(path=str(pl), lines=["日本.mp3"], exists=True, entries=1,
                              encoding="cp1252")
    before = pl.read_bytes()
    result = pg.write_playlist_plan([item])
    assert result["written"] == [] and len(result["errors"]) == 1
    assert pl.read_bytes() == before
