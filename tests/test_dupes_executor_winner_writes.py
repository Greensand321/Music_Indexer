"""The winner is the copy the user keeps: artwork/metadata writes must not damage it.

Before:
* artwork for anything that was not FLAC or MP4 went through an ID3 fallback
  that *prepended an ID3 header* to OGG, Opus, WAV and AIFF winners, leaving
  them unreadable (mutagen: "can't sync to MPEG frame") — and the step was
  reported as verified, because verification only looked at the sidecar file;
* tags and covers were saved in place, so a failure mid-save (disk full,
  crash) left a truncated winner while the losers were then quarantined;
* the metadata step wrote the whole planned tag dict, not the previewed
  changes, adding stray TRACK=/DISC=/YEAR= fields to FLAC/Vorbis files.
"""
from __future__ import annotations

import errno
import io
import shutil
from pathlib import Path

import pytest

mutagen = pytest.importorskip("mutagen")

import duplicate_consolidation_executor as ex
from duplicate_consolidation import ArtworkDirective
from duplicate_consolidation_executor import execute_consolidation_plan
from library_factory import encode_clip, ffmpeg_available
from _helpers.dupes_plans import config, group, plan, write

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="needs ffmpeg to make real audio files")


def _jpeg(color="red") -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, "JPEG")
    return buf.getvalue()


def _clip(dest: Path, ext: str, tmp_path: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(encode_clip(ext, 2.0, 440.0, tmp_path / "clips"), dest)
    return dest


def _source_with_sidecar(folder: Path, payload: bytes) -> Path:
    source = write(folder / "source.mp3", b"not read")
    (folder / "source.mp3.artwork").write_bytes(payload)
    return source


@pytest.mark.parametrize("ext", [".ogg", ".opus", ".wav", ".flac", ".mp3", ".m4a"])
def test_cover_embedding_keeps_every_format_readable(tmp_path, ext):
    payload = _jpeg()
    target = _clip(tmp_path / "lib" / f"Song{ext}", ext, tmp_path)
    kind_before = type(mutagen.File(str(target))).__name__
    length_before = mutagen.File(str(target)).info.length
    source = _source_with_sidecar(tmp_path / "lib", payload)
    directive = ArtworkDirective(source=str(source), target=str(target), reason="test")

    ok, detail = ex._apply_artwork(directive)

    assert ok, detail
    audio = mutagen.File(str(target))  # raised "can't sync to MPEG frame" for OGG/Opus/WAV
    assert type(audio).__name__ == kind_before
    assert audio.info.length == pytest.approx(length_before, abs=0.05)
    assert payload in ex._embedded_front_covers(audio)
    assert ex._verify_artwork_write(directive) == (True, "Artwork write verified.")


def test_corrupting_embed_is_caught_by_verification(tmp_path, monkeypatch):
    payload = _jpeg()
    target = _clip(tmp_path / "lib" / "Song.ogg", ".ogg", tmp_path)
    source = _source_with_sidecar(tmp_path / "lib", payload)
    directive = ArtworkDirective(source=str(source), target=str(target), reason="test")
    target.write_bytes(b"ID3\x04" + target.read_bytes())  # what the old fallback produced
    ok, _detail = ex._verify_artwork_write(directive)
    assert not ok


def test_failed_save_leaves_the_winner_byte_identical(tmp_path, monkeypatch):
    from mutagen.flac import FLAC

    payload = _jpeg()
    target = _clip(tmp_path / "lib" / "Song.flac", ".flac", tmp_path)
    before = target.read_bytes()
    source = _source_with_sidecar(tmp_path / "lib", payload)

    def disk_full_mid_save(self, *args, **kwargs):
        with open(self.filename, "r+b") as fh:
            fh.truncate(10)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(FLAC, "save", disk_full_mid_save)

    ok, detail = ex._apply_artwork(ArtworkDirective(source=str(source), target=str(target), reason="t"))
    assert not ok and "No space left" in detail
    assert target.read_bytes() == before

    ok, _detail, _skipped = ex._apply_metadata(str(target), {"album": "New"})
    assert not ok
    assert target.read_bytes() == before
    leftovers = sorted(p.name for p in target.parent.iterdir())
    assert leftovers == ["Song.flac", "Song.flac.artwork", "source.mp3", "source.mp3.artwork"]


def test_metadata_step_writes_only_the_previewed_changes(tmp_path):
    from mutagen.flac import FLAC

    winner = _clip(tmp_path / "Artist" / "Single" / "Song.flac", ".flac", tmp_path)
    tags = FLAC(str(winner))
    tags["title"], tags["artist"], tags["tracknumber"], tags["comment"] = ["Song"], ["Artist"], ["1/12"], ["keep me"]
    tags.save()
    loser = _clip(tmp_path / "Artist" / "Album" / "03 Song.mp3", ".mp3", tmp_path)
    planned = {
        "artist": "Artist", "title": "Song", "album": "Album", "album_type": "album",
        "track": 3, "tracknumber": "3/12", "disc": None, "date": None, "year": "2020",
        "genre": None, "cover_hash": "abc",
    }
    changes = {
        "album": {"from": None, "to": "Album"},
        "album_type": {"from": None, "to": "album"},
        "track": {"from": 1, "to": 3},
        "tracknumber": {"from": "1/12", "to": "3/12"},
        "year": {"from": "Unknown", "to": "2020"},
    }
    the_plan = plan(group(winner, [loser], planned_tags=planned, metadata_changes=changes))

    result = execute_consolidation_plan(the_plan, config(tmp_path, apply_metadata=True))

    assert result.success, [a.detail for a in result.actions if a.status in {"failed", "blocked"}]
    after = FLAC(str(winner))
    assert after["album"] == ["Album"]
    assert after["tracknumber"] == ["3/12"]
    assert after["date"] == ["2020"]  # the previewed year lands in the real date field
    assert after["comment"] == ["keep me"] and after["title"] == ["Song"]
    for stray in ("track", "year", "album_type", "cover_hash", "disc"):
        assert stray not in after, stray


def test_end_to_end_artwork_transfer_to_an_ogg_winner(tmp_path):
    payload = _jpeg("blue")
    winner = _clip(tmp_path / "Album" / "Song.ogg", ".ogg", tmp_path)
    loser = _clip(tmp_path / "Album" / "Song.mp3", ".mp3", tmp_path)
    (tmp_path / "Album" / "Song.mp3.artwork").write_bytes(payload)
    directive = ArtworkDirective(source=str(loser), target=str(winner), reason="Fill missing artwork")
    the_plan = plan(group(winner, [loser], artwork=[directive]))

    result = execute_consolidation_plan(the_plan, config(tmp_path, apply_artwork=True))

    assert result.success, [a.detail for a in result.actions if a.status in {"failed", "blocked"}]
    audio = mutagen.File(str(winner))
    assert type(audio).__name__ == "OggVorbis"
    assert payload in ex._embedded_front_covers(audio)
    assert not loser.exists()
