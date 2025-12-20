import os
import sys
import types

# Provide simple stubs for mutagen so the module imports
mutagen_stub = types.ModuleType("mutagen")


class DummyAudio:
    def __init__(self):
        self.tags = None
        self.pictures = []


def File(*_a, **_k):
    return DummyAudio()


mutagen_stub.File = File

id3_stub = types.ModuleType("id3")


class DummyID3Error(Exception):
    pass


id3_stub.ID3NoHeaderError = DummyID3Error
mutagen_stub.id3 = id3_stub
sys.modules.setdefault("mutagen", mutagen_stub)
sys.modules.setdefault("mutagen.id3", id3_stub)
acoustid_stub = types.SimpleNamespace(fingerprint_file=lambda *_a, **_k: (None, None))
sys.modules.setdefault("acoustid", acoustid_stub)

from library_sync import compute_library_sync_plan


def test_library_sync_plan_preview_reuses_plan(tmp_path):
    library_root = tmp_path / "library"
    incoming_root = tmp_path / "incoming"
    library_root.mkdir()
    incoming_root.mkdir()
    music_root = library_root / "Music"
    music_root.mkdir()

    incoming_song = incoming_root / "artist - track.mp3"
    incoming_song.write_text("audio")

    plan = compute_library_sync_plan(str(library_root), str(incoming_root))

    assert plan.destination_root == os.path.normpath(str(music_root))
    assert str(incoming_song) in plan.moves
    dest_path = plan.moves[str(incoming_song)]
    assert dest_path.startswith(str(music_root))

    html_path = tmp_path / "preview.html"
    plan.render_preview(str(html_path))
    content = html_path.read_text()

    assert "Library Sync (Dry Run Preview)" in content
    assert incoming_song.name in content
    assert f"{music_root.name}/" in content
