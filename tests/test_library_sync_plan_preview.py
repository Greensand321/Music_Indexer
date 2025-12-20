import os
import sys
import threading
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

from library_sync import LibrarySyncPlan, _compute_plan_items, compute_library_sync_plan, execute_library_sync_plan


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


def test_execute_library_sync_plan_moves_files(tmp_path):
    library_root = tmp_path / "library"
    incoming_root = tmp_path / "incoming"
    library_root.mkdir()
    incoming_root.mkdir()
    (library_root / "Music").mkdir()

    incoming_song = incoming_root / "artist - track.mp3"
    incoming_song.write_text("audio")

    plan = compute_library_sync_plan(str(library_root), str(incoming_root))
    dest_path = plan.moves[str(incoming_song)]

    summary = execute_library_sync_plan(plan)

    assert summary["moved"] == 1
    assert not summary.get("cancelled")
    assert dest_path and os.path.exists(dest_path)
    assert not incoming_song.exists()


def test_execute_library_sync_plan_honors_cancellation(tmp_path):
    library_root = tmp_path / "library"
    incoming_root = tmp_path / "incoming"
    library_root.mkdir()
    incoming_root.mkdir()
    (library_root / "Music").mkdir()

    incoming_song = incoming_root / "artist - track.mp3"
    incoming_song.write_text("audio")

    plan = compute_library_sync_plan(str(library_root), str(incoming_root))
    cancel_event = threading.Event()
    cancel_event.set()

    summary = execute_library_sync_plan(plan, cancel_event=cancel_event)

    assert summary["cancelled"]
    assert incoming_song.exists()


def test_preview_marks_duplicate_items(tmp_path):
    library_root = tmp_path / "library"
    incoming_root = tmp_path / "incoming"
    dest_root = library_root / "Music"
    dest_root.mkdir(parents=True)
    incoming_root.mkdir()

    src = incoming_root / "dup.mp3"
    dst = dest_root / "dup.mp3"
    src.write_text("audio")
    dst.write_text("audio")

    plan = LibrarySyncPlan(
        library_root=str(library_root),
        incoming_root=str(incoming_root),
        destination_root=str(dest_root),
        moves={str(src): str(dst)},
        tag_index={str(dst): {"leftover_tags": []}},
        decision_log=[],
    )
    plan.items = _compute_plan_items(plan.moves, destination_root=str(dest_root))
    html_path = tmp_path / "preview.html"
    plan.render_preview(str(html_path))
    content = html_path.read_text()

    assert "(duplicate)" in content
