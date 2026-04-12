import json
import os
import sys
import types
from pathlib import Path

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

from library_sync import LibrarySyncPlan, execute_plan


def _build_plan(tmp_path, filename="track.mp3"):
    library_root = tmp_path / "library"
    incoming_root = tmp_path / "incoming"
    dest_root = library_root / "Music"
    dest_root.mkdir(parents=True)
    incoming_root.mkdir()
    library_root.mkdir(exist_ok=True)

    src = incoming_root / filename
    src.write_text("new audio")
    dest = dest_root / "Artist" / filename

    plan = LibrarySyncPlan(
        library_root=str(library_root),
        incoming_root=str(incoming_root),
        destination_root=str(dest_root),
        moves={str(src): str(dest)},
        tag_index={str(dest): {"leftover_tags": []}},
        decision_log=[],
    )
    return plan, src, dest


def test_execute_plan_writes_audit_and_report(tmp_path):
    plan, src, dest = _build_plan(tmp_path)
    audit_path = tmp_path / "audit.json"
    report_path = tmp_path / "executed.html"
    playlist_path = tmp_path / "executed.m3u"

    summary = execute_plan(
        plan,
        audit_path=str(audit_path),
        executed_report_path=str(report_path),
        playlist_path=str(playlist_path),
        log_callback=lambda _m: None,
    )

    assert summary["moved"] == 1
    assert summary["audit_path"] == str(audit_path)
    assert dest.exists()
    assert not src.exists()

    payload = json.loads(audit_path.read_text())
    assert payload["items"][0]["status"] == "success"
    assert payload["summary"]["moved"] == 1

    content = report_path.read_text()
    assert "Library Sync (Executed" in content
    assert dest.name in content

    assert summary["playlist_path"] == str(playlist_path)
    assert playlist_path.exists()
    assert dest.name in playlist_path.read_text()


def test_execute_plan_handles_replacements_and_backups(tmp_path):
    plan, src, dest = _build_plan(tmp_path, "replace.mp3")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("old audio")

    audit_path = tmp_path / "audit_skip.json"
    report_path = tmp_path / "report_skip.html"

    summary = execute_plan(
        plan,
        audit_path=str(audit_path),
        executed_report_path=str(report_path),
        create_playlist=False,
        log_callback=lambda _m: None,
    )

    assert summary["moved"] == 0
    assert dest.read_text() == "old audio"
    assert src.exists()
    payload = json.loads(audit_path.read_text())
    assert payload["items"][0]["status"] == "skipped"
    assert payload["summary"]["review_required"]

    plan.allowed_replacements.add(str(dest))
    audit_path2 = tmp_path / "audit_replace.json"
    report_path2 = tmp_path / "report_replace.html"

    summary2 = execute_plan(
        plan,
        audit_path=str(audit_path2),
        executed_report_path=str(report_path2),
        create_playlist=False,
        log_callback=lambda _m: None,
    )

    assert summary2["moved"] == 1
    assert not src.exists()
    assert dest.read_text() == "new audio"
    assert summary2["backups"]
    backup_path = summary2["backups"][0]
    assert backup_path and os.path.exists(backup_path)
    assert Path(backup_path).read_text() == "old audio"
