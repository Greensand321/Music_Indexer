"""Opus library mirror backend (TOOL-05, TOOL-14, TOOL-20).

* cancel terminates the running ffmpeg and removes its partial output;
* output only gets its final name once encoded *and* tagged, so an interrupted
  run never leaves an untagged .opus that later runs skip forever;
* reserved non-music folders (Quarantine/, Trash/, Docs/) are not mirrored;
* per-tag log spam is gone; failures go to ``error_callback``.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from utils import opus_library_mirror as mirror


def _fake_ffmpeg(bin_dir: Path, mode: str) -> Path:
    """Install a fake ``ffmpeg`` on PATH.

    mode "hang": write a few bytes to the output, record its pid, sleep.
    mode "fail": exit 1 with an error message.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "ffmpeg"
    pid_file = bin_dir / "ffmpeg.pid"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        f"mode = {mode!r}\n"
        "out = sys.argv[-1]\n"
        "if mode == 'fail':\n"
        "    sys.stderr.write('Unknown encoder libopus')\n"
        "    sys.exit(1)\n"
        "open(out, 'wb').write(b'OggS-partial')\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return pid_file


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); check its state on Linux.
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        return "zombie" not in status.read_text().lower()
    return True


def _flac_library(root: Path) -> Path:
    from library_factory import TrackSpec, build_library

    lib = build_library(root, [
        TrackSpec("Basalt", "Fault Line", "Granite", 1, "2021", "Rock", ext=".flac"),
        TrackSpec("Basalt", "Magma", "Granite", 2, "2021", "Rock", ext=".mp3"),
        TrackSpec("Basalt", "Loser", None, ext=".mp3", relpath="Quarantine/loser.mp3"),
    ])
    (root / "Trash").mkdir()
    (root / "Trash" / "junk.txt").write_text("junk")
    (root / "Docs").mkdir()
    (root / "Docs" / "report.html").write_text("<html/>")
    return lib.root


def test_cancel_terminates_ffmpeg_and_removes_partial_output(tmp_path, monkeypatch):
    from library_factory import TrackSpec, build_library
    # Build the real FLAC first: the factory itself needs the real ffmpeg.
    build_library(tmp_path / "lib", [TrackSpec("A", "T", "B", 1, ext=".flac", relpath="in.flac")])
    src = tmp_path / "lib" / "in.flac"
    pid_file = _fake_ffmpeg(tmp_path / "bin", "hang")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    dest = tmp_path / "out" / "in.opus"
    dest.parent.mkdir()

    stop = threading.Event()
    result: dict = {}
    t = threading.Thread(target=lambda: result.update(
        r=mirror.convert_flac_to_opus(str(src), str(dest), False, cancel_event=stop)))
    t.start()
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.exists(), "fake ffmpeg never started"
    pid = int(pid_file.read_text())
    assert any(p.name.startswith(".in") for p in dest.parent.iterdir()), "partial file expected while running"

    stop.set()
    t.join(10)
    assert not t.is_alive()
    assert result["r"] == (False, "Cancelled.")
    assert not _pid_alive(pid), "ffmpeg must be terminated, not orphaned"
    assert list(dest.parent.iterdir()) == [], "partial output must be removed"


def test_mirror_cancel_reports_not_started_files(tmp_path, monkeypatch):
    lib = _flac_library(tmp_path / "Music")
    stop = threading.Event()
    stop.set()
    summary = mirror.mirror_library(str(lib), str(tmp_path / "Mirror"), False, cancel_event=stop)
    assert summary["cancelled"] is True
    assert summary["converted"] == summary["copied"] == 0


def test_metadata_failure_leaves_no_untagged_file(tmp_path, monkeypatch):
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    lib = _flac_library(tmp_path / "Music")
    monkeypatch.setattr(mirror, "_write_opus_metadata", lambda *a, **k: "Failed to write Opus metadata: boom")
    errors: list[str] = []
    summary = mirror.mirror_library(str(lib), str(tmp_path / "Mirror"), False, error_callback=errors.append)
    assert summary["errors"] == 1
    out = tmp_path / "Mirror" / "Basalt" / "Granite"
    assert not list(out.glob("*.opus")), "an untagged .opus must not be left behind"
    assert not [p for p in out.iterdir() if ".alphadex-partial" in p.name]
    # A second run converts it again instead of skipping it as "already exists".
    monkeypatch.undo()
    summary2 = mirror.mirror_library(str(lib), str(tmp_path / "Mirror"), False)
    assert summary2["converted"] == 1


def test_ffmpeg_failure_goes_to_error_callback(tmp_path, monkeypatch):
    lib = _flac_library(tmp_path / "Music")
    _fake_ffmpeg(tmp_path / "bin", "fail")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    infos: list[str] = []
    errors: list[str] = []
    summary = mirror.mirror_library(
        str(lib), str(tmp_path / "Mirror"), False,
        log_callback=infos.append, error_callback=errors.append,
    )
    assert summary["errors"] == 1
    assert any("Unknown encoder" in e for e in errors)
    assert not [m for m in infos if "normalized tag" in m]


def test_real_mirror_skips_reserved_folders_and_keeps_tags(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    lib = _flac_library(tmp_path / "Music")
    dest = tmp_path / "Mirror"
    logs: list[str] = []
    summary = mirror.mirror_library(str(lib), str(dest), False, log_callback=logs.append)

    assert summary["converted"] == 1 and summary["copied"] == 1 and summary["errors"] == 0
    assert sorted(summary["excluded_folders"]) == ["Docs", "Quarantine", "Trash"]
    assert not (dest / "Quarantine").exists()
    assert not (dest / "Trash").exists()
    assert not (dest / "Docs").exists()
    opus = dest / "Basalt" / "Granite" / "01 - Fault Line.opus"
    assert opus.exists()
    from mutagen.oggopus import OggOpus
    assert OggOpus(str(opus))["title"] == ["Fault Line"]
    assert not [p for p in dest.rglob("*") if ".alphadex-partial" in p.name]
    assert not [m for m in logs if "normalized tag" in m], "per-tag log spam is gone"
