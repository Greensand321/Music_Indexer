"""The silence-trimming ffmpeg call is bounded like the other fingerprint calls.

``trim_silence`` ran ffmpeg with no timeout (and with the GUI's stdin), so one
file ffmpeg stalled on hung the whole Duplicates scan — cancel could not
interrupt it — and a crash there leaked the temporary WAV.
"""
from __future__ import annotations

import os
import subprocess

import pytest

from _helpers.module_isolation import isolated_modules


@pytest.fixture
def cu(monkeypatch):
    with isolated_modules(fresh=["chromaprint_utils"]) as mods:
        module = mods["chromaprint_utils"]
        monkeypatch.setattr(module, "ensure_tool", lambda name: None)
        monkeypatch.setattr(module, "load_config", lambda: {"fingerprint_subprocess_timeout_sec": 7})
        yield module


def _outputs(cmds):
    return [c[-1] for c in cmds]


def test_trim_silence_has_a_timeout_and_no_stdin(cu, monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(kwargs)

        class P:
            returncode = 0
            stdout = b""
            stderr = b""

        return P()

    monkeypatch.setattr(cu.subprocess, "run", fake_run)
    song = tmp_path / "song.mp3"
    song.write_bytes(b"x")
    out = cu.trim_silence(str(song))
    os.remove(out)
    assert calls[0]["timeout"] == 7
    assert calls[0]["stdin"] is subprocess.DEVNULL


def test_trim_silence_timeout_is_a_fingerprint_error_and_cleans_up(cu, monkeypatch, tmp_path):
    made = []
    real_ntf = cu.tempfile.NamedTemporaryFile

    def tracking_ntf(*args, **kwargs):
        handle = real_ntf(*args, **kwargs)
        made.append(handle.name)
        return handle

    def stalled(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(cu.tempfile, "NamedTemporaryFile", tracking_ntf)
    monkeypatch.setattr(cu.subprocess, "run", stalled)
    song = tmp_path / "song.mp3"
    song.write_bytes(b"x")

    with pytest.raises(cu.FingerprintError, match="timeout"):
        cu.fingerprint_fpcalc(str(song), trim=True)
    assert made and not any(os.path.exists(p) for p in made)


def test_every_fingerprint_subprocess_gets_no_stdin(cu, monkeypatch, tmp_path):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd[0], kwargs.get("stdin"), kwargs.get("timeout")))

        class P:
            returncode = 0
            stdout = '{"fingerprint": "AQAAEw"}' if cmd[0] == "fpcalc" else b""
            stderr = "" if cmd[0] == "fpcalc" else b""

        return P()

    monkeypatch.setattr(cu.subprocess, "run", fake_run)
    song = tmp_path / "song.mp3"
    song.write_bytes(b"x")
    cu.fingerprint_fpcalc(str(song), trim=True)
    assert [s[0] for s in seen] == ["ffmpeg", "ffmpeg", "fpcalc"]
    assert all(stdin is subprocess.DEVNULL and timeout == 7 for _name, stdin, timeout in seen)
