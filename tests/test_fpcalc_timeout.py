import importlib
import subprocess
import sys


sys.modules.pop("chromaprint_utils", None)
cu = importlib.import_module("chromaprint_utils")


def test_fpcalc_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(cu, "ensure_tool", lambda name: None)
    monkeypatch.setattr(cu, "load_config", lambda: {"fingerprint_subprocess_timeout_sec": 1})

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    monkeypatch.setattr(cu.subprocess, "run", fake_run)

    audio_path = tmp_path / "song.mp3"
    audio_path.write_bytes(b"dummy")

    try:
        cu.fingerprint_fpcalc(str(audio_path), trim=False)
    except cu.FingerprintError as exc:
        assert str(exc) == "Fingerprint timeout"
    else:
        raise AssertionError("Expected FingerprintError")
