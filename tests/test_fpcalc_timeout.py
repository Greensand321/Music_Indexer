import subprocess

import pytest

from _helpers.module_isolation import isolated_modules


@pytest.fixture
def cu():
    # Always exercise the real module, even if something stubbed it.
    with isolated_modules(fresh=["chromaprint_utils"]) as mods:
        yield mods["chromaprint_utils"]


def test_fpcalc_timeout(cu, monkeypatch, tmp_path):
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
