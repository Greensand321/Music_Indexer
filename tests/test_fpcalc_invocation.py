import pytest

from _helpers.module_isolation import isolated_modules


@pytest.fixture
def cu():
    # Always exercise the real module, even if something stubbed it.
    with isolated_modules(fresh=["chromaprint_utils"]) as mods:
        yield mods["chromaprint_utils"]


def test_fpcalc_invocation_and_prefix(cu, monkeypatch, tmp_path):
    monkeypatch.setattr(cu, "ensure_tool", lambda name: None)
    monkeypatch.setattr(cu, "load_config", lambda: {})
    monkeypatch.setattr(cu, "strip_ext_prefix", lambda p: p[4:] if p.startswith("\\\\?\\") else p)

    # fpcalc runs on the intermediate WAV that ffmpeg writes, so that is the
    # path that must reach fpcalc without a Windows long-path prefix.
    long_tmp = "\\\\?\\C:\\temp\\fp_intermediate.wav"

    class FakeTmp:
        name = long_tmp

        def close(self):
            pass

    monkeypatch.setattr(cu.tempfile, "NamedTemporaryFile", lambda **kw: FakeTmp())

    commands = []

    def fake_run(cmd, stdout=None, stderr=None, text=None, **kw):
        commands.append(cmd)

        class P:
            returncode = 0
            stdout = '{"fingerprint": "1,2,3"}'
            stderr = ''

        return P()

    monkeypatch.setattr(cu.subprocess, 'run', fake_run)

    source = tmp_path / "song.flac"
    source.write_bytes(b"audio")
    fp = cu.fingerprint_fpcalc(str(source), trim=False)

    assert fp == '1 2 3'
    ffmpeg_cmd, fpcalc_cmd = commands
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert ffmpeg_cmd[-1] == long_tmp
    assert fpcalc_cmd[0] == "fpcalc"
    assert fpcalc_cmd[1] == '-json'
    # result path should not contain Windows long-path prefix
    assert not fpcalc_cmd[2].startswith('\\\\?\\')
    assert fpcalc_cmd[2] == "C:\\temp\\fp_intermediate.wav"


def test_fpcalc_missing_file_raises_before_running_tools(cu, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cu.subprocess, "run", lambda *a, **k: calls.append(a))

    with pytest.raises(cu.FingerprintError, match="file missing"):
        cu.fingerprint_fpcalc(str(tmp_path / "nope.flac"), trim=False)
    assert calls == []
