import importlib
import sys

# reload real module in case previous tests replaced it with a stub
sys.modules.pop('chromaprint_utils', None)
cu = importlib.import_module('chromaprint_utils')


def test_fpcalc_invocation_and_prefix(monkeypatch):
    monkeypatch.setattr(cu, "ensure_tool", lambda name: None)
    monkeypatch.setattr(cu, "strip_long_path_prefix", lambda p: p[4:] if p.startswith("\\\\?\\") else p)

    captured = {}

    def fake_run(cmd, stdout=None, stderr=None, text=None):
        captured['cmd'] = cmd
        class P:
            returncode = 0
            stdout = '{"fingerprint": "1,2,3"}'
            stderr = ''
        return P()

    monkeypatch.setattr(cu.subprocess, 'run', fake_run)

    path = "\\\\?\\C:\\music\\song.flac"
    fp = cu.fingerprint_fpcalc(path, trim=False)

    assert fp == '1 2 3'
    assert captured['cmd'][1] == '-json'
    assert captured['cmd'][2] == r"C:\music\song.flac"
