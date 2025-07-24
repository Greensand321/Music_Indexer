import os
import sys
import types
import importlib

import simple_duplicate_finder as sdf
from fingerprint_cache import flush_cache


def load_module(monkeypatch):
    acoustid_stub = types.ModuleType('acoustid')
    fp_map = {
        'a.mp3': '1 2',
        'b.flac': '1 2',
        'c.mp3': '9 9',
    }
    acoustid_stub.fingerprint_file = lambda p: (0, fp_map[os.path.basename(p)])
    monkeypatch.setitem(sys.modules, 'acoustid', acoustid_stub)
    importlib.reload(sdf)
    return sdf


def test_duplicate_detection(tmp_path, monkeypatch):
    sdf_mod = load_module(monkeypatch)
    (tmp_path / 'a.mp3').write_text('x')
    (tmp_path / 'b.flac').write_text('x')
    (tmp_path / 'c.mp3').write_text('x')
    db = tmp_path / 'fp.db'
    dups = sdf_mod.find_duplicates(str(tmp_path), db_path=str(db))
    pair = (str(tmp_path / 'b.flac'), str(tmp_path / 'a.mp3'))
    assert pair in dups or (pair[1], pair[0]) in dups
    flush_cache(str(db))

