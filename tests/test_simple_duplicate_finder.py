import os
import sys
import types
import importlib

import simple_duplicate_finder as sdf
from fingerprint_cache import flush_cache


def load_module(monkeypatch, fp_map=None):
    chroma_stub = types.ModuleType('chromaprint_utils')
    if fp_map is None:
        fp_map = {
            'a.mp3': '1 2',
            'b.flac': '1 2',
            'c.mp3': '9 9',
            'd.mp3': '1 1',
            'e.mp3': '2 1',
        }
    chroma_stub.fingerprint_fpcalc = lambda p, **kw: fp_map[os.path.basename(p)]
    monkeypatch.setitem(sys.modules, 'chromaprint_utils', chroma_stub)
    importlib.reload(sdf)
    return sdf


def test_duplicate_detection(tmp_path, monkeypatch):
    sdf_mod = load_module(monkeypatch)
    (tmp_path / 'a.mp3').write_text('x')
    (tmp_path / 'b.flac').write_text('x')
    (tmp_path / 'c.mp3').write_text('x')
    db = tmp_path / 'fp.db'
    dups, missing = sdf_mod.find_duplicates(str(tmp_path), db_path=str(db))
    pair = (str(tmp_path / 'b.flac'), str(tmp_path / 'a.mp3'))
    assert pair in dups or (pair[1], pair[0]) in dups
    assert missing == 0
    flush_cache(str(db))


def test_cross_prefix_detection(tmp_path, monkeypatch):
    sdf_mod = load_module(monkeypatch)
    (tmp_path / 'd.mp3').write_text('x')
    (tmp_path / 'e.mp3').write_text('x')
    db = tmp_path / 'fp.db'
    # Fuzzy prefix grouping should still compare them
    pair = (str(tmp_path / 'd.mp3'), str(tmp_path / 'e.mp3'))
    dups, _ = sdf_mod.find_duplicates(str(tmp_path), db_path=str(db), threshold=0.5)
    assert pair in dups or (pair[1], pair[0]) in dups
    flush_cache(str(db))


def test_fuzzy_prefix_grouping(tmp_path, monkeypatch):
    fp_map = {
        'f1.mp3': 'AAAAAAAAAAAAAAAAAAAA',
        'f2.flac': 'BAAAAAAAAAAAAAAAAAAA',
    }
    sdf_mod = load_module(monkeypatch, fp_map=fp_map)
    (tmp_path / 'f1.mp3').write_text('x')
    (tmp_path / 'f2.flac').write_text('x')
    db = tmp_path / 'fp.db'
    pair = (str(tmp_path / 'f2.flac'), str(tmp_path / 'f1.mp3'))
    dups, _ = sdf_mod.find_duplicates(str(tmp_path), db_path=str(db))
    assert pair in dups or (pair[1], pair[0]) in dups
    flush_cache(str(db))

