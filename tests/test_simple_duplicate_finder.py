import os
import sys
import types
import importlib

import simple_duplicate_finder as sdf
from fingerprint_cache import flush_cache, _ensure_db


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


def _naive_duplicate_pairs(sdf_mod, ordered_paths, fp_map, prefix_len=None, threshold=0.03):
    groups = []
    prefix_map = {}
    use_prefix = prefix_len is not None and prefix_len > 0
    for path in ordered_paths:
        fp = fp_map[os.path.basename(path)]
        prefix = fp[:prefix_len] if use_prefix else ""
        cand_groups = []
        if use_prefix:
            for key, groups_for_key in prefix_map.items():
                if sdf_mod.prefix_distance(prefix, key) <= sdf_mod.PREFIX_THRESHOLD:
                    for g in groups_for_key:
                        cand_groups.append((g, key))
        else:
            for g in prefix_map.get(prefix, []):
                cand_groups.append((g, prefix))
        placed = False
        for g, _key in cand_groups:
            dist = sdf_mod.fingerprint_distance(fp, g["fp"])
            if dist <= threshold:
                g["paths"].append(path)
                placed = True
                break
        if not placed:
            g = {"fp": fp, "paths": [path]}
            prefix_map.setdefault(prefix, []).append(g)
            groups.append(g)
    pairs = set()
    for g in groups:
        paths = g["paths"]
        if len(paths) <= 1:
            continue
        scored = sorted(
            paths, key=lambda p: sdf_mod._keep_score(p, sdf_mod.EXT_PRIORITY), reverse=True
        )
        keep = scored[0]
        for dup in scored[1:]:
            pairs.add(tuple(sorted((keep, dup))))
    return pairs


def test_duplicate_detection(tmp_path, monkeypatch):
    sdf_mod = load_module(monkeypatch)
    (tmp_path / 'a.mp3').write_text('x')
    (tmp_path / 'b.flac').write_text('x')
    (tmp_path / 'c.mp3').write_text('x')
    db = tmp_path / 'fp.db'
    _ensure_db(str(db))
    dups, missing = sdf_mod.find_duplicates(str(tmp_path), db_path=str(db), max_workers=1)
    pair = (str(tmp_path / 'b.flac'), str(tmp_path / 'a.mp3'))
    assert pair in dups or (pair[1], pair[0]) in dups
    assert missing == 0
    flush_cache(str(db))


def test_cross_prefix_detection(tmp_path, monkeypatch):
    sdf_mod = load_module(monkeypatch)
    (tmp_path / 'd.mp3').write_text('x')
    (tmp_path / 'e.mp3').write_text('x')
    db = tmp_path / 'fp.db'
    _ensure_db(str(db))
    # Fuzzy prefix grouping should still compare them
    pair = (str(tmp_path / 'd.mp3'), str(tmp_path / 'e.mp3'))
    dups, _ = sdf_mod.find_duplicates(
        str(tmp_path), db_path=str(db), threshold=0.5, max_workers=1
    )
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
    _ensure_db(str(db))
    pair = (str(tmp_path / 'f2.flac'), str(tmp_path / 'f1.mp3'))
    dups, _ = sdf_mod.find_duplicates(str(tmp_path), db_path=str(db), max_workers=1)
    assert pair in dups or (pair[1], pair[0]) in dups
    flush_cache(str(db))


def test_coarse_gating_regression(tmp_path, monkeypatch):
    fp_map = {
        'dup1.mp3': '1 2 3 4 5 6 7 8 9 10',
        'dup2.flac': '1 2 3 4 5 6 7 8 9 10',
        'near.mp3': '1 2 3 4 5 6 7 8 9 11',
        'far.mp3': '9 9 9 9 9 9 9 9 9 9',
    }
    sdf_mod = load_module(monkeypatch, fp_map=fp_map)
    for name in fp_map:
        (tmp_path / name).write_text('x')
    db = tmp_path / 'fp.db'
    _ensure_db(str(db))
    ordered_paths = [str(tmp_path / name) for name in sorted(fp_map)]
    monkeypatch.setattr(sdf_mod, '_walk_audio_files', lambda *_args, **_kw: iter(ordered_paths))

    expected = _naive_duplicate_pairs(
        sdf_mod, ordered_paths, fp_map, prefix_len=sdf_mod.FP_PREFIX_LEN
    )
    dups, _ = sdf_mod.find_duplicates(
        str(tmp_path), db_path=str(db), max_workers=1, threshold=0.2
    )
    actual = {tuple(sorted(pair)) for pair in dups}
    assert expected.issubset(actual)
    flush_cache(str(db))
