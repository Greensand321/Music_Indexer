import os
import sys
import types
import importlib


def load_module(monkeypatch):
    mutagen_stub = types.ModuleType('mutagen')
    def File(*a, **k):
        return None
    mutagen_stub.File = File
    id3_stub = types.ModuleType('id3')
    class DummyID3:
        pass
    id3_stub.ID3 = DummyID3
    id3_stub.ID3NoHeaderError = Exception
    mutagen_stub.id3 = id3_stub
    mp3_stub = types.ModuleType('mp3')
    class MP3:
        def __init__(self, *a, **k):
            self.tags = None
    mp3_stub.MP3 = MP3
    mutagen_stub.mp3 = mp3_stub
    monkeypatch.setitem(sys.modules, 'mutagen', mutagen_stub)
    monkeypatch.setitem(sys.modules, 'mutagen.id3', id3_stub)
    monkeypatch.setitem(sys.modules, 'mutagen.mp3', mp3_stub)

    acoustid_stub = types.ModuleType('acoustid')
    acoustid_stub.fingerprint_file = lambda p: (0, '')
    monkeypatch.setitem(sys.modules, 'acoustid', acoustid_stub)

    import tidal_sync
    importlib.reload(tidal_sync)
    return tidal_sync


def fake_read_tags(_):
    return {"artist": "A", "title": "T", "album": "AL"}


def fake_fingerprint(_path, log_callback=None):
    return "1 2 3 4 5 6 7 8"


def test_scan_downloads_prefix(tmp_path, monkeypatch):
    ts = load_module(monkeypatch)
    monkeypatch.setattr(ts, "_read_tags", fake_read_tags)
    monkeypatch.setattr(ts, "_fingerprint", fake_fingerprint)
    f = tmp_path / "song.mp3"
    f.write_text("x")
    items = ts.scan_downloads(str(tmp_path))
    assert items[0]["fp_prefix"] == fake_fingerprint("")[: ts.FP_PREFIX_LEN]


def test_match_downloads_prefix_lookup(monkeypatch):
    ts = load_module(monkeypatch)
    downloads = [
        {
            "artist": "A",
            "title": "T",
            "album": "AL",
            "path": "good.flac",
            "fingerprint": "1 2 3 4 5",
            "fp_prefix": "1 2 3 4 5"[: ts.FP_PREFIX_LEN],
        },
        {
            "artist": "A",
            "title": "T",
            "album": "AL",
            "path": "bad.flac",
            "fingerprint": "9 9 9 9 9",
            "fp_prefix": "9 9 9 9 9"[: ts.FP_PREFIX_LEN],
        },
    ]
    subpar = [
        {
            "artist": "A",
            "title": "T",
            "album": "AL",
            "path": "orig.mp3",
            "fingerprint": "1 2 3 4 5",
            "fp_prefix": "1 2 3 4 5"[: ts.FP_PREFIX_LEN],
        }
    ]
    matches = ts.match_downloads(subpar, downloads, threshold=0.1)
    assert matches[0]["download"] == "good.flac"
    assert matches[0]["candidates"] == []


def test_load_subpar_list_fingerprint(tmp_path, monkeypatch):
    ts = load_module(monkeypatch)

    called = {}

    def fake_get_fp(p, db, compute):
        called['p'] = p
        return "fp-x"

    monkeypatch.setattr(ts, "get_fingerprint", fake_get_fp)

    audio = tmp_path / "a.mp3"
    audio.write_text("x")
    list_file = tmp_path / "subpar.txt"
    list_file.write_text(
        f"A{ts.SUBPAR_DELIM}T{ts.SUBPAR_DELIM}AL{ts.SUBPAR_DELIM}{audio}\n"
    )

    items = ts.load_subpar_list(str(list_file))
    assert items[0]["fingerprint"] == "fp-x"
    assert items[0]["fp_prefix"] == "fp-x"[: ts.FP_PREFIX_LEN]
    assert called["p"] == str(audio)
