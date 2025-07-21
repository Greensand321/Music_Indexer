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
    received = []
    matches = ts.match_downloads(
        subpar,
        downloads,
        thresholds={"default": 0.1},
        result_callback=received.append,
    )
    assert matches[0]["download"] == "good.flac"
    assert matches[0]["candidates"] == []
    assert received == matches


def test_match_downloads_extension_threshold(monkeypatch):
    ts = load_module(monkeypatch)
    downloads = [
        {
            "artist": "A",
            "title": "T",
            "album": "AL",
            "path": "good.flac",
            "fingerprint": "1 2 3 4 6",
            "fp_prefix": "1 2 3 4 6"[: ts.FP_PREFIX_LEN],
        },
        {
            "artist": "A",
            "title": "T",
            "album": "AL",
            "path": "bad.mp3",
            "fingerprint": "1 2 3 4 6",
            "fp_prefix": "1 2 3 4 6"[: ts.FP_PREFIX_LEN],
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
    thresholds = {"default": 0.5, ".mp3": 0.1, ".flac": 0.5}
    matches = ts.match_downloads(subpar, downloads, thresholds=thresholds)
    assert matches[0]["download"] == "good.flac"


def test_match_downloads_mp3_override(monkeypatch):
    ts = load_module(monkeypatch)
    downloads = [
        {
            "artist": "A",
            "title": "T",
            "album": "AL",
            "path": "good.mp3",
            "fingerprint": "1 2 3 4 6",
            "fp_prefix": "1 2 3 4 6"[: ts.FP_PREFIX_LEN],
        },
        {
            "artist": "A",
            "title": "T",
            "album": "AL",
            "path": "bad.flac",
            "fingerprint": "1 2 3 4 6",
            "fp_prefix": "1 2 3 4 6"[: ts.FP_PREFIX_LEN],
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
    thresholds = {"default": 0.1, ".mp3": 0.3, ".flac": 0.1}
    matches = ts.match_downloads(subpar, downloads, thresholds=thresholds)
    assert matches[0]["download"] == "good.mp3"


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


def test_replace_file_backup(tmp_path, monkeypatch):
    ts = load_module(monkeypatch)

    orig = tmp_path / "track.mp3"
    new = tmp_path / "new.flac"
    orig.write_text("old")
    new.write_text("better")

    ts.replace_file(str(orig), str(new))

    backup = tmp_path / "__backup__" / "track.mp3"
    assert backup.exists()
    assert orig.read_text() == "better"
    assert backup.read_text() == "old"

    restored = ts.restore_backups(str(tmp_path))
    assert str(orig) in restored
    assert orig.read_text() == "old"
    assert not backup.exists()


def test_scan_downloads_parallel_identical(tmp_path, monkeypatch):
    ts = load_module(monkeypatch)

    def read_tags(p):
        base = os.path.splitext(os.path.basename(p))[0]
        return {"artist": base + "A", "title": base + "T", "album": base + "AL"}

    def fp(p, log_callback=None):
        base = os.path.splitext(os.path.basename(p))[0]
        return base + "-fp"

    monkeypatch.setattr(ts, "_read_tags", read_tags)
    monkeypatch.setattr(ts, "_fingerprint", fp)

    for idx in range(3):
        f = tmp_path / f"t{idx}.mp3"
        f.write_text("x")

    serial = ts.scan_downloads(str(tmp_path), max_workers=1)
    parallel = ts.scan_downloads(str(tmp_path), max_workers=2)

    key = lambda item: item["path"]
    serial_sorted = sorted(serial, key=key)
    parallel_sorted = sorted(parallel, key=key)

    assert serial_sorted == parallel_sorted
    for item in parallel_sorted:
        assert set(item.keys()) == {"artist", "title", "album", "path", "fingerprint", "fp_prefix"}


def test_load_subpar_and_scan_identical(tmp_path, monkeypatch):
    ts = load_module(monkeypatch)

    def fp(p, log_callback=None):
        return os.path.basename(p) + "_fp"

    monkeypatch.setattr(ts, "_fingerprint", fp)
    monkeypatch.setattr(
        ts,
        "_read_tags",
        lambda p: {"artist": "A", "title": "T", "album": "AL"},
    )

    def fake_get_fp(path, db_path, compute):
        return compute(path)[1]

    monkeypatch.setattr(ts, "get_fingerprint", fake_get_fp)

    audio = tmp_path / "t0.mp3"
    audio.write_text("x")

    list_file = tmp_path / "subpar.txt"
    list_file.write_text(
        f"A{ts.SUBPAR_DELIM}T{ts.SUBPAR_DELIM}AL{ts.SUBPAR_DELIM}{audio}\n"
    )

    subpar = ts.load_subpar_list(str(list_file))
    downloads = ts.scan_downloads(str(tmp_path))

    assert subpar[0]["fingerprint"] == downloads[0]["fingerprint"]

    matches = ts.match_downloads(subpar, downloads)
    assert matches[0]["download"] == str(audio)
