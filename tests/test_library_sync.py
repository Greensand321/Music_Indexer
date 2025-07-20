import sys
import types
from library_sync import compare_folders


def setup_modules(fps):
    mutagen_stub = types.ModuleType('mutagen')
    class Info:
        def __init__(self, bitrate=None, sample_rate=None):
            self.bitrate = bitrate
            self.sample_rate = sample_rate
    class DummyAudio:
        def __init__(self, path):
            self.tags = {
                'artist': ['a'],
                'title': ['b'],
                'album': ['c'],
            }
            if path.endswith('.mp3'):
                self.info = Info(bitrate=128000)
            else:
                self.info = Info(sample_rate=44100)
    mutagen_stub.File = lambda p, easy=True: DummyAudio(p)
    id3_stub = types.ModuleType('id3')
    id3_stub.ID3NoHeaderError = Exception
    mutagen_stub.id3 = id3_stub
    sys.modules['mutagen'] = mutagen_stub
    sys.modules['mutagen.id3'] = id3_stub
    import library_sync
    library_sync.MutagenFile = mutagen_stub.File

    ac_stub = types.ModuleType('acoustid')
    def fp_func(path):
        return 0, fps[path]
    ac_stub.fingerprint_file = fp_func
    sys.modules['acoustid'] = ac_stub


def test_compare_folders(tmp_path, monkeypatch):
    lib = tmp_path / 'lib'
    inc = tmp_path / 'inc'
    lib.mkdir(); inc.mkdir()
    f1 = lib / 'song1.mp3'
    f2 = lib / 'song2.flac'
    f1.write_text('a'); f2.write_text('b')
    i1 = inc / 'song1.flac'
    i2 = inc / 'song3.mp3'
    i1.write_text('c'); i2.write_text('d')

    fps = {
        str(f1): 'fp1',
        str(f2): 'fp2',
        str(i1): 'fp1',
        str(i2): 'fp3',
    }
    setup_modules(fps)

    new, existing, improvements = compare_folders(str(lib), str(inc), str(tmp_path / 'db.sqlite'))
    assert str(i2) in new
    assert existing == {}
    assert improvements == {str(i1): str(f1)}
