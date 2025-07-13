import os
import sys
import types
import tempfile

mutagen_stub = types.ModuleType('mutagen')
class DummyAudio:
    def __init__(self):
        self.tags = None
        self.pictures = []
def File(*a, **k):
    return DummyAudio()
mutagen_stub.File = File
id3_stub = types.ModuleType('id3')
id3_stub.ID3NoHeaderError = Exception
mutagen_stub.id3 = id3_stub
sys.modules['mutagen'] = mutagen_stub
sys.modules['mutagen.id3'] = id3_stub
acoustid_stub = types.ModuleType('acoustid')
acoustid_stub.fingerprint_file = lambda p: (0, 'hash')
sys.modules['acoustid'] = acoustid_stub

from music_indexer_api import build_dry_run_html

def test_phase_c_section(tmp_path):
    (tmp_path / 'song.mp3').write_text('x')
    html = tmp_path / 'out.html'
    build_dry_run_html(str(tmp_path), str(html), enable_phase_c=False)
    text = html.read_text()
    assert 'Phase C' not in text
    build_dry_run_html(str(tmp_path), str(html), enable_phase_c=True)
    text = html.read_text()
    assert 'Phase C' in text

