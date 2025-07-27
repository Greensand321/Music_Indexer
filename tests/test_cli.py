import types
import sys
import os
# stub modules
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
chroma_stub = types.ModuleType('chromaprint_utils')
chroma_stub.fingerprint_fpcalc = lambda p, **kw: 'hash'
sys.modules['chromaprint_utils'] = chroma_stub

from music_indexer_api import main

def test_cli_parsing(tmp_path, monkeypatch):
    called = {}
    def fake_build(root, out, log_callback=None, enable_phase_c=False, flush_cache=False, max_workers=None):
        called['args'] = (root, enable_phase_c, flush_cache, max_workers)
    monkeypatch.setattr('music_indexer_api.build_dry_run_html', fake_build)
    main([str(tmp_path), '--dry-run', '--enable-phase-c', '--flush-cache', '--max-workers', '2'])
    assert called['args'] == (str(tmp_path), True, True, 2)

