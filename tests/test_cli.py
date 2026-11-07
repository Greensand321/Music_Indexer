import types

import pytest

from _helpers.module_isolation import isolated_modules, make_mutagen_stub


@pytest.fixture
def indexer_api():
    chroma_stub = types.ModuleType('chromaprint_utils')
    chroma_stub.fingerprint_fpcalc = lambda p, **kw: 'hash'
    stubs = {**make_mutagen_stub(), 'chromaprint_utils': chroma_stub}
    with isolated_modules(stubs, fresh=['music_indexer_api']) as mods:
        yield mods['music_indexer_api']


def test_cli_parsing(tmp_path, monkeypatch, indexer_api):
    called = {}
    def fake_build(root, out, log_callback=None, enable_phase_c=False, flush_cache=False, max_workers=None):
        called['args'] = (root, enable_phase_c, flush_cache, max_workers)
    monkeypatch.setattr(indexer_api, 'build_dry_run_html', fake_build)
    indexer_api.main([str(tmp_path), '--dry-run', '--enable-phase-c', '--flush-cache', '--max-workers', '2'])
    assert called['args'] == (str(tmp_path), True, True, 2)

