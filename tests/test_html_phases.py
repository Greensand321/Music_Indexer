import types

import pytest

from _helpers.module_isolation import isolated_modules, make_mutagen_stub


@pytest.fixture
def indexer_api():
    chroma_stub = types.ModuleType("chromaprint_utils")
    chroma_stub.fingerprint_fpcalc = lambda p, **kw: "hash"
    stubs = {**make_mutagen_stub(), "chromaprint_utils": chroma_stub}
    with isolated_modules(stubs, fresh=["music_indexer_api"]) as mods:
        yield mods["music_indexer_api"]


def test_phase_c_section(tmp_path, indexer_api):
    # The indexer was refocused on routing only (duplicate detection lives in
    # the Duplicate Finder; see docs/features/indexer.md). ``enable_phase_c``
    # is still accepted for backward compatibility but is ignored, so the
    # dry-run report never contains a Phase C (cross-album duplicates) section.
    (tmp_path / "song.mp3").write_text("x")
    html = tmp_path / "out.html"

    indexer_api.build_dry_run_html(str(tmp_path), str(html), enable_phase_c=False)
    without_flag = html.read_text(encoding="utf-8")
    assert "Phase C" not in without_flag
    assert "song" in without_flag

    indexer_api.build_dry_run_html(str(tmp_path), str(html), enable_phase_c=True)
    with_flag = html.read_text(encoding="utf-8")
    assert "Phase C" not in with_flag
    # The flag is a no-op: the report is identical either way.
    assert with_flag == without_flag
