from music_indexer_api import sanitize
import importlib.util
from pathlib import Path


def _load_engine_sanitize():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "library_sync_indexer_engine"
        / "indexer_engine"
        / "music_indexer_api.py"
    )
    spec = importlib.util.spec_from_file_location(
        "library_sync_indexer_engine.indexer_engine.music_indexer_api",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.sanitize


def test_sanitize_handles_non_string_inputs():
    assert sanitize(None) == "Unknown"
    assert sanitize(2024) == "2024"
    assert sanitize(3.14) == "3.14"
    assert sanitize(["AC/DC", None, "Live"]) == "ACDC, Live"


def test_sanitize_preserves_entities_without_double_escaping():
    assert sanitize("A &amp; B") == "A &amp; B"
    assert sanitize("Rock & Roll") == "Rock & Roll"


def test_sanitize_strips_invalid_characters_and_whitespace():
    assert sanitize("  <> ") == "Unknown"
    assert sanitize("A/B<C>") == "ABC"


def test_engine_sanitize_matches_api_sanitize():
    engine_sanitize = _load_engine_sanitize()
    samples = [None, 2024, 3.14, ["AC/DC", None, "Live"], "A/B<C>", "Rock & Roll"]
    assert [engine_sanitize(sample) for sample in samples] == [
        sanitize(sample) for sample in samples
    ]
