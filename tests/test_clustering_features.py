"""Tests for selectable clustering features in clustered_playlists.

These cover the selection, sizing and cache-naming rules — the parts that
decide what a feature vector *means*. They run without numpy/sklearn/librosa
installed by stubbing those modules for the duration of this file only.

The stubs are installed through ``MonkeyPatch`` and torn down afterwards, and
the module under test is loaded as a private copy rather than through
``sys.modules["clustered_playlists"]``. Both are deliberate: an earlier draft
registered the fakes globally, and every later test file that imported numpy
then received an empty placeholder — the same cross-file stub bleed the
roadmap flags for the ad-hoc ``mutagen`` stubs elsewhere in the suite.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _importable(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False


class _Scaler:  # minimal stand-in for sklearn scalers
    def fit_transform(self, X):
        return X


def _package_stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    # A package needs __path__ so importlib.util.find_spec("pkg.sub") reports
    # "not found" for missing submodules instead of raising — the module under
    # test probes optional submodules (sklearn.manifold) at import time.
    mod.__path__ = []
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture(scope="module")
def cp():
    """A private copy of clustered_playlists importable without its heavy deps."""
    with pytest.MonkeyPatch.context() as mp:
        for name in ("numpy", "librosa", "essentia", "hdbscan"):
            if not _importable(name):
                mp.setitem(sys.modules, name, _package_stub(name))
        if not _importable("sklearn"):
            sk = _package_stub("sklearn")
            sk.cluster = _package_stub("sklearn.cluster", KMeans=object)
            sk.preprocessing = _package_stub(
                "sklearn.preprocessing",
                StandardScaler=_Scaler,
                MinMaxScaler=_Scaler,
                RobustScaler=_Scaler,
            )
            mp.setitem(sys.modules, "sklearn", sk)
            mp.setitem(sys.modules, "sklearn.cluster", sk.cluster)
            mp.setitem(sys.modules, "sklearn.preprocessing", sk.preprocessing)

        spec = importlib.util.spec_from_file_location(
            "_clustered_playlists_under_test", _REPO / "clustered_playlists.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


# ── Selection normalisation ───────────────────────────────────────────────


def test_default_selection_is_legacy_mfcc_plus_tempo(cp):
    sel = cp.normalize_feature_selection(None)
    assert sel == {
        "mfcc": True, "tempo": True, "chroma": False,
        "spectral": False, "energy": False, "onset_rate": False,
    }


def test_selection_returns_every_feature_key(cp):
    sel = cp.normalize_feature_selection({"chroma": True})
    assert set(sel) == set(cp.FEATURE_ORDER)
    assert sel["chroma"] is True
    assert sel["mfcc"] is False


def test_unknown_feature_keys_are_ignored(cp):
    sel = cp.normalize_feature_selection({"mfcc": True, "danceability": True})
    assert "danceability" not in sel
    assert sel["mfcc"] is True


def test_truthiness_is_coerced_to_bool(cp):
    sel = cp.normalize_feature_selection({"tempo": 1, "mfcc": 0})
    assert sel["tempo"] is True
    assert sel["mfcc"] is False


def test_empty_selection_is_rejected(cp):
    with pytest.raises(ValueError, match="at least one"):
        cp.normalize_feature_selection({"mfcc": False, "tempo": False})


# ── Vector sizing ─────────────────────────────────────────────────────────


def test_feature_order_and_dimensions_agree(cp):
    assert set(cp.FEATURE_ORDER) == set(cp.FEATURE_DIMENSIONS)


def test_legacy_length_is_preserved(cp):
    """Existing caches and Auto-DJ depend on the default staying 27 wide."""
    assert cp.feature_vector_length(None) == cp.FEATURE_VECTOR_LENGTH == 27


def test_all_features_length(cp):
    every = {name: True for name in cp.FEATURE_ORDER}
    assert cp.feature_vector_length(every) == 26 + 1 + 12 + 2 + 2 + 1


def test_single_feature_lengths(cp):
    for name, width in cp.FEATURE_DIMENSIONS.items():
        assert cp.feature_vector_length({name: True}) == width


# ── Selection keys & cache naming ─────────────────────────────────────────


def test_selection_key_follows_canonical_order_not_input_order(cp):
    a = cp.feature_selection_key({"tempo": True, "mfcc": True})
    b = cp.feature_selection_key({"mfcc": True, "tempo": True})
    assert a == b == "mfcc-tempo"


def test_legacy_selection_keeps_original_cache_filename(cp):
    assert cp.feature_cache_name(None, "librosa") == "features.npy"


def test_other_selections_get_distinct_cache_files(cp):
    legacy = cp.feature_cache_name(None, "librosa")
    with_chroma = cp.feature_cache_name({"mfcc": True, "tempo": True, "chroma": True})
    assert with_chroma != legacy
    assert with_chroma == "features_librosa_mfcc-tempo-chroma.npy"


def test_cache_is_keyed_by_engine_too(cp):
    """librosa and essentia vectors are not interchangeable, even at equal width."""
    assert cp.feature_cache_name(None, "essentia") != cp.feature_cache_name(None, "librosa")
    assert cp.feature_cache_name(None, "essentia") == "features_essentia_mfcc-tempo.npy"


# ── Scaler registry ───────────────────────────────────────────────────────


def test_scaler_registry_matches_wizard_choices(cp):
    assert set(cp.SCALERS) == {"standard", "minmax", "robust"}


# ── Assembly (needs real numpy) ───────────────────────────────────────────


def test_assemble_stacks_blocks_in_canonical_order(cp):
    np = pytest.importorskip("numpy")
    if not hasattr(np, "hstack"):
        pytest.skip("numpy is stubbed in this environment")
    sel = cp.normalize_feature_selection({"tempo": True, "mfcc": True, "energy": True})
    blocks = {
        "mfcc": np.arange(26, dtype=np.float32),
        "tempo": np.array([120.0], dtype=np.float32),
        "energy": np.array([0.5, 0.1], dtype=np.float32),
    }
    vec = cp._assemble_feature_vector(blocks, sel)
    assert vec.shape == (29,)
    # mfcc first, then tempo, then energy — regardless of dict insertion order
    assert vec[26] == 120.0
    assert list(vec[27:]) == [0.5, 0.1]
