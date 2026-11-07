"""Library Sync's own fingerprint distance (vendored engine, DUP-05/DUP-18 port).

``library_sync`` matches incoming tracks with
``library_sync_indexer_engine/.../near_duplicate_detector.fingerprint_distance``,
not the root module. Before this port that function Hamming-compared the
*compressed* Chromaprint bytes (one flipped bit in the real fingerprint
reshuffles every later byte) and returned 1.0 for ``bytes`` — which is what
pyacoustid hands back — so Library Sync only ever matched byte-identical
fingerprints. It now decodes the prints and compares the raw sub-fingerprints
(bit error rate), exactly like the fixed root module.

The fixtures are real fingerprints (``tests/data/dupes_real_fingerprints.json``).
The threshold tests pin Library Sync's *default* thresholds against them:
every same-recording pair must be a match (COLLISION / EXACT_MATCH) and every
different-recording pair must be NEW — not even LOW_CONFIDENCE.
"""
from __future__ import annotations

import itertools

import pytest

import library_sync
import near_duplicate_detector as root_nd
from library_sync_indexer_engine import config as vconfig
from library_sync_indexer_engine import near_duplicate_detector as vnd
from library_sync_types import MatchStatus, TrackRecord
from _helpers.dupes_chromaprint import compress_chromaprint, fake_raw_print, real_fingerprints

FPS = real_fingerprints()
REFERENCE = "songA.flac"
SAME_RECORDING_CODECS = [
    "songA.mp3_320", "songA.mp3_128", "songA.mp3_64",
    "songA.m4a_96", "songA.ogg_q2", "songA.opus_48", "songA.flac_untrimmed",
]
SHIFTED = ["songA_cut.mp3_128"]                 # starts 0.3 s later
DIFFERENT = ["songB.flac", "songC.mp3_128"]


def _ext(key: str) -> str:
    codec = key.split(".", 1)[1].split("_", 1)[0]
    return "." + codec


def _song(key: str) -> str:
    return key.split(".", 1)[0].split("_", 1)[0]


def _same_pairs():
    keys = [k for k in FPS if _song(k) == "songA" and k != "songA_lead_untrimmed.flac"]
    return list(itertools.combinations(keys, 2))


def _different_pairs():
    return [(a, b) for a, b in itertools.combinations(FPS, 2) if _song(a) != _song(b)]


# ── the metric ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", SAME_RECORDING_CODECS)
def test_same_recording_any_codec_is_near_zero(variant):
    assert vnd.fingerprint_distance(FPS[REFERENCE], FPS[variant]) <= 0.01


@pytest.mark.parametrize("variant", SHIFTED)
def test_shifted_start_stays_well_below_unrelated(variant):
    assert 0.02 <= vnd.fingerprint_distance(FPS[REFERENCE], FPS[variant]) <= 0.08


@pytest.mark.parametrize("a,b", _different_pairs())
def test_different_recordings_are_far_apart(a, b):
    assert vnd.fingerprint_distance(FPS[a], FPS[b]) >= 0.4


def test_bytes_fingerprints_compare_like_str():
    """pyacoustid returns bytes; the old code gave 1.0 for every bytes pair."""
    a, b = FPS[REFERENCE], FPS["songA.mp3_128"]
    expected = vnd.fingerprint_distance(a, b)
    assert expected < 0.01
    assert vnd.fingerprint_distance(a.encode(), b.encode()) == expected
    assert vnd.fingerprint_distance(a.encode(), b) == expected
    assert vnd.fingerprint_distance(bytearray(a.encode()), b) == expected


def test_empty_fingerprints_never_match():
    assert vnd.is_empty_chromaprint("AQAAAA")
    assert vnd.fingerprint_distance("AQAAAA", "AQAAAA") == 1.0
    assert vnd.fingerprint_distance(b"AQAAAA", b"AQAAAA") == 1.0
    assert vnd.fingerprint_distance("AQAAAA", FPS[REFERENCE]) == 1.0
    assert vnd.fingerprint_distance(None, FPS[REFERENCE]) == 1.0
    assert vnd.fingerprint_distance("", "") == 1.0


def test_matches_the_root_module_exactly():
    for a, b in itertools.combinations(FPS, 2):
        assert vnd.fingerprint_distance(FPS[a], FPS[b]) == root_nd.fingerprint_distance(FPS[a], FPS[b])


def test_plain_integer_lists_keep_their_semantics():
    """Synthetic "1,2,3" fixtures across the Library Sync tests still work."""
    assert vnd.fingerprint_distance("1,2,3,4", "1,2,3,4") == 0.0
    assert vnd.fingerprint_distance("1 2 3 4", "1 2 9 4") == 0.25


def test_decoder_round_trips_the_compressor():
    raw = fake_raw_print("vendored", 300)
    raw[0] |= 1 << 31
    assert vnd.decode_chromaprint(compress_chromaprint(raw, algorithm=2)) == (2, raw)


# ── coarse gate ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", SAME_RECORDING_CODECS + SHIFTED)
def test_coarse_gate_passes_same_recording_across_codecs(variant):
    """Banding the compressed bytes split a FLAC from its own lossy copies."""
    keys_a = vnd._coarse_fingerprint_keys(FPS[REFERENCE])
    keys_b = vnd._coarse_fingerprint_keys(FPS[variant].encode())
    assert vnd._coarse_gate(keys_a, keys_b)


def test_find_near_duplicates_groups_a_flac_and_its_mp3():
    infos = {
        "/lib/a.flac": {"fp": FPS[REFERENCE], "title": "Song", "primary": "Artist", "album": "Al", "ext": ".flac"},
        "/lib/a.mp3": {"fp": FPS["songA.mp3_64"].encode(), "title": "Song", "primary": "Artist", "album": "Al",
                       "ext": ".mp3"},
    }
    result = vnd.find_near_duplicates(infos, {".flac": 0, ".mp3": 1}, 0.1, max_workers=1)
    assert len(result.review_groups) == 1
    assert result.review_groups[0].winner == "/lib/a.flac"


# ── Library Sync's thresholds against the new metric ─────────────────────────

def _status(a: str, b: str, thresholds=None) -> tuple[MatchStatus, float, float]:
    thresholds = thresholds or vconfig.DEFAULT_FP_THRESHOLDS
    dist = library_sync.fingerprint_distance(FPS[a], FPS[b])
    thr = library_sync._select_pair_threshold(
        _ext(a), _ext(b), thresholds, vconfig.MIXED_CODEC_THRESHOLD_BOOST
    )
    margin = max(library_sync.NEAR_MISS_MARGIN_FLOOR, thr * library_sync.NEAR_MISS_MARGIN_RATIO)
    if dist == 0:
        return MatchStatus.EXACT_MATCH, dist, thr
    if dist <= thr:
        return MatchStatus.COLLISION, dist, thr
    if dist <= thr + margin:
        return MatchStatus.LOW_CONFIDENCE, dist, thr + margin
    return MatchStatus.NEW, dist, thr + margin


def test_library_sync_binds_the_vendored_metric():
    assert library_sync.fingerprint_distance is vnd.fingerprint_distance


@pytest.mark.parametrize("a,b", _same_pairs())
def test_default_thresholds_match_the_same_recording(a, b):
    status, dist, thr = _status(a, b)
    assert status in (MatchStatus.EXACT_MATCH, MatchStatus.COLLISION), (a, b, dist, thr)


def test_default_thresholds_match_a_leading_silence_copy():
    """0.6 s of extra leading silence (trim disabled) is still the same track."""
    status, dist, thr = _status(REFERENCE, "songA_lead_untrimmed.flac")
    assert status == MatchStatus.COLLISION, (dist, thr)


@pytest.mark.parametrize("a,b", _different_pairs())
def test_default_thresholds_keep_different_recordings_new(a, b):
    """With the old 0.3/0.35 defaults two different MP3s (0.43 apart) landed
    in the 0.35 + 25 % near-miss band and were sent to review."""
    status, dist, limit = _status(a, b)
    assert status == MatchStatus.NEW, (a, b, dist, limit)


def test_default_threshold_values():
    assert vconfig.DEFAULT_FP_THRESHOLDS["default"] == pytest.approx(0.2)
    for ext in (".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus"):
        assert vconfig.DEFAULT_FP_THRESHOLDS.get(ext, vconfig.DEFAULT_FP_THRESHOLDS["default"]) == pytest.approx(0.2)


# ── end to end through library_sync's matcher ────────────────────────────────

def _record(path: str, ext: str, fp) -> TrackRecord:
    return TrackRecord(
        track_id=path, path=path, normalized_path=path, ext=ext, bitrate=0,
        size=1, mtime=0.0, fingerprint=fp, tags={},
    )


@pytest.mark.parametrize("incoming_key,library_key,expected", [
    ("songA.flac", "songA.mp3_128", MatchStatus.COLLISION),
    ("songA.opus_48", "songA.flac", MatchStatus.COLLISION),
    ("songA_cut.mp3_128", "songA.m4a_96", MatchStatus.COLLISION),
    ("songB.flac", "songA.mp3_128", MatchStatus.NEW),
    ("songC.mp3_128", "songA.mp3_64", MatchStatus.NEW),
])
def test_match_tracks_end_to_end(incoming_key, library_key, expected):
    inc = _record("/inc/" + incoming_key, _ext(incoming_key), FPS[incoming_key])
    lib = _record("/lib/" + library_key, _ext(library_key), FPS[library_key])
    [result] = library_sync._match_tracks(
        [inc], [lib], vconfig.DEFAULT_FP_THRESHOLDS, {".flac": 3, ".mp3": 1},
        mixed_codec_threshold_boost=vconfig.MIXED_CODEC_THRESHOLD_BOOST,
    )
    assert result.status == expected, (result.distance, result.threshold_used)
    if expected == MatchStatus.COLLISION:
        assert result.existing is lib
        assert result.distance < 0.1
