"""Library Sync's candidate shortlist (area: syncmatch).

Library Sync must recognise a track the library already has — in any codec,
wherever it sits in the library — before the fingerprint distance can say
how close it is. The old shortlist keyed on the *whole compressed*
fingerprint string (so only byte-identical prints were ever found) and
otherwise compared the first 50 library tracks *with the incoming track's
extension*: a FLAC was never compared with its own MP3 copy once the library
held a few other FLACs, and a same-codec copy sorted after 50 others was
never compared at all. Both came out NEW (copied in again).

Fixtures: genuine Chromaprint fingerprints of three synthetic clips
(``tests/data/dupes_real_fingerprints.json``, the same ones
``test_vendored_fingerprint_distance.py`` pins the metric with) plus
synthetic prints from ``_helpers.dupes_chromaprint`` for the rest of the
library (random 32-bit sub-fingerprints: a different recording each).
"""
from __future__ import annotations

import itertools
import threading

import pytest

import library_sync
from _helpers.dupes_chromaprint import compress_chromaprint, fake_fingerprint, fake_raw_print, real_fingerprints
from library_sync_indexer_engine import config as vconfig
from library_sync_types import MatchStatus, TrackRecord, _fingerprint_signature

FPS = real_fingerprints()
DEFAULTS = vconfig.DEFAULT_FP_THRESHOLDS
PRIORITY = {".flac": 3, ".wav": 2, ".mp3": 1}
MATCHED = (MatchStatus.EXACT_MATCH, MatchStatus.COLLISION)


def _rec(name: str, ext: str, fp, length: float | None = None) -> TrackRecord:
    return TrackRecord(
        track_id=name, path=f"/music/{name}{ext}", normalized_path=f"/music/{name}{ext}", ext=ext,
        bitrate=0, size=1, mtime=0.0, fingerprint=fp, tags={}, duration=30, audio_length=length,
    )


def _others(count: int, ext: str, prefix: str, length: float | None = None) -> list[TrackRecord]:
    """``count`` different recordings, sorted before anything named later."""
    return [_rec(f"{prefix}{i:03d}", ext, fake_fingerprint(f"{prefix}-{i}"), length) for i in range(count)]


def _match(incoming, library, thresholds=None, **kw):
    return library_sync._match_tracks(
        list(incoming), list(library), thresholds or DEFAULTS, PRIORITY,
        mixed_codec_threshold_boost=vconfig.MIXED_CODEC_THRESHOLD_BOOST, **kw,
    )


@pytest.fixture
def distance_calls(monkeypatch):
    """Count the exact fingerprint comparisons the matcher makes."""
    calls = []
    real = library_sync.fingerprint_distance

    def counting(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(library_sync, "fingerprint_distance", counting)
    return calls


# ── the two reported failures ────────────────────────────────────────────────

def test_flac_finds_its_mp3_copy_among_other_flacs():
    """With 10 other FLACs in the library the FLAC's MP3 copy was never compared."""
    library = _others(10, ".flac", "a_flac") + [_rec("m_songA", ".mp3", FPS["songA.mp3_128"])] \
        + _others(5, ".mp3", "z_mp3")
    [res] = _match([_rec("in_songA", ".flac", FPS["songA.flac"])], library)
    assert res.status == MatchStatus.COLLISION, (res.status, res.distance)
    assert res.existing.track_id == "m_songA"
    assert res.distance < 0.01
    assert res.quality_label == "Potential Upgrade"


def test_same_codec_copy_sorted_after_sixty_tracks_is_found():
    """A re-encode (different print, same codec) after 60 other MP3s came out NEW."""
    library = _others(60, ".mp3", "a_mp3") + [_rec("zz_songA", ".mp3", FPS["songA.mp3_64"])]
    [res] = _match([_rec("in_songA", ".mp3", FPS["songA.mp3_320"])], library)
    assert res.status in MATCHED, (res.status, res.distance)
    assert res.existing.track_id == "zz_songA"


@pytest.mark.parametrize("incoming_key,library_key", [
    ("songA.opus_48", "songA.flac"),
    ("songA.m4a_96", "songA.ogg_q2"),
    ("songA_cut.mp3_128", "songA.m4a_96"),          # starts 0.3 s later (0.075)
    ("songA_lead_untrimmed.flac", "songA.mp3_128"),  # 0.6 s of leading silence (0.136)
])
def test_same_recording_is_found_anywhere_in_a_mixed_library(incoming_key, library_key):
    ext = "." + library_key.split(".", 1)[1].split("_", 1)[0]
    library = (
        _others(12, ".flac", "a") + _others(55, ".mp3", "b") + _others(6, ".m4a", "c")
        + [_rec("y_target", ext, FPS[library_key])] + _others(4, ".ogg", "z")
    )
    inc_ext = "." + incoming_key.split(".", 1)[1].split("_", 1)[0]
    [res] = _match([_rec("in", inc_ext, FPS[incoming_key])], library)
    assert res.status == MatchStatus.COLLISION, (res.status, res.distance)
    assert res.existing.track_id == "y_target"


def test_every_copy_in_the_library_is_a_candidate():
    library = [
        _rec("a_flac", ".flac", FPS["songA.flac"]),
        *_others(30, ".mp3", "m"),
        _rec("x_m4a", ".m4a", FPS["songA.m4a_96"]),
        _rec("y_ogg", ".ogg", FPS["songA.ogg_q2"]),
    ]
    [res] = _match([_rec("in", ".mp3", FPS["songA.mp3_128"])], library)
    compared = {c.track.track_id for c in res.candidates}
    assert {"a_flac", "x_m4a", "y_ogg"} <= compared
    assert res.existing.track_id == "a_flac"   # the closest copy
    assert len(res.candidates) < 10            # not the whole library


# ── unrelated tracks stay new (the low-confidence band) ──────────────────────

def _key_ext(key: str) -> str:
    return "." + key.split(".", 1)[1].split("_", 1)[0]


def _different_pairs():
    song = lambda k: k.split(".", 1)[0].split("_", 1)[0]  # noqa: E731
    return [(a, b) for a, b in itertools.permutations(FPS, 2) if song(a) != song(b)]


@pytest.mark.parametrize("a,b", _different_pairs())
def test_default_thresholds_keep_different_recordings_new(a, b):
    """Every different-recording pair (0.43+ apart), mixed codecs included,
    is NEW through the matcher itself — not even LOW_CONFIDENCE."""
    [res] = _match([_rec("in", _key_ext(a), FPS[a])], [_rec("lib", _key_ext(b), FPS[b])])
    assert res.status == MatchStatus.NEW, (res.status, res.distance, res.threshold_used, res.near_miss_margin)


@pytest.mark.parametrize("a,b", [("songB.flac", "songA.mp3_128"), ("songC.mp3_128", "songA.flac"),
                                 ("songC.mp3_128", "songA.mp3_64")])
def test_raised_threshold_does_not_send_unrelated_tracks_to_review(a, b):
    """At 0.35 (+ the mixed-codec boost) a 25 % band reached 0.47: two
    different songs 0.43 apart were LOW_CONFIDENCE and needed review."""
    [res] = _match([_rec("in", _key_ext(a), FPS[a])], [_rec("lib", _key_ext(b), FPS[b])],
                   thresholds={"default": 0.35})
    assert res.status == MatchStatus.NEW, (res.status, res.distance, res.threshold_used, res.near_miss_margin)


def test_near_miss_band_stops_below_the_different_recording_floor():
    assert library_sync._near_miss_margin(0.2) == pytest.approx(0.05)
    assert library_sync._near_miss_margin(0.01) == pytest.approx(library_sync.NEAR_MISS_MARGIN_FLOOR)
    assert library_sync._near_miss_margin(0.38) == pytest.approx(0.02)
    assert library_sync._near_miss_margin(0.5) == 0.0
    assert library_sync.DIFFERENT_RECORDING_FLOOR < 0.43


def test_missing_default_threshold_falls_back_to_the_engine_default():
    """``_select_threshold`` fell back to a hard-coded 0.3 (the old metric)."""
    assert library_sync._select_threshold(".mp3", {".flac": 0.1}) == pytest.approx(DEFAULTS["default"])
    assert library_sync._select_threshold(".flac", {".flac": 0.1}) == pytest.approx(0.1)


# ── prints that identify nothing never match ─────────────────────────────────

def _silence(n: int = 120) -> str:
    return compress_chromaprint([0x5A5A5A5A] * n)


@pytest.mark.parametrize("fp", ["AQAAAA", b"AQAAAA", "", "   ", None])
def test_empty_fingerprint_never_matches(fp, distance_calls):
    library = [_rec("same", ".mp3", fp)] + _others(20, ".mp3", "m") + [_rec("songA", ".flac", FPS["songA.flac"])]
    [res] = _match([_rec("in", ".mp3", fp)], library)
    assert res.status == MatchStatus.NEW
    assert res.existing is None and res.candidates == []
    assert distance_calls == []


@pytest.mark.parametrize("fp_factory", [
    lambda: compress_chromaprint(fake_raw_print("tiny", 10)),      # ~1 s: too short
    lambda: _silence(),                                             # long, but one distinct item
    lambda: compress_chromaprint([1, 2, 3, 4] * 40),                # long, 4 distinct items
])
def test_near_empty_fingerprint_matches_nothing_not_even_itself(fp_factory, distance_calls):
    fp = fp_factory()
    library = [_rec(f"copy{i}", ".flac", fp) for i in range(25)] + _others(10, ".flac", "f")
    [res] = _match([_rec("in", ".flac", fp)], library)
    assert res.status == MatchStatus.NEW
    assert res.existing is None and res.candidates == []
    assert distance_calls == []


def test_near_empty_library_prints_are_never_candidates(distance_calls):
    library = [_rec(f"s{i}", ".mp3", _silence()) for i in range(30)] + [_rec("t", ".mp3", "AQAAAA")]
    [res] = _match([_rec("in", ".flac", FPS["songA.flac"])], library)
    assert res.status == MatchStatus.NEW and res.candidates == []
    assert distance_calls == []


def test_items_shared_by_many_tracks_do_not_make_them_candidates(distance_calls):
    """A run of identical items in every library print (silence, a common
    intro) is a stop item: it must not shortlist the whole library."""
    indexed = [x for x in fake_raw_print("common-intro", 400) if library_sync._sampled_items([x])]
    shared = indexed[:8] * 2   # 16 items every print starts with, all of them indexed
    library = [
        _rec(f"t{i:03d}", ".mp3", compress_chromaprint(shared + fake_raw_print(f"body{i}", 240)))
        for i in range(60)
    ]
    incoming = _rec("in", ".mp3", compress_chromaprint(shared + fake_raw_print("new-body", 240)))
    [res] = _match([incoming], library)
    assert res.status == MatchStatus.NEW
    assert len(distance_calls) <= 3, len(distance_calls)


# ── the duration fallback ────────────────────────────────────────────────────

def _one_bit_per_item(seed: str, n: int = 240) -> tuple[str, str]:
    """A print and a copy with one flipped bit in *every* item: no exact item
    survives (the index cannot see it), yet the distance is 1/32 ≈ 0.031."""
    raw = fake_raw_print(seed, n)
    copy = [x ^ (1 << (i % 32)) for i, x in enumerate(raw)]
    return compress_chromaprint(raw), compress_chromaprint(copy)


def test_distorted_copy_without_shared_items_is_found_by_duration():
    original, copy = _one_bit_per_item("distorted")
    library = _others(40, ".mp3", "a", length=200.0) + [_rec("z_orig", ".mp3", original, 241.0)]
    profile = library_sync.PerformanceProfile()
    [res] = _match([_rec("in", ".flac", copy, 240.6)], library, profiler=profile)
    assert res.status == MatchStatus.COLLISION, (res.status, res.distance)
    assert res.existing.track_id == "z_orig"
    assert profile.fallback_shortlists == 1


def test_duration_fallback_only_compares_tracks_of_about_the_same_length(distance_calls):
    original, copy = _one_bit_per_item("too-long")
    library = [_rec("orig", ".mp3", original, 400.0)]
    [res] = _match([_rec("in", ".flac", copy, 240.0)], library)
    assert res.status == MatchStatus.NEW      # 160 s apart: not the same track
    assert distance_calls == []


def test_unknown_lengths_are_always_comparable():
    original, copy = _one_bit_per_item("no-length")
    [res] = _match([_rec("in", ".flac", copy, 240.0)], [_rec("orig", ".mp3", original, None)])
    assert res.status == MatchStatus.COLLISION


def test_new_track_costs_only_a_prescreen_per_same_length_track(distance_calls):
    """A genuinely new track is compared with every same-duration track, but
    only through the cheap pre-screen: no exact comparison survives it."""
    library = _others(150, ".mp3", "a", length=240.0) + _others(150, ".flac", "b", length=600.0)
    profile = library_sync.PerformanceProfile()
    [res] = _match([_rec("in", ".flac", fake_fingerprint("brand-new"), 240.0)], library, profiler=profile)
    assert res.status == MatchStatus.NEW
    assert len(distance_calls) <= 3, len(distance_calls)
    assert profile.fallback_shortlists + profile.signature_shortlists == 1


# ── fingerprint forms and plumbing ───────────────────────────────────────────

def test_bytes_fingerprints_match_like_text():
    library = _others(12, ".flac", "a") + [_rec("m", ".mp3", FPS["songA.mp3_128"].encode())]
    [res] = _match([_rec("in", ".flac", FPS["songA.flac"].encode())], library)
    assert res.status == MatchStatus.COLLISION and res.existing.track_id == "m"


def test_fingerprint_signature_accepts_bytes():
    """An old cache's BLOB fingerprint crashed ``TrackRecord.signature()``."""
    assert _fingerprint_signature(b"1,2,3") == "1 2 3"
    assert _fingerprint_signature(memoryview(b"AQAA 1")) == "AQAA 1"
    assert _rec("x", ".mp3", b"5 6 7").signature() == "5 6 7"
    assert _fingerprint_signature(b"") is None


def test_index_build_and_matching_stop_when_cancelled():
    cancel = threading.Event()
    cancel.set()
    library = _others(300, ".mp3", "a")
    index = library_sync._CandidateIndex(library, cancel_event=cancel)
    assert not index.complete and not index.built_from(library)
    assert _match([_rec("in", ".mp3", FPS["songA.mp3_128"])], library, cancel_event=cancel) == []


def test_recompute_matches_can_be_cancelled():
    cancel = threading.Event()
    cancel.set()
    scan = {"library_record_objects": _others(10, ".mp3", "a"),
            "incoming_record_objects": [_rec("in", ".mp3", FPS["songA.mp3_128"])]}
    out = library_sync.rematch_scan_result(scan, thresholds=DEFAULTS, mixed_codec_threshold_boost=0.03,
                                           cancel_event=cancel)
    assert out["partial"] is True and out["match_objects"] == []


def test_profile_counts_index_and_fallback_shortlists():
    library = _others(20, ".mp3", "a") + [_rec("m", ".mp3", FPS["songA.mp3_128"])]
    profile = library_sync.PerformanceProfile()
    _match([_rec("dup", ".flac", FPS["songA.flac"]), _rec("new", ".flac", fake_fingerprint("new"))],
           library, profiler=profile)
    assert profile.signature_shortlists >= 1
    assert profile.fallback_shortlists >= 1
    assert profile.extension_shortlists == 0
    assert len(profile.shortlist_sizes) == 2
