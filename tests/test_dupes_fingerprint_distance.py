"""Fingerprint distance on real Chromaprint fingerprints (DUP-05 / DUP-18).

Before: ``fingerprint_distance`` Hamming-compared the *compressed* base64
bytes (one flipped bit reshuffles every later byte, so a FLAC and its own MP3
scored 0.08–0.46) and returned 1.0 for the ``bytes`` pyacoustid hands back.
Now compressed prints are decoded and compared as raw 32-bit sub-fingerprints
(bit error rate, best of a few small alignments).

The pinned ranges come from real fingerprints of synthetic music produced by
the app's own pipeline (``tests/data/dupes_real_fingerprints.json``) and, when
ffmpeg + libchromaprint are available, from audio generated on the fly.
"""
from __future__ import annotations

import base64
import os
import shutil
import sys

import pytest

import near_duplicate_detector as nd
from config import EXACT_DUPLICATE_THRESHOLD, MIXED_CODEC_THRESHOLD_BOOST, NEAR_DUPLICATE_THRESHOLD
from _helpers.dupes_chromaprint import (
    compress_chromaprint,
    fake_fingerprint,
    fake_raw_print,
    flip_bits,
    real_fingerprints,
)

FPS = real_fingerprints()
SAME_RECORDING_LOSSY = [
    "songA.mp3_320", "songA.mp3_128", "songA.mp3_64",
    "songA.m4a_96", "songA.ogg_q2", "songA.opus_48",
]


def _compressed_byte_hamming(a: str, b: str) -> float:
    """The pre-fix metric, kept here only to show what the old code saw."""
    da = base64.urlsafe_b64decode(a + "=" * (-len(a) % 4))
    db = base64.urlsafe_b64decode(b + "=" * (-len(b) % 4))
    n = min(len(da), len(db))
    return sum((x ^ y).bit_count() for x, y in zip(da[:n], db[:n])) / (n * 8)


# ── decoding ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 7, 64, 500])
def test_decoder_round_trips_the_compressor(n):
    raw = fake_raw_print(f"rt{n}", n)
    raw[0] |= 1 << 31          # a gap of 32 bits exercises the 5-bit exception stream
    decoded = nd.decode_chromaprint(compress_chromaprint(raw, algorithm=2))
    assert decoded == (2, raw)


def test_decoder_accepts_bytes_and_padding():
    fp = FPS["songA.flac"]
    as_str = nd.decode_chromaprint(fp)
    assert as_str is not None and len(as_str[1]) > 100
    assert nd.decode_chromaprint(fp.encode("ascii")) == as_str
    assert nd.decode_chromaprint(fp + "=" * (-len(fp) % 4)) == as_str


def test_decoder_matches_libchromaprint():
    chromaprint = pytest.importorskip("chromaprint")
    try:
        ref, algorithm = chromaprint.decode_fingerprint(FPS["songA.mp3_64"].encode())
    except Exception as exc:  # library missing at runtime
        pytest.skip(f"libchromaprint unavailable: {exc}")
    assert nd.decode_chromaprint(FPS["songA.mp3_64"]) == (algorithm, [x & 0xFFFFFFFF for x in ref])


@pytest.mark.parametrize("text", [
    "1 2 3", "fp-test-001", "AAAAAAAAAAAAAAAAAAAA", "BAAAAAAAAAAAAAAAAAAA",
    base64.urlsafe_b64encode(b"\x00\x01\x02\x03\x04\x05").decode().rstrip("="),
    "", "   ", "AQAAAA",
])
def test_non_fingerprints_are_not_decoded(text):
    assert nd.decode_chromaprint(text) is None


def test_empty_prints_never_match_each_other():
    # fpcalc's answer for a clip too short to fingerprint. Two of them used to
    # be "identical" (distance 0.0), so any two short clips grouped as exact.
    assert nd.is_empty_chromaprint("AQAAAA")
    assert nd.fingerprint_distance("AQAAAA", "AQAAAA") == 1.0
    assert nd.fingerprint_distance(b"AQAAAA", FPS["songA.flac"]) == 1.0


# ── distance semantics ───────────────────────────────────────────────────────

def test_bytes_and_str_fingerprints_compare_the_same():
    """DUP-05: pyacoustid returns bytes; the Similarity page compared 1.0."""
    a, b = FPS["songA.flac"], FPS["songA.mp3_128"]
    expected = nd.fingerprint_distance(a, b)
    assert expected < EXACT_DUPLICATE_THRESHOLD
    assert nd.fingerprint_distance(a.encode(), b.encode()) == expected
    assert nd.fingerprint_distance(a.encode(), b) == expected


@pytest.mark.parametrize("variant", SAME_RECORDING_LOSSY)
def test_lossy_copies_of_the_same_recording_are_exact_duplicates(variant):
    reference = FPS["songA.flac"]
    distance = nd.fingerprint_distance(reference, FPS[variant])
    assert 0.0 <= distance <= EXACT_DUPLICATE_THRESHOLD, (variant, distance)


def test_the_old_metric_missed_these_pairs():
    """Documents the bug: the compressed-byte metric put most transcodes
    above the near + mixed-codec threshold."""
    reference = FPS["songA.flac"]
    missed = [
        v for v in SAME_RECORDING_LOSSY
        if _compressed_byte_hamming(reference, FPS[v]) > NEAR_DUPLICATE_THRESHOLD + MIXED_CODEC_THRESHOLD_BOOST
    ]
    assert len(missed) >= 4


@pytest.mark.parametrize("variant,low,high", [
    # 0.6 s of extra leading silence that was not trimmed (≈ 4.8 items of shift)
    ("songA_lead_untrimmed.flac", 0.0, NEAR_DUPLICATE_THRESHOLD),
    # same recording starting 0.3 s later (a fractional-item shift)
    ("songA_cut.mp3_128", EXACT_DUPLICATE_THRESHOLD, NEAR_DUPLICATE_THRESHOLD),
])
def test_shifted_copies_are_near_duplicates(variant, low, high):
    reference = FPS["songA.flac_untrimmed" if "untrimmed" in variant else "songA.flac"]
    distance = nd.fingerprint_distance(reference, FPS[variant])
    assert low <= distance <= high, (variant, distance)


@pytest.mark.parametrize("other", ["songB.flac", "songC.mp3_128"])
def test_different_recordings_stay_far_apart(other):
    for reference in ("songA.flac", "songA.mp3_64"):
        distance = nd.fingerprint_distance(FPS[reference], FPS[other])
        assert distance >= 0.3, (reference, other, distance)
        assert distance > NEAR_DUPLICATE_THRESHOLD + MIXED_CODEC_THRESHOLD_BOOST


def test_raw_distance_is_a_bit_error_rate():
    raw = fake_raw_print("ber", 200)
    assert nd.raw_fingerprint_distance(raw, raw) == 0.0
    assert nd.raw_fingerprint_distance(raw, flip_bits(raw, 64)) == pytest.approx(64 / (32 * 200), abs=1e-9)
    shifted = raw[3:]  # same audio, starting three items later
    assert nd.raw_fingerprint_distance(raw, shifted) == 0.0
    unrelated = fake_raw_print("other", 200)
    assert 0.4 < nd.raw_fingerprint_distance(raw, unrelated) < 0.6


def test_different_chromaprint_algorithms_are_not_comparable():
    raw = fake_raw_print("alg", 50)
    assert nd.fingerprint_distance(compress_chromaprint(raw, 1), compress_chromaprint(raw, 2)) == 1.0


def test_plain_integer_lists_keep_their_historical_semantics():
    # Synthetic fixtures across the suite are written against "fraction of
    # positions that differ"; no producer emits plain integer prints today.
    assert nd.fingerprint_distance("1 2 3", "1 2 4") == pytest.approx(1 / 3)
    assert nd.fingerprint_distance("1,2,3", "1 2 3") == 0.0


# ── the gates in front of the distance ──────────────────────────────────────

def test_coarse_keys_let_real_cross_codec_pairs_through():
    reference = set(nd._coarse_fingerprint_keys(FPS["songA.flac"]))
    for variant in SAME_RECORDING_LOSSY:
        assert reference & set(nd._coarse_fingerprint_keys(FPS[variant])), variant
    assert nd._coarse_fingerprint_keys(FPS["songA.flac"].encode())  # bytes do not crash


def test_find_near_duplicates_groups_a_flac_with_its_low_bitrate_mp3():
    infos = {
        "/lib/A/Song.flac": {"fp": FPS["songA.flac"], "title": "Song", "artist": "A", "album": "X", "ext": ".flac"},
        "/lib/A/Song.mp3": {"fp": FPS["songA.mp3_64"], "title": "Song", "artist": "A", "album": "X", "ext": ".mp3"},
        "/lib/A/Other.mp3": {"fp": FPS["songB.flac"], "title": "Song", "artist": "A", "album": "X", "ext": ".mp3"},
    }
    result = nd.find_near_duplicates(infos, {".flac": 0, ".mp3": 2}, NEAR_DUPLICATE_THRESHOLD,
                                     mixed_codec_boost=MIXED_CODEC_THRESHOLD_BOOST, max_workers=1)
    assert len(result.review_groups) == 1
    group = result.review_groups[0]
    assert {group.winner, *group.losers} == {"/lib/A/Song.flac", "/lib/A/Song.mp3"}


def test_consolidation_plan_groups_a_cross_codec_pair(tmp_path):
    from duplicate_consolidation import build_consolidation_plan

    flac, mp3, other = tmp_path / "Song.flac", tmp_path / "Song.mp3", tmp_path / "Other.mp3"
    for path in (flac, mp3, other):
        path.write_bytes(b"x")
    tags = {"artist": "Artist", "title": "Song", "album": "Album"}
    tracks = [
        {"path": str(flac), "fingerprint": FPS["songA.flac"], "ext": ".flac", "tags": dict(tags)},
        {"path": str(mp3), "fingerprint": FPS["songA.mp3_64"], "ext": ".mp3", "tags": dict(tags)},
        {"path": str(other), "fingerprint": FPS["songB.flac"], "ext": ".mp3", "tags": dict(tags)},
    ]
    plan = build_consolidation_plan(
        tracks,
        exact_duplicate_threshold=EXACT_DUPLICATE_THRESHOLD,
        near_duplicate_threshold=NEAR_DUPLICATE_THRESHOLD,
        mixed_codec_threshold_boost=MIXED_CODEC_THRESHOLD_BOOST,
    )
    assert len(plan.groups) == 1
    assert {plan.groups[0].winner_path, *plan.groups[0].losers} == {str(flac), str(mp3)}


# ── live: fingerprints of freshly encoded audio through the app's pipeline ──

def _libchromaprint_fpcalc(bin_dir):
    """An ``fpcalc`` stand-in backed by libchromaprint (via pyacoustid)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "fpcalc"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, sys, wave, chromaprint\n"
        "path = [a for a in sys.argv[1:] if not a.startswith('-')][-1]\n"
        "w = wave.open(path, 'rb'); sr, ch = w.getframerate(), w.getnchannels()\n"
        "frames = w.readframes(min(w.getnframes(), sr * 120))\n"
        "f = chromaprint.Fingerprinter(); f.start(sr, ch); f.feed(frames); fp = f.finish()\n"
        "fp = fp.decode('ascii') if isinstance(fp, bytes) else fp\n"
        "print(json.dumps({'duration': len(frames) / (2 * ch * sr), 'fingerprint': fp}))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


@pytest.fixture
def live_pipeline(tmp_path, monkeypatch):
    if os.name == "nt" or shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg and a POSIX shell")
    chromaprint = pytest.importorskip("chromaprint")
    try:
        chromaprint.Fingerprinter()
    except Exception as exc:
        pytest.skip(f"libchromaprint unavailable: {exc}")
    _libchromaprint_fpcalc(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ.get("PATH", ""))
    from _helpers.module_isolation import isolated_modules

    with isolated_modules(fresh=["chromaprint_utils"]) as mods:
        yield mods["chromaprint_utils"]


def test_live_same_tone_flac_and_mp3_are_exact_duplicates(live_pipeline, tmp_path):
    from library_factory import encode_clip

    cu = live_pipeline
    flac = encode_clip(".flac", 12.0, 262.0, tmp_path / "clips")
    mp3 = encode_clip(".mp3", 12.0, 262.0, tmp_path / "clips")
    fp_flac, fp_mp3 = cu.fingerprint_fpcalc(str(flac)), cu.fingerprint_fpcalc(str(mp3))
    assert fp_flac and fp_mp3
    assert nd.fingerprint_distance(fp_flac, fp_mp3) <= EXACT_DUPLICATE_THRESHOLD
    # A one-second clip is too short to fingerprint: "no fingerprint", not a match.
    assert cu.fingerprint_fpcalc(str(encode_clip(".flac", 1.0, 262.0, tmp_path / "clips"))) is None


def test_live_music_cross_codec_ranges(live_pipeline, tmp_path):
    np = pytest.importorskip("numpy")
    import subprocess
    import wave

    sr = 44100

    def chords(seed, seconds=20.0):
        rng = np.random.default_rng(seed)
        out = np.zeros(int(seconds * sr))
        base = int(rng.integers(48, 60))
        prog = [rng.choice([0, 3, 4, 5, 7, 9, 10], size=3, replace=False) for _ in range(8)]
        for i in range(int(seconds / 0.5)):
            s, e = int(i * 0.5 * sr), int((i + 1) * 0.5 * sr)
            t = np.arange(e - s) / sr
            env = np.exp(-3 * t)
            for k in prog[i % 8]:
                out[s:e] += env * np.sin(2 * np.pi * 440 * 2 ** ((base + k - 69) / 12) * t) * 0.2
            out[s:s + 2000] += rng.normal(0, 0.15, 2000) * np.exp(-np.arange(2000) / 300)
        return out / (np.abs(out).max() * 1.1)

    def encode(signal, name, args):
        wav = tmp_path / f"{name}.wav"
        pcm = (signal * 32767).astype("<i2")
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        out = tmp_path / name
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), *args, str(out)],
                       check=True, timeout=60)
        return live_pipeline.fingerprint_fpcalc(str(out))

    song, other = chords(1), chords(2)
    flac = encode(song, "song.flac", ["-c:a", "flac"])
    mp3 = encode(song, "song.mp3", ["-c:a", "libmp3lame", "-b:a", "64k"])
    different = encode(other, "other.flac", ["-c:a", "flac"])
    assert nd.fingerprint_distance(flac, mp3) <= EXACT_DUPLICATE_THRESHOLD
    assert nd.fingerprint_distance(flac, different) >= 0.3
