"""A short clip is not the full song (Library Sync matching, area: syncmatch).

A fingerprint covers the first 120 s of a track, so a 30-second clip's print
is a *prefix* of the full song's print. Library Sync scored pairs with the
overlap-only ``fingerprint_distance``, which compared just the shared 30 s
and returned ~0.0: the clip was an "exact match" of the song. Consequences:

* an incoming full song whose library copy is only a clip (a preview, an
  intro cut) was "already in the library" and skipped — never added;
* an incoming lossless clip of a lossy library song was a "possible upgrade",
  one Replace flag away from swapping the song for 30 seconds of it.

The matcher now scores pairs with ``match_distance`` — the vendored port of
the root ``duplicate_fingerprint_distance``, which counts the unmatched part
of the longer print as unrelated audio. A clip under half the song's printed
length lands above the review band (NEW); about half is borderline (review,
never an automatic decision); the same recording in another codec, or a copy
a few seconds shorter, matches exactly as before.

The candidate index still *finds* the song for the clip (they share their
sub-fingerprints), so its prefilter neither hides the pair nor decides it:
the song is compared, and the length-aware distance says "different".
"""
from __future__ import annotations

import itertools
import os

import pytest

import library_sync
import near_duplicate_detector as root_nd
from _helpers.dupes_chromaprint import compress_chromaprint, fake_raw_print, flip_bits, real_fingerprints
from library_sync_indexer_engine import config as vconfig
from library_sync_indexer_engine import near_duplicate_detector as vnd
from library_sync_types import MatchStatus, TrackRecord

ITEMS_PER_SECOND = 8          # ~0.124 s per Chromaprint sub-fingerprint
WINDOW = 950                  # ~120 s: what a print of a long track holds
DEFAULTS = vconfig.DEFAULT_FP_THRESHOLDS
PRIORITY = {".flac": 3, ".wav": 2, ".mp3": 1}
MATCHED = (MatchStatus.EXACT_MATCH, MatchStatus.COLLISION)

SONG = fake_raw_print("full song", WINDOW)


def _print(raw, *, flips: int = 0, seed: str = "codec") -> str:
    return compress_chromaprint(flip_bits(raw, flips, seed) if flips else raw)


def _clip(seconds: int, *, flips: int = 0) -> str:
    """The song's first ``seconds``, as its own (possibly re-encoded) file."""
    return _print(SONG[: seconds * ITEMS_PER_SECOND], flips=flips, seed=f"clip{seconds}")


FULL_FLAC = _print(SONG)
FULL_MP3 = _print(SONG, flips=300, seed="mp3")    # ~1 % of the bits: a lossy transcode


# ── the metric ───────────────────────────────────────────────────────────────

def test_matcher_uses_the_length_aware_metric():
    assert library_sync.match_distance is vnd.duplicate_fingerprint_distance
    # The overlap-only metric is still bound (and still the root's).
    assert library_sync.fingerprint_distance is vnd.fingerprint_distance


def test_vendored_length_aware_distance_matches_the_root():
    fps = list(real_fingerprints().values()) + [FULL_FLAC, FULL_MP3, _clip(30), _clip(60, flips=40), "1 2 3"]
    for a, b in itertools.combinations(fps, 2):
        assert vnd.duplicate_fingerprint_distance(a, b) == root_nd.duplicate_fingerprint_distance(a, b)
        assert vnd.fingerprint_distance(a, b) == root_nd.fingerprint_distance(a, b)


def test_overlap_only_distance_calls_a_clip_its_song():
    """The bug, measured: the old metric cannot tell a clip from its song."""
    assert vnd.fingerprint_distance(_clip(30), FULL_FLAC) == 0.0
    assert vnd.fingerprint_distance(_clip(30, flips=30), FULL_MP3) < 0.03
    assert vnd.duplicate_fingerprint_distance(_clip(30), FULL_FLAC) > 0.35
    assert vnd.duplicate_fingerprint_distance(_clip(60), FULL_FLAC) > 0.2


# ── through the matcher ──────────────────────────────────────────────────────

def _rec(name: str, ext: str, fp: str, length: float | None) -> TrackRecord:
    return TrackRecord(
        track_id=name, path=f"/music/{name}{ext}", normalized_path=f"/music/{name}{ext}", ext=ext,
        bitrate=0, size=1, mtime=0.0, fingerprint=fp, tags={}, duration=120, audio_length=length,
    )


def _match(incoming: TrackRecord, library: list[TrackRecord]):
    [res] = library_sync._match_tracks(
        [incoming], library, DEFAULTS, PRIORITY,
        mixed_codec_threshold_boost=vconfig.MIXED_CODEC_THRESHOLD_BOOST,
    )
    return res


@pytest.mark.parametrize("length", [30.0, None])
@pytest.mark.parametrize("inc_ext,lib_ext,lib_fp", [
    (".flac", ".mp3", FULL_MP3),    # was "Potential Upgrade"
    (".mp3", ".flac", FULL_FLAC),   # was "Already in the library"
    (".flac", ".flac", FULL_FLAC),  # was an exact match (distance 0.0)
])
def test_thirty_second_clip_is_new_although_the_index_finds_the_song(inc_ext, lib_ext, lib_fp, length):
    library = [_rec("song", lib_ext, lib_fp, 240.0)]
    res = _match(_rec("clip", inc_ext, _clip(30, flips=20 if inc_ext == ".mp3" else 0), length), library)
    assert res.status == MatchStatus.NEW, (res.status, res.distance)
    # Compared (the prefilter found it) and judged far: not hidden, not matched.
    assert [c.track.track_id for c in res.candidates] == ["song"]
    assert res.distance > res.threshold_used + res.near_miss_margin


def test_full_song_is_new_when_the_library_only_holds_a_clip():
    res = _match(_rec("song", ".mp3", FULL_MP3, 240.0), [_rec("clip", ".flac", _clip(30), 30.0)])
    assert res.status == MatchStatus.NEW, (res.status, res.distance)


def test_sixty_second_clip_is_at_most_sent_to_review():
    res = _match(_rec("clip", ".flac", _clip(60), 60.0), [_rec("song", ".mp3", FULL_MP3, 240.0)])
    assert res.status not in MATCHED, (res.status, res.distance)


def test_same_recording_in_another_codec_still_matches():
    res = _match(_rec("song", ".flac", FULL_FLAC, 240.0), [_rec("song", ".mp3", FULL_MP3, 240.0)])
    assert res.status == MatchStatus.COLLISION and res.distance < 0.02
    assert res.quality_label == "Potential Upgrade"


@pytest.mark.parametrize("track_seconds,shorter_by", [(100, 3), (100, 6), (60, 3), (240, 5)])
def test_a_copy_a_few_seconds_shorter_still_matches(track_seconds, shorter_by):
    """A different rip/edit that loses a few seconds at the end (tracks under
    the 120 s window; longer ones print the same 120 s either way)."""
    raw = fake_raw_print(f"track{track_seconds}", min(WINDOW, track_seconds * ITEMS_PER_SECOND))
    shorter = raw[: min(len(raw), (track_seconds - shorter_by) * ITEMS_PER_SECOND)]
    res = _match(
        _rec("copy", ".mp3", _print(shorter, flips=len(shorter) // 3), track_seconds - shorter_by),
        [_rec("orig", ".flac", _print(raw), track_seconds)],
    )
    before = vnd.fingerprint_distance(_print(shorter, flips=len(shorter) // 3), _print(raw))
    assert res.status == MatchStatus.COLLISION, (res.status, res.distance, before)
    assert res.distance - before < 0.03


def test_real_prints_keep_their_statuses():
    """The real-audio fixtures (all ~26 s, a few items apart) score as before
    under the length-aware metric, except the 0.6 s leading-silence copy,
    which moves slightly but stays a match."""
    fps = real_fingerprints()
    ref = fps["songA.flac"]
    for key, fp in fps.items():
        before = vnd.fingerprint_distance(ref, fp)
        after = vnd.duplicate_fingerprint_distance(ref, fp)
        if key == "songA_lead_untrimmed.flac":
            assert before < after < 0.2
        else:
            assert after == before, key


# ── end to end: compare_libraries and the plan ───────────────────────────────

PRINTS = {
    "song.mp3": FULL_MP3,            # library: the full song, lossy
    "other.mp3": _print(fake_raw_print("unrelated", WINDOW), flips=200),
    "clip.flac": _clip(30),          # incoming: a lossless 30 s clip of it
    "song.flac": FULL_FLAC,          # incoming: the same song, lossless
}


@pytest.fixture
def folders(tmp_path, monkeypatch):
    def fake(paths, db_path, log_callback=None, progress_callback=None, cancel_event=None, **_kw):
        return [(p, 120, PRINTS[os.path.basename(p)]) for p in paths]

    monkeypatch.setattr(library_sync.fingerprint_generator, "compute_fingerprints_parallel", fake)
    library = tmp_path / "library"
    incoming = tmp_path / "incoming"
    for folder, names in ((library, ("song.mp3", "other.mp3")), (incoming, ("clip.flac", "song.flac"))):
        for name in names:
            path = folder / "Artist" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode() * 64)
    return library, incoming


def test_compare_libraries_reports_the_clip_as_new(folders, tmp_path):
    library, incoming = folders
    result = library_sync.compare_libraries(
        str(library), str(incoming), str(tmp_path / "Docs" / "sync.db"), include_match_objects=True,
    )
    name = os.path.basename
    assert [name(p) for p in result["new"]] == ["clip.flac"]
    assert [(name(a), name(b)) for a, b in result["improved"]] == [("song.flac", "song.mp3")]
    assert result["existing_matches"] == []

    plan = library_sync.compute_library_sync_plan(
        str(library), str(incoming), transfer_mode="copy", match_results=result["match_objects"],
    )
    decisions = {name(item.source): item.decision for item in plan.items}
    assert decisions["clip.flac"] == library_sync.ExecutionDecision.COPY
    assert decisions["song.flac"] == library_sync.ExecutionDecision.REVIEW_REQUIRED
