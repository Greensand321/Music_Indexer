"""The match ladder, and the invariant that protects the user's library.

The invariant: uncertainty must never resolve to OWNED. A wrong "you already
have this" silently drops a wanted song and is never mentioned again; a wrong
"you're missing this" only costs a duplicate.
"""
from __future__ import annotations

import pytest

from playlist_gap_lexicon import load_lexicon
from playlist_gap_match import DifflibScorer, group_by_reason, match_all, match_one, summarize
from playlist_gap_snapshot import LibrarySnapshot, normalize_filename
from playlist_gap_types import (
    DEFAULT_THRESHOLDS,
    LibraryTrack,
    ReasonCode,
    Rung,
    SourceCapabilities,
    Thresholds,
    Verdict,
    WantedTrack,
    stronger,
)
from playlist_gap_parse import compare_key


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon()


def snapshot_of(*tracks: LibraryTrack) -> LibrarySnapshot:
    snap = LibrarySnapshot()
    for index, track in enumerate(tracks):
        track.filename_norm = track.filename_norm or normalize_filename(track.path)
        snap.tracks.append(track)
        if track.video_id:
            snap.by_video_id.setdefault(track.video_id, []).append(index)
        if track.norm_artist:
            snap.by_artist.setdefault(track.norm_artist, []).append(index)
        if track.norm_artist and track.norm_title:
            snap.by_artist_title.setdefault((track.norm_artist, track.norm_title), []).append(index)
        for token in track.filename_norm.split():
            snap.by_filename_token.setdefault(token, set()).add(index)
        for token in (track.norm_title or "").split():
            snap.by_title_token.setdefault(token, set()).add(index)
    return snap


def track(path, artist=None, title=None, duration=None, video_id=None) -> LibraryTrack:
    return LibraryTrack(
        path=path,
        ext=".flac",
        duration=duration,
        video_id=video_id,
        norm_artist=compare_key(artist) or None,
        norm_title=compare_key(title) or None,
        tags={"artist": artist, "title": title},
    )


def run(wanted, snap, lexicon, caps=None, thresholds=DEFAULT_THRESHOLDS):
    return match_one(
        wanted, snap,
        caps=caps or SourceCapabilities(duration=True),
        lexicon=lexicon, thresholds=thresholds,
    )


# ── rung 0 / 0b — identity ───────────────────────────────────────────────────


def test_rung0_video_id_from_filename(lexicon):
    snap = snapshot_of(track("/m/Not Sorted/Cupid [Qc7_zRjH808].opus", video_id="Qc7_zRjH808"))
    result = run(
        WantedTrack(row_id="1", display="anything at all", video_id="Qc7_zRjH808"),
        snap, lexicon, caps=SourceCapabilities(video_id=True),
    )
    assert result.verdict is Verdict.OWNED
    assert result.rung is Rung.FILE_IDENTITY
    assert result.reason_code is ReasonCode.FILE_ID_MATCH


def test_rung0_video_id_from_tag(lexicon):
    snap = snapshot_of(track("/m/song.opus", video_id="Qc7_zRjH808"))
    result = run(
        WantedTrack(row_id="1", display="x", video_id="Qc7_zRjH808"),
        snap, lexicon, caps=SourceCapabilities(video_id=True),
    )
    assert result.rung is Rung.IDENTITY


def test_identity_is_skipped_when_the_source_cannot_supply_ids(lexicon):
    """Capability gating: no video_id capability means rung 0 never runs."""
    snap = snapshot_of(track("/m/x.opus", video_id="Qc7_zRjH808"))
    result = run(
        WantedTrack(row_id="1", display="Nothing Like It", video_id="Qc7_zRjH808"),
        snap, lexicon, caps=SourceCapabilities(video_id=False),
    )
    assert result.verdict is Verdict.MISSING


# ── rung 2 — core title + artist ─────────────────────────────────────────────


def test_rung2_exact_match(lexicon):
    snap = snapshot_of(track("/m/Cupid.flac", "FIFTY FIFTY", "Cupid", duration=215))
    result = run(WantedTrack(row_id="1", display="FIFTY FIFTY - Cupid", duration=215), snap, lexicon)
    assert result.verdict is Verdict.OWNED
    assert result.rung is Rung.CORE_ARTIST


def test_rung2_title_part_is_matched_not_stripped(lexicon):
    snap = snapshot_of(track(
        "/m/x.flac", "Kate Bush", "Running Up That Hill (A Deal With God)"))
    result = run(
        WantedTrack(row_id="1", display="Kate Bush - Running Up That Hill (A Deal With God)"),
        snap, lexicon,
    )
    assert result.verdict is Verdict.OWNED


def test_modifier_difference_is_never_owned(lexicon):
    """Phase 1 cannot adjudicate, so it asks rather than guessing."""
    snap = snapshot_of(track("/m/Cupid.flac", "FIFTY FIFTY", "Cupid"))
    result = run(WantedTrack(row_id="1", display="FIFTY FIFTY - Cupid (Twin Version)"), snap, lexicon)
    assert result.verdict is Verdict.UNSURE
    assert result.rung is Rung.MODIFIER


def test_wrong_artist_same_title_does_not_match(lexicon):
    """Two unrelated tracks called "Resonance" must not be confused."""
    snap = snapshot_of(track("/m/x.flac", "HOME", "Resonance"))
    result = run(WantedTrack(row_id="1", display="Wavebeatmaker - Resonance"), snap, lexicon)
    assert result.verdict is Verdict.MISSING


# ── the duration gate ────────────────────────────────────────────────────────


def test_duration_gap_forces_missing_despite_an_exact_title(lexicon):
    """Cupid vs Cupid (Twin Version): same name, 41 seconds apart."""
    snap = snapshot_of(track("/m/Cupid.flac", "FIFTY FIFTY", "Cupid", duration=215))
    result = run(WantedTrack(row_id="1", display="FIFTY FIFTY - Cupid", duration=174), snap, lexicon)
    assert result.verdict is Verdict.MISSING
    assert result.reason_code is ReasonCode.DURATION_GAP


def test_small_duration_difference_is_still_owned(lexicon):
    snap = snapshot_of(track("/m/Cupid.flac", "FIFTY FIFTY", "Cupid", duration=215))
    result = run(WantedTrack(row_id="1", display="FIFTY FIFTY - Cupid", duration=217), snap, lexicon)
    assert result.verdict is Verdict.OWNED


def test_gate_is_skipped_when_duration_is_unavailable(lexicon):
    """Today's CSV has no duration column; the gate must simply not fire."""
    snap = snapshot_of(track("/m/Cupid.flac", "FIFTY FIFTY", "Cupid", duration=215))
    result = run(
        WantedTrack(row_id="1", display="FIFTY FIFTY - Cupid", duration=None),
        snap, lexicon, caps=SourceCapabilities(duration=False),
    )
    assert result.verdict is Verdict.OWNED


def test_no_fuzzy_score_can_override_a_failing_duration_gate(lexicon):
    """The core invariant, stated as a test."""
    snap = snapshot_of(track("/m/Yotto - Radiate.flac", "Yotto", "Radiate", duration=400))
    result = run(WantedTrack(row_id="1", display="Yotto - Radiate", duration=120), snap, lexicon)
    assert result.verdict is not Verdict.OWNED


# ── rung 5b — filename ───────────────────────────────────────────────────────


def test_rung5b_matches_an_untagged_file_by_filename(lexicon):
    """The primary path for a YouTube-sourced library: tags empty, name intact."""
    snap = snapshot_of(track("/m/Not Sorted/The Sways - Someday We Will Dream About Today.opus"))
    result = run(
        WantedTrack(row_id="1", display="The Sways - Someday We Will Dream About Today"),
        snap, lexicon,
    )
    assert result.verdict is Verdict.OWNED
    assert result.rung is Rung.FUZZY_FILENAME


def test_weak_filename_similarity_is_unsure_not_owned(lexicon):
    snap = snapshot_of(track("/m/Not Sorted/Yotto - Radiate Something Else Here.opus"))
    result = run(WantedTrack(row_id="1", display="Yotto - Radiate Something Different"), snap, lexicon)
    assert result.verdict in (Verdict.UNSURE, Verdict.MISSING)
    assert result.verdict is not Verdict.OWNED


# ── rung 6 and edge rows ─────────────────────────────────────────────────────


def test_nothing_found_is_missing(lexicon):
    result = run(WantedTrack(row_id="1", display="Yotto - Radiate"), snapshot_of(), lexicon)
    assert result.verdict is Verdict.MISSING
    assert result.reason_code is ReasonCode.NO_CANDIDATE


def test_mashup_with_no_artist_is_unsure(lexicon):
    result = run(WantedTrack(row_id="1", display="Self Aware x Babydoll"), snapshot_of(), lexicon)
    assert result.verdict is Verdict.UNSURE
    assert result.reason_code is ReasonCode.MASHUP_UNPARSED


def test_blank_row_is_reported_not_dropped(lexicon):
    result = run(WantedTrack(row_id="1", display=""), snapshot_of(), lexicon)
    assert result.verdict is Verdict.UNAVAILABLE
    assert result.reason_code is ReasonCode.SOURCE_UNAVAILABLE


def test_source_marked_unavailable(lexicon):
    result = run(
        WantedTrack(row_id="1", display="Gone", available=False),
        snapshot_of(), lexicon, caps=SourceCapabilities(availability=True),
    )
    assert result.verdict is Verdict.UNAVAILABLE


# ── precedence and stored decisions ──────────────────────────────────────────

@pytest.mark.parametrize("weaker,stronger_v", [
    (Verdict.OWNED, Verdict.UNSURE),
    (Verdict.UNSURE, Verdict.MISSING),
    (Verdict.MISSING, Verdict.IGNORED),
    (Verdict.IGNORED, Verdict.UNAVAILABLE),
])
def test_verdict_precedence_never_favours_owned(weaker, stronger_v):
    assert stronger(weaker, stronger_v) is stronger_v
    assert stronger(stronger_v, weaker) is stronger_v


def test_stored_user_decision_overrides_the_matcher(lexicon):
    snap = snapshot_of(track("/m/Cupid.flac", "FIFTY FIFTY", "Cupid"))
    wanted = [WantedTrack(row_id="row-1", display="FIFTY FIFTY - Cupid")]
    results = match_all(
        wanted, snap, caps=SourceCapabilities(), lexicon=lexicon,
        decisions={"row-1": (Verdict.IGNORED, "never want this")},
    )
    assert results[0].verdict is Verdict.IGNORED
    assert results[0].auto is False
    assert results[0].reason_code is ReasonCode.USER_DECISION


def test_summarize_and_group(lexicon):
    snap = snapshot_of(track("/m/Cupid.flac", "FIFTY FIFTY", "Cupid"))
    wanted = [
        WantedTrack(row_id="1", display="FIFTY FIFTY - Cupid"),
        WantedTrack(row_id="2", display="Yotto - Radiate"),
    ]
    results = match_all(wanted, snap, caps=SourceCapabilities(), lexicon=lexicon)
    counts = summarize(results)
    assert counts["owned"] == 1 and counts["missing"] == 1 and counts["total"] == 2
    assert set(group_by_reason(results)) == {"exact_match", "no_candidate"}


def test_cancellation_stops_matching(lexicon):
    snap = snapshot_of()
    wanted = [WantedTrack(row_id=str(i), display=f"A{i} - B{i}") for i in range(10)]
    state = {"n": 0}

    def cancel():
        state["n"] += 1
        return state["n"] > 3

    results = match_all(wanted, snap, caps=SourceCapabilities(), lexicon=lexicon,
                        should_cancel=cancel)
    assert len(results) < 10


def test_scorer_is_swappable(lexicon):
    class Always(DifflibScorer):
        def ratio(self, a, b):
            return 1.0

    snap = snapshot_of(track("/m/Not Sorted/completely different.opus"))
    result = match_one(
        WantedTrack(row_id="1", display="completely different"), snap,
        caps=SourceCapabilities(), lexicon=lexicon, scorer=Always(),
    )
    assert result.verdict is Verdict.OWNED


def test_mashup_is_still_detected_when_the_source_supplies_a_title(lexicon):
    """Regression: mashup detection read splits[0].origin, but a source with a
    title column contributes a higher-prior "provided" candidate that is
    word-for-word identical — so dedupe collapsed the two and the marker was
    lost, turning an "unsure" into a "missing"."""
    result = match_one(
        WantedTrack(row_id="1", display="Self Aware x Babydoll", title="Self Aware x Babydoll"),
        snapshot_of(),
        caps=SourceCapabilities(title=True),
        lexicon=lexicon,
    )
    assert result.reason_code is ReasonCode.MASHUP_UNPARSED
    assert result.verdict is Verdict.UNSURE
