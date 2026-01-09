import base64
import hashlib
import os
import threading

from duplicate_consolidation import (
    ArtworkDirective,
    ConsolidationPlan,
    DuplicateTrack,
    PlaylistImpact,
    GroupPlan,
    _extract_artwork_from_audio,
    _metadata_changes,
    build_consolidation_plan,
    _normalize_track,
    export_consolidation_preview,
)
from duplicate_consolidation_executor import ExecutionConfig, _apply_artwork, execute_consolidation_plan


def _snapshot(path):
    data = path.read_bytes()
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "mtime": int(stat.st_mtime), "sha256": hashlib.sha256(data).hexdigest()}


def _config(tmp_path, **overrides):
    defaults = dict(
        library_root=str(tmp_path),
        reports_dir=str(tmp_path / "reports"),
        playlists_dir=str(tmp_path / "Playlists"),
        quarantine_dir=str(tmp_path / "Quarantine"),
        cancel_event=threading.Event(),
        log_callback=lambda _m: None,
    )
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def test_artwork_extraction_and_embedding(monkeypatch, tmp_path):
    class DummyPic:
        def __init__(self, payload: bytes):
            self.data = payload
            self.width = 320
            self.height = 240

    class DummyAudio:
        def __init__(self, payload: bytes):
            self.pictures = [DummyPic(payload)]
            self.tags = {}

    payload = b"art-bytes"
    art, error = _extract_artwork_from_audio(DummyAudio(payload), "song.flac")
    assert not error
    assert len(art) == 1
    assert art[0].width == 320
    assert art[0].size == len(payload)

    source = tmp_path / "source.flac"
    target = tmp_path / "target.flac"
    source.write_text("audio")
    target.write_text("audio")
    (source.with_suffix(source.suffix + ".artwork")).write_bytes(payload)

    monkeypatch.setattr("duplicate_consolidation_executor.MutagenFile", None)
    ok, detail = _apply_artwork(ArtworkDirective(source=str(source), target=str(target), reason="test copy"))
    assert ok
    assert "Mutagen unavailable" in detail
    assert (tmp_path / "target.flac.artwork").read_bytes() == payload


def test_metadata_changes_diff_generation():
    winner = DuplicateTrack(
        path="track.flac",
        fingerprint="fp",
        ext=".flac",
        bitrate=256,
        sample_rate=44100,
        bit_depth=16,
        channels=2,
        tags={},
        current_tags={"artist": "Old Artist", "title": "Keep", "album": None},
    )
    planned = {"artist": "New Artist", "title": "Keep", "album": "Album Name"}
    changes = _metadata_changes(winner, planned)

    assert changes["artist"]["from"] == "Old Artist"
    assert changes["artist"]["to"] == "New Artist"
    assert "title" not in changes  # unchanged
    assert changes["album"]["from"] is None
    assert changes["album"]["to"] == "Album Name"


def test_playlist_rewrite_validation_dry_run(tmp_path):
    playlists_dir = tmp_path / "Playlists"
    playlists_dir.mkdir()
    winner = tmp_path / "winner.flac"
    loser = tmp_path / "loser.mp3"
    winner.write_text("winner")
    loser.write_text("loser")

    playlist = playlists_dir / "mix.m3u"
    playlist.write_text(os.path.relpath(str(loser), playlists_dir) + "\n")

    snapshot = {str(winner): _snapshot(winner), str(loser): _snapshot(loser)}
    plan = ConsolidationPlan(
        groups=[
            GroupPlan(
                group_id="g-dryrun",
                winner_path=str(winner),
                losers=[str(loser)],
                planned_winner_tags={"title": "Winner"},
                winner_current_tags={"title": "Old"},
                current_tags={str(winner): {"title": "Old"}, str(loser): {"title": "Old"}},
                metadata_changes={"title": {"from": "Old", "to": "Winner"}},
                winner_quality={"reasons": ["test"]},
                artwork=[],
                artwork_candidates=[],
                chosen_artwork_source={},
                artwork_status="unchanged",
                artwork_variant_id=1,
                artwork_variant_total=1,
                artwork_variant_label="Single artwork variant",
                artwork_unknown_tracks=[],
                artwork_unknown_reasons={},
                loser_disposition={str(loser): "quarantine"},
                playlist_rewrites={str(loser): str(winner)},
                playlist_impact=PlaylistImpact(playlists=1, entries=1),
                review_flags=[],
                context_summary={"album": [], "single": [], "unknown": [str(winner)]},
                context_evidence={str(winner): [], str(loser): []},
                tag_source=str(winner),
                placeholders_present=False,
                tag_source_reason="winner",
                tag_source_evidence=[],
                track_quality={str(winner): {}, str(loser): {}},
                group_confidence="High",
                group_match_type="Exact",
                grouping_metadata_key=("artist", "title"),
                grouping_thresholds={"exact": 0.01, "near": 0.03},
                grouping_decisions=[],
                artwork_evidence=[],
                fingerprint_distances={},
                library_state=snapshot,
            )
        ]
    )

    cfg = _config(
        tmp_path,
        allow_review_required=True,
        dry_run_execute=True,
    )

    result = execute_consolidation_plan(plan, cfg)
    assert result.success is True
    assert playlist.read_text().strip() == os.path.relpath(str(loser), playlists_dir)
    assert result.playlist_reports
    rewritten = result.playlist_reports[0].playlist
    assert rewritten != str(playlist)
    rewritten_text = os.path.relpath(str(winner), playlists_dir)
    with open(rewritten, encoding="utf-8") as handle:
        rewritten_lines = [ln.strip() for ln in handle]
    assert rewritten_text in rewritten_lines
    assert (tmp_path / "loser.mp3").exists()


def test_execute_from_preview_output_path(tmp_path):
    playlists_dir = tmp_path / "Playlists"
    playlists_dir.mkdir()
    winner = tmp_path / "winner.flac"
    loser = tmp_path / "loser.mp3"
    winner.write_text("winner")
    loser.write_text("loser")

    playlist = playlists_dir / "mix.m3u"
    playlist.write_text(os.path.relpath(str(loser), playlists_dir) + "\n")

    snapshot = {str(winner): _snapshot(winner), str(loser): _snapshot(loser)}
    plan = ConsolidationPlan(
        groups=[
            GroupPlan(
                group_id="g-preview",
                winner_path=str(winner),
                losers=[str(loser)],
                planned_winner_tags={"title": "Winner"},
                winner_current_tags={"title": "Old"},
                current_tags={str(winner): {"title": "Old"}, str(loser): {"title": "Old"}},
                metadata_changes={"title": {"from": "Old", "to": "Winner"}},
                winner_quality={"reasons": ["test"]},
                artwork=[],
                artwork_candidates=[],
                chosen_artwork_source={},
                artwork_status="unchanged",
                artwork_variant_id=1,
                artwork_variant_total=1,
                artwork_variant_label="Single artwork variant",
                artwork_unknown_tracks=[],
                artwork_unknown_reasons={},
                loser_disposition={str(loser): "quarantine"},
                playlist_rewrites={str(loser): str(winner)},
                playlist_impact=PlaylistImpact(playlists=1, entries=1),
                review_flags=[],
                context_summary={"album": [], "single": [], "unknown": [str(winner)]},
                context_evidence={str(winner): [], str(loser): []},
                tag_source=str(winner),
                placeholders_present=False,
                tag_source_reason="winner",
                tag_source_evidence=[],
                track_quality={str(winner): {}, str(loser): {}},
                group_confidence="High",
                group_match_type="Exact",
                grouping_metadata_key=("artist", "title"),
                grouping_thresholds={"exact": 0.01, "near": 0.03},
                grouping_decisions=[],
                artwork_evidence=[],
                fingerprint_distances={},
                library_state=snapshot,
            )
        ]
    )

    preview_path = tmp_path / "duplicate_preview.json"
    export_consolidation_preview(plan, str(preview_path))

    cfg = _config(
        tmp_path,
        allow_review_required=True,
        dry_run_execute=True,
    )

    result = execute_consolidation_plan(str(preview_path), cfg)
    assert result.success is True
    assert result.playlist_reports
    assert result.report_paths.get("audit")


def test_invalid_preview_output_fails_safely(tmp_path):
    preview_path = tmp_path / "duplicate_preview.json"
    preview_path.write_text("{\"plan\": \"invalid\"}")

    cfg = _config(tmp_path)

    result = execute_consolidation_plan(str(preview_path), cfg)
    assert result.success is False
    assert any("Invalid consolidation plan" in action.detail for action in result.actions)


def test_sidecar_artwork_in_report(monkeypatch, tmp_path):
    payload = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/xcAAgMBAp7Hc34AAAAASUVORK5CYII="
    )
    with_sidecar = tmp_path / "song.m4a"
    with_sidecar.write_text("audio")
    (tmp_path / "song.m4a.jpg").write_bytes(payload)

    monkeypatch.setattr("duplicate_consolidation.read_metadata", lambda _p, include_cover=True: ({}, [], None, None))
    monkeypatch.setattr("duplicate_consolidation._read_audio_file", lambda _p: (None, None))

    track = _normalize_track({"path": str(with_sidecar), "fingerprint": "1 2 3", "ext": ".m4a"})

    assert track.artwork
    assert track.trace["album_art"]["cover_count"] == 1
    assert track.trace["album_art"]["mp4_covr_missing"] is False

def test_missing_tags_and_placeholders_require_review(tmp_path):
    winner = tmp_path / "winner.flac"
    loser = tmp_path / "loser.flac"
    winner.write_text("winner")
    loser.write_text("loser")

    tracks = [
        {"path": str(winner), "fingerprint": "fp-tags", "ext": ".flac", "tags": {"title": "Demo Track"}},
        {"path": str(loser), "fingerprint": "fp-tags", "ext": ".flac", "tags": {"title": "Demo Track"}},
    ]
    plan = build_consolidation_plan(tracks)

    assert plan.requires_review
    assert plan.review_required_count > 0
    assert any("placeholder" in flag.lower() or "missing critical" in flag.lower() for flag in plan.groups[0].review_flags)

    cfg = _config(tmp_path)
    result = execute_consolidation_plan(plan, cfg)
    assert result.success is False
    assert (tmp_path / "loser.flac").exists()


def test_truncation_sets_review_required_count(tmp_path):
    tracks = [
        {"path": str(tmp_path / f"track-{idx}.flac"), "fingerprint": f"fp-{idx}", "ext": ".flac"}
        for idx in range(10)
    ]
    for track in tracks:
        open(track["path"], "w").close()

    plan = build_consolidation_plan(tracks, max_candidates=5)

    assert plan.review_flags
    assert plan.requires_review
    assert plan.review_required_count >= 1


def test_coarse_fingerprint_gate_allows_cross_codec_matches(tmp_path):
    flac = tmp_path / "Song.flac"
    m4a = tmp_path / "Song.m4a"
    flac.write_text("a")
    m4a.write_text("b")
    tracks = [
        {
            "path": str(flac),
            "fingerprint": " ".join(str(v) for v in range(40)),
            "ext": ".flac",
            "tags": {"artist": "Artist", "title": "Song"},
        },
        {
            "path": str(m4a),
            "fingerprint": " ".join([str(v) for v in range(39)] + ["99"]),
            "ext": ".m4a",
            "tags": {"artist": "Artist", "title": "Song"},
        },
    ]

    plan = build_consolidation_plan(tracks, near_duplicate_threshold=0.2)
    assert len(plan.groups) == 1
    group_paths = {plan.groups[0].winner_path, *plan.groups[0].losers}
    assert group_paths == {str(flac), str(m4a)}


def test_metadata_gate_uses_normalized_titles(tmp_path):
    base_a = tmp_path / "Track (Remastered 2014).flac"
    base_b = tmp_path / "Track.m4a"
    other = tmp_path / "Different.mp3"
    for p in (base_a, base_b, other):
        p.write_text("x")
    fp_a = " ".join(["10"] * 20 + ["12"] * 10)
    fp_b = " ".join(["10"] * 20 + ["12"] * 9 + ["11"])
    fp_other = " ".join(["10"] * 30)
    tracks = [
        {
            "path": str(base_a),
            "fingerprint": fp_a,
            "ext": ".flac",
            "tags": {"artist": "Artist", "title": "Track (Remastered 2014)"},
        },
        {
            "path": str(base_b),
            "fingerprint": fp_b,
            "ext": ".m4a",
            "tags": {"artist": "Artist", "title": "Track"},
        },
        {
            "path": str(other),
            "fingerprint": fp_other,
            "ext": ".mp3",
            "tags": {"artist": "Artist", "title": "Different Song"},
        },
    ]

    plan = build_consolidation_plan(tracks, near_duplicate_threshold=0.2)
    assert len(plan.groups) == 1
    group_paths = {plan.groups[0].winner_path, *plan.groups[0].losers}
    assert group_paths == {str(base_a), str(base_b)}
