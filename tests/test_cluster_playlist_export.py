"""Tests for cluster_playlist_export — Save cluster playlists (spec CP-10…CP-46).

Qt-free. Runs are fabricated with ``tests/_helpers/cluster_payloads.py``;
tags come from an injected ``tag_reader`` keyed by the track's index (the
helper names every track ``…/track_<index>.mp3``, and cluster = index % 3 by
default).
"""
from __future__ import annotations

import ast
import builtins
import dataclasses
import json
import os
import threading
import time
from pathlib import Path

import pytest

import cluster_graph_data as cgd
import cluster_playlist_export as cpe
import playlist_generator as pg
from _helpers.cluster_payloads import (
    make_cluster_run,
    make_member_files,
    run_block,
    track_path,
    write_cluster_run,
)

REPO = Path(__file__).resolve().parents[1]


# ── Helpers ───────────────────────────────────────────────────────────────


def _index(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def _reader(by_index=None, default=None, calls=None):
    """A tag reader: ``by_index[i]`` (or ``default``) for track i."""
    by_index = by_index or {}

    def read(path):
        if calls is not None:
            calls.append(path)
        try:
            i = _index(path)
        except (ValueError, IndexError):
            return dict(default or {})
        return dict(by_index.get(i, default or {}))
    return read


def _lib(tmp_path, kind="valid", files=True, name="Library", **kw):
    lib = tmp_path / name
    lib.mkdir(parents=True, exist_ok=True)
    payload, members = write_cluster_run(lib, kind=kind, files=files, **kw)
    return lib, payload, members


def _write(lib: Path, payload: dict, members: dict, files: bool = True) -> None:
    docs = lib / "Docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "cluster_info.json").write_text(json.dumps(payload), encoding="utf-8")
    (docs / "cluster_members.json").write_text(json.dumps(members), encoding="utf-8")
    if files:
        make_member_files(members["tracks"])


def _summary(lib, **kw):
    kw.setdefault("tag_reader", _reader())
    return cpe.summarize_cluster_run(lib, **kw)


def _plan(summary, ids=None, **kw):
    return cpe.plan_cluster_playlists(summary, set(summary.clusters) if ids is None else ids, **kw)


def _names(summary) -> dict:
    return {cid: f.name for cid, f in summary.clusters.items()}


def _tree(root: Path) -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            out[os.path.join(dirpath, d)] = "dir"
        for f in filenames:
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            out[p] = (st.st_mtime_ns, st.st_size)
    return out


def _by_cluster(n=20, n_clusters=3):
    out: dict = {}
    for i in range(n):
        out.setdefault(i % n_clusters, []).append(i)
    return out


# ── CP-12: refusals ───────────────────────────────────────────────────────


def test_legacy_payload_is_refused_with_reanalyse_message(tmp_path):
    lib, *_ = _lib(tmp_path, kind="legacy")
    summary = _summary(lib)
    assert summary.blocking_error == cpe.LEGACY_MESSAGE
    assert "Re-run the analysis" in summary.blocking_error and not summary.clusters
    plan = cpe.plan_cluster_playlists(summary, {0, 1})
    assert not plan.ok and plan.items == [] and plan.errors == [cpe.LEGACY_MESSAGE]

    # A run block but no members file (it failed to write) is refused the same way.
    lib2, *_ = _lib(tmp_path, name="L2")
    cgd.cluster_members_path(lib2).unlink()
    assert _summary(lib2).blocking_error == cpe.LEGACY_MESSAGE

    # No map at all: the existing load_cluster_data text.
    empty = tmp_path / "Empty"
    empty.mkdir()
    assert "No cluster data" in _summary(empty).blocking_error


def test_run_mismatch_is_refused(tmp_path):
    lib, *_ = _lib(tmp_path, kind="mismatched_run")
    assert "out of step with the map" in _summary(lib).blocking_error
    lib2, *_ = _lib(tmp_path, kind="misaligned_members", name="L2")
    assert "misaligned" in _summary(lib2).blocking_error


def test_all_noise_is_refused(tmp_path):
    lib, *_ = _lib(tmp_path, kind="all_noise")
    assert _summary(lib).blocking_error == cpe.ALL_NOISE_MESSAGE


def test_all_missing_is_refused(tmp_path):
    lib, *_ = _lib(tmp_path, files=False)
    assert _summary(lib).blocking_error == cpe.ALL_MISSING_MESSAGE


# ── CP-11: presence ───────────────────────────────────────────────────────


def test_missing_and_reserved_members_are_left_out_and_counted(tmp_path):
    lib = tmp_path / "Library"
    tracks = [track_path(str(lib), i) for i in range(20)]
    quarantined = str(lib / "Quarantine" / "track_00006.mp3")
    tracks[6] = quarantined                     # cluster 0, file exists but reserved
    payload, members = make_cluster_run(n=20, root=str(lib), tracks=tracks)
    _write(lib, payload, members)
    os.remove(tracks[3])                        # cluster 0, gone

    summary = _summary(lib)
    facts = summary.clusters[0]
    assert sorted(facts.missing) == sorted([tracks[3], quarantined])
    assert len(facts.present) == 5
    assert summary.missing_total == 2
    assert all(p not in (tracks[3], quarantined) for p, _e, _d in facts.present)
    plan = _plan(summary)
    row = next(r for r in plan.rows if r.cluster_id == 0)
    assert (row.tracks, row.missing) == (5, 2)
    item = next(i for i in plan.items if i.lines[2].endswith("cluster=0"))
    assert item.entries == 5
    assert not any("track_00003" in line or "Quarantine" in line for line in item.lines)


def test_presence_check_uses_long_paths(tmp_path, monkeypatch):
    lib, _payload, members = _lib(tmp_path, files=False)
    monkeypatch.setattr(cpe, "ensure_long_path", lambda p: "LONG:" + str(p))
    seen, reads = [], []

    def exists(path):
        seen.append(path)
        return True

    summary = _summary(lib, exists=exists, tag_reader=_reader(calls=reads))
    assert summary.blocking_error is None
    assert sorted(seen) == sorted("LONG:" + t for t in members["tracks"])
    assert reads and all(p.startswith("LONG:") for p in reads)
    # The long form never leaks into what is shown or written.
    for facts in summary.clusters.values():
        assert all(not p.startswith("LONG:") and "LONG:" not in e for p, e, _d in facts.present)


# ── CP-10: a moved library ────────────────────────────────────────────────


def test_moved_library_root_is_rebased_with_warning(tmp_path):
    old_root = os.path.abspath(str(tmp_path / "old" / "Music"))
    lib = tmp_path / "Library"
    tracks = [track_path(old_root, i) for i in range(19)] + [str(tmp_path / "elsewhere" / "x_00019.mp3")]
    failed = [os.path.join(old_root, "Artist 9", "broken.mp3")]
    payload, members = make_cluster_run(n=20, root=old_root, tracks=tracks, failed=failed)
    _write(lib, payload, members, files=False)
    rebased = [track_path(str(lib), i) for i in range(19)]
    make_member_files(rebased + [os.path.join(lib, "Artist 9", "broken.mp3")])

    summary = _summary(lib)
    assert summary.rebased_from == old_root
    assert any(old_root in w and "mapped to the current library" in w for w in summary.warnings)
    present = {p for f in summary.clusters.values() for p, _e, _d in f.present}
    assert present == set(rebased)
    # A path outside the old root is kept as stored (and is missing here).
    assert summary.clusters[19 % 3].missing == [tracks[19]]
    # The failed track was rebased too, so it is not counted as new.
    assert summary.new_tracks == 0


# ── CP-13: new tracks since the analysis ──────────────────────────────────


def test_new_tracks_are_counted_and_reserved_folders_ignored(tmp_path):
    lib, *_ = _lib(tmp_path)
    for rel in ("New Artist/n1.mp3", "New Artist/n2.flac", "New Artist/cover.jpg",
                "Quarantine/q.mp3", "Playlists/p.mp3", "Docs/d.mp3", "Trash/t.opus"):
        p = lib / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0")
    summary = _summary(lib)
    assert summary.new_tracks == 2
    assert "2 tracks were added after this analysis and are in no cluster." in summary.warnings
    assert not any("changed a lot" in w for w in summary.warnings)


def test_big_change_suggests_reanalysis(tmp_path):
    lib, _payload, members = _lib(tmp_path)          # 20 tracks: the limit is > 2
    for name in ("a.mp3", "b.mp3"):
        (lib / name).write_bytes(b"\0")
    assert not any("changed a lot" in w for w in _summary(lib).warnings)
    os.remove(members["tracks"][0])                  # 1 missing + 2 new = 3 > 2
    warnings = _summary(lib).warnings
    assert "Your library changed a lot since this analysis; consider re-analysing." in warnings


def test_audio_exts_match_the_analysis_worker():
    source = (REPO / "gui" / "workspaces" / "clustered_enhanced.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    worker_exts = next(
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_AUDIO_EXTS" for t in node.targets)
    )
    assert cpe.AUDIO_EXTS == frozenset(worker_exts)


def test_default_min_size_matches_hdbscan_default():
    pytest.importorskip("sklearn")
    import clustered_playlists

    assert cpe.DEFAULT_MIN_SIZE == clustered_playlists.DEFAULT_HDBSCAN_MIN_SIZE


# ── CP-26: the summary is read-only, cancellable, reports progress ───────


def test_summary_writes_nothing(tmp_path):
    lib, *_ = _lib(tmp_path)
    first = _summary(lib)
    folder = Path(first.run_folder)
    folder.mkdir(parents=True)
    (folder / "Cluster 00.m3u").write_text("#EXTM3U\n", encoding="utf-8")
    (lib / "Playlists" / "Clusters" / "older set").mkdir()
    before = _tree(lib)
    summary = _summary(lib)
    assert summary.blocking_error is None and summary.clusters
    assert _tree(lib) == before


def test_summary_honours_cancel_event(tmp_path):
    lib, *_ = _lib(tmp_path)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(cpe.ClusterPlaylistsCancelled):
        _summary(lib, cancel_event=cancel)

    cancel.clear()

    def reader(path):
        cancel.set()            # cancelled while the tags are being sampled
        return {}

    with pytest.raises(cpe.ClusterPlaylistsCancelled):
        _summary(lib, cancel_event=cancel, tag_reader=reader)


def test_summary_reports_progress(tmp_path):
    lib, *_ = _lib(tmp_path, n=450)
    seen = []
    _summary(lib, progress_callback=lambda d, t: seen.append((d, t)))
    assert len(seen) >= 3
    totals = {t for _d, t in seen}
    assert len(totals) == 1
    total = totals.pop()
    assert total == 450 + 3 * 50            # presence checks + capped tag reads
    dones = [d for d, _t in seen]
    assert dones == sorted(dones) and dones[-1] == total


# ── CP-14…CP-16: names ────────────────────────────────────────────────────


def test_names_are_deterministic(tmp_path):
    lib, *_ = _lib(tmp_path, n=30)
    tags = {i: {"genre": ["Rock", "Jazz", "Pop"][i % 3], "artist": f"A{i % 5}"} for i in range(30)}
    first = _names(_summary(lib, tag_reader=_reader(tags)))
    second = _names(_summary(lib, tag_reader=_reader(tags)))
    assert first == second
    assert first[0] == "Cluster 00 - Rock - ~90 BPM - low energy"


def test_name_uses_dominant_genre_tempo_and_energy(tmp_path):
    lib, *_ = _lib(tmp_path)
    groups = _by_cluster()
    tags = {i: {"genre": "Deep House"} for i in groups[0]}
    tags.update({i: {"genre": "  Techno "} for i in groups[2]})
    names = _names(_summary(lib, tag_reader=_reader(tags)))
    assert names == {
        0: "Cluster 00 - Deep House - ~90 BPM - low energy",
        1: "Cluster 01 - ~100 BPM",
        2: "Cluster 02 - Techno - ~110 BPM - high energy",
    }


def test_second_genre_is_appended_with_ampersand(tmp_path):
    lib, *_ = _lib(tmp_path, tempo=False, energy=False)
    groups = _by_cluster()
    tags = {}
    for k, i in enumerate(groups[0]):        # 7 tracks: 4 techno (one lower-case), 3 house
        tags[i] = {"genre": ["Techno", "techno", "Techno", "Techno", "House", "House", "House"][k]}
    for k, i in enumerate(groups[1]):        # 6 jazz, 1 blues (14 %: not appended)
        tags[i] = {"genre": "Blues" if k == 0 else "Jazz"}
    names = _names(_summary(lib, tag_reader=_reader(tags)))
    assert names[0] == "Cluster 00 - Techno & House"
    assert names[1] == "Cluster 01 - Jazz"


def test_artist_only_without_dominant_genre(tmp_path):
    lib, *_ = _lib(tmp_path, tempo=False, energy=False)
    groups = _by_cluster()
    tags = {}
    for k, i in enumerate(groups[0]):        # 7 different genres; Radiohead on 4 of 7
        tags[i] = {"genre": f"G{k}", "albumartist": "Radiohead" if k < 4 else f"Other {k}"}
    for k, i in enumerate(groups[1]):        # albumartist empty → falls back to artist
        tags[i] = {"artist": "Björk" if k < 5 else "X"}
    for k, i in enumerate(groups[2]):        # only 2 tracks of one artist: < 3
        tags[i] = {"artist": "Duo" if k < 2 else f"Solo {k}"}
    names = _names(_summary(lib, tag_reader=_reader(tags)))
    assert names[0] == "Cluster 00 - Radiohead"
    assert names[1] == "Cluster 01 - Björk"
    assert names[2] == "Cluster 02"
    # A dominant genre wins over a dominant artist.
    tags[groups[0][0]]["genre"] = tags[groups[0][1]]["genre"] = tags[groups[0][2]]["genre"] = "Rock"
    assert _names(_summary(lib, tag_reader=_reader(tags)))[0] == "Cluster 00 - Rock"


def test_various_artists_is_ignored(tmp_path):
    lib, *_ = _lib(tmp_path, tempo=False, energy=False)
    tags = {i: {"albumartist": "VARIOUS ARTISTS" if i % 2 else "Various Artists", "artist": ""}
            for i in range(20)}
    names = _names(_summary(lib, tag_reader=_reader(tags)))
    assert names == {0: "Cluster 00", 1: "Cluster 01", 2: "Cluster 02"}


def test_name_falls_back_to_cluster_number(tmp_path):
    def broken(path):
        raise OSError("unreadable")

    lib, *_ = _lib(tmp_path, tempo=False, energy=False)
    assert _names(_summary(lib, tag_reader=broken)) == {0: "Cluster 00", 1: "Cluster 01", 2: "Cluster 02"}


def test_cluster_ids_are_zero_padded_to_the_widest(tmp_path):
    labels = [0, 5, 120] * 6
    lib = tmp_path / "Library"
    payload, members = make_cluster_run(labels=labels, n_clusters=3, root=str(lib), tempo=False, energy=False)
    _write(lib, payload, members)
    names = _names(_summary(lib))
    assert names == {0: "Cluster 000", 5: "Cluster 005", 120: "Cluster 120"}

    small = tmp_path / "Small"
    payload, members = make_cluster_run(labels=[0, 7] * 6, n_clusters=2, root=str(small), tempo=False, energy=False)
    _write(small, payload, members)
    assert _names(_summary(small)) == {0: "Cluster 00", 7: "Cluster 07"}


def test_tag_sample_is_capped_for_many_clusters(tmp_path):
    lib = tmp_path / "Library"
    n_clusters, per = 100, 30
    labels = [i % n_clusters for i in range(n_clusters * per)]
    payload, members = make_cluster_run(labels=labels, n_clusters=n_clusters, root=str(lib),
                                        tempo=False, energy=False)
    _write(lib, payload, members, files=False)
    calls: list = []
    summary = _summary(lib, exists=lambda p: True, tag_reader=_reader(calls=calls))
    assert len(calls) == 100 * 20          # S = min(50, max(10, 2000 // 100)) = 20
    # Each cluster's sample is its 20 tracks closest to the centre.
    sampled = sorted(_index(p) for p in calls if _index(p) % n_clusters == 7)
    closest = sorted(sorted((i for i in range(len(labels)) if labels[i] == 7),
                            key=lambda i: members["centroid_distance"][i])[:20])
    assert sampled == closest
    assert len(summary.clusters) == 100


def test_energy_needs_three_values(tmp_path):
    lib = tmp_path / "Library"
    payload, members = make_cluster_run(n=20, root=str(lib), tempo=False)
    members["energy"] = [None] * 20
    members["energy"][0], members["energy"][1] = 0.1, 0.9
    _write(lib, payload, members)
    assert _names(_summary(lib)) == {0: "Cluster 00", 1: "Cluster 01", 2: "Cluster 02"}


# ── CP-17: sanitising ─────────────────────────────────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ('a<b>c:d"e/f\\g|h?i*j', "a-b-c-d-e-f-g-h-i-j"),
    ("tab\there\nnew\x00line", "tabherenewline"),
    ("  many    spaces  here ", "many spaces here"),
    ("  .hidden. ", "hidden"),
    ("...dots...", "dots"),
    ("CON", "CON_"),
    ("con", "con_"),
    ("nul.mix", "nul.mix_"),
    ("CON .mix", "CON .mix_"),
    ("COM¹", "COM1_"),
    ("CONIN$", "CONIN$_"),
    ("LPT0", "LPT0_"),
    ("Console", "Console"),
    ("COM10", "COM10"),
])
def test_sanitize_stem(raw, expected):
    assert cpe.sanitize_playlist_stem(raw) == expected


def test_sanitize_stem_trims_then_strips_and_falls_back():
    # A cut that would leave a trailing dot or space is stripped after the trim.
    assert cpe.sanitize_playlist_stem("a" * 98 + " .b", limit=100) == "a" * 98
    assert cpe.sanitize_playlist_stem("a" * 99 + ".b", limit=100) == "a" * 99
    assert len(cpe.sanitize_playlist_stem("x" * 300)) == 100
    assert cpe.sanitize_playlist_stem("", fallback="Cluster 03") == "Cluster 03"
    assert cpe.sanitize_playlist_stem(" . ", fallback="Cluster 03") == "Cluster 03"
    assert cpe.sanitize_playlist_stem("\x01\x02", fallback="Cluster 03") == "Cluster 03"
    # The device suffix never pushes a stem over its limit.
    assert len(cpe.sanitize_playlist_stem("nul." + "x" * 200, limit=50)) <= 50


# ── CP-18: user names and collisions ──────────────────────────────────────


def test_user_names_colliding_by_case_invalidate_the_plan(tmp_path):
    lib, *_ = _lib(tmp_path)
    summary = _summary(lib)
    plan = _plan(summary, names={0: "House", 1: "house"})
    assert not plan.ok
    assert plan.errors == ["Two playlists would be called House"]
    reasons = {r.cluster_id: r.reason for r in plan.rows}
    assert reasons[0] == reasons[1] == "Two playlists would be called House"
    assert cpe.write_cluster_playlists(plan)["refused"]
    assert not os.path.exists(summary.run_folder)
    # Unticking one of them makes the plan valid again.
    assert _plan(summary, {0, 2}, names={0: "House", 1: "house"}).ok
    # Names are sanitised (and an emptied name falls back to the cluster number).
    plan = _plan(summary, names={0: "a/b", 1: "  "})
    rows = {r.cluster_id: r.name for r in plan.rows}
    assert rows[0] == "a-b" and rows[1] == "Cluster 01"


def test_existing_file_matched_by_case_becomes_replace_with_exact_name(tmp_path):
    lib, *_ = _lib(tmp_path)
    folder = Path(_summary(lib).run_folder)
    folder.mkdir(parents=True)
    existing = folder / "house.m3u"
    existing.write_text("#EXTM3U\n", encoding="utf-8")
    st = os.stat(existing)

    summary = _summary(lib)
    assert summary.existing_files == {"house.m3u": ("house.m3u", (st.st_mtime_ns, st.st_size))}
    plan = _plan(summary, names={0: "House"})
    row = next(r for r in plan.rows if r.cluster_id == 0)
    assert row.action == cpe.ACTION_REPLACE
    item = plan.items[0]
    assert os.path.basename(item.path) == "house.m3u"
    assert item.exists and item.snapshot == (st.st_mtime_ns, st.st_size)
    assert item.lines[1] == "#PLAYLIST:House"
    assert plan.replace_count == 1 and plan.new_count == 2


def test_name_key_does_not_fold_sharp_s(tmp_path):
    assert cpe.name_key("Straße.m3u") != cpe.name_key("STRASSE.m3u")
    assert cpe.name_key("House.M3U") == cpe.name_key("house.m3u")
    lib, *_ = _lib(tmp_path)
    folder = Path(_summary(lib).run_folder)
    folder.mkdir(parents=True)
    (folder / "strasse.m3u").write_text("#EXTM3U\n", encoding="utf-8")
    plan = _plan(_summary(lib), names={0: "Straße", 1: "Strasse"})
    assert plan.ok
    actions = {r.cluster_id: r.action for r in plan.rows}
    assert actions[0] == cpe.ACTION_NEW and actions[1] == cpe.ACTION_REPLACE


# ── CP-19: noise ──────────────────────────────────────────────────────────


def test_noise_is_never_planned(tmp_path):
    lib, *_ = _lib(tmp_path, n=24, noise=4)
    summary = _summary(lib)
    assert summary.noise_count == 4 and -1 not in summary.clusters
    plan = _plan(summary, set(summary.clusters) | {-1})
    assert plan.rows[-1] == cpe.PlanRow(-1, "Unclustered", 4, 0, cpe.ACTION_SKIP, cpe.NOISE_REASON)
    assert len(plan.items) == 3
    assert not any(item.lines[2].endswith("cluster=-1") for item in plan.items)


# ── CP-20 / CP-21: run folders ────────────────────────────────────────────


def test_run_folder_name_kmeans_and_hdbscan(tmp_path):
    kmeans = run_block(20, 8, "/m")
    hdb = run_block(20, 3, "/m", method="hdbscan", min_cluster_size=5)
    assert cpe.run_folder_name(kmeans) == "2026-11-07 15.30.12 K-Means (k=8)"
    assert cpe.run_folder_name(hdb) == "2026-11-07 15.30.12 HDBSCAN (min 5)"
    for name in (cpe.run_folder_name(kmeans), cpe.run_folder_name(hdb)):
        assert ":" not in name and name.endswith(")")
    lib, *_ = _lib(tmp_path)
    summary = _summary(lib)
    assert summary.run_folder == os.path.join(
        os.path.abspath(lib), "Playlists", "Clusters", "2026-11-07 15.30.12 K-Means (k=3)")
    assert summary.run_folder_rel == "Playlists/Clusters/2026-11-07 15.30.12 K-Means (k=3)/"


def test_same_second_folder_of_another_run_gets_a_suffix(tmp_path):
    lib, _payload, members = _lib(tmp_path)
    clusters = lib / "Playlists" / "Clusters"
    base = cpe.run_folder_name(members["run"])

    def marked(folder: Path, run_id: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "Cluster 00.m3u").write_text(
            f"#EXTM3U\n#PLAYLIST:Cluster 00\n#ALPHADEX-CLUSTER:run={run_id};cluster=0\n", encoding="utf-8")

    marked(clusters / base, "20261107-153012-0000aa")
    assert os.path.basename(_summary(lib).run_folder) == f"{base} (2)"
    marked(clusters / f"{base} (2)", "20261107-153012-0000bb")
    assert os.path.basename(_summary(lib).run_folder) == f"{base} (3)"
    marked(clusters / f"{base} (3)", members["run"]["run_id"])     # this run's own files
    assert os.path.basename(_summary(lib).run_folder) == f"{base} (3)"

    other = tmp_path / "Other"
    other_lib, _p, other_members = _lib(tmp_path, name="Other")
    user_folder = other_lib / "Playlists" / "Clusters" / cpe.run_folder_name(other_members["run"])
    user_folder.mkdir(parents=True)
    (user_folder / "mine.m3u").write_text("#EXTM3U\nsong.mp3\n", encoding="utf-8")   # no marker
    assert os.path.basename(_summary(other_lib).run_folder) == user_folder.name
    assert other.exists()


def test_earlier_sets_are_listed_and_untouched(tmp_path):
    lib, *_ = _lib(tmp_path)
    earlier = lib / "Playlists" / "Clusters" / "2026-01-01 10.00.00 K-Means (k=4)"
    earlier.mkdir(parents=True)
    old = earlier / "Cluster 01.m3u"
    old.write_text("#EXTM3U\n#PLAYLIST:Cluster 01\n#ALPHADEX-CLUSTER:run=old;cluster=1\nx.mp3\n",
                   encoding="utf-8")
    before = _tree(earlier)
    summary = _summary(lib)
    assert summary.earlier_sets == [earlier.name]
    assert "1 earlier cluster playlist set is kept in Playlists/Clusters/." in summary.warnings
    result = cpe.write_cluster_playlists(_plan(summary))
    assert len(result["written"]) == 3 and not result["errors"]
    assert _tree(earlier) == before


def test_resave_same_run_replaces_and_leaves_old_names(tmp_path):
    lib, *_ = _lib(tmp_path)
    first_plan = _plan(_summary(lib))
    cpe.write_cluster_playlists(first_plan)
    first_names = sorted(os.path.basename(i.path) for i in first_plan.items)
    renamed_old = Path(first_plan.items[0].path)
    old_bytes = renamed_old.read_bytes()

    summary = _summary(lib)
    assert summary.run_folder == first_plan.run_folder           # same run, same folder
    assert sorted(e[0] for e in summary.existing_files.values()) == first_names
    plan = _plan(summary, names={0: "My favourites"})
    actions = {r.cluster_id: r.action for r in plan.rows}
    assert actions == {0: cpe.ACTION_NEW, 1: cpe.ACTION_REPLACE, 2: cpe.ACTION_REPLACE}
    assert plan.left_unchanged == [renamed_old.name]
    result = cpe.write_cluster_playlists(plan)
    assert len(result["written"]) == 3 and len(result["backups"]) == 2
    assert renamed_old.read_bytes() == old_bytes                  # left in place, unchanged
    assert (Path(plan.run_folder) / "My favourites.m3u").exists()


# ── CP-22 / CP-23: entries and header ─────────────────────────────────────


def test_entries_equal_plan_playlist_lines_relative_native(tmp_path):
    lib, *_ = _lib(tmp_path)
    plan = _plan(_summary(lib), order="library")
    for item in plan.items:
        entries = item.lines[3:]
        targets = [os.path.normpath(os.path.join(os.path.dirname(item.path), e)) for e in entries]
        assert entries == pg.plan_playlist(targets, item.path).lines
    first = plan.items[0].lines[3]
    assert first == os.path.join("..", "..", "..", "Artist 0", "Album", "track_00000.mp3")
    if os.sep == "\\":
        assert "/" not in first


def test_header_lines_and_entry_count(tmp_path):
    lib, *_ = _lib(tmp_path)
    summary = _summary(lib)
    plan = _plan(summary)
    assert [i.lines[2] for i in plan.items] == [
        f"#ALPHADEX-CLUSTER:run={summary.run_id};cluster={c}" for c in (0, 1, 2)
    ]
    for item, row in zip(plan.items, plan.rows):
        assert item.lines[0] == "#EXTM3U"
        assert item.lines[1] == f"#PLAYLIST:{row.name}"
        assert item.entries == len(item.lines) - 3 == row.tracks
        assert os.path.basename(item.path) == row.name + ".m3u"
        assert item.encoding == "utf-8" and not item.exists


# ── CP-46: the Windows path budget ────────────────────────────────────────


def test_long_library_root_drops_parts_then_trims_then_skips(tmp_path):
    # A library whose run folder leaves exactly 25 characters for a stem.
    folder_tail = len(os.path.join("", "Playlists", "Clusters", "2026-11-07 15.30.12 K-Means (k=3)")) + 1
    want_root = cpe.PATH_BUDGET - 25 - len(os.sep) - len(".m3u") - folder_tail
    pad = want_root - len(str(tmp_path)) - 1
    if pad < 1:
        pytest.skip("temporary folder already too deep for this layout")
    lib = tmp_path / ("L" * pad)
    assert len(str(lib)) == want_root
    payload, members = make_cluster_run(n=20, root=str(lib))
    _write(lib, payload, members)
    groups = _by_cluster()
    tags = {i: {"genre": "Deep House"} for i in groups[0]}
    summary = _summary(lib, tag_reader=_reader(tags))
    assert summary.clusters[0].name == "Cluster 00 - Deep House - ~90 BPM - low energy"

    plan = _plan(summary, names={2: "A very long custom playlist name indeed"})
    rows = {r.cluster_id: r for r in plan.rows}
    assert rows[0].name == "Cluster 00 - Deep House"            # parts dropped from the right
    assert rows[1].name == "Cluster 01 - ~100 BPM"               # fits as it is
    assert rows[2].name == "A very long custom playli"           # a user name is trimmed
    for item in plan.items:
        assert len(item.path) + len(cpe.BACKUP_SUFFIX) <= 259
    result = cpe.write_cluster_playlists(plan)
    assert len(result["written"]) == 3 and not result["errors"]

    # Deeper still: not even "Cluster 00" fits, so every row is skipped.
    deeper = dataclasses.replace(summary, run_folder=summary.run_folder + "x" * 20)
    plan = _plan(deeper)
    assert plan.items == []
    assert all(r.action == cpe.ACTION_SKIP and r.reason == cpe.TOO_LONG_REASON for r in plan.rows)


# ── CP-24 / CP-25: order and minimum size ─────────────────────────────────


def test_centre_and_library_orders(tmp_path):
    lib = tmp_path / "Library"
    payload, members = make_cluster_run(n=20, root=str(lib))
    groups = _by_cluster()
    for k, i in enumerate(groups[0]):            # cluster 0: the last track is the most typical
        members["centroid_distance"][i] = 10.0 - k
    members["centroid_distance"][groups[0][1]] = members["centroid_distance"][groups[0][2]]  # a tie
    _write(lib, payload, members)
    summary = _summary(lib)
    by_entry = {e: p for p, e, _d in summary.clusters[0].present}

    centre = [by_entry[e] for e in _plan(summary, order="centre").items[0].lines[3:]]
    dist = {p: d for p, _e, d in summary.clusters[0].present}
    assert centre == sorted(dist, key=lambda p: (dist[p], p.casefold()))
    assert _index(centre[0]) == groups[0][-1]

    library = [by_entry[e] for e in _plan(summary, order="library").items[0].lines[3:]]
    assert library == sorted(dist, key=str.casefold) and library != centre
    with pytest.raises(ValueError):
        _plan(summary, order="path")


def test_missing_members_are_dropped_from_the_order(tmp_path):
    lib, _payload, members = _lib(tmp_path)
    gone = members["tracks"][6]
    os.remove(gone)
    summary = _summary(lib)
    for order in cpe.ORDERS:
        item = _plan(summary, order=order).items[0]
        assert item.entries == 6
        assert not any(line.endswith("track_00006.mp3") for line in item.lines)


def test_min_size_counts_present_tracks_only(tmp_path):
    lib, _payload, members = _lib(tmp_path)
    for i in (2, 5):                             # cluster 2: 6 tracks, 4 still present
        os.remove(members["tracks"][i])
    summary = _summary(lib)
    row = next(r for r in _plan(summary).rows if r.cluster_id == 2)
    assert (row.action, row.reason) == (cpe.ACTION_SKIP, "too small (4 tracks)")
    assert len(_plan(summary).items) == 2
    row = next(r for r in _plan(summary, min_size=4).rows if r.cluster_id == 2)
    assert row.action == cpe.ACTION_NEW
    # Unticked clusters are skipped as "not selected"; ids beyond the run are ignored.
    plan = _plan(summary, {0, 99}, min_size=4)
    assert [r.reason for r in plan.rows] == ["", cpe.NOT_SELECTED_REASON, cpe.NOT_SELECTED_REASON]


# ── CP-28 / CP-39: the planner is pure and fast ───────────────────────────


def test_planner_does_no_file_io(tmp_path, monkeypatch):
    lib, *_ = _lib(tmp_path, noise=0)
    folder = Path(_summary(lib).run_folder)
    folder.mkdir(parents=True)
    (folder / "Cluster 01 - ~100 BPM.m3u").write_text("#EXTM3U\n", encoding="utf-8")
    summary = _summary(lib)

    def boom(*a, **k):
        raise AssertionError("the planner touched the file system")

    with monkeypatch.context() as m:
        for target, name in ((os.path, "exists"), (os.path, "isfile"), (os.path, "isdir"),
                             (os.path, "relpath"), (os, "stat"), (os, "scandir"),
                             (os, "listdir"), (builtins, "open")):
            m.setattr(target, name, boom)
        plans = [
            cpe.plan_cluster_playlists(summary, {0, 1, 2}, order=order, min_size=size, names=names)
            for order in cpe.ORDERS for size in (2, 7) for names in ({}, {0: "x/y"})
        ]
    assert all(p.ok for p in plans)
    assert any(p.replace_count == 1 for p in plans)


def test_replan_of_20k_members_is_fast():
    clusters = {}
    for cid in range(20):
        present = [(f"/m/a{cid}/t{i:05d}.mp3", f"../../../a{cid}/t{i:05d}.mp3", (i * 7919 % 1000) / 10)
                   for i in range(1000)]
        clusters[cid] = cpe.ClusterFacts(cid, f"Cluster {cid:02d}", present, [], ["Rock"], f"Cluster {cid:02d} - Rock")
    summary = cpe.RunSummary(
        library_root="/m", run={"run_id": "r", "method": "kmeans", "params": {"n_clusters": 20}},
        run_folder="/m/Playlists/Clusters/run", clusters=clusters,
    )
    t0 = time.perf_counter()
    plan = cpe.plan_cluster_playlists(summary, set(clusters), order="centre")
    elapsed = time.perf_counter() - t0
    assert len(plan.items) == 20 and sum(i.entries for i in plan.items) == 20_000
    assert elapsed < 0.5, f"re-plan took {elapsed:.3f}s"


# ── CP-27 / CP-28 / CP-30 / CP-31 / CP-36: writing ────────────────────────


def test_write_writes_exactly_the_plan(tmp_path):
    lib, *_ = _lib(tmp_path)
    summary = _summary(lib)
    plan = _plan(summary)
    before = _tree(lib)
    logs: list = []
    result = cpe.write_cluster_playlists(plan, log_callback=logs.append)
    assert result["refused"] is None and result["errors"] == [] and result["cancelled"] is False
    assert result["written"] == [i.path for i in plan.items]
    for item in plan.items:
        assert Path(item.path).read_bytes() == "".join(line + "\n" for line in item.lines).encode("utf-8")
    assert sorted(os.listdir(plan.run_folder)) == sorted(os.path.basename(i.path) for i in plan.items)
    added = set(_tree(lib)) - set(before)
    allowed = {i.path for i in plan.items} | {str(lib / "Docs" / "playlist_writes.log")}
    allowed |= {os.path.join(str(lib), "Playlists"), os.path.join(str(lib), "Playlists", "Clusters"),
                plan.run_folder}
    assert added <= allowed
    text, ok = cpe.result_message(plan, result)
    assert ok
    assert text == ("Saved 3 cluster playlists to Playlists/Clusters/2026-11-07 15.30.12 K-Means (k=3)/ "
                    "(0 replaced, backed up as .bak; 0 failed).")


def test_write_refuses_when_run_changed(tmp_path):
    lib, *_ = _lib(tmp_path)
    plan = _plan(_summary(lib))
    write_cluster_run(lib, files=True, run_id="20261108-090000-123456")   # a new analysis landed
    logs: list = []
    result = cpe.write_cluster_playlists(plan, log_callback=logs.append)
    assert result["refused"] == cpe.RUN_CHANGED_MESSAGE
    assert result["written"] == [] and not os.path.exists(plan.run_folder)
    assert not (lib / "Docs" / "playlist_writes.log").exists()
    assert cpe.result_message(plan, result) == (cpe.RUN_CHANGED_MESSAGE, False)
    assert any(cpe.RUN_CHANGED_MESSAGE in line for line in logs)


def test_write_uses_the_plans_library_root(tmp_path, monkeypatch):
    lib_a, *_ = _lib(tmp_path, name="A")
    lib_b, *_ = _lib(tmp_path, name="B", run_id="20261108-090000-bbbbbb")
    plan = _plan(_summary(lib_a))
    monkeypatch.chdir(lib_b)                 # nothing may depend on where we are
    result = cpe.write_cluster_playlists(plan)
    assert result["refused"] is None and len(result["written"]) == 3
    assert (lib_a / "Docs" / "playlist_writes.log").exists()
    assert not (lib_b / "Docs" / "playlist_writes.log").exists()
    assert not (lib_b / "Playlists").exists()


def test_replace_makes_bak_and_log_lines(tmp_path):
    lib, *_ = _lib(tmp_path)
    cpe.write_cluster_playlists(_plan(_summary(lib)))
    plan = _plan(_summary(lib), order="library")
    assert plan.replace_count == 3
    result = cpe.write_cluster_playlists(plan)
    assert len(result["backups"]) == 3
    for item, bak in zip(plan.items, result["backups"]):
        assert bak.startswith(item.path + ".") and bak.endswith(".bak")
        assert os.path.dirname(bak) == os.path.dirname(item.path)
    log = (lib / "Docs" / "playlist_writes.log").read_text(encoding="utf-8")
    assert log.count(" BACKUP ") == 3 and log.count(" WRITE ") == 6
    text, ok = cpe.result_message(plan, result)
    assert ok and "(3 replaced, backed up as .bak; 0 failed)" in text


def test_file_appearing_since_preview_is_skipped(tmp_path):
    lib, *_ = _lib(tmp_path)
    cpe.write_cluster_playlists(_plan(_summary(lib), {0}))        # cluster 0 exists now
    plan = _plan(_summary(lib))                                   # 0 = replace, 1 & 2 = new
    appeared = Path(plan.items[1].path)
    appeared.write_text("someone else's playlist\n", encoding="utf-8")
    changed = Path(plan.items[0].path)
    changed.write_text("edited after the preview\n", encoding="utf-8")
    result = cpe.write_cluster_playlists(plan)
    assert result["written"] == [plan.items[2].path]
    assert sorted(result["errors"]) == sorted([(str(appeared), "changed since the preview"),
                                               (str(changed), "changed since the preview")])
    assert appeared.read_text(encoding="utf-8") == "someone else's playlist\n"
    assert changed.read_text(encoding="utf-8") == "edited after the preview\n"
    text, ok = cpe.result_message(plan, result)
    assert not ok and "2 failed" in text


def test_cancel_between_files(tmp_path):
    lib, *_ = _lib(tmp_path)
    plan = _plan(_summary(lib))
    cancel = threading.Event()

    def log(line):
        if line.startswith("✓ Wrote"):
            cancel.set()

    result = cpe.write_cluster_playlists(plan, cancel_event=cancel, log_callback=log)
    assert result["cancelled"] is True
    assert result["written"] == [plan.items[0].path]
    assert not os.path.exists(plan.items[1].path)
    assert "WRITE" in (lib / "Docs" / "playlist_writes.log").read_text(encoding="utf-8")
    text, ok = cpe.result_message(plan, result)
    assert not ok and text.endswith("Cancelled before the rest were written.")
