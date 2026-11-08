"""Fabricated clustering runs for the cluster-playlist tests.

``make_cluster_run`` returns an index-aligned ``(payload, members)`` pair:
the map payload (``Docs/cluster_info.json``, optionally downsampled the way
``clustered_playlists.build_cluster_payload`` does it) and the full
membership (``Docs/cluster_members.json``), both carrying the same ``run``
block. ``write_cluster_run`` writes them (valid, or broken in one specific
way) and can create the member files on disk so presence checks pass.

Both builders check their own output through ``cluster_graph_data``, so a
test can never pass because the fake itself was malformed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cluster_graph_data as cgd

DEFAULT_RUN_ID = "20261107-153012-abc123"


def run_block(n: int, n_clusters: int, root: str, run_id: str = DEFAULT_RUN_ID,
              method: str = "kmeans", failed_count: int = 0, **extra) -> dict:
    params = {"n_clusters": n_clusters} if method == "kmeans" else {
        "min_cluster_size": extra.pop("min_cluster_size", 5), "min_samples": 5}
    run = {
        "schema": 1,
        "run_id": run_id,
        "created": "2026-11-07T15:30:12",
        "library_root": str(root),
        "method": method,
        "params": params,
        "features": "mfcc-tempo-energy",
        "feature_engine": "librosa",
        "normalization": "standard",
        "track_count": n,
        "failed_count": failed_count,
    }
    run.update(extra)
    return run


def track_path(root: str, i: int) -> str:
    return os.path.join(str(root), f"Artist {i % 4}", "Album", f"track_{i:05d}.mp3")


def _downsample_indices(n: int, m: int) -> list[int]:
    if m >= n:
        return list(range(n))
    return sorted({round(k * (n - 1) / (m - 1)) for k in range(m)})


def make_cluster_run(
    n: int = 20,
    n_clusters: int = 3,
    noise: int = 0,
    root: str = "/music",
    downsample_to: int | None = None,
    tempo: bool = True,
    energy: bool = True,
    run_id: str = DEFAULT_RUN_ID,
    method: str = "kmeans",
    labels: list | None = None,
    tracks: list | None = None,
    failed: list | None = None,
) -> tuple[dict, dict]:
    """An aligned ``(payload, members)`` pair.

    Labels are ``i % n_clusters`` with the last ``noise`` tracks unclustered;
    centre distances are ``i * 0.1`` (``None`` for noise). Tempo is
    ``90 + 10 * label`` BPM and energy ``label + 1 + 0.0001 * i``, so with
    three clusters cluster 0 is "low energy", cluster 2 "high energy" and
    cluster 1 neither (no float ties at the terciles).
    """
    if labels is None:
        labels = [i % n_clusters for i in range(n)]
        for k in range(min(noise, n)):
            labels[-(k + 1)] = -1
    n = len(labels)
    tracks = list(tracks) if tracks is not None else [track_path(root, i) for i in range(n)]
    failed = list(failed or [])
    run = run_block(n, n_clusters, root, run_id=run_id, method=method, failed_count=len(failed))
    members = {
        "run": run,
        "tracks": tracks,
        "labels": list(labels),
        "centroid_distance": [None if lab == -1 else round(i * 0.1, 6) for i, lab in enumerate(labels)],
        "failed": failed,
    }
    if tempo:
        members["tempo_bpm"] = [None if lab == -1 else 90.0 + 10 * lab for lab in labels]
    if energy:
        members["energy"] = [None if lab == -1 else lab + 1 + 0.0001 * i for i, lab in enumerate(labels)]

    indices = list(range(n))
    downsampled = downsample_to is not None and downsample_to < n
    if downsampled:
        indices = _downsample_indices(n, downsample_to)
    payload = {
        "X": [[float(i), 0.0] for i in indices],
        "X_2d": [[float(i % 7), float(i // 7)] for i in indices],
        "X_3d": [[float(i), float(i), float(i)] for i in indices],
        "labels": [labels[i] for i in indices],
        "tracks": [tracks[i] for i in indices],
        "cluster_info": {str(c): {"size": labels.count(c)} for c in sorted(set(labels)) if c >= 0},
        "X_downsampled": downsampled,
        "X_total_points": n,
        "X_indices": indices if downsampled else None,
        "run": dict(run),
    }
    cgd.validate_cluster_payload(payload)
    cgd.validate_cluster_members(members)
    cgd.check_members_match_payload(payload, members)
    return payload, members


KINDS = ("valid", "legacy", "mismatched_run", "misaligned_members", "all_noise")


def write_cluster_run(library, kind: str = "valid", files: bool = False, **kw) -> tuple[dict, dict]:
    """Write ``Docs/cluster_info.json`` and ``Docs/cluster_members.json``.

    ``kind``: ``valid``; ``legacy`` (a map without a run block, no members
    file); ``mismatched_run`` (the members file is from another run);
    ``misaligned_members`` (members labels one short); ``all_noise`` (every
    track unclustered). ``files=True`` also creates every member file under
    the library. Paths default to live under ``library``.
    """
    if kind not in KINDS:
        raise ValueError(kind)
    library = Path(library)
    kw.setdefault("root", str(library))
    if kind == "all_noise":
        kw["noise"] = kw.get("n", 20)
    payload, members = make_cluster_run(**kw)
    docs = library / "Docs"
    docs.mkdir(parents=True, exist_ok=True)
    if kind == "legacy":
        payload = {k: v for k, v in payload.items() if k != "run"}
    elif kind == "mismatched_run":
        members = json.loads(json.dumps(members))
        members["run"]["run_id"] = "20261106-101010-ffffff"
    elif kind == "misaligned_members":
        members = json.loads(json.dumps(members))
        members["labels"] = members["labels"][:-1]
    (docs / "cluster_info.json").write_text(json.dumps(payload), encoding="utf-8")
    if kind != "legacy":
        (docs / "cluster_members.json").write_text(
            json.dumps(members, separators=(",", ":")), encoding="utf-8"
        )
    cgd.load_cluster_data(library)   # the map payload always loads
    if files:
        make_member_files(members["tracks"])
    return payload, members


def make_member_files(tracks) -> None:
    for t in tracks:
        p = Path(t)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(b"\0")


def members_for_payload(payload: dict, run_id: str = DEFAULT_RUN_ID, root: str = "/music") -> tuple[dict, dict]:
    """Add a run block to an existing (not downsampled) map payload and build
    the matching members file (for ``playlists_helpers.write_cluster_payload``)."""
    labels = [int(v) for v in payload["labels"]]
    n_clusters = len({v for v in labels if v >= 0}) or 1
    new_payload, members = make_cluster_run(
        labels=labels, tracks=list(payload["tracks"]), n_clusters=n_clusters,
        root=root, run_id=run_id,
    )
    payload = dict(payload)
    payload["run"] = dict(members["run"])
    payload.setdefault("X_downsampled", False)
    payload.setdefault("X_total_points", len(labels))
    cgd.check_members_match_payload(payload, members)
    return payload, members
