"""Shared helpers for the Playlists / Clustered / Music Graph GUI tests.

* ``make_library`` — a tiny library of empty "audio" files, including the
  reserved folders, for workflows whose analysis is faked anyway.
* ``write_cluster_payload`` — fabricate ``Docs/cluster_info.json`` (valid,
  malformed, misaligned or large) and check it with ``cluster_graph_data``;
  a valid payload also gets a run block and a matching
  ``Docs/cluster_members.json`` (``with_members=True``).
* ``FakeClustering`` — a stand-in ``clustered_playlists`` module for worker
  tests (the worker imports it inside ``run()``), with a controllable gate so
  a test can hold a run "in extraction" and check cancellation. It writes the
  members file and returns the run block like the real engine
  (``write_members=True``).
"""
from __future__ import annotations

import json
import threading
import types
from pathlib import Path

import cluster_graph_data as cgd
from _helpers.cluster_payloads import members_for_payload

STANDARD_FILES = (
    "Artist A/Album 1/01 a.mp3",
    "Artist A/Album 1/02 b.mp3",
    "Artist A/Album 2/01 c.flac",
    "Artist B/Album X/01 d.mp3",
    "Quarantine/dupe.mp3",
    "Trash/x.mp3",
    "Not Sorted/n.mp3",
    "Docs/sample.wav",
)


def make_library(root: Path, files=STANDARD_FILES) -> Path:
    root = Path(root)
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\0")
    return root


def cluster_payload(n: int = 12, n_clusters: int = 3, noise: int = 0, tracks_root: str = "/music") -> dict:
    labels = [i % n_clusters for i in range(n)]
    for i in range(noise):
        labels[-(i + 1)] = -1
    return {
        "X_2d": [[float(i % 7), float(i // 7)] for i in range(n)],
        "X_3d": [[float(i), float(i), float(i)] for i in range(n)],
        "labels": labels,
        "tracks": [f"{tracks_root}/track_{i:05d}.mp3" for i in range(n)],
        "cluster_info": {str(c): {"size": labels.count(c)} for c in range(n_clusters)},
        "X_downsampled": False,
        "X_total_points": n,
    }


def write_cluster_payload(library: Path, kind: str = "valid", with_members: bool = True, **kw) -> Path:
    """Write ``Docs/cluster_info.json``; ``kind`` is valid|malformed|misaligned.

    A valid payload written ``with_members`` also carries a run block and gets
    a matching ``Docs/cluster_members.json`` (``_helpers.cluster_payloads``).
    """
    path = cgd.cluster_info_path(library)
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "malformed":
        path.write_text("{not json", encoding="utf-8")
        return path
    payload = cluster_payload(**kw)
    if kind == "misaligned":
        payload["labels"] = payload["labels"][:-1]
    if kind == "valid" and with_members:
        payload, members = members_for_payload(
            payload, root=kw.get("tracks_root", "/music"),
        )
        cgd.cluster_members_path(library).write_text(
            json.dumps(members, separators=(",", ":")), encoding="utf-8"
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    if kind == "valid":
        cgd.load_cluster_data(library)  # the helper itself must produce loadable data
    return path


class FakeClustering:
    """Fake ``clustered_playlists`` module. ``install(monkeypatch)`` puts it in sys.modules."""

    def __init__(self, n_clusters: int = 3, noise: int = 0, hold: bool = False,
                 write_members: bool = True) -> None:
        self.n_clusters = n_clusters
        self.noise = noise
        self.calls: list[dict] = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.saw_cancel = False
        self.write_payload = True
        self.write_members = write_members
        self.runs = 0

        class ClusteringCancelled(Exception):
            pass

        self.ClusteringCancelled = ClusteringCancelled

    def generate(self, tracks, root, method, params, log_callback=None, cancel_event=None,
                 progress_callback=None, **kw):
        import numpy as np

        self.calls.append({"method": method, "params": dict(params), "tracks": list(tracks),
                           "cancel_event": cancel_event, **kw})
        self.started.set()
        # Simulated extraction loop that honours cancel_event cooperatively.
        while not self.release.wait(0.02):
            if cancel_event is not None and cancel_event.is_set():
                self.saw_cancel = True
                raise self.ClusteringCancelled()
        if cancel_event is not None and cancel_event.is_set():
            self.saw_cancel = True
            raise self.ClusteringCancelled()
        if progress_callback is not None:
            progress_callback(len(tracks), len(tracks))
        if log_callback:
            log_callback(f"fake extraction of {len(tracks)} tracks")
        n = len(tracks)
        labels = np.array([i % self.n_clusters for i in range(n)])
        for i in range(min(self.noise, n)):
            labels[-(i + 1)] = -1
        X = np.random.RandomState(0).rand(n, 4) + labels[:, None] * 3.0
        info = {int(c): {"size": int((labels == c).sum())} for c in set(labels.tolist()) if c >= 0}
        self.runs += 1
        run = {
            "schema": 1, "run_id": f"20261107-1530{self.runs:02d}-fa{self.runs:04x}",
            "created": f"2026-11-07T15:30:{self.runs % 60:02d}", "library_root": str(root),
            "method": method, "params": dict(params), "features": "mfcc-tempo",
            "feature_engine": "librosa", "normalization": "standard",
            "track_count": n, "failed_count": 0,
        }
        result = {"X": X, "labels": labels, "tracks": list(tracks), "cluster_info": info,
                  "features": [], "failed": {}}
        if self.write_payload:
            docs = Path(root) / "Docs"
            docs.mkdir(parents=True, exist_ok=True)
            payload = {
                "X_2d": X[:, :2].tolist(), "X_3d": X[:, :3].tolist(),
                "labels": labels.tolist(), "tracks": list(tracks),
                "cluster_info": {str(k): v for k, v in info.items()},
            }
            if self.write_members:
                distances = [None] * n
                for c in set(labels.tolist()):
                    if c < 0:
                        continue
                    idx = np.flatnonzero(labels == c)
                    d = np.linalg.norm(X[idx] - X[idx].mean(axis=0), axis=1)
                    for i, v in zip(idx.tolist(), d.tolist()):
                        distances[i] = float(v)
                members = {"run": run, "tracks": list(tracks), "labels": labels.tolist(),
                           "centroid_distance": distances, "failed": []}
                (docs / "cluster_members.json").write_text(json.dumps(members), encoding="utf-8")
                payload["run"] = run
                result["run"] = run
                result["members_file"] = str(docs / "cluster_members.json")
            (docs / "cluster_info.json").write_text(json.dumps(payload), encoding="utf-8")
        return result

    def module(self) -> types.ModuleType:
        mod = types.ModuleType("clustered_playlists")
        mod.generate_clustered_playlists = self.generate
        mod.ClusteringCancelled = self.ClusteringCancelled
        return mod

    def install(self, monkeypatch) -> "FakeClustering":
        import sys

        monkeypatch.setitem(sys.modules, "clustered_playlists", self.module())
        return self
