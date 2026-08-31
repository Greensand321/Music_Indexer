"""Data layer for the in-app cluster graph.

Deliberately free of Qt, pyqtgraph and numpy so the loading, validation,
alignment and export rules can be exercised headlessly — the graph widgets
stay display-only, per the "GUI <-> backend separation" rule in CLAUDE.md.

The payload this reads is ``Docs/cluster_info.json``, written by
``clustered_playlists.generate_clustered_playlists``. Its arrays are
**index-aligned**: point *i* of ``X_2d``/``X_3d`` belongs to ``labels[i]`` and
``tracks[i]``. Everything here exists to keep that promise honest, because a
misalignment silently maps a selected dot to the wrong song.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

#: Cluster id used by HDBSCAN for points that belong to no cluster.
NOISE_LABEL = -1

#: Keys a usable payload must carry.
REQUIRED_KEYS = ("labels", "tracks")


class ClusterDataError(ValueError):
    """Raised when cluster_info.json is missing, unreadable or inconsistent."""


def cluster_info_path(library_path: str | os.PathLike[str]) -> Path:
    """Return the expected location of ``cluster_info.json`` for a library."""
    return Path(library_path) / "Docs" / "cluster_info.json"


def load_cluster_data(library_path: str | os.PathLike[str]) -> dict:
    """Load and validate the cluster payload for *library_path*.

    Raises
    ------
    ClusterDataError
        If the file is absent, unparseable, or internally inconsistent.
    """
    path = cluster_info_path(library_path)
    if not path.is_file():
        raise ClusterDataError(
            f"No cluster data at {path}. Run Clustered Playlists first."
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ClusterDataError(f"Could not read {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ClusterDataError(f"{path} does not contain a JSON object.")

    validate_cluster_payload(data)
    return data


def validate_cluster_payload(data: dict) -> None:
    """Check that a cluster payload is present, non-empty and index-aligned.

    The alignment check is the important one: coordinates, labels and tracks are
    consumed positionally, so a length mismatch means every downstream lookup
    (tooltip, selection, export) points at the wrong track.
    """
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ClusterDataError(
            f"cluster data is missing required key(s): {', '.join(missing)}"
        )

    coords = _raw_coords(data)
    if coords is None:
        raise ClusterDataError(
            "cluster data has neither 'X_2d' nor 'X_3d' coordinates to plot."
        )
    if not coords:
        raise ClusterDataError("cluster data contains no points to plot.")

    labels = data["labels"]
    tracks = data["tracks"]
    if not (len(coords) == len(labels) == len(tracks)):
        raise ClusterDataError(
            "cluster data is misaligned — coordinates, labels and tracks must "
            f"be the same length (got {len(coords)}, {len(labels)}, "
            f"{len(tracks)}). Re-run Clustered Playlists to regenerate it."
        )


def _raw_coords(data: dict) -> list | None:
    """Return the best available coordinate array, preferring the 2D embedding.

    Returns ``None`` only when no coordinate key is present at all; a key that
    exists but is empty yields ``[]`` so callers can tell "this payload has no
    coordinates" apart from "this payload has no points", which are different
    problems with different fixes.
    """
    for key in ("X_2d", "X_3d", "X"):
        coords = data.get(key)
        if coords:
            return coords
    for key in ("X_2d", "X_3d", "X"):
        if isinstance(data.get(key), list):
            return []
    return None


def graph_points(data: dict) -> Tuple[List[Tuple[float, float]], List[int], List[str]]:
    """Return ``(xy, labels, tracks)`` ready for plotting.

    ``X_2d`` is used when present — ``clustered_playlists`` computes it
    specifically for the flat scatter plot. Older payloads that predate it fall
    back to the first two components of ``X_3d``, which is a usable projection
    rather than an error.
    """
    validate_cluster_payload(data)
    raw = _raw_coords(data) or []

    xy: List[Tuple[float, float]] = []
    for point in raw:
        if len(point) < 2:
            raise ClusterDataError(
                f"coordinate {point!r} has fewer than two dimensions."
            )
        xy.append((float(point[0]), float(point[1])))

    labels = [int(v) for v in data["labels"]]
    tracks = [str(t) for t in data["tracks"]]
    return xy, labels, tracks


def normalize_cluster_info(raw: dict | None) -> Dict[int, dict]:
    """Return per-cluster info keyed by **int**.

    JSON object keys are always strings, so a payload round-tripped through
    ``cluster_info.json`` arrives as ``{"0": {...}}`` while the labels it
    describes are integers. Looking those up with an int key silently returned
    nothing, which is why per-cluster genre/tempo detail never appeared in the
    legend. Non-numeric keys are dropped rather than guessed at.
    """
    out: Dict[int, dict] = {}
    for key, value in (raw or {}).items():
        try:
            out[int(key)] = value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            continue
    return out


def cluster_counts(labels: Sequence[int]) -> Dict[int, int]:
    """Return ``{cluster_id: number_of_tracks}``, noise included."""
    counts: Dict[int, int] = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    return counts


def cluster_order(labels: Sequence[int]) -> List[int]:
    """Return cluster ids in display order: real clusters ascending, noise last."""
    unique = {int(v) for v in labels}
    real = sorted(v for v in unique if v != NOISE_LABEL)
    return real + ([NOISE_LABEL] if NOISE_LABEL in unique else [])


def summarize(labels: Sequence[int]) -> str:
    """Return a one-line human summary of a clustering."""
    counts = cluster_counts(labels)
    n_noise = counts.get(NOISE_LABEL, 0)
    n_clusters = len([c for c in counts if c != NOISE_LABEL])
    total = sum(counts.values())
    parts = [f"{total} tracks", f"{n_clusters} clusters"]
    if n_noise:
        parts.append(f"{n_noise} unclustered")
    return " · ".join(parts)


def selected_tracks(tracks: Sequence[str], indices: Iterable[int]) -> List[str]:
    """Return the track paths for *indices*, in stable ascending order.

    Selections are held in a set for cheap membership tests; sorting here keeps
    exports reproducible instead of depending on set iteration order.
    """
    out: List[str] = []
    for i in sorted({int(i) for i in indices}):
        if 0 <= i < len(tracks):
            out.append(tracks[i])
    return out


def selection_to_csv(paths: Sequence[str]) -> str:
    """Return CSV text (with header) listing the selected track paths."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["track_path"])
    for p in paths:
        writer.writerow([p])
    return buf.getvalue()


def selection_to_m3u(paths: Sequence[str], playlist_path: str | os.PathLike[str]) -> str:
    """Return ``.m3u`` text for *paths*, written relative to *playlist_path*.

    Relative entries keep the playlist portable if the library is moved, which
    is the same convention ``playlist_generator`` uses.
    """
    base = Path(playlist_path).parent
    lines = ["#EXTM3U"]
    for p in paths:
        try:
            lines.append(os.path.relpath(p, base))
        except ValueError:
            # Different drive on Windows — fall back to the absolute path.
            lines.append(str(p))
    return "\n".join(lines) + "\n"
