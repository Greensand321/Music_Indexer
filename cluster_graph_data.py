"""Data layer for the in-app cluster graph.

Deliberately free of Qt, pyqtgraph and numpy so the loading, validation,
alignment and export rules can be exercised headlessly — the graph widgets
stay display-only, per the "GUI <-> backend separation" rule in CLAUDE.md.

The payload this reads is ``Docs/cluster_info.json``, written by
``clustered_playlists.generate_clustered_playlists``. Its arrays are
**index-aligned**: point *i* of ``X_2d``/``X_3d`` belongs to ``labels[i]`` and
``tracks[i]``. Everything here exists to keep that promise honest, because a
misalignment silently maps a selected dot to the wrong song.

It also reads ``Docs/cluster_members.json`` — the run's full, never
downsampled membership that the cluster playlist export works from — and
checks that it belongs to the same run as the map payload (both carry the
same ``run`` block; see ``docs/specs/cluster_playlists.md`` §4). The same
index-alignment rule applies to that file.
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


# ── Full membership (Docs/cluster_members.json) ──────────────────────────

#: Keys a usable members file must carry (``run`` is checked separately).
MEMBERS_REQUIRED_KEYS = ("tracks", "labels")

_REANALYSE = "Run the analysis again to rebuild it."


def cluster_members_path(library_path: str | os.PathLike[str]) -> Path:
    """Return the expected location of ``cluster_members.json`` for a library."""
    return Path(library_path) / "Docs" / "cluster_members.json"


def load_cluster_members(library_path: str | os.PathLike[str]) -> dict:
    """Load and validate the full cluster membership for *library_path*.

    Raises
    ------
    ClusterDataError
        If the file is missing, unparseable, not a JSON object, lacks ``run``,
        ``run.run_id``, ``run.track_count``, ``tracks`` or ``labels``, or fails
        :func:`validate_cluster_members`.
    """
    path = cluster_members_path(library_path)
    if not path.is_file():
        raise ClusterDataError(
            f"No cluster membership at {path}. Run an analysis first."
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ClusterDataError(f"Could not read {path}: {exc}. {_REANALYSE}") from exc
    if not isinstance(data, dict):
        raise ClusterDataError(f"{path} does not contain a JSON object. {_REANALYSE}")
    run = data.get("run")
    missing = []
    if not isinstance(run, dict):
        missing.append("run")
    else:
        missing.extend(f"run.{k}" for k in ("run_id", "track_count") if k not in run)
    missing.extend(k for k in MEMBERS_REQUIRED_KEYS if k not in data)
    if missing:
        raise ClusterDataError(
            f"{path} is missing required key(s): {', '.join(missing)}. {_REANALYSE}"
        )
    validate_cluster_members(data)
    return data


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_cluster_members(data: dict) -> None:
    """Check a members file is index-aligned and internally consistent.

    ``tracks``, ``labels`` and ``centroid_distance`` must all be
    ``run.track_count`` long (as ``tempo_bpm`` / ``energy`` when present);
    labels must be ints (a bool is not); a clustered track must have a
    distance and a noise track must not.
    """
    run = data.get("run") if isinstance(data.get("run"), dict) else {}
    count = run.get("track_count")
    if not _is_int(count) or count < 0:
        raise ClusterDataError(f"cluster membership has an invalid track count ({count!r}). {_REANALYSE}")
    for key in ("tracks", "labels", "centroid_distance"):
        value = data.get(key)
        if not isinstance(value, list) or len(value) != count:
            got = len(value) if isinstance(value, list) else "missing"
            raise ClusterDataError(
                f"cluster membership is misaligned: '{key}' has {got} entries, "
                f"expected {count}. {_REANALYSE}"
            )
    for key in ("tempo_bpm", "energy"):
        if key in data and (not isinstance(data[key], list) or len(data[key]) != count):
            got = len(data[key]) if isinstance(data[key], list) else "not a list"
            raise ClusterDataError(
                f"cluster membership is misaligned: '{key}' has {got} entries, "
                f"expected {count}. {_REANALYSE}"
            )
    for i, (label, dist) in enumerate(zip(data["labels"], data["centroid_distance"])):
        if not _is_int(label):
            raise ClusterDataError(
                f"cluster membership has a non-integer label at {i} ({label!r}). {_REANALYSE}"
            )
        if label == NOISE_LABEL:
            if dist is not None:
                raise ClusterDataError(
                    f"cluster membership gives an unclustered track a centre distance (at {i}). {_REANALYSE}"
                )
        elif not _is_number(dist):
            raise ClusterDataError(
                f"cluster membership has no centre distance for a clustered track (at {i}). {_REANALYSE}"
            )
    failed = data.get("failed", [])
    if not isinstance(failed, list):
        raise ClusterDataError(f"cluster membership's 'failed' is not a list. {_REANALYSE}")


_OUT_OF_STEP = (
    "The saved cluster membership is out of step with the map (they come from "
    "different analyses). Re-run the analysis."
)


def check_members_match_payload(payload: dict, members: dict) -> None:
    """Raise :class:`ClusterDataError` unless *members* describes the same run
    as the map *payload*. Compares the stored paths (before any rebasing).

    The run ids must match, and the map's points must be the members' points:
    the same arrays when the map is not downsampled, otherwise the members at
    ``X_indices``.
    """
    p_run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    m_run = members.get("run") if isinstance(members.get("run"), dict) else {}
    if not p_run.get("run_id") or p_run.get("run_id") != m_run.get("run_id"):
        raise ClusterDataError(_OUT_OF_STEP)
    p_tracks = list(payload.get("tracks") or [])
    p_labels = list(payload.get("labels") or [])
    m_tracks = members["tracks"]
    m_labels = members["labels"]
    if not payload.get("X_downsampled"):
        if p_tracks != m_tracks or p_labels != m_labels:
            raise ClusterDataError(_OUT_OF_STEP)
        return
    indices = payload.get("X_indices")
    if (
        payload.get("X_total_points") != m_run.get("track_count")
        or not isinstance(indices, list)
        or len(indices) != len(p_tracks)
        or len(p_labels) != len(p_tracks)
    ):
        raise ClusterDataError(_OUT_OF_STEP)
    n = len(m_tracks)
    for pos, idx in enumerate(indices):
        if not _is_int(idx) or not 0 <= idx < n:
            raise ClusterDataError(_OUT_OF_STEP)
        if m_tracks[idx] != p_tracks[pos] or m_labels[idx] != p_labels[pos]:
            raise ClusterDataError(_OUT_OF_STEP)


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
