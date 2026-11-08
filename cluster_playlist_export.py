"""Save every cluster of a clustering run as a playlist — preview first.

Specification: ``docs/specs/cluster_playlists.md`` (requirements CP-n). Qt-free
and numpy-free, like :mod:`cluster_graph_data`, so every rule is testable
headlessly. Three steps, each its own function:

1. :func:`summarize_cluster_run` — **read-only**, cancellable, run as a job.
   Loads ``Docs/cluster_info.json`` and the full membership
   ``Docs/cluster_members.json``, refuses data that does not belong together,
   maps a moved library, checks which members are still present, counts
   tracks added since the analysis, samples tags to name each cluster, picks
   the run folder under ``Playlists/Clusters/`` and precomputes every present
   member's playlist entry.
2. :func:`plan_cluster_playlists` — **pure** (no file-system access at all):
   names, order, minimum size and the path budget turn the summary into dialog
   rows plus the exact :class:`playlist_generator.PlannedPlaylist` items.
3. :func:`write_cluster_playlists` — writes ``plan.items`` and nothing else,
   through :func:`playlist_generator.write_playlist_plan` (backups beside
   replaced files, ``Docs/playlist_writes.log``, ``verify=True``).

Nothing here writes outside step 3, and step 3 refuses a plan whose analysis
has been replaced since the preview.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, NamedTuple

import cluster_graph_data as cgd
import playlist_generator as pg
from utils.path_helpers import ensure_long_path
from utils.reserved_folders import is_reserved_path, iter_library_files

# ── Constants ─────────────────────────────────────────────────────────────

#: Audio files the analysis looks at. Must equal the analysis worker's set
#: (``gui.workspaces.clustered_enhanced._AUDIO_EXTS``); a test pins the two.
AUDIO_EXTS = frozenset({".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wav", ".opus"})

#: Where the run folders live, relative to the library.
CLUSTERS_DIR = ("Playlists", "Clusters")

#: Third header line of every generated file: ``#ALPHADEX-CLUSTER:run=…;cluster=…``.
MARKER_PREFIX = "#ALPHADEX-CLUSTER:"

#: Order options (key → dialog label). ``centre`` is the default (Q3).
ORDERS = {
    "centre": "Closest to the centre first",
    "library": "Library order",
}
DEFAULT_ORDER = "centre"

#: Minimum playlist size, counted in present tracks (Q4). The default matches
#: ``clustered_playlists.DEFAULT_HDBSCAN_MIN_SIZE``; a test pins the two.
DEFAULT_MIN_SIZE = 5
MIN_SIZE_RANGE = (2, 500)

#: Longest playlist stem (CP-17); the path budget (CP-46) may lower it.
NAME_LIMIT = 100
#: Longest path Windows opens without long-path support.
MAX_PATH = 259
#: The backup ``write_playlist_plan`` may create beside a replaced file.
BACKUP_SUFFIX = ".YYYYmmdd-HHMMSS-99.bak"
#: Longest allowed ``item.path`` so the file *and* its backup stay openable.
PATH_BUDGET = MAX_PATH - len(BACKUP_SUFFIX)

#: Reserved Windows device names (compared case-insensitively, CP-17).
_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{i}" for i in range(10)}
    | {f"LPT{i}" for i in range(10)}
)
_ILLEGAL = re.compile(r'[<>:"/\\|?*]')
_SPACES = re.compile(r"\s+")

#: How often the long loops poll ``cancel_event`` (CP-26).
_CANCEL_EVERY = 200

LEGACY_MESSAGE = (
    "This map was made before cluster playlists existed. Re-run the analysis to "
    "save a playlist per cluster. (Export playlist… still works for a selection.)"
)
ALL_NOISE_MESSAGE = (
    "This analysis put every track in 'Unclustered', so there is no cluster to "
    "save. Try a smaller minimum cluster size."
)
ALL_MISSING_MESSAGE = (
    "None of the analysed tracks are where they were; re-run the analysis."
)
RUN_CHANGED_MESSAGE = (
    "The analysis changed since the preview; open Save cluster playlists… again."
)
TOO_LONG_REASON = "path too long for Windows; move the library to a shorter folder"
NOT_SELECTED_REASON = "not selected"
NOISE_REASON = "not saved"

NOISE_LABEL = cgd.NOISE_LABEL

ACTION_NEW = "new"
ACTION_REPLACE = "replace"
ACTION_SKIP = "skip"


class ClusterPlaylistsCancelled(Exception):
    """The summary was cancelled (``cancel_event`` was set)."""


# ── Data ──────────────────────────────────────────────────────────────────


@dataclass
class ClusterFacts:
    """What the summary knows about one real cluster."""

    cluster_id: int
    #: ``"Cluster 03"``: the id zero-padded to the widest id in the run.
    label: str
    #: Present members as ``(path, playlist entry, centroid_distance)``.
    present: list = field(default_factory=list)
    #: Members left out: missing on disk or inside a reserved folder.
    missing: list = field(default_factory=list)
    #: Descriptor parts after the label, e.g. ``["Deep House", "~120 BPM"]``.
    parts: list = field(default_factory=list)
    #: The proposed (sanitised) name.
    name: str = ""


@dataclass
class RunSummary:
    """Output of :func:`summarize_cluster_run` (read-only facts)."""

    library_root: str
    run: dict = field(default_factory=dict)
    run_folder: str = ""
    rebased_from: str | None = None
    clusters: dict = field(default_factory=dict)       # id -> ClusterFacts
    noise_count: int = 0
    missing_total: int = 0
    new_tracks: int = 0
    map_points: int = 0
    map_total: int = 0
    downsampled: bool = False
    #: name key -> (exact file name, (st_mtime_ns, st_size)) of the ``.m3u``
    #: files already in the run folder.
    existing_files: dict = field(default_factory=dict)
    #: Existing files no proposed name targets (they stay as they are).
    left_unchanged: list = field(default_factory=list)
    #: Other run folders under ``Playlists/Clusters/`` (kept, never touched).
    earlier_sets: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    blocking_error: str | None = None

    @property
    def run_id(self) -> str:
        return str(self.run.get("run_id") or "")

    @property
    def run_folder_rel(self) -> str:
        """The run folder as the user sees it: ``Playlists/Clusters/<name>/``."""
        if not self.run_folder:
            return ""
        return "/".join((*CLUSTERS_DIR, os.path.basename(self.run_folder))) + "/"

    def cluster_ids(self) -> list:
        return sorted(self.clusters)


class PlanRow(NamedTuple):
    """One dialog row: ``(cluster_id, name, tracks, missing, action, reason)``."""

    cluster_id: int
    name: str
    tracks: int
    missing: int
    action: str
    reason: str


@dataclass
class ClusterPlaylistPlan:
    """Output of :func:`plan_cluster_playlists`: what Save would write."""

    run_id: str
    library_root: str
    run_folder: str
    rows: list = field(default_factory=list)
    items: list = field(default_factory=list)          # PlannedPlaylist
    errors: list = field(default_factory=list)         # str
    ok: bool = True
    left_unchanged: list = field(default_factory=list)

    @property
    def replace_count(self) -> int:
        return sum(1 for item in self.items if item.exists)

    @property
    def new_count(self) -> int:
        return sum(1 for item in self.items if not item.exists)


# ── Small pure helpers ────────────────────────────────────────────────────


def name_key(file_name: str) -> str:
    """Identity of a file name for "same file?" checks (§3).

    ``lower()`` rather than ``casefold()``: it matches how NTFS/APFS compare
    names (``casefold`` would make ``Straße`` and ``Strasse`` one file).
    """
    return file_name.lower()


def sanitize_playlist_stem(text: str, limit: int = NAME_LIMIT, *, fallback: str = "") -> str:
    """A file stem that is legal on Windows, macOS and Linux (CP-17).

    In order: NFKC-normalise; drop control characters; ``<>:"/\\|?*`` → ``-``;
    collapse whitespace; trim to ``limit``; strip leading/trailing spaces and
    dots; append ``_`` when the part before the first dot is a Windows device
    name (``CON`` → ``CON_``, ``nul.mix`` → ``nul.mix_``). An empty result
    becomes ``fallback`` (the planner passes ``Cluster <id>``).
    """
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
    text = _ILLEGAL.sub("-", text)
    text = _SPACES.sub(" ", text)
    limit = max(0, int(limit))
    text = text[:limit]
    text = text.strip(" .")
    head = text.split(".", 1)[0].rstrip(" ")
    if head.upper() in _DEVICE_NAMES:
        if len(text) + 1 > limit:
            text = text[: max(0, limit - 1)].rstrip(" .")
        text += "_"
    return text or fallback


def _fit_generated(label: str, parts: list, limit: int) -> str:
    """A generated name inside ``limit``: whole descriptor parts are dropped
    from the right before the stem is ever hard-trimmed (CP-17/CP-46)."""
    for k in range(len(parts), -1, -1):
        candidate = sanitize_playlist_stem(" - ".join([label, *parts[:k]]), limit=10_000)
        if len(candidate) <= limit:
            return candidate or label
    return sanitize_playlist_stem(label, limit=limit, fallback=label[:limit])


def method_label(run: dict) -> str:
    """``K-Means (k=8)`` / ``HDBSCAN (min 5)``: the method and its parameter."""
    method = str(run.get("method") or "")
    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    if method == "kmeans":
        label, value, fmt = "K-Means", params.get("n_clusters"), " (k={})"
    elif method == "hdbscan":
        label, value, fmt = "HDBSCAN", params.get("min_cluster_size"), " (min {})"
    else:
        label, value, fmt = sanitize_playlist_stem(method, fallback="Clusters"), None, ""
    suffix = ""
    if value is not None:
        try:
            suffix = fmt.format(int(value))
        except (TypeError, ValueError):
            suffix = ""
    return f"{label}{suffix}"


def run_folder_name(run: dict) -> str:
    """``<YYYY-MM-DD HH.MM.SS> <Method><param>`` (CP-20), e.g.
    ``2026-11-07 15.30.12 K-Means (k=8)``. No colons; legal on Windows."""
    created = str(run.get("created") or "")
    try:
        stamp = datetime.fromisoformat(created).strftime("%Y-%m-%d %H.%M.%S")
    except ValueError:
        stamp = sanitize_playlist_stem(str(run.get("run_id") or "run"), fallback="run")
    return f"{stamp} {method_label(run)}"


def _cluster_label(cluster_id: int, width: int) -> str:
    return f"Cluster {cluster_id:0{width}d}"


def _id_width(cluster_ids: Iterable[int]) -> int:
    ids = [c for c in cluster_ids if c >= 0]
    return max(2, len(str(max(ids)))) if ids else 2


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _same_root(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _rebase(path: str, old_root: str, new_root: str) -> str:
    """``path`` under ``old_root`` mapped to ``new_root`` (CP-10); paths not
    under it, or on another Windows drive, are kept as stored."""
    try:
        rel = os.path.relpath(path, old_root)
    except ValueError:
        return path
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return path
    return os.path.join(new_root, rel)


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ClusterPlaylistsCancelled()


def _default_tag_reader(path: str) -> dict:
    # Looked up at call time so tests can monkeypatch ``read_tags`` (or
    # ``read_metadata`` underneath it).
    from utils import audio_metadata_reader as amr
    return amr.read_tags(path)


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return str(value).strip()


def _dominant(values: list) -> list:
    """``[(display spelling, count)]`` counted case-insensitively, most
    frequent first (ties alphabetically by casefold)."""
    counts: dict = {}
    spellings: dict = {}
    for value in values:
        key = value.casefold()
        counts[key] = counts.get(key, 0) + 1
        spellings.setdefault(key, {})
        spellings[key][value] = spellings[key].get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out = []
    for key, count in ranked:
        spelling = sorted(spellings[key].items(), key=lambda kv: (-kv[1], kv[0].casefold(), kv[0]))[0][0]
        out.append((spelling, count))
    return out


def _round_to_5(value: float) -> int:
    return int((value / 5.0) + 0.5) * 5


def _read_marker_run(path: str) -> str | None:
    """The ``run=`` of the ``#ALPHADEX-CLUSTER`` marker in the first three
    lines of ``path``, or None (read-only)."""
    try:
        with open(ensure_long_path(path), "r", encoding="utf-8", errors="replace") as fh:
            for _ in range(3):
                line = fh.readline()
                if not line:
                    break
                line = line.strip().lstrip("﻿")
                if line.startswith(MARKER_PREFIX):
                    for field_ in line[len(MARKER_PREFIX):].split(";"):
                        k, _, v = field_.partition("=")
                        if k.strip() == "run":
                            return v.strip()
    except OSError:
        return None
    return None


def _list_m3u(folder: str) -> list:
    """``[(exact name, (st_mtime_ns, st_size))]`` of the ``.m3u`` files in
    ``folder`` (none when it does not exist)."""
    out = []
    try:
        with os.scandir(ensure_long_path(folder)) as it:
            for entry in it:
                if not entry.name.lower().endswith(".m3u"):
                    continue
                try:
                    if not entry.is_file():
                        continue
                    st = entry.stat()
                except OSError:
                    continue
                out.append((entry.name, (st.st_mtime_ns, st.st_size)))
    except OSError:
        return []
    return sorted(out, key=lambda e: e[0].casefold())


def _folder_is_foreign(folder: str, run_id: str) -> bool:
    """True when ``folder`` holds a generated file of *another* run (CP-20)."""
    for name, _stamp in _list_m3u(folder):
        marker = _read_marker_run(os.path.join(folder, name))
        if marker is not None and marker != run_id:
            return True
    return False


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ── 1. Summary (read-only) ────────────────────────────────────────────────


def summarize_cluster_run(
    library_root,
    *,
    tag_reader: Callable[[str], dict] | None = None,
    exists: Callable[[str], bool] = os.path.exists,
    cancel_event=None,
    progress_callback: Callable[[int, int], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> RunSummary:
    """Everything the dialog and the planner need, without writing anything.

    Returns a :class:`RunSummary`; when the data cannot be used,
    ``blocking_error`` says why (CP-12) and ``clusters`` is empty. Raises
    :class:`ClusterPlaylistsCancelled` when ``cancel_event`` is set.
    ``progress_callback(done, total)`` covers the presence checks and the tag
    reads.
    """
    log = log_callback or (lambda _m: None)
    progress = progress_callback or (lambda _d, _t: None)
    reader = tag_reader or _default_tag_reader
    summary = RunSummary(library_root=str(library_root))
    root = os.path.abspath(str(library_root))

    # ── Load and cross-check (CP-7…CP-9, CP-12) ──────────────────────────
    try:
        payload = cgd.load_cluster_data(library_root)
    except cgd.ClusterDataError as exc:
        summary.blocking_error = str(exc)
        return summary
    if not isinstance(payload.get("run"), dict) or not cgd.cluster_members_path(library_root).is_file():
        summary.blocking_error = LEGACY_MESSAGE
        return summary
    try:
        members = cgd.load_cluster_members(library_root)
        cgd.check_members_match_payload(payload, members)
    except cgd.ClusterDataError as exc:
        summary.blocking_error = str(exc)
        return summary

    run = dict(members["run"])
    summary.run = run
    summary.map_points = len(payload.get("tracks") or [])
    summary.map_total = int(payload.get("X_total_points") or summary.map_points)
    summary.downsampled = bool(payload.get("X_downsampled"))

    labels = [int(v) for v in members["labels"]]
    distances = list(members["centroid_distance"])
    real_ids = sorted({v for v in labels if v != NOISE_LABEL})
    if not real_ids:
        summary.blocking_error = ALL_NOISE_MESSAGE
        return summary
    _check_cancel(cancel_event)

    # ── Rebase a moved library (CP-10) ───────────────────────────────────
    tracks = [str(t) for t in members["tracks"]]
    failed = [str(t) for t in (members.get("failed") or [])]
    old_root = str(run.get("library_root") or "")
    if old_root and not _same_root(old_root, root):
        tracks = [_rebase(t, old_root, root) for t in tracks]
        failed = [_rebase(t, old_root, root) for t in failed]
        summary.rebased_from = old_root
        summary.warnings.append(
            f"This analysis was made for {old_root}; paths were mapped to the current library."
        )

    # ── Presence (CP-11) ─────────────────────────────────────────────────
    sizes: dict = {}
    for label in labels:
        sizes[label] = sizes.get(label, 0) + 1
    sample_size = min(50, max(10, 2000 // len(real_ids)))
    total = len(tracks) + sum(min(sample_size, sizes[c]) for c in real_ids)
    done = 0

    width = _id_width(real_ids)
    present: dict = {c: [] for c in real_ids}     # id -> [(index, path)]
    missing: dict = {c: [] for c in real_ids}
    for i, path in enumerate(tracks):
        if i % _CANCEL_EVERY == 0:
            _check_cancel(cancel_event)
            if i:
                progress(done, total)
        label = labels[i]
        here = not is_reserved_path(path, root) and bool(exists(ensure_long_path(path)))
        done += 1
        if label == NOISE_LABEL:
            summary.noise_count += 1
            if not here:
                summary.missing_total += 1
            continue
        if here:
            present[label].append((i, path))
        else:
            missing[label].append(path)
            summary.missing_total += 1
    progress(done, total)
    if not any(present.values()):
        summary.blocking_error = ALL_MISSING_MESSAGE
        return summary

    # ── Tracks added since the analysis (CP-13) ──────────────────────────
    known = {_path_key(p) for p in tracks}
    known.update(_path_key(p) for p in failed)
    for j, path in enumerate(iter_library_files(root, AUDIO_EXTS)):
        if j % _CANCEL_EVERY == 0:
            _check_cancel(cancel_event)
        if _path_key(str(path)) not in known:
            summary.new_tracks += 1
    if summary.new_tracks:
        summary.warnings.append(
            f"{_plural(summary.new_tracks, 'track')} {'was' if summary.new_tracks == 1 else 'were'} "
            "added after this analysis and "
            f"{'is' if summary.new_tracks == 1 else 'are'} in no cluster."
        )
    track_count = int(run.get("track_count") or len(tracks))
    if track_count and (summary.missing_total + summary.new_tracks) > 0.10 * track_count:
        summary.warnings.append(
            "Your library changed a lot since this analysis; consider re-analysing."
        )

    # ── Run folder (CP-20) and its existing files (CP-18/CP-21) ──────────
    clusters_dir = os.path.join(root, *CLUSTERS_DIR)
    base_name = run_folder_name(run)
    folder_name = base_name
    n = 2
    while (
        os.path.isdir(ensure_long_path(os.path.join(clusters_dir, folder_name)))
        and _folder_is_foreign(os.path.join(clusters_dir, folder_name), summary.run_id)
    ):
        folder_name = f"{base_name} ({n})"
        n += 1
    run_folder = os.path.join(clusters_dir, folder_name)
    summary.run_folder = run_folder
    for name, stamp in _list_m3u(run_folder):
        summary.existing_files[name_key(name)] = (name, stamp)
    try:
        with os.scandir(ensure_long_path(clusters_dir)) as it:
            others = [e.name for e in it if e.is_dir() and e.name != folder_name]
    except OSError:
        others = []
    summary.earlier_sets = sorted(others, key=str.casefold)
    if summary.earlier_sets:
        k = len(summary.earlier_sets)
        summary.warnings.append(
            f"{_plural(k, 'earlier cluster playlist set')} {'is' if k == 1 else 'are'} "
            "kept in Playlists/Clusters/."
        )

    # ── Entries (CP-22): computed once here, so the planner stays pure ───
    entry_anchor = os.path.join(run_folder, "_.m3u")
    tempo = members.get("tempo_bpm")
    energy = members.get("energy")
    terciles = None
    if isinstance(energy, list):
        values = [float(v) for v in energy if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if len(values) >= 3:
            terciles = statistics.quantiles(values, n=3)

    # ── Descriptors and names (CP-14…CP-16) ──────────────────────────────
    for cid in real_ids:
        here = present[cid]
        paths = [p for _i, p in here]
        entries = pg.relative_playlist_lines(paths, entry_anchor)
        facts = ClusterFacts(
            cluster_id=cid,
            label=_cluster_label(cid, width),
            present=[(p, e, float(distances[i])) for (i, p), e in zip(here, entries)],
            missing=list(missing[cid]),
        )
        sample = sorted(here, key=lambda ip: (float(distances[ip[0]]), ip[1].casefold()))[:sample_size]
        tags = []
        for _i, path in sample:
            _check_cancel(cancel_event)
            try:
                read = reader(ensure_long_path(path))
            except Exception:  # noqa: BLE001 - an unreadable file is "no tags"
                read = {}
            tags.append(read if isinstance(read, dict) else {})
            done += 1
            progress(done, total)
        facts.parts = _descriptor_parts(tags, [i for i, _p in here], tempo, energy, terciles)
        facts.name = _fit_generated(facts.label, facts.parts, NAME_LIMIT)
        summary.clusters[cid] = facts
    progress(total, total)

    proposed = {name_key(f.name + ".m3u") for f in summary.clusters.values()}
    summary.left_unchanged = [
        exact for key, (exact, _stamp) in sorted(summary.existing_files.items())
        if key not in proposed
    ]
    log(
        f"Cluster playlist preview: {_plural(len(real_ids), 'cluster')}, "
        f"{sum(len(f.present) for f in summary.clusters.values())} present track(s), "
        f"{summary.missing_total} missing, {summary.new_tracks} new since the analysis."
    )
    return summary


def _descriptor_parts(tags: list, indices: list, tempo, energy, terciles) -> list:
    """``[primary, tempo, energy]`` (only the parts that qualify, CP-15)."""
    parts: list = []

    # Genre: dominant when ≥ 3 sampled tracks carry one and the top share is
    # ≥ 40 %; a runner-up with ≥ 25 % is appended as "A & B".
    genres = [g for g in (_text(t.get("genre")) for t in tags) if g]
    primary = ""
    if len(genres) >= 3:
        ranked = _dominant(genres)
        top, top_n = ranked[0]
        if top_n / len(genres) >= 0.40:
            primary = top
            if len(ranked) > 1 and ranked[1][1] / len(genres) >= 0.25:
                primary = f"{top} & {ranked[1][0]}"
    if not primary and tags:
        artists = []
        for t in tags:
            artist = _text(t.get("albumartist")) or _text(t.get("artist"))
            if artist and artist.casefold() != "various artists":
                artists.append(artist)
        if artists:
            top, top_n = _dominant(artists)[0]
            if top_n >= 3 and top_n / len(tags) >= 0.50:
                primary = top
    if primary:
        parts.append(primary)

    if isinstance(tempo, list) and indices:
        bpm = [float(tempo[i]) for i in indices
               if isinstance(tempo[i], (int, float)) and not isinstance(tempo[i], bool) and tempo[i] > 0]
        if bpm and len(bpm) * 2 >= len(indices):
            rounded = _round_to_5(statistics.median(bpm))
            if rounded > 0:
                parts.append(f"~{rounded} BPM")

    if terciles is not None and isinstance(energy, list) and indices:
        vals = [float(energy[i]) for i in indices
                if isinstance(energy[i], (int, float)) and not isinstance(energy[i], bool)]
        if vals:
            median = statistics.median(vals)
            if median < terciles[0]:
                parts.append("low energy")
            elif median > terciles[1]:
                parts.append("high energy")
    return parts


# ── 2. Plan (pure) ────────────────────────────────────────────────────────


def _header(name: str, run_id: str, cluster_id: int) -> list:
    return [
        "#EXTM3U",
        f"#PLAYLIST:{name}",
        f"{MARKER_PREFIX}run={run_id};cluster={cluster_id}",
    ]


def plan_cluster_playlists(
    summary: RunSummary,
    cluster_ids,
    *,
    order: str = DEFAULT_ORDER,
    min_size: int = DEFAULT_MIN_SIZE,
    names: dict | None = None,
) -> ClusterPlaylistPlan:
    """Rows and items for ``cluster_ids`` (CP-16…CP-25, CP-46).

    **Pure**: no file-system access; everything comes from ``summary``. Items
    are in ascending cluster id; each holds the exact lines Save writes.
    ``names`` maps a cluster id to the user's edited name (sanitised here).
    """
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}; expected one of {sorted(ORDERS)}")
    names = dict(names or {})
    plan = ClusterPlaylistPlan(
        run_id=summary.run_id, library_root=summary.library_root, run_folder=summary.run_folder,
    )
    if summary.blocking_error:
        plan.errors.append(summary.blocking_error)
        plan.ok = False
        return plan

    selected = {int(c) for c in (cluster_ids or ())}
    min_size = max(1, int(min_size))
    stem_limit = min(NAME_LIMIT, PATH_BUDGET - len(summary.run_folder) - len(os.sep) - len(".m3u"))

    rows: dict = {}
    active: dict = {}       # id -> (stem, existing entry or None)
    for cid in summary.cluster_ids():
        facts: ClusterFacts = summary.clusters[cid]
        n_present, n_missing = len(facts.present), len(facts.missing)
        if cid in names:
            stem = sanitize_playlist_stem(names[cid], limit=max(stem_limit, 0), fallback=facts.label)
        else:
            stem = _fit_generated(facts.label, facts.parts, max(stem_limit, 0))
        if stem_limit < len(facts.label):
            rows[cid] = PlanRow(cid, facts.name, n_present, n_missing, ACTION_SKIP, TOO_LONG_REASON)
            continue
        if n_present < min_size:
            reason = f"too small ({_plural(n_present, 'track')})"
            rows[cid] = PlanRow(cid, stem, n_present, n_missing, ACTION_SKIP, reason)
            continue
        if cid not in selected:
            rows[cid] = PlanRow(cid, stem, n_present, n_missing, ACTION_SKIP, NOT_SELECTED_REASON)
            continue
        existing = summary.existing_files.get(name_key(stem + ".m3u"))
        file_name = existing[0] if existing else stem + ".m3u"
        if len(os.path.join(summary.run_folder, file_name)) > PATH_BUDGET:
            rows[cid] = PlanRow(cid, stem, n_present, n_missing, ACTION_SKIP, TOO_LONG_REASON)
            continue
        action = ACTION_REPLACE if existing else ACTION_NEW
        rows[cid] = PlanRow(cid, stem, n_present, n_missing, action, "")
        active[cid] = (stem, existing)

    # Two ticked rows with the same name key make the plan invalid (CP-18).
    by_key: dict = {}
    for cid, (stem, _existing) in active.items():
        by_key.setdefault(name_key(stem + ".m3u"), []).append(cid)
    for key, ids in sorted(by_key.items()):
        if len(ids) < 2:
            continue
        message = f"Two playlists would be called {active[ids[0]][0]}"
        plan.errors.append(message)
        plan.ok = False
        for cid in ids:
            rows[cid] = rows[cid]._replace(reason=message)

    targeted: set = set()
    for cid in sorted(active):
        stem, existing = active[cid]
        facts = summary.clusters[cid]
        if order == "centre":
            ordered = sorted(facts.present, key=lambda m: (m[2], m[0].casefold()))
        else:
            ordered = sorted(facts.present, key=lambda m: m[0].casefold())
        entries = [entry for _path, entry, _dist in ordered]
        file_name = existing[0] if existing else stem + ".m3u"
        targeted.add(name_key(file_name))
        plan.items.append(pg.PlannedPlaylist(
            path=os.path.join(summary.run_folder, file_name),
            lines=_header(stem, summary.run_id, cid) + entries,
            exists=existing is not None,
            entries=len(entries),
            snapshot=tuple(existing[1]) if existing else None,
        ))

    plan.rows = [rows[cid] for cid in summary.cluster_ids()]
    if summary.noise_count:
        plan.rows.append(PlanRow(
            NOISE_LABEL, "Unclustered", summary.noise_count, 0, ACTION_SKIP, NOISE_REASON,
        ))
    plan.left_unchanged = [
        exact for key, (exact, _stamp) in sorted(summary.existing_files.items())
        if key not in targeted
    ]
    return plan


# ── 3. Write ──────────────────────────────────────────────────────────────


def _current_run_id(library_root: str) -> str | None:
    path = cgd.cluster_members_path(library_root)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    run = data.get("run") if isinstance(data, dict) else None
    return str(run.get("run_id")) if isinstance(run, dict) and run.get("run_id") else None


def write_cluster_playlists(plan: ClusterPlaylistPlan, *, cancel_event=None, log_callback=None) -> dict:
    """Write exactly ``plan.items`` (CP-28, CP-30, CP-31).

    Refuses the whole plan, writing nothing, when it is invalid or when
    ``cluster_members.json`` under ``plan.library_root`` now belongs to another
    analysis. Otherwise :func:`playlist_generator.write_playlist_plan` with
    ``backup=True``, ``verify=True`` and the log in ``<plan.library_root>/Docs``.

    Returns the writer's dict (``written``, ``backups``, ``errors``,
    ``cancelled``) plus ``refused`` (the refusal message, or ``None``).
    """
    log = log_callback or (lambda _m: None)
    refused = {"written": [], "backups": [], "errors": [], "cancelled": False, "refused": None}
    if not plan.ok:
        refused["refused"] = (
            "The preview is not valid ({}); nothing was written.".format("; ".join(plan.errors))
        )
        log(f"! {refused['refused']}")
        return refused
    if _current_run_id(plan.library_root) != plan.run_id:
        refused["refused"] = RUN_CHANGED_MESSAGE
        log(f"! {RUN_CHANGED_MESSAGE}")
        return refused
    result = pg.write_playlist_plan(
        plan.items,
        backup=True,
        log_callback=log,
        log_dir=os.path.join(plan.library_root, "Docs"),
        verify=True,
        cancel_event=cancel_event,
    )
    result["refused"] = None
    return result


def result_message(plan: ClusterPlaylistPlan, result: dict) -> tuple[str, bool]:
    """The page-log summary line for a write (CP-31/CP-41) and whether it is
    a clean success."""
    if result.get("refused"):
        return str(result["refused"]), False
    written = len(result.get("written") or [])
    replaced = len(result.get("backups") or [])
    failed = len(result.get("errors") or [])
    folder = "/".join((*CLUSTERS_DIR, os.path.basename(plan.run_folder))) + "/"
    text = (
        f"Saved {_plural(written, 'cluster playlist')} to {folder} "
        f"({replaced} replaced, backed up as .bak; {failed} failed)."
    )
    if result.get("cancelled"):
        text += " Cancelled before the rest were written."
    return text, failed == 0 and not result.get("cancelled")
