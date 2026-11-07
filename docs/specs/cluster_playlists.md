# Spec: cluster playlists ("Save every cluster as a playlist")

_Status: ready for implementation (audited 2026-11-07, see the audit log at the end) ·
Written 2026-11-07 · Area: LISTEN → Sound Map (`docs/redesign/README.md` §5) · Supersedes
the ROADMAP item "Playlists straight from a clustering run (Designed, not started)"._

## 0. Summary

A clustering run groups the library into clusters, and the map shows them. Today the
only way to turn a cluster into a playlist is to select it on the map and use
**Export playlist…**, one cluster at a time. That export writes the file directly, with no
backup and no log. This feature adds **Save cluster playlists…**. It is a previewed,
one-click export of one `.m3u` per cluster, written into a per-run folder under
`Playlists/Clusters/`, through the existing `playlist_generator` write machinery.

The spec has three parts:

1. **Data.** The clustering run saves the full cluster membership and the ordering data in
   a new `Docs/cluster_members.json`. Today `cluster_info.json` is downsampled to 5 000
   points, so a playlist built from it would quietly lose tracks.
2. **Backend.** A new Qt-free module, `cluster_playlist_export.py`, does three things: a
   read-only summary (missing files, tags for naming, the exact playlist entries), a pure
   planner (names, order, minimum size → `PlannedPlaylist` items), and a write step. The
   write step is a thin wrapper around `playlist_generator.write_playlist_plan`.
3. **UI.** A dialog, plus a page-agnostic flow object that **owns its own button**, starts
   the jobs and talks to the page through a five-item contract (§8.4). The same code works
   on today's Graph page, today's Clustered Results tab, and the merged Sound Map page,
   so the parallel layout restructuring does not block or conflict with it.

Requirements are numbered **CP-n**. Each one is testable, and §10 maps tests to them.

---

## 1. Current state (verified against the code)

| Fact | Where |
|---|---|
| A clustering run writes the feature cache, `Docs/cluster_info.json` and `Docs/cluster_graph.html`, and **no playlists**. It logs "Cluster playlists are not written automatically…". If `Docs/` cannot be created, the run falls back to writing into the library root. | `clustered_playlists.generate_clustered_playlists` (`:930`, docstring `:944`, `docs` fallback `:992-997`, log line `:1089-1092`) |
| `cluster_info.json` is **downsampled** to `MAX_VISUALIZATION_POINTS = 5000` points (evenly spaced indices). `X`/`X_2d`/`X_3d`/`labels`/`tracks` are subset together, and `X_downsampled`, `X_total_points` and `X_indices` record this. Full-length labels/tracks exist only in the in-memory result. A small cluster in a big library can therefore have **no point at all on the map**. | `build_cluster_payload` (`:892`), `visualization_indices` (`:876`) |
| The payload records **no run identity**: no timestamp, method, params, feature selection or library root. `cluster_info[c]` carries `size`, `genres: []` and `tempo: [0, 0]` (placeholders). | `generate_clustered_playlists` (`:1094-1103`) |
| The returned result is index-aligned: `tracks`, `X` (**scaled**) and `labels`, plus `features` (the **raw, unscaled** vectors), `failed` (path → error) and `cluster_info`. `ClusterWorker` adds `metrics` and `config` (the run's config dict, which for a wizard run includes the wizard's `create_playlists` key). | `:1149-1157`; `gui/workspaces/clustered_enhanced.py:187-225` |
| A feature vector's blocks are always in `FEATURE_ORDER = (mfcc 26, tempo 1, chroma 12, spectral 2, energy 2, onset_rate 1)`. Only the selected blocks are present, so a block's column offset is the sum of the widths of the *selected* blocks before it. The `energy` block's first number is the RMS mean with librosa but the **spectral-energy mean** with Essentia. | `FEATURE_ORDER`, `FEATURE_DIMENSIONS`, `_assemble_feature_vector`, `_extract_audio_features_librosa`/`_essentia` |
| On cancel, `ClusteringCancelled` is raised before any artefact is written (last check `:1117`). Extracted vectors stay cached. A cancel that arrives *after* `:1117` lets the run write its artefacts, and `ClusterWorker` then still reports "Cancelled" (pre-existing; out of scope here). | `:1044-1050`, `:1065`, `:1106`, `:1117`; `ClusterWorker.run` |
| The analysis job is started as `_start_worker(worker, "Clustering")`: not a library write, `ask_before_quit=True`. Today it runs on the Clustered page, while the map is on the Graph page. | `clustered_enhanced.py:450` |
| The preview-first playlist API is: `plan_playlist(tracks, outfile) → PlannedPlaylist(path, lines, exists, entries, encoding="utf-8")`. It **touches the disk** (`os.path.exists(outfile)`). Its lines come from `_relative_lines` → `safe_relpath` (relative, native separators, absolute only across Windows drives). `write_playlist_plan(plan, backup=True, log_callback, log_dir)` writes `item.lines` verbatim with `\n`, makes `name.m3u.<YYYYmmdd-HHMMSS>[-n].bak` beside any file that exists **at write time**, and appends `BACKUP`/`WRITE`/`ERROR` lines to `<log_dir>/playlist_writes.log` once, after the loop. It does not use `ensure_long_path`, has no cancel and no "changed since preview" check. | `playlist_generator.py:524-556`, `:666-722` |
| Rewriting existing entries keeps their style: `playlist_entry(target, dir, like=entry)` keeps absolute/relative and the separator. New entries (no `like`) are relative with native separators (`\` on Windows). | `utils/path_helpers.py` |
| `update_playlists` (run after Indexer moves) and the Duplicates executor walk `Playlists/` **recursively**, and Repair's empty-list mode uses `rglob`. So playlists under `Playlists/Clusters/…` stay valid after later moves and quarantines, with no extra work. The members file is *not* updated by those moves. | `playlist_generator.py:465` (`os.walk(playlists_dir)`), `duplicate_consolidation_executor.py:198-204`, `gui/workspaces/playlists.py:165` (`rglob`) |
| `cluster_graph_data` is the Qt-free, numpy-free loader and validator (`load_cluster_data`, `validate_cluster_payload` (ignores unknown keys), `NOISE_LABEL = -1`, `cluster_order`, `selected_tracks`, `selection_to_m3u`, which emits `#EXTM3U`). | `cluster_graph_data.py` |
| The Graph page's **Export playlist…** writes `selection_to_m3u` text with `Path.write_text`, after `QFileDialog.getSaveFileName` (filter allows `.m3u8`). It makes no backup, writes no log and doesn't check the library-write lock. The page only loads data when pyqtgraph is available. | `gui/workspaces/graph.py:369-385`, `reload_data` |
| Legend checkboxes mean **visibility** (`cluster_toggled`); `visible_clusters()` returns the ids of the rows the *map* has. Visibility is remembered per cluster id across reloads. A click on a row selects the cluster. Swatches use an inline `setStyleSheet`. | `gui/widgets/cluster_legend.py` |
| The wizard's output step has **Create M3U playlists for each cluster** (`_create_playlists_cb`), **Generate cluster quality report** and **Open interactive graph after clustering**, all disabled. Dimensionality reduction and post-processing are disabled too. The keys `create_playlists`, `generate_report` and `open_graph` exist in the wizard's in-memory run config (not in `~/.soundvault_config.json`) and nothing reads them. | `gui/dialogs/clustering_wizard_dialog.py:336-372`, `:417-438` |
| The Clustered Results tab says "Playlists are not created automatically…". Tests pin both that text and the disabled wizard option. | `clustered_enhanced.py:596-599`; `tests/ui/test_ui_playlists_clustering.py::test_results_do_not_promise_cluster_playlists`, `::test_wizard_does_not_offer_unbuilt_options` |
| Job rules: `_start_worker(w, name, ask_before_quit=…, library_write=…)` and `_library_writes_blocked(action)` (shows a "Library Busy" information box). The latter also honours `controllers.tag_write_guard`. `job_registry().jobs(owner=…)` matches by the owner's class name. `main_window.play_tracks` switches the window to the Player page. | `gui/workspaces/base.py:144-180`, `gui/workers.py`, `gui/main_window.py:587-598` |

---

## 2. User stories and scope

| # | Story | v1? |
|---|---|---|
| S1 | **Save every cluster as a playlist.** "I clustered my library into 8 moods; give me 8 playlists." | **In** |
| S2 | **Save just these clusters.** "Only clusters 2, 5 and 6 are any good." Untick rows in the dialog, or hide clusters in the map legend first (hidden clusters start unticked). | **In** |
| S3 | **Regenerate after re-clustering.** "I re-ran with k=12; save the new set." Each run gets its own folder; earlier sets are kept (§6). Re-saving the *same* run replaces its files, with backups. | **In** |
| S4 | **Play a cluster.** Selecting a cluster in the legend and pressing *Play selection* on the map keeps working as it does today. A ▶ inside the save dialog is **out of v1** (see CP-40). | Existing behaviour only |
| S5 | **Offer it right after an analysis.** The wizard option "Offer to save a playlist per cluster when the analysis finishes" opens the same preview. | **In** |

**Out of v1** (each is a clean follow-up):

- Writing without the preview, ever.
- Deleting or retiring earlier run folders.
- A playlist of HDBSCAN noise.
- A per-playlist track cap ("the 50 most central").
- The "smooth path through the cluster" order (CP-5, deferred).
- A legend right-click "Save as playlist…" (CP-38, deferred).
- Playing a row from inside the dialog (CP-40, deferred).
- Remembering custom names across runs.
- Showing names in the map legend.
- `.m3u8`, absolute-path or forward-slash output options, and `#EXTINF` duration lines.
- A destination outside `<library>/Playlists/Clusters/`.
- Payloads written before this feature (they must be re-analysed, CP-12).
- The legacy Tk app (it gains the members file as a side effect of CP-1, and nothing else).
- The redesign's shared **Save playlist sheet**. The dialog should be built so it can grow into that sheet, but v1 doesn't need to be it.

---

## 3. Terms

- **Run.** One successful clustering run. It is identified by `run_id` and described by the `run` block (CP-2).
- **Members file.** `<docs>/cluster_members.json`, the full and never-downsampled membership of a run (CP-3). `<docs>` is the same folder `cluster_info.json` goes to, normally `<library>/Docs`.
- **Map payload.** `<library>/Docs/cluster_info.json`, as today, plus the `run` block.
- **Run folder.** `<library>/Playlists/Clusters/<run folder name>/` (CP-20).
- **Summary.** The output of the read-only phase (CP-26): the clusters, which members are present or missing, each present member's playlist entry, naming facts, the existing files in the run folder, and any warnings.
- **Plan.** The output of the pure planner (CP-28): rows for the dialog, plus the `PlannedPlaylist` items that will be written.
- **Name key.** `name.lower()` of a full file name (`stem + ".m3u"`), used for every "same file name?" comparison. `lower()` (simple case mapping) is used instead of `casefold()` because it matches how NTFS/APFS compare names (`casefold` would treat `Straße` and `Strasse` as one file; NTFS does not). `casefold` is still used for *sorting*.

---

## 4. Data contract

### 4.1 What the clustering run writes

**CP-1 — Members file is written by every successful run.**
`generate_clustered_playlists` writes `<docs>/cluster_members.json` after the last cancel
check (`:1117`), **before** `cluster_info.json`, using the same atomic `tmp` +
`os.replace` pattern. It is written compact (`separators=(",", ":")`, no indent) so a
20 000-track run stays around 3 MB. When a run is cancelled before that point, neither
file is written (this extends the existing `ClusteringCancelled` rule). If the members file
fails to write, the error is logged as
`! Could not save cluster membership (cluster playlists unavailable for this run): …`, and
`cluster_info.json` is still written. (A members file from an *earlier* run may then sit
beside the new map; CP-9 refuses that pair.)

**CP-2 — Both files carry the same `run` block.**
The map payload gains a top-level `"run"` key, and the members file carries the identical
object:

| key | value |
|---|---|
| `schema` | `1` |
| `run_id` | `<YYYYmmdd-HHMMSS>-<6 hex>`: the local start time of the run plus `secrets.token_hex(3)`, so two runs in one second still differ |
| `created` | ISO-8601 local time of the run start (`run_started`) |
| `library_root` | `os.path.abspath(root_path)` |
| `method` | `"kmeans"` or `"hdbscan"` (after `normalize_method`) |
| `params` | the `params` dict passed to `cluster_tracks` (as requested; `cluster_tracks` may still lower `n_clusters` for tiny libraries) |
| `features` | `feature_selection_key(selection)`, e.g. `"mfcc-tempo-energy"` |
| `feature_engine`, `normalization` | as used |
| `track_count` | number of clustered tracks (= length of the members arrays) |
| `failed_count` | `len(failed)` |

`build_cluster_payload` keeps its alignment behaviour unchanged. The `run` key is added to
its returned dict by `generate_clustered_playlists`, beside the arrays.
`validate_cluster_payload`, `cluster_graph_3d` and the legacy `cluster_graph_panel` ignore
unknown keys, so they need no change.

**CP-3 — Members file schema (index-aligned, full length).**

```json
{
  "run": { "...": "identical to cluster_info.json's run block" },
  "tracks":            ["/music/A/B/01 x.flac", "..."],
  "labels":            [3, 0, -1, "..."],
  "centroid_distance": [0.84, 1.92, null, "..."],
  "tempo_bpm":         [121.4, 98.0, "..."],
  "energy":            [0.112, 0.034, "..."],
  "failed":            ["/music/C/D/broken.mp3"]
}
```

- `tracks`, `labels` and `centroid_distance` all have length `run.track_count`. Element
  *i* of each describes the same track, in the same order as the run's returned `tracks`.
  **The CLAUDE.md index-alignment rule applies to this file too**, and nothing may subset
  one array without the others. The members file is **never downsampled**.
- `tempo_bpm` is present only when the `tempo` block was selected, and `energy` only when
  `energy` was. Both are taken from the **raw** `features` vectors, not the scaled `X`. The
  column of a block is the sum of `FEATURE_DIMENSIONS` of the *selected* blocks that come
  before it in `FEATURE_ORDER` (tempo is column 26 when MFCC is selected, 0 when not).
  `energy` is the energy block's *first* number: RMS mean with librosa, spectral-energy
  mean with Essentia. It is only ever compared within one run (CP-15), so the unit doesn't
  matter. Each is the same length as `tracks` when present. `null` means not available.
- Noise tracks (`label == -1`) have `centroid_distance: null`; every other track has a
  number.
- `failed` lists the paths that could not be analysed, as the keys of the run's `failed`
  mapping.
- (`path_rank` was removed in the audit with CP-5. A later "smooth path" order may add it
  as an optional key; readers must ignore unknown keys, so that needs no schema bump.)

**CP-4 — Centroid distance.**
For every cluster `c ≥ 0`, the centroid is the **mean of the scaled vectors `X`** of its
members. The same rule holds for K-Means and HDBSCAN. `centroid_distance[i]` is the
Euclidean distance from `X[i]` to its cluster's centroid. This is computed in a new numpy
helper, `clustered_playlists.cluster_centroid_distances(X, labels) -> list[float | None]`
(O(n·d); milliseconds for 20 000 tracks).

**CP-5 — (removed in audit: the "smooth path" order is deferred.)** The greedy
nearest-neighbour chain is O(n²) per cluster, needed a size limit, a per-cluster fallback
and an extra array, and v1 is complete without it: *Closest to the centre first* already
gives a meaningful order. It is a clean follow-up (see the note in CP-3).

**CP-6 — Returned result.**
The dict returned by `generate_clustered_playlists` gains `"run"` (the block) and
`"members_file"` (path, or `None` when it failed to write). Existing keys are unchanged.
`ClusterWorker` passes them through untouched.

### 4.2 Loading and validation (in `cluster_graph_data`, still Qt- and numpy-free)

**CP-7 — `cluster_members_path(library)`** returns `Path(library) / "Docs" /
"cluster_members.json"`. **`load_cluster_members(library)`** reads it, validates it, and
returns the dict. It raises `ClusterDataError` with an actionable message in four cases:
the file is missing, it is unparseable, it is not an object, or it lacks `run`,
`run.run_id`, `run.track_count`, `tracks` or `labels`.

**CP-8 — `validate_cluster_members(data)`** raises `ClusterDataError` when:

- `tracks`, `labels` and `centroid_distance` don't all have the length `run.track_count`;
- `tempo_bpm` or `energy` is present with a different length;
- `labels` contains anything that is not an `int` (a `bool` counts as not an `int`);
- `centroid_distance[i]` is `null` for a non-noise track, or a number for a noise track.

**CP-9 — `check_members_match_payload(payload, members)`** raises `ClusterDataError("…out
of step with the map… Re-run the analysis.")` in either of two cases. It compares the
**stored** paths (before any rebasing, CP-10).

- `payload["run"]["run_id"] != members["run"]["run_id"]`.
- The arrays disagree. The check depends on how the payload was written:
  - If `X_downsampled` is false: `payload.tracks == members.tracks` and
    `payload.labels == members.labels`.
  - If it is true: `payload.X_total_points == run.track_count`,
    `len(X_indices) == len(payload.tracks)`, and for every *i*,
    `members.tracks[X_indices[i]] == payload.tracks[i]` and
    `members.labels[X_indices[i]] == payload.labels[i]`.

### 4.3 Stale data and missing files

**CP-10 — Library root rebasing.**
Compare `run.library_root` with the current library path, using `normcase(abspath())` on
both. If they differ, every stored path under the old root (members **and** `failed`) is
mapped to `join(current_root, relpath(track, old_root))`. A path for which `relpath` raises
(another Windows drive) is kept as stored. The summary records `rebased_from = old_root`
and shows a warning: "This analysis was made for `<old>`; paths were mapped to the current
library." This covers the common Windows case of an external drive coming back under a
different letter.

**CP-11 — Present, missing and reserved.**
Every member is checked with `exists(ensure_long_path(track))` after rebasing (`exists` is
injectable, default `os.path.exists`; `ensure_long_path` makes Windows paths over 260
characters checkable and is a no-op elsewhere). A member that is missing, or that
`utils.reserved_folders.is_reserved_path(track, root)` reports as reserved, is **left out**
of its playlist. It is counted per cluster as `missing`, and its paths are kept for the
dialog's tooltip. The planner never re-points a missing track.

**CP-12 — Refusals (blocking).**
The summary returns `blocking_error` instead of clusters in six cases:

| Case | Message |
|---|---|
| No `cluster_info.json` | the existing `load_cluster_data` text |
| Payload has no `run` block, or no members file (legacy payload) | "This map was made before cluster playlists existed. Re-run the analysis to save a playlist per cluster. (Export playlist… still works for a selection.)" |
| CP-7/CP-8 fails | its message |
| CP-9 fails | its message |
| The run has no real cluster (every label is `-1`) | "This analysis put every track in 'Unclustered', so there is no cluster to save. Try a smaller minimum cluster size." |
| Every non-noise member is missing | "None of the analysed tracks are where they were; re-run the analysis." |

**CP-13 — New tracks since the analysis (warning, not blocking).**
The summary walks the library with `utils.reserved_folders.iter_library_files(root,
AUDIO_EXTS)`. `cluster_playlist_export.AUDIO_EXTS` is a frozenset equal to the analysis
worker's extension set (today `clustered_enhanced._AUDIO_EXTS`: `.mp3 .flac .m4a .aac .ogg
.wav .opus`); a test pins the two as equal, and the Sound Map merge may make the worker
import it. The analysis page itself is **not** edited for this. The summary counts the
files whose `normcase(normpath())` key is neither among the members (after rebasing) nor
among `failed`, and reports "N tracks were added after this analysis and are in no
cluster." If `(missing_total + new_tracks) > 10 %` of `run.track_count`, it adds "Your
library changed a lot since this analysis; consider re-analysing." Tracks retagged or
re-encoded in place are **not** detected, and files the worker skipped as unreadable count
as new (documented limitations).

---

## 5. Naming

**CP-14 — Deterministic names.**
With the same members file, the same tags and the same options, two summaries produce the
same proposed names. Names come only from this data and never from wall-clock time or set
iteration order.

**CP-15 — Descriptor facts per cluster.**

1. **Tag sample.** Per cluster, take the `S` **present** members with the smallest
   `centroid_distance`, ties broken by `casefold(path)`, where
   `S = min(50, max(10, 2000 // number_of_clusters))`. That caps the whole summary at about
   2 000 tag reads, even for an HDBSCAN run with hundreds of clusters. Read their tags with
   a tag reader called as `tag_reader(ensure_long_path(path))`. The default is
   `utils.audio_metadata_reader.read_tags`, looked up on that module at call time so tests
   can monkeypatch it (or `read_metadata` underneath it). A read that raises counts as "no
   tags".
2. **Genre.** Take the `genre` value (`read_tags` returns the first genre), stripped.
   Count it case-insensitively, and display the most frequent original spelling (ties are
   broken alphabetically by casefold). The *dominant genre* is the top genre when at least
   3 sampled tracks carry a genre and the top share is ≥ 40 % of them. If the runner-up's
   share is ≥ 25 %, it is appended as `A & B` (not `A / B`: CP-17 turns `/` into `-`, which
   would read as another name part).
3. **Artist** (used only when there is no dominant genre). Use `albumartist`, falling back
   to `artist`. Ignore `Various Artists` (casefold) and empty values. It qualifies when it
   covers ≥ 50 % of the sample and at least 3 tracks.
4. **Tempo** (only when `tempo_bpm` exists). Take the median BPM over **all** present
   members with a value > 0, provided at least half of them have one. Round it to the
   nearest 5 and render `~120 BPM`.
5. **Energy** (only when `energy` exists). Compute the run's terciles `t1` and `t2` with
   `statistics.quantiles(values, n=3)` over every non-null value in the members file. With
   fewer than 3 such values, no cluster gets an energy word. A cluster whose median (over
   its present members) is below `t1` renders `low energy`, and one above `t2` renders
   `high energy`. A medium cluster gets no energy word.

**CP-16 — Name shape.**

`Cluster <id> - <primary> - <tempo> - <energy>`

- Missing parts are dropped. `<primary>` is the dominant genre, or the artist when there is
  no dominant genre.
- `<id>` is the run's cluster id, the same number the legend shows ("Cluster 3"),
  zero-padded to the width of the largest id in the run, minimum 2 digits. The padding
  makes files sort in cluster order.
- When no part qualifies, the name is `Cluster 03`.
- Examples: `Cluster 03 - Deep House - ~120 BPM - high energy`,
  `Cluster 05 - Techno & House`, `Cluster 07 - Radiohead`, `Cluster 11`.
- Generated names never collide with each other, because each carries its unique id.

The (sanitised) name is used both as the file stem and as the `#PLAYLIST:` title (CP-23).

**CP-17 — Sanitising.**
`sanitize_playlist_stem(text, limit=100)` applies these rules in order:

1. NFKC-normalise (this also turns `COM¹` into `COM1` before rule 7).
2. Remove control characters (Unicode category `Cc`, which includes `\n`, `\t` and NUL).
3. Replace each of `<>:"/\|?*` with `-`.
4. Collapse runs of whitespace into one space.
5. Trim to `limit` characters.
6. Strip leading and trailing spaces and dots (a leading dot would make a hidden file;
   trailing ones are illegal on Windows). This runs *after* the trim so a cut can't leave
   a trailing dot or space.
7. If the part before the first dot, with trailing spaces removed, is a Windows device
   name (`CON PRN AUX NUL CONIN$ CONOUT$ COM0-9 LPT0-9`, case-insensitive), append `_` to
   the whole stem (`CON` → `CON_`, `nul.mix` → `nul.mix_`). Windows treats `CON.m3u` and
   `nul.mix.m3u` as the device.

If the result is empty, it becomes `Cluster <id>`. This is a new function. Don't import
`music_indexer_api.sanitize`: it pulls in the indexer stack and doesn't handle trailing
dots or device names. For generated names, whole descriptor parts are dropped from the
right before any hard trim (CP-46 can lower `limit`).

**CP-18 — User-edited names and collisions.**
The dialog lets the user edit each name, and the committed edit is sanitised by CP-17.
Two rules apply, both using the **name key** (§3):

- **Inside the plan:** two ticked rows with the same name key make the plan invalid.
  Save is disabled, and both rows show "Two playlists would be called X".
- **Against existing files:** the `.m3u` files already in the run folder (listed by the
  summary) are compared by name key. A match makes the row's action **Replace**, and the
  item targets the existing file's *exact* name. On a case-sensitive file system this
  avoids writing `house.m3u` beside `House.m3u`; on Windows it keeps the file's own
  spelling.

Run folders are per-run (CP-20), so a name can never collide with a playlist outside its
own run folder. That includes the user's own playlists in `Playlists/`. Custom names are
not persisted (v1).

**CP-19 — Noise is never a playlist.**
Label `-1` produces no plan item under any option. The dialog shows it as a disabled row,
"Unclustered (N) — not saved".

---

## 6. Destination, format and re-runs

**CP-20 — Run folder.**
The run folder is `<library>/Playlists/Clusters/<YYYY-MM-DD HH.MM.SS> <Method><param>/`,
built by `run_folder_name(run)` from `run.created`:

- `<Method>` is `K-Means` or `HDBSCAN`.
- `<param>` is ` (k=8)` for K-Means (from `params.n_clusters`) and ` (min 5)` for HDBSCAN
  (from `params.min_cluster_size`).

Example: `Playlists/Clusters/2026-11-07 15.30.12 K-Means (k=8)/`. The name has no colons
and ends in `)`, so it is legal on Windows. `write_playlist_plan` creates folders as needed.

The summary reads the first three lines of each `.m3u` already in that folder. If any
carries an `#ALPHADEX-CLUSTER` marker (CP-23) with a **different** `run=`, the folder
belongs to another run that started in the same second (in practice only possible around
a clock change). The summary then uses ` (2)`, ` (3)`, … appended to the folder name until
it finds a folder that is absent or holds only this run's files. This keeps CP-21's
"earlier folders are never modified" true.

**CP-21 — Re-runs keep history; re-saves replace.**
Saving a *different* run writes into that run's own new folder, and earlier run folders are
**never** modified or deleted. Saving the *same* run again (for example after first saving
only some clusters) writes into the same folder. Same-named files are replaced and backed
up (CP-31). Files from an earlier save of that run whose names no longer appear (for
example after a rename) are left in place, and the dialog lists them as "left unchanged".
The summary lists the other folders under `Playlists/Clusters/` (`earlier_sets`), and the
dialog says "N earlier cluster playlist sets are kept in Playlists/Clusters/".

**CP-22 — Entry style and encoding.**
Each playlist's entries are exactly the lines `playlist_generator.plan_playlist(tracks,
outfile).lines` would produce for the same tracks and file. Both go through the same
helper: `_relative_lines` becomes the public `playlist_generator.relative_playlist_lines
(tracks, outfile)` (the private name stays as an alias), and the **summary** calls it once
per present member against the run folder, in the worker thread. The planner then only
orders and concatenates them. (The planner can't call `plan_playlist`, which touches the
disk, CP-28.)

Entries are relative to the run folder with **native separators**:
`../../../Artist/Album/01 x.flac` on macOS/Linux and `..\..\..\Artist\Album\01 x.flac` on
Windows. They fall back to absolute paths only when no relative path exists (another
Windows drive). This is the same rule every other generated playlist follows. Later
rewrites by `update_playlists` and Repair keep each entry's own style
(`playlist_entry(…, like=entry)`).

Files are `.m3u`, UTF-8 without a BOM, with `\n` line endings: the `PlannedPlaylist`
default `encoding="utf-8"` as written by `write_playlist_plan`. A path that UTF-8 cannot
encode (an undecodable POSIX file name) makes that one file fail. The failure is reported in
`errors` and the file is not written, which is existing writer behaviour.

**CP-23 — Header lines.**
Three header lines are prepended to the entries:

1. `#EXTM3U`
2. `#PLAYLIST:<name>`
3. `#ALPHADEX-CLUSTER:run=<run_id>;cluster=<id>`

`PlannedPlaylist.entries` stays the track count. Every reader in the repo skips `#` lines.
`_rewrite_playlist_lines` treats `#PLAYLIST`/`#ALPHADEX` as plain comments, so they survive
renames. The marker is used by CP-20, and lets a later "retire earlier sets" feature
recognise generated files safely.

**CP-46 — Path-length budget (added in audit; Windows).**
`write_playlist_plan` opens plain paths, so on Windows without long-path support a path
over 259 characters fails. The planner keeps every item inside a budget, on every platform
(so tests on Linux see the same names):
`len(item.path) + len(".YYYYmmdd-HHMMSS-99.bak") <= 259`, which also covers the backup
`write_playlist_plan` may create beside it. When a generated name doesn't fit, descriptor
parts are dropped from the right, then the stem is trimmed (CP-17 with a lower `limit`). A
user-edited name is trimmed the same way. If even `Cluster <id>` doesn't fit, the row is
**Skip** with reason "path too long for Windows; move the library to a shorter folder".

---

## 7. Ordering and minimum size

**CP-24 — Order options** (dialog combo; default **Closest to the centre first**):

| Option | Key | Rule |
|---|---|---|
| Closest to the centre first | `centre` | Present members sorted by `(centroid_distance, casefold(path))`. The most typical tracks come first. |
| Library order | `library` | Present members sorted by `casefold(path)`, which is artist/album/track order for an indexed library. |

(The `path` option was removed with CP-5.)

**CP-25 — Minimum size.**
A spin box from 2 to 500, default **5**, matching `DEFAULT_HDBSCAN_MIN_SIZE`. It counts
**present** tracks. A cluster below it gets action **Skip** with reason "too small (N
tracks)", and is not ticked. Its checkbox is disabled while it is below the minimum, with
that reason as the tooltip. This setting is not the wizard's "Remove clusters smaller than
threshold", which would change the clustering itself (§9, CP-45).

---

## 8. Flow, modules and UI

### 8.1 Backend: `cluster_playlist_export.py` (new, top level, no Qt, no numpy)

| Function | Does | I/O |
|---|---|---|
| `summarize_cluster_run(library_root, *, tag_reader=None, exists=os.path.exists, cancel_event=None, progress_callback=None, log_callback=None) -> RunSummary` | Loads the payload and members file (CP-7…CP-9), rebases (CP-10), checks presence (CP-11), counts new tracks (CP-13), samples tags and computes descriptor facts (CP-15), chooses the run folder (CP-20), computes each present member's entry (CP-22), and lists the run folder's existing `.m3u` files with `(st_mtime_ns, st_size)` and the `earlier_sets` | Read-only |
| `plan_cluster_playlists(summary, cluster_ids, *, order="centre", min_size=5, names=None) -> ClusterPlaylistPlan` | Builds the rows and items (CP-16…CP-25, CP-46). Items are in ascending cluster id. Each item is a `PlannedPlaylist(path, lines=header + entries, exists=…, entries=n, snapshot=…)`, with `exists`/`snapshot` taken from the summary's listing | **Pure** (CP-28) |
| `write_cluster_playlists(plan, *, cancel_event=None, log_callback=None) -> dict` | Re-reads the members file of `plan.library_root` and refuses if its `run.run_id` changed (CP-30); otherwise calls `write_playlist_plan(plan.items, backup=True, log_callback=…, log_dir=<plan.library_root>/Docs, verify=True, cancel_event=…)` | Writes |
| `run_folder_name(run)`, `sanitize_playlist_stem(text, limit=100)`, `AUDIO_EXTS` | CP-20, CP-17, CP-13 | none |

`RunSummary` is a dataclass with these fields:

- `library_root` (as given), `run`, `run_folder` (absolute path), `rebased_from`
- `clusters: dict[int, ClusterFacts]`, where each `ClusterFacts` holds the id, the present
  members as `(path, entry, centroid_distance)`, the missing paths, the descriptor parts and
  the proposed name
- `noise_count`, `missing_total`, `new_tracks`, `map_points`, `map_total`, `downsampled`
- `existing_files` (name key → `(exact name, (mtime_ns, size))`), `left_unchanged`,
  `earlier_sets`, `warnings`, `blocking_error`

`ClusterPlaylistPlan` holds `run_id`, `library_root`, `run_folder`, `rows`, `items`,
`errors` and `ok`. Each row is `(cluster_id, name, tracks, missing, action ∈ {new,
replace, skip}, reason)`.

**CP-26 — The summary is read-only and cancellable.**
It never creates, modifies or deletes a file. It checks `cancel_event` at least every 200
members during the presence check and the walk, and between tag reads, and raises
`ClusterPlaylistsCancelled` when the event is set. It reports `progress_callback(done,
total)` over the presence checks and the tag reads. For 20 000 tracks it does about 20 000
`stat` calls, one library walk and at most ~2 000 tag reads (CP-15), which is why it runs
as a job.

**CP-27 — Shared writer extensions** (`playlist_generator`, backward compatible):

- `PlannedPlaylist.snapshot: tuple | None = None`, set by the cluster planner from the
  summary's listing. `plan_playlist` is **not** changed.
- `write_playlist_plan(..., verify=False, cancel_event=None)`, both keyword-only additions
  after the existing parameters; every current caller (`gui/workspaces/playlists.py`, the
  tests) is unaffected.
  - With `verify=True`, two kinds of item are **skipped**, reported in `errors` as
    `(path, "changed since the preview")` and logged as `ERROR <path>: changed since the
    preview`: an item planned as new whose file now exists, and an item planned as an
    overwrite whose `(st_mtime_ns, st_size)` no longer equals `snapshot`. An item planned
    as an overwrite whose file has disappeared is written as new (nothing can be lost).
  - With `cancel_event` set, writing stops **between** files. The files already written
    stay written, and the log lines gathered so far are still appended to
    `playlist_writes.log`.
  - The result always gains `"cancelled": bool`. Apart from that, with the defaults the
    behaviour and every file written are exactly today's, and the existing tests pass
    unchanged.

**CP-28 — The planner is pure and the plan is exact.**
`plan_cluster_playlists` does no file-system access at all (no `exists`, `stat`, `open`,
`relpath` or `abspath` on relative paths); everything it needs is in the summary.
`write_cluster_playlists` writes `plan.items` and nothing else: every file has exactly
`item.lines` and sits at exactly `item.path`. There is no second writer. All writing goes
through `write_playlist_plan`.

### 8.2 Preview-first flow (CLAUDE.md rule)

```
[Save cluster playlists…] ─► lock check ─► summary job (read-only, cancellable)
   ─► dialog = the preview (rows: name · tracks · missing · New/Replace/Skip; options)
   ─► [Save N playlists] ─► lock check ─► (if any Replace) question: "K existing files will be replaced (backed up first)"
   ─► write job (library_write=True) ─► write_playlist_plan(verify=True) ─► .bak + Docs/playlist_writes.log ─► result line
```

**CP-29 — Nothing is written before Save.**
Opening the dialog and changing options write nothing. These terminal paths write nothing
and each logs "Nothing was written." to the page log: cancelling the dialog, declining the
replace question, a blocked lock (either check), a cancelled summary (CP-35) and a
blocking summary error (CP-12).

**CP-30 — Plan signature.**
Two checks guard against changes since the preview.

- *Run change:* if `cluster_members.json`'s `run_id` changed between the summary and the
  write (a new analysis landed), the write refuses the **whole** plan with "The analysis
  changed since the preview; open Save cluster playlists… again." and writes nothing.
- *File change:* a file in the run folder that appeared or changed since the preview is
  skipped (CP-27 `verify`).

**CP-31 — Backups and log.**
Every replaced file first gets a `name.m3u.<YYYYmmdd-HHMMSS>[-n].bak` beside it. Every
backup, write and error is appended to `<library>/Docs/playlist_writes.log`. Both come from
the existing writer. The page log gets the `log_callback` lines and one summary line:
"Saved N cluster playlist(s) to Playlists/Clusters/<folder>/ (K replaced, backed up as
.bak; E failed)."

### 8.3 Jobs, lock and cancel: `gui/cluster_playlist_jobs.py` (new, page-agnostic)

`ClusterPlaylistFlow(page: WorkspaceBase, *, hidden_clusters=None, map_run_id=None,
cluster_colour=None)` is a `QObject` owned by the page. The three optional arguments are
callables the page supplies (CP-37, CP-39); with none of them the flow still works. It holds
`ClusterPlaylistSummaryWorker` and `ClusterPlaylistWriteWorker`, which are `QThread`s whose
`done` signal is not named `finished`, each with a real `cancel()` backed by a
`threading.Event`. Its interface:

- `button` — a `QPushButton` the flow creates and manages (text, enabled state, tooltip,
  Cancel mode). The page only places it.
- `start(only: set[int] | None = None)` — `None` means every cluster except the hidden ones.
- `refresh()` — re-evaluates availability (CP-37). Cheap: a `stat`, no JSON parsing.
- `offer_after_run(result)` — CP-44.
- `cancel()`, `is_busy()`, `on_library_changed()`
- `busy_changed(bool)`, `progress(int, int)`, `result_ready(str, bool)` (text, ok)

The flow starts jobs with `page._start_worker(...)`, so the jobs are owned by the page
(`job_registry().jobs(owner=page)` sees them), and logs through `page._log(message, level)`.

**CP-32 — Read-only job.**
`start()` first calls `page._library_writes_blocked("Save cluster playlists")` and does
nothing more (logging "Nothing was written.") if that returns true: while another page is
moving files, a presence check would be meaningless. The summary job then starts as
`page._start_worker(w, "Cluster playlist preview", ask_before_quit=False)`. Its
`Job.group` is `None` and `Job.ask_before_quit` is `False`.

**CP-33 — Write job.**
After the dialog is accepted, the flow calls `page._library_writes_blocked("Save cluster
playlists")` again. When the plan contains any Replace row, it then asks
`QMessageBox.question`. Last, it starts `page._start_worker(w, "Save cluster playlists",
library_write=True)`, with `ask_before_quit` left at its default `True`. The worker
receives the `ClusterPlaylistPlan` object the dialog showed, not a recomputation.

**CP-34 — Busy rules.**

- While either job of the flow runs, its button is disabled (or in Cancel mode, CP-35)
  and `busy_changed(True)` is emitted. A page that also runs analyses disables its Run
  buttons on it.
- While a clustering analysis runs **anywhere in the app**, the button is disabled with
  the tooltip "Wait for the analysis to finish." The flow detects this through the job
  registry: any running job named `ANALYSIS_JOB_NAME = "Clustering"` (a constant in
  `gui/cluster_playlist_jobs.py`; a test pins that the analysis page starts its worker
  under that name). The flow listens to `job_registry().jobs_changed` to update the
  button. This matters today because analysis and map live on different pages.
- Slots ignore any signal whose `sender()` is not the current worker (the PL-07 pattern).

**CP-35 — Cancel.**

- While the summary job runs, the flow's button reads "Cancel preview" and clicking it
  calls `cancel()`. A cancelled summary opens no dialog and logs "Cancelled. Nothing was
  written."
- The write job is short and shows no Cancel. Its `cancel()` (stop between files, CP-27) is
  used by the main window's quit path: quitting mid-write asks, cancels and waits
  (`ask_before_quit=True`).

**CP-36 — Library change.**
`on_library_changed()` (called by the page from `_on_library_changed`) cancels a running
summary and drops its result, clears any pending auto-offer (CP-44) and calls `refresh()`.
A running write is left to finish in the old library, which is the Playlists-page rule: it
writes exactly the confirmed plan, and checks and logs against `plan.library_root`, never
the page's new library.

### 8.4 UI on the Sound Map page

**Integration contract.** Whichever page shows the map (today `GraphWorkspace`; after the
merge, the Sound Map page) and whichever page runs the analysis (today the Clustered tab;
after the merge, the same page) each own one `ClusterPlaylistFlow` and do only this:

1. create it and add `flow.button` to their action area;
2. call `flow.refresh()` after loading (or failing to load) map data, and on show;
3. call `flow.on_library_changed()` from `_on_library_changed`;
4. map page only: pass `hidden_clusters`, `map_run_id` and `cluster_colour` callables;
5. analysis page only: call `flow.offer_after_run(result)` after a *successful* run.

Nothing else in the page is touched, so the layout restructuring can move the button
freely. Before the merge this means two flow instances; they are independent (the summary
is read-only, and writes are serialised by the library-write lock).

**CP-37 — Primary action.**
The flow's **Save cluster playlists…** button works independently of the map selection.

- **Enabled when:** a library is set, `cluster_members_path(library)` is a file, and no
  job blocks it (CP-34). It does **not** depend on the map having loaded, so the feature
  works without pyqtgraph.
- **Disabled otherwise**, with a tooltip:
  - no members file: "Analyse your library first. (Analyses made before this feature
    need to be re-run.)"
  - busy: CP-34
  - Everything else (legacy payload with a stale members file, mismatch, all missing) is
    reported by the summary's `blocking_error` (CP-12) in an information box.
- **Preselection:** every non-noise cluster of the *run*, minus `hidden_clusters()`. It is
  computed from the members file, not from the legend, because a small cluster can be
  absent from a downsampled map. Hidden clusters are only applied when
  `map_run_id() == summary.run.run_id`; otherwise everything is preselected and the
  warning "The map shows an older analysis; press Reload to see this one" is added.

**CP-38 — (removed in audit: the legend right-click "Save as playlist…" is deferred.)** S2
is fully served by the dialog's checkboxes and the legend's visibility, and adding a
context menu to the legend now would couple this feature to a widget the Sound Map merge
may restructure.

**CP-39 — Dialog contents** (`gui/dialogs/cluster_playlists_dialog.py`,
`ClusterPlaylistsDialog(summary, preselect, colour_for=None, parent=None)`, modal):

- **Header:** method and parameter, number of clusters, tracks, analysis time, and the
  destination as a library-relative path (`Playlists/Clusters/<folder>/`).
  - If the map is downsampled, it adds "The map shows 5,000 of 20,000 tracks; the
    playlists include all of them."
- **Warnings area:** `summary.warnings`, covering rebase, new tracks, "changed a lot",
  older map, earlier sets kept and files left unchanged. Shown only when non-empty.
- **Options:** an *Order* combo (CP-24) and a *Minimum tracks* spin box (CP-25).
- **Table:** one row per cluster in `cluster_order` (real clusters ascending, noise last).
  The columns are:
  - include checkbox
  - colour swatch, as a `QIcon` from a filled `QPixmap` when `colour_for` is given (no
    `setStyleSheet`; ratchet)
  - **Name**, editable
  - **Tracks** (present)
  - **Missing** (tooltip lists up to 20 paths)
  - **Action**: New, Replace or Skip plus reason
- **Buttons:** *Select all*, *Select none*, *Open folder* (only when the run folder exists;
  `QDesktopServices.openUrl(QUrl.fromLocalFile(...))`, never `xdg-open`), and **Save N
  playlists**.
  - The Save button's label states the counts, e.g. "Save 6 playlists (2 replace)".
  - It is disabled when N = 0 or the plan is invalid (CP-18).
  - *Cancel* closes the dialog.
- **Re-planning:** every option or checkbox change, and every **committed** name edit
  (`itemChanged`, not each keystroke), calls `plan_cluster_playlists` again on the GUI
  thread. Because entries come precomputed (CP-22), a re-plan of a 20 000-track summary
  takes well under 100 ms.
- **Contract with the flow:** `dialog.plan()` returns the current plan. After `exec()`
  returns `Accepted`, the flow reads `dialog.plan()`. Tests drive the dialog through
  `dialogs.on_exec(handler)` and rely on this.
- **Theming:** objectNames and theme tokens only. No hex colours, `font-size` literals or
  inline stylesheets, so `tests/test_gui_hygiene_ratchets.py` counts do not rise.

**CP-40 — (removed in audit: no ▶ in the dialog in v1.)** `main_window.play_tracks`
switches the window to the Player page, which would happen behind this modal dialog and
leave the user on another page when it closes. Playing a cluster stays possible through the
legend and *Play selection*. A ▶ belongs to the non-modal Save playlist sheet follow-up.

**CP-41 — Result.**
On completion the flow logs the CP-31 summary line with level `ok` (or `error` when
`errors` is non-empty, followed by one line per `(path, msg)`), and emits
`result_ready(text, ok)` so the page can show it in its own status area. A refusal (CP-30)
is logged with level `error` and emitted with `ok=False`. Then it calls `refresh()`.

**CP-42 — Lasso export uses the shared writer (related clean-up, independent of
CP-1…CP-41; land it with or after the Sound Map merge).**
`_on_export_m3u` keeps its `QFileDialog` (the native overwrite prompt remains the
confirmation) and the `#EXTM3U` first line. Three things change:

- It first checks `_library_writes_blocked("Export playlist")`, before the file dialog.
- It builds `plan_playlist(paths, target)`, prepends `#EXTM3U` to `lines`, and writes with
  `write_playlist_plan([item], backup=True, log_dir=<library>/Docs)`. A `.m3u8` target is
  UTF-8 as today.
- Replacing an existing playlist therefore leaves a `.bak` and a log line.

CSV export is unchanged.

**CP-43 — Texts that stop being true.**

| Text | Where | Change |
|---|---|---|
| Results hint ("Playlists are not created automatically…") | `clustered_enhanced.py:596-599` | Becomes "Save a playlist per cluster with **Save cluster playlists…** (or export a lasso selection from the map)." The analysis page's flow button sits beside *Open Visual Graph* until the merge removes the tab. |
| Advanced config summary's "Output" lines | `_display_config` | Replace "Playlists are not written automatically…" with "Save cluster playlists… writes one playlist per cluster", plus "Offers to save them when the run finishes" when `create_playlists` is on. |
| Log line | `clustered_playlists.py:1089-1092` | Becomes "→ Saved cluster membership; use Save cluster playlists… to write one playlist per cluster." |
| Docstrings | `generate_clustered_playlists`, `clustered_enhanced` module | Updated. |

---

## 9. Wizard options (output step)

**CP-44 — "Create M3U playlists for each cluster" is enabled.**

- **Relabel:** "Offer to save a playlist per cluster when the analysis finishes". Default
  unchecked.
- **Tooltip:** "Opens the Save cluster playlists preview after a successful run. Nothing is
  written until you click Save."
- **Config:** it keeps the wizard run-config key `create_playlists`, saved by
  `_save_step(4)`. This is the wizard's in-memory config, not `~/.soundvault_config.json`,
  so `load_config`/`save_config` are not involved and nothing is persisted.
- **Effect:** `flow.offer_after_run(result)` calls `start(None)` (every cluster except the
  hidden ones) when `result["config"].get("create_playlists")` is true and `result` has a
  `run` block. If the page is not visible at that moment, the offer is held and made the
  next time the page is shown (the flow watches the page's `Show` event with an event
  filter), so no modal appears over another page. A failed or cancelled run never offers
  (the page only calls it on success). Quick Start never offers, because its config has no
  `create_playlists` key.
- **Wording:** the output page's intro and hint text are updated to match.

**CP-45 — These stay disabled.**

- *Generate cluster quality report*: the report is always offered when scores exist.
- *Open interactive graph after clustering*: moot once analysis and map share the Sound
  Map page, and the merge owns that decision.
- *Dimensionality reduction*.
- *Remove clusters smaller than threshold*, *Minimum cluster size* and *Merge small
  clusters into 'Miscellaneous'*. These change the clustering and the map, not just the
  playlists. The dialog's minimum size (CP-25) is the playlist-only equivalent.

Their tooltips are unchanged.

---

## 10. Tests

All new tests follow `test_<module_or_feature>.py`, use `tmp_path`/`monkeypatch`, and keep
module stubs scoped (`tests/_helpers/module_isolation.py`, `monkeypatch.setitem`). The suite
must stay green in default, reversed and shuffled order.

### 10.1 Shared fakes

- **New `tests/_helpers/cluster_payloads.py`:**
  - `make_cluster_run(n=20, n_clusters=3, noise=0, root="/music", downsample_to=None,
    tempo=True, energy=True, run_id="20261107-153012-abc123")` returns an aligned
    `(payload, members)` pair. Distances are `i * 0.1` (and `null` for noise).
  - `write_cluster_run(library, kind="valid"|"legacy"|"mismatched_run"|"misaligned_members"|"all_noise",
    files=False, **kw)` writes both files; with `files=True` it also creates the member
    files on disk under `library` (so presence checks pass).
  - Both builders validate their own output through `cluster_graph_data`, as
    `write_cluster_payload` does today.
- **`tests/ui/playlists_helpers.py`:** `write_cluster_payload` gains `with_members=True`,
  delegating to the helper above. `FakeClustering` gains `write_members=True`: it writes the
  members file and the `run` block, and returns `run` (and its `config` passes through
  the worker). GUI tests that save need at least `min_size × clusters` present files (for
  example 15 files for 3 clusters), or they lower the minimum in the dialog.
- **Tag stubs:** backend tests inject `tag_reader`. GUI tests monkeypatch
  `utils.audio_metadata_reader.read_metadata` (the `_no_disk_tags` fixture in
  `tests/ui/test_ui_playlists_graph.py`; copy it, it is file-local) or `read_tags` with an
  event-gated function to hold the summary job.

### 10.2 Backend

| Test | CP |
|---|---|
| `tests/test_playlists_clustering_backend.py::test_members_file_is_written_with_the_run_block` (uses the existing `fake_engine` fixture) | 1, 2, 3, 6 |
| `…::test_members_file_is_full_length_when_the_map_is_downsampled` (monkeypatch `MAX_VISUALIZATION_POINTS`) | 3, 9 |
| `…::test_both_files_share_one_run_id` · `…::test_cancel_writes_neither_file` (extend `test_cancel_during_extraction_writes_no_artefacts`) · `…::test_members_write_failure_is_logged_and_map_still_written` | 1, 2 |
| `…::test_centroid_distance_on_a_toy_matrix` (call `cluster_centroid_distances` directly; noise → `None`) | 4 |
| `…::test_tempo_and_energy_columns_are_raw_and_only_when_selected` (with and without MFCC selected) | 3 |
| `tests/test_cluster_graph_data.py::test_load_members_missing_invalid_misaligned` (incl. bool labels, null distance on a clustered track) · `::test_members_must_match_payload_run_and_arrays` (full and downsampled) | 7, 8, 9 |
| **New `tests/test_cluster_playlist_export.py`:** | |
| `test_legacy_payload_is_refused_with_reanalyse_message` · `test_run_mismatch_is_refused` · `test_all_noise_is_refused` · `test_all_missing_is_refused` | 12 |
| `test_missing_and_reserved_members_are_left_out_and_counted` · `test_presence_check_uses_long_paths` (monkeypatch `ensure_long_path` and the injected `exists`; assert it receives the long form) | 11 |
| `test_moved_library_root_is_rebased_with_warning` (members and `failed`) | 10 |
| `test_new_tracks_are_counted_and_reserved_folders_ignored` · `test_big_change_suggests_reanalysis` · `test_audio_exts_match_the_analysis_worker` | 13 |
| `test_summary_writes_nothing` (snapshot the tree before and after) · `test_summary_honours_cancel_event` · `test_summary_reports_progress` | 26 |
| `test_names_are_deterministic` · `test_name_uses_dominant_genre_tempo_and_energy` · `test_second_genre_is_appended_with_ampersand` · `test_artist_only_without_dominant_genre` · `test_various_artists_is_ignored` · `test_name_falls_back_to_cluster_number` · `test_cluster_ids_are_zero_padded_to_the_widest` · `test_tag_sample_is_capped_for_many_clusters` · `test_energy_needs_three_values` (injected `tag_reader`) | 14–16 |
| `test_sanitize_stem` (parametrised: reserved characters, control characters, leading/trailing dots and spaces, trailing dot after trimming, `CON`, `nul.mix`, `COM¹`, `CONIN$`, empty → `Cluster 03`) | 17 |
| `test_user_names_colliding_by_case_invalidate_the_plan` · `test_existing_file_matched_by_case_becomes_replace_with_exact_name` · `test_name_key_does_not_fold_sharp_s` | 18 |
| `test_noise_is_never_planned` | 19 |
| `test_run_folder_name_kmeans_and_hdbscan` · `test_same_second_folder_of_another_run_gets_a_suffix` | 20 |
| `test_earlier_sets_are_listed_and_untouched` · `test_resave_same_run_replaces_and_leaves_old_names` | 21 |
| `test_entries_equal_plan_playlist_lines_relative_native` (compare with `plan_playlist(...).lines`) · `test_header_lines_and_entry_count` | 22, 23 |
| `test_long_library_root_drops_parts_then_trims_then_skips` | 46 |
| `test_centre_and_library_orders` · `test_missing_members_are_dropped_from_the_order` | 24 |
| `test_min_size_counts_present_tracks_only` | 25 |
| `test_planner_does_no_file_io` (monkeypatch `os.path.exists`/`os.stat`/`builtins.open`/`os.path.relpath` to raise) · `test_replan_of_20k_members_is_fast` (fake summary, generous bound, e.g. < 0.5 s) | 28, 39 |
| `test_write_writes_exactly_the_plan` · `test_write_refuses_when_run_changed` · `test_write_uses_the_plans_library_root` · `test_replace_makes_bak_and_log_lines` · `test_file_appearing_since_preview_is_skipped` · `test_cancel_between_files` | 27, 28, 30, 31, 36 |
| `tests/test_writers_playlists.py::test_verify_skips_changed_and_appeared_files` · `::test_verify_writes_a_vanished_overwrite_as_new` · `::test_defaults_unchanged` · `::test_cancel_event_stops_between_files_and_still_logs` | 27 |

### 10.3 GUI (`tests/ui/`, headless; read `tests/ui/README.md`)

New file `tests/ui/test_ui_playlists_cluster_export.py`. Flow and dialog tests use a bare
host workspace where possible (`make_workspace` with any page key, then construct a
`ClusterPlaylistFlow(ws)`), so they don't depend on the page being restructured; page
integration tests use `make_workspace("graph", …)` (or the Sound Map key after the merge)
and are guarded by `pytest.importorskip("pyqtgraph")`. They use `dialogs`, `threads` and
`qtbot`, and stub tags as in §10.1. Tests that would open the run folder monkeypatch
`QDesktopServices.openUrl`.

| Test | CP |
|---|---|
| `test_button_disabled_without_members_file_with_tooltip` · `test_button_enabled_without_the_map_loaded` | 37 |
| `test_legacy_or_mismatched_data_shows_the_blocking_message` (information box text; nothing written) | 12, 29 |
| `test_preview_job_is_read_only` (hold the summary with a gated `read_tags`; inspect `job_registry().jobs(owner=ws)`: `ask_before_quit is False`, `group is None`) | 32 |
| `test_dialog_preticks_all_but_hidden_clusters` (hide one via `_legend._rows[cid].checkbox`) · `test_cluster_absent_from_downsampled_map_is_preticked` · `test_hidden_clusters_ignored_when_map_is_older` | 37 |
| `test_noise_row_is_not_checkable` | 19, 39 |
| `test_options_replan_rows` (min size → Skip; order → `dialog.plan().items[0].lines` order changes) | 24, 25, 39 |
| `test_cancelling_the_dialog_writes_nothing` (`on_exec(Rejected)`; no `Playlists/Clusters`; "Nothing was written." logged) | 29 |
| `test_save_writes_exactly_the_shown_plan` (compare files with `dialog.plan().items`) | 28, 33 |
| `test_write_job_takes_the_library_write_lock` (`group == LIBRARY_WRITES`, `ask_before_quit is True`) | 33 |
| `test_replace_asks_and_backs_up` (`answer_yes`; `.bak` + `BACKUP` in `Docs/playlist_writes.log`) · `test_replace_declined_writes_nothing` (`answer_no`) | 31, 29 |
| `test_busy_library_blocks_preview_and_save` (`tag_write_guard.acquire("Tag Fixer apply")`; "Library Busy" information box; nothing written) | 32, 33 |
| `test_button_disabled_while_an_analysis_runs_on_another_page` (`FakeClustering(hold=True)` on the Clustered page, flow on the graph page) · `test_analysis_page_uses_the_shared_job_name` | 34 |
| `test_cancel_preview_opens_no_dialog` (gated tag reader; click the button in "Cancel preview" mode) | 35 |
| `test_library_change_cancels_preview_but_lets_write_finish` (gate the write by monkeypatching `cluster_playlist_export.write_cluster_playlists` with an event-gated wrapper) | 36 |
| `test_result_is_logged_and_emitted` (`qtbot.waitSignal(flow.result_ready)`) | 41 |
| `test_closing_the_window_mid_write_asks` (`main_window`, `dialogs`, gated write as above) | 35 |
| `test_lasso_export_backs_up_logs_and_respects_the_lock` | 42 |
| `tests/ui/test_ui_playlists_clustering.py::test_wizard_offer_option_is_enabled_and_saved` · `::test_successful_run_with_offer_opens_preview` (`FakeClustering(write_members=True)`, `on_exec(Rejected)`) · `::test_offer_waits_until_the_page_is_shown` (`make_workspace(..., show=False)`, then show) · `::test_failed_or_cancelled_run_never_offers` · `::test_quick_start_never_offers` | 44 |

**Existing tests to update (not delete):**

- `test_wizard_does_not_offer_unbuilt_options`: drop `_create_playlists_cb` from the
  disabled list, and assert that the rest stay disabled (CP-45).
- `test_results_do_not_promise_cluster_playlists`: rename it to
  `test_results_point_to_save_cluster_playlists`, and assert the new hint and button
  (CP-43).

**Static checks:** `tests/test_gui_hygiene_ratchets.py` counts must not rise; lower a
ceiling if the change removes inline styles. `tests/ui/test_ui_static_qt_api.py` must pass.
(`scripts/gui_snapshots.py` enumerates workspaces only, not dialogs, so it needs no change.)

---

## 11. Definition of done

Done means all four of the following hold:

- Every CP is implemented (removed CPs excepted).
- §10 is green, including on the Windows CI job.
- These docs are updated:
  - `docs/features/playlists_and_clustering.md`: "From groups to playlists", the wizard
    step-5 text, and "What's real today".
  - `docs/ROADMAP.md`: remove "Playlists straight from a clustering run"; move the
    `create_playlists` wizard option out of "Controls that are still switched off"; add
    the deferred follow-ups (smooth-path order, legend shortcut, play from the dialog).
  - `docs/gui_inventory.md`: the new button and dialog.
  - `docs/architecture.md`: add `cluster_members.json` and the run folder to the
    "What each run leaves behind in Docs/" table, and extend the index-alignment note to
    the members file.
  - `CLAUDE.md`: extend the index-alignment bullet to `Docs/cluster_members.json`.
- `docs/redesign/pages/06_playlists_clustering_graph.md` PL-15 is updated: its
  "cluster playlists are never created" / "create playlists" part is marked fixed; the
  other ignored wizard options it lists stay open.

---

## 12. Open questions for the owner (each has a default, so implementation isn't blocked)

| # | Question | Default if unanswered |
|---|---|---|
| Q1 | When you save a new run, should earlier sets be replaced or kept? | **Keep** (per-run folders, never deleted). "Retire earlier sets" can follow later, using the CP-23 marker. |
| Q2 | Should HDBSCAN's unclustered tracks ever become a playlist? | **Never** in v1. |
| Q3 | Default order inside a playlist? | **Closest to the centre first.** |
| Q4 | Default minimum size? | **5** present tracks. |
| Q5 | Name shape: genre first, artist only as fallback, with BPM and energy words? | **Yes**, as in CP-16. |
| Q6 | `.m3u` or `.m3u8`? | **`.m3u`, UTF-8**, like every other generated playlist. Trade-off: a few older Windows players read `.m3u` as ANSI and garble non-ASCII names; `.m3u8` would avoid that but break consistency with the rest of the app. Revisit with the Save playlist sheet. |
| Q7 | Analyses made before this feature: refuse, or allow a partial export from the map data? | **Refuse** and ask for a re-analysis. A partial export would silently drop tracks from downsampled maps. |
| Q8 | Should the wizard's "offer after analysis" default to on? | **Off.** |
| Q9 | Should the destination be choosable? | **No**: always `<library>/Playlists/Clusters/<run>/` in v1. |
| Q10 | Should the dialog let you play a cluster before saving? | **Not in v1** (CP-40): it would switch pages behind a modal dialog. Comes with the non-modal Save playlist sheet. |
| Q11 | Is a "smooth path through the cluster" order worth building? | **Deferred** (CP-5). Centre-first and library order ship in v1. |

---

## Audit log (2026-11-07)

Independent audit of the first draft against the code at `9fe4248`. Every cited file and
line was opened; callers of every changed function were grepped. Changes, with reasons:

**Correctness against the code**

1. **CP-22, CP-28, §8.1: the planner could not be both pure and built on `plan_playlist`.**
   `plan_playlist` calls `os.path.exists`, so `test_planner_does_no_file_io` (which makes
   `exists` raise) was impossible. Entries are now computed once in the summary through a
   public alias of the same helper (`relative_playlist_lines`), and the planner builds
   `PlannedPlaylist` itself. This also fixes re-plan speed (item 12).
2. **CP-37: preselection from `visible_clusters()` silently dropped clusters.** On a
   downsampled map a small cluster can have no point, hence no legend row, so it would
   not be preticked. Preselection is now "all clusters of the run minus the hidden ones",
   and hidden ones only apply when the map shows the same run (visibility is remembered by
   id across reloads, so an older map's choices could leak onto a new run).
3. **CP-15: `A / B` genre pairs.** CP-17 turns `/` into `-`, which reads as a name-part
   separator. Changed to `A & B`.
4. **CP-30/CP-36, §8.1: `write_cluster_playlists(plan, library_root)` took the page's
   library.** After a library change it would check the wrong members file and log to the
   wrong `Docs/`. The plan now carries `library_root`.
5. **CP-3: `energy_rms` is not RMS with Essentia** (it is spectral-energy mean). Renamed
   to `energy`, documented that it is only compared within one run. Clarified that the
   tempo column offset counts only *selected* blocks.
6. **CP-12: an all-noise HDBSCAN run had no defined behaviour.** Added a sixth refusal.
7. **CP-8/CP-9:** labels must be `int` and not `bool`; null distances only for noise;
   the downsampled check now also verifies `X_total_points` and `len(X_indices)`; the
   check compares stored (unrebased) paths.
8. **CP-27:** dropped the `plan_playlist` snapshot change (unused by anything; gold-plate);
   defined the vanished-overwrite case; required the log to be appended on cancel (the
   writer appends once, after the loop); stated the result's new `cancelled` key and that
   every current caller (`gui/workspaces/playlists.py`, two test modules) is unaffected.
9. **CP-1:** placed the write after the last cancel check; compact JSON for 20k-track
   libraries; noted the `Docs/` fallback (Terms) and the pre-existing late-cancel race in
   `ClusterWorker` (left out of scope, recorded in §1).
10. **CP-44:** the wizard's `create_playlists` is in-memory run config, not
    `~/.soundvault_config.json`; said so. Quick Start's "never offers" is explained by its
    config having no such key.
11. §1 line references corrected (`:1089-1092`, `:596-599`, `:144-180`, `:336-372`,
    executor `:198-204`, `playlists.py:165`) and facts added (map may lack small clusters,
    `play_tracks` switches pages, analysis job name, the page loads data only with pyqtgraph).

**Windows**

12. **New CP-46: path-length budget.** `write_playlist_plan` opens plain paths; a deep
    library root plus a 100-character name plus a `.bak` suffix exceeded 260 characters.
13. **CP-17:** device-name check now covers `nul.mix.m3u`-style names, `CONIN$`/`CONOUT$`,
    `COM0`/`LPT0` and superscript digits (after NFKC); strip now runs after the trim and
    also strips leading dots.
14. **CP-11/CP-15:** presence checks and tag reads go through `ensure_long_path`, so
    tracks with long paths aren't reported missing.
15. **§3 name key:** `lower()` instead of `casefold()` for file identity (NTFS doesn't fold
    `ß`).
16. **CP-20:** same-second run folders could make a new run modify an earlier run's folder,
    contradicting CP-21; added a marker check with a ` (2)` suffix.
17. **CP-10:** rebasing also covers `failed`, and survives a cross-drive `relpath` error.
    **CP-22:** example now shows Windows separators. **Q6:** recorded the ANSI-player
    trade-off.

**Scope (cut or deferred)**

18. **CP-5 removed** (smooth-path order: O(n²), size limit, fallback, extra array).
    `path_rank` dropped from CP-3, CP-4 helper renamed, CP-24's `path` row removed.
19. **CP-38 removed** (legend context menu: a third route to S2, and coupling to a widget
    under restructuring).
20. **CP-40 removed** (dialog ▶ would switch the main window to the Player behind a modal
    dialog). S4 now relies on existing *Play selection*. Added Q10/Q11.

**Decoupling from the page restructure**

21. **§8.3/§8.4:** the flow now owns its button and exposes a five-item integration
    contract; colours, hidden clusters and the map's run id are injected callables, and
    results go out through `page._log` and `result_ready` instead of a page-specific status
    widget (CP-41). The Open-folder action on the result line was dropped (no page-agnostic
    place for it; the dialog has one).
22. **CP-37:** availability is a `stat` of the members file, not "the map loaded", so the
    feature works without pyqtgraph and doesn't depend on the map page's state.
23. **CP-34:** "analysis runs on the page" missed today's layout (analysis and map on
    different pages). The flow now watches the job registry for the `Clustering` job name.
24. **CP-13:** no longer moves `_AUDIO_EXTS` out of `clustered_enhanced.py` (a file being
    restructured); a test pins equality instead.

**Scale**

25. **CP-15:** tag sample capped at ~2 000 reads in total (HDBSCAN can give hundreds of
    clusters). **CP-39:** re-plan on committed edits only, with precomputed entries, and a
    20k-member speed test. **CP-26:** explicit cancel granularity and cost.

**Consistency and testability**

26. Fixed references (§9 → §10 for tests; §11 → §9 for the wizard), the "8.1" test row
    (now CP-28), CP-29's "every path logs" (now only terminal paths), CP-42 marked
    independent of the rest, the `_no_disk_tags` fixture noted as file-local, gated-write
    technique specified for the mid-write tests, the `scripts/gui_snapshots.py` hedge
    removed (it doesn't enumerate dialogs), and the DoD's "PL-15 is marked fixed" narrowed
    (PL-15 also covers wizard options that stay disabled).
