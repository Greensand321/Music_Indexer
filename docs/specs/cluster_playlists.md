# Spec: cluster playlists ("Save every cluster as a playlist")

_Status: ready for implementation · Written 2026-11-07 · Area: LISTEN → Sound Map
(`docs/redesign/README.md` §5) · Supersedes the ROADMAP item "Playlists straight from a
clustering run (Designed, not started)"._

## 0. Summary

A clustering run groups the library into clusters, and the map shows them. Today the
only way to turn a cluster into a playlist is to select it on the map and use
**Export playlist…**, one cluster at a time. That export writes the file directly, with no
backup and no log. This feature adds **Save cluster playlists…**. It is a previewed,
one-click export of one `.m3u` per cluster, written into a per-run folder under
`Playlists/Clusters/`, through the existing `playlist_generator` plan/write machinery.

The spec has three parts:

1. **Data.** The clustering run saves the full cluster membership and the ordering data in
   a new `Docs/cluster_members.json`. Today `cluster_info.json` is downsampled to 5 000
   points, so a playlist built from it would quietly lose tracks.
2. **Backend.** A new Qt-free module, `cluster_playlist_export.py`, does three things: a
   read-only summary (missing files, tags for naming), a pure planner (names, order,
   minimum size → `PlannedPlaylist` items), and a write step. The write step is a thin
   wrapper around `playlist_generator.write_playlist_plan`.
3. **UI.** A dialog on the Sound Map page. A page-agnostic flow object starts the jobs, so
   the same code works on today's Graph page, today's Clustered Results tab, and the merged
   Sound Map page.

Requirements are numbered **CP-n**. Each one is testable, and §9 maps tests to them.

---

## 1. Current state (verified against the code)

| Fact | Where |
|---|---|
| A clustering run writes the feature cache, `Docs/cluster_info.json` and `Docs/cluster_graph.html`, and **no playlists**. It logs "Cluster playlists are not written automatically…". | `clustered_playlists.generate_clustered_playlists` (docstring at `:944`, log line at `:1090`) |
| `cluster_info.json` is **downsampled** to `MAX_VISUALIZATION_POINTS = 5000` points. `X`/`X_2d`/`X_3d`/`labels`/`tracks` are subset together, and `X_downsampled`, `X_total_points` and `X_indices` record this. Full-length labels/tracks exist only in the in-memory result. | `build_cluster_payload` (`:892`), `visualization_indices` (`:876`) |
| The payload records **no run identity**: no timestamp, method, params, feature selection or library root. `cluster_info[c]` carries `size`, `genres: []` and `tempo: [0, 0]` (placeholders). | `generate_clustered_playlists` (`:1094-1103`) |
| The returned result is index-aligned: `tracks`, `X` (**scaled**) and `labels`, plus `features` (the **raw, unscaled** vectors), `failed` (path → error) and `cluster_info`. | `:1149-1157` |
| A feature vector's blocks are always in `FEATURE_ORDER = (mfcc 26, tempo 1, chroma 12, spectral 2, energy 2, onset_rate 1)`. Only the selected blocks are present, so the BPM column and the RMS-mean column can be located from the selection alone. | `FEATURE_ORDER`, `FEATURE_DIMENSIONS`, `_assemble_feature_vector` |
| On cancel, `ClusteringCancelled` is raised before any artefact is written. Extracted vectors stay cached. | `:1044-1050`, `:1065`, `:1106`, `:1117` |
| The preview-first playlist API is: `plan_playlist(tracks, outfile) → PlannedPlaylist(path, lines, exists, entries, encoding="utf-8")`, where lines are `safe_relpath` (relative, native separators, absolute only across Windows drives); and `write_playlist_plan(plan, backup=True, log_callback, log_dir)`, which writes `item.lines` verbatim with `\n`, makes `name.m3u.<YYYYmmdd-HHMMSS>.bak` beside any file that exists **at write time**, and appends `BACKUP`/`WRITE`/`ERROR` lines to `<log_dir>/playlist_writes.log`. It has no cancel and no "changed since preview" check. | `playlist_generator.py:524-560`, `:666-722` |
| Rewriting existing entries keeps their style: `playlist_entry(target, dir, like=entry)` keeps absolute/relative and the separator. New entries (no `like`) are relative with native separators. | `utils/path_helpers.py` |
| `update_playlists` (run after Indexer moves) and the Duplicates executor walk `Playlists/` **recursively**, and Repair's empty-list mode uses `rglob`. So playlists under `Playlists/Clusters/…` stay valid after later moves and quarantines, with no extra work. | `playlist_generator.py` (`os.walk(playlists_dir)`), `duplicate_consolidation_executor.py:202`, `gui/workspaces/playlists.py` (`rglob`) |
| `cluster_graph_data` is the Qt-free, numpy-free loader and validator (`load_cluster_data`, `validate_cluster_payload`, `NOISE_LABEL = -1`, `cluster_order`, `selected_tracks`, `selection_to_m3u`, which emits `#EXTM3U`). | `cluster_graph_data.py` |
| The Graph page's **Export playlist…** writes `selection_to_m3u` text with `Path.write_text`, after `QFileDialog.getSaveFileName`. It makes no backup, writes no log and doesn't check the library-write lock. | `gui/workspaces/graph.py:369-385` |
| Legend checkboxes mean **visibility** (`cluster_toggled`). A click on a row selects the cluster. A right-click also selects, because `mousePressEvent` emits `selected` for every button. Swatches use an inline `setStyleSheet`. | `gui/widgets/cluster_legend.py` |
| The wizard's output step has **Create M3U playlists for each cluster** (`_create_playlists_cb`), **Generate cluster quality report** and **Open interactive graph after clustering**, all disabled. Dimensionality reduction and post-processing are disabled too. The config keys `create_playlists`, `generate_report` and `open_graph` exist and nothing reads them. | `gui/dialogs/clustering_wizard_dialog.py:338-372`, `:417-438` |
| The Clustered Results tab says "Playlists are not created automatically…". Tests pin both that text and the disabled wizard option. | `clustered_enhanced.py:596`; `tests/ui/test_ui_playlists_clustering.py::test_results_do_not_promise_cluster_playlists`, `::test_wizard_does_not_offer_unbuilt_options` |
| Job rules: `_start_worker(w, name, ask_before_quit=…, library_write=…)` and `_library_writes_blocked(action)`. The latter also honours `controllers.tag_write_guard`. | `gui/workspaces/base.py:144-180`, `gui/workers.py` (`Job.ask_before_quit`, `Job.group`, `LIBRARY_WRITES`) |

---

## 2. User stories and scope

| # | Story | v1? |
|---|---|---|
| S1 | **Save every cluster as a playlist.** "I clustered my library into 8 moods; give me 8 playlists." | **In** |
| S2 | **Save just these clusters.** "Only clusters 2, 5 and 6 are any good." Untick in the dialog, hide others in the legend first, or right-click a legend row → *Save as playlist…* | **In** |
| S3 | **Regenerate after re-clustering.** "I re-ran with k=12; save the new set." Each run gets its own folder; earlier sets are kept (§6). Re-saving the *same* run replaces its files, with backups. | **In** |
| S4 | **Play a cluster.** "Let me hear cluster 3 in the order it will be saved." A ▶ on each dialog row plays the planned order (the full membership, present files only) through the Player. Selecting a cluster in the legend and pressing *Play selection* on the map keeps working as it does today. | **In** (dialog ▶). Legend "Play this cluster" is **out** |
| S5 | **Offer it right after an analysis.** The wizard option "Offer to save a playlist per cluster when the analysis finishes" opens the same preview. | **In** |

**Out of v1** (each is a clean follow-up):

- Writing without the preview, ever.
- Deleting or retiring earlier run folders.
- A playlist of HDBSCAN noise.
- A per-playlist track cap ("the 50 most central").
- Remembering custom names across runs.
- Showing names in the map legend.
- `.m3u8`, absolute-path or forward-slash output options, and `#EXTINF` duration lines.
- A destination outside `<library>/Playlists/Clusters/`.
- Payloads written before this feature (they must be re-analysed, CP-12).
- The legacy Tk app.
- The redesign's shared **Save playlist sheet**. The dialog should be built so it can grow into that sheet, but v1 doesn't need to be it.

---

## 3. Terms

- **Run.** One successful clustering run. It is identified by `run_id` and described by the `run` block (CP-2).
- **Members file.** `<library>/Docs/cluster_members.json`, the full and never-downsampled membership of a run (CP-3).
- **Map payload.** `<library>/Docs/cluster_info.json`, as today, plus the `run` block.
- **Run folder.** `<library>/Playlists/Clusters/<run folder name>/` (CP-20).
- **Summary.** The output of the read-only phase (CP-26): the clusters, which members are present or missing, naming facts, the existing files in the run folder, and any warnings.
- **Plan.** The output of the pure planner (CP-28): rows for the dialog, plus the `PlannedPlaylist` items that will be written.

---

## 4. Data contract

### 4.1 What the clustering run writes

**CP-1 — Members file is written by every successful run.**
`generate_clustered_playlists` writes `Docs/cluster_members.json` using the same atomic
`tmp` + `os.replace` pattern as `cluster_info.json`. It writes the members file **before**
`cluster_info.json`. When a run is cancelled, neither file is written (this extends the
existing `ClusteringCancelled` rule). If the members file fails to write, the error is
logged as `! Could not save cluster membership (cluster playlists unavailable for this run): …`,
and `cluster_info.json` is still written.

**CP-2 — Both files carry the same `run` block.**
The map payload gains a top-level `"run"` key, and the members file carries the identical
object:

| key | value |
|---|---|
| `schema` | `1` |
| `run_id` | `<YYYYmmdd-HHMMSS>-<6 hex>`: the local start time of the run plus a random token, so two runs in one second still differ |
| `created` | ISO-8601 local time of the run start (`run_started`) |
| `library_root` | `os.path.abspath(root_path)` |
| `method` | `"kmeans"` or `"hdbscan"` (after `normalize_method`) |
| `params` | the `params` dict actually passed to `cluster_tracks` |
| `features` | `feature_selection_key(selection)`, e.g. `"mfcc-tempo-energy"` |
| `feature_engine`, `normalization` | as used |
| `track_count` | number of clustered tracks (= length of the members arrays) |
| `failed_count` | `len(failed)` |

`build_cluster_payload` keeps its alignment behaviour unchanged. The `run` key is added
beside the arrays. `cluster_graph_data.validate_cluster_payload` ignores unknown keys, so it
needs no change.

**CP-3 — Members file schema (index-aligned, full length).**

```json
{
  "run": { "...": "identical to cluster_info.json's run block" },
  "tracks":            ["/music/A/B/01 x.flac", "..."],
  "labels":            [3, 0, -1, "..."],
  "centroid_distance": [0.84, 1.92, null, "..."],
  "path_rank":         [0, 17, null, "..."],
  "tempo_bpm":         [121.4, 98.0, "..."],
  "energy_rms":        [0.112, 0.034, "..."],
  "failed":            ["/music/C/D/broken.mp3"]
}
```

- `tracks`, `labels`, `centroid_distance` and `path_rank` all have length `run.track_count`.
  Element *i* of each describes the same track, in the same order as the run's returned
  `tracks`. **The CLAUDE.md index-alignment rule applies to this file too**, and nothing may
  subset one array without the others. The members file is **never downsampled**.
- `tempo_bpm` is present only when the `tempo` block was selected, and `energy_rms` only
  when `energy` was. Both are taken from the **raw** `features` vectors, not the scaled `X`.
  The tempo column is the block's offset in `FEATURE_ORDER`. The energy value is the
  *first* number of the energy block (RMS mean). Each is the same length as `tracks` when
  present. `null` means not available.
- Noise tracks (`label == -1`) have `centroid_distance: null` and `path_rank: null`.
- `failed` lists the paths that could not be analysed, as the keys of the run's `failed`
  mapping.

**CP-4 — Centroid distance.**
For every cluster `c ≥ 0`, the centroid is the **mean of the scaled vectors `X`** of its
members. The same rule holds for K-Means and HDBSCAN. `centroid_distance[i]` is the
Euclidean distance from `X[i]` to its cluster's centroid. This is computed in a new helper,
`clustered_playlists.cluster_member_order(X, labels, path_limit=PATH_ORDER_MAX_TRACKS)`,
which returns `(centroid_distance, path_rank)` as lists.

**CP-5 — Smooth-path rank.**
For each cluster with at most `PATH_ORDER_MAX_TRACKS = 5000` members, `path_rank` gives each
member's position in a greedy nearest-neighbour chain through the scaled `X`:

- The chain starts at the member with the smallest centroid distance. On ties, the lowest
  index starts.
- Each next track is the nearest *remaining* member by Euclidean distance. On ties, the
  lowest index wins.
- Ranks run `0 … n_c − 1`.

Clusters larger than the limit get `null` ranks, because the chain is O(n²) and would stall
the run. This is the Auto-DJ walk (`playlist_engine.autodj_playlist`), but on the *scaled*
vectors the clustering used.

**CP-6 — Returned result.**
The dict returned by `generate_clustered_playlists` gains `"run"` (the block) and
`"members_file"` (path or `None`). Existing keys are unchanged. `ClusterWorker` passes them
through untouched.

### 4.2 Loading and validation (in `cluster_graph_data`, still Qt- and numpy-free)

**CP-7 — `cluster_members_path(library)`** returns `<library>/Docs/cluster_members.json`.
**`load_cluster_members(library)`** reads it, validates it, and returns the dict. It raises
`ClusterDataError` with an actionable message in four cases: the file is missing, it is
unparseable, it is not an object, or it lacks `run`, `run.run_id`, `tracks` or `labels`.

**CP-8 — `validate_cluster_members(data)`** raises `ClusterDataError` in three cases:

- `tracks`, `labels`, `centroid_distance` and `path_rank` don't all have the length
  `run.track_count`.
- `tempo_bpm` or `energy_rms` is present with a different length.
- `labels` contains a non-integer.

**CP-9 — `check_members_match_payload(payload, members)`** raises `ClusterDataError("…out
of step with the map… Re-run the analysis.")` in either of two cases:

- `payload["run"]["run_id"] != members["run"]["run_id"]`.
- The arrays disagree. The check depends on how the payload was written:
  - If `X_downsampled` is false, `payload.tracks == members.tracks` and
    `payload.labels == members.labels`.
  - If it is true, then for every *i*, `members.tracks[X_indices[i]] == payload.tracks[i]`
    and the labels match too.

### 4.3 Stale data and missing files

**CP-10 — Library root rebasing.**
Compare `run.library_root` with the current library path, using `normcase(abspath())` on
both. If they differ, every track under the old root is mapped to
`join(current_root, relpath(track, old_root))`. The summary records
`rebased_from = old_root` and shows a warning: "This analysis was made for `<old>`; paths
were mapped to the current library."

**CP-11 — Present, missing and reserved.**
Every member is checked with `os.path.exists` (injectable) after rebasing. A member that is
missing, or that `utils.reserved_folders.is_reserved_path(track, root)` reports as reserved,
is **left out** of its playlist. It is counted per cluster as `missing`, and its paths are
kept for the dialog's tooltip. The planner never re-points a missing track.

**CP-12 — Refusals (blocking).**
The summary returns `blocking_error` instead of clusters in five cases:

| Case | Message |
|---|---|
| No `cluster_info.json` | the existing `load_cluster_data` text |
| Payload has no `run` block, or no members file (legacy payload) | "This map was made before cluster playlists existed. Re-run the analysis to save a playlist per cluster. (Export playlist… still works for a selection.)" |
| CP-8 fails | its message |
| CP-9 fails | its message |
| Every clustered member is missing | "None of the analysed tracks are where they were; re-run the analysis." |

**CP-13 — New tracks since the analysis (warning, not blocking).**
The summary walks the library with `utils.reserved_folders.iter_library_files`, using the
audio extensions in `gui/workspaces/clustered_enhanced._AUDIO_EXTS`. Move that set to a
shared constant in `cluster_playlist_export` and import it from there. It counts the files
that are neither in `members.tracks` (after rebasing) nor in `members.failed`, and reports
"N tracks were added after this analysis and are in no cluster." If
`(missing_total + new_tracks) > 10 %` of `run.track_count`, it adds "Your library changed a
lot since this analysis; consider re-analysing." Tracks retagged or re-encoded in place
are **not** detected (documented limitation).

---

## 5. Naming

**CP-14 — Deterministic names.**
With the same members file, the same tags and the same options, two summaries produce the
same proposed names. Names come only from this data and never from wall-clock time or set
iteration order.

**CP-15 — Descriptor facts per cluster.**

1. **Tag sample.** Take the up-to-50 **present** members with the smallest
   `centroid_distance`. Ties go to `casefold(path)`. Read their tags with a tag reader. The
   default is `utils.audio_metadata_reader.read_tags`, looked up on that module at call
   time so tests can monkeypatch it. Reading at most `50 × clusters` files bounds the cost,
   and the most central tracks are the most representative.
2. **Genre.** Take the `genre` value (`read_tags` returns the first genre), stripped.
   Count it case-insensitively, and display the most frequent original spelling (ties are
   broken alphabetically by casefold). The *dominant genre* is the top genre when at least
   3 sampled tracks carry a genre and the top share is ≥ 40 % of them. If the runner-up's
   share is ≥ 25 %, append it as `A / B`.
3. **Artist** (used only when there is no dominant genre). Use `albumartist`, falling back
   to `artist`. Ignore `Various Artists` (casefold) and empty values. It qualifies when it
   covers ≥ 50 % of the sample and at least 3 tracks.
4. **Tempo** (only when `tempo_bpm` exists). Take the median BPM over **all** present
   members with a value > 0, provided at least half of them have one. Round it to the
   nearest 5 and render `~120 BPM`.
5. **Energy** (only when `energy_rms` exists). Compute the library terciles `t1` and `t2`
   (33rd and 67th percentiles, using `statistics.quantiles` over every non-null value in
   the members file). A cluster median below `t1` renders `low energy`, and one above `t2`
   renders `high energy`. A medium cluster gets no energy word.

**CP-16 — Name shape.**

`Cluster <id> - <primary> - <tempo> - <energy>`

- Missing parts are dropped. `<primary>` is the dominant genre, or the artist when there is
  no dominant genre.
- `<id>` is the run's cluster id, the same number the legend shows ("Cluster 3"),
  zero-padded to the width of the largest id in the run, minimum 2 digits. The padding
  makes files sort in cluster order.
- When no part qualifies, the name is `Cluster 03`.
- Examples: `Cluster 03 - Deep House - ~120 BPM - high energy`,
  `Cluster 07 - Radiohead`, `Cluster 11`.

The name is used both as the file stem and as the `#PLAYLIST:` title (CP-23).

**CP-17 — Sanitising.**
`sanitize_playlist_stem(text)` applies these rules in order:

1. NFKC-normalise.
2. Remove control characters.
3. Replace each of `<>:"/\|?*` with `-`.
4. Collapse runs of whitespace.
5. Strip leading and trailing spaces, and trailing dots.
6. If the whole stem is a Windows device name (`CON PRN AUX NUL COM1-9 LPT1-9`,
   case-insensitive), append `_`.
7. Trim to 100 characters. Generated names drop whole descriptor parts from the right
   before any hard trim.

If the result is empty, it becomes `Cluster <id>`. This is a new function. Don't import
`music_indexer_api.sanitize`: it pulls in the indexer stack and doesn't handle trailing
dots or device names.

**CP-18 — User-edited names and collisions.**
The dialog lets the user edit each name, and the edit is sanitised by CP-17. Two rules
apply:

- **Inside the plan:** two rows whose stems are equal under casefold make the plan invalid.
  Save is disabled, and both rows show "Two playlists would be called X".
- **Against existing files:** files already in the run folder (listed during the summary)
  are compared under casefold. A match makes the row's action **Replace**, and the item
  targets the existing file's *exact* name. On a case-sensitive file system this avoids
  writing `house.m3u` beside `House.m3u`.

Run folders are per-run, so a name can never collide with a playlist outside its own run
folder. That includes the user's own playlists in `Playlists/`. Custom names are not
persisted (v1).

**CP-19 — Noise is never a playlist.**
Label `-1` produces no plan item under any option. The dialog shows it as a disabled row,
"Unclustered (N) — not saved".

---

## 6. Destination, format and re-runs

**CP-20 — Run folder.**
The run folder is `<library>/Playlists/Clusters/<YYYY-MM-DD HH.MM.SS> <Method><param>/`,
built by `run_folder_name(run)`:

- `<Method>` is `K-Means` or `HDBSCAN`.
- `<param>` is ` (k=8)` for K-Means and ` (min 5)` for HDBSCAN.

Example: `Playlists/Clusters/2026-11-07 15.30.12 K-Means (k=8)/`. The folder name has no
colons, so it is legal on Windows. `write_playlist_plan` creates folders as needed. Two
runs started in the same second share a folder name. Their files then follow the replace
rule (CP-18), and the `#ALPHADEX-CLUSTER` header (CP-23) records which run wrote each file.

**CP-21 — Re-runs keep history; re-saves replace.**
Saving a *different* run writes into that run's own new folder, and earlier run folders are
**never** modified or deleted. Saving the *same* run again (for example after first saving
only some clusters) writes into the same folder. Same-named files are replaced and backed
up (CP-31). Files from an earlier save of that run whose names no longer appear are left
in place, and the dialog lists them as "left unchanged". The summary lists the other folders
under `Playlists/Clusters/` (`earlier_sets`), and the dialog says "N earlier cluster
playlist sets are kept in Playlists/Clusters/", with an *Open folder* action.

**CP-22 — Entry style and encoding.**
Each playlist's entries are exactly `playlist_generator.plan_playlist(tracks, outfile).lines`.
They are paths relative to the run folder with **native separators**, such as
`../../../Artist/Album/01 x.flac`, falling back to absolute paths only when no relative
path exists (another Windows drive). This is the same rule every other generated playlist
follows. Later rewrites by `update_playlists` and Repair keep each entry's own style
(`playlist_entry(…, like=entry)`).

Files are `.m3u`, UTF-8 without a BOM, with `\n` line endings: the `PlannedPlaylist`
default `encoding="utf-8"` as written by `write_playlist_plan`. A path that UTF-8 cannot
encode (an undecodable POSIX file name) makes that one file fail. The failure is reported in
`errors` and the file is not written, which is existing writer behaviour.

**CP-23 — Header lines.**
Three header lines are prepended to `plan_playlist`'s lines:

1. `#EXTM3U`
2. `#PLAYLIST:<name>`
3. `#ALPHADEX-CLUSTER:run=<run_id>;cluster=<id>`

`PlannedPlaylist.entries` stays the track count. Every reader in the repo skips `#` lines.
`_rewrite_playlist_lines` treats `#PLAYLIST`/`#ALPHADEX` as plain comments, so they survive
renames. The marker exists so a later "retire earlier sets" feature can recognise
generated files safely.

---

## 7. Ordering and minimum size

**CP-24 — Order options** (dialog combo; default **Closest to the centre first**):

| Option | Key | Rule |
|---|---|---|
| Closest to the centre first | `centre` | Present members sorted by `(centroid_distance, casefold(path))`. The most typical tracks come first. |
| Smooth path through the cluster | `path` | Present members sorted by `path_rank`. Missing members are dropped and the relative order is kept. If the cluster's ranks are `null` (over the limit, CP-5), the cluster falls back to `centre`, and its row notes "smooth path not available for clusters over 5,000 tracks". |
| Library order | `library` | Present members sorted by `casefold(path)`, which is artist/album/track order for an indexed library. |

**CP-25 — Minimum size.**
A spin box from 2 to 500, default **5**, matching `DEFAULT_HDBSCAN_MIN_SIZE`. It counts
**present** tracks. A cluster below it gets action **Skip** with reason "too small (N
tracks)", and is not ticked by default. Ticking it has no effect while it is below the
minimum: the checkbox is disabled, with that tooltip. This setting is not the wizard's
"Remove clusters smaller than threshold", which would change the clustering itself
(§11).

---

## 8. Flow, modules and UI

### 8.1 Backend: `cluster_playlist_export.py` (new, top level, no Qt, no numpy)

| Function | Does | I/O |
|---|---|---|
| `summarize_cluster_run(library_root, *, tag_reader=None, exists=os.path.exists, cancel_event=None, progress_callback=None, log_callback=None) -> RunSummary` | Loads the payload and members file (CP-7…CP-9), rebases (CP-10), checks presence (CP-11), counts new tracks (CP-13), samples tags and computes descriptor facts (CP-15), lists the run folder's existing `.m3u`/`.m3u8` files and `earlier_sets` | Read-only |
| `plan_cluster_playlists(summary, cluster_ids, *, order="centre", min_size=5, names=None) -> ClusterPlaylistPlan` | Builds the rows and items (CP-16…CP-25). Items are in ascending cluster id. Each item comes from `plan_playlist(...)` with the CP-23 header prepended. `exists` is taken from the summary's listing, not from the disk | **Pure**: no file-system access |
| `write_cluster_playlists(plan, library_root, *, cancel_event=None, log_callback=None) -> dict` | Re-reads the members file's `run.run_id`; refuses if it changed (CP-30); otherwise calls `write_playlist_plan(plan.items, backup=True, log_dir=<library>/Docs, verify=True, cancel_event=…)` | Writes |
| `run_folder_name(run)`, `sanitize_playlist_stem(text)`, `AUDIO_EXTS` | CP-20, CP-17, CP-13 | none |

`RunSummary` is a dataclass with these fields:

- `run`, `run_folder` (absolute path), `rebased_from`
- `clusters: dict[int, ClusterFacts]`, where each `ClusterFacts` holds the id, present paths
  with their distances and ranks, missing paths, descriptor parts, the proposed name and
  `path_available`
- `noise_count`, `missing_total`, `new_tracks`, `map_points`, `downsampled`
- `existing_files` (casefold → exact name), `earlier_sets`, `warnings`, `blocking_error`

`ClusterPlaylistPlan` holds `run_id`, `run_folder`, `rows`, `items`, `errors` and `ok`.
Each row is `(cluster_id, name, tracks, missing, action ∈ {new, replace, skip}, reason)`.

**CP-26 — The summary is read-only and cancellable.**
It never creates, modifies or deletes a file. It checks `cancel_event` between members and
between tag reads, and raises `ClusterPlaylistsCancelled` when the event is set. It reports
`progress_callback(done, total)` over the presence checks and the tag reads.

**CP-27 — Shared writer extensions** (`playlist_generator`, backward compatible):

- `PlannedPlaylist.snapshot: tuple | None = None`. `plan_playlist` sets it to
  `(st_mtime_ns, st_size)` when the file exists. The cluster planner copies the snapshot
  from the summary's listing.
- `write_playlist_plan(..., verify=False, cancel_event=None)`.
  - With `verify=True`, two kinds of item are **skipped** and reported in `errors`, with an
    `ERROR … changed since the preview` log line: an item planned as new whose file now
    exists, and an item planned as an overwrite whose `(mtime_ns, size)` no longer equals
    `snapshot`.
  - With `cancel_event` set, writing stops **between** files. The result gains
    `"cancelled": True`, and the files already written stay written and logged.
  - With the defaults, the behaviour is byte-for-byte today's, and the existing tests are
    unchanged.

**CP-28 — The plan is exact.**
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
Opening the dialog, changing options and pressing ▶ write nothing. Cancelling the dialog
writes nothing. Declining the replace question writes nothing. A blocked lock writes
nothing. Every one of these paths logs "Nothing was written." to the page log.

**CP-30 — Plan signature.**
Two checks guard against changes since the preview.

- *Run change:* if `cluster_members.json`'s `run_id` changed between the summary and the
  write (a new analysis landed), the write refuses the **whole** plan with "The analysis
  changed since the preview; open Save cluster playlists… again." and writes nothing.
- *File change:* a file in the run folder that appeared or changed since the preview is
  skipped (CP-27 `verify`).

**CP-31 — Backups and log.**
Every replaced file first gets a `name.m3u.<YYYYmmdd-HHMMSS>.bak` beside it. Every backup,
write and error is appended to `<library>/Docs/playlist_writes.log`. Both come from the
existing writer. The page log gets `log_callback` lines and one summary line: "Saved N
cluster playlist(s) to Playlists/Clusters/<folder>/ (K replaced, backed up as .bak; E
failed)."

### 8.3 Jobs, lock and cancel: `gui/cluster_playlist_jobs.py` (new, page-agnostic)

`ClusterPlaylistFlow(page: WorkspaceBase)` is a `QObject` owned by the page. It holds
`ClusterPlaylistSummaryWorker` and `ClusterPlaylistWriteWorker`, which are `QThread`s whose
`done` signal is not named `finished`, each with a real `cancel()` backed by a
`threading.Event`. Its interface:

- `start(preselect: set[int] | None)`
- `cancel()`
- `is_busy()`
- `on_library_changed()`
- `busy_changed(bool)`, `progress(int, int)`

**CP-32 — Read-only job.**
The summary job starts as `page._start_worker(w, "Cluster playlist preview",
ask_before_quit=False)`. Its `Job.group` is `None` and `Job.ask_before_quit` is `False`.
`start()` first calls `page._library_writes_blocked("Save cluster playlists")` and does
nothing if that returns true: while another page is moving files, a presence check would be
meaningless.

**CP-33 — Write job.**
After the dialog is accepted, the flow calls `_library_writes_blocked("Save cluster
playlists")` again. When the plan contains any Replace row, it then asks
`QMessageBox.question`. Last, it starts `page._start_worker(w, "Save cluster playlists",
library_write=True)`, with `ask_before_quit` left at its default `True`. The worker
receives the `ClusterPlaylistPlan` object the dialog showed, not a recomputation.

**CP-34 — Busy rules.**

- While either job runs, **Save cluster playlists…** and the legend's *Save as playlist…*
  are disabled, and the page's analysis **Run** buttons are disabled too.
- While an analysis job runs on the page, **Save cluster playlists…** is disabled with the
  tooltip "Wait for the analysis to finish."
- Slots ignore any signal whose `sender()` is not the current worker (the PL-07 pattern).

**CP-35 — Cancel.**

- While the summary job runs, the page shows a visible **Cancel**. Use the page's job card
  if the merged Sound Map has one; otherwise the Save button's text becomes "Cancel" and
  cancels. A cancelled summary opens no dialog and logs "Cancelled. Nothing was written."
- The write job's `cancel()` stops between files (CP-27). Quitting mid-write goes through
  the main window's ask, cancel and wait (`ask_before_quit=True`).

**CP-36 — Library change.**
A library change cancels a running summary and drops its result. A running write is left to
finish in the old library, which is the Playlists-page rule: it writes exactly the
confirmed plan. Any pending auto-offer (CP-44) is cleared.

### 8.4 UI on the Sound Map page

Integration contract: whichever class ends up as the Sound Map page (today `GraphWorkspace`
in `gui/workspaces/graph.py`; the merge is in progress) owns a `ClusterPlaylistFlow` and
provides the controls below. Until the merge lands, the Clustered Results tab gets the same
button, wired to its own flow instance (CP-43).

**CP-37 — Primary action.**
A **Save cluster playlists…** button sits in the map's action bar, beside *Play selection* /
*Export CSV…* / *Export playlist…*, but it works independently of the selection.

- **Enabled when:** a payload loaded **and** it has a `run` block, and no job blocks it
  (CP-34).
- **Disabled otherwise**, with a tooltip:
  - no data: "Analyse your library first."
  - legacy payload: the CP-12 legacy text
  - busy: CP-34
- **Preselection:** every non-noise cluster that is **visible** in the legend
  (`ClusterLegendWidget.visible_clusters()`).

**CP-38 — Legend shortcut.**
Right-clicking a legend row opens a context menu with **Save as playlist…**, which starts
the flow with `preselect={cluster_id}`. Noise rows have no menu. The legend gains a
`cluster_save_requested(int)` signal for this. The legend checkbox keeps meaning
*visibility*. It is never reused to mean "include".

**CP-39 — Dialog contents** (`gui/dialogs/cluster_playlists_dialog.py`,
`ClusterPlaylistsDialog(summary, preselect, parent)`, modal):

- **Header:** method and parameter, number of clusters, tracks, analysis time, and the
  destination as a library-relative path (`Playlists/Clusters/<folder>/`).
  - If the map is downsampled, it adds "The map shows 5,000 of 20,000 tracks; the
    playlists include all of them."
- **Warnings area:** `summary.warnings`, covering rebase, new tracks, "changed a lot" and
  earlier sets kept. Shown only when non-empty.
- **Options:** an *Order* combo (CP-24) and a *Minimum tracks* spin box (CP-25).
- **Table:** one row per cluster in `cluster_order` (real clusters ascending, noise last).
  The columns are:
  - include checkbox
  - colour swatch, as a `QIcon` from a filled `QPixmap` (no `setStyleSheet`; ratchet)
  - **Name**, editable
  - **Tracks** (present)
  - **Missing** (tooltip lists up to 20 paths)
  - **Action**: New, Replace or Skip plus reason
  - **▶** (plays the row's planned order)
- **Buttons:** *Select all*, *Select none*, *Open folder* (when the run folder exists;
  `QDesktopServices.openUrl`, never `xdg-open`), and **Save N playlists**.
  - The Save button's label states the counts, e.g. "Save 6 playlists (2 replace)".
  - It is disabled when N = 0 or the plan is invalid (CP-18).
  - *Cancel* closes the dialog.
- **Re-planning:** every option or checkbox change calls `plan_cluster_playlists` again.
  It is pure and fast, so it runs on the GUI thread.
- **Contract with the flow:** `dialog.plan()` returns the current plan. After `exec()`
  returns `Accepted`, the flow reads `dialog.plan()`. Tests drive the dialog through
  `dialogs.on_exec(handler)` and rely on this.
- **Theming:** objectNames and theme tokens only. No hex colours, `font-size` literals or
  inline stylesheets, so `tests/test_gui_hygiene_ratchets.py` counts do not rise.

**CP-40 — Play from the dialog.**
▶ emits `play_requested(paths, name)`, with the row's planned order and present tracks
only. The flow forwards it as `page.play_tracks_requested.emit(paths, name)`. Pressing ▶
writes nothing.

**CP-41 — Result.**
On completion, the page status and log show the CP-31 summary line, along with an *Open
folder* action. If `errors` is non-empty, the status role is `error` and the log lists each
`(path, msg)`.

**CP-42 — Lasso export uses the shared writer (related clean-up; coordinate with the Sound
Map merge).**
`_on_export_m3u` keeps its `QFileDialog` (the native overwrite prompt remains the
confirmation) and the `#EXTM3U` first line. Three things change:

- It first checks `_library_writes_blocked("Export playlist")`.
- It builds `plan_playlist(paths, target)`, prepends `#EXTM3U`, and writes with
  `write_playlist_plan([item], backup=True, log_dir=<library>/Docs)`.
- Replacing an existing playlist therefore leaves a `.bak` and a log line.

CSV export is unchanged.

**CP-43 — Texts that stop being true.**

| Text | Where | Change |
|---|---|---|
| Results hint ("Playlists are not created automatically…") | `clustered_enhanced.py:596` | Becomes "Save a playlist per cluster with **Save cluster playlists…** (or export a lasso selection from the map)." A **Save cluster playlists…** button is added beside *Open Visual Graph* until the merge removes the tab. |
| Advanced config summary's "Output" lines | `_display_config` | Mention the offer when `create_playlists` is on. |
| Log line | `clustered_playlists.py:1090` | Becomes "→ Saved cluster membership; use Save cluster playlists… on the map to write one playlist per cluster." |
| Docstrings | `generate_clustered_playlists`, `clustered_enhanced` | Updated. |

---

## 9. Wizard options (output step)

**CP-44 — "Create M3U playlists for each cluster" is enabled.**

- **Relabel:** "Offer to save a playlist per cluster when the analysis finishes". Default
  unchecked.
- **Tooltip:** "Opens the Save cluster playlists preview after a successful run. Nothing is
  written until you click Save."
- **Config:** it keeps the config key `create_playlists`, saved by `_save_step(4)`.
- **Effect:** after a *successful* run with `create_playlists=True`, the analysis page
  calls `flow.start(preselect=None)`, where `None` means every non-noise cluster. If the
  page is not visible at that moment, the offer is held and made the next time the page is
  shown (in `showEvent`), so no modal appears over another page. A failed or cancelled run
  never offers. Quick Start never offers.
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
    `(payload, members)` pair. Distances are `i * 0.1` and path ranks are positions within
    each cluster.
  - `write_cluster_run(library, kind="valid"|"legacy"|"mismatched_run"|"misaligned_members", **kw)`
    writes both files.
  - Both builders validate their own output through `cluster_graph_data`, as
    `write_cluster_payload` does today.
- **`tests/ui/playlists_helpers.py`:** `write_cluster_payload` gains `with_members=True`,
  delegating to the helper above. `FakeClustering` gains `write_members=True`: it writes the
  members file and the `run` block, and returns `run` in its result. GUI tests that save need
  at least `min_size × clusters` files (for example 15 files for 3 clusters), or they lower
  the minimum in the dialog.

### 10.2 Backend

| Test | CP |
|---|---|
| `tests/test_playlists_clustering_backend.py::test_members_file_is_written_with_the_run_block` (uses the existing `fake_engine` fixture) | 1, 2, 3 |
| `…::test_members_file_is_full_length_when_the_map_is_downsampled` (monkeypatch `MAX_VISUALIZATION_POINTS`) | 3, 9 |
| `…::test_both_files_share_one_run_id` · `…::test_cancel_writes_neither_file` | 1, 2 |
| `…::test_centroid_distance_on_a_toy_matrix` · `…::test_path_rank_is_a_greedy_chain_from_the_centre` · `…::test_path_rank_is_null_over_the_limit` (call `cluster_member_order` directly with a small `path_limit`) | 4, 5 |
| `…::test_tempo_and_energy_columns_are_raw_and_only_when_selected` | 3 |
| `tests/test_cluster_graph_data.py::test_load_members_missing_invalid_misaligned` · `::test_members_must_match_payload_run_and_arrays` (full and downsampled) | 7, 8, 9 |
| **New `tests/test_cluster_playlist_export.py`:** | |
| `test_legacy_payload_is_refused_with_reanalyse_message` · `test_run_mismatch_is_refused` · `test_all_missing_is_refused` | 12 |
| `test_missing_and_reserved_members_are_left_out_and_counted` | 11 |
| `test_moved_library_root_is_rebased_with_warning` | 10 |
| `test_new_tracks_are_counted_and_reserved_folders_ignored` · `test_big_change_suggests_reanalysis` | 13 |
| `test_summary_writes_nothing` (snapshot the tree before and after) · `test_summary_honours_cancel_event` · `test_summary_reports_progress` | 26 |
| `test_names_are_deterministic` · `test_name_uses_dominant_genre_tempo_and_energy` · `test_second_genre_is_appended` · `test_artist_only_without_dominant_genre` · `test_various_artists_is_ignored` · `test_name_falls_back_to_cluster_number` · `test_cluster_ids_are_zero_padded_to_the_widest` (injected `tag_reader`) | 14–16 |
| `test_sanitize_replaces_reserved_chars_trailing_dots_and_device_names` (parametrised) | 17 |
| `test_user_names_colliding_by_case_invalidate_the_plan` · `test_existing_file_matched_by_case_becomes_replace_with_exact_name` | 18 |
| `test_noise_is_never_planned` | 19 |
| `test_run_folder_name_kmeans_and_hdbscan` | 20 |
| `test_earlier_sets_are_listed_and_untouched` · `test_resave_same_run_replaces_and_leaves_old_names` | 21 |
| `test_entries_equal_plan_playlist_lines_relative_native` · `test_header_lines_and_entry_count` | 22, 23 |
| `test_centre_path_library_orders` · `test_path_order_falls_back_per_cluster` · `test_missing_members_keep_path_order` | 24 |
| `test_min_size_counts_present_tracks_only` | 25 |
| `test_planner_does_no_file_io` (monkeypatch `os.path.exists`/`os.stat`/`open` to raise) | 8.1 |
| `test_write_writes_exactly_the_plan` · `test_write_refuses_when_run_changed` · `test_replace_makes_bak_and_log_lines` · `test_file_appearing_since_preview_is_skipped` · `test_cancel_between_files` | 27, 28, 30, 31 |
| `tests/test_writers_playlists.py::test_verify_skips_changed_and_appeared_files` · `::test_defaults_unchanged` · `::test_cancel_event_stops_between_files` · `::test_plan_playlist_records_snapshot` | 27 |

### 10.3 GUI (`tests/ui/`, headless; read `tests/ui/README.md`)

New file `tests/ui/test_ui_playlists_cluster_export.py`. It uses `make_workspace("graph",
…)` (or the Sound Map key after the merge), `dialogs`, `threads` and `qtbot`. Tags are
stubbed with the existing `_no_disk_tags` pattern (monkeypatch
`utils.audio_metadata_reader.read_metadata`), and pyqtgraph-dependent tests are guarded by
`pytest.importorskip("pyqtgraph")`.

| Test | CP |
|---|---|
| `test_button_disabled_without_data_and_for_legacy_payload` (check the tooltips) | 37, 12 |
| `test_preview_job_is_read_only` (inspect `job_registry().jobs(owner=ws)`: `ask_before_quit is False`, `group is None`) | 32 |
| `test_dialog_preticks_only_visible_clusters` (hide one via `_legend._rows[cid].checkbox`) | 37 |
| `test_legend_menu_saves_one_cluster` (`dialogs.on_menu("Save as playlist…")`) | 38 |
| `test_noise_row_is_not_checkable` | 19, 39 |
| `test_options_replan_rows` (min size → Skip; order → ▶ order changes) | 24, 25, 39 |
| `test_cancelling_the_dialog_writes_nothing` (`on_exec(Rejected)`; no `Playlists/Clusters`) | 29 |
| `test_save_writes_exactly_the_shown_plan` (compare files with `dialog.plan().items`) | 28, 33 |
| `test_write_job_takes_the_library_write_lock` (`group == LIBRARY_WRITES`, `ask_before_quit is True`) | 33 |
| `test_replace_asks_and_backs_up` (`answer_yes`; `.bak` + `BACKUP` in `Docs/playlist_writes.log`) · `test_replace_declined_writes_nothing` (`answer_no`) | 31, 29 |
| `test_busy_library_blocks_preview_and_save` (`tag_write_guard.acquire("Tag Fixer apply")`; "Library Busy" information box; nothing written) | 32, 33 |
| `test_cancel_preview_opens_no_dialog` (a tag reader gated on an event, as `FakeClustering(hold=True)` does) | 35 |
| `test_library_change_cancels_preview_but_lets_write_finish` | 36 |
| `test_button_disabled_while_analysis_runs` (the merged page, or the Clustered tab with `FakeClustering(hold=True)`) | 34 |
| `test_dialog_play_emits_planned_order` (`qtbot.waitSignal(ws.play_tracks_requested)`) | 40 |
| `test_closing_the_window_mid_write_asks` (`main_window`, `dialogs`) | 35 |
| `test_lasso_export_backs_up_logs_and_respects_the_lock` | 42 |
| `tests/ui/test_ui_playlists_clustering.py::test_wizard_offer_option_is_enabled_and_saved` · `::test_successful_run_with_offer_opens_preview` (`FakeClustering(write_members=True)`, `on_exec(Rejected)`) · `::test_offer_waits_until_the_page_is_shown` · `::test_failed_or_cancelled_run_never_offers` | 44 |

**Existing tests to update (not delete):**

- `test_wizard_does_not_offer_unbuilt_options`: drop `_create_playlists_cb` from the
  disabled list, and assert that the rest stay disabled (CP-45).
- `test_results_do_not_promise_cluster_playlists`: rename it to
  `test_results_point_to_save_cluster_playlists`, and assert the new hint and button
  (CP-43).

**Static checks:** `tests/test_gui_hygiene_ratchets.py` counts must not rise; lower a
ceiling if the change removes inline styles. `tests/ui/test_ui_static_qt_api.py` must pass.
Add the dialog to `scripts/gui_snapshots.py` if that script enumerates dialogs.

---

## 11. Definition of done

Done means all four of the following hold:

- Every CP is implemented.
- §10 is green.
- These docs are updated:
  - `docs/features/playlists_and_clustering.md`: "From groups to playlists", the wizard
    step-5 text, and "What's real today".
  - `docs/ROADMAP.md`: remove "Playlists straight from a clustering run"; move the
    `create_playlists` wizard option out of "Controls that are still switched off".
  - `docs/gui_inventory.md`: the new button, menu and dialog.
  - `docs/architecture.md`: add `cluster_members.json` and the run folder to the
    "What each run leaves behind in Docs/" table, and extend the index-alignment note to
    the members file.
  - `CLAUDE.md`: extend the index-alignment bullet to `Docs/cluster_members.json`.
- `docs/redesign/pages/06_playlists_clustering_graph.md` PL-15 is marked fixed.

---

## 12. Open questions for the owner (each has a default, so implementation isn't blocked)

| # | Question | Default if unanswered |
|---|---|---|
| Q1 | When you save a new run, should earlier sets be replaced or kept? | **Keep** (per-run folders, never deleted). "Retire earlier sets" can follow later, using the CP-23 marker. |
| Q2 | Should HDBSCAN's unclustered tracks ever become a playlist? | **Never** in v1. |
| Q3 | Default order inside a playlist? | **Closest to the centre first.** |
| Q4 | Default minimum size? | **5** present tracks. |
| Q5 | Name shape: genre first, artist only as fallback, with BPM and energy words? | **Yes**, as in CP-16. |
| Q6 | `.m3u` or `.m3u8`? | **`.m3u`, UTF-8**, like every other generated playlist. |
| Q7 | Analyses made before this feature: refuse, or allow a partial export from the map data? | **Refuse** and ask for a re-analysis. A partial export would silently drop tracks from downsampled maps. |
| Q8 | Should the wizard's "offer after analysis" default to on? | **Off.** |
| Q9 | Should the destination be choosable? | **No**: always `<library>/Playlists/Clusters/<run>/` in v1. |
