# Root Directory Reorganization Spec

**Status: DEFERRED — planning only, not yet scheduled.**

This is a specification to execute later, not a task in progress. It exists so the
work doesn't have to be re-planned from scratch when the time comes.

## Why this is deferred

Other agents/sessions are actively working across this codebase right now. A
repo-wide file move + import rewrite touching dozens of modules would conflict
with almost anything else in flight — every open branch that touches a moved
file would need manual conflict resolution, and any branch based on the old
layout would silently break on merge. **Do not start this until there is a
"pencils down" window** — no other agents mid-task, no branches you care about
sitting unmerged.

When that window arrives, come back to this file, re-run the audit commands in
[Before you start](#before-you-start-re-verify-this-audit) to confirm nothing
has drifted since this was written (2026-08-31), and then execute.

## Scope

Two phases, to be done in order, as separate PRs:

- **Phase 1 — Junk removal.** Delete debris, relocate misplaced files. Zero
  import risk. Safe to do even if Phase 2 never happens.
- **Phase 2 — Domain-grouped package.** Move the ~35 real backend modules out
  of root into `alphadex/<domain>/`. Touches every import statement in the
  codebase. Do this in small, isolated, reviewable steps — not one big PR.

Both phases leave alone: `gui/`, `controllers/`, `plugins/`, `utils/`,
`docs/`, `mutagen_stub/`, `library_sync_indexer_engine/` (per `CLAUDE.md`,
this is an intentionally separate, already-drifted copy of the indexer engine
used only by `library_sync.py` — do not merge it into anything here), and the
two GUI entry points `alpha_dex_gui.py` / `main_gui.py`, which stay at root.

---

## Phase 1 — Junk removal

### Delete outright (and add to `.gitignore` so they don't come back)

| File | Why |
|---|---|
| `=0.13.0` | Stray file, almost certainly from a botched shell command (e.g. an unquoted `pip install foo>=0.13.0`). Not referenced anywhere. |
| `soundvault_crash.log`, `.log.1`–`.log.5` | ~5 MB of runtime crash logs. Should never have been committed. |
| `diff_player.txt`, `diff_player_prev.txt`, `player_diff.txt`, `player_full_diff.txt` | ~460 KB of old ad-hoc diff dumps, not source. |
| `crash_report.txt`, `crash_test.txt` | Debug/test output, not source. |
| `last_path.txt` | Runtime state (last-used folder path). Belongs in `~/.soundvault_config.json`-style user state, not the repo. |
| `player_106bd0e.py` | Hash-suffixed filename — smells like an accidental duplicate/backup of another player file. **Diff it against the real player module it shadows before deleting**, in case it holds an uncommitted fix. |

Suggested `.gitignore` additions:
```
*.log
*.log.[0-9]
last_path.txt
crash_report.txt
crash_test.txt
```

### Relocate (not junk, just misfiled)

| File(s) | Move to | Why |
|---|---|---|
| `test_cluster_graph_demo.py`, `test_clustering_integration.py` | `tests/` | Orphaned — every other test lives in `tests/` (42 modules per `CLAUDE.md`); pytest still collects them from root today only because `pytest.ini` sets `testpaths = tests` — **verify these two are even currently being run in CI/local `pytest` before moving them silently into scope**, since they may not be collected at all right now. |
| `alphadex.html`, `alphadex3.html` | `docs/prototypes/` (new folder) | Static prototype/mockup HTML, not part of either GUI's runtime. |
| `demo_tile_proposals.py` | `scripts/` or delete | Not imported anywhere (confirmed). Looks like a UI mockup/demo used to prototype the "Liquid Glass" tile styling in the Tools workspace. |
| `update_lg.py`, `update_tools.py` | `scripts/` or delete | Not imported anywhere (confirmed). These are one-shot codemod scripts that read/rewrite `demo_tile_proposals.py` and `gui/workspaces/tools.py` via **hardcoded relative paths**, meaning they only work run from repo root and were almost certainly already run once and forgotten. **Ask before deleting** — confirm they're not still needed for a change in progress. |
| `diagnose_cluster_graph.py` | `scripts/` | Not imported anywhere (confirmed) — it's a standalone CLI diagnostic (`python diagnose_cluster_graph.py <library_path>`), not a library module. Belongs with dev tooling, not the app package. |
| `fingerprint_examples.py` | `scripts/` or `examples/` | Not imported anywhere (confirmed) — example/reference usage, not library code. |
| `duplicate_bucketing_poc.py` | `scripts/` or delete | Name says proof-of-concept. **Verify it's not imported by anything before moving** — not checked as thoroughly as the others above during this audit. |

### End state after Phase 1

```
Music_Indexer/
├── CLAUDE.md  README.md  requirements.txt  pytest.ini  .gitignore
├── alpha_dex_gui.py            # unchanged
├── main_gui.py                 # unchanged
├── *.py                        # ~35 backend/legacy-GUI modules, still flat at root
├── controllers/  gui/  plugins/  utils/  scripts/
├── library_sync_indexer_engine/  mutagen_stub/
├── docs/
│   └── prototypes/              # NEW: alphadex.html, alphadex3.html
└── tests/                       # +2 files moved in
```

This alone is safe to do at any time, independent of Phase 2, and independent
of what other agents are doing elsewhere in the repo (it touches no import
statements) — **as long as** nothing else in flight is mid-edit on
`demo_tile_proposals.py`, `update_lg.py`, `update_tools.py`, or the crash-log
files specifically. Check `git log --all --since="1 week ago" -- <file>`
for each before deleting/moving it.

---

## Phase 2 — Domain-grouped package

### Architecture finding from this audit (not yet in `CLAUDE.md`)

`CLAUDE.md` already documents two known violations of "no tkinter in backend
modules": `plugins/acoustid_plugin.py` and
`controllers/library_index_controller.py`. This audit found **two more**,
undocumented until now:

- `cluster_graph_panel.py` imports `tkinter` directly.
- `update_genres.py` imports `tkinter` directly.

Neither can go into a tkinter-free backend package as-is. Options at
execution time: split UI code out of them first (matches the spirit of the
architecture rule), or accept the violation and place them under a
`legacy_gui/` grouping instead of `alphadex/`. Recommend flagging this to
whoever executes Phase 2 rather than silently choosing one.

Also confirmed importing `tkinter` directly (expected, not new): `main_gui.py`,
`library_sync_review.py` (already documented in `CLAUDE.md` as "Library Sync
review-first UI panel"), `unsorted_popup.py`.

### Proposed structure

```
alphadex/
├── __init__.py
├── core/
│   ├── config.py                       # 14 importers repo-wide — highest fan-in of any root module
│   └── dry_run_coordinator.py          # used by music_indexer_api.py + 4 test files
├── indexing/
│   ├── music_indexer_api.py            # "Core scan / relocation logic" (CLAUDE.md)
│   ├── validator.py                    # "Validate library folder structure" (CLAUDE.md)
│   └── indexer_control.py              # cancel-event; only consumer is gui/workspaces/indexer.py
├── duplicates/
│   ├── duplicate_consolidation.py
│   ├── duplicate_consolidation_executor.py
│   ├── duplicate_scan_engine.py
│   ├── near_duplicate_detector.py
│   ├── simple_duplicate_finder.py
│   ├── fingerprint_cache.py
│   ├── fingerprint_generator.py
│   ├── chromaprint_utils.py            # "fpcalc wrapper" (CLAUDE.md)
│   ├── audio_norm.py                   # confirmed importers: fingerprint_generator.py + its test
│   └── cache_prewarmer.py              # confirmed importer: controllers/import_controller.py
├── library_sync/
│   ├── library_sync.py
│   ├── library_sync_review.py          # tkinter-coupled review UI — see architecture note above
│   ├── library_sync_review_report.py
│   ├── library_sync_review_state.py
│   └── library_sync_types.py
├── tagging/
│   ├── tag_fixer.py
│   ├── metadata_service.py
│   └── update_genres.py                # tkinter-coupled — see architecture note above
├── playlists/
│   ├── playlist_engine.py
│   ├── playlist_generator.py
│   ├── clustered_playlists.py
│   ├── cluster_graph_panel.py          # tkinter-coupled — see architecture note above
│   └── cluster_graph_3d.py
└── diagnostics/
    ├── crash_logger.py
    └── crash_watcher.py
```

`legacy_gui/` (or leave at root next to `main_gui.py` — pick one at execution
time) for the confirmed tkinter-only files that don't belong in a backend
package: `unsorted_popup.py`, and whatever comes out of the
`cluster_graph_panel.py` / `update_genres.py` split decision above.

### Confidence levels — be honest about what's verified vs. guessed

This audit checked actual `import` fan-in for the modules listed with a
confirmed-importer note above. It did **not** trace every function call for
every file — for anything without a "confirmed importer" note in the tree
above (most of `duplicates/`, `playlists/`, `diagnostics/`), the placement is
based on the filename and `CLAUDE.md`'s existing one-line descriptions, which
is good enough to plan against but should be re-confirmed with
`grep -rln "import <module>"` immediately before that specific file is moved,
not assumed correct from this document alone.

### Execution plan — staged, not a single PR

Do **not** move all ~35 files and rewrite every importer in one PR. Go
domain-by-domain, smallest/most-isolated first:

1. **`library_sync/`** first — it's the most self-contained (5 files that
   mostly reference each other and the separate `library_sync_indexer_engine/`,
   which stays untouched). Move it, update its own internal imports plus
   whatever external files import `library_sync*`, run the full test suite,
   smoke-test the Library Sync workspace in the running app (per `CLAUDE.md`'s
   own rule: don't claim a UI change works without actually running it).
2. **`tagging/`** next (3 files, resolve the `update_genres.py` tkinter
   question first).
3. **`duplicates/`** — largest domain, do this once the pattern from steps 1–2
   is proven safe.
4. **`playlists/`** — resolve the `cluster_graph_panel.py` tkinter question first.
5. **`indexing/`** and **`core/`** last — `config.py` alone has 14 importers
   scattered across the whole codebase including `main_gui.py` (470 KB), so
   this step touches the most files of any single step. Doing it last means
   the import-rewrite process has already been proven on four smaller domains
   first.
6. **`diagnostics/`** — smallest, can go anywhere in the sequence; included
   last here only because it's lowest-value/lowest-urgency.

After each step: full `pytest` run, plus a manual smoke test of whichever
workspace(s) that domain's modules actually feed into. Update `pytest.ini`'s
`pythonpath` setting if needed once modules stop being importable by bare
name from repo root.

### What Phase 2 does *not* include

- Making the project pip-installable (`pyproject.toml` / `setup.py`) — that's
  a separate decision or an add-on, not required for the directory layout
  change itself.
- Touching `gui/`, `controllers/`, `plugins/`, `utils/` internals — only their
  `import` statements change, to point at the new `alphadex.<domain>.<module>`
  paths.
- Resolving the `cluster_graph_panel.py` / `update_genres.py` tkinter
  violations beyond deciding where they land — actually splitting UI from
  logic in those two files is its own task, not a prerequisite blocking this
  reorg (though doing it at the same time is reasonable if whoever executes
  this wants to).

---

## Before you start: re-verify this audit

Things may have changed since 2026-08-31. Re-run before executing either phase:

```bash
# re-confirm the junk/misplaced file list is still accurate
ls -la | grep -v '^d'

# re-confirm nothing has grown a new importer since this was written
grep -rln "import <module_name>" --include="*.py" .

# re-confirm the tkinter-import list for Phase 2
for f in *.py; do
  grep -lq "^import tkinter\|^from tkinter" "$f" 2>/dev/null && echo "$f"
done

# sanity check nothing else is mid-edit on files this touches
git log --all --since="1 week ago" --name-only -- <file>
```
