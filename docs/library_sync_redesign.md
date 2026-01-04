# Library Sync UI Redesign (Current Implementation)

## Overview
The Library Sync UI is a review-first comparison tool for an existing library and an incoming folder. It provides two synchronized lists (Incoming Tracks and Existing Tracks), a text inspector, and an optional plan/preview/execute pipeline for copy/move operations. File operations only occur after the user explicitly runs **Execute**—scans and previews are non-destructive.

**Backend alignment (current state):** `library_sync.py` performs independent scans of the *Existing Library* and *Incoming Folder*, reuses cached fingerprints where size/mtime match, computes new fingerprints in parallel when needed, and matches via indexed shortlists (fingerprint signature ➜ extension ➜ fallback pool). Matches are classified as *New*, *Collision*, *Exact Match*, or *Low Confidence* with a quality label (*Potential Upgrade* / *Keep Existing*) derived from format priority and bitrate. `library_sync_review.py` drives the UI, tracks session state, and wires up plan preview and execution.

## User Flow (Start ➜ Finish)
1. **Pick folders:** Select an *Existing Library* folder and an *Incoming Folder*.
2. **Configure scan:** Adjust the global threshold, optional per-format overrides, preset name, and report version.
3. **Scan libraries:** Scan each folder independently (both are cancellable).
4. **Review results:** Inspect Incoming/Existing lists and the text inspector for match details.
5. **Build plan (optional):** Choose transfer mode (copy/move), then **Build Plan** and **Preview** to generate the HTML preview.
6. **Execute (optional):** Run **Execute** to apply the plan; optionally emit a playlist of transferred files.
7. **Export logs / save session:** Use **Export Logs…** and **Save Session** to persist review data.

## UI Components and Controls
### Folder Selection
- **Existing Library:** Path picker for the destination library root.
- **Incoming Folder:** Path picker for the new content root.

### Scan Configuration
- **Global Threshold:** Default fingerprint distance cutoff.
- **Preset Name:** Stored with the scan session for reference.
- **Per-format overrides:** Text box with `ext=threshold` per line.
- **Report Version:** Stored in the scan session metadata.
- **Scan State:** Read-only indicator of the current scan lifecycle state.

### Scan Controls
- **Scan / Cancel:** Independent buttons for library and incoming scans.
- **Progress + Phase labels:** Per-scan progress bar, phase text, and partial-result notice on cancellation.

### Plan & Execution
- **Build Plan:** Computes a deterministic move/copy plan.
- **Preview:** Renders the plan preview HTML.
- **Transfer mode toggle:** Switches between **Copy Originals** and **Move Originals**.
- **Execute:** Applies the current plan (requires a matching, previewed plan).
- **Output playlist:** Optionally emit a playlist of transferred files after execution.
- **Open Preview / Cancel:** Open the preview HTML or cancel the active plan task.

### Primary Lists (two panels)
- **Incoming Tracks:** Shows each incoming file with chips and fingerprint distance.
- **Existing Tracks:** Shows each library file and the number of incoming tracks that map to it as best matches.

### Detail Inspector
Read-only text panel summarizing:
- Track metadata and file details.
- Match status, fingerprint distance, threshold used, confidence score, and quality label.

### Logs
- Scrollable log panel capturing scans, plan actions, and execution events.
- **Export Logs…** writes the log to disk.
- **Save Session** persists the current scan configuration and folders to config.

## Status Chips (Current)
- **Incoming list:** New, Collision, Exact Match, Low Confidence, Potential Upgrade, Keep Existing, Missing Metadata, Partial.
- **Existing list:** Best Match, Potential Upgrade, Keep Existing, Partial, or Unmatched.

## Execution Pipeline (Opt-in)
- **Plan generation:** `compute_library_sync_plan()` remaps indexer-derived destinations under the selected library root and attaches per-file decisions (*COPY*, *REPLACE*, or *SKIP* with reasons).
- **Preview:** Renders a dry-run HTML preview via the indexer HTML renderer.
- **Execution:** `execute_library_sync_plan()` writes JSON audit logs plus HTML reports, optionally creates a “LibrarySync_Added_YYYY-mm-dd_hhmm.m3u8” playlist, and writes backups before replacements. Cancellations are honored between steps.

## Current Gaps (Not Yet Wired)
- **Per-item copy/replace flags** are modeled in `library_sync_review_state.py`, but the UI does not expose buttons or menus to set them yet.
- **Export report UI** is not exposed (the report helpers exist in `library_sync_review_report.py`, but there is no export button).
- **Filters/sorts and quick actions** (e.g., “Collisions Only” or “Sort by Quality Delta”) from earlier design notes are not present.

## Acceptance Criteria (Current Implementation)
- Two primary lists: *Incoming Tracks* and *Existing Tracks*.
- Independent scan controls with cancellation and partial-result indicators.
- Explicit plan build/preview/execute controls with a transfer-mode toggle.
- Non-destructive review unless **Execute** is run.
- Logs and session persistence are available from the UI.
