# Library Sync UI Redesign

## Overview
This redesign reframes the Library Sync tool as a review-first comparison interface between an existing music library and an incoming folder. The UI prioritizes clarity, safety, and trust by clearly distinguishing incoming content from existing content, surfacing duplicates and collisions, and enabling informed decisions through non-destructive review actions. All automated file operations and playlist updates are intentionally excluded from this tool.

**Backend alignment (current state):** The shipping backend in `library_sync.py` already follows this review-first contract. It performs independent scans of the *Existing Library* and *Incoming Folder*, reuses cached fingerprints where size/mtime match, computes new fingerprints in parallel when needed, and matches via indexed shortlists (fingerprint signature ➜ extension ➜ fallback pool). Matches are classified as *New*, *Collision*, *Exact Match*, or *Low Confidence* with a quality label (*Potential Upgrade* / *Keep Existing*) derived from format priority and bitrate. Execution helpers remain opt-in and write dry-run previews plus JSON/HTML audit logs; file operations are blocked unless a plan is explicitly executed.

## User Flow (Start ➜ Finish)
1. **Pick folders:** The user independently selects an *Existing Library* folder and an *Incoming Folder*.
2. **Configure scan:** The user adjusts fingerprint distance thresholds (global or per-format) and optional quality tie-breaker settings, then starts scans for each folder independently.
3. **Scan libraries:** Each folder is scanned separately using indexed matching and cached fingerprints; progress and cancellation are supported.
4. **Review results:** Results are displayed in two primary lists—**Incoming Tracks** and **Existing Tracks**—with status chips indicating matches, collisions, confidence, and quality relationships.
5. **Inspect details:** Selecting a row opens an inspector showing metadata, fingerprint metrics, quality comparison, and the best-matched counterpart when applicable.
6. **Flag actions:** Users flag items for potential copy or replacement as part of a review-only plan; no file operations occur.
7. **Export plan:** The user exports a report summarizing all findings and flags for later manual execution or use by future controlled tools.

## UI Components and Controls
### Folder Selection
- **Existing Library selector:** Path picker that remembers the last valid library root.
- **Incoming Folder selector:** Independent path picker for new content.

### Scan Controls
- **Scan Library / Scan Incoming:** Independent scan buttons with cancellable progress.
- **Threshold presets:** Dropdown for common profiles (e.g., strict, normal, loose) mapped to fingerprint distance values.
- **Per-format overrides (optional disclosure):** Table allowing threshold overrides per extension.

### Primary Lists (exactly two)
- **Incoming Tracks:** Displays all scanned incoming files. Each row is labeled with a status chip such as *New*, *Collision*, *Exact Match*, or *Low Confidence*.
- **Existing Tracks:** Displays all scanned library files. Rows are highlighted when they are the best match for one or more incoming tracks.
- *Incoming Tracks* always contains all incoming files; “new” versus “matched” is indicated via status chips and filters rather than separate lists.

### Detail Inspector
- Shows detailed information for the selected item, including tags, duration, bitrate (when available), format-based quality score, fingerprint distance, threshold used, and the single best-matched counterpart (lowest distance). If multiple potential matches exist, only the best candidate is actionable; others are informational.
- Optional shortcuts such as “open containing folder” and “play preview” are shown only when supported by the runtime environment.

**Implementation snapshot:** Each scan produces normalized `TrackRecord` entries (path, ext, bitrate, size, mtime, duration, fingerprint, tags). The inspector can display `MatchResult` payloads that already include the best match, candidate shortlist (with distances), threshold applied, near-miss margin, confidence score, and quality comparison labels. These payloads also flag when a result is partial due to cancellation, allowing the UI to show “Partial scan” states without blocking retries.

### Status Chips
- Each row may display one or more chips, including *New*, *Collision*, *Exact Match*, *Low Confidence*, *Potential Upgrade*, *Keep Existing*, or *Missing Metadata*. Chips are used consistently throughout the UI to avoid ambiguity.

### Minimal Action Buttons (non-destructive only)
- **Filter: Collisions Only:** Toggles visibility to only matched items.
- **Sort: Quality Delta:** Sorts incoming tracks by quality difference relative to their matched library counterpart.
- **Flag for Copy:** Marks selected incoming tracks for inclusion in the export plan.
- **Flag for Replace:** Marks the incoming–existing pair (best match only) as a potential replacement candidate.
- **Export Plan / Report:** Generates a CSV or JSON summary of the current review state.
- **Clear Flags:** Removes all review flags without affecting scan results.
- No UI control, context menu, shortcut, or background task performs file moves, replacements, or playlist updates.

**Execution pipeline (opt-in):**
- **Plan generation:** `compute_library_sync_plan()` builds a deterministic move/route plan by reusing the indexer engine’s routing rules, remapping destinations under the selected library root, and attaching per-file decisions (*COPY*, *REPLACE*, or *SKIP* with reasons). Dry-run previews reuse the indexer HTML renderer for consistency.
- **Execution:** `execute_plan()` consumes the plan, writes JSON audit logs plus executed HTML reports, creates a “LibrarySync_Added_YYYY-mm-dd_hhmm.m3u8” playlist of transferred files (when enabled), and writes backups before replacements. Cancellations are honored between steps; skipped or review-required items are summarized in the report.

### Progress and Status
- Separate progress indicators are shown for library and incoming scans. Each scan can be cancelled independently. A log panel displays scan configuration, counts, skipped files, and warnings.

## Determining “Incoming” vs “Existing”
### Matching Logic
- Incoming tracks are compared against the existing library using an indexed fingerprint lookup rather than exhaustive pairwise comparison. Cached fingerprints are reused when available.

### Thresholds
- Per-format thresholds take precedence over the global threshold. Distances below the active threshold are treated as matches; distances above are treated as non-matches.

### Confidence Indicators
- **Exact Match:** Fingerprint distance equals zero.
- **Collision:** Distance is below the active threshold.
- **Low Confidence:** Distance is above the threshold but within a defined margin (e.g., threshold + fixed delta or percentage), prompting manual review.

### Quality Comparison
- When a match exists, a quality score is computed using extension priority and bitrate when available. If bitrate is unavailable, extension priority alone is used. Incoming items with higher scores are labeled *Potential Upgrade*; lower-quality matches are labeled *Keep Existing*.

## Error Handling and Edge Cases
- Scans support large libraries through incremental progress updates, cancellation after the current file completes processing, and reuse of cached fingerprints.
- Missing metadata is tolerated and flagged without blocking classification.
- Unsupported or unreadable files are skipped with logged reasons.
- Cancelled scans preserve partial results and are clearly labeled as *Partial*.
- Path and permission errors surface as inline banners and log entries without disabling retry.
- Threshold inputs are validated; invalid values are rejected with inline feedback and reverted to the last known good configuration.

**Performance profiling hooks:** An optional `PerformanceProfile` collector records cache hits, fingerprint computations, shortlist sizes, and scan/match durations. UI surfaces can render these metrics for troubleshooting slow comparisons without affecting the main workflow.

## Export Plan Schema (Minimum)
- Exported reports must include incoming path, existing path (nullable), match status, fingerprint distance, threshold used, quality scores, user flags, notes, timestamp, and a scan configuration hash.

## Acceptance Criteria
- The UI contains exactly two primary lists: *Incoming Tracks* and *Existing Tracks*.
- No list, button, shortcut, or background process performs copy, replace, or playlist updates.
- Users can independently select folders, configure thresholds, and scan each source.
- Incoming tracks are clearly labeled via chips rather than separated into additional lists.
- Matching, quality comparison, and confidence indicators are visible and consistent.
- Exported reports fully represent the review state and require no file access to generate.
- Scan cancellation preserves completed results and logs without corrupting state.
