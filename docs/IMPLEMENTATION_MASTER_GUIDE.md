# Library Sync & Per-Item Flags — Master Implementation Guide

**Status:** Ready for implementation
**Branch:** `claude/audit-startup-splash-PwNwu`
**Last Updated:** 2026-03-20

---

## Document Structure

This guide combines:
1. **Per-Item Review Flags** — Feature allowing users to override auto-decisions on individual tracks
2. **Library Sync** — Complete workflow for merging two music libraries safely
3. **Integration Points** — How flags feed into plan building
4. **Implementation Checklist** — All tasks needed to build this system

---

## Quick Reference

- **Per-Item Flags Spec:** See `docs/FEATURE_SPECIFICATION.md` (complete 400-line reference)
- **Library Sync Spec:** See `docs/LIBRARY_SYNC_COMPLETE_SPEC.md` (complete 750-line reference)
- **This Document:** Implementation guide + checklist

---

# Part 1: Per-Item Review Flags

## What It Does

Allows users to right-click incoming tracks during Library Sync review and explicitly flag them:
- 📋 **Copy** — Force file to be copied (overrides skip decisions)
- ↻ **Replace** — Allow file to overwrite its match in library
- ✕ **Clear flag** — Remove the override
- 📝 **Add note** — Document the decision

## How It Works

```
User right-clicks track in table
        ↓
Context menu appears (4 options)
        ↓
User selects action (Copy / Replace / Clear / Note)
        ↓
ReviewStateStore stores the flag
        ↓
Flag emoji appears in Flag column (📋 or ↻)
        ↓
When building plan, flags override auto-decisions
        ↓
Plan reflects user's choices ("User flag: Copy")
        ↓
Execution respects the flag
```

## Key Behaviors

| Action | When to Use | Effect |
|--------|-----------|--------|
| 📋 Copy | "This should be copied despite system decision" | Forces COPY in execution |
| ↻ Replace | "This is better quality, use it instead" | Forces REPLACE in execution |
| ✕ Clear | "Changed my mind, use auto-decision" | Reverts to auto-decided |
| 📝 Note | "Document why I chose this" | Shows in HTML preview |

## Data Structures

### ReviewStateStore
```python
{
  "copy": ["track_001", "track_003"],           # IDs of files to copy
  "replace": {
    "track_002": "track_existing_002"           # incoming → existing mapping
  },
  "notes": {
    "track_001": "Higher bitrate",              # per-track notes
    "track_002": "Check metadata"
  }
}
```

### MatchResult (from scan)
```python
{
  "incoming_file": AudioFile,
  "existing_file": AudioFile (or None),
  "status": "NEW|MATCHED|COLLISION|UPGRADE",
  "distance": float,                            # 0.0-1.0
  "track_id": str                               # Stable ID for flagging
}
```

---

# Part 2: Library Sync Workflow

## 7-Stage Architecture

### **Stage 1: INPUT**
User provides:
- Existing library path
- Incoming folder path
- Fingerprint threshold (0.0–1.0, default 0.3)

**UI:** Form with path inputs, Browse buttons, threshold slider

**Validation:**
- Both paths exist ✓
- Both are directories ✓
- Both are readable ✓

**Result:** Paths validated, ready to scan

---

### **Stage 2: SCAN**

**Process:**
1. Fingerprint all files in existing library (parallel thread)
2. Fingerprint all files in incoming folder (parallel thread)
3. Compare fingerprints using threshold
4. Classify each incoming file

**Classification Logic:**
```
For each incoming file:
  If 0 matches found:
    → Status: NEW (copy it)
  If 1 exact match (distance ~0.0):
    → Status: MATCHED (skip it)
  If 1 very close match (distance < 0.1):
    → Status: COLLISION (keep existing)
  If 1 distant match (0.1-0.3):
    → Status: UPGRADE (user decides)
  If 2+ matches:
    → Status: COLLISION (ambiguous)
```

**Result:** MatchResult objects with distances and classifications

---

### **Stage 3: REVIEW**

**UI:** Incoming tracks table (5 columns)
```
┌────────────────┬──────────┬──────────┬──────────┬──────────────┐
│ Track          │ Status   │ Distance │ Flag     │ Note         │
├────────────────┼──────────┼──────────┼──────────┼──────────────┤
│ song1.mp3      │ New      │ —        │          │              │
│ song2.mp3      │ Collision│ 0.045    │ 📋 Copy  │              │
│ song3.mp3      │ Upgrade  │ 0.085    │ ↻ Replace│ Higher brate │
└────────────────┴──────────┴──────────┴──────────┴──────────────┘
```

**User Actions:**
- Right-click any row → context menu
- Select Copy / Replace / Clear / Add note
- Flags stored in ReviewStateStore
- Notes displayed inline

**Result:** ReviewStateStore with user overrides + optional notes

---

### **Stage 4: PLAN BUILDING**

**Input:** MatchResults + ReviewStateStore flags

**Algorithm:**
```
For each MatchResult:
  1. Check ReviewStateStore for flags
  2. If flagged for Copy:
     → Decision: COPY
  3. Else if flagged for Replace:
     → Decision: REPLACE
  4. Else (no flag, auto-decide):
     → If status == NEW: COPY
     → If status == MATCHED: SKIP
     → If status == COLLISION: SKIP
     → If status == UPGRADE: SKIP

Result: ExecutionPlan {
  "copy_operations": [...],
  "replace_operations": [...],
  "skip_operations": [...],
  "summary": {
    "to_copy": 85,
    "to_replace": 12,
    "to_skip": 103
  }
}
```

**Result:** ExecutionPlan with operations and reasons

---

### **Stage 5: PREVIEW**

**Generate:** LibrarySyncPreview.html showing:
- All operations in table format
- Decision for each (COPY / REPLACE / SKIP)
- Reason (auto-decided or "User flag: Copy")
- User's notes (if any)
- Estimated size impact

**User Reviews:**
- Scans for accuracy
- Checks flagged items appear correctly
- Verifies no unintended consequences
- Can go back to Stage 3 to re-flag if needed

**Result:** User approval or request to modify

---

### **Stage 6: EXECUTION**

**Operations:**

**COPY:**
```
source: /path/to/incoming/file.mp3
destination: /path/to/library/file.mp3
action: copy file to library
```

**REPLACE:**
```
source: /path/to/incoming/file.mp3
destination: /path/to/existing/file.mp3
action: backup existing, then copy source over it
result: existing file → /Library/Backups/LibrarySyncBackups/file.mp3.backup
```

**SKIP:**
```
action: do nothing
```

**Progress:** Show:
- Overall progress (X / 200 files)
- Current file being copied
- Speed (MB/s)
- Time remaining

**Error Handling:**
- Source disappeared → Skip, log error
- Disk full → Pause, ask user to free space
- Permission denied → Pause, ask user to fix permissions
- File locked → Retry 3 times, then skip
- Corrupted file → Log warning, try to copy anyway

**Result:** Files copied/replaced per plan, all actions logged

---

### **Stage 7: REPORT**

**Generate:** LibrarySyncExecutionReport.html showing:
- Summary (X copied, Y replaced, Z skipped)
- Detailed list of each operation
- Errors (if any) and recovery steps
- Backup locations for replaced files
- Time taken

**Example Output:**
```
✓ Execution Complete

Copied:   85 files (350 MB)
Replaced: 12 files
Skipped:  103 files
Failed:   0 files

Time: 38 minutes
Library growth: +397 MB

Backups: /Library/Backups/LibrarySyncBackups/
(Kept for 30 days)
```

**Result:** User can verify success and recover if needed

---

## Full User Journey Example

```
1. Open Library Sync workspace
   User selects:
   - Existing: /mnt/music/
   - Incoming: /mnt/usb/NewMusic/
   - Threshold: 0.3

2. Click "Scan"
   System fingerprints both folders (~5-10 minutes)
   Results: 200 incoming files classified
   - NEW: 50
   - MATCHED: 50
   - COLLISION: 60
   - UPGRADE: 40

3. Review incoming tracks table
   User notices:
   - Song "remix.mp3" is COLLISION but looks NEW
   - Song "upgrade.flac" is UPGRADE and is better quality
   User actions:
   - Right-click remix.mp3 → Copy (override skip)
   - Right-click upgrade.flac → Replace (use better version)
   - Add note: "Better quality" to upgrade.flac
   Repeat for ~30 files needing flags

4. Click "Build Plan"
   System builds plan with overrides:
   - Copy: 50 (auto) + 1 (flagged) = 51
   - Replace: 12 (flagged)
   - Skip: 137 (auto)

5. Click "Preview Plan"
   HTML shows:
   - All 200 operations
   - Flagged items marked "User flag: Copy/Replace"
   - User's notes displayed
   User verifies: "Looks good"

6. Click "Execute Plan"
   System copies/replaces files with progress
   - 51 files copied (200 MB)
   - 12 files replaced (backup created)
   - 137 files skipped (do nothing)
   ~30 minutes elapsed

7. View report
   Summary: +51 new files, +12 upgraded
   Backups saved at /Library/Backups/...
   Done!

Result: Library now has 51 new songs + 12 quality upgrades
```

---

# Part 3: Implementation Checklist

## Phase 1: Data Structures & Backend (No UI)

### ReviewStateStore Class
- [ ] Create `ReviewStateStore` class
  - [ ] `flag_for_copy(track_id)` — add to copy list
  - [ ] `flag_for_replace(track_id, existing_id)` — add to replace dict
  - [ ] `clear_flags(track_id)` — remove all flags for track
  - [ ] `add_note(track_id, note_text)` — store note
  - [ ] `get_flags()` — return full state dict
  - [ ] `clear_all()` — reset on rescan
  - [ ] `__len__()` — count flagged items

### MatchResult Enhancement
- [ ] Add `track_id` field to MatchResult (stable ID from normalized path)
- [ ] Ensure `track_id` is unique per scan
- [ ] Add method `get_track_id_from_path()` to generate deterministically

### Path Resolution
- [ ] Create `resolve_review_flags_to_paths()` function
  - [ ] Takes: ReviewStateStore + List[MatchResult]
  - [ ] Returns: dict with `copy_only_paths`, `allowed_replacement_paths`
  - [ ] Converts track_ids back to file paths

### Plan Building Integration
- [ ] Modify `compute_library_sync_plan()` to accept:
  - [ ] `copy_only_paths` parameter
  - [ ] `allowed_replacement_paths` parameter
- [ ] Update decision logic in `_compute_plan_items()`:
  - [ ] If path in `copy_only_paths` → force COPY
  - [ ] If path in `allowed_replacement_paths` → allow REPLACE
  - [ ] Otherwise → auto-decide as before

---

## Phase 2: Scan & Fingerprinting

### Scan Logic
- [ ] Implement `scan_incoming_folder()` function
  - [ ] Find all audio files recursively
  - [ ] Extract metadata for each
  - [ ] Generate Chromaprint fingerprints
  - [ ] Cache fingerprints
  - [ ] Return List[AudioFile] with fingerprints

### Comparison Logic
- [ ] Implement `compare_fingerprints()` function
  - [ ] For each incoming file, find matches in library
  - [ ] Calculate distance for each match
  - [ ] Classify as NEW/MATCHED/COLLISION/UPGRADE
  - [ ] Return List[MatchResult]

### Threading
- [ ] Use daemon threads for scan operations
- [ ] Update UI via `widget.after(0, callback)` from worker threads
- [ ] Show progress bars for both library and incoming folder

### Error Handling
- [ ] Handle missing files gracefully
- [ ] Handle corrupted audio files (skip with warning)
- [ ] Handle permission denied (show error, don't crash)
- [ ] Handle out of disk space (pause, ask user)

---

## Phase 3: UI — Review Stage

### Incoming Tracks Table
- [ ] Create table widget with 5 columns:
  - [ ] Track (file name)
  - [ ] Status (NEW / MATCHED / COLLISION / UPGRADE)
  - [ ] Distance (numeric, right-aligned, "—" for NEW)
  - [ ] Flag (emoji: empty / 📋 / ↻)
  - [ ] Note (text, truncated if long)

### Table Features
- [ ] Sortable columns (click header to sort)
- [ ] Filterable (filter by status, show only flagged, search by name)
- [ ] Expandable rows (click to expand → show detailed view)
- [ ] User-friendly row height (30-40px per row)

### Context Menu (Right-Click)
- [ ] Implement right-click context menu with 4 options:
  - [ ] 📋 Copy (enable/disable based on status)
  - [ ] ↻ Replace (enable/disable if no match exists)
  - [ ] ✕ Clear flag (enable if already flagged)
  - [ ] 📝 Add note (always enabled)

### Copy Flag Action
- [ ] Get track_id from clicked row (stored in UserRole)
- [ ] Call `ReviewStateStore.flag_for_copy(track_id)`
- [ ] Update row: Flag column shows 📋
- [ ] Log: "Flagged [track_id] for copy"

### Replace Flag Action
- [ ] Get track_id and validate match exists
  - [ ] If no match: Show warning dialog "Cannot Replace — No existing match found"
  - [ ] If match exists: Continue
- [ ] Call `ReviewStateStore.flag_for_replace(track_id, existing_id)`
- [ ] Update row: Flag column shows ↻
- [ ] Log: "Flagged [track_id] to replace [existing_id]"

### Clear Flag Action
- [ ] Get track_id from clicked row
- [ ] Call `ReviewStateStore.clear_flags(track_id)`
- [ ] Update row: Flag column becomes empty
- [ ] Log: "Cleared flags for [track_id]"

### Add Note Action
- [ ] Show dialog: "Add Note — Notes for this track:"
  - [ ] Pre-fill with existing note if any
  - [ ] Multi-line text area
  - [ ] [OK] [Cancel] buttons
- [ ] Call `ReviewStateStore.add_note(track_id, note_text)`
- [ ] Update row: Note column shows note (truncated)
- [ ] Log: "Added note to [track_id]"

### Summary Row
- [ ] Show at bottom of table:
  - [ ] "NEW: X | MATCHED: Y | COLLISION: Z | UPGRADE: W"
  - [ ] Update counts as user flags items
  - [ ] "Flagged: X items" (count of flagged tracks)

### Detailed View (Click Row)
- [ ] Expand row to show:
  - [ ] Incoming file details (path, size, bitrate, codec, metadata)
  - [ ] Existing match details (if any)
  - [ ] Fingerprint distance and comparison
  - [ ] User action (current flag + note)
  - [ ] Plan impact (what will happen if execute)

---

## Phase 4: Plan Building & Preview

### Build Plan Stage
- [ ] Create "Build Plan" button in toolbar
- [ ] On click:
  - [ ] Disable input form (grayed out)
  - [ ] Show progress dialog ("Building plan...")
  - [ ] Get ReviewStateStore flags
  - [ ] Call `resolve_review_flags_to_paths()`
  - [ ] Call `compute_library_sync_plan(copy_only_paths, allowed_replacement_paths)`
  - [ ] Store ExecutionPlan in memory
  - [ ] Generate LibrarySyncPreview.html
  - [ ] Show success message with summary

### Preview Generation
- [ ] Create `generate_library_sync_preview_html()` function
  - [ ] Header section: paths, threshold, timestamp
  - [ ] Summary section: counts (copy, replace, skip)
  - [ ] Copy operations table: source, destination, reason, note
  - [ ] Replace operations table: source, existing, reason, note
  - [ ] Skip operations table: incoming, existing, reason
  - [ ] Escape HTML special characters (security)
  - [ ] Write to `/Library/Docs/LibrarySyncPreview.html`

### Preview Dialog
- [ ] Create dialog with buttons:
  - [ ] [Preview in Browser] — open HTML in default browser
  - [ ] [Edit Flags] — go back to Stage 3 (preserve flags)
  - [ ] [Execute] — proceed to Stage 6

---

## Phase 5: Execution

### Execute Plan Stage
- [ ] Create "Execute Plan" button
- [ ] On click:
  - [ ] Show confirmation dialog (list what will happen)
  - [ ] If user confirms:
    - [ ] Disable form (grayed out)
    - [ ] Show execution progress dialog
    - [ ] Run execution in daemon thread

### Copy Operation
- [ ] Implement `execute_copy(source_path, dest_path)` function
  - [ ] Validate source exists
  - [ ] Create destination folder if needed
  - [ ] Check if destination already exists (edge case)
  - [ ] Copy file using `shutil.copy2()` (preserves metadata)
  - [ ] Log: "COPY [source] → [destination]"
  - [ ] Return success/failure

### Replace Operation
- [ ] Implement `execute_replace(source_path, dest_path)` function
  - [ ] Validate source and destination exist
  - [ ] Create backup: `dest_path + ".backup"`
  - [ ] Delete destination file
  - [ ] Copy source to destination
  - [ ] Log: "REPLACE [source] → [destination] (backup: [backup])"
  - [ ] On error: Restore from backup, log error

### Skip Operation
- [ ] Implement `execute_skip()` function
  - [ ] Do nothing
  - [ ] Log: "SKIP [file]"

### Execution Progress
- [ ] Update progress dialog with:
  - [ ] Overall progress bar (X / 200 files)
  - [ ] Current file name and size
  - [ ] Current operation (copying / replacing)
  - [ ] Speed (MB/s)
  - [ ] Time remaining (estimate)
  - [ ] Counts so far: X copied, Y replaced, Z skipped
  - [ ] Cancel button (finishes current file, then stops)

### Execution Logging
- [ ] Log every operation:
  - [ ] Timestamp
  - [ ] Operation type (COPY / REPLACE / SKIP)
  - [ ] Source and destination paths
  - [ ] File size and result (success/failure)
  - [ ] Error message if failed

---

## Phase 6: Execution Report

### Report Generation
- [ ] Create `generate_library_sync_execution_report_html()` function
  - [ ] Header: title, start time, end time, status
  - [ ] Summary: counts (copied, replaced, skipped, failed)
  - [ ] Successful operations table: all copy/replace operations
  - [ ] Failed operations table (if any): path, error message
  - [ ] Backup information: location, count, retention policy
  - [ ] Timing information: duration in minutes
  - [ ] Write to `/Library/Docs/LibrarySyncExecutionReport.html`

### Report Dialog
- [ ] Show after execution completes:
  - [ ] Summary (X copied, Y replaced, Z skipped, W failed)
  - [ ] Time taken
  - [ ] Status (✓ SUCCESS / ⚠ PARTIAL / ✗ FAILED)
  - [ ] Backup location (if replacements made)
  - [ ] Buttons: [View Report] [View Backups] [Done]

### Post-Execution Cleanup
- [ ] Clear ReviewStateStore (optional — reset for next scan)
- [ ] Clear ExecutionPlan from memory
- [ ] Re-enable form for new sync
- [ ] Optionally clear incoming tracks table

---

## Phase 7: UI — Workspace Integration

### Library Sync Workspace
- [ ] Create new tab or workspace called "Library Sync"
- [ ] Add to workspace selector in main_gui.py

### Main Panel Layout
- [ ] Top section: Input form
  - [ ] Existing library path input + Browse button
  - [ ] Incoming folder path input + Browse button
  - [ ] Threshold slider (0.0 - 1.0)
  - [ ] Preset buttons (Exact / Close / Loose / Reset)
  - [ ] Status message ("Ready to scan", "Scanning...", etc.)
  - [ ] Buttons: [Scan] [Clear] [Settings]

- [ ] Middle section: Incoming tracks table
  - [ ] Table with 5 columns (Track, Status, Distance, Flag, Note)
  - [ ] Summary row at bottom
  - [ ] Expandable rows for detail view

- [ ] Bottom section: Action buttons
  - [ ] [Build Plan]
  - [ ] [Preview Plan]
  - [ ] [Execute Plan]

### Progress Indicators
- [ ] Progress bar for existing library scan
- [ ] Progress bar for incoming folder scan
- [ ] Overall status message

### Input Validation UI
- [ ] Show checkmarks (✓) when paths are valid
- [ ] Show errors (✗) if paths invalid
- [ ] Disable Scan button if validation fails

---

## Phase 8: Error Handling & Edge Cases

### Validation
- [ ] Existing library path:
  - [ ] Must exist ✓
  - [ ] Must be directory ✓
  - [ ] Must be readable ✓
  - [ ] Error if empty (no audio files) ⚠
- [ ] Incoming folder path:
  - [ ] Must exist ✓
  - [ ] Must be directory ✓
  - [ ] Must be readable ✓
  - [ ] Error if empty (no audio files) ⚠
- [ ] Threshold:
  - [ ] Must be numeric
  - [ ] Must be 0.0 - 1.0

### Error Dialogs
- [ ] Path not found → show error, don't proceed
- [ ] Permission denied → show error, suggest solutions
- [ ] Disk full during execution → pause, ask to free space
- [ ] File locked during copy → retry 3x, then skip
- [ ] Corrupted audio → log warning, try to copy anyway
- [ ] Replace without match → show warning, don't allow
- [ ] Replace on file that was deleted → treat as copy instead

### Graceful Degradation
- [ ] If match objects unavailable:
  - [ ] ReviewStateStore flags stored
  - [ ] Plan building proceeds with auto-decisions
  - [ ] Flags silently ignored (logged as warning)
  - [ ] No crash
- [ ] If scan interrupted:
  - [ ] Show partial results
  - [ ] Allow user to rescan
  - [ ] Clear all state for fresh scan

### Cancellation
- [ ] Scan can be cancelled mid-way
  - [ ] Partial results shown (what's done so far)
- [ ] Execution can be cancelled mid-way
  - [ ] Finishes current file being copied
  - [ ] Stops, shows what completed
  - [ ] User can re-run to finish remaining files

---

## Phase 9: Testing

### Unit Tests
- [ ] ReviewStateStore
  - [ ] `test_flag_for_copy()` — add to copy list
  - [ ] `test_flag_for_replace()` — add to replace dict
  - [ ] `test_clear_flags()` — remove flags
  - [ ] `test_add_note()` — store note
  - [ ] `test_get_flags()` — return state dict

- [ ] Comparison logic
  - [ ] `test_classify_new_track()` — 0 matches → NEW
  - [ ] `test_classify_exact_match()` — ~0.0 distance → MATCHED
  - [ ] `test_classify_collision()` — < 0.1 distance → COLLISION
  - [ ] `test_classify_upgrade()` — 0.1-0.3 distance → UPGRADE

- [ ] Plan building
  - [ ] `test_flagged_copy_overrides_skip()` — copy flag forces COPY
  - [ ] `test_flagged_replace_overrides_skip()` — replace flag forces REPLACE
  - [ ] `test_unflagged_auto_decides()` — no flag → auto-decision
  - [ ] `test_cannot_replace_without_match()` — validation works

- [ ] Path resolution
  - [ ] `test_resolve_flags_to_paths()` — track_id → file path conversion
  - [ ] `test_handle_missing_match_objects()` — graceful degradation

### Integration Tests
- [ ] `test_full_scan_to_plan_workflow()` — complete flow
- [ ] `test_user_flags_affect_plan()` — flags actually change plan
- [ ] `test_preview_generation()` — HTML generated correctly
- [ ] `test_execution_copies_files()` — files actually copied
- [ ] `test_execution_replaces_files()` — files actually replaced
- [ ] `test_execution_creates_backups()` — backups made for replaced files

### UI Tests
- [ ] `test_right_click_context_menu()` — menu appears
- [ ] `test_copy_flag_shows_emoji()` — flag appears in table
- [ ] `test_note_dialog_appears()` — dialog opens on note action
- [ ] `test_table_sortable()` — columns can be sorted
- [ ] `test_table_filterable()` — can filter by status

### Manual Testing
- [ ] Scan two small test libraries (5-10 files each)
- [ ] Flag 2-3 items in different ways
- [ ] Verify plan shows correct decisions
- [ ] Preview HTML looks correct
- [ ] Execute plan and verify files copied/replaced
- [ ] Check execution report
- [ ] Verify backups created
- [ ] Test cancellation mid-execution

---

## Phase 10: Documentation & Polish

### Code Documentation
- [ ] Docstrings for all major functions
- [ ] Comments explaining complex logic
- [ ] Type hints for parameters and returns
- [ ] Architecture diagram in code comments

### User Documentation
- [ ] Add Library Sync section to README.md
- [ ] Document per-item flags feature
- [ ] Screenshot examples of workflow
- [ ] Troubleshooting guide for common errors

### Performance Optimization
- [ ] Profile scan stage (optimize fingerprinting)
- [ ] Profile plan building (optimize comparison)
- [ ] Cache fingerprints from previous scans
- [ ] Show loading indicators for long operations

### Polish
- [ ] Consistent emoji usage (📋, ↻, ✕, 📝, etc.)
- [ ] Clear status messages ("Ready to scan", "Scanning...", "Building plan...")
- [ ] Intuitive button layout
- [ ] Helpful tooltips on all controls
- [ ] Keyboard shortcuts (if applicable)

---

# Implementation Order

## Recommended Build Sequence

```
1. Phase 1: Data structures (ReviewStateStore, MatchResult enhancement)
   → No UI needed, can test independently

2. Phase 2: Scan & fingerprinting
   → Backend logic, no UI (but output visible in logs)

3. Phase 3: UI — Review stage
   → Build table + context menu

4. Phase 4: Plan building
   → Integrate with Phase 1 data

5. Phase 5: Execution
   → Make changes actually happen

6. Phase 6: Reports
   → Show user what happened

7. Phase 7: Workspace integration
   → Wire into main_gui.py

8. Phase 8: Error handling
   → Make everything robust

9. Phase 9: Testing
   → Verify all cases

10. Phase 10: Polish & documentation
    → Make it shine
```

---

# Key Decisions

| Decision | Why |
|----------|-----|
| Track IDs from normalized paths | Stable across rescans (unless path changes) |
| ReviewStateStore in-memory only | Prevents stale flags, simpler implementation |
| Flags cleared on rescan | Match results change with threshold, old flags invalid |
| HTML preview before execution | Prevents accidents, shows what will happen |
| Backups for replaced files (kept 30 days) | Safety margin, user can recover |
| Parallel scan threads | Fast fingerprinting on both sides |
| Daemon threads for long ops | UI stays responsive |
| No deletions from incoming | Safe by default, user decides |

---

# Integration Points with Existing Code

| Module | Integration Point | Change Needed |
|--------|------------------|---------------|
| `library_sync.py` | Plan building function | Add `copy_only_paths` and `allowed_replacement_paths` params |
| `near_duplicate_detector.py` | Fingerprint comparison | Use for MatchResult generation |
| `main_gui.py` | Library Sync workspace | Add new tab/workspace |
| `config.py` | User settings | Store threshold, preserve last paths |
| `library_sync_review.py` | Existing review UI | Enhance with flags feature |
| `fingerprint_cache.py` | Cache management | Use for scan optimization |

---

# Success Criteria

Verify after implementation:

- ✅ Scan fingerprints 1000 files in < 10 minutes
- ✅ User can flag 100 tracks in < 10 minutes
- ✅ Plan building for 500 files takes < 30 seconds
- ✅ HTML preview renders in < 2 seconds
- ✅ Execution copies 100 files in < 5 minutes
- ✅ Execution report generated in < 10 seconds
- ✅ Zero data loss (no files deleted by mistake)
- ✅ Backups created for all replaced files
- ✅ All operations logged with timestamps
- ✅ User can resume after cancellation
- ✅ All tests pass (unit + integration + manual)
- ✅ Documentation complete and clear

---

# Quick Links to Detailed Specs

For more detailed information, refer to:

- **Per-Item Flags Feature:** `docs/FEATURE_SPECIFICATION.md` (400 lines)
  - Complete breakdown of every user interaction
  - All edge cases and special behaviors
  - Technical data flows
  - Quality of life details
  - Example user journeys

- **Library Sync Complete System:** `docs/LIBRARY_SYNC_COMPLETE_SPEC.md` (750 lines)
  - High-level architecture diagram
  - 7-stage workflow in detail
  - UI mockups and forms
  - Algorithm pseudocode
  - Configuration options
  - Performance targets

---

# Ready to Begin?

This document provides:
1. ✅ Complete feature understanding (2 specs)
2. ✅ Full workflow documentation (7 stages)
3. ✅ Implementation checklist (100+ tasks)
4. ✅ Testing guide (unit + integration + manual)
5. ✅ Integration points with existing code
6. ✅ Success criteria for verification

**All tasks are actionable and specific.**
**No vague "implement X" — each task is concrete.**
**Can be implemented in any order, but suggested sequence provided.**

---

**Status:** 🟢 Ready for implementation
**Branch:** `claude/audit-startup-splash-PwNwu`
**Session:** Ready to build

