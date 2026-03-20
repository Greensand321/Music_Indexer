# Library Sync Per-Item Review — Testing Guide

## Status
Phase 4: Testing & Polish - Manual Testing Plan

## Code Quality Checks (Automated)
✅ **Syntax validation:**
- `library_sync.py` — syntax check passed
- `gui/workspaces/library_sync.py` — syntax check passed

✅ **Import verification:**
- ReviewStateStore imports correctly
- ReviewStateStore -> MatchResult dependency verified
- All new function signatures accept correct parameters

---

## Manual Integration Tests

These tests require a full working environment with:
- Python 3.11+
- All dependencies from `requirements.txt` installed
- FFmpeg on PATH
- VLC/libVLC installed

### Test Setup

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare test libraries:**
   - Create test existing library: `/tmp/music_existing/` with 3-5 sample MP3 files
   - Create test incoming folder: `/tmp/music_incoming/` with:
     - 2 new files (not in existing)
     - 1 file that's an exact match (same fingerprint as existing)
     - 1 file that's a potential upgrade (higher quality codec)

3. **Run the application:**
   ```bash
   python main_gui.py
   ```

---

## Test Scenarios

### Test 1: Scan and Display Incoming Tracks with Track IDs
**Objective:** Verify that scan displays tracks and stores track_ids properly

**Steps:**
1. Open Library Sync workspace
2. Set "Existing Library" to `/tmp/music_existing/`
3. Set "Incoming Folder" to `/tmp/music_incoming/`
4. Click "🔍 Scan Both Libraries"

**Expected Results:**
- ✓ Scan completes without error
- ✓ Incoming Tracks table shows 4 rows with columns: Track | Status | Distance | Flag | Note
- ✓ Flag and Note columns are initially empty
- ✓ Each row displays file name and match status (New/Collision/etc.)

**Failure Points:**
- Table columns don't display (check _build_ui() for column configuration)
- Tracks don't appear (check _populate_results() and match_objects storage)
- No track data shown (check include_match_objects=True in SyncScanWorker)

---

### Test 2: Right-Click Context Menu
**Objective:** Verify context menu appears and displays correct options

**Steps:**
1. After successful scan, right-click on one of the new incoming tracks
2. Observe the context menu

**Expected Results:**
- ✓ Menu appears with options: 📋 Copy | ↻ Replace | ✕ Clear flag | 📝 Add note
- ✓ Menu actions are responsive

**Failure Points:**
- Menu doesn't appear (check setContextMenuPolicy and customContextMenuRequested signal)
- Actions don't trigger (check lambda connections)

---

### Test 3: Flag Track for Copy
**Objective:** Verify copy flag is set and displayed

**Steps:**
1. Right-click on a "New" track → select "📋 Copy"
2. Check the Flag column on that row

**Expected Results:**
- ✓ Flag column shows "📋 Copy"
- ✓ Log area shows "Flagged [track_id] for copy"
- ✓ Flag persists when you click on other rows and back
- ✓ Can right-click → "✕ Clear flag" to remove it

**Failure Points:**
- Flag not displayed (check _update_incoming_item_flags() column index)
- Flag doesn't persist (check ReviewStateStore storing correctly)
- Clear flag doesn't work (check unflag_copy and flags.replace cleanup)

---

### Test 4: Flag Track for Replace
**Objective:** Verify replace flag is set only when a match exists

**Steps:**
1. Find a track with match (Status = "Collision" or "Upgrade")
2. Right-click → select "↻ Replace"
3. Check Flag column

**Expected Results:**
- ✓ Flag column shows "↻ Replace"
- ✓ Log shows "Flagged [track_id] to replace [existing_id]"

**Try on a New track (no match):**
4. Right-click a "New" track → select "↻ Replace"

**Expected Results:**
- ✓ Warning dialog: "Cannot Replace — No existing match found for this track."
- ✓ Flag is NOT set

**Failure Points:**
- Replace allowed on unmatched tracks (check match validation in _flag_for_replace())
- Flag persists incorrectly (check _update_incoming_item_flags() logic)

---

### Test 5: Add Notes to Tracks
**Objective:** Verify notes are stored and displayed

**Steps:**
1. Right-click a track → select "📝 Add note"
2. Type a note (e.g., "Higher bitrate version")
3. Click OK

**Expected Results:**
- ✓ Note column shows the note text (truncated if long)
- ✓ Log shows "Added note to [track_id]"
- ✓ Can edit note again by right-clicking and selecting "📝 Add note"
- ✓ Empty note clears the display

**Failure Points:**
- Note dialog doesn't appear (check QInputDialog call)
- Note not displayed (check _update_incoming_item_flags() column 4)
- Note overwrites flag (columns should be independent)

---

### Test 6: Build Plan with Flagged Items
**Objective:** Verify plan builder respects user flags

**Steps:**
1. After scan, flag 1-2 items:
   - Flag a "New" track for copy
   - Flag a matched track for replace
2. Click "📋 Build Plan"
3. Wait for plan to complete
4. Click "👁 Preview Plan" to view HTML

**Expected Results:**
- ✓ Plan builds without error
- ✓ Preview opens in browser
- ✓ In preview, flagged tracks show appropriate actions (COPY or REPLACE)
- ✓ Log shows "Building Library Sync plan and preview…" then success message

**HTML Preview Check:**
- Flagged-for-copy items show in COPY section
- Flagged-for-replace items show in REPLACE section
- Reason column shows "User flag" for overridden items

**Failure Points:**
- Plan fails to build (check SyncBuildWorker error handling)
- Flags ignored in plan (check resolve_review_flags_to_paths() conversion)
- HTML preview doesn't reflect flags (check plan.items generation)

---

### Test 7: Scan After Clearing Flags
**Objective:** Verify old flags don't persist across scans

**Steps:**
1. Set some flags on current scan results
2. Change threshold value (e.g., 0.3 → 0.25)
3. Click "↺ Recompute Matches"
4. Wait for new scan

**Expected Results:**
- ✓ Old flags are cleared (Flag and Note columns now empty)
- ✓ New match results displayed
- ✓ Log shows scan starting fresh

**Failure Points:**
- Old flags still visible (check clear_all() call in _on_scan())
- New results mixed with old (check _incoming_table.clear())

---

### Test 8: Clear Flag Action
**Objective:** Verify individual flag clearing

**Steps:**
1. Flag a track for copy
2. Right-click it → "✕ Clear flag"

**Expected Results:**
- ✓ Flag column becomes empty
- ✓ Log shows "Cleared flags for [track_id]"
- ✓ Flag is actually gone (not just visually hidden)

**Failure Points:**
- Clear fails on replace flag (check flags.replace cleanup in _flag_clear())
- Clear doesn't update display (check _update_incoming_item_flags())

---

## Known Limitations (By Design)

1. **Session persistence:** Flags are cleared when you close the workspace. This is intentional (in-memory only for now). User can take notes before closing if needed.

2. **Threshold changes:** Recomputing with new thresholds clears all flags. This is intentional because best matches may change and replace flags need reconciliation.

3. **No undo/redo:** Individual flag changes are not undoable. Users must clear and re-flag.

4. **Match object dependency:** The resolve_review_flags_to_paths() function requires match objects. If scan doesn't return them, flags will be silently ignored during plan building. This is a safety fallback.

---

## Success Criteria Checklist

- [ ] Scan displays tracks with all 5 columns (Track, Status, Distance, Flag, Note)
- [ ] Right-click context menu appears with 5 options
- [ ] Copy flag can be set and displayed (emoji visible)
- [ ] Replace flag validates that match exists (warning on no match)
- [ ] Notes can be added, edited, and cleared
- [ ] Plan builder runs with flagged items
- [ ] HTML preview shows flagged items with correct actions
- [ ] Plan execution respects flags (files are copied/replaced as flagged)
- [ ] Clearing flags from GUI removes them from display
- [ ] New scan clears old flags automatically

---

## Debugging Tips

If tests fail:

1. **Check log area in GUI** — shows detailed operation results
2. **Check browser console** (for HTML preview) — JavaScript errors if any
3. **Check Python console** — traceback if background worker crashes
4. **Verify sync folders exist** and contain audio files
5. **Enable debug mode** in library_sync.py (set `debug = True`)
6. **Check file permissions** — existing lib should be readable, incoming should be readable

---

## Test Results Log

| Test | Date | Result | Notes |
|------|------|--------|-------|
| (pending manual run) | — | — | Full environment required |

