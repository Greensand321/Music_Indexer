# Library Sync Per-Item Review Flags — Complete Specification

## Overview

The **Per-Item Review Flags** feature allows users to override automatic file disposition decisions on a track-by-track basis during Library Sync operations. Instead of accepting the system's auto-decided copy/skip/replace logic, users can right-click individual incoming tracks to flag them with explicit instructions.

---

## The Problem It Solves

**Scenario:** You're syncing an incoming folder to your existing library. The system fingerprints everything and decides:
- Track A: "This is new → copy it"
- Track B: "This matches existing file → skip it"
- Track C: "This is a potential upgrade → mark for review"

**But you know:**
- Track A is actually a duplicate you want to skip (system threshold was wrong)
- Track B is a better quality version you want to replace the existing one
- Track C isn't an upgrade at all — you want to skip it

**Without this feature:** You can't tell the system your preferences. It builds a plan based on auto-logic, and you're stuck with it.

**With this feature:** You flag each track with your decision, and the plan builder respects your overrides.

---

## User Workflow (Step-by-Step)

### Step 1: Open Library Sync Workspace
```
User clicks: Library Sync tab (or workspace selector)
System shows: Empty form with path inputs and threshold controls
```

### Step 2: Set Input Paths
```
User enters:
- "Existing Library": /mnt/library/Music/
- "Incoming Folder": /mnt/incoming/NewTracks/
- "Threshold": 0.3 (optional override)

System stores paths for scan
```

### Step 3: Run Scan
```
User clicks: "🔍 Scan Both Libraries" button

System:
1. Fingerprints all files in existing library
2. Fingerprints all files in incoming folder
3. Compares fingerprints using threshold
4. Generates MatchResult objects with track_ids
5. Displays results in incoming tracks table

UI updates:
- Two progress bars fill (existing scan, incoming scan)
- Status changes from "Scanning..." to "Complete"
- Incoming tracks table populates with results
```

### Step 4: Review Results Table

**Table displays (5 columns):**

| Column | Content | Example |
|--------|---------|---------|
| **Track** | File name | `Song Title.mp3` |
| **Status** | Match classification | New / Collision / Upgrade / Matched |
| **Distance** | Fingerprint similarity (0.0–1.0) | 0.045 (lower = more similar) |
| **Flag** | User override status | (empty) or 📋 Copy or ↻ Replace |
| **Note** | User-added context | "Higher bitrate" |

**Key detail:** Each row has a hidden `track_id` (stable ID based on normalized file path) used for flag lookups.

### Step 5: Right-Click to Flag (The Core Action)

User right-clicks an incoming track row → **Context menu appears** with 5 options:

```
┌─────────────────────────────────┐
│ 📋 Copy                         │
│ ↻ Replace                       │
│ ✕ Clear flag                    │
│ ─────────────────────────────   │
│ 📝 Add note                     │
└─────────────────────────────────┘
```

#### **Option 1: 📋 Copy**
```
What happens:
- Flag is SET for this track's track_id
- UI updates: Flag column shows "📋 Copy"
- Log shows: "Flagged [track_id] for copy"

Behavior:
- Overrides skip decision → file WILL be copied
- Removes any replace flag if it was set
- Persists across other row selections
- Can be cleared with "✕ Clear flag"

Use case:
- "This looks new to me, copy it despite the distance"
- "This should be included even though it matches"
```

#### **Option 2: ↻ Replace**
```
What happens:
- System validates: Does this track have a match?
  - YES: Sets replace flag, Flag column shows "↻ Replace"
  - NO: Shows warning: "Cannot Replace — No existing match found"
- If valid: Log shows "Flagged [track_id] to replace [existing_id]"

Behavior:
- Destination path (existing match) is added to allowed_replacements
- File WILL be written over the existing file
- Can be cleared with "✕ Clear flag"
- Only works if a matching file exists in library

Use case:
- "This incoming file is better quality, replace the library copy"
- "This is an upgraded codec version I want to use"
- "The existing file is corrupted, use this one instead"
```

#### **Option 3: ✕ Clear flag**
```
What happens:
- Both copy AND replace flags removed
- UI updates: Flag column becomes empty
- Log shows: "Cleared flags for [track_id]"

Behavior:
- Reverts to auto-decided behavior
- Note is NOT cleared (notes persist)

Use case:
- "Actually, I changed my mind about this one"
- "Use the system's decision after all"
```

#### **Option 4: 📝 Add note**
```
What happens:
- Dialog opens: "Add Note — Notes for this track:"
- Pre-filled with existing note if one exists
- User types multi-line note
- Clicks OK

UI updates:
- Note column shows the note text (truncated if long)
- Note persists even if flag is cleared
- Log shows: "Added note to [track_id]"

Behavior:
- Notes are independent of flags
- Can add note without flagging
- Note appears in HTML preview
- Cleared when scan restarts

Use case:
- "Higher bitrate version"
- "Check metadata before copying"
- "Potential test file, review first"
- "This file is problematic"
```

### Step 6: Review Multiple Tracks (Repeat Step 5)

User can flag as many tracks as needed:
```
Loop:
1. Right-click incoming track
2. Select action from menu
3. See flag appear in Flag column
4. Optionally add note
5. Click next track or repeat

System maintains all flags in ReviewStateStore
Flags persist until scan is restarted or explicitly cleared
```

### Step 7: Build Plan (Flags Are Applied Here)

```
User clicks: "📋 Build Plan"

System executes:
1. Retrieves all flags from ReviewStateStore
2. Converts track_ids to source file paths
3. Calls compute_library_sync_plan() with overrides:
   - copy_only_paths = [list of flagged-for-copy paths]
   - allowed_replacement_paths = [list of replaceable destinations]
4. Plan builder generates execution decisions:
   - Flagged-for-copy files → ExecutionDecision.COPY
   - Flagged-for-replace destinations → ExecutionDecision.REPLACE
   - Unflagged files → Auto-decided (COPY/SKIP based on match)

UI updates:
- Progress bar shows computation progress
- Status shows "Building plan + preview..."
- Plan status shows counts and summary
```

### Step 8: Preview Plan (HTML Report)

```
User clicks: "👁 Preview Plan" button

System:
- Opens HTML preview at /Library/Docs/LibrarySyncPreview.html
- Shows all planned operations

Preview shows (for each file):
- Source path
- Destination path
- **Execution decision** (COPY / REPLACE / SKIP)
- **Reason** (why this decision was made)
  - Normal reasons: "Exact duplicate", "Destination exists"
  - **With flags:** "User flag: Copy" or "User flag: Replace"
- Notes (if any were added)

User can:
- Verify flags were applied correctly
- See which files will be affected
- Check for unintended consequences
- Decide whether to proceed
```

### Step 9: Execute Plan (Flags Control Behavior)

```
User clicks: "▶ Execute Plan" button

System:
1. Iterates through planned items
2. For each item with ExecutionDecision:
   - COPY: Copies source to destination
   - REPLACE: Overwrites existing file with source
   - SKIP: Leaves file alone
3. Logs every action to /Library/Docs/execution_report.html
4. Reports results: "X files copied, Y replaced, Z skipped"

Critical: Flags determine which files actually get moved/replaced
```

---

## What Happens in Different Scenarios

### **Scenario 1: New Track, No Match**

**Status shown:** New
**Auto-decision:** Copy
**Available actions:**
- ✓ Copy (makes it explicit)
- ✗ Replace (disabled — no match to replace)
- ✓ Clear flag (reverts to auto)

**If flagged for Copy:** Copied (same as auto)
**If not flagged:** Copied (auto-decided)

---

### **Scenario 2: Matched Track (Collision)**

**Status shown:** Collision
**Auto-decision:** Skip (keep existing)
**Available actions:**
- ✓ Copy (force copy despite match)
- ✓ Replace (replace existing with this)
- ✓ Clear flag (skip, use auto)

**Outcomes:**
| Flag | Result |
|------|--------|
| (none) | Skip — keep existing file |
| Copy | Copy — adds new file (might create duplicates) |
| Replace | Replace — overwrites existing with incoming |

---

### **Scenario 3: Upgrade Candidate**

**Status shown:** Upgrade
**Auto-decision:** Mark for review (user decides)
**Available actions:**
- ✓ Copy (add both to library)
- ✓ Replace (use newer version)
- ✓ Clear flag (manually skip)

**Outcomes:**
| Flag | Result |
|------|--------|
| (none) | Depends on user choice in plan building |
| Copy | Copy — keeps both versions |
| Replace | Replace — overwrites with upgrade |

---

### **Scenario 4: User Adds Note but No Flag**

**Status shown:** (any)
**Auto-decision:** (whatever system decides)
**User action:** Adds note "Higher bitrate"

**Result:**
- Note is stored and displayed
- File gets auto-decided treatment
- Note appears in HTML preview for reference
- User can flag later if needed

---

## Edge Cases & Special Behaviors

### **What Happens When You Rescan?**

```
Old state:
- Track A: Flagged for copy, Note: "Check this"
- Track B: Flagged for replace
- Track C: Note added

User changes threshold and clicks "↺ Recompute"

New state:
- ALL flags cleared ✓
- ALL notes cleared ✓
- Table repopulated with new scan results
- User can flag new results fresh

Reason: Match results change with new threshold,
so old flags may no longer apply to same tracks
```

### **What If Existing Match Is Deleted?**

```
User flags Track A for replace
- System validates match exists
- User deletes the existing file manually
- User builds plan anyway

Result:
- Plan builder finds no destination file
- Treats it as new copy (not replace)
- File is copied instead
- No error (graceful degradation)
```

### **What If User Flags, Then Changes Settings?**

```
Scenario 1: User flags Track A, changes folder, scans again
Result: Old flags are cleared (different folder = different tracks)

Scenario 2: User flags Track A, changes threshold, computes again
Result: Old flags are cleared (different matches possible)

Reason: Flags are tied to specific track_ids from specific scan
New scan = new track_ids = old flags invalid
```

### **Can You Flag the Same Track Twice?**

```
User flags Track A for Copy
User right-clicks Track A again, flags for Replace

Result:
- Copy flag is removed
- Replace flag is set
- Only one active flag per track

Behavior: Last action wins
```

### **What If No Match Objects Are Available?**

```
Scenario: Scan completes but include_match_objects=False
(shouldn't happen, but could if code is changed)

Result:
- ReviewStateStore has flags
- But resolve_review_flags_to_paths() gets empty match_results
- Flags are silently ignored
- Plan builds with auto-decisions only
- No error (safety fallback)

Log shows: No warning (silent degradation)
```

---

## Technical Data Flow

### **Backend → Frontend Data Path**

```
compare_libraries() scan
    ↓
Returns: {
    "match_objects": [MatchResult(...), MatchResult(...), ...]
    "new": [...],
    "matched": [...],
    ...
}
    ↓
GUI stores in self._current_match_results
    ↓
_populate_results() extracts track_id from each MatchResult
    ↓
Tree items created with track_id in Qt.UserRole
    ↓
User right-clicks → menu handler gets track_id
    ↓
ReviewStateStore.flag_for_copy(track_id) stores the flag
```

### **Frontend → Backend Data Path**

```
User flags multiple tracks
    ↓
ReviewStateStore accumulates all flags:
{
    "copy": ["track_001", "track_003"],
    "replace": {"track_002": "track_existing_002"},
    "notes": {"track_001": "Higher bitrate"}
}
    ↓
User clicks "Build Plan"
    ↓
SyncBuildWorker receives review_flags object
    ↓
resolve_review_flags_to_paths() converts:
{
    "copy": ["/path/to/incoming/file1.mp3"],
    "replace": ["/path/to/existing/file2.mp3"]
}
    ↓
compute_library_sync_plan() receives as parameters:
    copy_only_paths = ["/path/to/incoming/file1.mp3"]
    allowed_replacement_paths = ["/path/to/existing/file2.mp3"]
    ↓
_compute_plan_items() applies overrides:
    - Items in copy_only_paths → forced to COPY
    - Items in allowed_replacement_paths → allowed to REPLACE
    ↓
Plan items respect user decisions
```

---

## Visual Design

### **Table Appearance**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Incoming Tracks                                                     │
├────────────────┬──────────┬──────────┬──────────┬──────────────────┤
│ Track          │ Status   │ Distance │ Flag     │ Note             │
├────────────────┼──────────┼──────────┼──────────┼──────────────────┤
│ song1.mp3      │ New      │ —        │          │                  │
│ song2.mp3      │ Collision│ 0.045    │ 📋 Copy  │                  │
│ song3.mp3      │ Upgrade  │ 0.085    │ ↻ Replace│ Higher bitrate   │
│ song4.m4a      │ Matched  │ 0.000    │          │ Check first      │
│ song5.flac     │ New      │ —        │ 📋 Copy  │                  │
└────────────────┴──────────┴──────────┴──────────┴──────────────────┘
```

### **Context Menu Appearance**

```
Right-click on row → menu appears at cursor

Menu items (left-aligned with emoji):
┌─────────────────────────────────┐
│ 📋 Copy                         │ ← Enable file to be copied
│ ↻ Replace                       │ ← Allow overwrite of match
│ ✕ Clear flag                    │ ← Remove all flags
│ ─────────────────────────────── │ ← Separator
│ 📝 Add note                     │ ← Open note dialog
└─────────────────────────────────┘

Click outside menu to dismiss
```

### **Note Dialog**

```
Title: "Add Note"
Prompt: "Notes for this track:"
Input: Multi-line text area
Buttons: [OK] [Cancel]

If note already exists:
  Input field pre-filled with existing text
  User can edit or clear
```

### **HTML Preview Flagged Items**

```
For each flagged item, preview shows extra info:

Source File: incoming/song.mp3
Destination: library/song.mp3
Decision: COPY
Reason: User flag: Copy
Note: Higher bitrate version
───────────────────────────────────────

Source File: incoming/upgrade.flac
Destination: library/upgrade.flac
Decision: REPLACE
Reason: User flag: Replace existing
Note: Better quality codec
───────────────────────────────────────
```

---

## Limitations (By Design)

### **Session Not Persistent**
- Flags are cleared when app closes
- Notes are cleared when app closes
- Scan restart clears all flags
- **Rationale:** Prevents stale flags from different scans

### **No Undo/Redo**
- No way to undo individual flag changes
- Can only clear and re-flag
- **Workaround:** Check preview before executing

### **No Bulk Operations**
- Cannot flag multiple tracks at once
- Must right-click each track individually
- **Future enhancement:** Select multiple + bulk flag

### **Match Object Dependency**
- Requires match objects from scan
- Silent graceful degradation if unavailable
- **Safety measure:** Plan still works, just ignores flags

---

## Quality of Life Details

### **Log Output**

Every user action is logged:
```
"Flagged track_001 for copy"
"Flagged track_002 to replace track_existing_002"
"Cleared flags for track_003"
"Added note to track_004"
"Building Library Sync plan and preview…"
"Plan built — 47 file operation(s). Preview written."
```

**User benefit:** Can trace what flags were set before building plan

### **Emoji Indicators**

Chosen for quick visual recognition:
- 📋 = Copy (clipboard = transfer/copy concept)
- ↻ = Replace (circular arrow = overwrite/replace concept)
- ✕ = Clear (X = remove/clear concept)
- 📝 = Note (notepad = documentation concept)

### **Color/Highlighting**

(If implemented in future):
- Flagged rows could be highlighted
- Copy flags in one color, replace in another
- Note column text bolder/italic

---

## Example User Journey

### **Full Example: Syncing 100 Tracks**

```
1. User opens Library Sync
   - Sets path to /library/Music (existing)
   - Sets path to /incoming/NewMusic (incoming)
   - Keeps threshold at 0.3

2. Clicks "Scan" → waits 30 seconds

3. Results: 100 incoming tracks
   - 30 are "New" (no match)
   - 40 are "Matched" (exact duplicates, will skip)
   - 20 are "Collision" (similar, keep existing)
   - 10 are "Upgrade" (better quality, marked for review)

4. User reviews and flags:
   - Finds a "New" track that's actually a duplicate
     → Right-click → "Copy" (makes decision explicit)
   - Finds 3 "Upgrade" tracks that ARE better quality
     → Right-click each → "Replace"
   - Adds notes: "Check bitrate first", "Better metadata", etc.
   - Finds 2 "Collision" tracks that look like errors
     → Notes: "Verify fingerprint" (doesn't flag, just documents)

5. Clicks "Build Plan"
   - System builds plan with 4 copies (30 auto-new + 1 flagged) + 3 replacements
   - Skips 60 (40 auto-matched + 20 auto-collision)
   - Takes ~10 seconds

6. Clicks "Preview Plan"
   - Opens HTML showing 67 operations
   - Scans for flagged items to verify
   - Sees notes appear next to flagged items
   - Confirms "This looks right"

7. Clicks "Execute Plan"
   - Files are copied/replaced per plan
   - Execution report written to Docs/
   - Log shows: "Copied 4 files, replaced 3 files, skipped 60"

8. Done!
```

---

## Why This Design Was Chosen

| Design Choice | Alternative | Why Chosen |
|---------------|-------------|-----------|
| Right-click context menu | Buttons in sidebar | Minimal UI, close to source |
| Track-by-track flagging | Bulk operations | Intentional focus per item |
| In-memory only (no persistence) | Save/load sessions | Prevents stale flags |
| Track IDs in UserRole | Stored in separate list | Keeps data with UI element |
| Two params to plan builder | Single "flags" object | More flexible, explicit |
| Silent fallback if no match objects | Error dialog | Graceful degradation |

---

## Success Metrics

After implementation, verify:
- ✓ Flagged items appear in preview with "User flag" reason
- ✓ Right-click menu appears instantly (< 50ms)
- ✓ Flags don't persist across scans (intentional)
- ✓ Notes can be 200+ characters without breaking UI
- ✓ Plan builds 10x faster than scan (< 5 seconds for 1000 files)
- ✓ Replacing doesn't create backup of existing file
- ✓ Execution report logs all flagged decisions
- ✓ No file is copied/deleted incorrectly due to flag logic

---

## This Is What The Feature Does

You now have a complete picture of:
- ✅ **What problem it solves** (user overrides for file disposition)
- ✅ **How users interact with it** (right-click menu, flags, notes)
- ✅ **The exact workflow** (scan → flag → build → preview → execute)
- ✅ **Edge cases** (rescans, missing matches, no objects)
- ✅ **Data flow** (track_ids, ReviewStateStore, path conversion)
- ✅ **Visual design** (table, menu, dialogs)
- ✅ **Technical details** (parameters, decisions, HTML output)
- ✅ **Why each choice was made** (rationale table)

Ready to review in the test environment! 🎯

