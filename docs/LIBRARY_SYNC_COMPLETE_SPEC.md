# Library Sync — Complete System Specification

## Overview

**Library Sync** is a workflow for comparing two music libraries (existing vs. incoming) and systematically updating the existing library with new or better files from the incoming folder. It's designed to be safe, reversible, and give users full control before any files are touched.

**Core principle:** Preview-first. Every destructive or moving operation must be explicitly approved by the user after seeing exactly what will happen.

---

## The Problem It Solves

**Scenario:** You have a music library at `/mnt/music/` with 10,000 songs. You get a new USB drive with 500 songs to add. Some are new, some are duplicates of what you already have, some might be better versions (higher bitrate, better metadata).

**Without Library Sync:** You manually sort the USB drive, listen to samples, rename files, check for duplicates, delete old versions. Hours of work. Risk of mistakes.

**With Library Sync:**
1. Point to both folders
2. System fingerprints both
3. Shows you a plan of what will happen
4. You verify or override decisions
5. One click → done

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LIBRARY SYNC WORKFLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INPUT STAGE                                                 │
│  ├─ User provides: existing_library_path, incoming_folder_path │
│  ├─ User sets: fingerprint threshold (0.0-1.0)                 │
│  └─ Result: Two file paths ready for scan                       │
│                                                                 │
│  2. SCAN STAGE                                                  │
│  ├─ Fingerprint all files in existing library                   │
│  ├─ Fingerprint all files in incoming folder                    │
│  ├─ Compare fingerprints using threshold                        │
│  └─ Result: MatchResult objects describing relationships        │
│                                                                 │
│  3. REVIEW STAGE                                                │
│  ├─ Display incoming files in table                             │
│  ├─ User right-clicks to flag decisions (Optional)              │
│  ├─ User adds notes (Optional)                                  │
│  └─ Result: ReviewStateStore with user overrides                │
│                                                                 │
│  4. PLAN BUILDING STAGE                                         │
│  ├─ Take MatchResults + ReviewStateStore                        │
│  ├─ Compute disposition for each file                           │
│  │  (COPY / SKIP / REPLACE / QUARANTINE)                        │
│  ├─ Determine source & destination paths                        │
│  ├─ Group by disposition type                                   │
│  └─ Result: ExecutionPlan with operations                       │
│                                                                 │
│  5. PREVIEW STAGE                                               │
│  ├─ Generate HTML report of all operations                      │
│  ├─ Show: source, destination, decision, reason, note           │
│  ├─ User reviews in browser                                     │
│  ├─ User can go back and re-flag if needed                      │
│  └─ Result: User approval or modification                       │
│                                                                 │
│  6. EXECUTION STAGE                                             │
│  ├─ For each operation in ExecutionPlan:                        │
│  │  ├─ COPY: Copy source → destination                          │
│  │  ├─ SKIP: Do nothing                                         │
│  │  ├─ REPLACE: Backup existing, then copy source               │
│  │  └─ QUARANTINE: Move to quarantine folder                    │
│  ├─ Log every action                                            │
│  ├─ Handle errors gracefully                                    │
│  └─ Result: Files moved, library updated                        │
│                                                                 │
│  7. REPORT STAGE                                                │
│  ├─ Generate execution report                                   │
│  ├─ Show: files copied, replaced, skipped, quarantined          │
│  ├─ Show: errors (if any)                                       │
│  ├─ Show: timing                                                │
│  └─ Result: User can verify success                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: INPUT

### **What Happens**

User opens the Library Sync panel and is presented with a form.

### **UI Elements**

```
┌────────────────────────────────────────────────────────┐
│  LIBRARY SYNC                                          │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Existing Library:                                     │
│  [  /mnt/library/Music/  ]  [📁 Browse]               │
│  ✓ Folder exists and is readable                       │
│                                                        │
│  Incoming Folder:                                      │
│  [  /mnt/usb/NewMusic/   ]  [📁 Browse]               │
│  ✓ Folder exists and is readable                       │
│                                                        │
│  Fingerprint Threshold: [0.30 ◄─► 1.00]               │
│  Tooltip: "Lower = stricter matching, Higher = looser" │
│                                                        │
│  Preset Thresholds:                                    │
│  [Exact (0.02)] [Close (0.1)] [Loose (0.3)] [Reset]   │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │ ℹ Status: Ready to scan                          │ │
│  │ Files in library: scanning...                    │ │
│  │ Files in incoming: scanning...                   │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  [🔍 Scan] [Clear] [Settings]                         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### **Validation**

```
On "Scan" click:

1. Existing Library path:
   - Must exist ✓
   - Must be a directory ✓
   - Must be readable ✓
   - Cannot be empty (should have audio files) ⚠

2. Incoming Folder path:
   - Must exist ✓
   - Must be a directory ✓
   - Must be readable ✓
   - Cannot be empty (should have audio files) ⚠

3. Threshold:
   - Must be numeric
   - Must be between 0.0 and 1.0

If validation fails:
  → Show error dialog
  → Don't start scan
  → User fixes and tries again

If validation passes:
  → Disable form (grayed out)
  → Show progress bars
  → Start scanning both libraries in parallel
```

### **Stored State**

```python
{
  "existing_library_path": str,      # User's last choice
  "incoming_folder_path": str,        # User's last choice
  "threshold": float,                 # 0.0 - 1.0, default 0.3
  "last_scan_time": datetime,         # When last scan completed
  "last_scan_file_count": int         # How many files in last scan
}
```

---

## Stage 2: SCAN

### **What Happens**

System fingerprints both libraries and compares them.

### **Process (Parallel)**

```
THREAD 1: Scan Existing Library
├─ Recursively find all audio files
├─ Filter by extension (.mp3, .flac, .m4a, etc.)
├─ For each file:
│  ├─ Extract metadata (artist, title, duration)
│  ├─ Generate Chromaprint fingerprint
│  ├─ Cache fingerprint in SQLite
│  └─ Show progress (X / Y files fingerprinted)
└─ Return: List[AudioFile] with fingerprints

THREAD 2: Scan Incoming Folder
├─ Recursively find all audio files
├─ Filter by extension
├─ For each file:
│  ├─ Extract metadata
│  ├─ Generate Chromaprint fingerprint
│  ├─ Don't cache (temporary)
│  └─ Show progress (X / Y files fingerprinted)
└─ Return: List[AudioFile] with fingerprints

MAIN THREAD: Monitor
├─ Update progress bar 1: existing_files / total_existing
├─ Update progress bar 2: incoming_files / total_incoming
├─ When both complete:
│  ├─ Start comparison
│  └─ Display results
```

### **Comparison Phase**

```
For each incoming file:
  1. Get its fingerprint
  2. Find all existing files within threshold
     (distance <= user_threshold)
  3. Classify result:
     - 0 matches → Status: NEW
     - 1 match, distance ~0.0 → Status: MATCHED (exact duplicate)
     - 1 match, distance < 0.1 → Status: COLLISION (very similar)
     - 1 match, distance 0.1-0.3 → Status: UPGRADE (possible better version)
     - 2+ matches → Status: COLLISION (ambiguous, keep existing)
  4. Create MatchResult object:
     {
       "incoming_file": AudioFile,
       "existing_file": AudioFile (or None),
       "status": str,
       "distance": float,
       "track_id": str  # stable ID for flagging
     }

Result: List[MatchResult] (one per incoming file)
```

### **UI During Scan**

```
┌────────────────────────────────────────────────────────┐
│  SCANNING...                                           │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Existing Library:                                     │
│  ████████████░░░░░░░░░░░░░░░░░░░░  234 / 500 files   │
│                                                        │
│  Incoming Folder:                                      │
│  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░   45 / 200 files   │
│                                                        │
│  Estimated time remaining: 2 minutes                  │
│                                                        │
│  [Cancel]                                              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### **After Scan Completes**

```
Form re-enables, results table populated with incoming files

Status line shows:
"Scan complete: 500 files in library, 200 incoming files
 Fingerprints generated and compared
 Ready to review and build plan"
```

---

## Stage 3: REVIEW

### **What Happens**

User reviews incoming files and optionally flags decisions.

### **UI: Incoming Tracks Table**

```
┌──────────────────────────────────────────────────────────────────┐
│  Incoming Tracks (200 files)                                     │
├────────────────┬──────────┬──────────┬──────────┬────────────────┤
│ Track          │ Status   │ Distance │ Flag     │ Note           │
├────────────────┼──────────┼──────────┼──────────┼────────────────┤
│ song1.mp3      │ New      │ —        │          │                │
│ song2.mp3      │ Collision│ 0.045    │ 📋 Copy  │                │
│ song3.mp3      │ Upgrade  │ 0.085    │ ↻ Replace│ Higher bitrate │
│ song4.m4a      │ Matched  │ 0.000    │          │ Check first    │
│ song5.flac     │ New      │ —        │ 📋 Copy  │                │
│ ...            │ ...      │ ...      │ ...      │ ...            │
└────────────────┴──────────┴──────────┴──────────┴────────────────┘

Columns are sortable / filterable

Summary row at bottom:
"NEW: 50 | COLLISION: 60 | UPGRADE: 40 | MATCHED: 50"
```

### **Status Classification**

| Status | Meaning | Auto-Decision | Can Override |
|--------|---------|---------------|--------------|
| **NEW** | No match in existing library | Copy | Flag to force copy |
| **MATCHED** | Exact duplicate (distance ~0.0) | Skip | Flag to copy anyway |
| **COLLISION** | Very similar, but not exact | Skip | Flag to copy or replace |
| **UPGRADE** | Similar but might be better version | Mark for review | Flag to copy or replace |

### **User Actions (Right-Click Menu)**

See Stage 3 of Per-Item Flags spec (already detailed):
- 📋 Copy
- ↻ Replace
- ✕ Clear flag
- 📝 Add note

### **Table Filtering / Sorting**

User can:
- **Sort** by: Track name, Status, Distance, Flag, Note
- **Filter** by: Show only flagged items, show only NEW, show only MATCHED, etc.
- **Search** by: Track name (substring match)

Example:
```
[⚙ Filters ▼]
☑ Show NEW
☑ Show MATCHED
☑ Show COLLISION
☑ Show UPGRADE
☑ Show only flagged
📝 Search: "remix"

[Clear filters]
```

### **Detailed View (Click on Row)**

Clicking a row expands it to show:

```
┌─ song2.mp3 (expand/collapse)
├─ Incoming File:
│  ├─ Path: /mnt/usb/NewMusic/Song 2 Remix.mp3
│  ├─ Size: 4.2 MB
│  ├─ Duration: 3:45
│  ├─ Bitrate: 320 kbps
│  ├─ Codec: MP3
│  ├─ Metadata: Artist / Title / Album (extracted)
│  └─ Fingerprint: [show visual]
│
├─ Existing Match:
│  ├─ Path: /mnt/music/song_2.mp3
│  ├─ Size: 3.1 MB
│  ├─ Duration: 3:44
│  ├─ Bitrate: 192 kbps
│  ├─ Codec: MP3
│  ├─ Metadata: Artist / Title / Album (extracted)
│  └─ Fingerprint: [show visual]
│
├─ Comparison:
│  ├─ Fingerprint distance: 0.045 (very close)
│  ├─ Duration diff: +1 second (likely same song)
│  ├─ Status: COLLISION (similar, not exact)
│  └─ Auto-decision: SKIP (keep existing)
│
├─ User Action:
│  ├─ Flag: 📋 Copy
│  └─ Note: "Might be remix, check"
│
└─ Plan Impact:
   ├─ With flag, will: COPY incoming file
   ├─ Resulting files: BOTH incoming and existing in library
   ├─ Size impact: +4.2 MB
   └─ Reversible: Yes (can delete incoming copy)
```

---

## Stage 4: PLAN BUILDING

### **What Happens**

System takes MatchResults + ReviewStateStore and decides exactly what to do with each file.

### **Algorithm**

```
For each MatchResult:

  1. Check ReviewStateStore for flags
     ├─ If flagged for COPY:
     │  └─ Decision: COPY to library
     │
     ├─ If flagged for REPLACE:
     │  └─ Decision: REPLACE existing file
     │
     └─ If not flagged (auto-decision):
        ├─ If status == NEW:
        │  └─ Decision: COPY
        │
        ├─ If status == MATCHED (exact duplicate):
        │  └─ Decision: SKIP
        │
        ├─ If status == COLLISION (very similar):
        │  └─ Decision: SKIP (keep existing)
        │
        └─ If status == UPGRADE (possible better):
           └─ Decision: SKIP (user must flag to use it)

  2. For COPY decisions:
     ├─ Determine destination path:
     │  ├─ Artist folder structure? (if exists)
     │  ├─ Naming convention? (if set)
     │  └─ Default: /library/[filename from incoming]
     └─ Create CopyOperation:
        {
          "source": "/path/to/incoming/file.mp3",
          "destination": "/path/to/library/file.mp3",
          "decision": "COPY",
          "reason": "New file" or "User flag: Copy",
          "note": "user note or empty"
        }

  3. For REPLACE decisions:
     ├─ Validate: does existing match exist?
     ├─ YES:
     │  └─ Create ReplaceOperation:
     │     {
     │       "source": "/path/to/incoming/file.mp3",
     │       "destination": "/path/to/existing/file.mp3",
     │       "decision": "REPLACE",
     │       "reason": "User flag: Replace",
     │       "note": "Better quality" or similar
     │     }
     └─ NO: Log warning, treat as COPY instead

  4. For SKIP decisions:
     └─ CreateSkipOperation:
        {
          "incoming": "/path/to/incoming/file.mp3",
          "existing": "/path/to/existing/file.mp3" (or null),
          "decision": "SKIP",
          "reason": "Exact duplicate" or "Keep existing",
          "note": ""
        }

Result: ExecutionPlan = {
  "copy_operations": [...],
  "replace_operations": [...],
  "skip_operations": [...],
  "summary": {
    "total_incoming": 200,
    "to_copy": 85,
    "to_replace": 12,
    "to_skip": 103
  }
}
```

### **UI During Build**

```
User clicks: [Build Plan]

Dialog appears:
┌──────────────────────────────────────────────────┐
│  Building plan...                                │
│  ████████████░░░░░░░░░░░░░░░░░░  167 / 200      │
│  Estimated time: 15 seconds                      │
│  [Cancel]                                        │
└──────────────────────────────────────────────────┘

When complete:
┌──────────────────────────────────────────────────┐
│  ✓ Plan built successfully                       │
│                                                  │
│  Summary:                                        │
│  - Copy: 85 files (350 MB)                       │
│  - Replace: 12 files (47 MB)                     │
│  - Skip: 103 files (0 MB)                        │
│                                                  │
│  Preview will be saved to:                       │
│  /Library/Docs/LibrarySyncPreview.html           │
│                                                  │
│  [Preview in Browser] [Edit Flags] [Execute]    │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Stage 5: PREVIEW

### **What Happens**

System generates HTML report and user reviews before execution.

### **HTML Preview Structure**

```html
LibrarySyncPreview.html
├─ Header
│  ├─ Title: "Library Sync Preview"
│  ├─ Generated: 2026-03-20 15:34:22
│  ├─ Existing Library: /mnt/music/
│  ├─ Incoming Folder: /mnt/usb/NewMusic/
│  └─ Threshold: 0.3
│
├─ Summary Section
│  ├─ Total incoming files: 200
│  ├─ Copy: 85 (350 MB) [expandable list]
│  ├─ Replace: 12 (47 MB) [expandable list]
│  ├─ Skip: 103 (0 MB) [expandable list]
│  └─ Overall library growth: +397 MB
│
├─ Copy Operations
│  ├─ Heading: "85 Files to Copy"
│  └─ Table:
│     ┌──────────────┬─────────────┬──────────┬──────────────┐
│     │ Source       │ Destination │ Size     │ Reason       │
│     ├──────────────┼─────────────┼──────────┼──────────────┤
│     │ song1.mp3    │ song1.mp3   │ 4.2 MB   │ New file     │
│     │ song2.mp3    │ song2.mp3   │ 3.8 MB   │ User flag    │
│     │ ...          │ ...         │ ...      │ ...          │
│     └──────────────┴─────────────┴──────────┴──────────────┘
│
├─ Replace Operations
│  ├─ Heading: "12 Files to Replace"
│  └─ Table:
│     ┌──────────────┬────────────┬─────────┬─────────────┐
│     │ Source       │ Existing   │ Old→New │ Reason      │
│     ├──────────────┼────────────┼─────────┼─────────────┤
│     │ upgrade.flac │ song_3.mp3 │ 3→5 MB  │ User flag   │
│     │ ...          │ ...        │ ...     │ ...         │
│     └──────────────┴────────────┴─────────┴─────────────┘
│
├─ Skip Operations
│  ├─ Heading: "103 Files to Skip"
│  └─ Collapsible table (defaults to collapsed)
│     ┌──────────────┬────────────┬─────────────────────┐
│     │ Incoming     │ Existing   │ Reason              │
│     ├──────────────┼────────────┼─────────────────────┤
│     │ song4.mp3    │ song4.mp3  │ Exact duplicate     │
│     │ ...          │ ...        │ ...                 │
│     └──────────────┴────────────┴─────────────────────┘
│
└─ Footer
   ├─ Last updated: [timestamp]
   ├─ Status: Ready to execute
   └─ Note: "Review carefully before executing!"
```

### **User Review Checklist**

When viewing HTML preview, user should check:

```
□ Copy count looks reasonable (not unexpectedly high)
□ Replace operations have notes explaining why
□ No critical files in "Skip" that should be copied
□ No old files being replaced that shouldn't be
□ Estimated size increase is acceptable
□ All flagged items show up with "User flag" reason
□ No duplicate operations
```

### **Going Back to Edit**

If user finds an issue, they can:

```
Option 1: Click [Edit Flags] in preview dialog
  → Returns to Step 3 (Review stage)
  → Flags are preserved
  → User modifies flags
  → Click [Build Plan] again
  → New preview generated

Option 2: Close preview, manually adjust in table
  → Return to incoming tracks table
  → Right-click to modify flags
  → Click [Build Plan] again
```

---

## Stage 6: EXECUTION

### **What Happens**

System performs all planned operations: copying and replacing files.

### **Execution Process**

```
User clicks: [Execute Plan]

Dialog appears:
┌─────────────────────────────────────────────┐
│ ⚠ Confirm Execution                         │
├─────────────────────────────────────────────┤
│                                             │
│ This will:                                  │
│ • Copy 85 files (350 MB)                    │
│ • Replace 12 existing files                 │
│ • Skip 103 files                            │
│                                             │
│ This action cannot be undone (but files     │
│ are not deleted, just moved/copied)         │
│                                             │
│ Continue?                                   │
│ [No, go back] [Yes, execute]               │
│                                             │
└─────────────────────────────────────────────┘
```

### **During Execution**

```
┌─────────────────────────────────────────────┐
│ EXECUTING SYNC PLAN...                      │
├─────────────────────────────────────────────┤
│                                             │
│ Overall progress:                           │
│ █████████████░░░░░░░░░░░░░░░░░  97 / 200  │
│                                             │
│ Current operation:                          │
│ Copying: "song_45.flac" (8.2 MB)            │
│ ████████████░░░░░░░░░░░░░  45%             │
│ Speed: 15 MB/s                              │
│ Estimated time remaining: 4 minutes         │
│                                             │
│ Completed so far:                           │
│ • Copied: 45 files (189 MB)                 │
│ • Replaced: 7 files                         │
│ • Skipped: 45 files                         │
│                                             │
│ [Cancel execution]  (may leave partial)    │
│                                             │
└─────────────────────────────────────────────┘
```

### **Core Operations**

#### **COPY Operation**
```python
def execute_copy(source_path, dest_path):
  """Copy source file to destination in library"""

  # Validation
  if not os.path.exists(source_path):
    log("ERROR", f"Source not found: {source_path}")
    return False

  if not os.path.exists(os.path.dirname(dest_path)):
    log("INFO", f"Creating destination folder: {os.path.dirname(dest_path)}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

  # If destination already exists, this is a conflict
  if os.path.exists(dest_path):
    log("WARNING", f"Destination exists, treating as REPLACE: {dest_path}")
    return execute_replace(source_path, dest_path)

  # Copy file
  try:
    shutil.copy2(source_path, dest_path)  # preserves metadata
    log("COPY", f"{source_path} → {dest_path}")
    return True
  except Exception as e:
    log("ERROR", f"Copy failed: {e}")
    return False
```

#### **REPLACE Operation**
```python
def execute_replace(source_path, dest_path):
  """Replace existing file with source"""

  # Validation
  if not os.path.exists(source_path):
    log("ERROR", f"Source not found: {source_path}")
    return False

  if not os.path.exists(dest_path):
    log("WARNING", f"Destination not found, treating as COPY: {dest_path}")
    return execute_copy(source_path, dest_path)

  # Back up existing file (safety)
  backup_path = dest_path + ".backup"
  try:
    shutil.copy2(dest_path, backup_path)
    log("BACKUP", f"Backed up to: {backup_path}")
  except Exception as e:
    log("WARNING", f"Could not create backup: {e}")

  # Replace file
  try:
    os.remove(dest_path)
    shutil.copy2(source_path, dest_path)
    log("REPLACE", f"{source_path} → {dest_path} (backup: {backup_path})")
    # Note: backup is kept for safety, can be deleted later
    return True
  except Exception as e:
    log("ERROR", f"Replace failed: {e}")
    # Try to restore backup
    if os.path.exists(backup_path):
      shutil.copy2(backup_path, dest_path)
      log("RESTORE", f"Restored backup to: {dest_path}")
    return False
```

#### **SKIP Operation**
```python
def execute_skip(incoming_path, existing_path):
  """Do nothing - skip this file"""
  log("SKIP", f"Skipped: {incoming_path}")
  return True
```

### **Error Handling**

```
Errors encountered:
│
├─ Source file disappeared
│  └─ Action: Skip this file, log error, continue
│
├─ Destination disk full
│  └─ Action: Pause, show error, wait for user to free space, resume
│
├─ Permission denied on destination folder
│  └─ Action: Pause, show error, user must fix permissions, resume
│
├─ File locked (in use by another process)
│  └─ Action: Retry 3 times with 2-second delays, then skip
│
├─ Corrupted source file
│  └─ Action: Try to copy anyway, log warning if it fails
│
└─ Path name conflict (too long, invalid characters)
   └─ Action: Sanitize filename, log action, proceed
```

### **Cancellation**

```
User clicks [Cancel execution]:

If currently copying a file:
  └─ Finish current copy (don't interrupt mid-file)
  └─ Stop next operations
  └─ Report what completed successfully
  └─ Suggest user can re-run to finish remaining files

Result:
  ├─ Some files copied (not rolled back)
  ├─ Some files not yet processed
  ├─ No files deleted (safe to resume)
  └─ Log shows exactly what completed
```

---

## Stage 7: REPORT

### **What Happens**

System generates execution report after all operations complete.

### **Execution Report HTML**

```html
LibrarySyncExecutionReport.html
├─ Header
│  ├─ Title: "Library Sync Execution Report"
│  ├─ Execution started: 2026-03-20 15:45:00
│  ├─ Execution completed: 2026-03-20 16:23:00 (38 minutes)
│  ├─ Status: ✓ SUCCESS / ⚠ PARTIAL / ✗ FAILED
│  └─ Operator: (username or "Unknown")
│
├─ Summary Section
│  ├─ Planned: 200 files
│  ├─ Successfully processed: 198 files
│  ├─ Failed: 2 files
│  ├─ Files copied: 85 (350 MB)
│  ├─ Files replaced: 12 (now using better quality)
│  ├─ Files skipped: 101
│  └─ Total time: 38 minutes
│
├─ Successful Operations (grouped)
│  ├─ COPY - 85 files
│  │  └─ Table:
│  │     ┌──────────────┬─────────────┬──────────┐
│  │     │ Source       │ Destination │ Size     │
│  │     ├──────────────┼─────────────┼──────────┤
│  │     │ song1.mp3    │ song1.mp3   │ 4.2 MB   │
│  │     │ ...          │ ...         │ ...      │
│  │     └──────────────┴─────────────┴──────────┘
│  │
│  └─ REPLACE - 12 files
│     └─ Table:
│        ┌──────────────┬────────────┬────────────────┐
│        │ Source       │ Existing   │ Status         │
│        ├──────────────┼────────────┼────────────────┤
│        │ upgrade.flac │ song_3.mp3 │ Replaced (bak) │
│        │ ...          │ ...        │ ...            │
│        └──────────────┴────────────┴────────────────┘
│
├─ Failed Operations (if any)
│  ├─ Heading: "2 Files Failed"
│  ├─ Table:
│  │  ┌──────────────┬────────────┬──────────────────┐
│  │  │ Source       │ Destination│ Error            │
│  │  ├──────────────┼────────────┼──────────────────┤
│  │  │ corrupted.mp3│ song_99.mp3│ Read error       │
│  │  │ locked.flac  │ song_100.fl│ File in use      │
│  │  └──────────────┴────────────┴──────────────────┘
│  │
│  └─ Notes:
│     "These files were not processed. You can:
│      1. Fix the source files and re-run sync
│      2. Manually copy them
│      3. Skip them for now"
│
├─ Backup Information
│  ├─ Heading: "Backups Created (for replaced files)"
│  └─ Location: /Library/Backups/LibrarySyncBackups/
│     ├─ song_3.mp3.backup (created 2026-03-20 16:15)
│     ├─ song_4.m4a.backup (created 2026-03-20 16:18)
│     └─ ... 10 more backups
│     │
│     └─ Note: "Backups are kept for 30 days.
│        Delete the LibrarySyncBackups folder to reclaim space."
│
└─ Footer
   ├─ Next steps:
   │  "1. Review the new files in your library
   │   2. Listen to replaced files to confirm quality
   │   3. Clean up the incoming folder if no longer needed"
   │
   └─ Support:
      "If any files are missing or corrupted, refer to
       /Library/Backups/LibrarySyncBackups/ for recovery."
```

### **Status Dialog**

```
After execution completes, show:

┌──────────────────────────────────────────┐
│ ✓ Sync Complete                          │
├──────────────────────────────────────────┤
│                                          │
│ Copied:   85 files (350 MB)              │
│ Replaced: 12 files                       │
│ Skipped:  103 files                      │
│ Failed:   0 files                        │
│                                          │
│ Time: 38 minutes                         │
│ Library size: +397 MB                    │
│                                          │
│ Execution report: /Library/Docs/         │
│ LibrarySyncExecutionReport.html           │
│                                          │
│ Backups: /Library/Backups/                │
│ LibrarySyncBackups/ (30 days retention)   │
│                                          │
│ [View Report] [View Backups] [Done]      │
│                                          │
└──────────────────────────────────────────┘
```

---

## How Per-Item Flags Integrate

The flags from Stage 3 (Review) affect Stage 4 (Plan Building):

```
ReviewStateStore flags:
{
  "copy": ["track_001", "track_003"],
  "replace": {"track_002": "track_existing_002"}
}
        ↓
resolve_review_flags_to_paths() converts to:
{
  "copy_only_paths": ["/path/to/track_001.mp3"],
  "allowed_replacement_paths": ["/path/to/track_existing_002.mp3"]
}
        ↓
compute_library_sync_plan() receives as parameters:
        ↓
Plan builder treats flagged items specially:
├─ copy_only_paths items → forced to COPY (even if would auto-skip)
├─ allowed_replacement_paths → allowed to REPLACE (if flagged)
└─ All other items → auto-decided
        ↓
ExecutionPlan includes:
├─ CopyOperation with reason "User flag: Copy"
├─ ReplaceOperation with reason "User flag: Replace"
└─ SkipOperation (unchanged)
```

---

## Key Design Principles

### **1. Preview-First, Never Destructive**
✓ Every operation shown in preview before execution
✓ User must explicitly click [Execute]
✓ No auto-execution or background syncing
✓ All changes reversible (backups kept, no deletions)

### **2. User Control Over AI Decisions**
✓ System provides smart defaults (fingerprinting)
✓ User can override on any single file
✓ Flags make decisions explicit and reviewable
✓ No silent sorting or background processing

### **3. No Data Loss**
✓ Incoming files never deleted (copied, not moved)
✓ Existing files backed up before replace
✓ SKIP = do nothing (safest choice)
✓ Execution log shows exactly what happened

### **4. Clear Feedback Loop**
✓ Progress bars during long operations
✓ Detailed HTML previews before execution
✓ Execution reports after completion
✓ All errors logged and visible

### **5. Modularity**
✓ Scan decoupled from plan building
✓ Plan building decoupled from execution
✓ Can rebuild plan multiple times before executing
✓ Can re-run on same folders (idempotent)

---

## Configuration & Settings

User can configure:

```
Thresholds:
├─ Default fingerprint threshold (0.0 - 1.0)
├─ Preset buttons (Exact / Close / Loose)
└─ Per-session override

Execution Behavior:
├─ Backup old files before replace? (YES/NO)
├─ Create execution report? (YES/NO, default YES)
├─ Keep backups for (1-90 days, default 30)
└─ Log level (INFO / WARNING / ERROR)

UI Preferences:
├─ Auto-expand detailed view for flagged items? (YES/NO)
├─ Show fingerprint distance units? (distance / percentage)
├─ Default sort column in table? (Track / Status / Distance)
└─ Remember last paths? (YES/NO)

Folder Structure:
├─ Preserve artist folder structure? (YES/NO)
├─ Naming convention for copied files? (filename / artist_title / auto)
└─ Skip certain folder patterns? (Not Sorted, Trash, etc.)
```

---

## Workflow Comparison

### **Without Library Sync**
```
User approach:
1. Mount USB drive manually
2. Open two file browsers (existing lib + incoming)
3. Listen to samples of potential duplicates
4. Manually move/copy files one by one
5. Deal with name conflicts manually
6. Hope nothing gets deleted by mistake
7. No record of what was done

Time: 2-4 hours for 500 files
Risk: HIGH (manual operations)
Reversibility: LOW (hope you didn't delete by mistake)
Confidence: LOW (did I miss anything?)
```

### **With Library Sync**
```
System workflow:
1. Select two folders (existing, incoming)
2. Click "Scan" (automatic, 5-10 minutes)
3. See table of all files (NEW/MATCHED/COLLISION/UPGRADE)
4. Optionally flag decisions (5-10 minutes for complex decisions)
5. Click "Build Plan" (automatic, < 1 minute)
6. Review preview (5 minutes to check)
7. Click "Execute" (automatic, 10-30 minutes)
8. Full execution report generated

Time: 30-60 minutes total (including review)
Risk: LOW (preview before execution, backups)
Reversibility: HIGH (backups available)
Confidence: HIGH (see everything before it happens)
```

---

## Success Metrics

After full implementation, verify:

- ✓ Scan fingerprints 1000 files in < 10 minutes
- ✓ User can flag 100 tracks in < 10 minutes
- ✓ Plan building for 500 files takes < 30 seconds
- ✓ HTML preview renders in browser in < 2 seconds
- ✓ Execution copies 100 files in < 5 minutes
- ✓ Execution report generated in < 10 seconds
- ✓ Zero data loss (no files deleted)
- ✓ Backups created for all replaced files
- ✓ All operations logged with timestamps
- ✓ User can resume after cancellation

---

## File Structure Impact

After Library Sync execution:

```
Before:
/mnt/music/
├─ 500 files
├─ ~2 GB
└─ Already organized

After:
/mnt/music/
├─ 500 + 85 = 585 files
├─ ~2 GB + 350 MB = ~2.35 GB
├─ Some files replaced with better versions
└─ Organization preserved

/Library/Docs/
├─ LibrarySyncPreview.html (kept)
├─ LibrarySyncExecutionReport.html (new)
└─ LibrarySyncBackups/ (new if replacements made)
   ├─ song_3.mp3.backup
   ├─ song_4.m4a.backup
   └─ ... (12 backups total)
```

---

## You Now Fully Understand Library Sync

✅ **Purpose:** Safely merge incoming music into existing library
✅ **Workflow:** 7 stages from input to execution
✅ **Preview-first:** Every operation shown before execution
✅ **User control:** Flags let you override auto-decisions
✅ **Safety:** No deletions, backups for replacements
✅ **Feedback:** Detailed progress and reports throughout
✅ **Reversibility:** Can recover from any operation
✅ **Performance:** Designed to handle 1000+ files
✅ **Modularity:** Each stage independent, can re-run stages
✅ **Data integrity:** No loss, no corruption, no silent failures

**Ready to build!** 🚀

