# Library Sync Per-Item Review Implementation Plan

**Session ID:** `claude/audit-startup-splash-PwNwu`
**Created:** 2026-03-20
**Status:** Planning Phase

---

## Vision

Enable users to manually override automatic file disposition decisions during Library Sync. Currently, the plan builder uses fingerprint matching to auto-decide whether each incoming file should be COPIED, REPLACED, or SKIPPED. Users should be able to:

- Flag incoming files for copy regardless of auto-decision
- Flag incoming files to replace their existing counterpart
- Attach notes to explain their reasoning
- See visual indicators of flags in the UI
- Have those flags respected when the plan is built and executed

This closes the gap between **infrastructure** (ReviewStateStore class exists and is complete) and **integration** (nothing uses it yet).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│ GUI: gui/workspaces/library_sync.py                     │
│ - Shows incoming/existing tracks                        │
│ - Right-click context menu for flagging                 │
│ - Visual indicators of user selections                  │
└────────────────────────┬────────────────────────────────┘
                         │ ReviewStateStore
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Backend: library_sync_review_state.py                   │
│ - In-memory storage: copy set, replace dict, notes      │
│ - Already complete, no changes needed                   │
└────────────────────────┬────────────────────────────────┘
                         │ review_flags param
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Plan Builder: library_sync.py                           │
│ - compute_library_sync_plan()                           │
│ - build_library_sync_preview()                          │
│ - _compute_plan_items() [auto-decision logic]           │
│                                                         │
│ NEW: Check user flags before auto-deciding              │
└─────────────────────────────────────────────────────────┘
```

---

## Task Checklist

### Phase 1: Backend Integration (30 min)
- [ ] **1.1** Modify `library_sync.py` — add `review_flags` parameter
  - [ ] 1.1.1 Add to `compute_library_sync_plan()` signature
  - [ ] 1.1.2 Add to `build_library_sync_preview()` signature
  - [ ] 1.1.3 Modify `_compute_plan_items()` to check flags before auto-logic
  - [ ] 1.1.4 Wire parameter through all call chains
- [ ] **1.2** Add import for ReviewStateStore in `library_sync.py`
- [ ] **1.3** Unit test: flag item, verify plan respects it

### Phase 2: GUI Frontend (60 min)
- [ ] **2.1** Initialize ReviewStateStore in `LibrarySyncWorkspace.__init__()`
- [ ] **2.2** Enhance incoming table display
  - [ ] 2.2.1 Add "Flag" and "Note" columns to header
  - [ ] 2.2.2 Store track_id in tree items (Qt.UserRole)
- [ ] **2.3** Implement context menu for incoming tracks
  - [ ] 2.3.1 Right-click handler: `_on_incoming_context_menu()`
  - [ ] 2.3.2 Menu actions: Copy, Replace, Clear flag, Add note
- [ ] **2.4** Implement flag actions
  - [ ] 2.4.1 `_flag_for_copy(track_id)` — store in ReviewStateStore
  - [ ] 2.4.2 `_flag_for_replace(track_id, existing_id)` — validate match exists
  - [ ] 2.4.3 `_flag_clear(track_id)` — clear both copy and replace
  - [ ] 2.4.4 `_edit_note(track_id)` — modal dialog for note entry
- [ ] **2.5** Update table display on flag changes
  - [ ] 2.5.1 `_update_flag_display(item, track_id)` — show emoji + status
  - [ ] 2.5.2 `_update_note_display(item, track_id)` — show note or "…"
- [ ] **2.6** Handle scan result events
  - [ ] 2.6.1 Store match results after scan completes
  - [ ] 2.6.2 Call `reconcile_best_matches()` on results change

### Phase 3: Worker Integration (25 min)
- [ ] **3.1** Modify `SyncBuildWorker` to accept and pass review_flags
  - [ ] 3.1.1 Add `review_flags` to `__init__()`
  - [ ] 3.1.2 Pass to `build_library_sync_preview()` call in `run()`
- [ ] **3.2** Modify `_on_build_plan()` to pass ReviewStateStore to worker

### Phase 4: Testing & Polish (30 min)
- [ ] **4.1** Manual test: flag item for copy, verify plan includes it
- [ ] **4.2** Manual test: flag item to replace, verify plan executes replace
- [ ] **4.3** Manual test: reconciliation after re-scanning (best match changed)
- [ ] **4.4** Edge case: clear flags before rebuild
- [ ] **4.5** Verify HTML preview reflects final plan (with flags applied)

### Phase 5: Documentation (15 min)
- [ ] **5.1** Update this plan document with final decisions
- [ ] **5.2** Add docstrings to new UI methods
- [ ] **5.3** Update `docs/gui_inventory.md` if control layout changed

---

## Detailed Implementation Notes

### 1.1 Backend: ReviewFlags Integration

**File:** `library_sync.py`

**Current state:**
- `compute_library_sync_plan()` takes no user overrides
- `_compute_plan_items()` auto-decides: NEW → COPY, MATCH (lower quality) → SKIP, MATCH (higher quality) → REPLACE

**Changes:**

```python
# Line 893: Update signature
def compute_library_sync_plan(
    library_root: str,
    incoming_folder: str,
    *,
    db_path: str | None = None,
    log_callback: Callable | None = None,
    progress_callback: Callable | None = None,
    cancel_event: threading.Event | None = None,
    review_flags: "ReviewStateStore | None" = None,  # ← ADD THIS
    transfer_mode: str = "move",
) -> LibrarySyncPlan:
```

```python
# Line 845: In _compute_plan_items(), add flag check BEFORE auto-decision
def _compute_plan_items(results: list[MatchResult], review_flags=None) -> list[PlanItem]:
    items = []
    for result in results:
        incoming_id = result.incoming.track_id

        # USER OVERRIDE: Check review flags first
        if review_flags:
            if review_flags.is_copy_flagged(incoming_id):
                items.append(PlanItem(action=COPY, incoming=result.incoming, existing=None, reason="User flag"))
                continue
            elif replace_target := review_flags.replace_target(incoming_id):
                if result.existing and result.existing.track_id == replace_target:
                    items.append(PlanItem(action=REPLACE, incoming=result.incoming, existing=result.existing, reason="User flag"))
                    continue

        # AUTO-DECISION: Original logic (if no flag override)
        # ... existing code ...
```

**Import needed:**
```python
from library_sync_review_state import ReviewStateStore  # At top of file
```

---

### 2.1-2.6 Frontend: UI Integration

**File:** `gui/workspaces/library_sync.py`

**Current state:**
- ReviewStateStore not instantiated
- Incoming table has 3 columns: Track, Status, Distance
- No right-click menu
- No mechanism to store track IDs in UI items

**Changes:**

```python
# Line 167: In __init__()
def __init__(self, library_path: str = "", parent: QtWidgets.QWidget | None = None) -> None:
    super().__init__(library_path, parent)
    self._scan_result = None
    self._worker: SyncScanWorker | None = None
    self._build_worker: SyncBuildWorker | None = None
    self._exec_worker: SyncExecuteWorker | None = None

    # NEW: Initialize review state store
    from library_sync_review_state import ReviewStateStore
    self._review_state_store = ReviewStateStore()
    self._current_match_results = []  # Store results after scan

    self._build_ui()
```

```python
# Line 344: Update incoming table header
self._incoming_table.setHeaderLabels(["Track", "Status", "Distance", "Flag", "Note"])
self._incoming_table.setColumnWidth(0, 200)
self._incoming_table.setColumnWidth(1, 80)
self._incoming_table.setColumnWidth(2, 70)
self._incoming_table.setColumnWidth(3, 60)
self._incoming_table.setColumnWidth(4, 100)

# Enable context menu
self._incoming_table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
self._incoming_table.customContextMenuRequested.connect(self._on_incoming_context_menu)
```

```python
# NEW METHOD: Context menu handler
def _on_incoming_context_menu(self, pos: QtCore.QPoint) -> None:
    """Right-click menu for incoming track."""
    item = self._incoming_table.itemAt(pos)
    if not item:
        return

    track_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
    if not track_id:
        return

    menu = QtWidgets.QMenu()
    menu.addAction("📋 Copy", lambda: self._flag_for_copy(track_id))
    menu.addAction("↻ Replace", lambda: self._flag_for_replace(track_id))
    menu.addAction("✕ Clear flag", lambda: self._flag_clear(track_id))
    menu.addSeparator()
    menu.addAction("📝 Add note", lambda: self._edit_note(track_id))
    menu.exec(self._incoming_table.mapToGlobal(pos))
```

```python
# NEW METHODS: Flag actions
def _flag_for_copy(self, track_id: str) -> None:
    """Mark track for copy, remove any replace flag."""
    self._review_state_store.unflag_copy(track_id)  # Clear replace if present
    self._review_state_store.flag_for_copy(track_id)
    self._update_incoming_item_flags(track_id)

def _flag_for_replace(self, track_id: str) -> None:
    """Mark track to replace existing counterpart."""
    # Find the matching result for validation
    match = next((r for r in self._current_match_results
                  if r.incoming.track_id == track_id), None)
    if not match or not match.existing:
        QtWidgets.QMessageBox.warning(self, "Cannot Replace",
            "No existing match found for this track.")
        return

    self._review_state_store.unflag_copy(track_id)  # Can't do both
    self._review_state_store.flag_for_replace(match)
    self._update_incoming_item_flags(track_id)

def _flag_clear(self, track_id: str) -> None:
    """Clear both copy and replace flags."""
    self._review_state_store.unflag_copy(track_id)
    # No method to clear replace, so we'd need to add one
    # OR: clear by not having any flag set
    # TODO: Add clear_replace() method to ReviewStateStore
    self._update_incoming_item_flags(track_id)

def _edit_note(self, track_id: str) -> None:
    """Show dialog to edit note for this track."""
    current_note = self._review_state_store.note_for(track_id) or ""
    text, ok = QtWidgets.QInputDialog.getMultiLineText(
        self, "Add Note", "Notes for this track:", current_note
    )
    if ok:
        self._review_state_store.set_note(track_id, text)
        self._update_incoming_item_flags(track_id)

def _update_incoming_item_flags(self, track_id: str) -> None:
    """Refresh the display for a single incoming item."""
    # Find item by track_id
    for i in range(self._incoming_table.topLevelItemCount()):
        item = self._incoming_table.topLevelItem(i)
        if item.data(0, QtCore.Qt.ItemDataRole.UserRole) == track_id:
            # Update flag column (index 3)
            if self._review_state_store.is_copy_flagged(track_id):
                item.setText(3, "📋 Copy")
            elif self._review_state_store.replace_target(track_id):
                item.setText(3, "↻ Replace")
            else:
                item.setText(3, "")

            # Update note column (index 4)
            note = self._review_state_store.note_for(track_id)
            item.setText(4, note if note else "")
            break
```

```python
# MODIFY: In method that populates incoming table, store track_id
# (Find _populate_results() or similar and update it)
for result in results:
    item = QtWidgets.QTreeWidgetItem([
        result.incoming.path,
        str(result.status),
        f"{result.distance:.3f}" if result.distance else "—",
        "",  # Flag (empty initially)
        "",  # Note (empty initially)
    ])
    # Store track_id for context menu lookup
    item.setData(0, QtCore.Qt.ItemDataRole.UserRole, result.incoming.track_id)
    self._incoming_table.addTopLevelItem(item)
```

---

### 3.1-3.2 Worker: Pass Flags to Plan Builder

**File:** `gui/workspaces/library_sync.py`

**Changes to `SyncBuildWorker`:**

```python
# Line 70: Update __init__
def __init__(self, library_root: str, incoming_folder: str, transfer_mode: str,
             review_flags=None) -> None:  # ← ADD review_flags
    super().__init__()
    self.library_root = library_root
    self.incoming_folder = incoming_folder
    self.transfer_mode = transfer_mode
    self.review_flags = review_flags  # ← STORE IT
    self._cancel_event = threading.Event()
    self.plan = None
    self.preview_html_path: str = ""
```

```python
# Line 102: In run(), pass to build_library_sync_preview()
plan = library_sync.build_library_sync_preview(
    self.library_root,
    self.incoming_folder,
    output_html,
    log_callback=_log,
    progress_callback=_progress,
    cancel_event=self._cancel_event,
    transfer_mode=self.transfer_mode,
    review_flags=self.review_flags,  # ← PASS IT
)
```

**Changes to `_on_build_plan()` method:**

```python
# Around line 524 (find where SyncBuildWorker is instantiated)
self._build_worker = SyncBuildWorker(
    self.library_root,
    self.incoming_folder,
    self._transfer_toggle.text(),
    review_flags=self._review_state_store,  # ← ADD THIS
)
```

---

### 4.1-4.5 Testing Strategy

**Manual tests to run before marking complete:**

1. **Copy flag**: Flag incoming track for copy → run plan builder → verify plan shows COPY action
2. **Replace flag**: Flag incoming track to replace existing → verify plan shows REPLACE action
3. **Clear flag**: Flag item → right-click Clear → verify flag visually disappears
4. **Notes persist**: Add note → close/reopen workspace → note still there (if session persists)
5. **Reconciliation**: Change threshold → re-scan → verify replace flags rebind to new best match
6. **Plan HTML**: After plan build, open preview HTML and verify flags influenced final copy/move/skip decisions

**No unit tests needed yet** (ReviewStateStore already tested; this is GUI integration testing)

---

## Flexible Decisions (May Change)

| Decision | Current Plan | Notes |
|----------|--------------|-------|
| Flag indicators | Emoji + text | Could use checkboxes, colors, or icons instead |
| Note UI | Modal dialog | Could use inline edit, sidebar, or separate panel |
| Replace validation | Lookup current match | Could disable Replace if no match exists |
| Preview refresh | Manual rebuild | Could auto-rebuild on flag changes (advanced) |
| Session persistence | In-memory only | Could save/load flags to JSON file |
| Undo/redo | Not implemented | Could add if needed later |

---

## Known Gaps / Assumptions

1. **ReviewStateStore.clear_replace()** may need to be added if no method exists to clear replace flags independently
2. **Track ID stability**: Assumes track_id doesn't change between scans (verify in code)
3. **Qt.UserRole**: May need adjustment if other code uses Qt.UserRole on these items
4. **match.existing validation**: When replacing, need to ensure existing_id is valid
5. **Best match reconciliation**: Need to call `reconcile_best_matches()` after re-scan with threshold change

---

## Success Criteria

✅ User can right-click incoming track and choose COPY or REPLACE
✅ Flags are visually indicated in table (emoji in "Flag" column)
✅ Notes are visible and editable (column shows truncated text)
✅ Plan builder respects flags: flagged-for-copy items are always copied
✅ Plan builder respects flags: flagged-for-replace items always replace
✅ Reconciliation works: if best match changes, replace flag rebinds or clears
✅ HTML preview reflects final plan (with flags applied)
✅ All existing tests still pass

---

## Revision History

| Date | Change | By |
|------|--------|-----|
| 2026-03-20 | Created initial plan | Claude |
| (pending) | (section updates as work progresses) | (user/claude) |

