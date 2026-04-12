# Library Sync Features

## Per-Item Review Flags (NEW)

**Status:** ✅ Implemented and ready for testing
**Documentation:** See `library_sync_per_item_review.md` and `library_sync_per_item_review_testing.md`

### What It Does

During Library Sync, users can now override automatic file disposition decisions on a per-item basis:

- **Flag for Copy**: Force an incoming file to be copied regardless of whether a match exists
- **Flag for Replace**: Force replacement of an existing library file with an incoming upgrade
- **Add Notes**: Attach freeform notes to explain flagging decisions

### User Workflow

1. **Scan** — Select existing library and incoming folder, run scan
2. **Review** — Browse scan results in the incoming tracks table
3. **Flag** — Right-click incoming tracks to set flags and add notes
4. **Build Plan** — Flags automatically override auto-decisions during plan building
5. **Execute** — Files are copied/replaced according to flagged decisions

### Implementation

**Backend** (`library_sync.py`):
- `compute_library_sync_plan()` accepts `copy_only_paths` and `allowed_replacement_paths`
- `build_library_sync_preview()` passes overrides to the plan builder
- `resolve_review_flags_to_paths()` converts track IDs to file paths

**Frontend** (`gui/workspaces/library_sync.py`):
- ReviewStateStore initialized in LibrarySyncWorkspace
- Incoming tracks table has Flag and Note columns
- Right-click context menu with flagging options
- Visual indicators (emojis) for flag status

**Data Flow**:
```
User Right-Click
    ↓
ReviewStateStore (store track_id → flag mapping)
    ↓
Scan Result Stored (MatchResult objects with track_ids)
    ↓
Build Plan Called
    ↓
resolve_review_flags_to_paths() (convert track_ids → paths)
    ↓
compute_library_sync_plan() (apply path overrides)
    ↓
Plan items respect user flags
```

### Files Modified

- `library_sync.py` — Backend integration
- `gui/workspaces/library_sync.py` — Frontend UI
- `CLAUDE.md` — Updated Known Gaps section
- `docs/library_sync_per_item_review.md` — Implementation plan
- `docs/library_sync_per_item_review_testing.md` — Testing guide

### Testing

Full integration testing requires running the application with:
1. Prepared test libraries (existing + incoming folders)
2. All dependencies installed
3. Full GUI interaction

See `library_sync_per_item_review_testing.md` for detailed manual test scenarios.

### Known Limitations

- **Session persistence**: Flags clear when workspace closes (in-memory only)
- **Threshold changes**: Recomputing with different thresholds clears old flags
- **No undo/redo**: Individual flag changes cannot be undone
- **Match object requirement**: Flags are silently ignored if match objects unavailable

### Future Enhancements

- Persistent flag sessions (save/load from JSON)
- Flag history/undo
- Bulk flag operations (flag all "Upgrade" matches at once)
- Integration with config auto-save

