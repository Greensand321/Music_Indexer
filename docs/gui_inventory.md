# AlphaDEX GUI Inventory

A plain-English reference for every screen, panel, control, and dialog in the
current desktop application (`main_gui.py` + `library_sync_review.py`). Use
this as a requirements baseline when building a replacement frontend.

---

## 1. Application Window

The root window (`SoundVaultImporterApp`) is a single resizable Tkinter window.
It contains three fixed sections stacked vertically before the tab area:

### 1.1 Top Ribbon (always visible)

| Element | Type | Purpose |
|---|---|---|
| Choose Library… | Button | Opens a folder-picker to select the active music library |
| Library name | Label | Displays the name (last folder segment) of the open library |
| Library path | Label (below ribbon) | Shows the full filesystem path of the open library |
| Library stats | Label (below path) | Displays track count / folder count summary after a scan |
| Theme | Dropdown (Combobox) | Selects the active ttk theme (e.g. default, clam, vista) |
| UI scale | Dropdown (Combobox) | Sets display scaling factor (1.25 → 6.0); triggers a full UI rebuild |
| Exit | Button | Exits the application cleanly |

### 1.2 Menu Bar

**File**
- Open Library… — folder picker; same as "Choose Library…" button
- Validate Library — checks folder structure against expected AlphaDEX layout
- Exit — quit

**Settings**
- Metadata Services… — opens the Metadata Services configuration window

**Tools**
- Library Sync… — opens the standalone Library Sync tool window
- Use Library Sync (Review) — toggle/checkbutton to switch Library Sync tab to the review-first mode
- *(separator)*
- Fix Tags via AcoustID — starts an AcoustID tag scan on the current library
- Generate Library Index… — writes an HTML index of the library
- Export Artist/Title List… — dialog to export all artist/title pairs to a text file
- Export List by Codec… — dialog to export file paths grouped by audio codec
- Playlist Artwork — opens the `Playlists/` folder in the file manager
- Playlist Repair… — opens the Playlist Repair dialog
- File Clean Up… — opens the File Cleanup dialog
- Similarity Inspector… — opens the Similarity Inspector dialog
- Metadata Fuzzy Duplicate Finder… — opens the Fuzzy Duplicate Finder dialog
- Duplicate Bucketing POC… — opens the Duplicate Bucketing POC dialog
- Duplicate Pair Review… — opens the Duplicate Pair Review dialog
- Duplicate Scan Engine… — opens the Duplicate Scan Engine dialog
- M4A Tester… — opens the M4A Tester dialog
- Opus Tester… — opens the Opus Tester dialog
- *(separator)*
- Reset Tag-Fix Log — clears the Tag Fixer scan history

**Debug**
- Enable Verbose Logging — sets the logger to DEBUG level

**Help**
- View Crash Log… — opens a read-only window showing the last 50 lines of `soundvault_crash.log`

---

## 2. Main Tab Bar (Notebook)

Tabs are ordered at runtime. The full list (in code order):

1. **Log**
2. **Indexer**
3. **Playlist Creator**
4. **Tag Fixer**
5. **Duplicate Finder**
6. **Player**
7. **Library Compression**
8. **Library Sync**
9. **Help**

---

## 3. Tab: Log

A single full-screen read-only scrollable text area. All timestamped messages
from background operations (indexing, tag fixing, etc.) are appended here in
real time. Automatically scrolls to the latest entry.

---

## 4. Tab: Indexer

Controls the file-organization and rename pipeline.

| Element | Type | Purpose |
|---|---|---|
| Dry Run | Checkbox | When checked, only writes the HTML preview; no files are moved |
| Enable Cross-Album Scan (Phase 3) | Checkbox | Activates the third scan phase that checks across album boundaries |
| Flush Cache | Checkbox | Forces the fingerprint cache to be cleared before the run |
| Create Playlists | Checkbox | Generates `.m3u` playlists in the `Playlists/` folder on a full run |
| Max Workers | Text entry (narrow) | Number of parallel worker threads (blank = system default) |
| Start Indexer | Button | Launches the indexer in a background thread |
| Open 'Not Sorted' Folder | Button | Opens the `Not Sorted/` subfolder in the file manager |
| Cancel | Button (disabled until running) | Signals the running indexer to stop |
| Phase A progress bar | Progress bar | Progress of Phase A (file discovery / metadata scan) |
| Phase B progress bar | Progress bar | Progress of Phase B (file moves / renames) |
| Phase C progress bar | Progress bar | Progress of Phase C (cross-album scan, when enabled) |
| Status label | Label | Short current-status message; hovering shows the full text (tooltip) |
| Run log | Scrolled text area | Detailed per-file log of indexer decisions during the current run |

---

## 5. Tab: Playlist Creator

A two-column layout. Left: a fixed list of tool names. Right: the active tool
panel (swapped when a tool is selected).

### 5.1 Tool Selector (left column)

A vertical listbox. Clicking an entry loads that tool's panel on the right.
Available tools:
1. Interactive – KMeans
2. Interactive – HDBSCAN
3. Genre Normalizer
4. Year Assistant
5. Tempo/Energy Buckets
6. Auto-DJ

---

### 5.2 Panel: Interactive – KMeans

Wraps the `ClusterGraphPanel` interactive scatter plot. Users can explore the
clustered library as a 2-D map. Panel-level controls exposed alongside the
graph:

**Graph area** — the cluster scatter plot itself (mouse-interactive)

**Side panel controls**

| Element | Type | Purpose |
|---|---|---|
| Cluster Loader section | LabelFrame | Houses controls for computing/reloading clusters |
| N Clusters | Spinbox / entry | Number of K-Means clusters to compute |
| Engine selector | Dropdown | Choose feature extraction engine (librosa or essentia) |
| Recompute Clusters | Button | Re-runs K-Means with the current settings |
| Temporary Playlist section | LabelFrame | Build a playlist from points selected on the graph |
| Add Selected | Button | Adds the currently selected graph points to the temp playlist |
| Remove Selected | Button | Removes highlighted items from the temp playlist |
| Clear | Button | Empties the temp playlist |
| Save Playlist | Button | Writes the temp playlist to the Playlists folder |
| Playlist list | Listbox (scrollable) | Shows items currently in the temp playlist |

---

### 5.3 Panel: Interactive – HDBSCAN

Same layout as Interactive – KMeans but with HDBSCAN-specific parameters in the
Cluster Loader:

| Element | Type | Purpose |
|---|---|---|
| Min Cluster Size | Entry | Minimum number of points required to form a cluster |
| Min Samples | Entry (optional) | Controls how conservative cluster boundaries are |
| Cluster Selection Epsilon | Entry (optional) | Merges clusters closer than this distance |
| Engine selector | Dropdown | Feature extraction engine |
| Recompute Clusters | Button | Re-runs HDBSCAN |

---

### 5.4 Panel: Genre Normalizer

| Element | Type | Purpose |
|---|---|---|
| Library label | Label | Shows the currently open library path |
| Mapping file label | Label | Shows the path where `Docs/.genre_mapping.json` will be saved |
| Scan Genres button | Button | Scans all audio files and collects their raw genre tags |
| Scan status label | Label | Shows scan progress ("Scanning genres…" / "Found N genres.") |
| Scan progress bar | Progress bar | Advances during the genre scan |
| Initialize Normalizer button | Button | Takes the JSON from the Paste area, writes genres to files |
| Save Mapping button | Button | Saves the current mapping JSON to `Docs/.genre_mapping.json` |
| Apply To Songs button | Button | Applies a saved or pasted JSON mapping to embedded tags |
| Init / apply status labels | Labels | Reports how many files were rewritten |
| LLM Prompt Template section | LabelFrame + ScrolledText (read-only) | Shows the prompt to paste into an AI assistant |
| Copy Prompt | Button | Copies the prompt template to the clipboard |
| Raw Genre List section | LabelFrame + ScrolledText (read-only) | Populated with all unique genres found during the scan |
| Copy Raw List | Button | Copies the raw genre list to the clipboard |
| Paste JSON Mapping Here section | LabelFrame + ScrolledText (editable) | User pastes the AI-generated JSON mapping |

---

### 5.5 Panel: Year Assistant

A two-step workflow for filling in missing release years.

**Step 1 — Generate AI Instructions**

| Element | Type | Purpose |
|---|---|---|
| Library label | Label | Shows current library |
| Instructions section | LabelFrame + ScrolledText (read-only) | The system prompt to paste into an AI |
| Copy Instructions | Button | Copies the prompt to the clipboard |
| Tracks missing year section | LabelFrame + ScrolledText (read-only) | List of "Artist – Title" lines that have no year tag |
| Copy Track List | Button | Copies the track list to the clipboard |

**Step 2 — Import Results + Apply**

| Element | Type | Purpose |
|---|---|---|
| Paste AI Results section | LabelFrame + ScrolledText (editable) | User pastes the AI's JSON response |
| Parse Results | Button | Validates and loads the JSON into the review table |
| Status label | Label | Shows current step state |
| Results table | Treeview | Columns: Track, Year, Confidence, Notes, Status |
| Apply Selected | Button | Writes the year tag for rows selected in the table |
| Apply High Confidence | Button | Writes year tags for all rows with confidence = "high" |
| Dry Run checkbox | Checkbox | When checked, shows what would change without writing files |

---

### 5.6 Panel: Tempo/Energy Buckets

| Element | Type | Purpose |
|---|---|---|
| Tempo/Energy Buckets section | LabelFrame | Houses all bucket config |
| Tempo Engine section | LabelFrame | Chooses feature extraction engine |
| Engine radio / dropdown | Selector | librosa vs. essentia |
| BPM Low / BPM High entries | Text entries | Defines the BPM range for the bucket |
| Energy Low / Energy High entries | Text entries | Defines the energy range for the bucket |
| Add Bucket | Button | Adds the configured bucket to the list |
| Bucket list | Listbox / table | Shows configured buckets |
| Remove Selected | Button | Removes the highlighted bucket |
| Generate Playlists | Button | Runs the bucketing and writes `.m3u` files |
| Progress bar | Progress bar | Advances during feature extraction |
| Run Log section | LabelFrame + ScrolledText | Timestamped messages from the playlist job |
| Bucket Summary section | LabelFrame | Shows a count of tracks per bucket after a run |

---

### 5.7 Panel: Auto-DJ

| Element | Type | Purpose |
|---|---|---|
| Auto-DJ Settings section | LabelFrame | Houses all config |
| Output path label + Browse button | Path row | Where to save the generated playlist |
| Seed track field + Browse | Path row | Optional starting track for the similarity chain |
| Max tracks entry | Text entry | Maximum number of tracks to include |
| Similarity threshold entry | Text entry | Minimum similarity score to chain to the next track |
| Generate | Button | Starts the Auto-DJ playlist generation |
| Auto-DJ Log section | LabelFrame + ScrolledText | Shows generation progress messages |

---

## 6. Tab: Tag Fixer

A scan-and-review table for AcoustID / MusicBrainz metadata suggestions.

| Element | Type | Purpose |
|---|---|---|
| Folder path entry | Text entry | Path to scan (pre-filled from the active library) |
| Browse… | Button | Folder picker |
| Scan | Button | Starts an AcoustID metadata scan in the background |
| Exclude 'no diff' | Checkbox | Hides rows where no metadata change was proposed |
| Exclude 'skipped' | Checkbox | Hides rows that the scanner skipped |
| Show All | Checkbox | Overrides both exclude filters and shows every scanned file |
| Verbose Debug | Checkbox | Enables extra debug output in the log tab |
| API status indicator | Colored dot + label | Green = AcoustID accessible, amber = untested, red = failed |
| Progress bar | Progress bar | Advances during the scan |
| Results table | Treeview (multi-select, sortable columns) | Columns: File, Score, Old Artist, New Artist, Old Title, New Title, Old Album, New Album, Genres, Suggested Genre |
| Selected count label | Label | "Selected: N" — updates as rows are selected |
| Ctrl+A | Keyboard shortcut | Selects all rows in the table |
| Apply field checkboxes | Checkboxes | Artist / Title / Album / Genres — controls which fields get written |
| Apply Selected | Button | Writes the checked fields for all selected rows |

Row color coding: white = high confidence match, yellow = changed/medium confidence, red/pink = low confidence.

---

## 7. Tab: Duplicate Finder

A lightweight launcher tab. The actual duplicate workflow lives in a separate
floating window (see Section 13).

| Element | Type | Purpose |
|---|---|---|
| Library path label | Label | Shows the currently open library |
| Description text | Label | Explains the workflow in one sentence |
| Open Duplicate Finder | Button | Opens the `DuplicateFinderShell` window (disabled until a library is open) |

---

## 8. Tab: Player

An in-app audio player with playlist-building tools.

### 8.1 Controls bar (top)

| Element | Type | Purpose |
|---|---|---|
| Status label | Label | Current playback state (e.g. "Playing…", "Stopped") |
| Search label + entry | Text entry | Live filter applied to the library table as you type |
| Reload | Button | Rescans the library folder and refreshes the table |

### 8.2 Library table (left side)

| Column | Description |
|---|---|
| Title | Track title |
| Artist | Artist name |
| Album | Album name |
| Length | Duration (mm:ss) |
| Play | A "▶" button; clicking starts playback for that row |

Clicking a row populates the album art panel on the right. Selecting multiple
rows and using the playlist builder adds them to the temp queue.

### 8.3 Art + playlist panel (right side)

| Element | Type | Purpose |
|---|---|---|
| Album art caption | Label | "Select a track to view album art" placeholder |
| Album art image | Image label | Displays embedded cover art for the selected track |
| Add Playlist | Button | Toggles between library view and a loaded playlist view |
| Load Playlist | Button | Opens a file picker to load an `.m3u/.m3u8` playlist into the table |
| Playlist Builder section | LabelFrame | Temporary queue management |
| Playlist status label | Label | Shows how many tracks are in the temp queue |
| Playlist listbox | Listbox (scrollable, multi-select) | Items in the temp playlist |
| Add Selected | Button | Adds the highlighted library rows to the temp playlist |
| Remove Selected | Button | Removes highlighted items from the temp playlist |
| Clear | Button | Empties the temp playlist |
| Save Playlist | Button | Writes the temp playlist to an `.m3u` file |
| Current Playlists | Button | Shows a message box listing all playlists in the library |

---

## 9. Tab: Library Compression

Converts a FLAC library to an Opus mirror (96 kbps) for a more compact copy.

| Element | Type | Purpose |
|---|---|---|
| Title label | Bold heading label | "Library Compression" |
| Description label | Label | Explains the FLAC-to-Opus conversion |
| Source Library section | LabelFrame + entry + Browse | Folder to read from |
| Destination Mirror section | LabelFrame + entry + Browse | Folder to write the Opus mirror to |
| Overwrite existing files | Checkbox | Whether to re-encode files that already exist at the destination |
| Status label | Label | "Ready to mirror library." / "Mirroring library…" / "Completed." |
| Progress bar | Progress bar | Advances as files are processed |
| Progress count label | Label | "Progress: N / M" |
| Start | Button | Begins the mirroring process |
| Open Report | Button (initially disabled) | Opens the HTML mirror report once the run finishes |

---

## 10. Tab: Library Sync

Hosts the `LibrarySyncReviewPanel` (review-first sync workflow).
Full detail is in Section 17 below.

---

## 11. Tab: Help

An LLM chat assistant panel for asking questions about the application.

| Element | Type | Purpose |
|---|---|---|
| Chat history | ScrolledText (read-only) | Displays the conversation history |
| Chat input | Text entry | User types a question here |
| Send | Button | Submits the question to the assistant |

---

## 12. Settings Window: Metadata Services

Opened from **Settings → Metadata Services…**. Non-modal window.

| Element | Type | Purpose |
|---|---|---|
| Service | Dropdown (Combobox, read-only) | Choose between "AcoustID" and "MusicBrainz" |
| **When AcoustID is selected:** | | |
| API Key | Text entry | AcoustID application API key |
| **When MusicBrainz is selected:** | | |
| App | Text entry | Application name for the User-Agent string |
| Version | Text entry | Application version |
| Contact | Text entry | Contact email for the User-Agent string |
| Test Connection | Button | Validates the current credentials against the service |
| Status label | Label | "Testing…" / green success message / red error message |
| Save | Button (enabled after successful test) | Persists the credentials to `~/.soundvault_config.json` |

---

## 13. Floating Window: Duplicate Finder

Opened from **Tab 7: Duplicate Finder → Open Duplicate Finder**. Resizable
top-level window. This is the primary deduplication workflow.

### 13.1 Library Selection

| Element | Type | Purpose |
|---|---|---|
| Library Root entry | Text entry (wide) | Path to the library to scan |
| Browse… | Button | Folder picker for the library root |
| Playlist Folder entry | Text entry (wide) | Folder where playlists will be updated after execution |
| Browse… | Button | Folder picker for the playlist folder |

### 13.2 Controls Row

| Element | Type | Purpose |
|---|---|---|
| Scan Library | Button | Fingerprints all audio files and builds duplicate groups |
| Preview | Button | Generates `Docs/duplicate_preview.html` and `duplicate_preview.json` |
| Open Report | Button (disabled until executed) | Opens the last execution HTML report |
| Execute | Button | Two-step confirmation: first click arms it (label changes), second click runs |
| ⚙️ Thresholds | Button | Opens the Threshold Settings popup (see Section 13.5) |
| Show different artwork variants | Checkbox | When checked, groups that differ only by artwork are also shown |
| Show no-op groups | Checkbox | When unchecked, hides groups that require no action |
| Update Playlists | Checkbox | When checked, playlist files are updated after execution |
| Quarantine Duplicates | Checkbox | Default action: move losers to `Quarantine/` folder |
| Delete losers (permanent) | Checkbox | Alternative to quarantine; irreversible; requires confirmation |

### 13.3 Progress + Status Bar

| Element | Type | Purpose |
|---|---|---|
| Progress bar | Progress bar | Shows combined fingerprinting / grouping / preview progress (0–100 %) |
| Status label | Label | "Idle" / "Scanning…" / "Grouping…" / "Done" |

### 13.4 Results Area (two side-by-side sections)

**Duplicate Groups list (left)**

| Element | Type | Purpose |
|---|---|---|
| Fingerprint status label | Label | Reports cache hits, total fingerprints computed |
| Groups table | Treeview | Columns: Group ID, Track Title, Count, Status |

Clicking a row in the groups table loads that group's details on the right.

**Group Details inspector (right)**

| Element | Type | Purpose |
|---|---|---|
| Group disposition label | Label | "Group disposition" |
| Group disposition selector | Combobox (disabled until a group is selected) | Per-group override: "Default (global)" / "Retain" / "Quarantine" / "Delete" |
| Group details text | ScrolledText (read-only) | Full details of the selected group: file paths, winner, planned actions, artwork hashes |

### 13.5 Log

| Element | Type | Purpose |
|---|---|---|
| Log text area | ScrolledText (read-only) | Timestamped messages from scan, preview, and execute phases |

### 13.6 Threshold Settings Popup (from ⚙️ button)

A small modal dialog with labeled text entries for each fingerprinting and
matching parameter. All values are saved to `~/.soundvault_config.json`.

| Field | Description |
|---|---|
| Exact duplicate threshold | Fingerprint distance at or below which two tracks are considered exact duplicates |
| Near duplicate threshold | Distance below which they are considered near-duplicates |
| Mixed-codec boost | Additional distance allowance when comparing lossless vs. lossy files |
| Fingerprint offset (ms) | How many milliseconds into the track the fingerprint window starts |
| Fingerprint duration (ms) | Length of the audio window used for fingerprinting |
| Silence threshold (dB) | Level below which audio is considered silence (for trimming) |
| Silence min length (ms) | Minimum duration of silence to trim |
| Trim lead max (ms) | Maximum amount of leading silence to strip |
| Trim trail max (ms) | Maximum amount of trailing silence to strip |
| Trim padding (ms) | Silence buffer left after trimming |

Single **OK** button saves and closes. Clearing the plan is triggered automatically.

---

## 14. Dialog: Similarity Inspector

Opened from **Tools → Similarity Inspector…**

| Element | Type | Purpose |
|---|---|---|
| Description label | Label | "Compare two tracks using the Duplicate Finder fingerprint pipeline." |
| Song A row | Label + entry + Browse | First audio file to compare |
| Song B row | Label + entry + Browse | Second audio file to compare |
| Run | Button | Computes fingerprints and displays the comparison report |
| Duplicate Finder Report | Button | Runs a full duplicate pair report for the two selected files |
| Close | Button | Closes the dialog |
| Advanced ▸ | Toggle button | Expands / collapses the Advanced Overrides panel |
| **Advanced Overrides panel** (collapsible) | LabelFrame | Per-run parameter overrides |
| Trim leading/trailing silence | Checkbox | Enable silence trimming before fingerprinting |
| Fingerprint offset (ms) | Entry | Start offset |
| Fingerprint duration (ms) | Entry | Window length |
| Silence threshold (dB) | Entry | Silence level |
| Silence min length (ms) | Entry | Min silence duration |
| Exact duplicate threshold | Entry | Distance cutoff for "exact" label |
| Near duplicate threshold | Entry | Distance cutoff for "near duplicate" label |
| Mixed-codec boost | Entry | Codec penalty boost |
| Results section | LabelFrame + ScrolledText (read-only) | Shows: codec, duration, sample rate, raw fingerprint distance, effective threshold, verdict |

---

## 15. Dialog: Metadata Fuzzy Duplicate Finder

Opened from **Tools → Metadata Fuzzy Duplicate Finder…**

### 15.1 Metadata Match Settings

| Element | Type | Purpose |
|---|---|---|
| Min word length | Entry | Minimum word length to include in matching (default: 4) |
| Min shared words | Entry | How many words must match between two tracks (default: 2) |
| Ignore words seen in >N tracks | Entry | Filters out stop-word-like terms (default: 200) |
| Search fields checkboxes | Checkboxes | Title / Artist / Album / Album Artist / Genre / Filename |

### 15.2 Fingerprint Filter

| Element | Type | Purpose |
|---|---|---|
| Exact threshold | Entry | Fingerprint distance for exact match |
| Near threshold | Entry | Fingerprint distance for near match |
| Mixed-codec boost | Entry | Boost for cross-codec comparisons |
| Include pairs without fingerprints | Checkbox | Keeps pairs where fingerprinting fails |
| Skip fingerprint check | Checkbox | Uses metadata-only matching (no audio analysis) |
| Help tip | Label | Explains the above two options |

### 15.3 Actions + Results

| Element | Type | Purpose |
|---|---|---|
| Run Scan | Button | Starts the metadata + fingerprint scan |
| Cancel | Button (enabled while running) | Stops the scan |
| Include non-matching fingerprints in review | Checkbox | Adds fingerprint-non-match pairs to the output |
| Close | Button | Closes the dialog |
| Progress label + status label | Labels | Show current step and file count |
| Progress bar | Progress bar | Advances during scanning |
| Candidate Results table | Treeview | Columns: Verdict, Distance, Shared Words, Matched Words, Track A, Track B |
| Summary label | Label | "Found N candidate pairs" |
| Selected Pair Details | LabelFrame + ScrolledText | Shows full paths and shared word list for the selected row |
| Send Matches to Duplicate Pair Review | Button (enabled after scan) | Sends the results to the Duplicate Pair Review window |

---

## 16. Dialog: Duplicate Pair Review

Opened from **Tools → Duplicate Pair Review…** or from the Fuzzy Duplicate
Finder "Send Matches" button.

| Element | Type | Purpose |
|---|---|---|
| Progress label | Label | "Pair N of M" |
| Left Track panel | LabelFrame | Album art image, winner badge, title, metadata summary |
| Right Track panel | LabelFrame | Same layout as Left |
| Status label | Label | Fingerprint distance and verdict for the current pair |
| ◀ Prev | Button | Navigate to the previous pair |
| Next ▶ | Button | Navigate to the next pair |
| Switch | Button | Swaps the designated winner between left and right |
| Prefer MP3 | Button | Forces the MP3 file to be the winner regardless of codec priority |
| Jump to: entry + Go | Entry + Button | Jump directly to pair number N |
| Yes | Button (also Enter key) | Confirms the current winner selection and advances |
| No | Button (also Backspace key) | Rejects the proposed action and advances |
| Tag | Button | Opens a sub-dialog to edit tags on the current pair |
| Close | Button | Closes the window |

Keyboard shortcuts: Enter = Yes, Backspace = No, ← = Prev, → = Next.

---

## 17. Tab / Panel: Library Sync (LibrarySyncReviewPanel)

Embedded directly in the **Library Sync** tab. Also used by the standalone
Library Sync window opened via **Tools → Library Sync…**.

### 17.1 Folders section

| Element | Type | Purpose |
|---|---|---|
| Existing Library label + entry + Browse | Row | Path to the destination/reference library |
| Incoming Folder label + entry + Browse | Row | Path to the new content folder to compare |

### 17.2 Scan Configuration section

| Element | Type | Purpose |
|---|---|---|
| Global Threshold | Text entry | Default fingerprint distance cutoff for matching |
| Preset Name | Text entry | Label saved with the session for reference |
| Report Version | Text entry (integer) | Version number stored in the session metadata |
| Scan State | Read-only label | Shows current lifecycle state (Idle / Ready / Scanning / Complete / Cancelled) |
| Per-format overrides text area | Multi-line text box | One `ext=threshold` line per format (e.g. `.flac=0.3`) |
| Recompute Matches | Button | Re-runs match logic on already-scanned data using updated thresholds |
| Save Session | Button | Persists current folders and config to `~/.soundvault_config.json` |

### 17.3 Scan Progress section (two side-by-side columns)

Each column (Existing Library / Incoming Folder) contains:

| Element | Type | Purpose |
|---|---|---|
| Column title | Label | "Existing Library" or "Incoming Folder" |
| Progress status label | Label | "Idle" / "Scanning…" / file count |
| Progress bar | Progress bar | Advances as files are fingerprinted |
| Phase label | Label | Current phase description (e.g. "Fingerprinting…") |
| Partial result notice | Label (amber) | Shown when a scan was cancelled mid-way |
| Scan | Button | Starts the scan for this column |
| Cancel | Button | Cancels the active scan for this column |

### 17.4 Review Results section

**Incoming Tracks list** (left panel)

| Column | Description |
|---|---|
| Track | Filename of the incoming file |
| Chips | Status badges: New / Collision / Exact Match / Low Confidence / Potential Upgrade / Keep Existing / Missing Metadata / Partial |
| Distance | Raw fingerprint distance to best match |

**Existing Tracks list** (right panel)

| Column | Description |
|---|---|
| Track | Filename of the existing file |
| Chips | Status badges: Best Match / Potential Upgrade / Keep Existing / Partial / Unmatched |
| Best Matches | Count of incoming tracks that map to this file |

**Match summary label** — "N matches found" / "No matches yet."

**Inspector** — Read-only text area showing full match details when a row is
selected in either list: metadata, file path, match status, fingerprint
distance, threshold used, confidence score, quality label (Potential Upgrade /
Keep Existing).

### 17.5 Plan & Execution section

| Element | Type | Purpose |
|---|---|---|
| Plan status label | Label | "No plan built." / progress messages |
| Plan progress bar | Progress bar | Advances during plan build and execution |
| Plan phase label | Label | Current phase description |
| Build Plan | Button | Computes a deterministic copy/move plan from matched results |
| Preview | Button | Renders an HTML dry-run preview of the plan |
| Copy Originals / Move Originals | Toggle button | Switches the transfer mode; label updates to show current mode |
| Execute | Button | Applies the plan (requires a matching previewed plan) |
| Output playlist | Checkbox | When checked, execution also writes an `.m3u8` playlist of transferred files |
| Open Preview | Button (disabled until preview exists) | Opens the preview HTML in the browser |
| Cancel | Button (disabled unless a task is running) | Stops the active plan/execution task |

### 17.6 Logs section

| Element | Type | Purpose |
|---|---|---|
| Log text area | Scrollable text (read-only) | Timestamped scan, plan, and execution events (capped at 400 entries) |
| Export Logs… | Button | Saves the current log to a user-chosen file |

### 17.7 Report Preview Dialog (modal)

Shown internally before saving a report. Not directly user-triggered at present
(the button path is not yet wired).

| Element | Type | Purpose |
|---|---|---|
| Summary labels | Labels | New / Collisions+Exact / Low Confidence / Flagged Copy / Flagged Replace counts |
| Save Report | Button | Confirms and proceeds |
| Cancel | Button | Dismisses without saving |

---

## 18. Dialog: Export Artist/Title List

Opened from **Tools → Export Artist/Title List…**

| Element | Type | Purpose |
|---|---|---|
| Library Path section | LabelFrame + label | Shows the library being scanned |
| Output File section | LabelFrame + label | Shows where `Docs/artist_title_list.txt` will be written |
| Options section | LabelFrame | |
| Exclude flac files | Checkbox | Skips FLAC tracks from the export |
| Add album song duplicates | Checkbox | Includes tracks that appear more than once per album |
| Progress bar | Progress bar | Advances during scanning |
| Progress status label | Label | "Scanning…" / "N tracks found" |
| Export Log section | LabelFrame + ScrolledText | Timestamped log of the export |
| Start Export | Button | Begins scanning and writing the list |
| Open File | Button (enabled after export) | Opens the output file |
| Close | Button | Dismisses the dialog |

---

## 19. Dialog: Export List by Codec

Opened from **Tools → Export List by Codec…**

| Element | Type | Purpose |
|---|---|---|
| Library Path section | LabelFrame + label | Shows the library being scanned |
| Output File section | LabelFrame + label | Shows where the codec list will be written |
| Options section | LabelFrame | |
| Codec filter | Checkboxes or dropdown | Selects which codec(s) to include in the export |
| Omit paths | Checkbox | Exports filenames only, not full paths |
| Progress bar | Progress bar | Advances during scanning |
| Progress status label | Label | Scan status text |
| Export Log section | LabelFrame + ScrolledText | Timestamped log |
| Start Export | Button | Begins the scan |
| Open File | Button (enabled after export) | Opens the output file |
| Close | Button | Dismisses the dialog |

---

## 20. Dialog: Playlist Repair

Opened from **Tools → Playlist Repair…**

| Element | Type | Purpose |
|---|---|---|
| Description label | Label | Explains what the repair does |
| Playlists section | LabelFrame | |
| Playlist listbox | Listbox (scrollable) | Shows selected playlist files to repair |
| Add… | Button | Opens a multi-file picker for `.m3u` / `.m3u8` files |
| Remove Selected | Button | Removes highlighted items from the list |
| Clear | Button | Empties the list |
| Prefer Opus when searching for missing FLAC tracks | Checkbox | If a FLAC track is missing, looks for an `.opus` equivalent |
| Status label | Label | Current operation state |
| Progress bar | Progress bar | Advances as playlists are processed |
| Progress count label | Label | "Progress: N / M" |
| Repair Log section | LabelFrame + ScrolledText | Per-file repair messages |
| Run Repair | Button | Starts the repair in a background thread |
| Open Report | Button (enabled after repair) | Opens the HTML repair report |
| Close | Button | Closes the dialog |

---

## 21. Dialog: File Cleanup

Opened from **Tools → File Clean Up…**

| Element | Type | Purpose |
|---|---|---|
| Title label | Bold label | "File Clean Up" |
| Description label | Label | Explains it removes trailing `(N)` or " copy" suffixes |
| Library Root entry | Text entry | Path to the library (pre-filled from current library) |
| Browse… | Button | Folder picker |
| Status label | Label | "Ready to scan." / "Scanning files…" / completion summary |
| Execute | Button | Starts the rename pass in a background thread |
| Cancel / Close | Button | Cancels before running; changes to "Close" after completion |

---

## 22. Dialog: M4A Tester

Opened from **Tools → M4A Tester…**

| Element | Type | Purpose |
|---|---|---|
| Description label | Label | "Select an M4A file to validate album art and metadata parsing." |
| M4A File section | LabelFrame | |
| File entry | Text entry (read-only) | Path of the selected file |
| Browse… | Button | Opens a file picker filtered to `.m4a` files |
| Album Art section | LabelFrame | Displays the extracted cover image (or "No album art found") |
| Metadata section | LabelFrame + ScrolledText | Shows: Title, Artist, Album, Album Artist, Track, Disc, Year, Date, Genre, Compilation |
| Close | Button | Dismisses the dialog |

---

## 23. Dialog: Opus Tester

Opened from **Tools → Opus Tester…**

Same layout as M4A Tester but targets `.opus` files. Displays:

| Section | Contents |
|---|---|
| Opus File | File entry (read-only) + Browse |
| Album Art | Cover image display |
| Metadata | All embedded Opus tags |
| Controls | Close button |

---

## 24. Dialog: Duplicate Bucketing POC

Opened from **Tools → Duplicate Bucketing POC…**

| Element | Type | Purpose |
|---|---|---|
| Description label | Label | "Select a folder to scan for duplicate bucketing." |
| Folder entry | Text entry | Folder to scan (pre-filled from current library) |
| Browse… | Button | Folder picker |
| Run | Button | Runs the bucketing algorithm and saves an HTML report |
| Close | Button | Closes the dialog |
| Status label | Label | "Idle" / "Running…" / "Completed" / "Failed" |

After completion, a yes/no dialog offers to open the HTML report.

---

## 25. Dialog: Duplicate Scan Engine

Opened from **Tools → Duplicate Scan Engine…**

A more advanced, experimental duplicate scanner using a staged pipeline
(audio headers → fingerprint LSH → chroma verification). Results are written
to a SQLite database.

### Paths section

| Element | Type | Purpose |
|---|---|---|
| Library Root entry + Browse | Row | Folder to scan |
| Database entry + Browse | Row | SQLite output database (default: `Docs/duplicate_scan.db`) |

### Scan Settings section

All values are text entries:

| Setting | Description |
|---|---|
| Sample rate (Hz) | Audio sample rate for fingerprinting (default: 11025) |
| Max analysis seconds | Maximum audio duration to analyze per file |
| Duration tolerance (ms) | Absolute duration difference allowed when matching |
| Duration tolerance ratio | Relative duration difference allowed |
| FP bands | Number of fingerprint LSH bands |
| Min band collisions | Minimum LSH band matches to flag as a candidate |
| FP distance threshold | Maximum fingerprint distance for a match |
| Chroma offset frames | Frame offset used in chroma comparison |
| Chroma match threshold | Score above which tracks are considered a confirmed match |
| Chroma possible threshold | Score above which tracks are flagged as possible matches |

### Controls + Log

| Element | Type | Purpose |
|---|---|---|
| Run Scan | Button | Starts the staged scan in a background thread |
| Status label | Label | "Idle" / "Running…" / "Done" |
| Log text area | ScrolledText (read-only) | Progress messages from each scan stage |

---

## 26. Crash Log Viewer

Opened from **Help → View Crash Log…**

A read-only scrolled text window showing the last 50 lines of
`soundvault_crash.log`. No interactive controls other than closing the window.

---

## 27. Inline Widgets and Shared Patterns

| Pattern | Description |
|---|---|
| **Tooltip** | Any label with a `Tooltip` attachment shows a pop-up with the full text on hover (e.g., the truncated status label in the Indexer tab) |
| **Progress Dialog** (ProgressDialog class) | A modal "Working…" window with a determinate progress bar and a Cancel button; used by long operations that do not have their own inline progress UI |
| **Unsorted Popup** (UnsortedPopup, from `unsorted_popup.py`) | A popup that appears when unrecognized files are found in the `Not Sorted/` folder; lets the user decide how to handle them |
| **Browse… buttons** | All folder/file pickers remember the last-used path via `load_last_path()` / `save_last_path()` |
| **Scrollable text areas** | Read-only areas use `ScrolledText` with `state="disabled"`; they become normal briefly when new text is appended, then disabled again |
| **Treeview sort** | The Tag Fixer table supports column-header clicking to sort ascending/descending |
| **Background threading** | All long operations run in daemon threads; GUI updates are dispatched back to the main thread via `after()` |

---

*Document generated from a full read of `main_gui.py` (11,587 lines) and
`library_sync_review.py` (1,269 lines) on 2026-03-15.*
