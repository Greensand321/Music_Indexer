# Playlist Gap — interface mockups

Five interaction models for the Playlist Gap workspace, built to be compared and
narrowed down to one. Concept stage — see `docs/playlist_gap_feature_plan.md` for
the feature itself.

## Current direction

| File | Model | In one line |
|---|---|---|
| **`06-hybrid-v2.html`** | **Hybrid v2** ★ | **Tiles and tabs at once. The tile strip is the live overview *and* the navigation; each step gets the whole pane below it. All six panes are filled in — click the tiles to move between them.** |

Round 1, kept for reference only:

| File | Model | In one line |
|---|---|---|
| `01-flow-tiles.html` | Flow Tiles | Every step is a tile on one board, wired in order. |
| `02-tabbed-steps.html` | Tabbed Steps | One step at a time behind a tab bar that tracks completion. |
| `03-workbench.html` | Workbench | Setup rail, three-column bucket board, docked evidence panel. |
| `04-triage-queue.html` | Triage Queue | An inbox — list, reading pane, keyboard-driven. |
| `05-answer-first.html` | Answer First | Opens on the download list; machinery collapsed behind a summary. |
| `index.html` | — | All six behind a switcher, for side-by-side comparison. |

Open `index.html` in a browser and press `1`–`6` (`1` is Hybrid v2).

### What v2 changed, and why

- **Tiles moved to where they earn their keep** — a persistent strip that reports live state and
  doubles as the step navigation, plus a full **Summary** board at the end. They are no longer the
  main workflow.
- **Every pane is drawn**, not just one, so the whole shape is visible rather than partial.
- **Setup is deliberately small** — a source dropdown, a playlist URL, a name, and the library
  folder inclusion list. Two fields and a scan, as it should be.
- **Triage got the depth**: a queue on the left, a full evidence comparison in the middle with
  **cover art on both sides**, and a **ranked list of every possible counterpart** on the right,
  clickable, plus manual library search and a "none of these" escape. Nothing is accepted blindly.
- **Compare became a real pane** — a ladder readout showing how many rows each rung resolved, so
  the result is inspectable instead of a black box.

### Revision 2.1 — the second pass, and saved sources

- **A seventh step: Verify.** The pipeline now spans **two passes** — find and sort, then check
  what actually downloaded. Downloaders often fetch the wrong recording (you asked for the remix
  and got the original), which leaves a duplicate *and* the wanted track still missing, while the
  ledger marks the row done. The Verify pane checks source ID first, fingerprints the rest, and
  splits the folder into five outcomes; only the *wrong version* group needs a decision, and its
  fix writes the correction back into the ledger. The tile strip shows the
  `you download` break between the two passes.
- **Saved sources.** Setup now leads with a list of saved sources — name, source, URL, row count,
  *new since last run*, and a ▶ button per row — plus **Run all**, which builds one combined list
  with each track only once. The routine monthly use is one press.

## Editing

`build_mockups.py` is the **source of truth** — it emits all six HTML files, so the
standalone mockups and the switcher can't drift apart. Edit the generator, then:

```bash
python docs/mockups/playlist_gap/build_mockups.py
```

Palette and sidebar come from the app's real `MIDNIGHT` theme in
`gui/themes/tokens.py`, so the mockups sit at roughly the density the PySide6 build
would have. Counts and track names are the user's real data from a YouTube Music
export, not placeholders.

These are interface sketches only. No Qt code, no wiring, nothing imported by the app.
