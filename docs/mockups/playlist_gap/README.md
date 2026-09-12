# Playlist Gap — interface mockups

Five interaction models for the Playlist Gap workspace, built to be compared and
narrowed down to one. Concept stage — see `docs/playlist_gap_feature_plan.md` for
the feature itself.

| File | Model | In one line |
|---|---|---|
| `01-flow-tiles.html` | Flow Tiles | Every step is a tile on one board, wired in order. |
| `02-tabbed-steps.html` | Tabbed Steps | One step at a time behind a tab bar that tracks completion. |
| `03-workbench.html` | Workbench | Setup rail, three-column bucket board, docked evidence panel. |
| `04-triage-queue.html` | Triage Queue | An inbox — list, reading pane, keyboard-driven. |
| `05-answer-first.html` | Answer First | Opens on the download list; machinery collapsed behind a summary. |
| `index.html` | — | All five behind a switcher, for side-by-side comparison. |

Open `index.html` in a browser and press `1`–`5`.

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
