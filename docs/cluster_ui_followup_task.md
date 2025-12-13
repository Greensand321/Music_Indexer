# Task: Stabilize clustered playlist UI rendering & controls

## Problem
Users reported two regressions after running either K-Means or HDBSCAN scans:
1. The interactive cluster graph renders at an incorrect, undersized default until the window is manually resized, suggesting the canvas layout is not recalculated when it first appears.
2. Buttons inside the Playlist Creator tab occasionally trigger the wrong plugin after a scan finishes—for example, clicking **Interactive – KMeans** might open **Auto-DJ** instead—implying selection state is not kept in sync with the visible panel.

## Scope
- `cluster_graph_panel.py`
- `main_gui.py`
- Manual verification steps in the Playlist Creator tab

## Acceptance Criteria
- The cluster graph redraws itself to the correct size immediately after clustering finishes, without requiring the user to move or resize the window.
- Plugin buttons in the Playlist Creator tab always open the panel that matches the clicked entry, even after running clustering jobs or rebuilding the UI.
- Regression notes or logs explain how selection state is preserved and how the canvas redraw is triggered.

## Notes
- Consider debouncing resize handling to avoid excessive redraws when the window is being resized.
- Preserve the last selected plugin when rebuilding the UI (e.g., after scaling changes) to prevent misrouted actions.
