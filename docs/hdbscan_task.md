# Task: Fix HDBSCAN visualization & parameter handling in Clustered Playlists

## Problem
The interactive HDBSCAN view in the Playlist Creator tab often renders a single-color scatter plot because clustering is recomputed on raw, unscaled feature vectors (`np.vstack(self.features)`) while offline generation scales features with `StandardScaler`. Without normalization or dimensionality reduction, HDBSCAN frequently labels all points as noise, leading to the uninformative visualization.

## Scope
- `cluster_graph_panel.py`: interactive reclustering and parameter dialog.
- `main_gui.py`: plugin wiring for interactive HDBSCAN/KMeans panels.
- `clustered_playlists.py`: feature extraction and storage feeding the interactive view.

## Goals
1. Ensure the interactive HDBSCAN workflow uses the same preprocessing as offline generation (feature scaling and optional dimensionality reduction) so clusters can form when density exists.
2. Let users adjust key HDBSCAN parameters before running the initial interactive plot, not only after clicking “Redo Values.”
3. Keep K-Means behavior unchanged while preventing shared-state regressions between the two interactive modes.
4. Make the visualization clearly differentiate noise points versus clustered points after preprocessing fixes.

## Proposed Approach
- Pass scaled (and optionally reduced) features into the interactive panel, reusing the standardized matrix already produced for playlist generation instead of the raw feature list.
- Centralize HDBSCAN parameter defaults so both the initial render and the edit dialog share the same values, and surface the dialog before the first run (or persist last-used values from `cluster_params`).
- Update recoloring logic if needed to keep noise (`label == -1`) visibly distinct after preprocessing changes.
- Add defensive logging to confirm when HDBSCAN returns only noise versus multiple clusters to aid troubleshooting.

## Acceptance Criteria
- Interactive HDBSCAN plots use normalized feature data and, when clusters exist, show multiple colors; an all-noise result is logged explicitly rather than silently appearing unclustered.
- Users can configure `min_cluster_size`, `min_samples`, and `cluster_selection_epsilon` prior to the first interactive HDBSCAN run, with sensible defaults and validation.
- K-Means interactive behavior remains the same, and shared UI elements (buttons, dialogs) still function for both clustering modes.
- Manual test instructions added to the issue or commit notes describing how to verify the corrected visualization and parameter entry.
