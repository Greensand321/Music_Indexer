# Interactive Graphs & Enhanced Clustering — Comprehensive Implementation Plan

**Status:** 🔴 Not Started (Major Gap Identified)
**Priority:** High - Core Feature
**Estimated Effort:** 40-60 hours

---

## Current Problems Identified

### 1. **Graph Visualization Not Working**
- `cluster_graph_panel.py` exists but uses **Tkinter** (matplotlib/tkinter backend)
- Main app is **Qt-based** (gui/workspaces using PyQt6/PySide6)
- **Result:** Graph code unreachable from Qt, creates a mismatch
- `gui/workspaces/graph.py` only shows placeholder messages

### 2. **Missing User Prompts & Controls**
- Clustering UI has basic K-means/HDBSCAN selector
- **Missing prompts for:**
  - Feature selection (which features to use?)
  - Feature normalization preferences
  - Algorithm parameters (eps, samples in HDBSCAN, etc.)
  - Post-processing options (merge small clusters, remove outliers?)
  - Output preferences (playlists, visualization, report)
- Users can't understand or control the clustering process

### 3. **HTML Graphs Open Externally**
- Current design: generate HTML, open in browser
- **Problem:** Breaks flow, separate window, can't interact with main app while viewing
- **User preference:** Keep everything in-app, embedded visualization

### 4. **Backend Gaps**
- Feature extraction happens but no way to inspect features
- No dimension reduction visualization (PCA, t-SNE, UMAP)
- No cluster quality metrics shown
- No way to adjust parameters and re-cluster interactively
- No drill-down into cluster details
- No export/playlist generation from graph selection

---

## Solution Architecture

### Phase 1: Port Tkinter Graph to Qt (Backend Independence)

Create a reusable, backend-agnostic interactive graph widget:

```
┌─────────────────────────────────────────────────────────────┐
│ Interactive Scatter Plot Widget (Pure Qt)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✓ Scatter plot with zooming/panning                       │
│  ✓ Point selection (lasso, rectangle, by cluster)          │
│  ✓ Hover tooltips with metadata                            │
│  ✓ Legend showing clusters                                 │
│  ✓ Color by cluster, by quality, by selected, etc.         │
│  ✓ Export selection as playlist                            │
│  ✓ Interactive parameter tuning                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principle:** Use PyQtGraph (Qt-native, high-performance) instead of matplotlib + tkinter.

### Phase 2: Enhanced Clustering UI with Prompts

Create a **Clustering Wizard** dialog:

```
Step 1: Feature Selection
┌─────────────────────────────────────────────────────┐
│ ☑ Tempo           ← BPM based rhythm               │
│ ☑ MFCC            ← Timbre (13 coefficients)       │
│ ☑ Chroma          ← Harmonic content               │
│ ☑ Spectral        ← Brightness                     │
│ ☑ Energy          ← Loudness characteristics       │
│ ☑ Onset rate      ← Percussion density             │
│                                                     │
│ [ⓘ Info] [Sample]                                 │
└─────────────────────────────────────────────────────┘

Step 2: Normalization & Preprocessing
┌─────────────────────────────────────────────────────┐
│ Normalization:  ⊙ Standard (Z-score)               │
│                 ⊙ MinMax (0-1)                     │
│                 ⊙ RobustScaler (outlier-resistant) │
│                                                     │
│ Dimensionality reduction:                           │
│ ⊙ None (use all features)                          │
│ ⊙ PCA (faster, linear)                             │
│ ⊙ t-SNE (better visualization, slow)               │
│ ⊙ UMAP (balanced, fast)                            │
│                                                     │
│ [ⓘ Info] [Preview]                                │
└─────────────────────────────────────────────────────┘

Step 3: Algorithm Selection & Parameters
┌─────────────────────────────────────────────────────┐
│ Algorithm: ⊙ K-Means  ⊙ HDBSCAN  ⊙ DBSCAN         │
│                                                     │
│ K-Means Parameters:                                 │
│ K (clusters):        [8____________] (2-100)        │
│ Max iterations:      [300__________] (10-1000)      │
│ Random seed:         [42___________] (for repro)    │
│ Init method:         [k-means++▼]   (k-means++, random)│
│                                                     │
│ [ⓘ Info] [Preset: Pop/Rock] [Preset: Electronic]   │
└─────────────────────────────────────────────────────┘

Step 4: Post-Processing
┌─────────────────────────────────────────────────────┐
│ ☑ Remove outliers (clusters < 5 tracks)            │
│ ☑ Merge small clusters into "Misc"                 │
│ ☑ Remove silence/noise cluster                     │
│                                                     │
│ Min cluster size: [5___________]                    │
│                                                     │
│ [ⓘ Info]                                           │
└─────────────────────────────────────────────────────┘

Step 5: Output & Visualization
┌─────────────────────────────────────────────────────┐
│ ☑ Create playlists for each cluster                │
│ ☑ Generate interactive graph                        │
│ ☑ Create cluster quality report                    │
│ ☑ Export cluster CSV                               │
│                                                     │
│ Playlist naming: [Cluster {N} - {Label}▼]          │
│                                                     │
│ [ⓘ Info]                                           │
└─────────────────────────────────────────────────────┘

             [< Back] [Next >] [Cancel] [Run]
```

### Phase 3: Embedded Interactive Graph

Replace external HTML with in-app visualization:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Visual Music Graph (Embedded in Qt)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Toolbar: [🔍 Zoom] [↔ Pan] [⊟ Fit] [■ Select] [●●●○ Filters]   │
│           [💾 Save] [📄 Report] [🎵 Playlist] [↺ Refresh]        │
│                                                                     │
│  ┌────────────────────────────────────────┐  ┌────────────────┐   │
│  │                                        │  │   Clusters     │   │
│  │      ●  ●●  ●     ●●●                 │  │ ☑ Cluster 0 ●  │   │
│  │    ●  ●●●●●●● ● ●●●●●●●●●            │  │ ☑ Cluster 1 ●  │   │
│  │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●       │  │ ☑ Cluster 2 ●  │   │
│  │  ●●●●●●●●●●●●●●●●●●●●●●●●●●●●       │  │ ☑ Cluster 3 ●  │   │
│  │    ●  ●●●●●●● ● ●●●●●●●●●            │  │ ☑ Cluster 4 ●  │   │
│  │      ●  ●●  ●     ●●●                 │  │ ☐ Outliers ●   │   │
│  │                                        │  │                │   │
│  │          (Interactive Scatter)         │  │ ▼ Size by:     │   │
│  │          • Zoom: mouse wheel           │  │ ⊙ Cluster      │   │
│  │          • Pan: click+drag             │  │ ⊙ Popularity   │   │
│  │          • Lasso: polygon selection    │  │ ⊙ BPM          │   │
│  │          • Hover: metadata tooltip     │  │ ⊙ Duration     │   │
│  │          • Click: track details        │  │                │   │
│  │                                        │  │ ▼ Color by:    │   │
│  │                                        │  │ ⊙ Cluster      │   │
│  │                                        │  │ ⊙ Genre        │   │
│  │                                        │  │ ⊙ Artist       │   │
│  │                                        │  │ ⊙ Quality      │   │
│  │                                        │  │ ⊙ Selection    │   │
│  │                                        │  └────────────────┘   │
│  └────────────────────────────────────────┘                       │
│                                                                     │
│  Selection: 0 points selected                                      │
│  [→ Create Playlist] [→ Export CSV] [Clear Selection]              │
│                                                                     │
│  Cluster Details:                                                   │
│  Cluster 2: 47 tracks | Avg BPM: 128 | Genres: Techno, House     │
│  [View All] [Details] [Create Playlist]                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase 4: Interactive Tools & Features

#### 4A: Selection & Exploration Tools

```python
# User can select points via:
- Lasso selection (free-form polygon)
- Rectangle selection (drag corner-to-corner)
- Cluster filter (click legend to show/hide)
- Distance selection (click + radius)
- By metadata (search artist, filter genre, etc.)

# Selection actions:
- Create playlist from selection
- Export as CSV
- Merge clusters
- Re-assign to different cluster
- View details panel
```

#### 4B: Hover Tooltips

```
When hovering over point:
┌──────────────────────────────┐
│ 🎵 Track Details             │
├──────────────────────────────┤
│ Artist:    The Chemical Bros │
│ Title:     Block Rockin Beats│
│ Album:     Dig Your Own Hole │
│ Cluster:   Cluster 3 (Techno)│
│ BPM:       130               │
│ Duration:  4:32              │
│ Genres:    Techno, Electronic│
│ [▶ Play] [→ Details]         │
└──────────────────────────────┘
```

#### 4C: Interactive Parameter Tuning

```
Live parameter adjustment (for supported algorithms):

K-Means:
  K (clusters): [8 ←────|────→ 20]  (slider)
  [↺ Re-cluster] [Show deltas]

HDBSCAN:
  Min points: [5 ←────|────→ 50]
  Min cluster size: [2 ←────|────→ 20]
  [↺ Re-cluster] [Show impact]

Results update in real-time or on-demand
```

#### 4D: Dimensionality Reduction Selector

```
Current view: 2D (Features 0-1)

Choose dimensions:
⊙ First 2 PCA components (explains 89% variance)
⊙ t-SNE projection (2D, computed from all features)
⊙ UMAP projection (2D, computed from all features)
⊙ Manual: X-axis: [Tempo▼] Y-axis: [MFCC avg▼]

[Compute] [Save as view]
```

### Phase 5: Cluster Quality Metrics & Reports

```
┌─────────────────────────────────────────────────────────┐
│ Cluster Quality Report                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ Silhouette Score:     0.42 (fair cohesion)             │
│ Davies-Bouldin Index: 1.23 (lower is better)           │
│ Calinski-Harabasz:    145.7 (higher is better)         │
│ Within-cluster SSE:   2341.5                           │
│ Between-cluster SSE:  1890.3                           │
│                                                         │
│ Cluster-by-cluster breakdown:                          │
│ ┌─────────────────────────────────────────────────┐   │
│ │ Cluster 0 (Techno):                             │   │
│ │   Size: 47 tracks                               │   │
│ │   Silhouette: 0.51 (good)                       │   │
│ │   Centroid distance: 4.2                        │   │
│ │   Avg distance to centroid: 1.3                 │   │
│ │   Genres: Techno (89%), Electronic (11%)        │   │
│ │   Tempo range: 120-135 BPM                      │   │
│ │   [View] [Edit] [Merge] [Delete]                │   │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
│ ┌─────────────────────────────────────────────────┐   │
│ │ Cluster 1 (Ambient):                            │   │
│ │   Size: 23 tracks                               │   │
│ │   Silhouette: 0.38 (fair)                       │   │
│ │   ...                                           │   │
│ └─────────────────────────────────────────────────┘   │
│                                                         │
│ [↺ Re-cluster] [Export Report] [Suggest improvements]  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Detailed Implementation Plan

### Phase 1: Qt Graph Widget (20 hours)

#### 1.1 Dependencies & Setup
- [ ] Add `pyqtgraph` to requirements.txt (high-performance Qt visualization)
- [ ] Add `scikit-learn` for quality metrics (if not already present)
- [ ] Add `scipy` for distance calculations (if not already present)

#### 1.2 Create `ScatterPlotWidget` (PyQtGraph-based)
**File:** `gui/widgets/scatter_plot_widget.py`

```python
class InteractiveScatterPlot(QtWidgets.QWidget):
    """High-performance scatter plot with interactive features."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use PyQtGraph for rendering
        self.plot_widget = pg.PlotWidget()
        self.scatter = None
        self._setup_interactions()

    def set_data(self, X, clusters, labels=None, colors=None):
        """Update plot data."""
        # X: (n_samples, 2) array (2D projection)
        # clusters: (n_samples,) array of cluster IDs
        # labels: (n_samples,) array of track names
        # colors: (n_samples,) array of RGB tuples
        pass

    def set_selection(self, indices):
        """Highlight selected points."""
        pass

    def on_points_selected(self, callback):
        """Register callback for lasso/rectangle selection."""
        pass

    def on_point_clicked(self, callback):
        """Register callback when point is clicked."""
        pass

    def on_point_hover(self, callback):
        """Register callback for hover tooltips."""
        pass

    def export_selection(self, format='csv'):
        """Export selected points."""
        pass
```

#### 1.3 Create `ClusterLegendWidget`
**File:** `gui/widgets/cluster_legend_widget.py`

- Checkboxes for each cluster (show/hide)
- Cluster color indicator
- Track count per cluster
- Genre summary
- Click to highlight all tracks in cluster

#### 1.4 Create `TrackDetailsPanel`
**File:** `gui/widgets/track_details_panel.py`

- Shows metadata when point hovered/clicked
- Album art (if available)
- Metadata: artist, title, album, BPM, genres
- Quick actions: play, create playlist, view full details

### Phase 2: Clustering Wizard Dialog (15 hours)

#### 2.1 Feature Selection Dialog
**File:** `gui/dialogs/feature_selector_dialog.py`

```python
class FeatureSelectorDialog(QtWidgets.QDialog):
    """Multi-step wizard for clustering configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_wizard()

    def get_features(self) -> dict:
        """Return selected features and options."""
        return {
            "tempo": self.tempo_cb.isChecked(),
            "mfcc": self.mfcc_cb.isChecked(),
            "chroma": self.chroma_cb.isChecked(),
            "spectral": self.spectral_cb.isChecked(),
            "energy": self.energy_cb.isChecked(),
            "onset_rate": self.onset_cb.isChecked(),
        }

    def get_normalization(self) -> str:
        """Return normalization method."""
        return self.norm_combo.currentText()

    def get_reduction_method(self) -> str:
        """Return dimensionality reduction method."""
        return self.reduction_combo.currentText()

    def get_algorithm_params(self) -> dict:
        """Return algorithm parameters."""
        return {
            "algorithm": self.algo_combo.currentText(),
            "k": self.k_spinbox.value(),
            "min_cluster_size": self.min_size_spinbox.value(),
            ...
        }
```

#### 2.2 Add Preset Configurations
- Genre presets (EDM, Hip-Hop, Jazz, Classical, etc.)
- Mood presets (High Energy, Relaxing, Energetic, etc.)
- Tempo presets (Fast, Medium, Slow)

#### 2.3 Feature Preview Panel
- Show sample feature values
- Visualization of feature distributions
- Correlation matrix between features

### Phase 3: Enhanced Clustered Workspace UI (15 hours)

#### 3.1 Redesign `ClusteredWorkspace`

```
┌─────────────────────────────────────────────────────┐
│ Clustered Playlists Workspace                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│ [📊 Quick Start] [⚙ Advanced] [📖 Help]           │
│                                                     │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Quick Start: Run with recommended defaults     │ │
│ │ Genre: [All▼]  Algorithm: [K-Means▼] K: [8]   │ │
│ │ Engine: [librosa▼]                            │ │
│ │ [📊 Run] [⚙ Advanced Options]                 │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Progress:                                       │ │
│ │ ████████░░░░░░░░░░ (45%)                       │ │
│ │ Extracting features: Track 152/500              │ │
│ │ Time remaining: ~2 min                          │ │
│ │                                                 │ │
│ │ [✕ Cancel]                                    │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Results:                                        │ │
│ │ ✓ Clustering complete (8 clusters, 2 outliers) │ │
│ │ ✓ Playlists created in Music/Playlists/        │ │
│ │ ✓ Quality metrics computed                     │ │
│ │ ✓ Graph data generated                         │ │
│ │                                                 │ │
│ │ [📊 View Graph] [📄 Quality Report] [🎵 Open]│ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘
```

#### 3.2 Add Feature Extraction Status
- Show which tracks are being processed
- Estimated time remaining
- Cancel button
- Detailed log of any errors

#### 3.3 Results Summary Panel
- Number of clusters created
- Number of tracks per cluster
- Quality metrics (Silhouette, Davies-Bouldin)
- Outliers count
- Links to: view graph, quality report, open in player

### Phase 4: Graph Workspace Redesign (10 hours)

#### 4.1 Replace `graph.py` with embedded visualization

```python
class GraphWorkspace(WorkspaceBase):
    """Interactive cluster visualization within app."""

    def _build_ui(self):
        # Toolbar
        toolbar = self._build_toolbar()
        self.content_layout.addLayout(toolbar)

        # Main layout: scatter plot + legend + details
        main_layout = QtWidgets.QHBoxLayout()

        # Left: scatter plot
        self.scatter = InteractiveScatterPlot()
        self.scatter.on_point_clicked.connect(self._on_point_clicked)
        self.scatter.on_point_hover.connect(self._on_hover)
        self.scatter.on_selection_changed.connect(self._on_selection_changed)
        main_layout.addWidget(self.scatter, 3)

        # Right: legend + details
        side_layout = QtWidgets.QVBoxLayout()
        self.legend = ClusterLegendWidget()
        self.legend.on_cluster_toggled.connect(self.scatter.toggle_cluster)
        side_layout.addWidget(self.legend)
        self.details = TrackDetailsPanel()
        side_layout.addWidget(self.details)
        main_layout.addLayout(side_layout, 1)

        self.content_layout.addLayout(main_layout)

        # Bottom: selection actions
        bottom_layout = self._build_bottom_actions()
        self.content_layout.addLayout(bottom_layout)
```

#### 4.2 Implement Toolbar Actions
- Zoom (fit, in, out)
- Pan mode
- Select mode (lasso, rectangle)
- Filters/display options
- Save image
- Generate report
- Create playlist from selection

#### 4.3 Load Cluster Data
- Auto-detect cluster data from library
- Show data status and modification time
- Allow refresh/reload
- Show warning if data is stale

### Phase 5: Quality Metrics & Reports (10 hours)

#### 5.1 Create `ClusterMetricsCalculator`
**File:** `cluster_metrics.py`

```python
class ClusterMetricsCalculator:
    """Compute cluster quality metrics."""

    @staticmethod
    def silhouette_score(X, labels):
        """Silhouette coefficient (-1 to 1, higher is better)."""
        pass

    @staticmethod
    def davies_bouldin_index(X, labels):
        """Davies-Bouldin index (lower is better)."""
        pass

    @staticmethod
    def calinski_harabasz_score(X, labels):
        """Calinski-Harabasz index (higher is better)."""
        pass

    @staticmethod
    def within_cluster_sse(X, labels, centers):
        """Within-cluster sum of squared errors."""
        pass

    @staticmethod
    def between_cluster_sse(centers, cluster_sizes):
        """Between-cluster sum of squared errors."""
        pass

    @staticmethod
    def per_cluster_metrics(X, labels):
        """Compute metrics for each cluster."""
        pass
```

#### 5.2 Create `ClusterQualityReport`
**File:** `gui/dialogs/cluster_quality_dialog.py`

- Show overall metrics
- Show per-cluster metrics
- Suggestions for improvement
- Export as PDF/HTML

#### 5.3 Create `SuggestionEngine`
- "These clusters are too small, consider merging"
- "Silhouette score is low, try different K value"
- "Try dimensionality reduction for better separation"
- "Consider removing outlier cluster"

### Phase 6: Backend: Dimensionality Reduction (8 hours)

#### 6.1 PCA Reduction
```python
def reduce_pca(X, n_components=2):
    """PCA reduction for visualization."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    return X_reduced, explained_var, pca
```

#### 6.2 t-SNE Reduction
```python
def reduce_tsne(X, n_components=2):
    """t-SNE for better cluster separation visualization."""
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, random_state=42)
    X_reduced = tsne.fit_transform(X)
    return X_reduced
```

#### 6.3 UMAP Reduction
```python
def reduce_umap(X, n_components=2, n_neighbors=15):
    """UMAP: fast, preserves both local and global structure."""
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)
    X_reduced = reducer.fit_transform(X)
    return X_reduced
```

#### 6.4 Caching Computed Projections
- Store PCA/t-SNE/UMAP in library cache
- Avoid recomputing on workspace reload
- Version tracking (invalidate if parameters change)

### Phase 7: Interactive Features (12 hours)

#### 7.1 Point Selection Tools

```python
class SelectionManager:
    """Handle different selection modes."""

    def lasso_select(self, polygon_points):
        """Select points inside polygon."""
        pass

    def rectangle_select(self, x1, y1, x2, y2):
        """Select points in rectangle."""
        pass

    def cluster_select(self, cluster_id):
        """Select entire cluster."""
        pass

    def distance_select(self, center, radius):
        """Select points within radius."""
        pass

    def metadata_filter(self, genre=None, artist=None, tempo_range=None):
        """Filter by metadata."""
        pass

    def clear_selection(self):
        """Clear all selections."""
        pass
```

#### 7.2 Selection Actions

- **Create Playlist:** "Cool Tracks" from 42 selected points
- **Export CSV:** Download metadata of selected tracks
- **View Details:** Open details panel for each selected track
- **Merge Clusters:** Combine cluster A & B
- **Re-assign:** Move selected tracks to different cluster
- **Create Subcluster:** Cluster selected tracks separately
- **Play:** Queue selected tracks

#### 7.3 Visualization Options

```python
class VisualizationOptions:
    """Control how points are displayed."""

    size_by = {
        "cluster": fixed_size,
        "popularity": play_count,
        "bpm": tempo_value,
        "duration": track_length,
    }

    color_by = {
        "cluster": cluster_id,
        "genre": primary_genre,
        "artist": artist_hash,
        "quality": bitrate,
        "selection": is_selected,
    }

    shape_by = {
        "cluster": fixed_circle,
        "is_favorite": heart_vs_circle,
    }
```

### Phase 8: Backend Integration & Cleanup (10 hours)

#### 8.1 Update `clustered_playlists.py`
- Add output of feature vectors (for visualization)
- Add cluster metadata (size, genres, tempo range)
- Add cluster centers (for quality metrics)
- Return all needed data for visualization

#### 8.2 Update `cluster_graph_panel.py`
- Keep existing tkinter code for compatibility
- Mark as deprecated, reference new Qt implementation
- Document migration path

#### 8.3 Create `cluster_visualization_backend.py`
- Unified backend for loading cluster data
- Handles PCA/t-SNE/UMAP computation
- Caching layer
- Version management

#### 8.4 Update Main Workspace Navigation
- Link from Clustered → Graph automatically
- Pass cluster data through workspace context
- Handle missing data gracefully

---

## Implementation Checklist

### Phase 1: Qt Graph Widget (20 hrs)
- [ ] Add PyQtGraph to requirements
- [ ] Create InteractiveScatterPlot widget
- [ ] Implement point selection (lasso, rectangle)
- [ ] Implement hover tooltips with metadata
- [ ] Implement zoom/pan controls
- [ ] Create ClusterLegendWidget
- [ ] Create TrackDetailsPanel
- [ ] Write unit tests

### Phase 2: Clustering Wizard (15 hrs)
- [ ] Create multi-step wizard dialog
- [ ] Feature selection step
- [ ] Normalization selector
- [ ] Algorithm parameter configuration
- [ ] Post-processing options
- [ ] Output preferences
- [ ] Add preset configurations
- [ ] Write tests

### Phase 3: Enhanced Clustered Workspace (15 hrs)
- [ ] Redesign UI layout
- [ ] Add quick start mode
- [ ] Add advanced options button
- [ ] Feature extraction progress display
- [ ] Results summary panel
- [ ] Quality metrics computation
- [ ] Link to graph/report

### Phase 4: Graph Workspace Redesign (10 hrs)
- [ ] Replace placeholder with real visualization
- [ ] Add toolbar with actions
- [ ] Load cluster data on startup
- [ ] Show data status
- [ ] Implement dimension selector
- [ ] Add filter/display options

### Phase 5: Quality Metrics & Reports (10 hrs)
- [ ] Implement metrics calculator
- [ ] Create quality report dialog
- [ ] Add per-cluster metrics view
- [ ] Implement suggestion engine
- [ ] Export report functionality

### Phase 6: Dimensionality Reduction (8 hrs)
- [ ] Implement PCA reduction
- [ ] Implement t-SNE reduction
- [ ] Implement UMAP reduction (optional)
- [ ] Caching layer
- [ ] Performance optimization

### Phase 7: Interactive Features (12 hrs)
- [ ] Selection tools (lasso, rectangle, cluster, distance)
- [ ] Metadata filtering
- [ ] Create playlist from selection
- [ ] Export selection as CSV
- [ ] Re-assign cluster functionality
- [ ] Visualization options (size/color/shape by)

### Phase 8: Integration & Polish (10 hrs)
- [ ] Update clustered_playlists.py for new backend
- [ ] Update cluster_graph_panel.py documentation
- [ ] Create cluster_visualization_backend.py
- [ ] Update main workspace navigation
- [ ] Comprehensive testing
- [ ] Documentation & user guide

---

## Architecture Overview

```
GUI Layer (Qt)
├── gui/workspaces/clustered.py (Enhanced UI + wizard)
├── gui/workspaces/graph.py (Embedded interactive graph)
├── gui/widgets/scatter_plot_widget.py (PyQtGraph visualization)
├── gui/widgets/cluster_legend_widget.py (Cluster selector)
├── gui/widgets/track_details_panel.py (Metadata display)
├── gui/dialogs/feature_selector_dialog.py (Multi-step wizard)
└── gui/dialogs/cluster_quality_dialog.py (Quality report)

Backend Layer (Business Logic)
├── clustered_playlists.py (Enhanced to output feature vectors)
├── cluster_visualization_backend.py (Unified visualization backend)
├── cluster_metrics.py (Quality metric calculations)
├── cluster_reduction.py (PCA/t-SNE/UMAP reducers)
└── cluster_suggestion_engine.py (Quality improvement suggestions)

Legacy Layer (For Compatibility)
└── cluster_graph_panel.py (Marked deprecated, kept for reference)
```

---

## Data Flow (New Architecture)

```
User runs clustering
    ↓
ClusterWorker calls generate_clustered_playlists()
    ↓
generate_clustered_playlists():
  - Extracts features (returns X, labels, metadata)
  - Computes PCA projection for visualization
  - Saves cluster metadata (sizes, genres, centers)
  - Creates playlists
    ↓
Results in /Library/Clusters/
  - features.npy (feature vectors)
  - labels.npy (cluster assignments)
  - metadata.json (cluster info)
  - pca_projection.npy (2D for visualization)
  - playlists/ (m3u files)
    ↓
User opens Visual Music Graph
    ↓
GraphWorkspace loads cluster data
    ↓
cluster_visualization_backend loads and caches data
    ↓
InteractiveScatterPlot renders data with PyQtGraph
    ↓
User interacts: hover, click, select, filter
    ↓
Actions: create playlist, export, merge clusters, etc.
```

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Graph rendering | **PyQtGraph** | Native Qt, high performance, interactive |
| Dimensionality reduction | **scikit-learn** | Proven, well-tested (PCA), supports UMAP |
| Advanced clustering metrics | **scikit-learn** | Industry standard |
| UMAP (optional) | **umap-learn** | Better preservation of global structure |
| Caching | **joblib** or **pickle** | Simple, built-in to scikit-learn |

---

## Success Criteria

- ✅ Interactive graph renders smoothly with 1000+ points
- ✅ All selection modes work: lasso, rectangle, cluster, distance, metadata
- ✅ Hover tooltips show instantly (< 100ms)
- ✅ Zoom/pan responsive and smooth
- ✅ Clustering parameters explainable to users
- ✅ Quality metrics shown for each cluster
- ✅ Create playlist from graph selection works
- ✅ All features stay in-app (no external HTML)
- ✅ Can switch between PCA/t-SNE/UMAP projections
- ✅ Suggestions engine provides actionable feedback

---

## Estimated Timeline

- **Phase 1 (Qt Widget):** 20 hours → 📊 Core visualization
- **Phase 2 (Wizard):** 15 hours → 🧙 User prompts
- **Phase 3 (Enhanced UI):** 15 hours → 🎨 Better UX
- **Phase 4 (Graph workspace):** 10 hours → 📈 Embedded graphs
- **Phase 5 (Metrics):** 10 hours → 📊 Quality feedback
- **Phase 6 (Dimensionality):** 8 hours → 🔍 Different views
- **Phase 7 (Interactive):** 12 hours → ⚙️ Tools & actions
- **Phase 8 (Integration):** 10 hours → ✅ Polish & test

**Total: 100 hours (2-3 weeks at 35-40 hrs/week)**

---

## Recommended Build Sequence

1. **Start with Phase 1** (Qt Widget) - foundation for everything
2. **Parallel: Phase 2** (Wizard) - doesn't depend on Phase 1
3. **Phase 3** (Enhanced UI) - ties together 1 & 2
4. **Phase 4** (Graph workspace) - integrates Phase 1
5. **Phase 5** (Metrics) - backend only, can run in parallel
6. **Phase 6** (Dimensionality) - improves Phase 1
7. **Phase 7** (Interactive) - advanced Phase 1 features
8. **Phase 8** (Integration) - final polish

Can start Phases 1, 2, 5 simultaneously for speed.

---

## Open Questions

1. Should we keep tkinter cluster_graph_panel.py for backward compatibility?
   - Recommendation: Mark deprecated, keep for reference only

2. Should we support UMAP for better visualization?
   - Recommendation: Yes, as optional feature (requires umap-learn)

3. Should PCA/t-SNE be computed on-demand or pre-computed?
   - Recommendation: Pre-compute during clustering, cache results

4. Should users be able to manually edit cluster assignments?
   - Recommendation: Yes, as advanced feature (merges, re-assignment)

5. Should we generate a PDF report or just HTML/on-screen?
   - Recommendation: HTML on-screen + export as PDF if requested

