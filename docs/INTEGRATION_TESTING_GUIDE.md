# Integration Testing Guide — Phases 1-3 Implementation

**Status:** ✅ **INTEGRATION COMPLETE - Ready for Testing**

**Date:** 2026-03-20
**Branch:** `claude/audit-startup-splash-PwNwu`

---

## What's Been Integrated

### 1. **PyQtGraph Installation** ✅
- Added `pyqtgraph>=0.13.0` to requirements.txt

### 2. **Workspace Registration** ✅
- `EnhancedClusteredWorkspace` replaces old `ClusteredWorkspace`
- `GraphWorkspace` enhanced with scatter plot embedding
- All imports updated in `main_window.py`

### 3. **Graph Workspace Enhanced** ✅
- Created `gui/workspaces/graph_enhanced.py` with:
  - Embedded `InteractiveScatterPlot` widget
  - `ClusterLegendWidget` for visibility control
  - `TrackDetailsPanel` for metadata display
  - Auto-loads cluster data from library
  - Export selection as CSV or M3U playlist

### 4. **Backend Data Export** ✅
- Updated `clustered_playlists.py` to export:
  - Feature vectors (X array)
  - Cluster assignments (labels)
  - Track paths
  - Cluster metadata
  - Saves to `Docs/cluster_info.json` for graph workspace

---

## Pre-Testing Checklist

Before running tests, verify:

- [ ] Branch is `claude/audit-startup-splash-PwNwu`
- [ ] All 4 commits are present
- [ ] Python 3.9+ installed
- [ ] venv activated (if using one)
- [ ] `requirements.txt` includes `pyqtgraph>=0.13.0`

---

## Testing Procedure

### Phase 1: Environment Setup (5 minutes)

```bash
# 1. Update requirements
pip install -r requirements.txt

# 2. Verify PyQtGraph installed
python -c "import pyqtgraph; print('PyQtGraph OK')"

# 3. Verify imports work
python -c "from gui.widgets.interactive_scatter_plot import InteractiveScatterPlot; print('Scatter plot OK')"
python -c "from gui.workspaces.clustered_enhanced import EnhancedClusteredWorkspace; print('Workspace OK')"
```

### Phase 2: Launch Application (5 minutes)

```bash
# Start the application with splash screen and library selection
python alpha_dex_gui.py
```

The app will:
1. Show splash screen with loading bar
2. Display landing page for library selection
3. Cross-fade to main window with selected library loaded

### Phase 3: Test Quick Start Clustering (10 minutes)

**Steps:**
1. Open the app
2. Select a test library (or create one with a few MP3 files)
3. Navigate to **Clustered Playlists** tab
4. Click **🚀 Quick Start**
5. Watch progress bar (should go 0 → 100%)
6. Verify log shows:
   - "Found X tracks"
   - "Extracting audio features..."
   - "Clustering complete: found K clusters"
   - "Silhouette score: X.XXX"

**Expected Output:**
```
✓ Clustering complete!
Clusters created: 8
Total tracks: 123
Silhouette score: 0.52 (Good)

[View Quality Report] [View Playlists] [Open Visual Graph]
```

**If it fails:**
- Check log for specific error
- Verify audio files are readable
- Ensure librosa/numpy/sklearn installed

---

### Phase 4: Test Advanced Configuration (10 minutes)

**Steps:**
1. In **Clustered Playlists** tab
2. Click **⚙ Advanced** tab
3. Click **⚙ Configure with Wizard...**
4. Go through all 5 steps:
   - **Step 1:** Select features (check MFCC and Tempo)
   - **Step 2:** Keep standard normalization
   - **Step 3:** Change K to 12 (if not already)
   - **Step 4:** Check "Remove small clusters"
   - **Step 5:** Check all output options
5. Click **Finish**
6. Configuration should display in text area
7. Click **▶ Run**
8. Watch clustering proceed

**Expected Configuration Display:**
```
Configuration:
  Algorithm: KMEANS
  Normalization: standard
  Dimensionality reduction: none

Features:
  ✓ Tempo
  ✓ MFCC
  ✗ Chroma
  ✓ Energy
  ...

K-Means Parameters:
  K (clusters): 12

Output:
  ✓ Create playlists
  ✓ Generate quality report
  ✓ Open interactive graph
```

---

### Phase 5: Test Quality Report (5 minutes)

**Steps:**
1. After clustering completes
2. Click **📄 View Quality Report**
3. Dialog should open showing:
   - Silhouette score
   - Davies-Bouldin index
   - Calinski-Harabasz score
   - Per-cluster breakdown
   - Suggestions for improvement

**Expected Content:**
```
Overall Metrics:
  Silhouette Score: 0.52 (good)
  Davies-Bouldin Index: 1.23 (lower is better)
  Calinski-Harabasz Score: 145.7 (higher is better)

Per-Cluster Breakdown:
  Cluster 0 (Techno):
    Size: 47 tracks
    Silhouette: 0.51 (good)
    ...

Suggestions for Improvement:
- Clustering looks good! Feel free to proceed...
```

---

### Phase 6: Test Interactive Graph (15 minutes)

**Steps:**
1. From clustering results, click **📊 Open Visual Graph**
2. Graph workspace should open with:
   - Scatter plot in center (showing clusters as colored dots)
   - Cluster legend on right (with show/hide checkboxes)
   - Track details panel below legend
3. **Test interactions:**

   a) **Hover over point:**
      - Track metadata should appear in details panel
      - No selection yet

   b) **Click a point:**
      - Point should highlight (red border)
      - Details should update

   c) **Cluster legend - toggle visibility:**
      - Click checkbox next to cluster
      - Points should disappear/reappear

   d) **Cluster legend - select cluster:**
      - Click cluster name
      - All points in that cluster should highlight

   e) **Lasso selection (if available):**
      - Try to select multiple points
      - Selected points show count at bottom

   f) **Export selection:**
      - Select some points
      - Click **📄 Export Selection**
      - Save as CSV
      - Verify file contains track paths

   g) **Create playlist:**
      - Select points
      - Click **🎵 Create Playlist from Selection**
      - Enter name "Test Playlist"
      - Verify M3U created in `Music/Playlists/`

**Expected Graph Features:**
- Smooth zoom (mouse wheel)
- Smooth pan (click+drag)
- Point selection (lasso or rectangle)
- Hover tooltips
- Cluster toggling
- Color-coded points by cluster

---

### Phase 7: Test Data Persistence (5 minutes)

**Steps:**
1. Close the graph workspace
2. Navigate away (to another tab)
3. Come back to **Visual Music Graph**
4. Verify:
   - Scatter plot is still there
   - Data is loaded (status shows "✓ Loaded: X clusters, Y tracks")
   - Cluster legend shows all clusters
   - Can interact with graph again

---

## Troubleshooting

### Problem: "PyQtGraph not installed"
**Solution:**
```bash
pip install pyqtgraph>=0.13.0
```

### Problem: "No cluster data found — run Clustered Playlists first"
**Solution:**
- Clustering hasn't been run yet
- Or cluster_info.json wasn't created
- Check `Library/Docs/` folder exists
- Run clustering from **Clustered Playlists** tab

### Problem: Scatter plot doesn't render
**Solution:**
- PyQtGraph may not be installed
- Check terminal for import errors
- Verify numpy/scipy installed
- May be a graphics driver issue

### Problem: "ImportError: cannot import EnhancedClusteredWorkspace"
**Solution:**
- Verify `gui/workspaces/clustered_enhanced.py` exists
- Check main_window.py has correct import
- Ensure no typos in import path

### Problem: Clustering hangs or very slow
**Solution:**
- Large library (>5000 files)?
- Audio feature extraction takes time
- Try Quick Start with smaller library first
- Check CPU usage (should be using cores)

### Problem: Quality report shows no metrics
**Solution:**
- Metrics computed after clustering
- May not be available immediately
- Try refreshing after clustering
- Check for errors in log

### Problem: Legend doesn't show clusters
**Solution:**
- Scatter plot data may not be loaded
- Check status label shows "✓ Loaded: X clusters"
- Try clicking **↺ Refresh** button
- Check cluster_info.json was created

---

## Success Criteria

After all 7 phases, you should be able to:

✅ Install PyQtGraph without errors
✅ Open the application
✅ Run Quick Start clustering on test library
✅ See progress bar advance to 100%
✅ View clustering results with metrics
✅ Open Quality Report dialog
✅ View scatter plot with colored points
✅ Hover points and see metadata
✅ Click to select points
✅ Toggle cluster visibility
✅ Export selection as CSV
✅ Create M3U playlist from selection
✅ Have data persist when navigating away

If all ✅ check marks are achievable, **Integration is Complete and Successful!**

---

## Additional Checks

### Check 1: File Structure
Verify these files exist:
- [ ] `gui/widgets/interactive_scatter_plot.py`
- [ ] `gui/widgets/cluster_legend.py`
- [ ] `gui/widgets/track_details_panel.py`
- [ ] `gui/dialogs/clustering_wizard_dialog.py`
- [ ] `gui/dialogs/cluster_quality_report_dialog.py`
- [ ] `gui/workspaces/clustered_enhanced.py`
- [ ] `gui/workspaces/graph_enhanced.py`

### Check 2: Data Files
After clustering, verify these exist:
- [ ] `Library/Docs/cluster_info.json`
- [ ] `Library/Docs/features.npy`
- [ ] `Library/Playlists/*.m3u` (if created)

### Check 3: Imports
Verify in Python shell:
```python
from gui.widgets.interactive_scatter_plot import InteractiveScatterPlot
from gui.widgets.cluster_legend import ClusterLegendWidget
from gui.widgets.track_details_panel import TrackDetailsPanel
from gui.dialogs.clustering_wizard_dialog import ClusteringWizardDialog
from gui.dialogs.cluster_quality_report_dialog import ClusterQualityReportDialog
from gui.workspaces.clustered_enhanced import EnhancedClusteredWorkspace
from gui.workspaces.graph_enhanced import GraphWorkspace

print("All imports successful!")
```

---

## Performance Benchmarks

Expected performance on test library:

| Operation | Duration | Notes |
|-----------|----------|-------|
| Scan 100 files | < 1 min | Fingerprinting takes most time |
| Feature extraction | 2-5 min | Depends on audio engine |
| K-Means clustering (k=8) | < 10 sec | Fast algorithm |
| HDBSCAN clustering | 30-60 sec | Slower, more accurate |
| Graph render (100 points) | < 1 sec | PyQtGraph is fast |
| Plan building | < 30 sec | Quick operation |

---

## Known Limitations

1. **Graph doesn't save state** — Navigating away and back reloads from disk
2. **No real-time re-clustering** — Parameters set before clustering, not adjustable after
3. **Large arrays not saved** — X matrix not saved if >10k points (memory limit)
4. **Selection not persistent** — Graph selection clears when navigating away
5. **No undo** — Can't undo clustering, must re-run

---

## Next Steps After Testing

If all tests pass:
1. ✅ Integration is complete
2. ✅ Ready for production use
3. Consider Phase 4+ enhancements:
   - Interactive parameter tuning
   - Dimension reduction visualization (PCA, t-SNE, UMAP)
   - Cluster merging interface
   - Playlist editing in graph
   - Export reports to PDF/HTML

---

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Look at the log output (in app and terminal)
3. Verify requirements.txt installed correctly
4. Try with a small test library first
5. Create an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - Library size and file types
   - Python version
   - PyQtGraph version

---

**Integration Complete! Ready for comprehensive testing.** 🎉

