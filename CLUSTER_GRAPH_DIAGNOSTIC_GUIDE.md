# 3D Cluster Graph — Diagnostic & Testing Guide

## Quick Start — Test the Visualization Without Real Data

If you're seeing a blank screen when opening the 3D graph, use the demo tools to verify the Three.js visualization is working:

### Option 1: Test Button in GUI (Easiest)

**Qt GUI (alpha_dex_gui.py):**
1. Click **Music Graph** in the sidebar
2. Click **"Test with Demo Data"** button
3. A browser window opens with 10 colored points in 3D space

**Tkinter GUI (main_gui.py):**
1. Go to **Playlist Creator** tab
2. Select any clustering method (KMeans or HDBSCAN)
3. Click **"Test 3D (Demo)"** button
4. A browser window opens with 10 colored points in 3D space

### Option 2: Command Line

```bash
python test_cluster_graph_demo.py
```

Creates `Docs/cluster_graph_demo.html` with 10 random test points and opens it in your browser.

---

## Understanding the Issue — Why You See Nothing

The most likely causes of a blank 3D graph:

### 1. **No Cluster Data Generated Yet**
- The cluster graph HTML requires `Docs/cluster_info.json`
- This is only created AFTER running **Clustered Playlists**
- **Solution:** Go to Clustered Playlists workspace → Run clustering → Try again

### 2. **HTML File Not Generated**
- Even if `cluster_info.json` exists, `cluster_graph.html` must be created
- This should happen automatically during clustering
- **Solution:** Use **"Regenerate HTML"** button, or run `python diagnose_cluster_graph.py <library_path>`

### 3. **Browser JavaScript Error**
- The Three.js library might fail to load or render
- **Solution:**
  - Press **F12** in browser to open Developer Console
  - Look for red error messages
  - Check that `three.min.js` CDN is accessible

### 4. **Invalid Data Structure**
- `cluster_info.json` might be corrupted or missing required fields
- **Solution:** Run diagnostic tool (see below)

---

## Diagnostic Tools

### Command-Line Diagnostic

```bash
python diagnose_cluster_graph.py ~/Music
```

This validates:
- ✓ Library and Docs folders exist
- ✓ `cluster_info.json` is readable JSON
- ✓ Required keys present (X_3d, labels, tracks)
- ✓ Data dimensions consistent
- ✓ Clustering summary (number of clusters, noise points)
- ✓ HTML file generated and contains Three.js code

**Output Example:**
```
======================================================================
AlphaDEX Cluster Graph Diagnostic Tool
======================================================================

1. Library path: /home/user/Music
   ✓ Library directory exists

2. Docs folder: /home/user/Music/Docs
   ✓ Docs directory exists

3. Cluster info file: /home/user/Music/Docs/cluster_info.json
   ✓ cluster_info.json exists
   ✓ JSON is valid

4. Data structure validation
   ✓ All required keys present

5. Data dimensions
   - X_3d points: 5432
   - Labels: 5432
   - Tracks: 5432
   ✓ All dimensions consistent

6. Clustering summary
   - Clusters: 8
   - Noise points: 127
   - Cluster IDs: [0, 1, 2, 3, 4, 5, 6, 7]

7. Sample data (first 3 points)
   [0] Cluster  0 | (   5.58,  -19.00,   -9.00) | /Music/Artist/track.mp3
   [1] Cluster  1 | ( -11.07,    9.46,    7.07) | /Music/Artist/another.mp3
   [2] Cluster  0 | (  15.69,  -16.52,   -3.12) | /Music/Artist/third.mp3

8. HTML visualization file: /home/user/Music/Docs/cluster_graph.html
   ✓ cluster_graph.html exists (27.3 KB)

9. HTML content validation
   ✓ DOCTYPE
   ✓ Three.js CDN
   ✓ Scene container
   ✓ Data embedded

10. Recommendations
   ✓ Everything looks good!
   → Try opening in a modern browser (Chrome, Firefox, Safari)
   → File: /home/user/Music/Docs/cluster_graph.html
   → Press F12 in browser to check console for errors

======================================================================
```

---

## The 3D Visualization — What You Should See

If everything works correctly:

### Visual Elements
- **Dark background** — "space" aesthetic
- **Colored points** — Each point is a song; colors represent clusters
- **Rotating grid** — Ground reference plane
- **Axis lines** — Show X (red), Y (green), Z (blue) directions

### Controls
| Action | Control |
|--------|---------|
| Rotate view | Click + Drag mouse |
| Zoom | Scroll wheel |
| Pan | Right-click + Drag |
| View presets | Buttons at bottom: XY, XZ, YZ, 3D |
| Reset camera | "Reset View" button |

### Interaction
- **Hover over points** — See song title, artist, cluster ID
- **Click points** — Select individual tracks
- **Shift-click** — Multi-select for playlists
- **Export selected** — CSV or M3U buttons at bottom

### Data Display
- **Top-left HUD** — Shows track count, cluster count
- **Top-right Legend** — Click cluster colors to hide/show
- **Bottom selection bar** — Shows selected count and export options

---

## Data Flow — How It All Connects

```
┌─────────────────────────────────────┐
│  Run Clustered Playlists            │
│  (Extract audio features, cluster)  │
└────────────────┬────────────────────┘
                 │
                 ├─→ Extract librosa features (MFCC, tempo, etc.)
                 ├─→ Normalize features
                 ├─→ Run clustering (KMeans or HDBSCAN)
                 ├─→ Compute 3D embedding (UMAP or t-SNE)
                 │
                 v
┌─────────────────────────────────────┐
│  Docs/cluster_info.json             │
│  (Contains X_3d, labels, tracks)    │
└────────────────┬────────────────────┘
                 │
                 v
┌─────────────────────────────────────┐
│  cluster_graph_3d.py                │
│  (Generates HTML from JSON)         │
└────────────────┬────────────────────┘
                 │
                 v
┌─────────────────────────────────────┐
│  Docs/cluster_graph.html            │
│  (Self-contained Three.js app)      │
└────────────────┬────────────────────┘
                 │
                 v
┌─────────────────────────────────────┐
│  Browser (Chrome, Firefox, Safari)  │
│  Renders 3D visualization           │
└─────────────────────────────────────┘
```

---

## Browser Console Debugging

If you see nothing or errors, check the browser console:

1. Open the generated HTML file in your browser
2. Press **F12** (or Cmd+Option+I on Mac) to open Developer Tools
3. Click **Console** tab
4. Look for red error messages

### Common Issues & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `CLUSTER_DATA is null` | No data embedded | Regenerate HTML |
| `THREE is not defined` | Three.js didn't load | Check internet connection, CDN access |
| `Can't read property 'X_3d' of null` | JSON parsing failed | Check cluster_info.json format |
| `WebGL context lost` | Graphics card issue | Try different browser, update drivers |

---

## Files Generated

### By Clustered Playlists Workspace
- **`Docs/cluster_info.json`** — The core data file (JSON)
  - Contains: 3D coordinates, cluster labels, track paths, metadata

### By cluster_graph_3d.py
- **`Docs/cluster_graph.html`** — The visualization (self-contained HTML)
  - Size: ~27 KB (includes Three.js, CSS, JavaScript, embedded data)
  - No external dependencies except Three.js CDN

### Demo Files
- **`Docs/cluster_graph_demo.html`** — Test visualization (10 random points)

---

## Performance Notes

- **Small libraries (< 5,000 tracks)** — Instant visualization
- **Large libraries (5,000–50,000 tracks)** — Still smooth; takes a few seconds to render
- **Very large libraries (> 50,000 tracks)** — Downsamples to 5,000 points for performance; all data kept in labels

---

## Troubleshooting Checklist

- [ ] Run Clustered Playlists first (creates cluster_info.json)
- [ ] Check that `Docs/cluster_info.json` exists and is readable
- [ ] Use `python diagnose_cluster_graph.py` to validate data
- [ ] Try the demo: "Test 3D (Demo)" button
- [ ] Open in a modern browser (Chrome, Firefox, Safari)
- [ ] Press F12 to check for JavaScript errors
- [ ] Check internet connection (Three.js loads from CDN)
- [ ] If on a corporate network, whitelist `cdnjs.cloudflare.com`

---

## Need Help?

1. **Visualization not appearing?** → Run diagnostic tool
2. **Cluster data looks wrong?** → Check Clustered Playlists settings
3. **Browser errors?** → Check console (F12) for specific error messages
4. **Performance issues?** → Library might be too large; try filtering to subset of tracks

---

## Technical Details

### Data Format (cluster_info.json)

```json
{
  "X_3d": [[x1, y1, z1], [x2, y2, z2], ...],
  "labels": [0, 1, 0, 2, -1, ...],
  "tracks": ["/path/to/track1.mp3", ...],
  "metadata": [{"title": "...", "artist": "..."}, ...],
  "cluster_info": {
    "0": {"size": 234},
    "1": {"size": 156},
    ...
  },
  "X_downsampled": false,
  "X_total_points": 1000
}
```

### 3D Coordinates
- **X_3d** contains normalized 3D coordinates from audio feature embedding
- Values are typically in range [-50, 50]
- Spatial distance ≈ audio similarity (librosa features)
- Generated by UMAP or t-SNE dimensionality reduction

### Cluster Labels
- **0, 1, 2, ...** = Regular clusters
- **-1** = Noise points (HDBSCAN only)

---

## See Also

- `cluster_graph_3d.py` — HTML generation module
- `CLAUDE.md` — Project architecture and guidelines
