"""Generate a standalone Three.js 3D cluster graph HTML page.

This module produces a self-contained HTML file that visualises
``cluster_info.json`` data as an interactive 3D scatter plot.  The HTML
embeds Three.js from a CDN and needs **no** local build step.

Usage from other modules::

    from cluster_graph_3d import generate_cluster_graph_html

    # After clustering writes cluster_info.json:
    html_path = generate_cluster_graph_html(library_path)

The generated file is written to ``<library>/Docs/cluster_graph.html``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_cluster_graph_html(
    library_path: str,
    cluster_info_path: str | None = None,
    output_path: str | None = None,
    log_callback=None,
) -> str:
    """Generate a Three.js 3D cluster visualisation HTML file.

    Parameters
    ----------
    library_path:
        Root of the music library.  ``Docs/cluster_info.json`` is read
        from here when *cluster_info_path* is ``None``.
    cluster_info_path:
        Explicit path to the cluster JSON.  Falls back to
        ``<library_path>/Docs/cluster_info.json``.
    output_path:
        Where to write the HTML.  Falls back to
        ``<library_path>/Docs/cluster_graph.html``.
    log_callback:
        Optional ``(str) -> None`` logger.

    Returns
    -------
    str
        Absolute path of the generated HTML file.

    Raises
    ------
    FileNotFoundError
        If the cluster info JSON does not exist.
    """
    if log_callback is None:
        log_callback = lambda msg: None  # noqa: E731

    docs = os.path.join(library_path, "Docs")
    os.makedirs(docs, exist_ok=True)

    if cluster_info_path is None:
        cluster_info_path = os.path.join(docs, "cluster_info.json")

    if not os.path.isfile(cluster_info_path):
        raise FileNotFoundError(
            f"Cluster data not found at {cluster_info_path}.  "
            "Run Clustered Playlists first to generate it."
        )

    with open(cluster_info_path, "r", encoding="utf-8") as fh:
        cluster_data = json.load(fh)

    # Validate required keys
    _validate_cluster_data(cluster_data)

    if output_path is None:
        output_path = os.path.join(docs, "cluster_graph.html")

    html = _render_html(cluster_data)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    log_callback(f"✓ 3D cluster graph written to {output_path}")
    logger.info("Wrote cluster_graph.html to %s", output_path)
    return os.path.abspath(output_path)


def generate_cluster_graph_html_from_data(
    cluster_data: dict,
    output_path: str,
    log_callback=None,
) -> str:
    """Generate HTML directly from an in-memory cluster data dict.

    Parameters
    ----------
    cluster_data:
        Dict with keys ``X_3d``, ``labels``, ``tracks``, and optionally
        ``cluster_info`` and ``metadata``.
    output_path:
        Where to write the HTML file.
    log_callback:
        Optional logger.

    Returns
    -------
    str
        Absolute path of the generated HTML file.
    """
    if log_callback is None:
        log_callback = lambda msg: None  # noqa: E731

    _validate_cluster_data(cluster_data)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    html = _render_html(cluster_data)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    log_callback(f"✓ 3D cluster graph written to {output_path}")
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"X_3d", "labels", "tracks"}


def _validate_cluster_data(data: dict) -> None:
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(
            f"cluster_info.json is missing required keys: {missing}"
        )

    if not data["X_3d"]:
        raise ValueError("X_3d is empty — no points to visualise")

    if len(data["X_3d"]) != len(data["labels"]):
        raise ValueError(
            f"Length mismatch: X_3d has {len(data['X_3d'])} entries "
            f"but labels has {len(data['labels'])}"
        )


def _render_html(cluster_data: dict) -> str:
    """Return a complete HTML page string embedding *cluster_data* as JSON."""

    # Inline the cluster data so the HTML is truly self-contained.
    data_json = json.dumps(cluster_data, separators=(",", ":"))

    return _HTML_TEMPLATE.replace("/*__CLUSTER_DATA__*/", data_json)


# ---------------------------------------------------------------------------
# HTML + Three.js template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AlphaDEX — 3D Cluster Graph</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:100%;height:100%;overflow:hidden;background:#0a0a0f;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
#scene-container{position:absolute;top:0;left:0;width:100%;height:100%}

/* HUD overlay */
#hud{position:absolute;top:12px;left:12px;pointer-events:none;z-index:10}
#hud h1{font-size:16px;font-weight:600;color:#94a3b8;margin-bottom:4px}
#hud .stats{font-size:12px;color:#64748b}

/* Tooltip */
#tooltip{position:absolute;display:none;padding:8px 12px;background:rgba(15,23,42,.92);border:1px solid #334155;border-radius:6px;font-size:12px;pointer-events:none;z-index:20;max-width:280px;backdrop-filter:blur(4px)}
#tooltip .tt-title{font-weight:600;color:#f1f5f9;margin-bottom:2px}
#tooltip .tt-artist{color:#94a3b8}
#tooltip .tt-cluster{color:#64748b;font-size:11px;margin-top:2px}

/* Legend */
#legend{position:absolute;top:12px;right:12px;background:rgba(15,23,42,.85);border:1px solid #1e293b;border-radius:8px;padding:10px 14px;z-index:10;max-height:60vh;overflow-y:auto;min-width:140px;backdrop-filter:blur(4px)}
#legend h2{font-size:12px;color:#94a3b8;margin-bottom:6px;font-weight:500}
.legend-item{display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer;font-size:12px;user-select:none}
.legend-item:hover{color:#f8fafc}
.legend-swatch{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.legend-item.hidden{opacity:.35}

/* Bottom bar */
#bottom-bar{position:absolute;bottom:0;left:0;right:0;display:flex;align-items:center;justify-content:space-between;padding:8px 16px;background:rgba(15,23,42,.8);border-top:1px solid #1e293b;z-index:10;backdrop-filter:blur(4px)}
#bottom-bar .controls{display:flex;gap:8px}
#bottom-bar button{background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:4px;padding:4px 10px;font-size:11px;cursor:pointer;transition:background .15s}
#bottom-bar button:hover{background:#334155}
#bottom-bar button.active{background:#2563eb;border-color:#3b82f6;color:#fff}
#bottom-bar .help{font-size:11px;color:#475569}

/* Selection info */
#selection-bar{position:absolute;bottom:44px;left:50%;transform:translateX(-50%);display:none;background:rgba(30,41,59,.95);border:1px solid #334155;border-radius:8px;padding:8px 16px;z-index:15;backdrop-filter:blur(4px);text-align:center}
#selection-bar .sel-count{font-size:13px;font-weight:500}
#selection-bar .sel-actions{display:flex;gap:6px;margin-top:6px}
#selection-bar button{background:#1e293b;color:#cbd5e1;border:1px solid #334155;border-radius:4px;padding:4px 10px;font-size:11px;cursor:pointer}
#selection-bar button:hover{background:#334155}
#selection-bar button.primary{background:#2563eb;border-color:#3b82f6;color:#fff}

/* No data message */
#no-data{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;z-index:30}
#no-data h2{font-size:20px;margin-bottom:8px;color:#f1f5f9}
#no-data p{color:#64748b;font-size:14px}
</style>
</head>
<body>

<div id="scene-container"></div>

<div id="hud">
  <h1>AlphaDEX — 3D Cluster Graph</h1>
  <div class="stats" id="stats-text">Loading…</div>
</div>

<div id="tooltip">
  <div class="tt-title" id="tt-title"></div>
  <div class="tt-artist" id="tt-artist"></div>
  <div class="tt-cluster" id="tt-cluster"></div>
</div>

<div id="legend"><h2>Clusters</h2></div>

<div id="bottom-bar">
  <div class="controls">
    <button id="btn-reset" title="Reset camera">Reset View</button>
    <button id="btn-top" title="Top-down (XY)">XY</button>
    <button id="btn-front" title="Front (XZ)">XZ</button>
    <button id="btn-side" title="Side (YZ)">YZ</button>
    <button id="btn-select" title="Toggle selection mode">Select</button>
    <button id="btn-export-csv" title="Export selected tracks as CSV">Export CSV</button>
    <button id="btn-export-m3u" title="Export selected tracks as M3U playlist">Export M3U</button>
  </div>
  <div class="help">Drag to orbit · Scroll to zoom · Right-drag to pan · Click point to select</div>
</div>

<div id="selection-bar">
  <div class="sel-count" id="sel-count">0 tracks selected</div>
  <div class="sel-actions">
    <button onclick="clearSelection()">Clear</button>
    <button class="primary" onclick="exportSelectedCSV()">Export CSV</button>
    <button class="primary" onclick="exportSelectedM3U()">Export M3U</button>
  </div>
</div>

<div id="no-data" style="display:none">
  <h2>No cluster data found</h2>
  <p>Run <strong>Clustered Playlists</strong> first to generate cluster data.</p>
</div>

<!-- Three.js from CDN -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>

<script>
// ═══════════════════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════════════════
const CLUSTER_DATA = /*__CLUSTER_DATA__*/null;

if (!CLUSTER_DATA || !CLUSTER_DATA.X_3d || CLUSTER_DATA.X_3d.length === 0) {
  document.getElementById("no-data").style.display = "block";
  document.getElementById("hud").style.display = "none";
  document.getElementById("legend").style.display = "none";
  document.getElementById("bottom-bar").style.display = "none";
  throw new Error("No cluster data");
}

const positions = CLUSTER_DATA.X_3d;
const labels    = CLUSTER_DATA.labels;
const tracks    = CLUSTER_DATA.tracks;
const metadata  = CLUSTER_DATA.metadata || [];
const clusterInfo = CLUSTER_DATA.cluster_info || {};

const N = positions.length;

// ═══════════════════════════════════════════════════════════════════════════
// COLOUR PALETTE
// ═══════════════════════════════════════════════════════════════════════════
function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) { r = g = b = l; }
  else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1; if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1/3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1/3);
  }
  return [r, g, b];
}

const uniqueLabels = [...new Set(labels)].sort((a, b) => a - b);
const clusterColors = {};
const noiseColor = [0.4, 0.4, 0.4];

uniqueLabels.forEach((lbl, i) => {
  if (lbl < 0) { clusterColors[lbl] = noiseColor; return; }
  const hue = i / Math.max(uniqueLabels.filter(l => l >= 0).length, 1);
  clusterColors[lbl] = hslToRgb(hue, 0.85, 0.6);
});

// ═══════════════════════════════════════════════════════════════════════════
// SCENE SETUP
// ═══════════════════════════════════════════════════════════════════════════
const container = document.getElementById("scene-container");

const scene    = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
scene.fog = new THREE.FogExp2(0x0a0a0f, 0.003);

const camera   = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 5000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

// Ambient light for gentle fill
scene.add(new THREE.AmbientLight(0x404060, 0.5));

// ═══════════════════════════════════════════════════════════════════════════
// COMPUTE BOUNDS & NORMALISE
// ═══════════════════════════════════════════════════════════════════════════
let minX = Infinity, minY = Infinity, minZ = Infinity;
let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

for (let i = 0; i < N; i++) {
  const [x, y, z] = positions[i];
  if (x < minX) minX = x; if (x > maxX) maxX = x;
  if (y < minY) minY = y; if (y > maxY) maxY = y;
  if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
}

const rangeX = maxX - minX || 1;
const rangeY = maxY - minY || 1;
const rangeZ = maxZ - minZ || 1;
const maxRange = Math.max(rangeX, rangeY, rangeZ);
const SCALE = 100;  // Map to [-50, 50] cube

function normalise(x, y, z) {
  return [
    ((x - minX) / maxRange - 0.5 * rangeX / maxRange) * SCALE,
    ((y - minY) / maxRange - 0.5 * rangeY / maxRange) * SCALE,
    ((z - minZ) / maxRange - 0.5 * rangeZ / maxRange) * SCALE,
  ];
}

// ═══════════════════════════════════════════════════════════════════════════
// POINTS GEOMETRY
// ═══════════════════════════════════════════════════════════════════════════
const clusterVisibility = {};
uniqueLabels.forEach(l => { clusterVisibility[l] = true; });

const geometry = new THREE.BufferGeometry();
const posArr   = new Float32Array(N * 3);
const colArr   = new Float32Array(N * 3);
const sizeArr  = new Float32Array(N);

const BASE_SIZE = Math.max(1.5, Math.min(8, 400 / Math.sqrt(N)));

for (let i = 0; i < N; i++) {
  const [nx, ny, nz] = normalise(...positions[i]);
  posArr[i * 3]     = nx;
  posArr[i * 3 + 1] = ny;
  posArr[i * 3 + 2] = nz;

  const c = clusterColors[labels[i]] || noiseColor;
  colArr[i * 3]     = c[0];
  colArr[i * 3 + 1] = c[1];
  colArr[i * 3 + 2] = c[2];

  sizeArr[i] = BASE_SIZE;
}

geometry.setAttribute("position", new THREE.BufferAttribute(posArr, 3));
geometry.setAttribute("color",    new THREE.BufferAttribute(colArr, 3));
geometry.setAttribute("size",     new THREE.BufferAttribute(sizeArr, 1));

// Custom shader for round, glowing points
const pointsMaterial = new THREE.ShaderMaterial({
  vertexShader: `
    attribute float size;
    varying vec3 vColor;
    void main() {
      vColor = color;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * (200.0 / -mvPosition.z);
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    varying vec3 vColor;
    void main() {
      float d = length(gl_PointCoord - vec2(0.5));
      if (d > 0.5) discard;
      float glow = 1.0 - smoothstep(0.0, 0.5, d);
      float alpha = glow * 0.9;
      vec3 col = vColor + vec3(0.15) * (1.0 - d * 2.0);
      gl_FragColor = vec4(col, alpha);
    }
  `,
  vertexColors: true,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
});

const points = new THREE.Points(geometry, pointsMaterial);
scene.add(points);

// ═══════════════════════════════════════════════════════════════════════════
// SUBTLE GRID HELPER
// ═══════════════════════════════════════════════════════════════════════════
const gridHelper = new THREE.GridHelper(SCALE * 1.2, 20, 0x1a1a2e, 0x12121f);
gridHelper.position.y = -SCALE * 0.5;
scene.add(gridHelper);

// ═══════════════════════════════════════════════════════════════════════════
// CAMERA POSITION
// ═══════════════════════════════════════════════════════════════════════════
camera.position.set(SCALE * 0.8, SCALE * 0.6, SCALE * 0.8);
camera.lookAt(0, 0, 0);

// ═══════════════════════════════════════════════════════════════════════════
// ORBIT CONTROLS (inline — no import needed)
// ═══════════════════════════════════════════════════════════════════════════
const orbitState = {
  target: new THREE.Vector3(0, 0, 0),
  spherical: new THREE.Spherical(),
  sphericalDelta: new THREE.Spherical(),
  panOffset: new THREE.Vector3(),
  rotateStart: new THREE.Vector2(),
  panStart: new THREE.Vector2(),
  isDragging: false,
  button: -1,
  damping: 0.12,
};

(function initOrbit() {
  const offset = new THREE.Vector3();
  offset.copy(camera.position).sub(orbitState.target);
  orbitState.spherical.setFromVector3(offset);
})();

container.addEventListener("mousedown", e => {
  orbitState.isDragging = true;
  orbitState.button = e.button;
  orbitState.rotateStart.set(e.clientX, e.clientY);
  orbitState.panStart.set(e.clientX, e.clientY);
});

container.addEventListener("mousemove", e => {
  if (!orbitState.isDragging) return;

  if (orbitState.button === 0) {
    // Left button — orbit
    const dx = (e.clientX - orbitState.rotateStart.x) * 0.005;
    const dy = (e.clientY - orbitState.rotateStart.y) * 0.005;
    orbitState.spherical.theta -= dx;
    orbitState.spherical.phi   -= dy;
    orbitState.spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, orbitState.spherical.phi));
    orbitState.rotateStart.set(e.clientX, e.clientY);
  } else if (orbitState.button === 2) {
    // Right button — pan
    const dx = (e.clientX - orbitState.panStart.x) * 0.05;
    const dy = (e.clientY - orbitState.panStart.y) * 0.05;

    const panLeft = new THREE.Vector3();
    panLeft.setFromMatrixColumn(camera.matrix, 0);
    panLeft.multiplyScalar(-dx);

    const panUp = new THREE.Vector3();
    panUp.setFromMatrixColumn(camera.matrix, 1);
    panUp.multiplyScalar(dy);

    orbitState.panOffset.add(panLeft).add(panUp);
    orbitState.panStart.set(e.clientX, e.clientY);
  }
});

container.addEventListener("mouseup",   () => { orbitState.isDragging = false; });
container.addEventListener("mouseleave", () => { orbitState.isDragging = false; });
container.addEventListener("contextmenu", e => e.preventDefault());

container.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = e.deltaY > 0 ? 1.1 : 0.9;
  orbitState.spherical.radius *= factor;
  orbitState.spherical.radius = Math.max(5, Math.min(SCALE * 5, orbitState.spherical.radius));
}, { passive: false });

function updateOrbit() {
  orbitState.target.add(orbitState.panOffset);
  orbitState.panOffset.multiplyScalar(0);

  const offset = new THREE.Vector3();
  offset.setFromSpherical(orbitState.spherical);
  camera.position.copy(orbitState.target).add(offset);
  camera.lookAt(orbitState.target);
}

// ═══════════════════════════════════════════════════════════════════════════
// RAYCASTER — hover & click
// ═══════════════════════════════════════════════════════════════════════════
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = BASE_SIZE * 0.6;
const mouse = new THREE.Vector2();
let hoveredIndex = -1;

const tooltip   = document.getElementById("tooltip");
const ttTitle   = document.getElementById("tt-title");
const ttArtist  = document.getElementById("tt-artist");
const ttCluster = document.getElementById("tt-cluster");

function getTrackName(i) {
  if (metadata[i] && metadata[i].title) return metadata[i].title;
  const p = tracks[i] || "";
  return p.split(/[/\\]/).pop() || `Track ${i}`;
}

function getArtistName(i) {
  if (metadata[i] && metadata[i].artist) return metadata[i].artist;
  return "";
}

container.addEventListener("mousemove", e => {
  mouse.x =  (e.clientX / window.innerWidth)  * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(points);

  if (intersects.length > 0) {
    const idx = intersects[0].index;
    if (!clusterVisibility[labels[idx]]) { hideTooltip(); return; }
    hoveredIndex = idx;
    const name   = getTrackName(idx);
    const artist = getArtistName(idx);
    ttTitle.textContent   = name;
    ttArtist.textContent  = artist;
    ttCluster.textContent = `Cluster ${labels[idx]}`;
    tooltip.style.display = "block";
    tooltip.style.left    = (e.clientX + 14) + "px";
    tooltip.style.top     = (e.clientY + 14) + "px";
    document.body.style.cursor = "pointer";
  } else {
    hideTooltip();
  }
});

function hideTooltip() {
  hoveredIndex = -1;
  tooltip.style.display = "none";
  document.body.style.cursor = "default";
}

// ═══════════════════════════════════════════════════════════════════════════
// SELECTION
// ═══════════════════════════════════════════════════════════════════════════
const selected = new Set();
let selectMode = false;

const selBar   = document.getElementById("selection-bar");
const selCount = document.getElementById("sel-count");

function updateSelectionUI() {
  if (selected.size > 0) {
    selBar.style.display = "block";
    selCount.textContent = `${selected.size} track${selected.size !== 1 ? "s" : ""} selected`;
  } else {
    selBar.style.display = "none";
  }
  updatePointAppearance();
}

function updatePointAppearance() {
  const col = geometry.attributes.color.array;
  const sz  = geometry.attributes.size.array;

  for (let i = 0; i < N; i++) {
    if (!clusterVisibility[labels[i]]) {
      col[i*3] = col[i*3+1] = col[i*3+2] = 0;
      sz[i] = 0;
      continue;
    }

    const c = clusterColors[labels[i]] || noiseColor;

    if (selected.has(i)) {
      // Selected: bright white-ish glow
      col[i*3]     = Math.min(1, c[0] + 0.4);
      col[i*3 + 1] = Math.min(1, c[1] + 0.4);
      col[i*3 + 2] = Math.min(1, c[2] + 0.4);
      sz[i] = BASE_SIZE * 2;
    } else {
      col[i*3]     = c[0];
      col[i*3 + 1] = c[1];
      col[i*3 + 2] = c[2];
      sz[i] = BASE_SIZE;
    }
  }

  geometry.attributes.color.needsUpdate = true;
  geometry.attributes.size.needsUpdate  = true;
}

container.addEventListener("click", e => {
  if (orbitState.isDragging) return;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(points);

  if (intersects.length > 0) {
    const idx = intersects[0].index;
    if (!clusterVisibility[labels[idx]]) return;

    if (e.shiftKey || e.ctrlKey || e.metaKey || selectMode) {
      if (selected.has(idx)) selected.delete(idx);
      else selected.add(idx);
    } else {
      selected.clear();
      selected.add(idx);
    }
    updateSelectionUI();
  } else if (!e.shiftKey && !e.ctrlKey && !e.metaKey) {
    clearSelection();
  }
});

function clearSelection() {
  selected.clear();
  updateSelectionUI();
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════════════════════
function getSelectedTracks() {
  return [...selected].sort((a, b) => a - b).map(i => tracks[i]);
}

function exportSelectedCSV() {
  const sel = getSelectedTracks();
  if (sel.length === 0) { alert("No tracks selected"); return; }

  let csv = "track_path\n";
  sel.forEach(t => { csv += '"' + t.replace(/"/g, '""') + '"\n'; });

  downloadText(csv, "cluster_selection.csv", "text/csv");
}

function exportSelectedM3U() {
  const sel = getSelectedTracks();
  if (sel.length === 0) { alert("No tracks selected"); return; }

  let m3u = "#EXTM3U\n";
  sel.forEach(t => { m3u += t + "\n"; });

  downloadText(m3u, "cluster_selection.m3u", "audio/x-mpegurl");
}

function downloadText(text, filename, mime) {
  const blob = new Blob([text], { type: mime });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ═══════════════════════════════════════════════════════════════════════════
// LEGEND
// ═══════════════════════════════════════════════════════════════════════════
(function buildLegend() {
  const legend = document.getElementById("legend");
  const validClusters = uniqueLabels.filter(l => l >= 0);

  validClusters.forEach(lbl => {
    const count = labels.filter(l => l === lbl).length;
    const c = clusterColors[lbl];
    const hex = "#" + [c[0], c[1], c[2]].map(v => {
      const h = Math.round(v * 255).toString(16);
      return h.length === 1 ? "0" + h : h;
    }).join("");

    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-swatch" style="background:${hex}"></span>
      <span>Cluster ${lbl} (${count})</span>`;
    item.addEventListener("click", () => {
      clusterVisibility[lbl] = !clusterVisibility[lbl];
      item.classList.toggle("hidden", !clusterVisibility[lbl]);
      updatePointAppearance();
    });
    legend.appendChild(item);
  });

  const noiseCount = labels.filter(l => l < 0).length;
  if (noiseCount > 0) {
    const item = document.createElement("div");
    item.className = "legend-item";
    item.innerHTML = `<span class="legend-swatch" style="background:#666"></span>
      <span>Noise (${noiseCount})</span>`;
    item.addEventListener("click", () => {
      clusterVisibility[-1] = !clusterVisibility[-1];
      item.classList.toggle("hidden", !clusterVisibility[-1]);
      updatePointAppearance();
    });
    legend.appendChild(item);
  }
})();

// ═══════════════════════════════════════════════════════════════════════════
// BUTTONS
// ═══════════════════════════════════════════════════════════════════════════
function setCameraView(theta, phi) {
  orbitState.spherical.theta = theta;
  orbitState.spherical.phi = phi;
}

document.getElementById("btn-reset").addEventListener("click", () => {
  orbitState.target.set(0, 0, 0);
  orbitState.spherical.set(SCALE * 1.2, Math.PI / 4, Math.PI / 4);
});

document.getElementById("btn-top").addEventListener("click",   () => setCameraView(0, 0.01));
document.getElementById("btn-front").addEventListener("click", () => setCameraView(0, Math.PI / 2));
document.getElementById("btn-side").addEventListener("click",  () => setCameraView(Math.PI / 2, Math.PI / 2));

document.getElementById("btn-select").addEventListener("click", function() {
  selectMode = !selectMode;
  this.classList.toggle("active", selectMode);
  this.textContent = selectMode ? "Select (ON)" : "Select";
});

document.getElementById("btn-export-csv").addEventListener("click", exportSelectedCSV);
document.getElementById("btn-export-m3u").addEventListener("click", exportSelectedM3U);

// ═══════════════════════════════════════════════════════════════════════════
// HUD STATS
// ═══════════════════════════════════════════════════════════════════════════
const nClusters = uniqueLabels.filter(l => l >= 0).length;
document.getElementById("stats-text").textContent =
  `${N.toLocaleString()} tracks · ${nClusters} cluster${nClusters !== 1 ? "s" : ""}` +
  (CLUSTER_DATA.X_downsampled ? ` (downsampled from ${CLUSTER_DATA.X_total_points.toLocaleString()})` : "");

// ═══════════════════════════════════════════════════════════════════════════
// RENDER LOOP
// ═══════════════════════════════════════════════════════════════════════════
function animate() {
  requestAnimationFrame(animate);
  updateOrbit();
  renderer.render(scene, camera);
}
animate();

// Handle window resize
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""
