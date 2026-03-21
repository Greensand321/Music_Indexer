/**
 * Three.js 3D Cluster Viewer for AlphaDEX Music Indexer
 * Renders cluster data with interactive orbit controls, point selection, and hover tooltips
 */

let scene, camera, renderer, controls, raycaster, mouse;
let points = null;
let pointsData = {
    positions: [],
    colors: [],
    clusters: [],
    tracks: [],
    metadata: [],
};
let selectedIndices = new Set();
let hoveredIndex = -1;

// Python bridge
let pythonBridge = null;

// Initialization
function init() {
    setupThreeJs();
    setupInteraction();
    setupPythonBridge();
    animate();
}

/**
 * Set up Three.js scene, camera, renderer
 */
function setupThreeJs() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    scene.fog = new THREE.Fog(0x1a1a1a, 300, 1000);

    // Camera
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 10000);
    camera.position.set(80, 80, 80);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowShadowMap;
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // Lighting
    // Ambient light for overall illumination
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    // Directional light for depth
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(150, 150, 150);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.left = -200;
    dirLight.shadow.camera.right = 200;
    dirLight.shadow.camera.top = 200;
    dirLight.shadow.camera.bottom = -200;
    dirLight.shadow.camera.near = 0.1;
    dirLight.shadow.camera.far = 1000;
    scene.add(dirLight);

    // Point light for accent
    const pointLight = new THREE.PointLight(0x6366f1, 0.4);
    pointLight.position.set(-100, 100, 0);
    scene.add(pointLight);

    // Axes helper (RGB = XYZ)
    const axesHelper = new THREE.AxesHelper(50);
    scene.add(axesHelper);

    // Grid
    const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x222222);
    gridHelper.position.y = -100;
    scene.add(gridHelper);

    // Orbit controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 2;
    controls.enableZoom = true;
    controls.zoomSpeed = 1;

    // Raycaster for picking
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 10;
    mouse = new THREE.Vector2();

    // Handle resize
    window.addEventListener('resize', onWindowResize, false);
}

/**
 * Set up interaction: mouse move for hover, click for selection
 */
function setupInteraction() {
    document.addEventListener('mousemove', onMouseMove, false);
    document.addEventListener('click', onMouseClick, false);
}

/**
 * Handle mouse move for hover detection
 */
function onMouseMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    if (points) {
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(points);

        if (intersects.length > 0) {
            const index = intersects[0].index;
            setHoveredPoint(index);
        } else {
            setHoveredPoint(-1);
        }
    }
}

/**
 * Handle mouse click for selection
 */
function onMouseClick(event) {
    if (points) {
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObject(points);

        if (intersects.length > 0) {
            const index = intersects[0].index;
            toggleSelection(index);
        }
    }
}

/**
 * Set hovered point and update visual
 */
function setHoveredPoint(index) {
    hoveredIndex = index;

    if (index >= 0 && pointsData.metadata[index]) {
        const meta = pointsData.metadata[index];
        updateHoverInfo(meta);

        // Highlight point (scale up)
        if (points) {
            const sizes = points.geometry.attributes.size.array;
            sizes.fill(6);
            sizes[index] = 12;
            points.geometry.attributes.size.needsUpdate = true;
        }
    } else {
        hideHoverInfo();

        // Reset sizes
        if (points) {
            points.geometry.attributes.size.fill(6);
            points.geometry.attributes.size.needsUpdate = true;
        }
    }
}

/**
 * Toggle point selection
 */
function toggleSelection(index) {
    if (selectedIndices.has(index)) {
        selectedIndices.delete(index);
    } else {
        selectedIndices.add(index);
    }

    updatePointColors();
    updateSelectionInfo();
    notifySelectionChanged();
}

/**
 * Update point colors based on cluster and selection state
 */
function updatePointColors() {
    if (!points) return;

    const colors = points.geometry.attributes.color.array;
    const baseColors = pointsData.colors;

    for (let i = 0; i < baseColors.length; i++) {
        const [r, g, b] = baseColors[i];
        const normalizedR = r / 255;
        const normalizedG = g / 255;
        const normalizedB = b / 255;

        if (selectedIndices.has(i)) {
            // Brighten selected points
            colors[i * 3] = Math.min(normalizedR * 1.5, 1);
            colors[i * 3 + 1] = Math.min(normalizedG * 1.5, 1);
            colors[i * 3 + 2] = Math.min(normalizedB * 1.5, 1);
        } else {
            colors[i * 3] = normalizedR;
            colors[i * 3 + 1] = normalizedG;
            colors[i * 3 + 2] = normalizedB;
        }
    }

    points.geometry.attributes.color.needsUpdate = true;
}

/**
 * Update hover info display
 */
function updateHoverInfo(metadata) {
    const hoverEl = document.getElementById('hover-info');
    const trackName = metadata.title || metadata.path.split('/').pop() || 'Unknown';
    hoverEl.innerHTML = `
        <strong>${escapeHtml(trackName)}</strong><br>
        Artist: ${escapeHtml(metadata.artist)}<br>
        Album: ${escapeHtml(metadata.album)}<br>
        Cluster: ${metadata.cluster || 'N/A'}
    `;
    hoverEl.classList.add('visible');
}

/**
 * Hide hover info
 */
function hideHoverInfo() {
    const hoverEl = document.getElementById('hover-info');
    hoverEl.classList.remove('visible');
}

/**
 * Update selection info display
 */
function updateSelectionInfo() {
    const infoEl = document.getElementById('selection-info');
    const count = selectedIndices.size;

    if (count > 0) {
        infoEl.innerHTML = `<strong>Selected:</strong> ${count} point(s)<br>Ctrl+Click to deselect`;
        infoEl.classList.add('visible');
    } else {
        infoEl.classList.remove('visible');
    }
}

/**
 * Load cluster data from Python
 */
function loadClusterData(clusterDataJson) {
    try {
        const clusterData = JSON.parse(clusterDataJson);

        // Extract arrays
        const positions = clusterData.positions || [];
        const colors = clusterData.colors || [];
        const clusters = clusterData.clusters || [];
        const tracks = clusterData.tracks || [];
        const metadata = clusterData.metadata || [];

        if (positions.length === 0) {
            console.error('No position data provided');
            return;
        }

        // Store data
        pointsData = {
            positions,
            colors,
            clusters,
            tracks,
            metadata,
        };

        // Create points
        createPointCloud();

        // Auto-fit camera to data bounds
        fitCameraToData();

        console.log(`Loaded ${positions.length} points`);
    } catch (error) {
        console.error('Failed to load cluster data:', error);
    }
}

/**
 * Create Three.js point cloud from data
 */
function createPointCloud() {
    // Remove old points
    if (points) {
        scene.remove(points);
    }

    const { positions, colors } = pointsData;

    // Create geometry
    const geometry = new THREE.BufferGeometry();

    // Positions
    const positionArray = new Float32Array(positions.length * 3);
    for (let i = 0; i < positions.length; i++) {
        const pos = positions[i];
        positionArray[i * 3] = pos[0];
        positionArray[i * 3 + 1] = pos[1];
        positionArray[i * 3 + 2] = pos[2];
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));

    // Colors (RGB → normalized)
    const colorArray = new Float32Array(colors.length * 3);
    for (let i = 0; i < colors.length; i++) {
        const [r, g, b] = colors[i];
        colorArray[i * 3] = r / 255;
        colorArray[i * 3 + 1] = g / 255;
        colorArray[i * 3 + 2] = b / 255;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

    // Sizes
    const sizeArray = new Float32Array(colors.length).fill(6);
    geometry.setAttribute('size', new THREE.BufferAttribute(sizeArray, 1));

    // Material
    const material = new THREE.PointsMaterial({
        size: 6,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.8,
    });

    // Points
    points = new THREE.Points(geometry, material);
    points.castShadow = true;
    points.receiveShadow = true;
    scene.add(points);
}

/**
 * Auto-fit camera to data bounds
 */
function fitCameraToData() {
    if (!points) return;

    const geometry = points.geometry;
    geometry.computeBoundingBox();
    const boundingBox = geometry.boundingBox;

    const center = boundingBox.getCenter(new THREE.Vector3());
    const size = boundingBox.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));

    camera.position.set(center.x + cameraZ * 0.5, center.y + cameraZ * 0.5, center.z + cameraZ * 0.8);
    camera.lookAt(center);
    controls.target.copy(center);
    controls.update();
}

/**
 * Notify Python of selection change
 */
function notifySelectionChanged() {
    if (pythonBridge) {
        const indices = Array.from(selectedIndices);
        pythonBridge.onSelectionChanged(JSON.stringify(indices));
    }
}

/**
 * Set up Python bridge via qwebchannel
 */
function setupPythonBridge() {
    // Wait for qwebchannel to be available
    if (typeof qt !== 'undefined') {
        new QWebChannel(qt.webChannelTransport, function (channel) {
            pythonBridge = channel.objects.graphBridge;

            if (pythonBridge) {
                console.log('Python bridge connected');

                // Fetch initial cluster data
                pythonBridge.getClusterData(function (data) {
                    loadClusterData(data);
                });

                // Connect signals
                pythonBridge.clusterDataUpdated.connect(function (data) {
                    loadClusterData(data);
                });
            } else {
                console.warn('graphBridge not available on channel');
            }
        });
    } else {
        console.warn('Qt WebChannel not available (running outside Qt environment)');
        // For development: use dummy data
        loadDummyData();
    }
}

/**
 * Load dummy data for development/testing
 */
function loadDummyData() {
    const numPoints = 100;
    const positions = [];
    const colors = [];
    const clusters = [];
    const metadata = [];

    for (let i = 0; i < numPoints; i++) {
        const x = (Math.random() - 0.5) * 100;
        const y = (Math.random() - 0.5) * 100;
        const z = (Math.random() - 0.5) * 100;
        const cluster = Math.floor(Math.random() * 5);

        positions.push([x, y, z]);
        colors.push([100 + Math.random() * 100, 100 + Math.random() * 100, 200]);
        clusters.push(cluster);
        metadata.push({
            title: `Track ${i}`,
            artist: `Artist ${cluster}`,
            album: `Album ${cluster}`,
            cluster,
        });
    }

    loadClusterData(
        JSON.stringify({
            positions,
            colors,
            clusters,
            metadata,
        })
    );
}

/**
 * Animation loop
 */
function animate() {
    requestAnimationFrame(animate);

    // Update controls
    if (controls) {
        controls.update();
    }

    // Render
    renderer.render(scene, camera);
}

/**
 * Handle window resize
 */
function onWindowResize() {
    const w = window.innerWidth;
    const h = window.innerHeight;

    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

/**
 * Utility: escape HTML in strings
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;',
    };
    return text.replace(/[&<>"']/g, (m) => map[m]);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
