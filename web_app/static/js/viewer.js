// static/js/viewer.js

// Initialize 3DMol viewer
function initViewer() {
    try {
        const viewer = $3Dmol.createViewer('viewer', {
            defaultcolors: $3Dmol.rasmolElementColors,
            backgroundColor: 0xffffff,
            width: '100%',
            height: '100%'
        });
        viewer.setStyle({}, { cartoon: {} });
        viewer.zoomTo();
        viewer.render();
        return viewer;
    } catch (error) {
        console.error('Error initializing 3DMol viewer:', error);
        return null;
    }
}

async function loadPDB(pdbPath) {
    try {
        window.viewer.clear();
        const response = await fetch(`/pdb/${pdbPath}`);
        const pdbData = await response.text();

        window.viewer.addModel(pdbData, 'pdb');
        window.viewer.setStyle({}, { cartoon: {} });
        window.viewer.zoomTo();
        window.viewer.render();

        window.AppState.currentModel = pdbData;

        // Compute residue coordinates for 3D overlays
        try {
            const model = window.viewer.getModel();
            const caAtoms = model ? model.selectedAtoms({ atom: 'CA' }) : [];
            window.residueCoords = caAtoms.map(a => ({ x: a.x, y: a.y, z: a.z }));
        } catch (e) {
            console.warn('Failed to compute CA coordinates', e);
            window.residueCoords = [];
        }

        document.getElementById('viewer-controls').style.display = 'block';
    } catch (error) {
        console.error('Error loading PDB:', error);
        throw new Error('Failed to load protein structure');
    }
}

function clear3DOverlay() {
    try { window.viewer.removeAllShapes(); window.viewer.render(); } catch (e) {}
}

function maybeUpdate3DOverlay() {
    const noLayer = document.getElementById('noLayer3D');
    if (noLayer && noLayer.checked) { clear3DOverlay(); return; }
    
    if (!Array.isArray(window.residueCoords) || window.residueCoords.length === 0) return;
    if (!window.AppState.currentArcConnections || window.AppState.currentArcConnections.length === 0) { 
        clear3DOverlay(); 
        return; 
    }
    
    try {
        window.viewer.removeAllShapes();
        const n = window.AppState.currentSequence ? window.AppState.currentSequence.length : window.residueCoords.length;
        const weights = window.AppState.currentArcConnections.map(e => e[2]);
        const wmin = Math.min(...weights), wmax = Math.max(...weights);
        const norm = (w) => (wmax !== wmin) ? (w - wmin) / (wmax - wmin) : 0.5;
        
        window.AppState.currentArcConnections.forEach(([i, j, w]) => {
            if (i < 0 || j < 0 || i >= n || j >= n) return;
            const a = window.residueCoords[i];
            const b = window.residueCoords[j];
            if (!a || !b) return;
            const t = norm(w);
            const radius = 0.1 + 0.4 * t;
            const blue = Math.round((0.5 + 0.5 * t) * 255);
            window.viewer.addCylinder({
                start: a, end: b,
                radius: radius,
                fromCap: 1, toCap: 1,
                color: `rgb(0,0,${blue})`
            });
        });
        window.viewer.render();
    } catch (e) {
        console.warn('Failed updating 3D overlay', e);
    }
}

// Expose functions needed globally
window.loadPDB = loadPDB;
window.maybeUpdate3DOverlay = maybeUpdate3DOverlay;
window.clear3DOverlay = clear3DOverlay;

// Setup Event Listeners on Load
document.addEventListener('DOMContentLoaded', function () {
    window.viewer = initViewer();

    // Viewer Controls
    const controls = {
        'toggle-backbone': () => { window.viewer.setStyle({}, { line: {} }); window.viewer.render(); },
        'toggle-sidechains': () => { window.viewer.setStyle({}, { stick: {} }); window.viewer.render(); },
        'color-cartoon': () => { window.viewer.setStyle({}, { cartoon: {} }); window.viewer.render(); },
        'color-chain': () => { window.viewer.setStyle({}, { cartoon: { colorscheme: 'chainHetatm' } }); window.viewer.render(); },
        'color-residue': () => { window.viewer.setStyle({}, { cartoon: { colorscheme: 'shapely' } }); window.viewer.render(); },
        'zoom-to-fit': () => { window.viewer.zoomTo(); window.viewer.render(); },
        'reset-view': () => { window.viewer.zoomTo(); window.viewer.setStyle({}, { cartoon: {} }); window.viewer.render(); }
    };

    Object.entries(controls).forEach(([id, handler]) => {
        const element = document.getElementById(id);
        if (element) element.addEventListener('click', handler);
    });

    // Handle Resize
    function handleResize() {
        if (window.viewer) {
            const container = document.getElementById('viewer');
            if (container) {
                const canvas = container.querySelector('canvas');
                if (canvas) { canvas.style.width = '100%'; canvas.style.height = '100%'; }
                window.viewer.resize();
                window.viewer.render();
            }
        }
    }
    handleResize();
    let resizeTimer;
    window.addEventListener('resize', function () {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(handleResize, 100);
    });
});