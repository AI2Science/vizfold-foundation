// static/js/attention.js

function initAttentionUI() {
    const menu = document.getElementById('attn-dropdown-menu');
    if (!menu) return;
    
    // Dropdown handlers
    menu.querySelectorAll('a').forEach(a => {
        a.onclick = (e) => {
            e.preventDefault();
            window.AppState.attnType = a.getAttribute('data-type');
            document.getElementById('attnDropdown').textContent = a.textContent;
            if (window.AppState.attnData) updateLayerControls();
            updateAttentionDisplay(getCurrentAssets());
        };
    });

    // Default selection
    window.AppState.attnType = 'msa_row';
    const attnBtn = document.getElementById('attnDropdown');
    if (attnBtn) attnBtn.textContent = 'MSA Row Attention';

    // Layer Range Slider
    const range = document.getElementById('layerRange');
    if (range) {
        range.oninput = () => {
            document.getElementById('layerValue').textContent = range.value;
            updateAttentionDisplay(getCurrentAssets());
        };
    }

    // 3D Overlay Checkbox
    const noLayerChk = document.getElementById('noLayer3D');
    if (noLayerChk) {
        noLayerChk.onchange = () => {
            if (noLayerChk.checked) {
                window.clear3DOverlay();
            } else {
                window.maybeUpdate3DOverlay();
            }
        };
    }

    // Trigger generation
    triggerVizGeneration();
    pollVizUntilReady();
    setupArcInteractions();
    setupHeatmapInteractions();
}

function setAttnLoading(loading) {
    const loadingEl = document.getElementById('attn-loading');
    const contentEl = document.getElementById('attn-content');
    if (!loadingEl || !contentEl) return;
    loadingEl.style.display = loading ? 'block' : 'none';
    contentEl.style.display = loading ? 'none' : 'flex';
}

function updateLayerControls() {
    const range = document.getElementById('layerRange');
    const type = window.AppState.attnType;
    const data = window.AppState.attnData;
    const layers = (type === 'msa_row') ? (data?.msa_row?.layers || []) : (data?.triangle_start?.layers || []);
    
    if (!range) return;
    
    if (layers.length === 0) {
        range.disabled = true;
        range.min = 0; range.max = 0; range.value = 0;
        document.getElementById('layerValue').textContent = '-';
        return;
    }
    
    layers.sort((a,b)=>a-b);
    range.disabled = false;
    range.min = layers[0];
    range.max = layers[layers.length-1];
    
    if (!layers.includes(parseInt(range.value))) {
        range.value = layers[0];
    }
    document.getElementById('layerValue').textContent = range.value;
}

function getCurrentAssets() {
    if (!window.AppState.attnData) return null;
    const L = document.getElementById('layerRange').value;
    const type = window.AppState.attnType;
    const assets = (type === 'msa_row') 
        ? window.AppState.attnData.msa_row.assets[String(L)] 
        : window.AppState.attnData.triangle_start.assets[String(L)];
    return assets || null;
}

function updateAttentionDisplay(assets) {
    if (!assets) {
        setAttnLoading(true);
        clearArcAndHeatmap();
        return;
    }
    setAttnLoading(false);
    fetchAndRenderAttention(assets.attn_file_url);
}

function clearArcAndHeatmap() {
    const svg = document.getElementById('arcSvg');
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const ctx = document.getElementById('heatmapCanvas').getContext('2d');
    ctx.clearRect(0,0,ctx.canvas.width, ctx.canvas.height);
}

async function fetchAndRenderAttention(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error('failed to fetch attention file');
        const text = await res.text();
        const parsed = parseAttentionFile(text);
        
        populateHeadSelector(Object.keys(parsed).map(x=>parseInt(x,10)).sort((a,b)=>a-b));
        
        const head = getSelectedHead();
        const edges = preprocessConnections(parsed[head] || []);
        
        window.AppState.currentArcConnections = edges;
        window.AppState.currentSequence = document.getElementById('sequence').value.trim();
        
        // Reset arc view
        window.AppState.arcView.xMin = 0; 
        window.AppState.arcView.xMax = window.AppState.currentSequence.length;
        
        renderArcDiagram(window.AppState.currentArcConnections, window.AppState.currentSequence);
        renderHeatmap(parsed[head] || [], window.AppState.currentSequence.length);
        
        window.maybeUpdate3DOverlay();
    } catch (e) {
        console.warn('fetchAndRenderAttention error', e);
        clearArcAndHeatmap();
    }
}

function parseAttentionFile(text) {
    const lines = text.split(/\r?\n/);
    const heads = {};
    let currentHead = null;
    for (let raw of lines) {
        const line = raw.trim();
        if (!line) continue;
        if (line.toLowerCase().startsWith('layer')) {
            const parts = line.replace(',', '').split(/\s+/);
            const headIdx = parseInt(parts[parts.length-1], 10);
            currentHead = headIdx;
            if (!(currentHead in heads)) heads[currentHead] = [];
        } else {
            const parts = line.split(/\s+/);
            if (parts.length >= 3) {
                const r1 = parseInt(parts[0], 10);
                const r2 = parseInt(parts[1], 10);
                const w = parseFloat(parts[2]);
                if (!Number.isNaN(r1) && !Number.isNaN(r2) && !Number.isNaN(w)) {
                    heads[currentHead].push([r1, r2, w]);
                }
            }
        }
    }
    return heads;
}

function populateHeadSelector(heads) {
    const sel = document.getElementById('headSelect');
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = '';
    heads.forEach(h => {
        const opt = document.createElement('option');
        opt.value = String(h);
        opt.textContent = `Head ${h}`;
        sel.appendChild(opt);
    });
    if (heads.length > 0) {
        sel.value = heads.includes(parseInt(current,10)) ? current : String(heads[0]);
    }
    sel.onchange = () => {
        const assets = getCurrentAssets();
        if (assets) fetchAndRenderAttention(assets.attn_file_url);
    };
}

function getSelectedHead() {
    const sel = document.getElementById('headSelect');
    return sel && sel.value ? parseInt(sel.value, 10) : 0;
}

// ----------------- Rendering Logic ----------------- //

function renderArcDiagram(connections, sequence) {
    const svg = document.getElementById('arcSvg');
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const n = sequence.length;
    const width = svg.clientWidth || svg.parentElement.clientWidth;
    const height = svg.getAttribute('height') ? parseInt(svg.getAttribute('height'),10) : 480;
    const padding = 20;
    const baselineY = height - 40;
    
    const view = window.AppState.arcView;
    if (view.xMax === null) { view.xMin = 0; view.xMax = n; }
    
    const viewSpan = Math.max(view.xMax - view.xMin, 1e-6);
    const xScale = (width - 2*padding) / viewSpan;
    const xCenter = (fIndex) => padding + xScale * (fIndex - view.xMin); 
    const xTick = (i) => xCenter(i + 0.5);

    // Draw baseline labels
    const iStart = Math.max(0, Math.floor(view.xMin));
    const iEnd = Math.min(n, Math.ceil(view.xMax));
    
    for (let i=iStart; i<iEnd; i++) {
        const tx = document.createElementNS('http://www.w3.org/2000/svg','text');
        const xpix = Math.round(xTick(i));
        tx.setAttribute('x', xpix);
        tx.setAttribute('y', baselineY + 14);
        tx.setAttribute('text-anchor','middle');
        tx.setAttribute('font-size','10');
        tx.textContent = sequence[i];
        
        if (window.AppState.attnType === 'triangle_start' && 
            window.AppState.currentProteinMeta && 
            window.AppState.currentProteinMeta.residue_idx === i) {
            tx.setAttribute('fill', 'blue');
            tx.setAttribute('font-weight', 'bold');
        }
        svg.appendChild(tx);
    }

    if (!connections || connections.length === 0) return;
    
    const weights = connections.map(c=>c[2]);
    const wmin = Math.min(...weights), wmax = Math.max(...weights);
    const norm = (w)=> (wmax!==wmin) ? (w - wmin)/(wmax-wmin) : 0.5;

    const drawableHeight = baselineY - padding;
    let maxAbsDist = 0;
    for (const [r1, r2] of connections) {
        const df = Math.abs((r2 + 0.5) - (r1 + 0.5));
        if (df > maxAbsDist) maxAbsDist = df;
    }
    const rawMaxHeightPx = maxAbsDist * 0.5 * ((width - 2*padding) / n);
    const targetMaxHeight = 0.75 * drawableHeight;
    const heightScale = rawMaxHeightPx > 0 ? (targetMaxHeight / rawMaxHeightPx) : 1.0;

    const SAMPLES = 60;
    connections.forEach(([r1, r2, w]) => {
        const f1 = r1 + 0.5;
        const f2 = r2 + 0.5;
        const px1 = xCenter(f1);
        const px2 = xCenter(f2);
        const baseXScale = (width - 2*padding) / n;
        const heightPx = Math.abs(f2 - f1) * 0.5 * baseXScale * heightScale;
        
        const path = document.createElementNS('http://www.w3.org/2000/svg','path');
        let d = '';
        for (let i=0; i<=SAMPLES; i++) {
            const t = i / SAMPLES;
            const xpx = px1 + (px2 - px1) * t;
            const ypx = baselineY - heightPx * Math.sin(Math.PI * t);
            d += (i===0 ? `M ${xpx} ${ypx}` : ` L ${xpx} ${ypx}`);
        }
        const nw = norm(w);
        const strokeWidth = (0.5 + 3* nw).toFixed(2);
        const blue = Math.round((0.5 + 0.5 * nw) * 255);
        path.setAttribute('d', d);
        path.setAttribute('stroke', `rgb(0,0,${blue})`);
        path.setAttribute('stroke-width', strokeWidth);
        path.setAttribute('fill','none');
        path.setAttribute('opacity','0.9');
        svg.appendChild(path);
    });
}

function preprocessConnections(connections) {
    const pairs = new Map();
    for (const item of connections) {
        const r1 = item[0]|0, r2 = item[1]|0; const w = Number(item[2]);
        if (!Number.isFinite(w)) continue;
        if (r1 === r2) continue;
        const a = Math.min(r1, r2), b = Math.max(r1, r2);
        const key = a + '-' + b;
        const prev = pairs.get(key);
        if (prev === undefined || w > prev) pairs.set(key, w);
    }
    const arr = Array.from(pairs.entries()).map(([key, w]) => {
        const [a, b] = key.split('-').map(x=>parseInt(x,10));
        return [a, b, w];
    });
    arr.sort((x,y)=> y[2]-x[2]);
    return arr.slice(0, 50); // TOP_K = 50
}

function renderHeatmap(connections, seqLen) {
    const canvas = document.getElementById('heatmapCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);
    
    const grid = new Float32Array(seqLen*seqLen);
    let wmin = Infinity, wmax = -Infinity;
    for (const [r1,r2,w] of connections) {
        const idx = r1*seqLen + r2;
        grid[idx] = w;
        if (w<wmin) wmin=w;
        if (w>wmax) wmax=w;
    }
    if (!isFinite(wmin) || !isFinite(wmax)) { wmin=0; wmax=1; }
    const norm = (v)=> (wmax!==wmin)? (v-wmin)/(wmax-wmin) : 0.5;
    
    const img = ctx.createImageData(seqLen, seqLen);
    for (let y=0;y<seqLen;y++){
        for (let x=0;x<seqLen;x++){
            const v = norm(grid[y*seqLen+x] || 0);
            const i = (y*seqLen + x)*4;
            const r = 230 - Math.floor(180*v);
            const g = 240 - Math.floor(200*v);
            const b = 255;
            img.data[i]=r; img.data[i+1]=g; img.data[i+2]=b; img.data[i+3]=255;
        }
    }
    
    const off = document.createElement('canvas');
    off.width = seqLen; off.height = seqLen;
    off.getContext('2d').putImageData(img,0,0);
    
    window.AppState.lastHeatSource = off;
    window.AppState.lastHeatSeqLen = seqLen;
    
    const hView = window.AppState.heatView;
    if (hView.w === null || hView.h === null) { 
        window.AppState.heatView = { x: 0, y: 0, w: seqLen, h: seqLen }; 
    }
    redrawHeatmap();
}

function redrawHeatmap() {
    const canvas = document.getElementById('heatmapCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0,0,canvas.width, canvas.height);
    
    if (!window.AppState.lastHeatSource) return;
    const W = canvas.width, H = canvas.height;
    const hView = window.AppState.heatView;
    const seqLen = window.AppState.lastHeatSeqLen;
    
    const sx = Math.max(0, Math.min(seqLen-1, hView.x));
    const sy = Math.max(0, Math.min(seqLen-1, hView.y));
    const sw = Math.max(1, Math.min(seqLen - sx, hView.w));
    const sh = Math.max(1, Math.min(seqLen - sy, hView.h));
    
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(window.AppState.lastHeatSource, sx, sy, sw, sh, 0, 0, W, H);
}

// ----------------- Interactions ----------------- //

function setupArcInteractions() {
    const svg = document.getElementById('arcSvg');
    let isDown = false; let startX = 0; let startMin = 0;
    
    svg.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDown = true;
        startX = e.clientX;
        startMin = window.AppState.arcView.xMin;
        document.body.style.userSelect = 'none';
        svg.style.cursor = 'grabbing';
    });
    
    window.addEventListener('mouseup', ()=> {
        isDown = false;
        document.body.style.userSelect = '';
        svg.style.cursor = '';
    });
    
    window.addEventListener('mousemove', (e)=>{
        if (!isDown || !window.AppState.currentSequence) return;
        e.preventDefault();
        
        const width = svg.clientWidth || svg.parentElement.clientWidth;
        const padding = 20;
        const drawable = Math.max(1, width - 2*padding);
        const span = Math.max(1e-6, window.AppState.arcView.xMax - window.AppState.arcView.xMin);
        const dxPx = e.clientX - startX;
        const dIndex = dxPx / drawable * span;
        const n = window.AppState.currentSequence.length;
        
        let newMin = startMin - dIndex;
        newMin = Math.max(0, Math.min(n - span, newMin));
        window.AppState.arcView.xMin = newMin; 
        window.AppState.arcView.xMax = newMin + span;
        renderArcDiagram(window.AppState.currentArcConnections, window.AppState.currentSequence);
    });

    svg.addEventListener('wheel', (e)=>{
        if (!window.AppState.currentSequence) return;
        e.preventDefault();
        const rect = svg.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const width = svg.clientWidth || svg.parentElement.clientWidth;
        const padding = 20;
        const drawable = Math.max(1, width - 2*padding);
        const n = window.AppState.currentSequence.length;
        const span = Math.max(1e-6, window.AppState.arcView.xMax - window.AppState.arcView.xMin);
        const mouseT = Math.max(0, Math.min(1, (x - padding) / drawable));
        const mouseIndex = window.AppState.arcView.xMin + mouseT * span;
        
        const factor = (e.deltaY < 0) ? 0.9 : 1.1; // zoom in/out
        let newSpan = Math.max(5, Math.min(n, span * factor));
        let newMin = mouseIndex - mouseT * newSpan;
        newMin = Math.max(0, Math.min(n - newSpan, newMin));
        
        window.AppState.arcView.xMin = newMin; 
        window.AppState.arcView.xMax = newMin + newSpan;
        renderArcDiagram(window.AppState.currentArcConnections, window.AppState.currentSequence);
    }, { passive: false });
}

function setupHeatmapInteractions() {
    const canvas = document.getElementById('heatmapCanvas');
    let isDown = false; let startX=0, startY=0; let startView=null;
    
    canvas.addEventListener('mousedown', (e)=>{
        isDown = true;
        startX = e.clientX; startY = e.clientY;
        startView = { ...window.AppState.heatView };
    });
    
    window.addEventListener('mouseup', ()=> isDown=false);
    
    window.addEventListener('mousemove', (e)=>{
        if (!isDown || !window.AppState.lastHeatSource) return;
        const W = canvas.clientWidth; const H = canvas.clientHeight;
        const dxPx = e.clientX - startX; const dyPx = e.clientY - startY;
        
        const dX = dxPx / Math.max(1, W) * startView.w;
        const dY = dyPx / Math.max(1, H) * startView.h;
        const seqLen = window.AppState.lastHeatSeqLen;
        
        const maxX = Math.max(0, seqLen - startView.w);
        const maxY = Math.max(0, seqLen - startView.h);
        
        window.AppState.heatView.x = Math.max(0, Math.min(maxX, startView.x - dX));
        window.AppState.heatView.y = Math.max(0, Math.min(maxY, startView.y - dY));
        redrawHeatmap();
    });
    
    canvas.addEventListener('wheel', (e)=>{
        if (!window.AppState.lastHeatSource) return;
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const W = canvas.clientWidth; const H = canvas.clientHeight;
        const tx = Math.max(0, Math.min(1, mx / Math.max(1, W)));
        const ty = Math.max(0, Math.min(1, my / Math.max(1, H)));
        const factor = (e.deltaY < 0) ? 0.9 : 1.1;
        const seqLen = window.AppState.lastHeatSeqLen;
        
        let newW = Math.max(10, Math.min(seqLen, window.AppState.heatView.w * factor));
        let newH = Math.max(10, Math.min(seqLen, window.AppState.heatView.h * factor));
        
        const cx = window.AppState.heatView.x + tx * window.AppState.heatView.w;
        const cy = window.AppState.heatView.y + ty * window.AppState.heatView.h;
        let newX = cx - tx * newW;
        let newY = cy - ty * newH;
        
        newX = Math.max(0, Math.min(seqLen - newW, newX));
        newY = Math.max(0, Math.min(seqLen - newH, newY));
        
        window.AppState.heatView = { x: newX, y: newY, w: newW, h: newH };
        redrawHeatmap();
    }, { passive: false });
}

// ----------------- Polling ----------------- //

async function triggerVizGeneration() {
    if (!window.AppState.currentProteinMeta) return;
    try {
        await fetch('/viz/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ protein_id: window.AppState.currentProteinMeta.protein_id })
        });
    } catch (e) {
        console.warn('Failed to trigger viz generation:', e);
    }
}

async function fetchVizList() {
    if (!window.AppState.currentProteinMeta) return null;
    const params = new URLSearchParams({
        protein_id: window.AppState.currentProteinMeta.protein_id,
        residue_idx: window.AppState.currentProteinMeta.residue_idx
    });
    const res = await fetch(`/viz/list?${params.toString()}`);
    if (!res.ok) return null;
    return await res.json();
}

async function pollVizUntilReady() {
    if (window.AppState.attnPollTimer) clearInterval(window.AppState.attnPollTimer);
    setAttnLoading(true);
    
    const attempt = async () => {
        try {
            const data = await fetchVizList();
            if (data && ((data.msa_row.layers && data.msa_row.layers.length>0) || (data.triangle_start.layers && data.triangle_start.layers.length>0))) {
                window.AppState.attnData = data;
                updateLayerControls();
                updateAttentionDisplay(getCurrentAssets());
                if (window.AppState.attnPollTimer) clearInterval(window.AppState.attnPollTimer);
            }
        } catch (e) {
            console.warn('viz list fetch failed', e);
        }
    };
    
    await attempt();
    window.AppState.attnPollTimer = setInterval(attempt, 4000);
}

// Expose necessary functions for index.html onclicks
window.initAttentionUI = initAttentionUI;