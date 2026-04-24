import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import PlotComponent from 'react-plotly.js'
const Plot = PlotComponent.default || PlotComponent;
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

function App() {
  const [meta, setMeta] = useState(null)
  
  // Trace Explorer State
  const [viewMode, setViewMode] = useState('evolution') 
  const [iteration, setIteration] = useState(0)
  const [layer, setLayer] = useState(0)
  const [heatmapData, setHeatmapData] = useState([[0]])
  
  // Structure Viewer State
  const [pdbText, setPdbText] = useState(null)
  const [renderStyle, setRenderStyle] = useState('cartoon') 
  const [colorMode, setColorMode] = useState('confidence') 
  const viewerRef = useRef(null)

  // NEW: State to track which amino acid you clicked
  const [selectedResidue, setSelectedResidue] = useState(null)

  // 1. Fetch Meta and PDB on load
  useEffect(() => {
    axios.get(`${API_BASE}/meta`)
      .then(response => setMeta(response.data))
      .catch(error => console.error("Error fetching meta:", error))

    axios.get(`${API_BASE}/structure?t=${Date.now()}`)
      .then(response => setPdbText(response.data))
      .catch(error => console.error("Error fetching PDB:", error))
  }, [])

  // 2. Dynamic Fetching based on View Mode
  useEffect(() => {
    if (!meta) return;

    if (viewMode === 'evolution') {
      axios.get(`${API_BASE}/tensor/activations/recycle_${iteration}_s_z`)
        .then(response => {
          // Backend already averaged across channels
          setHeatmapData(response.data.data);
        })
        .catch(() => {
          console.log(`No data for iteration ${iteration}`);
          setHeatmapData([[0]]); 
        })
    } 
    else if (viewMode === 'attention') {
      const layerStr = layer.toString().padStart(3, '0');
      axios.get(`${API_BASE}/tensor/attention/layer_${layerStr}`)
        .then(response => {
          // Backend already averaged across all attention heads
          setHeatmapData(response.data.data);
        })
        .catch(() => {
          console.log(`No data for layer ${layerStr}`);
          setHeatmapData([[0]]);
        })
    }
  }, [iteration, layer, viewMode, meta])

  // 3. Render 3Dmol viewer
  useEffect(() => {
    if (pdbText && viewerRef.current && window.$3Dmol) {
      viewerRef.current.innerHTML = ''; 
      let viewer = window.$3Dmol.createViewer(viewerRef.current, { backgroundColor: '#1e1e1e' });
      viewer.addModel(pdbText, "pdb");
      
      let colorConfig = {};
      if (colorMode === 'spectrum') {
        colorConfig = { color: 'spectrum' };
      } else if (colorMode === 'confidence') {
        colorConfig = { colorscheme: { prop: 'b', gradient: 'rwb', min: 0.5, max: 1.0 } };
      }

      let styleObj = {};
      styleObj[renderStyle] = colorConfig;
      viewer.setStyle({}, styleObj);

      viewer.setClickable({}, true, function(atom) {
        viewer.removeAllLabels(); 
        viewer.addLabel(`${atom.resn} ${atom.resi}`, { 
          position: atom, backgroundColor: '#333333', fontColor: 'white', backgroundOpacity: 0.8 
        });
        viewer.render();
        
        // NEW: Tell React which residue number was clicked!
        setSelectedResidue(atom.resi);
      });

      viewer.zoomTo();
      viewer.render();
    }
  }, [pdbText, renderStyle, colorMode])

  // NEW: Calculate the dynamic crosshair shapes for Plotly
  let crosshairShapes = [];
  if (selectedResidue && heatmapData.length > 1) {
    const idx = selectedResidue - 1; // PDB is 1-indexed, JavaScript arrays are 0-indexed
    const maxIdx = heatmapData.length - 1;
    crosshairShapes = [
      { type: 'line', x0: idx, x1: idx, y0: 0, y1: maxIdx, line: { color: '#ff0055', width: 2, dash: 'dot' } }, // Vertical
      { type: 'line', x0: 0, x1: maxIdx, y0: idx, y1: idx, line: { color: '#ff0055', width: 2, dash: 'dot' } }  // Horizontal
    ];
  }

  return (
    <div className="dashboard-container">
      
      <div className="panel structure-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <h2 style={{ margin: 0 }}>Structure Viewer {meta && <span className="status-badge">Model: {meta.model_name}</span>}</h2>
          <div style={{ display: 'flex', gap: '10px' }}>
            <select value={renderStyle} onChange={(e) => setRenderStyle(e.target.value)} style={{ padding: '4px', backgroundColor: '#333', color: 'white', border: 'none', borderRadius: '4px' }}>
              <option value="cartoon">Cartoon</option>
              <option value="stick">Sticks</option>
              <option value="sphere">Surface (Spheres)</option>
            </select>
            <select value={colorMode} onChange={(e) => setColorMode(e.target.value)} style={{ padding: '4px', backgroundColor: '#333', color: 'white', border: 'none', borderRadius: '4px' }}>
              <option value="confidence">Color: Confidence</option>
              <option value="spectrum">Color: Rainbow</option>
            </select>
          </div>
        </div>
        <div ref={viewerRef} style={{ flex: 1, position: 'relative', border: '1px solid #333', borderRadius: '4px' }}></div>
      </div>

      <div className="panel trace-panel">
        <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginBottom: '10px' }}>
          <h2 style={{ margin: 0 }}>Trace Explorer</h2>
          {/* NEW: Display the currently selected residue */}
          {selectedResidue && <span style={{ color: '#ff0055', fontWeight: 'bold' }}>Target: Residue {selectedResidue}</span>}
          <select value={viewMode} onChange={(e) => setViewMode(e.target.value)} style={{ padding: '4px 8px', backgroundColor: '#333', color: 'white', border: 'none', borderRadius: '4px' }}>
            <option value="evolution">Trunk Evolution (s_z)</option>
            <option value="attention">ESM-2 Attention</option>
          </select>
        </div>
        
        <Plot
          data={[{ z: heatmapData, type: 'heatmap', colorscale: 'Viridis' }]}
          layout={{ 
            width: 500, height: 500, 
            paper_bgcolor: '#1e1e1e', plot_bgcolor: '#1e1e1e', font: { color: '#a0a0a0' },
            margin: { t: 30, r: 30, l: 30, b: 30 },
            xaxis: { title: 'Residue i' }, yaxis: { title: 'Residue j', autorange: 'reversed' },
            shapes: crosshairShapes // NEW: Injects the targeting laser into the chart!
          }}
          config={{ displayModeBar: false }}
        />
      </div>

      <div className="panel timeline-panel">
        {viewMode === 'evolution' ? (
          <>
            <h2>Recycling Iterations</h2>
            <input type="range" min="0" max={meta ? 3 : 0} value={iteration} onChange={(e) => setIteration(parseInt(e.target.value))} style={{ width: '60%', margin: '0 20px' }} />
            <span>Iteration: {iteration}</span>
          </>
        ) : (
          <>
            <h2>ESM-2 Transformer Layers</h2>
            <input type="range" min="0" max="35" value={layer} onChange={(e) => setLayer(parseInt(e.target.value))} style={{ width: '60%', margin: '0 20px' }} />
            <span>Layer: {layer}</span>
          </>
        )}
      </div>

    </div>
  )
}

export default App