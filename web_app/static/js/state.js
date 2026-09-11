// static/js/state.js

// Global State Management
window.AppState = {
    currentPredictionId: null,
    eventSource: null,
    currentModel: null,
    currentProteinMeta: null, // {protein_id, residue_idx}
    attnData: null,           // cached list from /viz/list
    attnType: 'msa_row',
    attnPollTimer: null,
    
    // Attention View States
    arcView: { xMin: 0, xMax: null },
    currentArcConnections: [],
    currentSequence: '',
    
    // Heatmap View States
    heatView: { x: 0, y: 0, w: null, h: null },
    lastHeatSource: null,
    lastHeatSeqLen: 0
};