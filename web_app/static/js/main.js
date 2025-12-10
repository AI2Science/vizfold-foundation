// static/js/main.js

function cancelPrediction() {
    if (!window.AppState.currentPredictionId) {
        console.log('No prediction to cancel');
        return;
    }

    const cancelButton = document.getElementById('cancel-btn');
    const runButton = document.getElementById('run-btn');
    const outputElement = document.getElementById('command-output');

    cancelButton.disabled = true;

    fetch('/cancel_prediction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prediction_id: window.AppState.currentPredictionId })
    })
    .then(response => {
        if (!response.ok) throw new Error('Network response was not ok');
        return response.json();
    })
    .then(data => {
        if (data.status === 'cancelled' || data.status === 'already_done') {
            outputElement.textContent += '\n' + (data.message || 'Prediction was cancelled.') + '\n';
            if (window.AppState.eventSource) {
                window.AppState.eventSource.close();
                window.AppState.eventSource = null;
            }
            document.getElementById('loading').style.display = 'none';
            runButton.disabled = false;
            cancelButton.style.display = 'none';
            cancelButton.disabled = false;
            window.AppState.currentPredictionId = null;
        } else {
            outputElement.textContent += '\nError: ' + (data.message || 'Failed to cancel prediction') + '\n';
            cancelButton.disabled = false;
        }
    })
    .catch(error => {
        console.error('Error cancelling prediction:', error);
        outputElement.textContent += '\nError cancelling prediction: ' + error.message + '\n';
        cancelButton.disabled = false;
        if (window.AppState.eventSource) {
            window.AppState.eventSource.close();
            window.AppState.eventSource = null;
        }
        document.getElementById('loading').style.display = 'none';
        runButton.disabled = false;
        cancelButton.style.display = 'none';
        window.AppState.currentPredictionId = null;
    });
}

function submitForm(event) {
    event.preventDefault();
    const form = document.getElementById('protein-form');
    const loading = document.getElementById('loading');
    const outputElement = document.getElementById('command-output');
    const viewerControls = document.getElementById('viewer-controls');
    const runButton = document.getElementById('run-btn');
    const cancelButton = document.getElementById('cancel-btn');

    outputElement.textContent = 'Starting prediction...\n';
    
    // Manage Collapses (using Bootstrap API)
    new bootstrap.Collapse(document.getElementById('outputCollapse'), { toggle: false }).show();
    new bootstrap.Collapse(document.getElementById('threeDCollapse'), { toggle: false }).hide();
    new bootstrap.Collapse(document.getElementById('attnCollapse'), { toggle: false }).hide();

    loading.style.display = 'block';
    viewerControls.style.display = 'none';
    runButton.disabled = true;
    cancelButton.style.display = 'inline-block';

    window.AppState.currentPredictionId = Date.now().toString();
    const formData = new FormData(form);

    if (window.AppState.eventSource) window.AppState.eventSource.close();

    formData.append('prediction_id', window.AppState.currentPredictionId);
    const params = new URLSearchParams();
    for (const [key, value] of formData.entries()) {
        params.append(key, value);
    }

    window.AppState.eventSource = new EventSource(`/process?${params.toString()}`);

    window.AppState.eventSource.onmessage = function (event) {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'output') {
                outputElement.textContent += data.message + '\n';
                outputElement.scrollTop = outputElement.scrollHeight;
            } else if (data.type === 'error') {
                outputElement.textContent += '\nError: ' + data.message + '\n';
                stopProcessing();
            } else if (data.type === 'complete') {
                outputElement.textContent += '\nPrediction complete! Loading structure...\n';
                stopProcessing();
                
                // Show output sections
                new bootstrap.Collapse(document.getElementById('outputCollapse'), { toggle: false }).hide();
                new bootstrap.Collapse(document.getElementById('threeDCollapse'), { toggle: false }).show();
                new bootstrap.Collapse(document.getElementById('attnCollapse'), { toggle: false }).show();

                if (data.pdb_file) {
                    window.loadPDB(data.pdb_file);
                }
                
                const protId = document.getElementById('protein-id').value;
                const resIdx = parseInt(document.getElementById('residue-idx').value || '0');
                window.AppState.currentProteinMeta = { protein_id: protId, residue_idx: resIdx };
                window.initAttentionUI();
            }
        } catch (e) {
            console.error('Error processing message:', e);
            stopProcessing();
        }
    };

    window.AppState.eventSource.onerror = function () {
        outputElement.textContent += '\nConnection error/closed.\n';
        stopProcessing();
    };

    function stopProcessing() {
        loading.style.display = 'none';
        runButton.disabled = false;
        cancelButton.style.display = 'none';
        if (window.AppState.eventSource) {
            window.AppState.eventSource.close();
            window.AppState.eventSource = null;
        }
        window.AppState.currentPredictionId = null;
    }
}

// ----------------- Protein List Logic ----------------- //

async function loadProteins() {
    try {
        const response = await fetch('/proteins');
        const proteins = await response.json();
        const dropdownMenu = document.getElementById('protein-dropdown-menu');
        
        dropdownMenu.innerHTML = '';
        if (proteins.length === 0) {
            dropdownMenu.innerHTML = '<li><a class="dropdown-item disabled" href="#">No proteins available</a></li>';
            return;
        }

        proteins.forEach(protein => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.className = 'dropdown-item';
            a.href = '#';
            a.textContent = `${protein.protein_id}`;
            a.onclick = () => selectProtein(protein);
            li.appendChild(a);
            dropdownMenu.appendChild(li);
        });

        const li = document.createElement('li');
        const a = document.createElement('a');
        a.className = 'dropdown-item';
        a.href = '#';
        a.textContent = `Clear`;
        a.onclick = () => clearProtein();
        li.appendChild(a);
        dropdownMenu.appendChild(li);

    } catch (error) {
        console.error('Error loading proteins:', error);
    }
}

function selectProtein(protein) {
    document.getElementById('proteinDropdown').textContent = `${protein.protein_id}`;
    document.getElementById('protein-id').value = protein.protein_id;
    document.getElementById('description').value = protein.description;
    document.getElementById('sequence').value = protein.sequence;
    document.getElementById('residue-idx').value = protein.residue_idx;
    
    window.loadPDB(protein.pdb_file);
    window.AppState.currentProteinMeta = { protein_id: protein.protein_id, residue_idx: protein.residue_idx };
    window.initAttentionUI();
    
    new bootstrap.Collapse(document.getElementById('outputCollapse'), { toggle: false }).hide();
    new bootstrap.Collapse(document.getElementById('threeDCollapse'), { toggle: false }).show();
    new bootstrap.Collapse(document.getElementById('attnCollapse'), { toggle: false }).show();
}

function clearProtein() {
    document.getElementById('proteinDropdown').textContent = `Select a protein`;
    document.getElementById('protein-id').value = '';
    document.getElementById('description').value = '';
    document.getElementById('sequence').value = '';
    document.getElementById('residue-idx').value = '';
    if(window.viewer) window.viewer.clear();
    document.getElementById('viewer-controls').style.display = 'none';
    
    window.AppState.currentProteinMeta = null;
    window.AppState.attnData = null;
    
    // Hide sections
    new bootstrap.Collapse(document.getElementById('outputCollapse'), { toggle: false }).hide();
    new bootstrap.Collapse(document.getElementById('threeDCollapse'), { toggle: false }).hide();
    new bootstrap.Collapse(document.getElementById('attnCollapse'), { toggle: false }).hide();
}

// Init
document.addEventListener('DOMContentLoaded', loadProteins);

// Expose to window for HTML onclick handlers
window.cancelPrediction = cancelPrediction;
window.submitForm = submitForm;