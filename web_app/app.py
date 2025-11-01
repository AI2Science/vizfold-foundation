import os
import subprocess
import time
import json
import glob
import shutil
import threading
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

# Ensure project root is importable when running from web_app
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './web_tmp_dir'
# Store running processes
running_processes = {}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
DATA_DIR = "/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data"

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def stream_output(process, pdb_file, protein_id, prediction_id):
    """Stream output from subprocess to client."""
    def generate():
        # Stream stdout line-by-line in text mode
        for line in iter(process.stdout.readline, ''):
            line_str = line.rstrip('\n')
            if line_str:
                yield f"data: {json.dumps({'type': 'output', 'message': line_str})}\n\n"
        
        # Check for errors
        process.wait()
            
        if process.returncode != 0:
            for folder in running_processes[prediction_id]['output_folders']:
                if os.path.exists(folder):
                    try:
                        shutil.rmtree(folder)
                    except Exception as e:
                        print(f'Error removing folder {folder}: {str(e)}')
            # Check if process was killed (negative return code indicates termination by signal)
            if process.returncode < 0:
                # Process was cancelled/terminated
                yield f"data: {json.dumps({'type': 'cancelled', 'message': 'Prediction was cancelled'})}\n\n"
            else:
                # Process had a normal error, try to read stderr
                try:
                    error = process.stderr.read()
                    yield f"data: {json.dumps({'type': 'error', 'message': error})}\n\n"
                except Exception:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred while reading the error output'})}\n\n"
        else:                                             
            # Send completion message with PDB file info
            yield f"""data: {json.dumps({
                'type': 'complete',
                'pdb_file': pdb_file,
                'protein_id': protein_id
            })}\n\n"""

        # Clean up the process from the running processes
        if prediction_id in running_processes:
            del running_processes[prediction_id]
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # for nginx
            'Connection': 'keep-alive'
        }
    )

def parse_fasta_file(fasta_path):
    """Parse a FASTA file and return protein details."""
    all_outputs = os.listdir(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), 'outputs'))
    with open(fasta_path, 'r') as f:
        content = f.read().strip()
        if not content:
            return None
        lines = content.split('\n')
        protein_id = fasta_path.split('/')[-1].split('.')[0]
        description = lines[0][1:]
        description_protein = description.split('|')[0]
        sequence = ''.join(lines[1:])
        residue_idx = -1
        for i in all_outputs:
            if i.startswith(f'my_outputs_align_{protein_id}_demo_tri_'):
                residue_idx = i.split('_')[-1]
                break
        if (residue_idx == -1):
            return None
        pdb_file = f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}/predictions/{description_protein if description_protein else protein_id}_model_1_ptm_relaxed.pdb'
        if not os.path.exists(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), pdb_file)):
            return None
        return {
            'protein_id': protein_id,
            'description': description,
            'sequence': sequence,
            'residue_idx': int(residue_idx),
            'pdb_file': pdb_file
        }

def get_paths_for_protein(protein_id, residue_idx):
    base = os.path.abspath(app.config['UPLOAD_FOLDER'])
    attn_map_dir = os.path.join(base, f'outputs/attention_files_{protein_id}_demo_tri_{residue_idx}')
    return attn_map_dir

def detect_available_layers(attn_map_dir, residue_idx):
    layers_msa = set()
    layers_tri = set()
    if os.path.isdir(attn_map_dir):
        for fname in os.listdir(attn_map_dir):
            if fname.startswith('msa_row_attn_layer') and fname.endswith('.txt'):
                try:
                    L = int(fname.replace('msa_row_attn_layer', '').replace('.txt', ''))
                    layers_msa.add(L)
                except ValueError:
                    pass
            if fname.startswith('triangle_start_attn_layer') and f"residue_idx_{residue_idx}" in fname and fname.endswith('.txt'):
                try:
                    core = fname.split('triangle_start_attn_layer')[1]
                    L = int(core.split('_')[0])
                    layers_tri.add(L)
                except Exception:
                    pass
    return sorted(layers_msa), sorted(layers_tri)


@app.route('/cancel_prediction', methods=['POST'])
def cancel_prediction():
    prediction_id = request.json.get('prediction_id')
    if not prediction_id:
        return jsonify({'status': 'error', 'message': 'No prediction ID provided'}), 400
        
    if prediction_id in running_processes:
        process_info = running_processes[prediction_id]
        try:
            # Terminate the process group to ensure all child processes are killed
            if process_info['process'].poll() is None:  # Check if process is still running
                try:
                    # Try to terminate gracefully first
                    os.killpg(os.getpgid(process_info['process'].pid), 15)  # SIGTERM
                    
                    # Wait a bit for the process to terminate
                    try:
                        process_info['process'].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate
                        os.killpg(os.getpgid(process_info['process'].pid), 9)  # SIGKILL
                    return jsonify({
                        'status': 'cancelled',
                        'message': 'Prediction was successfully cancelled'
                    })
                except ProcessLookupError:
                    # Process already terminated
                    pass
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'Error cancelling prediction: {str(e)}'
                    }), 500
            
            # Clean up
            del running_processes[prediction_id]
            return jsonify({'status': 'already_done', 'message': 'Process was already terminated'})
            
        except Exception as e:
            # Clean up even if there was an error
            if prediction_id in running_processes:
                del running_processes[prediction_id]
            return jsonify({
                'status': 'error',
                'message': f'Error cancelling prediction: {str(e)}'
            }), 500
    
    return jsonify({'status': 'not_found', 'message': 'No running prediction found with that ID'})

@app.route('/process', methods=['GET', 'POST'])
def process():
    prediction_id = None
    if request.method == 'GET':
        # Handle SSE connection
        sequence = request.args.get('sequence', '').strip()
        description = request.args.get('description', '').strip()
        residue_idx = int(request.args.get('residue_idx', 1))
        protein_id = request.args.get('protein_id', 'demo').strip()
        prediction_id = request.args.get('prediction_id', None)
    else:
        # Handle form submission
        sequence = request.form.get('sequence', '').strip()
        description = request.form.get('description', '').strip()
        protein_id = request.form.get('protein_id', 'demo').strip()
        prediction_id = request.form.get('prediction_id', None)
        try:
            residue_idx = int(request.form.get('residue_idx', 1))
        except (ValueError, TypeError):
            residue_idx = 1  # Default value if conversion fails
    
    if sequence == '':
        return jsonify({'error': 'Protein sequence is required'}), 400
    
    description = description if description else protein_id
    description_protein = description.split('|')[0]
    fasta_exists = os.path.exists(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'fasta_{protein_id}'))
    output_exists = os.listdir(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}/predictions')) if os.path.exists(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}/predictions')) else []
    if fasta_exists or len(output_exists) > 0:
        protein_id = f"{protein_id}_new"
    # Format FASTA content
    fasta_content = f">{description}\n{sequence}"
    
    # Run OpenFold in a subprocess with streaming output
    fasta_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'fasta_{protein_id}')
    os.makedirs(fasta_dir, exist_ok=True)
    
    # Save FASTA file
    fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
    with open(fasta_path, 'w') as f:
        f.write(fasta_content)
    
    # Define output directories
    attn_map_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/attention_files_{protein_id}_demo_tri_{residue_idx}')
    output_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}')
    image_output_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/attention_images_{protein_id}_demo_tri_{residue_idx}')
    
    data_dir = os.path.realpath(os.path.expanduser(DATA_DIR))
    # Create output directories
    os.makedirs(attn_map_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    
    #Build the command
    cmd = [
        'python3', '-u', 'run_pretrained_openfold.py',
        fasta_dir,
        f'{data_dir}/pdb_mmcif/mmcif_files',
        '--output_dir', output_dir,
        '--config_preset', 'model_1_ptm',
        '--uniref90_database_path', f'{data_dir}/uniref90/uniref90.fasta',
        '--mgnify_database_path', f'{data_dir}/mgnify/mgy_clusters_2022_05.fa',
        '--pdb70_database_path', f'{data_dir}/pdb70/pdb70',
        '--uniclust30_database_path', f'{data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08',
        '--bfd_database_path', f'{data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
        '--save_outputs',
        '--cpus', '8',
        '--model_device', 'cuda:0',
        '--attn_map_dir', attn_map_dir,
        '--num_recycles_save', '1',
        '--triangle_residue_idx', str(residue_idx),
        '--demo_attn'
    ]

    if (os.path.exists(os.path.abspath(f'./examples/monomer/alignments/fasta_dir_{protein_id}'))):
        cmd.append('--use_precomputed_alignments')
        cmd.append(os.path.abspath('./examples/monomer/alignments'))

    # Store the process
    process = subprocess.Popen(
        cmd,
        cwd='..',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout to prevent blocking
        text=True,
        bufsize=1,  # line buffered
        universal_newlines=True,
        start_new_session=True  # Required for process group management
    )
    
    # Store the process in the global dictionary
    running_processes[prediction_id] = {
        'process': process,
        'start_time': time.time(),
        'output_folders': [
            fasta_dir,
            output_dir,
            attn_map_dir,
            image_output_dir
        ]
    }
    
    # Path to the expected PDB file
    pdb_file = f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}/predictions/{description_protein if description_protein else protein_id}_model_1_ptm_relaxed.pdb'
    # Return the streaming response
    return stream_output(process, pdb_file, protein_id, prediction_id)

@app.route('/pdb/<path:filename>')
def serve_pdb(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], os.path.join('outputs', filename))

@app.route('/proteins')
def list_proteins():
    """List all available proteins from FASTA files."""
    fasta_files = glob.glob(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), 'fasta_*/*.fasta'))
    print(fasta_files)
    proteins = []
    for fasta_file in fasta_files:
        protein_data = parse_fasta_file(fasta_file)
        if protein_data:
            proteins.append(protein_data)
    return jsonify(proteins)

@app.route('/viz/list')
def list_viz():
    protein_id = request.args.get('protein_id')
    residue_idx = request.args.get('residue_idx', type=int)
    if not protein_id or residue_idx is None:
        return jsonify({'error': 'protein_id and residue_idx required'}), 400

    attn_map_dir = get_paths_for_protein(protein_id, residue_idx)
    layers_msa, layers_tri = detect_available_layers(attn_map_dir, residue_idx)

    def arc_png_path(attn_type, L):
        if attn_type == 'msa_row':
            fname = f"msa_row_head_0_layer_{L}_{protein_id}_arc.png"
            # Note: multiple heads exist; frontend can pick head; here we default head 0
            path = os.path.join('attention_images_' + f"{protein_id}_demo_tri_{residue_idx}", 'msa_row_attention_plots', fname)
        else:
            fname = f"tri_start_res_{residue_idx}_head_0_layer_{L}_{protein_id}_arc.png"
            path = os.path.join('attention_images_' + f"{protein_id}_demo_tri_{residue_idx}", 'tri_start_attention_plots', fname)
        return f"/outputs/{path}"

    def heatmap_html_path(attn_type, L):
        if attn_type == 'msa_row':
            fname = f"msa_row_layer{L}_heatmap_grid.html"
        else:
            fname = f"triangle_start_layer{L}_res{residue_idx}_heatmap_grid.html"
        path = os.path.join('attention_images_' + f"{protein_id}_demo_tri_{residue_idx}", 'heatmaps', fname)
        return f"/outputs/{path}"

    def attn_file_path(attn_type, L):
        if attn_type == 'msa_row':
            fname = f"msa_row_attn_layer{L}.txt"
        else:
            fname = f"triangle_start_attn_layer{L}_residue_idx_{residue_idx}.txt"
        path = os.path.join(f"attention_files_{protein_id}_demo_tri_{residue_idx}", fname)
        return f"/outputs/{path}"

    result = {
        'protein_id': protein_id,
        'residue_idx': residue_idx,
        'msa_row': {
            'layers': layers_msa,
            'assets': {str(L): {
                'attn_file_url': attn_file_path('msa_row', L),
            } for L in layers_msa}
        },
        'triangle_start': {
            'layers': layers_tri,
            'assets': {str(L): {
                'attn_file_url': attn_file_path('triangle_start', L),
            } for L in layers_tri}
        }
    }
    return jsonify(result)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
