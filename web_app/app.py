import os
import subprocess
import time
import json
import glob
import shutil
import threading
import sys
import multiprocessing
import shlex
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.middleware.proxy_fix import ProxyFix

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

def stream_output(process, completion_payload, protein_id, prediction_id):
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
            payload = completion_payload() if callable(completion_payload) else completion_payload
            if not isinstance(payload, dict):
                payload = {'pdb_file': payload}
            payload = {
                'type': 'complete',
                'protein_id': protein_id,
                **payload,
            }
            yield f"data: {json.dumps(payload)}\n\n"

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


def _mmcif_to_pdb_atom_site_only(mmcif_path: str, pdb_path: str) -> None:
    """Convert a (Model)mmCIF file to PDB using only the _atom_site loop.

    This avoids external dependencies (gemmi/biopython). It is intended for visualization
    in 3Dmol and is not a fully general mmCIF->PDB converter.
    """
    cols = []
    rows = []
    in_loop = False
    in_atom_site = False

    def _finalize_atom_site():
        return

    with open(mmcif_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith('loop_'):
                in_loop = True
                in_atom_site = False
                cols = []
                rows = []
                continue

            if in_loop and line.startswith('_'):
                cols.append(line)
                if line.startswith('_atom_site.'):
                    in_atom_site = True
                continue

            if in_loop and (line.startswith('#') or line.startswith('loop_') or line.startswith('_')):
                if in_atom_site and cols and rows:
                    break
                if line.startswith('loop_'):
                    in_loop = True
                    in_atom_site = False
                    cols = []
                    rows = []
                else:
                    in_loop = False
                    in_atom_site = False
                continue

            if in_loop and in_atom_site and cols:
                try:
                    toks = shlex.split(line, posix=True)
                except ValueError:
                    toks = line.split()
                if len(toks) != len(cols):
                    continue
                rows.append(dict(zip(cols, toks)))

    if not rows:
        raise ValueError(f"No _atom_site records found in {mmcif_path}")

    def get(r, key, default=''):
        return r.get(key, default)

    with open(pdb_path, 'w') as out:
        serial = 1
        for r in rows:
            group = get(r, '_atom_site.group_PDB', 'ATOM')
            rec = 'HETATM' if group.upper().startswith('HET') else 'ATOM'
            atom_name = get(r, '_atom_site.auth_atom_id', '') or get(r, '_atom_site.label_atom_id', '')
            resn = get(r, '_atom_site.auth_comp_id', '') or get(r, '_atom_site.label_comp_id', '')
            chain = get(r, '_atom_site.auth_asym_id', '') or get(r, '_atom_site.label_asym_id', '')
            resseq = get(r, '_atom_site.auth_seq_id', '') or get(r, '_atom_site.label_seq_id', '')
            icode = get(r, '_atom_site.pdbx_PDB_ins_code', '')
            altloc = get(r, '_atom_site.label_alt_id', '')
            x = get(r, '_atom_site.Cartn_x', '0.0')
            y = get(r, '_atom_site.Cartn_y', '0.0')
            z = get(r, '_atom_site.Cartn_z', '0.0')
            occ = get(r, '_atom_site.occupancy', '1.00')
            b = get(r, '_atom_site.B_iso_or_equiv', '0.00')
            elem = get(r, '_atom_site.type_symbol', '').strip()

            if altloc in ('.', '?'):
                altloc = ''
            if icode in ('.', '?'):
                icode = ''
            if not chain or chain in ('.', '?'):
                chain = 'A'
            chain = chain[0]

            try:
                resseq_int = int(float(resseq))
            except Exception:
                resseq_int = 1

            an = atom_name.strip()
            if len(an) == 0:
                an = 'X'
            if len(an) < 4:
                an = an.rjust(4)
            else:
                an = an[:4]

            try:
                xf = float(x); yf = float(y); zf = float(z)
            except Exception:
                xf = yf = zf = 0.0
            try:
                occf = float(occ)
            except Exception:
                occf = 1.0
            try:
                bf = float(b)
            except Exception:
                bf = 0.0

            out.write(
                f"{rec:<6}{serial:>5} {an}{altloc:1}{resn:>3} {chain:1}{resseq_int:>4}{icode:1}   "
                f"{xf:>8.3f}{yf:>8.3f}{zf:>8.3f}{occf:>6.2f}{bf:>6.2f}          {elem:>2}\n"
            )
            serial += 1
        out.write("END\n")


def _ensure_viewable_structure(rel_path: str) -> str | None:
    """Ensure the structure is viewable in 3Dmol (prefer PDB). Converts mmCIF/ModelCIF to PDB if needed."""
    if not rel_path:
        return None
    base = os.path.abspath(app.config['UPLOAD_FOLDER'])
    abs_path = os.path.join(base, rel_path)
    if not os.path.exists(abs_path):
        return rel_path

    lower = rel_path.lower()
    if lower.endswith('.pdb'):
        return rel_path
    if lower.endswith('.cif') or lower.endswith('.mmcif'):
        pdb_rel = rel_path + '.pdb'
        pdb_abs = os.path.join(base, pdb_rel)
        try:
            if (not os.path.exists(pdb_abs)) or (os.path.getmtime(pdb_abs) < os.path.getmtime(abs_path)):
                os.makedirs(os.path.dirname(pdb_abs), exist_ok=True)
                _mmcif_to_pdb_atom_site_only(abs_path, pdb_abs)
            return pdb_rel
        except Exception as e:
            print(f"[WARN] mmCIF->PDB conversion failed for {abs_path}: {e}")
            return rel_path
    return rel_path

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


@app.route('/viz/generate', methods=['POST'])
def viz_generate():
    """Compatibility endpoint. The frontend calls this to trigger visualization generation."""
    return jsonify({'status': 'ok'})


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
        runner = request.args.get('runner', 'openfold').strip().lower()
    else:
        # Handle form submission
        sequence = request.form.get('sequence', '').strip()
        description = request.form.get('description', '').strip()
        protein_id = request.form.get('protein_id', 'demo').strip()
        prediction_id = request.form.get('prediction_id', None)
        runner = request.form.get('runner', 'openfold').strip().lower()
        try:
            residue_idx = int(request.form.get('residue_idx', 1))
        except (ValueError, TypeError):
            residue_idx = 1  # Default value if conversion fails
    
    if sequence == '':
        return jsonify({'error': 'Protein sequence is required'}), 400
    
    description = description if description else protein_id
    description_protein = description.split('|')[0]

    # ESMFold: return cached results immediately if a completed run already exists for this
    # protein_id + residue_idx.  This must happen before the fasta_exists dedup check below,
    # which would otherwise rename protein_id to "{id}_new" on every re-submission (because
    # the fasta dir from the first run persists), causing the ESMFold block to look in a
    # directory that doesn't exist and re-run the full model unnecessarily.
    if runner == 'esmfold':
        _base = os.path.abspath(app.config['UPLOAD_FOLDER'])
        _esmf_cached = os.path.join(_base, f'outputs/esmf_outputs_{protein_id}_demo_tri_{residue_idx}')
        _pdb_cached  = os.path.join(_esmf_cached, 'structure', 'predicted.pdb')
        if os.path.isfile(_pdb_cached):
            # Sync attention text files into the canonical attn_map_dir so /viz/list works.
            _attn_src = os.path.join(_esmf_cached, 'attention_files')
            _attn_dst = os.path.join(_base, f'outputs/attention_files_{protein_id}_demo_tri_{residue_idx}')
            if os.path.isdir(_attn_src):
                os.makedirs(_attn_dst, exist_ok=True)
                for _fname in os.listdir(_attn_src):
                    _s = os.path.join(_attn_src, _fname)
                    _d = os.path.join(_attn_dst, _fname)
                    if os.path.isfile(_s) and not os.path.exists(_d):
                        shutil.copy2(_s, _d)
            _pdb_rel = os.path.relpath(_pdb_cached, _base)
            _cached_payload = {'type': 'complete', 'protein_id': protein_id,
                               'pdb_file': _pdb_rel, 'residue_idx': residue_idx}

            def _cached_gen(_p=_cached_payload):
                yield f"data: {json.dumps({'type': 'output', 'message': 'ESMFold output already exists; using cached results.'})}\n\n"
                yield f"data: {json.dumps(_p)}\n\n"

            return Response(_cached_gen(), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache',
                                     'X-Accel-Buffering': 'no',
                                     'Connection': 'keep-alive'})

    fasta_exists = os.path.exists(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'fasta_{protein_id}'))
    output_exists = os.listdir(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}/predictions')) if os.path.exists(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}/predictions')) else []
    prot_old = protein_id
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
    boltz_out_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/boltz_outputs_{protein_id}_demo_tri_{residue_idx}')
    
    data_dir = os.path.realpath(os.path.expanduser(DATA_DIR))
    # Create output directories
    os.makedirs(attn_map_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(boltz_out_dir, exist_ok=True)

    runner = runner or 'openfold'
    if runner not in ('openfold', 'boltz', 'esmfold'):
        return jsonify({'error': f"Unknown runner '{runner}'. Choose 'openfold', 'boltz', or 'esmfold'."}), 400

    proc_env = os.environ.copy()

    if runner == 'openfold':
        # Ensure tools like jackhmmer can resolve shared libraries (e.g., libopenblas.so.0)
        # even when the web server is started outside an activated conda env.
        lib_dirs = []
        conda_prefix = proc_env.get('CONDA_PREFIX', '').strip()
        if conda_prefix:
            lib_dirs.append(os.path.join(conda_prefix, 'lib'))

        # Fallback: infer env prefix from the jackhmmer binary on PATH.
        try:
            jackhmmer_bin = shutil.which('jackhmmer', path=proc_env.get('PATH'))
        except Exception:
            jackhmmer_bin = None
        if jackhmmer_bin:
            # <env>/bin/jackhmmer -> <env>/lib
            env_prefix = os.path.dirname(os.path.dirname(os.path.realpath(jackhmmer_bin)))
            lib_dirs.append(os.path.join(env_prefix, 'lib'))

        # Fallback: use the interpreter prefix.
        try:
            lib_dirs.append(os.path.join(sys.prefix, 'lib'))
        except Exception:
            pass

        # Apply LD_LIBRARY_PATH update (preserve existing)
        lib_dirs = [d for d in lib_dirs if d and os.path.isdir(d)]
        if lib_dirs:
            prev_ld = proc_env.get('LD_LIBRARY_PATH', '')
            prev_parts = [p for p in prev_ld.split(':') if p] if prev_ld else []
            new_parts = []
            for d in lib_dirs:
                if d not in prev_parts and d not in new_parts:
                    new_parts.append(d)
            proc_env['LD_LIBRARY_PATH'] = ':'.join(new_parts + prev_parts)

        # Build the command
        cmd = [
            sys.executable, '-u', 'run_pretrained_openfold.py',
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
            '--cpus', str(max(multiprocessing.cpu_count() - 2, 1)),
            '--model_device', 'cuda:0',
            '--attn_map_dir', attn_map_dir,
            '--num_recycles_save', '1',
            '--triangle_residue_idx', str(residue_idx),
            '--demo_attn'
        ]

        if (os.path.exists(os.path.abspath(f'../examples/monomer/fasta_dir_{prot_old}'))):
            # Only enable precomputed alignments when the expected subdir exists.
            # Some examples store alignments under alignments/<id>_1 but the UI may use <id>.
            align_base = os.path.abspath('../examples/monomer/alignments')
            want_dir = os.path.join(align_base, prot_old)
            if not os.path.isdir(want_dir):
                try:
                    # Prefer an exact alias like <id>_1 if present.
                    cand = os.path.join(align_base, f"{prot_old}_1")
                    if os.path.isdir(cand):
                        # Create a symlink so OpenFold can find alignments/<id>
                        os.symlink(os.path.basename(cand), want_dir)
                    else:
                        # Otherwise, try any single matching prefix directory.
                        matches = [d for d in os.listdir(align_base) if d.startswith(prot_old + '_')]
                        if len(matches) == 1:
                            os.symlink(matches[0], want_dir)
                except Exception:
                    pass
            if os.path.isdir(want_dir):
                cmd.append('--use_precomputed_alignments')
                cmd.append(align_base)

        print(os.path.abspath(f'../examples/monomer/fasta_dir_{prot_old}'))

        process = subprocess.Popen(
            cmd,
            cwd='..',
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout to prevent blocking
            text=True,
            bufsize=1,  # line buffered
            universal_newlines=True,
            start_new_session=True  # Required for process group management
        )

        # Path to the expected PDB file
        pdb_file = f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}/predictions/{description_protein if description_protein else protein_id}_model_1_ptm_relaxed.pdb'
        completion_payload = {'pdb_file': pdb_file, 'residue_idx': residue_idx}
        output_folders = [fasta_dir, output_dir, attn_map_dir, image_output_dir]

    elif runner == 'boltz':
        # Boltz runner (Boltz-2 + VizFold tracer) integration.
        # We generate a minimal input.yaml for boltz predict and route trace outputs into `attn_map_dir`
        # to keep existing attention UI unchanged.
        boltz_input_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'boltz_inputs_{protein_id}')
        os.makedirs(boltz_input_dir, exist_ok=True)
        boltz_yaml_path = os.path.join(boltz_input_dir, 'input.yaml')
        boltz_fasta_path = os.path.join(boltz_input_dir, 'input.fasta')

        # Boltz expects YAML schema like scripts/boltz/inputs/input.yaml
        with open(boltz_yaml_path, 'w') as f:
            f.write('version: 1\n')
            f.write('sequences:\n')
            f.write('  - protein:\n')
            f.write('      id: A\n')
            f.write(f'      sequence: {sequence}\n')
            f.write('      msa: empty\n')

        with open(boltz_fasta_path, 'w') as f:
            f.write(f">{protein_id}\n{sequence}\n")

        # Wire tracer
        trace_dir = os.path.abspath(os.path.join(PROJECT_ROOT, 'boltz_trace'))
        proc_env['PYTHONPATH'] = f"{trace_dir}:{proc_env.get('PYTHONPATH', '')}" if proc_env.get('PYTHONPATH') else trace_dir
        proc_env['BOLTZ_SAVE_ATTN'] = '1'
        proc_env['BOLTZ_TRACE_DIR'] = attn_map_dir
        proc_env['BOLTZ_ACT_DIR'] = os.path.join(boltz_out_dir, 'act_npz')
        proc_env['BOLTZ_TRACE_HEAD'] = 'all'
        proc_env['BOLTZ_TRACE_TOPK'] = proc_env.get('BOLTZ_TRACE_TOPK', '50')
        proc_env['BOLTZ_TRACE_RESIDUES'] = str(residue_idx)
        # Default to OpenFold's typical evoformer stack depth so the layer slider is consistent.
        # Users can override by exporting BOLTZ_TRACE_LAYERS.
        if not proc_env.get('BOLTZ_TRACE_LAYERS'):
            proc_env['BOLTZ_TRACE_LAYERS'] = ','.join(str(i) for i in range(48))
        proc_env['BOLTZ_TRACE_DEBUG'] = proc_env.get('BOLTZ_TRACE_DEBUG', '1')

        boltz_cache = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), 'boltz_cache')
        os.makedirs(boltz_cache, exist_ok=True)
        pred_dir = os.path.join(boltz_out_dir, 'pred')
        os.makedirs(pred_dir, exist_ok=True)

        boltz_bin = proc_env.get('BOLTZ_BIN') or shutil.which('boltz')
        if not boltz_bin:
            return jsonify({
                'error': "Boltz runner selected but 'boltz' was not found on PATH. Activate the environment that has boltz installed, or set BOLTZ_BIN to the boltz executable path."
            }), 500

        cmd = [
            boltz_bin, 'predict', boltz_yaml_path,
            '--cache', boltz_cache,
            '--out_dir', pred_dir,
            '--no_kernels',
            '--seed', '0',
            '--override',
        ]

        process = subprocess.Popen(
            cmd,
            cwd='..',
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
            env=proc_env,
        )

        def _resolve_boltz_structure_payload():
            # Prefer PDB if present; otherwise use CIF.
            # We return the relative path under UPLOAD_FOLDER so /pdb/<path> can serve it.
            rel_base = os.path.relpath(os.path.abspath(app.config['UPLOAD_FOLDER']), os.path.abspath(app.config['UPLOAD_FOLDER']))
            _ = rel_base  # keep local variable usage explicit
            candidates = []
            for ext in ('*.pdb', '*.cif', '*.mmcif'):
                candidates.extend(glob.glob(os.path.join(pred_dir, '**', ext), recursive=True))
            candidates = [c for c in candidates if os.path.isfile(c)]
            if not candidates:
                # Fallback to directory listing on client side
                return {'pdb_file': None, 'residue_idx': residue_idx}
            # pick newest
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            chosen = candidates[0]
            rel = os.path.relpath(chosen, os.path.abspath(app.config['UPLOAD_FOLDER']))
            rel_view = _ensure_viewable_structure(rel)
            return {'pdb_file': rel_view, 'residue_idx': residue_idx}

        completion_payload = _resolve_boltz_structure_payload
        output_folders = [fasta_dir, boltz_input_dir, boltz_out_dir, attn_map_dir]

    else:
        # ESMfold runner via vizfold.backends.esmfold.ESMFoldRunner.
        # Runs run_pretrained_esmf.py as a subprocess so output streams to the client.
        # Attention text files land in esmf_out_dir/attention_files/ and are copied
        # into attn_map_dir/ on completion so the existing /viz/list endpoint works unchanged.
        esmf_out_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']),
                                    f'outputs/esmf_outputs_{protein_id}_demo_tri_{residue_idx}')
        os.makedirs(esmf_out_dir, exist_ok=True)

        # Expose vizfold package (lives at project root) to the subprocess
        proc_env['PYTHONPATH'] = (
            f"{PROJECT_ROOT}:{proc_env['PYTHONPATH']}" if proc_env.get('PYTHONPATH')
            else PROJECT_ROOT
        )

        device = proc_env.get('ESMF_DEVICE', 'cuda')
        trace_mode = proc_env.get('ESMF_TRACE_MODE', 'attention')
        top_k = proc_env.get('ESMF_TRACE_TOPK', '50')

        cmd = [
            sys.executable, '-u', 'run_pretrained_esmf.py',
            '--fasta', fasta_path,
            '--out', esmf_out_dir,
            '--device', device,
            '--trace_mode', trace_mode,
            '--top_k', top_k,
        ]

        process = subprocess.Popen(
            cmd,
            cwd='..',
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )

        def _resolve_esmf_payload():
            # Copy attention text files into attn_map_dir so /viz/list can find them.
            esmf_attn_src = os.path.join(esmf_out_dir, 'attention_files')
            if os.path.isdir(esmf_attn_src):
                os.makedirs(attn_map_dir, exist_ok=True)
                for fname in os.listdir(esmf_attn_src):
                    src = os.path.join(esmf_attn_src, fname)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(attn_map_dir, fname))
            # Locate the PDB written by ESMFoldRunner.
            pdb_path = os.path.join(esmf_out_dir, 'structure', 'predicted.pdb')
            if os.path.isfile(pdb_path):
                pdb_rel = os.path.relpath(pdb_path, os.path.abspath(app.config['UPLOAD_FOLDER']))
                return {'pdb_file': pdb_rel, 'residue_idx': residue_idx}
            return {'pdb_file': None, 'residue_idx': residue_idx}

        completion_payload = _resolve_esmf_payload
        output_folders = [fasta_dir, esmf_out_dir, attn_map_dir]

    # Store the process in the global dictionary
    running_processes[prediction_id] = {
        'process': process,
        'start_time': time.time(),
        'output_folders': output_folders
    }

    # Return the streaming response
    return stream_output(process, completion_payload, protein_id, prediction_id)

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
            protein_data['runner'] = 'openfold'
            proteins.append(protein_data)

    # Also include Boltz predictions, which do not follow OpenFold's output directory convention.
    # We scan for boltz output directories and construct entries compatible with the frontend.
    base = os.path.abspath(app.config['UPLOAD_FOLDER'])
    boltz_dirs = glob.glob(os.path.join(base, 'outputs', 'boltz_outputs_*_demo_tri_*'))
    seen = {p.get('protein_id') for p in proteins if isinstance(p, dict)}
    for bdir in boltz_dirs:
        try:
            bname = os.path.basename(bdir)
            # boltz_outputs_{protein_id}_demo_tri_{residue_idx}
            if not bname.startswith('boltz_outputs_'):
                continue
            parts = bname.split('_demo_tri_')
            if len(parts) != 2:
                continue
            protein_id = parts[0].replace('boltz_outputs_', '')
            residue_idx = int(parts[1])

            # Find the FASTA we saved during /process (preferred), else skip
            fasta_path = os.path.join(base, f'fasta_{protein_id}', f'{protein_id}.fasta')
            if not os.path.exists(fasta_path):
                continue
            with open(fasta_path, 'r') as f:
                content = f.read().strip()
            if not content:
                continue
            lines = content.splitlines()
            description = lines[0][1:] if lines[0].startswith('>') else protein_id
            sequence = ''.join([ln.strip() for ln in lines[1:] if ln.strip() and not ln.startswith('>')])

            # Resolve structure file under pred/**
            pred_dir = os.path.join(bdir, 'pred')
            candidates = []
            for ext in ('*.pdb', '*.cif', '*.mmcif'):
                candidates.extend(glob.glob(os.path.join(pred_dir, '**', ext), recursive=True))
            candidates = [c for c in candidates if os.path.isfile(c)]
            if not candidates:
                continue
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            chosen = candidates[0]
            pdb_file = os.path.relpath(chosen, base)
            pdb_file = _ensure_viewable_structure(pdb_file)

            if protein_id in seen:
                continue
            proteins.append({
                'protein_id': protein_id,
                'description': description,
                'sequence': sequence,
                'residue_idx': residue_idx,
                'pdb_file': pdb_file,
                'runner': 'boltz',
            })
            seen.add(protein_id)
        except Exception as e:
            print(f"[WARN] Failed parsing boltz output dir {bdir}: {e}")

    # Also include ESMfold predictions.
    esmf_dirs = glob.glob(os.path.join(base, 'outputs', 'esmf_outputs_*_demo_tri_*'))
    for edir in esmf_dirs:
        try:
            ename = os.path.basename(edir)
            if not ename.startswith('esmf_outputs_'):
                continue
            parts = ename.split('_demo_tri_')
            if len(parts) != 2:
                continue
            protein_id = parts[0].replace('esmf_outputs_', '')
            residue_idx = int(parts[1])

            fasta_path = os.path.join(base, f'fasta_{protein_id}', f'{protein_id}.fasta')
            if not os.path.exists(fasta_path):
                continue
            with open(fasta_path, 'r') as f:
                content = f.read().strip()
            if not content:
                continue
            lines = content.splitlines()
            description = lines[0][1:] if lines[0].startswith('>') else protein_id
            sequence = ''.join([ln.strip() for ln in lines[1:] if ln.strip() and not ln.startswith('>')])

            pdb_path = os.path.join(edir, 'structure', 'predicted.pdb')
            if not os.path.isfile(pdb_path):
                continue
            pdb_file = os.path.relpath(pdb_path, base)

            if protein_id in seen:
                continue
            proteins.append({
                'protein_id': protein_id,
                'description': description,
                'sequence': sequence,
                'residue_idx': residue_idx,
                'pdb_file': pdb_file,
                'runner': 'esmfold',
            })
            seen.add(protein_id)
        except Exception as e:
            print(f"[WARN] Failed parsing esmf output dir {edir}: {e}")

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
    app.run(host='0.0.0.0', port=9000, debug=True)
