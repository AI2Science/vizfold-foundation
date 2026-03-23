# Boltz inference and tracing (VizFold)

This doc describes how to run **Boltz-2** inference and extract **attention-style traces** in the same text format used by VizFold’s arc visualization utilities.

The repo ships a small example input under `scripts/boltz/inputs/` for validation. To run on your own target, override `IN_YAML` / `IN_FASTA` (see below).

---

## Quickstart (ICE / coe-gpu H100)

> **Goal:** create the env on **scratch** (not `$HOME`), then submit a trace job that produces `attn_txt/` + `arc_png/` and passes validation.

### 1) Create the environment (one-time)

```bash
cd "$(git rev-parse --show-toplevel)"

# scratch location (recommended)
export SCR=/storage/ice1/2/0/$USER

# RECOMMENDED: use a fresh env prefix to avoid RDKit conflicts from older installs
export ENV="$SCR/conda/envs/boltz_clean_fresh"
rm -rf "$ENV"

# Build the env
bash scripts/boltz/setup_boltz_env.sh

# ---- Sanity check: RDKit must import compiled symbols (rdBase + Mol) ----
# If this fails, your env is inconsistent; rebuild with a new prefix.
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi
module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

python - <<'PY'
from rdkit import rdBase
from rdkit.Chem import Mol
print("RDKit OK:", rdBase.rdkitVersion)
PY
```

### 2) Run the example trace job
```bash
# Use the env you just created
export BOLTZ_ENV="$ENV"

JOBID=$(sbatch --export=ALL scripts/boltz/run_boltz_trace.sbatch | awk '{print $4}')
echo "JOBID=$JOBID"
```

### 3) Verify it completed + find outputs
```bash
# status (wait until COMPLETED)
sacct -j "$JOBID" --format=JobID,State,ExitCode,Elapsed -n

# see the run directory path + pass/fail
tail -n 80 outputs/boltz_runs/slurm-${JOBID}.out
tail -n 120 outputs/boltz_runs/slurm-${JOBID}.err

# extract OUT_RUN from slurm output (copy/paste-safe)
OUT_RUN=$(grep -m1 '^\[INFO\] OUT_RUN=' outputs/boltz_runs/slurm-${JOBID}.out | sed 's/.*OUT_RUN=//')
echo "OUT_RUN=$OUT_RUN"

# guard: if empty, stop (usually means job not finished yet or slurm out path mismatch)
[ -n "$OUT_RUN" ] || { echo "[ERROR] OUT_RUN is empty. Check outputs/boltz_runs/slurm-${JOBID}.out" >&2; exit 2; }

# quick counts
find "$OUT_RUN/attn_txt" -name "*.txt" | wc -l
find "$OUT_RUN/act_npz"  -name "*.npz" | wc -l
find "$OUT_RUN/arc_png"  -name "*.png" | wc -l
```

## Environment notes (ICE/H100)
- Boltz runs on GPU.
- On ICE/H100, run with `--no_kernels` to avoid CUDA kernel / cuBLAS symbol issues (the provided runner already uses this).

## Output layout
A run produces:
- `pred/` : structure prediction outputs (e.g., CIF, pLDDT, PAE)
- `attn_txt/` : attention-style trace text files
- `arc_png/` : arc diagram PNGs generated from `attn_txt/`
- `act_npz/` : lightweight activation summaries (optional; only written when `BOLTZ_ACT_DIR` is set, which the provided runner does)

## Attention-weight availability + proxy fallback

Boltz versions differ in whether they expose **true attention weights** during inference. The tracer tries to capture attention weights when they exist, but can fall back to a proxy so that `attn_txt/` files and `arc_png/` visualizations are still produced.

### When true attention weights are available
If the underlying attention module exposes an attention tensor that includes a **head dimension** (e.g., something that can be interpreted as `(H, N, N)` after reshaping/averaging), the tracer exports **multi-head** traces for each selected layer:
- `msa_row_attn_layer{L}.txt` contains multiple `Layer {L} Head {h}` blocks
- Triangle start/end files contain multiple `Layer {L} Head {h}` blocks

### When true attention weights are NOT available (proxy mode)
Some Boltz versions do not expose a usable attention-weight tensor through the attention modules/hooks. In that case, the tracer falls back to **proxy weights** derived from the TriangleAttention output so downstream visualization still works.

Proxy mode works like this:
- It looks for a square pair representation shaped like `(N, N, C)` (or `(B, N, N, C)`) in the module output.
- It computes the **L2 norm across channels** to convert it into a single `(N, N)` weight matrix and normalizes it.
- This yields a **single-head** weight matrix with shape `(1, N, N)`.

Important notes about proxy mode:
- The “Head {h}” label in trace files is **not a real attention head** in proxy mode (it is a derived single-head view).
- In proxy mode, `msa_row_attn_layer{L}.txt` is still generated for **format compatibility**, but it is also derived from the triangle output (not true MSA-row attention).
- Proxy weights are intended for **visualization/diagnostics and format compatibility**, not as a perfect replacement for true attention weights.

## Trace text formats

All trace files use repeated blocks of:
- header line: `Layer {L} Head {h}`
- followed by edges: `i j weight`

### MSA row attention
- File: `msa_row_attn_layer{L}.txt`
- Contents:
  - Multi-head blocks when true weights exist
  - A single proxy head block when proxy mode is used

### Triangle attention (per residue)
- Files:
  - `triangle_start_attn_layer{L}_residue_idx_{r}.txt`
  - `triangle_end_attn_layer{L}_residue_idx_{r}.txt`
- Contents:
  - Multi-head blocks when true weights exist
  - A single proxy head block when proxy mode is used
- Edge format:
  - `i j weight` (with `i == r` for the row-based triangle traces)

## Configuration knobs

### Inputs
Default example inputs live here:
- `scripts/boltz/inputs/input.yaml`
- `scripts/boltz/inputs/input.fasta`

To run on your own target, override:
```bash
export IN_YAML=/path/to/your_input.yaml
export IN_FASTA=/path/to/your_input.fasta
```
