#!/usr/bin/env bash
set -eo pipefail

# Repo root
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# --------- User-configurable paths (override via env) ----------
SCR="${SCR:-/storage/ice1/2/0/$USER}" # ICE default (user-based), override if needed
BOLTZ_ENV="${BOLTZ_ENV:-$SCR/conda/envs/boltz}" # default to your boltz env on scratch
BOLTZ_CACHE="${BOLTZ_CACHE:-$SCR/boltz_cache}"
OUT_BASE="${OUT_BASE:-$SCR/issue42/outputs}"

# Inputs (kept in repo)
IN_YAML="${IN_YAML:-$REPO_ROOT/scripts/boltz/inputs/input.yaml}"
IN_FASTA="${IN_FASTA:-$REPO_ROOT/scripts/boltz/inputs/input.fasta}"

# Tracer lives in repo
TRACE_DIR="${TRACE_DIR:-$REPO_ROOT/boltz_trace}"

# --------- Trace config (sbatch --export can override) ----------
TRACE_HEAD="${BOLTZ_TRACE_HEAD:-all}"
TRACE_TOPK="${BOLTZ_TRACE_TOPK:-50}"
TRACE_RESIDUES="${BOLTZ_TRACE_RESIDUES:-18}"
TRACE_LAYERS="${BOLTZ_TRACE_LAYERS:-0}"
TRACE_DEBUG="${BOLTZ_TRACE_DEBUG:-1}"

RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
OUT_RUN="$OUT_BASE/$RUN_ID"
OUT_PRED="$OUT_RUN/pred"
OUT_ATTN="$OUT_RUN/attn_txt"
OUT_ARC="$OUT_RUN/arc_png"
OUT_ACT="$OUT_RUN/act_npz"
mkdir -p "$OUT_PRED" "$OUT_ATTN" "$OUT_ARC" "$OUT_ACT"

echo "[INFO] OUT_RUN=$OUT_RUN"
echo "[INFO] TRACE_HEAD=$TRACE_HEAD TRACE_TOPK=$TRACE_TOPK TRACE_RESIDUES=$TRACE_RESIDUES TRACE_LAYERS=$TRACE_LAYERS"

# Fail fast on missing inputs
[ -f "$IN_YAML" ]  || { echo "[ERROR] Missing IN_YAML:  $IN_YAML" >&2; exit 2; }
[ -f "$IN_FASTA" ] || { echo "[ERROR] Missing IN_FASTA: $IN_FASTA" >&2; exit 2; }

# Put caches on scratch
mkdir -p "$SCR"/{tmp,pip-cache,xdg-cache,hf-cache,torch-cache,wandb}
export TMPDIR="$SCR/tmp"
export PIP_CACHE_DIR="$SCR/pip-cache"
export XDG_CACHE_HOME="$SCR/xdg-cache"
export HF_HOME="$SCR/hf-cache"
export TORCH_HOME="$SCR/torch-cache"
export WANDB_DIR="$SCR/wandb"
export WANDB_MODE=offline

# ---- Load modules / conda reliably in batch ----
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi

module load anaconda3 >/dev/null 2>&1 || true

# make conda activate work in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Guard against empty env
if [ -z "${BOLTZ_ENV:-}" ]; then
  echo "[ERROR] BOLTZ_ENV is empty. Set BOLTZ_ENV or keep the default." >&2
  exit 2
fi

conda activate "$BOLTZ_ENV"

echo "[INFO] which python: $(which python)"
echo "[INFO] python: $(python -V)"
python -c "import torch; print('[INFO] torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

echo "[INFO] Running boltz predict..."
(
  export PYTHONPATH="$TRACE_DIR:${PYTHONPATH:-}"
  export BOLTZ_SAVE_ATTN=1
  export BOLTZ_TRACE_DIR="$OUT_ATTN"
  export BOLTZ_ACT_DIR="$OUT_ACT"

  export BOLTZ_TRACE_HEAD="$TRACE_HEAD"
  export BOLTZ_TRACE_TOPK="$TRACE_TOPK"
  export BOLTZ_TRACE_RESIDUES="$TRACE_RESIDUES"
  export BOLTZ_TRACE_LAYERS="$TRACE_LAYERS"
  export BOLTZ_TRACE_DEBUG="$TRACE_DEBUG"

  boltz predict "$IN_YAML" \
    --cache "$BOLTZ_CACHE" \
    --out_dir "$OUT_PRED" \
    --no_kernels \
    --seed 0 \
    --override
)

echo "[INFO] Attn txt:"
ls -l "$OUT_ATTN" || true
echo "[INFO] Act npz:"
ls -l "$OUT_ACT" || true

# Plot arc PNGs (no tracer on PYTHONPATH here)
export ATTN_DIR="$OUT_ATTN"
export REPO="$REPO_ROOT"
export OUT_DIR="$OUT_ARC"
export FASTA="$IN_FASTA"
export TRACE_LAYERS TRACE_RESIDUES TRACE_TOPK

python - <<'PY'
import os, sys

def parse_int_list(s, default):
    s = (s or "").strip()
    if not s:
        return default
    return [int(x) for x in s.split(",") if x.strip()]

sys.path.insert(0, os.environ["REPO"])
from visualize_attention_arc_diagram_demo_utils import generate_arc_diagrams, parse_fasta_sequence

attn_dir = os.environ["ATTN_DIR"]
out_dir  = os.environ["OUT_DIR"]
seq      = parse_fasta_sequence(os.environ["FASTA"])

layers   = parse_int_list(os.environ.get("TRACE_LAYERS", "0"), [0])
residues = parse_int_list(os.environ.get("TRACE_RESIDUES", "18"), [18])
topk     = int(os.environ.get("TRACE_TOPK", "50"))

print("[INFO] plotting layers=", layers, "residues=", residues, "topk=", topk)

for L in layers:
    generate_arc_diagrams(attn_dir, seq, out_dir, "BOLTZ",
                          attention_type="msa_row", top_k=topk, layer_idx=L)

for L in layers:
    for r in residues:
        generate_arc_diagrams(attn_dir, seq, out_dir, "BOLTZ",
                              attention_type="triangle_start", residue_indices=[r],
                              top_k=topk, layer_idx=L)
        generate_arc_diagrams(attn_dir, seq, out_dir, "BOLTZ",
                              attention_type="triangle_end", residue_indices=[r],
                              top_k=topk, layer_idx=L)

print("[INFO] Wrote arc PNGs to", out_dir)
PY

echo "[INFO] Arc PNGs count: $(find "$OUT_ARC" -maxdepth 1 -type f -name '*.png' | wc -l)"
ls -1 "$OUT_ARC" | head -n 50 || true

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || true)"
python - <<PY
import json, os, time
manifest = {
  "run_dir": "$OUT_RUN",
  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
  "repo": {"path": "$REPO_ROOT", "git_sha": "$GIT_SHA"},
  "inputs": {"yaml": "$IN_YAML", "fasta": "$IN_FASTA"},
  "outputs": {"pred": "$OUT_PRED", "attn_txt": "$OUT_ATTN", "act_npz": "$OUT_ACT", "arc_png": "$OUT_ARC"},
  "trace": {"head": "$TRACE_HEAD", "topk": "$TRACE_TOPK", "residues": "$TRACE_RESIDUES", "layers": "$TRACE_LAYERS"},
  "boltz": {"no_kernels": True, "seed": 0, "cache": "$BOLTZ_CACHE"},
}
with open(os.path.join("$OUT_RUN", "manifest.json"), "w") as f:
  json.dump(manifest, f, indent=2)
print("[INFO] wrote", os.path.join("$OUT_RUN", "manifest.json"))
PY

python scripts/boltz/validate_boltz_traces.py \
  --run_dir "$OUT_RUN" \
  --layers "$TRACE_LAYERS" \
  --residues "$TRACE_RESIDUES" \
  --expect_heads 4 || { echo "[FAIL] validate_boltz_traces.py" >&2; exit 1; }

echo "[PASS] validate_boltz_traces.py"

echo "[DONE] $OUT_RUN"