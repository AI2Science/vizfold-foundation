#!/bin/bash
#SBATCH --job-name=esmf_vizfold
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/esmf_%j.out
#SBATCH --error=outputs/logs/esmf_%j.err
#
# ESMFold inference with VizFold trace export on ICE.
# Usage:
#   export FASTA=/path/to/seq.fasta OUTDIR=outputs/esmf_run
#   sbatch run_esmf_ice.slurm

#
# For a quick smoke test (<5 min): use a short FASTA and --trace_mode none

set -e

# Defaults if not set
FASTA="${FASTA:-examples/monomer/fasta_dir_6KWC/6KWC.fasta}"
OUTDIR="${OUTDIR:-outputs/esmf_ice_${SLURM_JOB_ID}}"
TRACE_MODE="${TRACE_MODE:-attention+activations}"
DEVICE="${DEVICE:-cuda}"

mkdir -p outputs/logs
mkdir -p "$OUTDIR"

cd "${OPENFOLD_HOME:-.}"

# Sanity check: fail fast if the environment is not set up
python -c "import torch; import transformers" 2>/dev/null || {
    echo "ERROR: torch or transformers not importable."
    echo "Activate the environment vizfold install esmfold created, then resubmit:"
    echo "  source \$(vizfold status | awk '/esmfold/ { print \$NF }')/bin/activate"
    exit 1
}

echo "FASTA=$FASTA OUTDIR=$OUTDIR TRACE_MODE=$TRACE_MODE"
echo "Running ESMFold..."

python scripts/esmfold/run_pretrained_esmf.py \
  --fasta "$FASTA" \
  --out "$OUTDIR" \
  --model facebook/esmfold_v1 \
  --device "$DEVICE" \
  --trace_mode "$TRACE_MODE" \
  --layers all \
  --save_fp16

echo "Done. Outputs in $OUTDIR"
