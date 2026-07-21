#!/bin/bash
# Fold one sequence on a GPU. Site-independent: every path comes from OPENFOLD_*,
# and install.sh prints the invocation for the cluster it installed on.
#
#   OPENFOLD_PREFIX=<prefix> run/fold.sh 6KWC_1
set -euo pipefail

PREFIX=${OPENFOLD_PREFIX:-$HOME/openfold}
REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
ENV_NAME=${OPENFOLD_ENV_NAME:-openfold-env}

INPUT_ID=${1:-${OPENFOLD_INPUT_ID:-6KWC_1}}
[ $# -gt 0 ] && shift                     # the rest pass through to the python script
DATA=${OPENFOLD_DATA_DIR:-$PREFIX/data}
FASTA_DIR=${OPENFOLD_FASTA_DIR:-$REPO/examples/monomer/fasta_dir_${INPUT_ID%_*}}
ALIGNMENT_DIR=${OPENFOLD_ALIGNMENT_DIR:-$REPO/examples/monomer/alignments}
OUTPUT_DIR=${OPENFOLD_OUTPUT_DIR:-$PREFIX/outputs/$INPUT_ID}
CONFIG_PRESET=${OPENFOLD_CONFIG_PRESET:-model_1_ptm}
MODEL_DEVICE=${OPENFOLD_MODEL_DEVICE:-cuda:0}
CPUS=${OPENFOLD_CPUS:-${SLURM_CPUS_PER_TASK:-8}}

die() { echo "FATAL: $*" >&2; exit 1; }

MM=$PREFIX/bin/micromamba
[ -x "$MM" ] || die "nothing installed at $PREFIX; run install.sh first"
export MAMBA_ROOT_PREFIX=$PREFIX/mamba
set +u   # the conda gcc hook reads SYS_SYSROOT unset
eval "$("$MM" shell hook --shell bash)"
micromamba activate "$ENV_NAME"
set -u

nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null ||
    die "no GPU visible; run this inside a GPU allocation"
test -d "$ALIGNMENT_DIR/$INPUT_ID" ||
    die "no precomputed alignments at $ALIGNMENT_DIR/$INPUT_ID"
mkdir -p "$OUTPUT_DIR"

cd "$REPO"
set -x
python3 -u run_pretrained_openfold.py \
    "$FASTA_DIR" \
    "$DATA/pdb_mmcif/mmcif_files" \
    --use_precomputed_alignments "$ALIGNMENT_DIR" \
    --uniref90_database_path "$DATA/uniref90/uniref90.fasta" \
    --mgnify_database_path "$DATA/mgnify/mgy_clusters_2022_05.fa" \
    --pdb70_database_path "$DATA/pdb70/pdb70" \
    --uniclust30_database_path "$DATA/uniclust30/uniclust30_2018_08/uniclust30_2018_08" \
    --bfd_database_path "$DATA/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" \
    --output_dir "$OUTPUT_DIR" \
    --config_preset "$CONFIG_PRESET" \
    --model_device "$MODEL_DEVICE" \
    --cpus "$CPUS" \
    "$@"
