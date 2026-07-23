#!/bin/bash
# Fold one sequence on a GPU. Site-independent: every path comes from OPENFOLD_*,
# and install.sh prints the invocation for the cluster it installed on.
#
#   OPENFOLD_PREFIX=<prefix> run/fold.sh 6KWC_1
set -euo pipefail

# OPENFOLD_HOME if a scheduler spooled this script (BASH_SOURCE would then miss its
# siblings); otherwise BASH_SOURCE, for a direct run. config.sh sets REPO and die().
. "${OPENFOLD_HOME:-$(dirname "${BASH_SOURCE[0]}")/..}/install/config.sh"
PREFIX=${OPENFOLD_PREFIX:-$HOME/openfold}
ENV_NAME=${OPENFOLD_ENV_NAME:-openfold-env}
ENV_PREFIX=${OPENFOLD_ENV_PREFIX:-$PREFIX/mamba/envs/$ENV_NAME}

# torch and DeepSpeed JIT-compile at import; a site's module env (CC/CXX/CPATH/
# PATH/LD_LIBRARY_PATH pointing at its own gcc) makes that compile fail with
# "cannot execute cc1". Re-exec once in a curated env -- the treatment the install
# build uses -- keeping HOME, a clean PATH, the SLURM/CUDA GPU binding, and every
# OPENFOLD_* the caller and config set.
if [ -z "${OPENFOLD_CLEAN_ENV:-}" ]; then
    clean=(HOME="$HOME" PATH="$ENV_PREFIX/bin:/usr/bin:/bin" OPENFOLD_CLEAN_ENV=1)
    [ -n "${TMPDIR:-}" ] && clean+=("TMPDIR=$TMPDIR")
    # torch/DeepSpeed autotune their JIT kernels here; on NFS $HOME it is glacial
    # (79 s vs 8 s inference). Keep it on node-local disk.
    clean+=("TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/openfold-triton-$(id -u)}")
    while IFS= read -r kv; do clean+=("$kv"); done < <(
        env | grep -E '^(OPENFOLD_[A-Z0-9_]*|SLURM_[A-Z0-9_]*|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|NVIDIA_VISIBLE_DEVICES)=')
    exec env -i "${clean[@]}" bash "$0" "$@"
fi

INPUT_ID=${1:-${OPENFOLD_INPUT_ID:-6KWC_1}}
[ $# -gt 0 ] && shift                     # the rest pass through to the python script
DATA=${OPENFOLD_DATA_DIR:-$PREFIX/data}
FASTA_DIR=${OPENFOLD_FASTA_DIR:-$REPO/examples/monomer/fasta_dir_${INPUT_ID%_*}}
ALIGNMENT_DIR=${OPENFOLD_ALIGNMENT_DIR:-$REPO/examples/monomer/alignments}
OUTPUT_DIR=${OPENFOLD_OUTPUT_DIR:-$PREFIX/outputs/$INPUT_ID}
CONFIG_PRESET=${OPENFOLD_CONFIG_PRESET:-model_1_ptm}
MODEL_DEVICE=${OPENFOLD_MODEL_DEVICE:-cuda:0}
CPUS=${OPENFOLD_CPUS:-${SLURM_CPUS_PER_TASK:-8}}

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
