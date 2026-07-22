#!/bin/bash
# NCSA Delta. Reached from ../../install.sh, or run directly from a checkout.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/interactive.sh"
. "$REPO/install/config.sh"

# Project space is /work/nvme/<allocation>/<user>, and it names the account too,
# so a wrong guess bills the wrong project. Only asked for when it is needed.
allocation() {
    local found=()
    for dir in /work/nvme/*/"$USER"; do
        [ -d "$dir" ] && found+=("$(basename "$(dirname "$dir")")")
    done
    interactive::choose OPENFOLD_ALLOCATION allocation "${found[@]}"
}

if [ -n "${OPENFOLD_PREFIX:-}" ]; then
    PREFIX=$OPENFOLD_PREFIX
else
    ALLOCATION=$(allocation)
    PREFIX=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "/work/nvme/$ALLOCATION/$USER/openfold")
fi

export OPENFOLD_PREFIX=$PREFIX OPENFOLD_HOME=$REPO
export OPENFOLD_AF2_ROOT=${OPENFOLD_AF2_ROOT:-/sw/external/alphafold2/data_hyun_official}
export OPENFOLD_GPU_PARTITION=${OPENFOLD_GPU_PARTITION:-gpuA100x4-interactive}
SETUP=$REPO/install/setup.sh
mkdir -p "$PREFIX"

if [ -n "${SLURM_STEP_ID:-}" ]; then
    LAUNCH=(bash)                                     # already on the node
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    LAUNCH=(srun --ntasks=1)                          # salloc leaves you off it
else
    ALLOCATION=${ALLOCATION:-$(allocation)}
    ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" "$ALLOCATION-delta-cpu")
    PARTITION=$(interactive::resolve OPENFOLD_PARTITION "slurm partition" cpu)
    LAUNCH=(
        sbatch --job-name=openfold-install
        --account="$ACCOUNT" --partition="$PARTITION"
        --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00
        --output="$PREFIX/install-%j.log" --export=ALL
    )
fi
# Only for the smoke test setup.sh prints; unknown here is fine.
if [ -n "${ALLOCATION:-}" ]; then export OPENFOLD_GPU_ACCOUNT=$ALLOCATION-delta-gpu; fi

echo "${LAUNCH[0]} $SETUP"
exec "${LAUNCH[@]}" "$SETUP"
