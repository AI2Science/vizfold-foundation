#!/bin/bash
# Georgia Tech PACE Phoenix (ClusterName "phoenix-slurm"). From ../../install.sh.
# No database mirror; setup.sh fetches the parameters and example templates.
# The GPU queue is mixed hardware, so <site>.json pins an A100.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/interactive.sh"
. "$REPO/install/config.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 20 GB; the env and databases go on ~/scratch.
PREFIX=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "$HOME/scratch/openfold")
ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" \
    "$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)")
export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-$ACCOUNT}

export OPENFOLD_PREFIX=$PREFIX OPENFOLD_HOME=$REPO
SETUP=$REPO/install/setup.sh
mkdir -p "$PREFIX"

if [ -n "${SLURM_STEP_ID:-}" ]; then
    LAUNCH=(bash)
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    LAUNCH=(srun --ntasks=1)
else
    PARTITION=$(interactive::resolve OPENFOLD_PARTITION "slurm partition" "${OPENFOLD_PARTITION:-cpu-small}")
    LAUNCH=(
        sbatch --job-name=openfold-install
        --account="$ACCOUNT" --partition="$PARTITION"
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --time=02:00:00
        --output="$PREFIX/install-%j.log" --export=ALL
    )
fi

echo "${LAUNCH[0]} $SETUP"
exec "${LAUNCH[@]}" "$SETUP"
