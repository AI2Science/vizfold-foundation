#!/bin/bash
# Nexus (SLURM ClusterName "nexus-dev"). Reached from ../../install.sh.
#
# No database mirror and no project filesystem, so this installs under $HOME and
# setup.sh fetches the parameters and example templates itself.
#
# The GPU is a 10 GB A100 vGPU behind a 535 driver. Inference is fine -- torch
# ships prebuilt cubins -- but OpenMM JITs through NVRTC, and a 12.8 toolkit
# emits PTX this driver rejects (CUDA_ERROR_UNSUPPORTED_PTX_VERSION), which
# surfaces as "Minimization failed after 100 attempts". Capping CUDA lower does
# not help: no consistent stack solves at <=12.2. So relaxation is skipped here
# until the driver is updated, and the example is one that fits in 10 GB.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/interactive.sh"

# Bulk data belongs on the shared /projects volume, not $HOME. Everything large
# hangs off the prefix: the env, the package cache, parameters and templates.
for candidate in "/projects/$USER" /projects/*/"$USER" /projects; do
    [ -d "$candidate" ] && [ -w "$candidate" ] && { BASE=$candidate; break; }
done
PREFIX=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "${BASE:-$HOME}/openfold")
ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" \
    "$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)")

export OPENFOLD_PREFIX=$PREFIX OPENFOLD_HOME=$REPO
export OPENFOLD_GPU_ACCOUNT=$ACCOUNT
export OPENFOLD_GPU_PARTITION=${OPENFOLD_GPU_PARTITION:-gpu}
export OPENFOLD_GPU_RESOURCES=${OPENFOLD_GPU_RESOURCES:---cpus-per-task=8 --mem=24G}
export OPENFOLD_EXAMPLE=${OPENFOLD_EXAMPLE:-1UBQ_1}
export OPENFOLD_FOLD_ARGS=${OPENFOLD_FOLD_ARGS:---skip_relaxation}
SETUP=$REPO/install/setup.sh
mkdir -p "$PREFIX"

if [ -n "${SLURM_STEP_ID:-}" ]; then
    LAUNCH=(bash)                                     # already on the node
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    LAUNCH=(srun --ntasks=1)
else
    # debug has 2 cores and 5.5 GB, too small to build in; gpu is the roomy queue.
    PARTITION=$(interactive::resolve OPENFOLD_PARTITION "slurm partition" gpu)
    LAUNCH=(
        sbatch --job-name=openfold-install
        --account="$ACCOUNT" --partition="$PARTITION"
        --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G --time=01:00:00
        --output="$PREFIX/install-%j.log" --export=ALL
    )
fi

echo "${LAUNCH[0]} $SETUP"
exec "${LAUNCH[@]}" "$SETUP"
