#!/bin/bash
# NCSA Delta. Reached from ../../install.sh, or run directly from a checkout.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/interactive.sh"
. "$REPO/install/config.sh"
config::site_defaults "${BASH_SOURCE[0]}"

die() { echo "FATAL: $*" >&2; exit 1; }

# Project space is /work/nvme/<allocation>/<user>, and it names the accounts too.
# An allocation is only usable if it has both: a directory to install into and
# accounts to charge. Having a directory is not enough -- this cluster hands out
# project space whose accounts you cannot submit to, and vice versa.
usable() {
    local dir alloc accounts
    accounts=$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | sort -u)
    for dir in /work/nvme/*/"$USER"; do
        [ -d "$dir" ] || continue
        alloc=$(basename "$(dirname "$dir")")
        grep -qx "$alloc-delta-cpu" <<<"$accounts" &&
            grep -qx "$alloc-delta-gpu" <<<"$accounts" && echo "$alloc"
    done
}

# Never a question: pick one and say which, with the variable that changes it.
# An existing install wins, so re-running lands where the last one did.
allocation() {
    local usable installed
    usable=$(usable)
    [ -n "$usable" ] || return 1
    installed=$(while read -r a; do
        [ -n "$a" ] && [ -d "/work/nvme/$a/$USER/openfold" ] && echo "$a"
    done <<<"$usable" | head -1)
    echo "${installed:-$(head -1 <<<"$usable")}"
}

# Sets ALLOCATION. Checked after resolving, not with `|| die` on the substitution:
# a failing $( ) inside an argument does not fail the assignment, it just yields
# an empty default -- which silently becomes /work/nvme//$USER.
require_allocation() {
    local default
    default=$(allocation) || default=
    ALLOCATION=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$default")
    [ -n "$ALLOCATION" ] ||
        die "no usable allocation: need /work/nvme space and <alloc>-delta-{cpu,gpu} accounts"
}

if [ -n "${OPENFOLD_PREFIX:-}" ]; then
    PREFIX=$OPENFOLD_PREFIX
else
    require_allocation
    PREFIX=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "/work/nvme/$ALLOCATION/$USER/openfold")
fi

export OPENFOLD_PREFIX=$PREFIX OPENFOLD_HOME=$REPO
SETUP=$REPO/install/setup.sh
mkdir -p "$PREFIX"

if [ -n "${SLURM_STEP_ID:-}" ]; then
    LAUNCH=(bash)                                     # already on the node
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    LAUNCH=(srun --ntasks=1)                          # salloc leaves you off it
else
    [ -n "${ALLOCATION:-}" ] || require_allocation
    ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" "$ALLOCATION-delta-cpu")
    PARTITION=$(interactive::resolve OPENFOLD_PARTITION "slurm partition" "${OPENFOLD_PARTITION:-cpu}")
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
