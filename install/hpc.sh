#!/bin/bash
# hpc.sh -- shared SLURM site flow. Library. A <site>.sh sources it, loads its
# <site>.json, resolves its prefix (and non-default accounts), then calls hpc::submit.
#   . "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
#   config::site_defaults "${BASH_SOURCE[0]}"
#   hpc::submit "<default prefix>" ["<default account>"]
# GPU account = build account + OPENFOLD_GPU_ACCOUNT_SUFFIX (Anvil bills GPUs apart);
# the rest (_GPU_PARTITION, _GPU_RESOURCES, _GPU_GRES, _EXAMPLE) setup.sh reads.

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "hpc.sh is a library" >&2; exit 1; }
[ -n "${HPC_SH:-}" ] && return 0
HPC_SH=1

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"        # REPO, die
. "$(dirname "${BASH_SOURCE[0]}")/interactive.sh"

# Pick a per-allocation prefix cluster (Delta-family): scan <root>/*/<user> for dirs
# whose allocation has every given account suffix, preferring one that already holds
# an install. $1 = dir root, $2.. = required account suffixes. Echoes the allocation.
hpc::allocation() {
    local root=$1; shift
    local dir alloc accounts s ok found=()
    accounts=$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | sort -u)
    for dir in "$root"/*/"$USER"; do
        [ -d "$dir" ] || continue
        alloc=$(basename "$(dirname "$dir")"); ok=1
        for s in "$@"; do grep -qx "$alloc$s" <<<"$accounts" || ok=0; done
        [ "$ok" = 1 ] && found+=("$alloc")
    done
    [ ${#found[@]} -gt 0 ] || return 1
    for alloc in "${found[@]}"; do
        [ -d "$root/$alloc/$USER/openfold" ] && { echo "$alloc"; return 0; }
    done
    echo "${found[0]}"
}

hpc::submit() {
    local prefix_default=$1 account_default=${2:-}
    local PREFIX ACCOUNT PARTITION SETUP LAUNCH

    PREFIX=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "$prefix_default")
    [ -n "$PREFIX" ] || die "no install prefix; set OPENFOLD_PREFIX"
    ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" \
        "${account_default:-$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)}")
    export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-$ACCOUNT${OPENFOLD_GPU_ACCOUNT_SUFFIX:-}}
    export OPENFOLD_PREFIX=$PREFIX OPENFOLD_HOME=$REPO
    SETUP=$REPO/install/setup.sh
    mkdir -p "$PREFIX"

    if [ -n "${SLURM_STEP_ID:-}" ]; then
        LAUNCH=(bash)                                    # already on the node
    elif [ -n "${SLURM_JOB_ID:-}" ]; then
        LAUNCH=(srun --ntasks=1)                         # salloc leaves you off it
    else
        [ -n "$ACCOUNT" ] || die "no slurm account; set OPENFOLD_ACCOUNT"
        PARTITION=$(interactive::resolve OPENFOLD_PARTITION "slurm partition" "${OPENFOLD_PARTITION:-}")
        [ -n "$PARTITION" ] || die "no build partition; set OPENFOLD_PARTITION or its <site>.json"
        LAUNCH=(
            sbatch --job-name=openfold-install
            --account="$ACCOUNT" --partition="$PARTITION"
            --nodes=1 --ntasks=1 --cpus-per-task="${OPENFOLD_BUILD_CPUS:-8}"
            --mem="${OPENFOLD_BUILD_MEM:-24G}" --time="${OPENFOLD_BUILD_TIME:-02:00:00}"
            --output="$PREFIX/install-%j.log" --export=ALL
        )
    fi
    echo "${LAUNCH[0]} $SETUP"
    exec "${LAUNCH[@]}" "$SETUP"
}
