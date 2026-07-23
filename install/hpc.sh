#!/bin/bash
# hpc.sh -- shared SLURM site flow. Library: a <site>.sh sources it, loads <site>.json, resolves prefix/accounts, then calls hpc::submit.

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "hpc.sh is a library" >&2; exit 1; }
[ -n "${HPC_SH:-}" ] && return 0
HPC_SH=1

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"        # REPO, die
. "$(dirname "${BASH_SOURCE[0]}")/interactive.sh"

# Delta-family: pick an allocation under $1 whose accounts (suffixes $2..) all exist, preferring one that holds an install.
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
            ${OPENFOLD_BUILD_GRES:+--gres="$OPENFOLD_BUILD_GRES"}   # Delta-AI rejects CPU-only jobs
            --output="$PREFIX/install-%j.log" --export=ALL
        )
    fi
    echo "${LAUNCH[0]} $SETUP"
    exec "${LAUNCH[@]}" "$SETUP"
}
