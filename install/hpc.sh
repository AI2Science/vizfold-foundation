#!/bin/bash
# hpc.sh -- shared SLURM flow. Base declares site::* hooks as no-ops; a sourced sites/<name>.sh overrides the ones it needs; hpc::run assembles and executes them.

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "hpc.sh is a library" >&2; exit 1; }
[ -n "${HPC_SH:-}" ] && return 0
HPC_SH=1

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"        # REPO, die
. "$(dirname "${BASH_SOURCE[0]}")/interactive.sh"

# Hooks a site contributes. Each sets the *_DEFAULT that hpc::run declares (bash dynamic scope); unset ones stay no-ops.
site::prefix()      { :; }
site::account()     { :; }
site::gpu_account() { :; }

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

# Resolve ~/scratch (a symlink on PACE) to the user's scratch root, dropping any subdir it points into.
hpc::scratch_root() {
    local s; s=$(readlink -f "$HOME/scratch") || return 1
    case "$s" in */"$USER"/*) echo "${s%%/"$USER"/*}/$USER" ;; *) echo "$s" ;; esac
}

# Run the assembled hooks, then submit setup.sh to the scheduler (or run it in place when there is none).
hpc::run() {
    if [ -z "${SLURM_JOB_ID:-}" ] && ! command -v sbatch >/dev/null 2>&1; then
        exec bash "$REPO/install/setup.sh"          # no scheduler: install here
    fi
    local PREFIX_DEFAULT= ACCOUNT_DEFAULT= GPU_ACCOUNT_DEFAULT= PREFIX ACCOUNT PARTITION SETUP LAUNCH
    site::prefix; site::account; site::gpu_account

    PREFIX=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "$PREFIX_DEFAULT")
    [ -n "$PREFIX" ] || die "no install prefix; set OPENFOLD_PREFIX or add site::prefix"
    [ -n "$ACCOUNT_DEFAULT" ] || ACCOUNT_DEFAULT=$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)
    ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" "$ACCOUNT_DEFAULT")
    export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-${GPU_ACCOUNT_DEFAULT:-$ACCOUNT${OPENFOLD_GPU_ACCOUNT_SUFFIX:-}}}
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
            ${OPENFOLD_BUILD_GRES:+--gres="$OPENFOLD_BUILD_GRES"}
            --output="$PREFIX/install-%j.log" --export=ALL
        )
    fi
    echo "${LAUNCH[0]} $SETUP"
    exec "${LAUNCH[@]}" "$SETUP"
}
