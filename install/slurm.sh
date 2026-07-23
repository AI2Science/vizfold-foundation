#!/bin/bash
# slurm.sh -- shared SLURM flow. Base declares slurm::* hooks as no-ops; a sourced sites/<name>.sh overrides the ones it needs; slurm::run assembles and executes them.

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "slurm.sh is a library" >&2; exit 1; }
[ -n "${SLURM_SH:-}" ] && return 0
SLURM_SH=1

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"        # REPO, die
. "$(dirname "${BASH_SOURCE[0]}")/interactive.sh"
export PATH="$(dirname "${BASH_SOURCE[0]}"):$PATH"

# A site overrides this to export the account-specific vars its <site>.json templates reference
# ($ALLOC, OPENFOLD_ACCOUNT, OPENFOLD_SCRATCH, ...). No-op by default; runs before <site>.json is filled.
slurm::discover() { :; }

# Delta-family: pick an allocation under $1 whose accounts (suffixes $2..) all exist, preferring one that holds an install.
slurm::allocation() {
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

# Resolve, prompt for, and memoize (in ALLOC) the /work/nvme allocation whose accounts (suffixes $@) all exist.
slurm::nvme_alloc() {
    [ -n "${ALLOC:-}" ] && return
    ALLOC=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(slurm::allocation /work/nvme "$@" || true)")
    [ -n "$ALLOC" ] || die "no usable allocation: need /work/nvme space and an <alloc> with account suffix(es): $*"
    export ALLOC
}

# The user's Slurm default account, overridable inline.
slurm::default_account() { echo "${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)}"; }

# Resolve ~/scratch (a symlink on PACE) to the user's scratch root, dropping any subdir it points into.
slurm::scratch_root() {
    local s; s=$(readlink -f "$HOME/scratch") || return 1
    case "$s" in */"$USER"/*) echo "${s%%/"$USER"/*}/$USER" ;; *) echo "$s" ;; esac
}

# Build the scheduler argv for setup.sh, one argument per line.
# $1 account, $2 partition, $3 the literal --pty (or empty when stdout is not a terminal).
slurm::launch_args() {
    if [ -n "${SLURM_STEP_ID:-}" ]; then
        printf '%s\n' bash                                   # already on the node
        return
    fi
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        printf '%s\n' srun --ntasks=1                        # salloc leaves you off it
        return
    fi
    printf '%s\n' srun -u
    [ -n "$3" ] && printf '%s\n' "$3"
    printf '%s\n' --job-name=vizfold-install "--account=$1" "--partition=$2" \
        --nodes=1 --ntasks=1 "--cpus-per-task=${OPENFOLD_BUILD_CPUS:-8}" \
        "--mem=${OPENFOLD_BUILD_MEM:-24G}" "--time=${OPENFOLD_BUILD_TIME:-02:00:00}"
    [ -n "${OPENFOLD_BUILD_GRES:-}" ] && printf '%s\n' "--gres=$OPENFOLD_BUILD_GRES"
    return 0
}

# Run the assembled hooks, then run setup.sh on the scheduler (or here when there is none).
slurm::run() {
    if [ -z "${SLURM_JOB_ID:-}" ] && ! command -v srun >/dev/null 2>&1; then
        exec bash "$REPO/install/setup.sh"          # no scheduler: install here
    fi
    local PREFIX ACCOUNT PARTITION SETUP PTY
    # OPENFOLD_PREFIX/ACCOUNT come pre-resolved: inline env, or <site>.json templates expanded off slurm::discover's vars.
    PREFIX=$(interactive::resolve OPENFOLD_PREFIX "install prefix" "${OPENFOLD_PREFIX:-}")
    [ -n "$PREFIX" ] || die "no install prefix; set OPENFOLD_PREFIX or its <site>.json"
    ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" "${OPENFOLD_ACCOUNT:-$(slurm::default_account)}")
    export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-${ACCOUNT:+$ACCOUNT${OPENFOLD_GPU_ACCOUNT_SUFFIX:-}}}
    export OPENFOLD_PREFIX=$PREFIX OPENFOLD_HOME=$REPO
    SETUP=$REPO/install/setup.sh
    mkdir -p "$PREFIX"

    if [ -z "${SLURM_STEP_ID:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
        [ -n "$ACCOUNT" ] || die "no slurm account; set OPENFOLD_ACCOUNT"
        PARTITION=$(interactive::resolve OPENFOLD_PARTITION "slurm partition" "${OPENFOLD_PARTITION:-}")
        [ -n "$PARTITION" ] || die "no build partition; set OPENFOLD_PARTITION or its <site>.json"
    fi

    # -t 1 must be tested here, not inside launch_args: command substitution makes stdout a pipe.
    PTY=; [ -t 1 ] && PTY=--pty
    local LAUNCH=()
    while IFS= read -r arg; do LAUNCH+=("$arg"); done < <(slurm::launch_args "$ACCOUNT" "${PARTITION:-}" "$PTY")
    echo "${LAUNCH[0]} $SETUP"
    exec "${LAUNCH[@]}" "$SETUP"
}
