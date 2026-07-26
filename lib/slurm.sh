#!/bin/bash
# slurm.sh -- shared SLURM flow. Base declares slurm::* hooks as no-ops; a sourced sites/<name>.sh overrides the ones it needs; slurm::run assembles and executes them.

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "slurm.sh is a library" >&2; exit 1; }
[ -n "${SLURM_SH:-}" ] && return 0
SLURM_SH=1

. "$(dirname "${BASH_SOURCE[0]}")/config.sh"        # REPO, OF, die
. "$(dirname "${BASH_SOURCE[0]}")/interactive.sh"

# The site profiles, beside the libs that read them: nothing here is about any one backend.
SITES=$REPO/sites

# A site overrides this to export the atoms its <site>.json templates reference, before they are filled.
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

# The /work/nvme allocation whose accounts (suffixes $@) all exist, naming those accounts off the same suffixes.
slurm::nvme_alloc() {
    if [ -z "${OPENFOLD_ALLOCATION:-}" ]; then
        OPENFOLD_ALLOCATION=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(slurm::allocation /work/nvme "$@" || true)")
        [ -n "$OPENFOLD_ALLOCATION" ] || die "no usable allocation: need /work/nvme space and an <alloc> with account suffix(es): $*"
    fi
    export OPENFOLD_ALLOCATION
    export OPENFOLD_ACCOUNT=${OPENFOLD_ACCOUNT:-$OPENFOLD_ALLOCATION$1}
    [ -n "${2:-}" ] && export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-$OPENFOLD_ALLOCATION$2}
    return 0
}

# The user's Slurm default account, overridable inline.
slurm::default_account() { echo "${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)}"; }

# Three sources: Delta's login nodes cannot always reach slurmctld. Lower-cased to match sites/.
slurm::cluster() {
    local name=${SLURM_CLUSTER_NAME:-}
    [ -n "$name" ] || name=$(scontrol show config 2>/dev/null | awk '$1 == "ClusterName" { print $3 }')
    [ -n "$name" ] || name=$(awk -F= '/^[ \t]*ClusterName[ \t]*=/ { gsub(/[ \t]/, "", $2); print $2 }' \
        "${SLURM_CONF:-/etc/slurm/slurm.conf}" 2>/dev/null)
    echo "$name" | tr '[:upper:]' '[:lower:]'
}

# The prefix when nothing else settles one. No site overrides this hook; delta-gh.json sets OPENFOLD_PREFIX instead (it shares /work/nvme with x86 Delta and must not share the env).
slurm::default_prefix() { echo "${OPENFOLD_BASE:+$OPENFOLD_BASE/vizfold}"; }

# Resolve ~/scratch (a symlink on PACE) to the user's scratch root, dropping any subdir it points into.
slurm::scratch_root() {
    local s; [ -d "$HOME/scratch" ] || return 1; s=$(readlink -f "$HOME/scratch")
    case "$s" in */"$USER"/*) echo "${s%%/"$USER"/*}/$USER" ;; *) echo "$s" ;; esac
}

# The scheduler argv for setup.sh, one per line. $1 account, $2 partition, $3 --pty or empty.
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

# Pick the site and settle the layers under it -- every installer starts here. Exports OPENFOLD_SITE.
vizfold::settle_site() {
    local cluster site prefix
    cluster=$(slurm::cluster)
    [ -n "$cluster" ] && [ -f "$SITES/$cluster.sh" ] || cluster=local
    site=$(interactive::resolve OPENFOLD_SITE "site" "$cluster")
    test -f "$SITES/$site.sh" ||
        die "no site script for $site; have: $(cd "$SITES" && echo *.sh | sed 's/\.sh//g')"
    export OPENFOLD_SITE=$site

    . "$SITES/$site.sh"                                 # register slurm::discover
    [ -n "${OPENFOLD_PREFIX:-}" ] || slurm::discover    # the atoms the <site>.json templates need
    config::site_defaults "$SITES/$site.sh"             # fill + expand <site>.json off those atoms
    config::load                                        # then the previous install's answers

    prefix=$(interactive::resolve OPENFOLD_PREFIX "install prefix" \
        "${OPENFOLD_PREFIX:-$(slurm::default_prefix)}")
    # Empty is allowed: only the OpenFold build, which needs room for an env, insists (slurm::run).
    [ -n "$prefix" ] && export OPENFOLD_PREFIX=$prefix
    return 0
}

# Run the assembled hooks, then run setup.sh on the scheduler (or here when there is none).
slurm::run() {
    if [ -z "${SLURM_JOB_ID:-}" ] && ! command -v srun >/dev/null 2>&1; then
        exec bash "$OF/install/setup.sh"          # no scheduler: install here
    fi
    local PREFIX ACCOUNT PARTITION SETUP PTY
    PREFIX=${OPENFOLD_PREFIX:-}
    [ -n "$PREFIX" ] || die "no install prefix; set OPENFOLD_PREFIX or its <site>.json"
    # May already be set: inline env, slurm::discover, or a <site>.json template off its vars.
    ACCOUNT=$(interactive::resolve OPENFOLD_ACCOUNT "slurm account" "${OPENFOLD_ACCOUNT:-$(slurm::default_account)}")
    export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-${ACCOUNT:+$ACCOUNT${OPENFOLD_GPU_ACCOUNT_SUFFIX:-}}}
    export OPENFOLD_PREFIX=$PREFIX OPENFOLD_HOME=$REPO OPENFOLD_ACCOUNT=$ACCOUNT
    SETUP=$OF/install/setup.sh
    mkdir -p "$PREFIX"

    if [ -z "${SLURM_STEP_ID:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
        [ -n "$ACCOUNT" ] || die "no slurm account; set OPENFOLD_ACCOUNT"
        PARTITION=$(interactive::resolve OPENFOLD_PARTITION "slurm partition" "${OPENFOLD_PARTITION:-}")
        [ -n "$PARTITION" ] || die "no build partition; set OPENFOLD_PARTITION or its <site>.json"
        export OPENFOLD_PARTITION=$PARTITION
    fi

    # -t 1 must be tested here, not inside launch_args: command substitution makes stdout a pipe.
    PTY=; [ -t 1 ] && PTY=--pty
    local LAUNCH=()
    while IFS= read -r arg; do LAUNCH+=("$arg"); done < <(slurm::launch_args "$ACCOUNT" "${PARTITION:-}" "$PTY")
    echo "${LAUNCH[0]} $SETUP"
    exec "${LAUNCH[@]}" "$SETUP"
}
