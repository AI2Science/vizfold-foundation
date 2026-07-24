#!/bin/bash
# Every site's fully-resolved install config, snapshotted. Run: bash install/tests/site_config.sh
# Accept an intended change: bash install/tests/site_config.sh -u
#
# A <site>.json that restates a default and one that omits it must resolve identically, so this runs
# the real flow -- slurm::discover, <site>.json templating, slurm::run's exports, setup.sh's
# defaults -- and compares what config::save would write plus the vars that drive the build job.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$(cd ../../.. && pwd)
EXPECTED=tests/site_config.expected
SANDBOX=${TMPDIR:-/tmp}/vizfold-site-config-$$
trap 'rm -rf "$SANDBOX"' EXIT

# The one login-specific atom each discover reads off the real cluster, as inline env supplies it.
# nexus-dev probes absolute paths instead, so it gets the answer and skips discover.
site_env() {
    case $1 in
        anvil)          echo 'export PROJECT=/anvil/projects/x-test' ;;
        delta|delta-gh) echo 'export ALLOC=bbka' ;;
        expanse)        echo 'export OPENFOLD_ALLOCATION=abc123' ;;
        ice-slurm)      echo 'SCRATCH=/storage/ice1/x-test' ;;
        phoenix-slurm)  echo 'SCRATCH=/storage/scratch1/x-test' ;;
        nexus-dev)      echo 'export OPENFOLD_BASE=/projects/x-test; skip=1' ;;
    esac
}

resolve() {
    local site=$1 SCRATCH= skip= arch=x86_64
    eval "$(site_env "$site")"
    [ "$site" = delta-gh ] && arch=aarch64                     # the one non-x86 site

    export USER=x-test HOME=$SANDBOX/home
    export VIZFOLD_CONFIG=$SANDBOX/$site.json OPENFOLD_HOME=$REPO

    . ./slurm.sh
    . "sites/$site.sh"
    # After the sources, so these win: slurm.sh defines its own scratch_root.
    uname() { [ "${1:-}" = -m ] && echo "$arch" || command uname "$@"; }
    sacctmgr() { echo bbka; }
    slurm::scratch_root() { echo "$SCRATCH"; }

    [ -n "$skip" ] || slurm::discover
    config::site_defaults "sites/$site.sh"

    # What slurm::run settles before handing off to setup.sh.
    export OPENFOLD_PREFIX=${OPENFOLD_PREFIX:-$(slurm::default_prefix)}
    export OPENFOLD_ACCOUNT=${OPENFOLD_ACCOUNT:-$(slurm::default_account)}
    export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-${OPENFOLD_ACCOUNT:+$OPENFOLD_ACCOUNT${OPENFOLD_GPU_ACCOUNT_SUFFIX:-}}}

    . "$SANDBOX/setup-defs.sh"
    export CONDA_PREFIX=$OPENFOLD_PREFIX/envs/vizfold-openfold
    setup::config
    setup::fold_vars
    setup::config_save >/dev/null

    echo "## $site"
    echo "account=$OPENFOLD_ACCOUNT partition=${OPENFOLD_PARTITION:-}"
    echo "build=${OPENFOLD_BUILD_CPUS:-8}/${OPENFOLD_BUILD_MEM:-24G}/${OPENFOLD_BUILD_TIME:-02:00:00}/${OPENFOLD_BUILD_GRES:-}"
    echo "env=$ENV_DIR arch=$arch max_cuda=$MAX_CUDA override_cuda=$CONDA_OVERRIDE_CUDA"
    echo "launch=$LAUNCH"
    cat "$VIZFOLD_CONFIG"
}

mkdir -p "$SANDBOX/home"
# setup.sh's defaults, taken from setup.sh so they cannot drift; a file, not <(), for bash 3.2.
sed '/^main() {/,$d' setup.sh > "$SANDBOX/setup-defs.sh"
# Subshell per site: exported OPENFOLD_* and the site's hooks must not leak into the next.
# Substitute the quoted value, not the bare path: a $REPO of /w would rewrite every /work path.
actual=$(for f in sites/*.json; do f=${f##*/}; (resolve "${f%.json}" 2>/dev/null); done |
    sed "s#\"$REPO\"#\"{REPO}\"#g; s#\"$SANDBOX/#\"{SANDBOX}/#g")

if [ "${1:-}" = -u ]; then
    printf '%s\n' "$actual" > "$EXPECTED"
    echo "updated $EXPECTED ($(grep -c '^## ' <<<"$actual") sites)"
elif [ "$actual" = "$(cat "$EXPECTED" 2>/dev/null)" ]; then
    echo "ok   $(grep -c '^## ' <<<"$actual") sites resolve unchanged"
else
    echo "FAIL site config drifted:"
    diff "$EXPECTED" <(printf '%s\n' "$actual")
    exit 1
fi
