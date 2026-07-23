#!/bin/bash

# Install OpenFold on any HPC cluster in one command; add a cluster as install/sites/<ClusterName>.sh.
set -euo pipefail

die() { echo "FATAL: $*" >&2; exit 1; }

bootstrap::config() {
    REPO_URL=${OPENFOLD_REPO_URL:-https://github.com/AI2Science/vizfold-foundation.git}
    BRANCH=${OPENFOLD_BRANCH:-main}
    SRC=${OPENFOLD_SRC:-$HOME/openfold-src}   # outlives the job; the editable install points here
}

# Piped into bash BASH_SOURCE is unusable; under sbatch it is the spool copy.
bootstrap::find_repo() {
    if [ -n "${OPENFOLD_HOME:-}" ]; then
        REPO=$OPENFOLD_HOME
    elif [ -f "${SLURM_SUBMIT_DIR:-$PWD}/setup.py" ]; then
        REPO=${SLURM_SUBMIT_DIR:-$PWD}
    else
        # Walk up from this file; a copy saved outside a checkout finds nothing.
        REPO=$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null &&
            until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)
    fi
}

# No checkout found: fall back to our own clone and keep it current, so a re-run after a fix doesn't reuse stale code.
bootstrap::sync_checkout() {
    [ -f "$REPO/setup.py" ] && return
    REPO=$SRC
    if [ -d "$REPO/.git" ]; then
        # Point origin at REPO_URL so OPENFOLD_REPO_URL applies to re-runs, not just the first clone.
        git -C "$REPO" remote set-url origin "$REPO_URL" 2>/dev/null
        git -C "$REPO" fetch -q origin "$BRANCH" &&
            git -C "$REPO" reset -q --hard FETCH_HEAD ||
            echo "warning: could not update $REPO, using it as-is" >&2
    else
        git clone -q --branch "$BRANCH" "$REPO_URL" "$REPO"
    fi
}

# Pin REPO for the libraries (config.sh reads OPENFOLD_HOME) before sourcing them.
bootstrap::libs() {
    test -f "$REPO/setup.py" || die "$REPO is not an OpenFold checkout"
    export OPENFOLD_HOME=$REPO
    . "$REPO/install/hpc.sh"          # pulls in config.sh + interactive.sh; declares the site::* hooks and hpc::run
    SITES=$REPO/install/sites
}

bootstrap::pick_site() {
    local cluster
    cluster=$(scontrol show config 2>/dev/null | awk '$1 == "ClusterName" { print $3 }') || true
    [ -n "${cluster:-}" ] && [ -f "$SITES/$cluster.sh" ] || cluster=local
    SITE=$(interactive::resolve OPENFOLD_SITE "site" "$cluster")
    test -f "$SITES/$SITE.sh" ||
        die "no site script for $SITE; have: $(cd "$SITES" && echo *.sh | sed 's/\.sh//g')"
    export OPENFOLD_SITE=$SITE
}

bootstrap::dispatch() {
    config::site_defaults "$SITES/$SITE.sh"   # <site>.json defaults (fills unset)
    . "$SITES/$SITE.sh"                        # register this site's hook overrides
    hpc::run                                   # execute the assembled function set
}

main() {
    bootstrap::config
    bootstrap::find_repo
    bootstrap::sync_checkout
    bootstrap::libs
    bootstrap::pick_site
    bootstrap::dispatch
}
main
