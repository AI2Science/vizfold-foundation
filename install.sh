#!/bin/bash

# Install OpenFold on any HPC cluster in one command; add a cluster as install/sites/<ClusterName>.sh.
set -euo pipefail

REPO_URL=${OPENFOLD_REPO_URL:-https://github.com/AI2Science/vizfold-foundation.git}
BRANCH=${OPENFOLD_BRANCH:-main}
SRC=${OPENFOLD_SRC:-$HOME/openfold-src}   # outlives the job; the editable install points here

die() { echo "FATAL: $*" >&2; exit 1; }

# Piped into bash BASH_SOURCE is unusable; under sbatch it is the spool copy.
if [ -n "${OPENFOLD_HOME:-}" ]; then
    REPO=$OPENFOLD_HOME
elif [ -f "${SLURM_SUBMIT_DIR:-$PWD}/setup.py" ]; then
    REPO=${SLURM_SUBMIT_DIR:-$PWD}
else
    # Walk up from this file; a copy saved outside a checkout finds nothing.
    REPO=$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" 2>/dev/null &&
        until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)
fi
if [ ! -f "$REPO/setup.py" ]; then
    # Our clone: keep it current so a re-run after a fix doesn't reuse stale code.
    REPO=$SRC
    if [ -d "$REPO/.git" ]; then
        git -C "$REPO" fetch -q origin "$BRANCH" &&
            git -C "$REPO" reset -q --hard FETCH_HEAD ||
            echo "warning: could not update $REPO, using it as-is" >&2
    else
        git clone -q --branch "$BRANCH" "$REPO_URL" "$REPO"
    fi
fi
test -f "$REPO/setup.py" || die "$REPO is not an OpenFold checkout"

# Pin REPO for the libraries (config.sh reads OPENFOLD_HOME) before sourcing them.
export OPENFOLD_HOME=$REPO
. "$REPO/install/interactive.sh"
. "$REPO/install/config.sh"
SITES=$REPO/install/sites

CLUSTER=$(scontrol show config 2>/dev/null | awk '$1 == "ClusterName" { print $3 }') || true
[ -n "${CLUSTER:-}" ] && [ -f "$SITES/$CLUSTER.sh" ] || CLUSTER=local
SITE=$(interactive::resolve OPENFOLD_SITE "site" "$CLUSTER")
test -f "$SITES/$SITE.sh" ||
    die "no site script for $SITE; have: $(cd "$SITES" && echo *.sh | sed 's/\.sh//g')"

export OPENFOLD_SITE=$SITE
exec bash "$SITES/$SITE.sh"
