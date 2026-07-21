#!/bin/bash
# Install OpenFold, anywhere. Add a cluster as install/sites/<ClusterName>.sh.
#
#   curl -sL https://raw.githubusercontent.com/yasithdev/vizfold-foundation/delta-install-upstream/install.sh | bash
#   ./install.sh
set -euo pipefail

REPO_URL=${OPENFOLD_REPO_URL:-https://github.com/yasithdev/vizfold-foundation.git}
BRANCH=${OPENFOLD_BRANCH:-delta-install-upstream}   # -> AI2Science/main once merged
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
    # This clone is ours, so keep it current: re-running the one-liner after a fix
    # must not silently reuse the checkout it made last time.
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

. "$REPO/install/interactive.sh"
SITES=$REPO/install/sites

CLUSTER=$(scontrol show config 2>/dev/null | awk '$1 == "ClusterName" { print $3 }') || true
[ -n "${CLUSTER:-}" ] && [ -f "$SITES/$CLUSTER.sh" ] || CLUSTER=local
SITE=$(interactive::resolve OPENFOLD_SITE "site" "$CLUSTER")
test -f "$SITES/$SITE.sh" ||
    die "no site script for $SITE; have: $(cd "$SITES" && echo *.sh | sed 's/\.sh//g')"

export OPENFOLD_HOME=$REPO
exec bash "$SITES/$SITE.sh"
