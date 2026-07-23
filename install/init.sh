#!/bin/bash

# Install a model backend (OpenFold today) on this cluster; dispatch on SLURM ClusterName, add a cluster as install/sites/<ClusterName>.sh. Invoked by `vizfold init`.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # already cloned by the bootstrap
die() { echo "FATAL: $*" >&2; exit 1; }

# Pin REPO for the libraries (config.sh reads OPENFOLD_HOME) before sourcing them.
init::libs() {
    test -f "$REPO/setup.py" || die "$REPO is not a vizfold checkout; re-run the bootstrap installer"
    export OPENFOLD_HOME=$REPO
    . "$REPO/install/slurm.sh"          # pulls in config.sh + interactive.sh; declares the slurm::* hooks and slurm::run
    SITES=$REPO/install/sites
}

init::pick_site() {
    local cluster
    cluster=$(scontrol show config 2>/dev/null | awk '$1 == "ClusterName" { print $3 }') || true
    [ -n "${cluster:-}" ] && [ -f "$SITES/$cluster.sh" ] || cluster=local
    SITE=$(interactive::resolve OPENFOLD_SITE "site" "$cluster")
    test -f "$SITES/$SITE.sh" ||
        die "no site script for $SITE; have: $(cd "$SITES" && echo *.sh | sed 's/\.sh//g')"
    export OPENFOLD_SITE=$SITE
}

init::dispatch() {
    . "$SITES/$SITE.sh"                                 # register slurm::discover
    [ -n "${OPENFOLD_PREFIX:-}" ] || slurm::discover    # export the account-specific vars the <site>.json templates need
    config::site_defaults "$SITES/$SITE.sh"             # fill + expand <site>.json (templates resolve off the discovered vars)
    slurm::run
}

main() {
    init::libs
    init::pick_site
    init::dispatch
}
main
