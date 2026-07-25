#!/bin/bash

# Install the OpenFold backend on this cluster; dispatch on the cluster name slurm::cluster resolves (SLURM_CLUSTER_NAME, then scontrol, then slurm.conf), add a cluster as backends/openfold/install/sites/<ClusterName>.sh. Invoked by `vizfold install openfold`.
set -euo pipefail

# Checkout root (already cloned by the bootstrap) and the OpenFold backend subtree it lives in.
REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
OF=$REPO/backends/openfold
die() { echo "FATAL: $*" >&2; exit 1; }

# Pin REPO for the libraries (config.sh reads OPENFOLD_HOME) before sourcing them.
init::libs() {
    test -f "$OF/setup.py" || die "$REPO is not a vizfold checkout; re-run the bootstrap installer"
    export OPENFOLD_HOME=$REPO
    . "$OF/install/slurm.sh"          # pulls in config.sh + interactive.sh; declares the slurm::* hooks and slurm::run
    SITES=$OF/install/sites
}

init::pick_site() {
    local cluster
    cluster=$(slurm::cluster)
    [ -n "$cluster" ] && [ -f "$SITES/$cluster.sh" ] || cluster=local
    SITE=$(interactive::resolve OPENFOLD_SITE "site" "$cluster")
    test -f "$SITES/$SITE.sh" ||
        die "no site script for $SITE; have: $(cd "$SITES" && echo *.sh | sed 's/\.sh//g')"
    export OPENFOLD_SITE=$SITE
}

init::dispatch() {
    . "$SITES/$SITE.sh"                                 # register slurm::discover
    [ -n "${OPENFOLD_PREFIX:-}" ] || slurm::discover    # export the account-specific vars the <site>.json templates need
    config::site_defaults "$SITES/$SITE.sh"             # fill + expand <site>.json (templates resolve off the discovered vars)
    config::load                                        # then the previous install's answers, under all of it
    slurm::run
}

main() {
    init::libs
    init::pick_site
    init::dispatch
}
main
