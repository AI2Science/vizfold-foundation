#!/bin/bash

# Install the OpenFold backend on this cluster. The site -- which cluster this is, which allocation
# and prefix it settles on -- is the platform's business and lives in lib/slurm.sh; add a cluster as
# sites/<ClusterName>.sh. Invoked by `vizfold install openfold`.
set -euo pipefail

# Checkout root (already cloned by the bootstrap) and the OpenFold backend subtree it lives in.
REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
OF=$REPO/backends/openfold
die() { echo "FATAL: $*" >&2; exit 1; }

main() {
    test -f "$OF/setup.py" || die "$REPO is not a vizfold checkout; re-run the bootstrap installer"
    # Pin REPO for the libraries (config.sh reads OPENFOLD_HOME) before sourcing them.
    export OPENFOLD_HOME=$REPO
    . "$REPO/lib/slurm.sh"        # config.sh + interactive.sh, the slurm::* hooks, settle_site, run
    vizfold::settle_site
    slurm::run
}
main
