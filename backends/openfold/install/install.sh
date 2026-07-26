#!/bin/bash

# Install the OpenFold backend on this cluster. Site selection lives in lib/slurm.sh; add a cluster
# as sites/<ClusterName>.sh. Invoked by `vizfold install openfold`.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
OF=$REPO/backends/openfold
die() { echo "FATAL: $*" >&2; exit 1; }

main() {
    test -f "$OF/setup.py" || die "no openfold backend at $OF; set OPENFOLD_HOME to a vizfold checkout"
    # Pin REPO for the libraries (config.sh reads OPENFOLD_HOME) before sourcing them.
    export OPENFOLD_HOME=$REPO
    . "$REPO/lib/slurm.sh"        # config.sh + interactive.sh, the slurm::* hooks, settle_site, run
    vizfold::settle_site
    slurm::run
}
main
