#!/bin/bash

# Install the OpenFold backend. Site selection lives in lib/slurm.sh; add a cluster as sites/<ClusterName>.sh.
set -euo pipefail

. "${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/lib/slurm.sh"

main() {
    test -f "$OF/setup.py" || die "no openfold backend at $OF; set OPENFOLD_HOME to a vizfold checkout"
    export OPENFOLD_HOME=$REPO   # setup.sh finds lib/ off it, without walking up
    vizfold::settle_site
    slurm::run
}
main
