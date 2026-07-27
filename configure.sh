#!/bin/bash

# What `vizfold install repo` runs once the checkout is there: settle which cluster this is, where
# the install prefix goes, which AlphaFold2 mirror holds the protein databases, and what the
# scheduler will accept -- then write all of it to ~/.config/vizfold/vizfold.json. A backend install
# afterwards only builds its environment; without this it would have to settle the site itself.
set -euo pipefail

. "${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/lib/slurm.sh"

main() {
    vizfold::settle_site
    export OPENFOLD_HOME=$REPO VIZFOLD_ENV_BASE=$(vizfold::env_base)
    # Defers to whatever is set: overwriting it would move someone's run history.
    export VIZFOLD_DB=${VIZFOLD_DB:-$(vizfold::prefix)/vizfold.db}
    # Recorded, so it has to exist: status reads a configured-but-absent prefix as a broken config.
    mkdir -p "$(vizfold::env_base)"
    config::save
    cat <<MSG

The proteins are under $REPO/examples/monomer, alignments beside them:

  vizfold list proteins

Then a backend to fold with: vizfold install openfold (or esmfold).
MSG
}

# Sourced (tests/configure_repo.sh) this file is just its definitions; only an execution configures.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
main "$@"
