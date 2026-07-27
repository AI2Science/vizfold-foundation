#!/bin/bash
# Which account an install charges. Run: bash tests/slurm_account.sh
#
# Nexus sets no DefaultAccount, so the old lookup returned nothing and the prompt came up empty with
# no hint that `pearc26-tutorial` was sitting right there in the associations.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); export REPO OPENFOLD_HOME=$REPO
export USER=x-test VIZFOLD_CONFIG=/nonexistent/vizfold-test.json
. "$REPO/lib/slurm.sh"
set +e
fail=0

# $1 what `show user … DefaultAccount` prints, $2 what `show assoc … Account` prints.
stub() {
    DEFAULT=$1 ASSOC=$2
    sacctmgr() { case "$*" in *DefaultAccount*) printf '%s' "$DEFAULT" ;; *Account*) printf '%s' "$ASSOC" ;; esac; }
}
check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else
    echo "FAIL $1"; echo "  want [$3]"; echo "  got  [$2]"; fail=1; fi; }

stub 'bbka' 'bbka
other'
check "a DefaultAccount is taken as given" "$(OPENFOLD_ACCOUNT= slurm::default_account)" "bbka"

# The reported bug: Nexus prints nothing for DefaultAccount.
stub '' 'pearc26-tutorial'
check "no DefaultAccount falls back to the one association" "$(OPENFOLD_ACCOUNT= slurm::default_account)" "pearc26-tutorial"

# sacctmgr repeats an account per association; one account is still one answer.
stub '' 'pearc26-tutorial
pearc26-tutorial'
check "a repeated association is still one account" "$(OPENFOLD_ACCOUNT= slurm::default_account)" "pearc26-tutorial"

stub '' 'alpha
beta'
check "several associations name none: charging one is the user's call" "$(OPENFOLD_ACCOUNT= slurm::default_account)" ""

stub '' ''
check "no slurm accounting at all is not an error" "$(OPENFOLD_ACCOUNT= slurm::default_account)" ""

stub 'bbka' 'bbka'
check "an explicit OPENFOLD_ACCOUNT wins" "$(OPENFOLD_ACCOUNT=mine slurm::default_account)" "mine"

# `vizfold install repo` writes the config from settle_site alone, never reaching slurm::run, so the
# account has to be settled there or it lands empty in a config that claims to be complete.
stub '' 'pearc26-tutorial'
got=$( export OPENFOLD_SITE=nexus-dev OPENFOLD_BASE=/projects/x-test OPENFOLD_ACCOUNT= OPENFOLD_PREFIX=/tmp/x
       interactive::resolve() { echo "$3"; }
       vizfold::settle_site >/dev/null 2>&1; echo "$OPENFOLD_ACCOUNT" )
check "settle_site records it, for the config install repo writes" "$got" "pearc26-tutorial"

exit $fail
