#!/bin/bash
# SDSC Expanse (ClusterName "expanse"). Reached from ../../install.sh.
# No database mirror; setup.sh fetches the parameters and example templates.
# No default account is set here, so the first association is used unless overridden.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

ACCT=${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep . | head -1)}
hpc::submit "/expanse/lustre/projects/$ACCT/$USER/openfold" "$ACCT"
