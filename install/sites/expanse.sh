#!/bin/bash

# SDSC Expanse ("expanse"). No mirror; no default account, so the first association is used.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

ACCT=${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep . | head -1)}
hpc::submit "/expanse/lustre/projects/$ACCT/$USER/openfold" "$ACCT"
