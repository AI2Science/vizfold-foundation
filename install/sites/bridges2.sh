#!/bin/bash

# PSC Bridges-2 ("bridges2"). AF2 mirror in <site>.json; account = grant id = /ocean project dir.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME (/jet) is tiny; install under the grant's /ocean project space.
ACCT=${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)}
hpc::submit "/ocean/projects/$ACCT/$USER/openfold" "$ACCT"
