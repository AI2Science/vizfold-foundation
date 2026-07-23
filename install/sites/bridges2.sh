#!/bin/bash
# PSC Bridges-2 (ClusterName "bridges2"). Reached from ../../install.sh.
# AF2 mirror in <site>.json. Project space and account share the grant id.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME (/jet) is tiny; install under the grant's /ocean project space.
ACCT=${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null)}
hpc::submit "/ocean/projects/$ACCT/$USER/openfold" "$ACCT"
