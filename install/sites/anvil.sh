#!/bin/bash
# Purdue Anvil (ClusterName "anvil"). Reached from ../../install.sh.
# No database mirror; setup.sh fetches the parameters and example templates.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 100 GB; the env and databases go on scratch. GPU jobs charge <account>-gpu.
export OPENFOLD_GPU_ACCOUNT_SUFFIX=-gpu
hpc::submit "/anvil/scratch/$USER/openfold"
