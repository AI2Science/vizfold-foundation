#!/bin/bash
# Purdue Anvil (ClusterName "anvil"). Reached from ../../install.sh.
# No database mirror; setup.sh fetches the parameters and example templates.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is small; install on $PROJECT (snapshotted, for software, not purged),
# not $SCRATCH (purged after 30 days of no access, no warning). GPU charges <account>-gpu.
export OPENFOLD_GPU_ACCOUNT_SUFFIX=-gpu
if [ -n "${PROJECT:-}" ]; then DEFAULT=$PROJECT/$USER/openfold; else DEFAULT=/anvil/scratch/$USER/openfold; fi
hpc::submit "$DEFAULT"
