#!/bin/bash

# Purdue Anvil ("anvil"). No mirror; GPU jobs charge <account>-gpu.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $PROJECT (snapshotted, for software), not $SCRATCH (purged after 30 idle days). GPU charges <account>-gpu.
export OPENFOLD_GPU_ACCOUNT_SUFFIX=-gpu
if [ -n "${PROJECT:-}" ]; then DEFAULT=$PROJECT/$USER/openfold; else DEFAULT=/anvil/scratch/$USER/openfold; fi
hpc::submit "$DEFAULT"
