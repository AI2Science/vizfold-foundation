#!/bin/bash

# NCSA Delta ("delta"). Per-allocation /work/nvme + -delta-{cpu,gpu} accounts; AF2 mirror in <site>.json.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

ALLOC=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(hpc::allocation /work/nvme -delta-cpu -delta-gpu || true)")
[ -n "$ALLOC" ] || die "no usable allocation: need /work/nvme space and <alloc>-delta-{cpu,gpu} accounts"
export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-$ALLOC-delta-gpu}
hpc::submit "/work/nvme/$ALLOC/$USER/openfold" "$ALLOC-delta-cpu"
