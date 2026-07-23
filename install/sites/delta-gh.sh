#!/bin/bash

# NCSA Delta-AI ("delta-gh"). Grace-Hopper aarch64; build on a GH200 node (no CPU queue); no mirror.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# aarch64: setup.sh uses environment-aarch64.yml (CUDA 13 / py3.13, nomkl, no deepspeed).
ALLOC=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(hpc::allocation /work/nvme -dtai-gh || true)")
[ -n "$ALLOC" ] || die "no usable allocation: need /work/nvme space and <alloc>-dtai-gh account"
hpc::submit "/work/nvme/$ALLOC/$USER/openfold" "$ALLOC-dtai-gh"
