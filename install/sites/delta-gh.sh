#!/bin/bash

# NCSA Delta-AI ("delta-gh"). Grace-Hopper aarch64; build on a GH200 node (no CPU queue); no mirror.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# environment.yml pins mkl (x86-only) + deepspeed=cuda (no aarch64 build): the env can't solve here. Refuse before burning GH200 hours.
[ "$(uname -m)" = aarch64 ] && die "Delta-AI is aarch64: environment.yml (mkl, deepspeed=cuda) has no aarch64 solve. Needs an aarch64 env variant (nomkl/openblas + deepspeed via pip)."

ALLOC=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(hpc::allocation /work/nvme -dtai-gh || true)")
[ -n "$ALLOC" ] || die "no usable allocation: need /work/nvme space and <alloc>-dtai-gh account"
hpc::submit "/work/nvme/$ALLOC/$USER/openfold" "$ALLOC-dtai-gh"
