#!/bin/bash

# NCSA Delta ("delta"). Per-allocation /work/nvme + -delta-{cpu,gpu} accounts; AF2 mirror in <site>.json.

site::_alloc() {
    [ -n "${ALLOC:-}" ] && return
    ALLOC=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(hpc::allocation /work/nvme -delta-cpu -delta-gpu || true)")
    [ -n "$ALLOC" ] || die "no usable allocation: need /work/nvme space and <alloc>-delta-{cpu,gpu} accounts"
}
site::prefix()      { site::_alloc; PREFIX_DEFAULT=/work/nvme/$ALLOC/$USER/openfold; }
site::account()     { site::_alloc; ACCOUNT_DEFAULT=$ALLOC-delta-cpu; }
site::gpu_account() { site::_alloc; GPU_ACCOUNT_DEFAULT=$ALLOC-delta-gpu; }
