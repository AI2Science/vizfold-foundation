#!/bin/bash

# NCSA Delta-AI ("delta-gh"). Grace-Hopper aarch64 (setup.sh uses environment-aarch64.yml); build on a GH200 node, no CPU queue.

site::_alloc() {
    [ -n "${ALLOC:-}" ] && return
    ALLOC=$(interactive::resolve OPENFOLD_ALLOCATION allocation "$(hpc::allocation /work/nvme -dtai-gh || true)")
    [ -n "$ALLOC" ] || die "no usable allocation: need /work/nvme space and <alloc>-dtai-gh account"
}
site::prefix()  { site::_alloc; PREFIX_DEFAULT=/work/nvme/$ALLOC/$USER/openfold; }
site::account() { site::_alloc; ACCOUNT_DEFAULT=$ALLOC-dtai-gh; }
