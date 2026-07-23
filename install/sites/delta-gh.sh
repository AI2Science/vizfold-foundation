#!/bin/bash

# NCSA Delta-AI ("delta-gh"). Grace-Hopper aarch64 (setup.sh uses environment-aarch64.yml); build on a GH200 node, no CPU queue.
# /work/nvme is SHARED with x86 Delta, so use an -gh suffix: the aarch64 env must not clobber Delta's openfold prefix.

slurm::_alloc()  { slurm::nvme_alloc -dtai-gh; }
slurm::prefix()  { slurm::_alloc; PREFIX_DEFAULT=/work/nvme/$ALLOC/$USER/openfold-gh; }
slurm::account() { slurm::_alloc; ACCOUNT_DEFAULT=$ALLOC-dtai-gh; }
