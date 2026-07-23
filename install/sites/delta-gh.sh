#!/bin/bash

# NCSA Delta-AI ("delta-gh"). Grace-Hopper aarch64 (setup.sh uses environment-aarch64.yml); build on a GH200 node, no CPU queue.

slurm::_alloc()  { slurm::nvme_alloc -dtai-gh; }
slurm::prefix()  { slurm::_alloc; PREFIX_DEFAULT=/work/nvme/$ALLOC/$USER/openfold; }
slurm::account() { slurm::_alloc; ACCOUNT_DEFAULT=$ALLOC-dtai-gh; }
