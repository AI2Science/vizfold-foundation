#!/bin/bash

# NCSA Delta ("delta"). Per-allocation /work/nvme + -delta-{cpu,gpu} accounts; AF2 mirror in <site>.json.

slurm::_alloc()      { slurm::nvme_alloc -delta-cpu -delta-gpu; }
slurm::prefix()      { slurm::_alloc; PREFIX_DEFAULT=/work/nvme/$ALLOC/$USER/openfold; }
slurm::account()     { slurm::_alloc; ACCOUNT_DEFAULT=$ALLOC-delta-cpu; }
slurm::gpu_account() { slurm::_alloc; GPU_ACCOUNT_DEFAULT=$ALLOC-delta-gpu; }
