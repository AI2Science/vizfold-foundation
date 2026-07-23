#!/bin/bash

# NCSA Delta ("delta"). Per-allocation /work/nvme + -delta-{cpu,gpu} accounts; AF2 mirror in <site>.json.

site::_alloc()      { hpc::nvme_alloc -delta-cpu -delta-gpu; }
site::prefix()      { site::_alloc; PREFIX_DEFAULT=/work/nvme/$ALLOC/$USER/openfold; }
site::account()     { site::_alloc; ACCOUNT_DEFAULT=$ALLOC-delta-cpu; }
site::gpu_account() { site::_alloc; GPU_ACCOUNT_DEFAULT=$ALLOC-delta-gpu; }
