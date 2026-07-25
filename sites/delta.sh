#!/bin/bash

# NCSA Delta ("delta"). Per-allocation /work/nvme, and a separate account per queue (named by slurm::nvme_alloc from its own suffixes); delta.json templates the base off the discovered $OPENFOLD_ALLOCATION.

slurm::discover() { slurm::nvme_alloc -delta-cpu -delta-gpu; }
