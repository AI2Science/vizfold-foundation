#!/bin/bash

# NCSA Delta. Per-allocation /work/nvme, a separate account per queue; delta.json templates the base off the discovered $OPENFOLD_ALLOCATION.

slurm::discover() { slurm::nvme_alloc -delta-cpu -delta-gpu; }
