#!/bin/bash

# NCSA Delta ("delta"). Per-allocation /work/nvme + -delta-{cpu,gpu} accounts; prefix/accounts templated in <site>.json off $ALLOC.

slurm::discover() { slurm::nvme_alloc -delta-cpu -delta-gpu; }
