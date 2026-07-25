#!/bin/bash

# NCSA Delta ("delta"). Per-allocation /work/nvme, and a separate account per queue; <site>.json templates the base off the discovered $ALLOC.

slurm::discover() { slurm::nvme_alloc -delta-cpu -delta-gpu; }
