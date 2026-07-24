#!/bin/bash

# Purdue Anvil ("anvil"). GPU jobs charge <account>-gpu (suffix in <site>.json); prefix templates off OPENFOLD_BASE = $PROJECT/$USER (not the purged $SCRATCH), falling back to /anvil/scratch/$USER.

slurm::discover() { OPENFOLD_BASE=${PROJECT:+$PROJECT/$USER}; export OPENFOLD_BASE=${OPENFOLD_BASE:-/anvil/scratch/$USER}; }
