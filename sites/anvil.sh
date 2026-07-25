#!/bin/bash

# Purdue Anvil ("anvil"). GPU jobs charge <account>-gpu (suffix in <site>.json); the prefix defaults to $OPENFOLD_BASE/vizfold (slurm::default_prefix), with OPENFOLD_BASE = $PROJECT/$USER (not the purged $SCRATCH), falling back to /anvil/scratch/$USER.

slurm::discover() { OPENFOLD_BASE=${PROJECT:+$PROJECT/$USER}; export OPENFOLD_BASE=${OPENFOLD_BASE:-/anvil/scratch/$USER}; }
