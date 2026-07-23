#!/bin/bash

# Purdue Anvil ("anvil"). GPU jobs charge <account>-gpu (suffix in <site>.json); install on $PROJECT, not purged $SCRATCH.

slurm::prefix() { PREFIX_DEFAULT=${PROJECT:+$PROJECT/$USER/openfold}; PREFIX_DEFAULT=${PREFIX_DEFAULT:-/anvil/scratch/$USER/openfold}; }
