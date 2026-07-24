#!/bin/bash

# GT PACE Phoenix ("phoenix-slurm"). AF2 mirror + A100 gres in <site>.json; prefix templates off OPENFOLD_BASE = the user's /storage/scratch1 root (purged after 60 idle days).

slurm::discover() { OPENFOLD_BASE=$(slurm::scratch_root) || die "cannot resolve ~/scratch"; export OPENFOLD_BASE; }
