#!/bin/bash

# GT PACE ICE ("ice-slurm"). AF2 mirror + A100 gres in <site>.json; prefix templates off OPENFOLD_BASE = the user's /storage/ice1 scratch root (purged at semester end).

slurm::discover() { OPENFOLD_BASE=$(slurm::scratch_root) || die "cannot resolve ~/scratch"; export OPENFOLD_BASE; }
