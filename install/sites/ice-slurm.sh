#!/bin/bash

# GT PACE ICE ("ice-slurm"). AF2 mirror + A100 gres in <site>.json; install on the user's /storage/ice1 scratch root (purged at semester end).

site::prefix() { local s; s=$(hpc::scratch_root) || die "cannot resolve ~/scratch"; PREFIX_DEFAULT=$s/openfold; }
