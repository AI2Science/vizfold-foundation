#!/bin/bash

# GT PACE Phoenix ("phoenix-slurm"). AF2 mirror + A100 gres in <site>.json; install on the user's /storage/scratch1 root (purged after 60 idle days).

slurm::prefix() { local s; s=$(slurm::scratch_root) || die "cannot resolve ~/scratch"; PREFIX_DEFAULT=$s/openfold; }
