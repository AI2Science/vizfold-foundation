#!/bin/bash

# GT PACE Phoenix ("phoenix-slurm"). No mirror; A100 gres in <site>.json; install on the user's /storage/scratch1 root (purged after 60 idle days).

site::prefix() { local s; s=$(hpc::scratch_root) || die "cannot resolve ~/scratch"; PREFIX_DEFAULT=$s/openfold; }
