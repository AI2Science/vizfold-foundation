#!/bin/bash

# GT PACE Phoenix ("phoenix-slurm"). No mirror reachable; A100 gres in <site>.json.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 20 GB; install on scratch. ~/scratch is a symlink, so use its real
# /storage/scratch1 target (purged after 60 idle days; OPENFOLD_PREFIX to persist).
SCRATCH=$(readlink -f "$HOME/scratch") || die "cannot resolve ~/scratch"
hpc::submit "$SCRATCH/openfold"
