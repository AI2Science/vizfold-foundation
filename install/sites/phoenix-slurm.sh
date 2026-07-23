#!/bin/bash

# GT PACE Phoenix ("phoenix-slurm"). No mirror reachable; A100 gres in <site>.json.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 20 GB; install under ~/scratch (resolved to real /storage/scratch1, purged after 60 idle days). OPENFOLD_PREFIX to persist.
SCRATCH=$(readlink -f "$HOME/scratch" 2>/dev/null || echo "$HOME/scratch")
hpc::submit "$SCRATCH/openfold"
