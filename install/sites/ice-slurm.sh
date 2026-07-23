#!/bin/bash

# GT PACE ICE ("ice-slurm"). AF2 mirror + A100 gres pinned in <site>.json.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 30 GB; install under ~/scratch, resolved to its real /storage path. Set OPENFOLD_PREFIX to a project volume to persist.
SCRATCH=$(readlink -f "$HOME/scratch" 2>/dev/null || echo "$HOME/scratch")
hpc::submit "$SCRATCH/openfold"
