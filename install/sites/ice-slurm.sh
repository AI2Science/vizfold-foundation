#!/bin/bash
# Georgia Tech PACE ICE (ClusterName "ice-slurm"). Reached from ../../install.sh.
# AF2 mirror named in <site>.json; the mixed GPU queue pins an A100 there too.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 30 GB, so install under ~/scratch -- but resolve the symlink to its real
# /storage path so every user gets their own concrete location, not a link. Set
# OPENFOLD_PREFIX to a project volume if you have one and want it to persist.
SCRATCH=$(readlink -f "$HOME/scratch" 2>/dev/null || echo "$HOME/scratch")
hpc::submit "$SCRATCH/openfold"
