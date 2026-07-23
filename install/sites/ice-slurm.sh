#!/bin/bash

# GT PACE ICE ("ice-slurm"). AF2 mirror + A100 gres pinned in <site>.json.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 30 GB; install on scratch. ~/scratch is a symlink, so use its real /storage
# target. Set OPENFOLD_PREFIX to a project volume to persist (scratch is purged).
SCRATCH=$(readlink -f "$HOME/scratch") || die "cannot resolve ~/scratch"
hpc::submit "$SCRATCH/openfold"
