#!/bin/bash
# Georgia Tech PACE Phoenix (ClusterName "phoenix-slurm"). From ../../install.sh.
# No database mirror reachable here; setup.sh fetches the parameters and templates.
# The mixed GPU queue pins an A100 in <site>.json.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 20 GB, so install under ~/scratch -- but resolve the symlink to its real
# /storage/scratch1 path so every user gets a concrete location, not a link. That
# scratch is purged after 60 idle days; set OPENFOLD_PREFIX to a project volume to persist.
SCRATCH=$(readlink -f "$HOME/scratch" 2>/dev/null || echo "$HOME/scratch")
hpc::submit "$SCRATCH/openfold"
