#!/bin/bash
# Georgia Tech PACE ICE (ClusterName "ice-slurm"). Reached from ../../install.sh.
# AF2 mirror named in <site>.json; the mixed GPU queue pins an A100 there too.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 30 GB, so the env and databases go on ~/scratch. This account has no
# persistent project space, and ICE scratch is wiped at semester end -- set
# OPENFOLD_PREFIX to a project volume if you have one and want it to persist.
hpc::submit "$HOME/scratch/openfold"
