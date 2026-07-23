#!/bin/bash
# Georgia Tech PACE Phoenix (ClusterName "phoenix-slurm"). From ../../install.sh.
# No database mirror reachable here; setup.sh fetches the parameters and templates.
# The mixed GPU queue pins an A100 in <site>.json.
set -euo pipefail

REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
. "$REPO/install/hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# $HOME is 20 GB, so the env and databases go on ~/scratch. This account has no
# persistent project space, and Phoenix scratch is purged after 60 idle days --
# set OPENFOLD_PREFIX to a project volume if you have one and want it to persist.
hpc::submit "$HOME/scratch/openfold"
