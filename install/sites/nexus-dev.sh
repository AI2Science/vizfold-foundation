#!/bin/bash
# Nexus (ClusterName "nexus-dev"). Reached from ../../install.sh.
# No database mirror; setup.sh fetches the parameters and example templates.
# The GPU is a 10 GB A100 vGPU with a 535 driver; setup.sh pins NVRTC to match.
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# Bulk data goes on the shared /projects volume, not $HOME.
for c in "/projects/$USER" /projects/*/"$USER" /projects; do
    [ -d "$c" ] && [ -w "$c" ] && { BASE=$c; break; }
done
hpc::submit "${BASE:-$HOME}/openfold"
