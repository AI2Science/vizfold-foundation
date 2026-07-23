#!/bin/bash

# Nexus ("nexus-dev"). No mirror; 10 GB A100 vGPU on a 535 driver (setup.sh pins NVRTC).
set -euo pipefail

. "$(dirname "${BASH_SOURCE[0]}")/../hpc.sh"
config::site_defaults "${BASH_SOURCE[0]}"

# Bulk data goes on the shared /projects volume, not $HOME.
for c in "/projects/$USER" /projects/*/"$USER" /projects; do
    [ -d "$c" ] && [ -w "$c" ] && { BASE=$c; break; }
done
hpc::submit "${BASE:-$HOME}/openfold"
