#!/bin/bash

# Nexus ("nexus-dev"). No mirror; 10 GB A100 vGPU on a 535 driver (setup.sh pins NVRTC); bulk data on shared /projects, not $HOME.

slurm::prefix() {
    local c base
    for c in "/projects/$USER" /projects/*/"$USER" /projects; do
        [ -d "$c" ] && [ -w "$c" ] && { base=$c; break; }
    done
    PREFIX_DEFAULT=${base:-$HOME}/openfold
}
