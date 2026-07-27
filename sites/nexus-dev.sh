#!/bin/bash

# Nexus. AF2 mirror on the staging volume; 10 GB A100 vGPU on a 535 driver (setup.sh pins NVRTC); OPENFOLD_BASE = a writable /projects dir, not $HOME.

slurm::discover() {
    local c
    for c in "/projects/$USER" /projects/*/"$USER" /projects; do
        [ -d "$c" ] && [ -w "$c" ] && { export OPENFOLD_BASE=$c; return; }
    done
    export OPENFOLD_BASE=$HOME
}
