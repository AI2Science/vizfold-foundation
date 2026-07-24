#!/bin/bash

# NCSA Delta-AI ("delta-gh"). Grace-Hopper aarch64 (setup.sh uses environment-aarch64.yml); build on a GH200 node, no CPU queue.
# /work/nvme is SHARED with x86 Delta, so <site>.json templates an -gh prefix suffix: the aarch64 env must not clobber Delta's.

slurm::discover() { slurm::nvme_alloc -dtai-gh; }
