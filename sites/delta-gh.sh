#!/bin/bash

# NCSA Delta-AI ("delta-gh"). Grace-Hopper aarch64, GH200 build node, no CPU queue. /work/nvme is shared
# with x86 Delta, so <site>.json suffixes the prefix -gh: the aarch64 env must not clobber Delta's.

slurm::discover() { slurm::nvme_alloc -dtai-gh; }
