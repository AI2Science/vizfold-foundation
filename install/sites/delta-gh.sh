#!/bin/bash

# NCSA Delta-AI ("delta-gh"). Grace-Hopper aarch64 (setup.sh uses environment-aarch64.yml); build on a GH200 node, no CPU queue.

site::_alloc()  { hpc::nvme_alloc -dtai-gh; }
site::prefix()  { site::_alloc; PREFIX_DEFAULT=/work/nvme/$ALLOC/$USER/openfold; }
site::account() { site::_alloc; ACCOUNT_DEFAULT=$ALLOC-dtai-gh; }
