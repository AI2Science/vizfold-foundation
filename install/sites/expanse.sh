#!/bin/bash

# SDSC Expanse ("expanse"). No mirror; no default account, so the first association is used for both the account and the /expanse dir the prefix templates off.

slurm::discover() { export OPENFOLD_ACCOUNT=${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep . | head -1)}; }
