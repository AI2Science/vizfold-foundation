#!/bin/bash

# SDSC Expanse ("expanse"). No mirror; no default account, so the first association is used for both account and /expanse dir.

slurm::_acct() { [ -n "${ACCT:-}" ] || ACCT=${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep . | head -1)}; }
slurm::prefix()  { slurm::_acct; PREFIX_DEFAULT=/expanse/lustre/projects/$ACCT/$USER/openfold; }
slurm::account() { slurm::_acct; ACCOUNT_DEFAULT=$ACCT; }
