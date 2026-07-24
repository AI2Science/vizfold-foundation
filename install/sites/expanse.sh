#!/bin/bash

# SDSC Expanse ("expanse"). No mirror; no default account, so the first association is the atom: expanse.json templates both the slurm account and the /expanse project dir off $ALLOC.

slurm::discover() { ALLOC=$(interactive::resolve OPENFOLD_ALLOCATION allocation "${ALLOC:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep . | head -1)}"); [ -n "$ALLOC" ] || die "no usable allocation; set OPENFOLD_ALLOCATION"; export ALLOC; }
