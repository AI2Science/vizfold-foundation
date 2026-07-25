#!/bin/bash

# SDSC Expanse ("expanse"). No mirror; no default account, so the first association is the atom -- it doubles as the slurm account and as the /expanse project dir expanse.json templates.

slurm::discover() { ALLOC=$(interactive::resolve OPENFOLD_ALLOCATION allocation "${ALLOC:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep . | head -1)}"); [ -n "$ALLOC" ] || die "no usable allocation; set OPENFOLD_ALLOCATION"; export ALLOC OPENFOLD_ACCOUNT=${OPENFOLD_ACCOUNT:-$ALLOC}; }
