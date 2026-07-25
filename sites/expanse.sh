#!/bin/bash

# SDSC Expanse ("expanse"). No mirror; no default account, so the first association is the atom -- it doubles as the slurm account and as the /expanse project dir expanse.json templates.

slurm::discover() { OPENFOLD_ALLOCATION=$(interactive::resolve OPENFOLD_ALLOCATION allocation "${OPENFOLD_ALLOCATION:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep . | head -1)}"); [ -n "$OPENFOLD_ALLOCATION" ] || die "no usable allocation; set OPENFOLD_ALLOCATION"; export OPENFOLD_ALLOCATION OPENFOLD_ACCOUNT=${OPENFOLD_ACCOUNT:-$OPENFOLD_ALLOCATION}; }
