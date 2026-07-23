#!/bin/bash

# SDSC Expanse ("expanse"). No mirror; no default account, so the first association is used for both account and /expanse dir.

site::_acct() { [ -n "${ACCT:-}" ] || ACCT=${OPENFOLD_ACCOUNT:-$(sacctmgr -nP show assoc user="$USER" format=Account 2>/dev/null | grep . | head -1)}; }
site::prefix()  { site::_acct; PREFIX_DEFAULT=/expanse/lustre/projects/$ACCT/$USER/openfold; }
site::account() { site::_acct; ACCOUNT_DEFAULT=$ACCT; }
