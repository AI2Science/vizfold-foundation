#!/bin/bash
# What `vizfold install repo` leaves behind: a full config, a prefix that exists, and the run
# database it was already pointed at. Run: bash tests/configure_repo.sh
#
# The prefix matters as much as the keys: status reads a recorded-but-absent prefix as a broken
# config, so writing the answer without creating the directory would redden a fresh install.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SANDBOX=${TMPDIR:-/tmp}/vizfold-configure-$$
trap 'rm -rf "$SANDBOX"' EXIT
fail=0
mkdir -p "$SANDBOX"

run() { # $1... extra env
    rm -rf "$SANDBOX/pfx" "$SANDBOX/config.json"
    env OPENFOLD_HOME="$REPO" VIZFOLD_CONFIG="$SANDBOX/config.json" OPENFOLD_SITE=local \
        OPENFOLD_PREFIX="$SANDBOX/pfx" "$@" bash "$REPO/configure.sh" >/dev/null 2>&1
}

check() { if [ "$2" = "$3" ]; then echo "ok   $1"; else
    echo "FAIL $1"; echo "  want $3"; echo "  got  $2"; fail=1; fi; }

run
keys=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$SANDBOX/config.json" 2>/dev/null)
check "the config carries the whole 19-key schema" "$keys" "19"
check "the recorded prefix exists" "$([ -d "$SANDBOX/pfx/envs" ] && echo yes || echo no)" "yes"

get() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],""))' "$SANDBOX/config.json" "$1"; }
check "OPENFOLD_HOME is the checkout that ran it" "$(get OPENFOLD_HOME)" "$REPO"
check "VIZFOLD_DB defaults under the prefix" "$(get VIZFOLD_DB)" "$SANDBOX/pfx/vizfold.db"

# A second install must not move someone's run history.
run VIZFOLD_DB="$SANDBOX/mine.db"
check "an existing VIZFOLD_DB is left alone" "$(get VIZFOLD_DB)" "$SANDBOX/mine.db"

exit $fail
