#!/bin/bash
# setup::link_mirror against the three uniclust30 layouts the mirrors actually ship.
# Run: bash tests/link_mirror.sh
#
# Every cluster lays uniclust30 out differently, so this is the one install step whose inputs vary
# per site. The failure it guards is quiet: the fold falls back to a full MSA search against a
# database that is not there.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SANDBOX=${TMPDIR:-/tmp}/vizfold-link-mirror-$$
trap 'rm -rf "$SANDBOX"' EXIT

run_case() {
    local name=$1 populate=$2
    rm -rf "$SANDBOX"; mkdir -p "$SANDBOX"
    AF2=$SANDBOX/mirror DATA=$SANDBOX/data UNICLUST=$SANDBOX/data/uniclust30/uniclust30_2018_08
    mkdir -p "$AF2" "$DATA"
    "$populate"

    # Definitions only: the guard at the foot of setup.sh returns when sourced.
    export OPENFOLD_HOME=$REPO VIZFOLD_CONFIG=$SANDBOX/none.json
    . "$REPO/backends/openfold/install/setup.sh"
    log() { :; }
    setup::link_mirror >/dev/null
    CASE=$name
}

assert_link() {
    local target=$1
    [ -L "$target" ] || { echo "FAIL [$CASE] $target is not a symlink"; exit 1; }
}
assert_missing() {
    [ ! -e "$1" ] || { echo "FAIL [$CASE] $1 should not exist"; exit 1; }
}

populate_flat() {
    mkdir -p "$AF2/uniref90" "$AF2/uniclust30"
    : > "$AF2/uniref90/uniref90.fasta"
    : > "$AF2/uniclust30/uniclust30_2018_08_a3m.ffdata"
    : > "$AF2/uniclust30/uniclust30_2018_08_cs219.ffindex"
}
populate_nested() {
    mkdir -p "$AF2/uniref90" "$AF2/uniclust30/uniclust30_2018_08"
    : > "$AF2/uniref90/uniref90.fasta"
    : > "$AF2/uniclust30/uniclust30_2018_08/uniclust30_2018_08_a3m.ffdata"
}
populate_uniref30_only() {
    mkdir -p "$AF2/uniref30"
    : > "$AF2/uniref30/UniRef30_2023_02_a3m.ffdata"
    : > "$AF2/uniref30/UniRef30_2023_02_cs219.ffindex"
}

run_case "single-nested" populate_flat
assert_link "$DATA/uniref90"
assert_missing "$DATA/uniclust30/uniclust30_2018_08/uniclust30_2018_08"   # a dir, not a stray file
assert_link "$UNICLUST/uniclust30_2018_08_a3m.ffdata"
assert_link "$UNICLUST/uniclust30_2018_08_cs219.ffindex"
echo "ok   a single-nested mirror stages uniclust30 into the writable dir"

run_case "double-nested" populate_nested
assert_link "$UNICLUST/uniclust30_2018_08_a3m.ffdata"
echo "ok   a double-nested mirror resolves to the same place"

run_case "uniref30-only" populate_uniref30_only
assert_link "$UNICLUST/uniclust30_2018_08_a3m.ffdata"
assert_link "$UNICLUST/uniclust30_2018_08_cs219.ffindex"
[ "$(readlink "$UNICLUST/uniclust30_2018_08_a3m.ffdata")" = "$AF2/uniref30/UniRef30_2023_02_a3m.ffdata" ] ||
    { echo "FAIL [uniref30-only] the alias does not point at the uniref30 file"; exit 1; }
echo "ok   with no uniclust30, uniref30 is aliased under the uniclust30 names"

# The mirror is read-only in practice.
run_case "mirror-untouched" populate_flat
[ -z "$(find "$AF2" -type l)" ] || { echo "FAIL [mirror-untouched] a link was created inside the mirror"; exit 1; }
echo "ok   nothing is written into the mirror itself"
