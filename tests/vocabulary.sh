#!/bin/bash
# One vocabulary: no name the config never carries, no site key nothing consumes. Run: bash tests/vocabulary.sh
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
VIZFOLD_CONFIG=/nonexistent/vocabulary-test.json; export VIZFOLD_CONFIG
. "$REPO/lib/config.sh"

fail=0
schema=$(printf '%s\n' $VIZFOLD_CONFIG_KEYS | sort)

# Site keys the install consumes but never persists: a re-install re-reads the same file.
install_only="OPENFOLD_ALLOCATION OPENFOLD_BASE OPENFOLD_BUILD_CPUS OPENFOLD_BUILD_GRES
OPENFOLD_BUILD_MEM OPENFOLD_BUILD_TIME OPENFOLD_GPU_ACCOUNT_SUFFIX"

report() { # $1 label, $2 newline-separated offenders, $3 remedy
    if [ -z "$2" ]; then echo "ok   $1"; else
        echo "FAIL $1"; printf '       %s\n' $2; echo "       -> $3"; fail=1
    fi
}

# Every resolved("X") in the CLI must be a schema key, or the binary expects what no install writes.
cli=$(grep -oE 'resolved\("[A-Z0-9_]+"\)' "$REPO/cli/src/core/config.rs" |
    sed 's/resolved("//; s/")//' | sort -u)
report "every name the CLI resolves is in the config schema" \
    "$(comm -23 <(printf '%s\n' $cli) <(printf '%s\n' $schema))" \
    "add it to VIZFOLD_CONFIG_KEYS, or read a name the install already settles"

# Every <site>.json key must be persisted or knowingly install-only -- nothing set and forgotten.
known=$(printf '%s\n' $schema $install_only | sort -u)
site_keys=$(python3 -c '
import json, pathlib
print(*sorted({k for f in pathlib.Path("sites").glob("*.json") for k in json.loads(f.read_text())}))')
report "every site key is persisted or knowingly install-only" \
    "$(comm -23 <(printf '%s\n' $site_keys) <(printf '%s\n' $known))" \
    "add it to VIZFOLD_CONFIG_KEYS, or to install_only here if the install consumes it and stops"

# An unprovided $VAR expands to empty: this is how "$ALLOC-delta-cpu" once became the account "-delta-cpu".
templated=$(grep -ohE '\$\{?[A-Z0-9_]+\}?' sites/*.json | tr -d '${}' | sort -u)
provided=$(printf '%s\n' $schema $install_only USER HOME \
    $(grep -ohE 'export [A-Z0-9_]+|[A-Z0-9_]+=' sites/*.sh lib/slurm.sh | tr -d '=' | sed 's/export //') |
    sort -u)
report "every site template atom is provided by a discover hook or the schema" \
    "$(comm -23 <(printf '%s\n' $templated) <(printf '%s\n' $provided))" \
    "export it from the site's slurm::discover, or it silently expands to empty"

# Either side alone would judge the other's config stale, so the two spellings stay one list.
cli_schema=$(sed -n '/pub const CONFIG_KEYS/,/^];/p' "$REPO/cli/src/core/config.rs" |
    grep -oE '"[A-Z0-9_]+"' | tr -d '"' | sort)
report "the CLI's copy of the schema is the schema" \
    "$(comm -3 <(printf '%s\n' $cli_schema) <(printf '%s\n' $schema) | tr -d '\t')" \
    "make CONFIG_KEYS in cli/src/core/config.rs and VIZFOLD_CONFIG_KEYS name the same set"

exit $fail
