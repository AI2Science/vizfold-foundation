#!/bin/bash
# ~/.config/vizfold/vizfold.json: what the install resolved, for whatever drives it later. Flat map; sourcing fills unset vars (inline wins).

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "config.sh is a library" >&2; exit 1; }
[ -n "${CONFIG_SH:-}" ] && return 0
CONFIG_SH=1

# The base every script shares. OPENFOLD_HOME wins, else walk up from this lib (<repo>/install/) to the checkout's setup.py.
REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
die() { echo "FATAL: $*" >&2; exit 1; }

config::file() {
    echo "${VIZFOLD_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/vizfold/vizfold.json}"
}

# Fill unset vars from a JSON file, never overwriting -- so inline > user file > site defaults.
config::fill() {
    local file=$1 label=${2:-config} key value
    [ -r "$file" ] && command -v python3 >/dev/null || return 0
    echo "$label: $file" >&2
    # `if`, not `&&`: a skipped last line would return non-zero and abort a set -e caller.
    while IFS='=' read -r key value; do
        if [ -n "$key" ] && [ -z "${!key:-}" ]; then export "$key=$value"; fi
    done < <(python3 -c '
import json, sys
try:
    items = json.load(open(sys.argv[1])).items()
except Exception:
    sys.exit(0)
for k, v in items:
    if isinstance(v, str) and "\n" not in v:
        print(f"{k}={v}")' "$file" 2>/dev/null)
    return 0
}

config::load() { config::fill "$(config::file)" "config"; }

# <site>.sh loads its own <site>.json: same basename, beside it.
config::site_defaults() { config::fill "${1%.sh}.json" "site defaults"; }

# Only names that are set are written, so an unused one leaves no empty key.
config::save() {
    local file
    file=$(config::file)
    mkdir -p "${file%/*}"
    python3 -c '
import json, os, sys
path, names = sys.argv[1], sys.argv[2:]
with open(path, "w") as f:
    json.dump({n: os.environ[n] for n in names if os.environ.get(n)},
              f, indent=2, sort_keys=True)
    f.write("\n")' "$file" "$@" &&
        echo "wrote $file" || echo "warning: could not write $file" >&2
}

config::load
