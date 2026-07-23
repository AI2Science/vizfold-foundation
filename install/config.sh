#!/bin/bash
# What the install resolved, for anything that drives it later -- rust-core, the
# portals, a shell. Sourcing this file loads them; an inlined variable always
# wins, the file only fills what is unset.
#
#   ~/.config/vizfold/vizfold.json    {"OPENFOLD_PREFIX": "/work/...", ...}
#
# Flat, so a consumer can read it as a string map and export it verbatim.

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "config.sh is a library" >&2; exit 1; }
[ -n "${CONFIG_SH:-}" ] && return 0
CONFIG_SH=1

# The base every install script shares: where the checkout is, and how to fail.
# OPENFOLD_HOME wins (install.sh exports it); otherwise walk up from this library,
# which lives at <repo>/install/, to the setup.py at the checkout root.
REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && until [ -f setup.py ] || [ "$PWD" = / ]; do cd ..; done; pwd)}
die() { echo "FATAL: $*" >&2; exit 1; }

config::file() {
    echo "${VIZFOLD_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/vizfold/vizfold.json}"
}

# Fill unset variables from a JSON file. Never overwrites, so the first caller
# to set a value keeps it: inline environment, then the user's file, then the
# site defaults -- highest precedence first, each only filling what is missing.
config::fill() {
    local file=$1 label=${2:-config} key value
    [ -r "$file" ] && command -v python3 >/dev/null || return 0
    echo "$label: $file" >&2
    # `if`, not `&&`: a skipped last line would make the loop -- and sourcing this
    # file -- return non-zero, which aborts a `set -e` caller.
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
