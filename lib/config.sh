#!/bin/bash
# ~/.config/vizfold/vizfold.json: what the install resolved, for whatever drives it later. Flat map; sourcing fills unset vars (inline wins).

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "config.sh is a library" >&2; exit 1; }
[ -n "${CONFIG_SH:-}" ] && return 0
CONFIG_SH=1

# The checkout root every backend shares. OPENFOLD_HOME wins (exported by `vizfold install`);
# otherwise it is one level up from this neutral lib at lib/config.sh.
REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# The OpenFold backend subtree, for OpenFold's own scripts (esmfold never reads OF). Backend-local
# files (setup.py, environment.yml, install/) live here; shared demo assets (examples/) at the root.
OF=${OPENFOLD_DIR:-$REPO/backends/openfold}
die() { echo "FATAL: $*" >&2; exit 1; }

config::file() {
    echo "${VIZFOLD_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/vizfold/vizfold.json}"
}

# Fill unset vars from a JSON file, never overwriting -- so inline > user file > site defaults.
# Values are templates: $VAR/${VAR} resolve against the environment first, then against other keys in
# the same file, recursively -- so a <site>.json builds OPENFOLD_PREFIX off OPENFOLD_BASE off a
# discovered $ALLOC, in any key order. No commands run; an unresolved name expands to empty.
config::fill() {
    local file=$1 label=${2:-config} key value
    [ -r "$file" ] && command -v python3 >/dev/null || return 0
    echo "$label: $file" >&2
    # `if`, not `&&`: a skipped last line would return non-zero and abort a set -e caller.
    while IFS='=' read -r key value; do
        if [ -n "$key" ] && [ -z "${!key:-}" ]; then export "$key=$value"; fi
    done < <(python3 -c '
import json, os, re, sys
try:
    scope = {k: v for k, v in json.load(open(sys.argv[1])).items() if isinstance(v, str) and "\n" not in v}
except Exception:
    sys.exit(0)
ref = re.compile(r"\$\{(\w+)\}|\$(\w+)")
def resolve(name, seen):
    if name in os.environ: return os.environ[name]           # inline / discovered / user-file wins
    if name in scope and name not in seen: return expand(scope[name], seen | {name})
    return ""                                                # unknown -> empty (discovery dies if an atom is missing)
def expand(val, seen):
    return ref.sub(lambda m: resolve(m.group(1) or m.group(2), seen), val)
for k, v in scope.items():
    if k not in os.environ:                                  # fill unset only
        print(f"{k}={expand(v, {k})}")' "$file" 2>/dev/null)
    return 0
}

# Activate a micromamba env ($2, a name or path) via its binary ($1). set +u: the conda gcc hook reads SYS_SYSROOT unset.
mamba::activate() { set +u; eval "$("$1" shell hook --shell bash)"; micromamba activate "$2"; set -u; }

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
