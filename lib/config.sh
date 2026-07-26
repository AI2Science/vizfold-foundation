#!/bin/bash
# ~/.config/vizfold/vizfold.json: what the install resolved, for whatever drives it later. Flat map; sourcing fills unset vars (inline wins).

[ "${BASH_SOURCE[0]}" = "$0" ] && { echo "config.sh is a library" >&2; exit 1; }
[ -n "${CONFIG_SH:-}" ] && return 0
CONFIG_SH=1

# The checkout root every backend shares; OPENFOLD_HOME is exported by `vizfold install`.
REPO=${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# OpenFold-only subtree (setup.py, environment.yml, install/); shared assets like examples/ stay at the root.
OF=$REPO/backends/openfold
die() { echo "FATAL: $*" >&2; exit 1; }

# Progress line, shared so every installer's output reads the same.
log() { echo "== $* (+$((SECONDS))s)"; }

# The install root and the one env base under it, each env a fixed vizfold-<backend> name.
# Mirrored by env_base()/env_dir() in cli/src/core/config.rs.
vizfold::prefix() { echo "${OPENFOLD_PREFIX:-$HOME/openfold}"; }
vizfold::env_base() { echo "${VIZFOLD_ENV_BASE:-$(vizfold::prefix)/envs}"; }
vizfold::env() { echo "$(vizfold::env_base)/vizfold-$1"; }

config::file() {
    echo "${VIZFOLD_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/vizfold/vizfold.json}"
}

# Fill unset vars from a JSON file, never overwriting -- so inline > user file > site defaults.
# Values are templates: $VAR/${VAR} resolves against the environment first, then against the file's
# own keys, recursively and in any key order. No commands run; an unresolved name expands to empty.
config::fill() {
    local file=$1 label=${2:-config} key value
    [ -r "$file" ] && command -v python3 >/dev/null || return 0
    echo "$label: $file" >&2
    # `if`, not `&&`: a skipped last line would return non-zero and abort a set -e caller.
    while IFS='=' read -r key value; do
        # An empty value is "not settled", never an answer: it must not mask the layer below it.
        if [ -n "$key" ] && [ -n "$value" ] && [ -z "${!key:-}" ]; then export "$key=$value"; fi
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
    if k not in os.environ:
        print(f"{k}={expand(v, {k})}")' "$file" 2>/dev/null)
    return 0
}

# Activate a micromamba env ($2, a name or path) via its binary ($1). set +u: the conda gcc hook reads SYS_SYSROOT unset.
mamba::activate() { set +u; eval "$("$1" shell hook --shell bash)"; micromamba activate "$2"; set -u; }

# micromamba at <prefix>/bin/micromamba, downloaded once: every backend's environment and the Node
# one `vizfold serve` builds come from this one copy.
mamba::ensure() {
    local prefix=$1 mm=$1/bin/micromamba build
    if ! "$mm" --version >/dev/null 2>&1; then
        case "$(uname -s)-$(uname -m)" in
            Linux-aarch64|Linux-arm64)   build=linux-aarch64 ;;
            Linux-*)                     build=linux-64 ;;
            Darwin-arm64|Darwin-aarch64) build=osx-arm64 ;;
            Darwin-*)                    build=osx-64 ;;
            *) die "no micromamba build for $(uname -s)-$(uname -m)" ;;
        esac
        mkdir -p "$prefix"
        curl -Ls "https://micro.mamba.pm/api/micromamba/$build/latest" | tar -xj -C "$prefix" bin/micromamba
    fi
    echo "$mm"
}

# The previous install's answers. Called explicitly, and deliberately not at source time: loading
# here would put them in the environment before a caller has run its own discovery, and nothing
# downstream can then tell a value the user chose from one an earlier install invented. The
# installers call this after their <site>.json, which fixes the precedence at
#   inline env > slurm::discover > <site>.json > saved vizfold.json > built-in default
config::load() { config::fill "$(config::file)" "config"; }

# <site>.sh loads its own <site>.json: same basename, beside it.
config::site_defaults() { config::fill "${1%.sh}.json" "site defaults"; }

# The config's schema: the same binary reads it everywhere, so the key set is fixed -- same on every cluster, whichever backend installed last.
VIZFOLD_CONFIG_KEYS="OPENFOLD_HOME OPENFOLD_PREFIX OPENFOLD_SITE OPENFOLD_DATA_DIR OPENFOLD_AF2_ROOT
VIZFOLD_ENV_BASE OPENFOLD_ENV_PREFIX ESMFOLD_ENV_PREFIX OPENFOLD_MAX_CUDA OPENFOLD_DRIVER_CUDA
OPENFOLD_ACCOUNT OPENFOLD_PARTITION OPENFOLD_GPU_ACCOUNT OPENFOLD_GPU_PARTITION
OPENFOLD_GPU_RESOURCES OPENFOLD_GPU_GRES OPENFOLD_GPU_TIME OPENFOLD_EXAMPLE
VIZFOLD_DB"

# Every key, every time -- empty for what this install did not settle. config::load has already put
# the previous install's values in the environment, so a second backend rewrites them rather than
# dropping the keys it does not know about.
config::save() {
    local file
    file=$(config::file)
    mkdir -p "${file%/*}"
    python3 -c '
import json, os, sys
path, names = sys.argv[1], sys.argv[2:]
with open(path, "w") as f:
    json.dump({n: os.environ.get(n, "") for n in names}, f, indent=2, sort_keys=True)
    f.write("\n")' "$file" $VIZFOLD_CONFIG_KEYS ||
        die "could not write $file"   # not a warning: everything downstream reads this file
    echo "wrote $file"

}
