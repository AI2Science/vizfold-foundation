#!/bin/bash
# Every site's fully-resolved install config, snapshotted. Run: bash tests/site_config.sh (-u to accept).
# Runs the real flow -- discover, templating, slurm::run's exports, setup.sh's defaults -- so a
# <site>.json that restates a default and one that omits it are proven to resolve identically.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
EXPECTED=tests/site_config.expected
SANDBOX=${TMPDIR:-/tmp}/vizfold-site-config-$$
trap 'rm -rf "$SANDBOX"' EXIT

# The one login-specific atom each discover reads off the cluster; nexus-dev probes paths and skips it.
site_env() {
    case $1 in
        delta|delta-gh) echo 'export OPENFOLD_ALLOCATION=bbka' ;;
        ice-slurm)      echo 'SCRATCH=/storage/ice1/x-test' ;;
        phoenix-slurm)  echo 'SCRATCH=/storage/scratch1/x-test' ;;
        nexus-dev)      echo 'export OPENFOLD_BASE=/projects/x-test; skip=1' ;;
    esac
}

resolve() {
    local site=$1 SCRATCH= skip= arch=x86_64
    eval "$(site_env "$site")"
    [ "$site" = delta-gh ] && arch=aarch64                     # the one non-x86 site

    export USER=x-test HOME=$SANDBOX/home OPENFOLD_SITE=$site
    export VIZFOLD_CONFIG=$SANDBOX/$site.json OPENFOLD_HOME=$REPO

    . "$REPO/lib/slurm.sh"
    . "$SITES/$site.sh"
    # After the sources, so these win: slurm.sh defines its own scratch_root.
    uname() { [ "${1:-}" = -m ] && echo "$arch" || command uname "$@"; }
    sacctmgr() { echo bbka; }
    slurm::scratch_root() { echo "$SCRATCH"; }

    [ -n "$skip" ] || slurm::discover
    config::site_defaults "$SITES/$site.sh"

    # What slurm::run settles before handing off to setup.sh.
    export OPENFOLD_PREFIX=${OPENFOLD_PREFIX:-$(slurm::default_prefix)}
    export OPENFOLD_ACCOUNT=${OPENFOLD_ACCOUNT:-$(slurm::default_account)}
    export OPENFOLD_GPU_ACCOUNT=${OPENFOLD_GPU_ACCOUNT:-${OPENFOLD_ACCOUNT:+$OPENFOLD_ACCOUNT${OPENFOLD_GPU_ACCOUNT_SUFFIX:-}}}

    . "$SANDBOX/setup-defs.sh"
    export CONDA_PREFIX=$OPENFOLD_PREFIX/envs/vizfold-openfold
    setup::config
    setup::fold_vars
    setup::config_save >/dev/null

    echo "## $site"
    echo "account=$OPENFOLD_ACCOUNT partition=${OPENFOLD_PARTITION:-}"
    echo "build=${OPENFOLD_BUILD_CPUS:-8}/${OPENFOLD_BUILD_MEM:-24G}/${OPENFOLD_BUILD_TIME:-02:00:00}/${OPENFOLD_BUILD_GRES:-}"
    echo "env=$ENV_DIR arch=$arch max_cuda=$MAX_CUDA override_cuda=$CONDA_OVERRIDE_CUDA"
    cat "$VIZFOLD_CONFIG"
}

mkdir -p "$SANDBOX/home"
# setup.sh's defaults, taken from setup.sh so they cannot drift; a file, not <(), for bash 3.2.
sed '/^main() {/,$d' backends/openfold/install/setup.sh > "$SANDBOX/setup-defs.sh"
# Subshell per site, so nothing leaks. Substitute the quoted value: a $REPO of /w would rewrite /work.
actual=$(for f in sites/*.json; do f=${f##*/}; (resolve "${f%.json}" 2>/dev/null); done |
    sed "s#\"$REPO\"#\"{REPO}\"#g; s#\"$SANDBOX/#\"{SANDBOX}/#g")

# One binary reads this file on every cluster, so the key set must not depend on the site.
shapes=$(for f in "$SANDBOX"/*.json; do
    python3 -c 'import json,sys; print(*sorted(json.load(open(sys.argv[1]))))' "$f"
done | sort -u)
if [ "$(printf '%s\n' "$shapes" | wc -l)" -ne 1 ]; then
    echo "FAIL config key set differs by site:"; printf '%s\n' "$shapes"; exit 1
fi

# ...nor on which backend installed last.
cp "$SANDBOX/delta.json" "$SANDBOX/before.json"
(export VIZFOLD_CONFIG=$SANDBOX/delta.json ESMFOLD_ENV_PREFIX=/envs/vizfold-esmfold
 . "$REPO/lib/config.sh" && config::load && config::save) >/dev/null 2>&1
python3 - "$SANDBOX/before.json" "$SANDBOX/delta.json" <<'PY' || exit 1
import json, sys
before, after = (json.load(open(p)) for p in sys.argv[1:3])
lost = {k: v for k, v in before.items() if v and after.get(k) != v}
if lost or set(after) != set(before):
    print("FAIL a second backend's install dropped or changed:", lost or set(before) ^ set(after))
    sys.exit(1)
print("ok   a second backend's install preserves every settled value")
PY

# The snapshot above walks the layers by hand; settle_site is what both installers actually call.
(export USER=x-test HOME=$SANDBOX/home OPENFOLD_ALLOCATION=bbka
 export VIZFOLD_CONFIG=$SANDBOX/settle.json OPENFOLD_HOME=$REPO
 unset OPENFOLD_SITE OPENFOLD_PREFIX OPENFOLD_ACCOUNT OPENFOLD_GPU_ACCOUNT
 . "$REPO/lib/slurm.sh"
 sacctmgr() { echo bbka; }
 slurm::cluster() { echo delta; }
 vizfold::settle_site >/dev/null 2>&1
 got="$OPENFOLD_SITE $OPENFOLD_PREFIX"
 want="delta /work/nvme/bbka/x-test/vizfold"
 if [ "$got" = "$want" ]; then
     echo "ok   settle_site lands a site on the values the snapshot records"
 else
     echo "FAIL settle_site drifted from the snapshot"; echo "  want: $want"; echo "  got:  $got"; exit 1
 fi) || exit 1

# Installing ESMFold first must decide nothing for a later OpenFold install: it used to persist its
# own $HOME fallback, and OpenFold then skipped discovery -- build on a quota'd home, CPU account for GPUs.
(export USER=x-test HOME=$SANDBOX/home OPENFOLD_ALLOCATION=bbka
 export VIZFOLD_CONFIG=$SANDBOX/esmfold-first.json OPENFOLD_HOME=$REPO
 unset OPENFOLD_SITE OPENFOLD_PREFIX OPENFOLD_ACCOUNT OPENFOLD_GPU_ACCOUNT

 # What `vizfold install esmfold` records on a machine with no site (the workstation case).
 . "$REPO/lib/config.sh"
 export ESMFOLD_ENV_PREFIX=$SANDBOX/home/openfold/envs/vizfold-esmfold
 config::save >/dev/null

 python3 - "$SANDBOX/esmfold-first.json" <<'PY' || exit 1
import json, sys
cfg = json.load(open(sys.argv[1]))
if cfg.get("OPENFOLD_PREFIX"):
    print("FAIL the esmfold install persisted a prefix nobody chose:", cfg["OPENFOLD_PREFIX"])
    sys.exit(1)
PY

 # Now OpenFold, on Delta: its own discovery must win.
 (. "$REPO/lib/slurm.sh"
  sacctmgr() { echo bbka; }
  slurm::cluster() { echo delta; }
  vizfold::settle_site >/dev/null 2>&1
  got="$OPENFOLD_PREFIX $OPENFOLD_ACCOUNT $OPENFOLD_GPU_ACCOUNT"
  want="/work/nvme/bbka/x-test/vizfold bbka-delta-cpu bbka-delta-gpu"
  if [ "$got" = "$want" ]; then
      echo "ok   an esmfold-first install leaves openfold's discovery intact"
  else
      echo "FAIL esmfold-first derailed the openfold install"
      echo "  want: $want"; echo "  got:  $got"; exit 1
  fi)) || exit 1

# The first two keys below are set by every <site>.json, so config::fill has a rival value to prefer
# -- a key no site writes would test nothing.
(export OPENFOLD_PARTITION=chosen-partition OPENFOLD_AF2_ROOT=$SANDBOX/chosen-mirror \
        VIZFOLD_DB=$SANDBOX/chosen.db
 resolve delta >/dev/null 2>&1
 python3 - "$SANDBOX/delta.json" "$SANDBOX" <<'PY' || exit 1
import json, sys
cfg, sandbox = json.load(open(sys.argv[1])), sys.argv[2]
want = {
    "OPENFOLD_PARTITION": "chosen-partition",
    "OPENFOLD_AF2_ROOT": f"{sandbox}/chosen-mirror",
    "VIZFOLD_DB": f"{sandbox}/chosen.db",
}
kept = {k: cfg.get(k) for k in want}
if kept != want:
    print("FAIL the install overwrote a value set inline:", kept, "wanted", want)
    sys.exit(1)
print("ok   an inline choice beats the site file that also sets it")
PY
) || exit 1

if [ "${1:-}" = -u ]; then
    printf '%s\n' "$actual" > "$EXPECTED"
    echo "updated $EXPECTED ($(grep -c '^## ' <<<"$actual") sites)"
elif [ "$actual" = "$(cat "$EXPECTED" 2>/dev/null)" ]; then
    echo "ok   $(grep -c '^## ' <<<"$actual") sites resolve unchanged"
else
    echo "FAIL site config drifted:"
    diff "$EXPECTED" <(printf '%s\n' "$actual")
    exit 1
fi
