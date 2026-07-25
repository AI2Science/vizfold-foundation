#!/bin/bash

# Install the ESMFold backend into an environment that can fold on its own: its own Python, PyTorch,
# Transformers, and the `esmfold` package with its `vizfold-esmfold` entrypoint. No SLURM/site
# machinery and no AF2 databases -- ESMFold needs no CUDA build and pulls its weights from
# HuggingFace at run time. Invoked by `vizfold install esmfold`. Idempotent: skips the pip work if
# the environment already imports what it needs.
set -euo pipefail

# config.sh is the backend-neutral shared install lib (repo-root lib/), owned by no backend. It
# fills unset vars from ~/.config/vizfold/vizfold.json (so an openfold install's PREFIX etc. carry
# over here). OPENFOLD_HOME is exported by `vizfold install`; the fallback finds lib/ from here.
CFG=${OPENFOLD_HOME:+$OPENFOLD_HOME/lib/config.sh}
. "${CFG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../lib" && pwd)/config.sh}"

log() { echo "== $* (+$((SECONDS))s)"; }

REPO=${OPENFOLD_HOME:-$REPO}
ESM=$REPO/backends/esmfold
PREFIX=${OPENFOLD_PREFIX:-$HOME/openfold}
ENV=${ESMFOLD_ENV_PREFIX:-$(vizfold::env esmfold)}
# The package needs >=3.10, which a cluster login node's python3 routinely is not -- so the
# environment brings its own rather than inheriting whichever one happens to be on PATH.
PYTHON_VERSION=3.11
test -f "$ESM/pyproject.toml" || die "no esmfold project at $ESM; is $REPO a vizfold checkout?"

esmfold::present() { "$ENV/bin/python" -c 'import torch, transformers, esmfold' 2>/dev/null; }

esmfold::env() {
    log "env $ENV (python $PYTHON_VERSION)"
    local mm; mm=$(mamba::ensure "$PREFIX")
    export MAMBA_ROOT_PREFIX=$PREFIX/mamba
    mkdir -p "$(dirname "$ENV")"
    # --no-rc so a user ~/.condarc envs_dirs/channels cannot redirect it, as the OpenFold install does.
    "$mm" create -y --no-rc -p "$ENV" -c conda-forge "python=$PYTHON_VERSION" pip
}

esmfold::install() {
    # torch first, from its own wheel index when one is set (a CUDA build): pyproject asks only for
    # torch>=2.1, so the package install below finds it satisfied and leaves this build alone.
    log torch
    local index=()
    [ -n "${ESMFOLD_PIP_INDEX_URL:-}" ] && index=(--index-url "$ESMFOLD_PIP_INDEX_URL")
    "$ENV/bin/pip" install ${index[@]+"${index[@]}"} "${ESMFOLD_TORCH_SPEC:-torch}"
    # The project declares the rest and installs the `esmfold` package with its entrypoint.
    log package
    "$ENV/bin/pip" install "$ESM"
}

# Prove the environment can fold on its own: its own interpreter, every import the backend makes,
# and the entrypoint on its PATH -- not merely that the pip commands exited zero.
esmfold::verify() {
    log verify
    "$ENV/bin/python" - <<'PY'
import shutil, sys, torch, transformers
from esmfold.cli import main
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
entrypoint = shutil.which("vizfold-esmfold", path=f"{sys.prefix}/bin")
assert entrypoint, "vizfold-esmfold is not in the environment"
print("entrypoint", entrypoint)
PY
}

# Record what was resolved so `vizfold status` and the DB commands see this install.
esmfold::config_save() {
    log config
    export OPENFOLD_HOME=$REPO OPENFOLD_PREFIX=$PREFIX ESMFOLD_ENV_PREFIX=$ENV
    export VIZFOLD_ENV_BASE=$(vizfold::env_base)
    export VIZFOLD_DB=${VIZFOLD_DB:-$PREFIX/vizfold.db}
    config::save
}

main() {
    if esmfold::present; then
        log "already installed at $ENV"
    else
        [ -x "$ENV/bin/pip" ] || esmfold::env
        esmfold::install
    fi
    esmfold::verify
    esmfold::config_save
    cat <<EOF
== ready (+$((SECONDS))s)

ESMFold env: $ENV

Fold the bundled example (downloads facebook/esmfold_v1 on first run):

  $ENV/bin/vizfold-esmfold \\
    --fasta $REPO/examples/monomer/fasta_dir_6KWC/6KWC.fasta \\
    --out $PREFIX/outputs/esmf_6KWC --trace_mode none
EOF
}
main "$@"
