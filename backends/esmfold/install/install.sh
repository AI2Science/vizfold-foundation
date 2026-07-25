#!/bin/bash

# Install the ESMFold backend into an environment that can fold on its own: no AF2 databases and no
# CUDA build, weights come from HuggingFace at run time. The build runs here rather than on a
# compute node -- a pip install needs neither -- but the site still decides *where*, so this asks
# the platform the same question `vizfold install openfold` does.
# Invoked by `vizfold install esmfold`. Idempotent.
set -euo pipefail

# lib/ is the backend-neutral shared install machinery, owned by no backend; slurm.sh pulls in
# config.sh and interactive.sh. OPENFOLD_HOME is exported by `vizfold install`; the fallback finds
# lib/ from here.
LIB=${OPENFOLD_HOME:+$OPENFOLD_HOME/lib}
. "${LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../lib" && pwd)}/slurm.sh"

# Cluster, allocation, install prefix, then the saved config under all of it. Without this the
# prefix defaulted to $HOME/openfold -- which put a multi-GB torch environment on a quota'd home
# and, once persisted, made every later OpenFold install skip its own discovery.
vizfold::settle_site

ESM=$REPO/backends/esmfold
PREFIX=$(vizfold::prefix)
ENV=${ESMFOLD_ENV_PREFIX:-$(vizfold::env esmfold)}
# A cluster login node's python3 is routinely older than the package needs, so we bring our own.
PYTHON_VERSION=3.11
# Beside the prefix, like setup.sh's caches. MAMBA_ROOT_PREFIX already parks the conda packages
# there; without this the 2.5-3 GB torch wheel still lands in ~/.cache/pip, which is exactly the
# quota this install moved off $HOME to avoid. Its own name, not .openfold-pip: `vizfold uninstall
# esmfold` must not delete the other backend's cache.
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PREFIX/../.esmfold-pip}
test -f "$ESM/pyproject.toml" || die "no esmfold project at $ESM; is $REPO a vizfold checkout?"

esmfold::present() { "$ENV/bin/python" -c 'import torch, transformers, esmfold' 2>/dev/null; }

esmfold::env() {
    log "env $ENV (python $PYTHON_VERSION)"
    local mm; mm=$(mamba::ensure "$PREFIX")
    export MAMBA_ROOT_PREFIX=$PREFIX/mamba
    mkdir -p "$(dirname "$ENV")"
    # --no-rc so a user ~/.condarc envs_dirs/channels cannot redirect it.
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

# Not merely that pip exited zero: every import the backend makes, and the entrypoint it runs.
esmfold::verify() {
    log verify
    "$ENV/bin/python" - <<'PY'
import shutil, sys, torch, transformers
from esmfold.cli import main
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
entrypoint = shutil.which("esmfold", path=f"{sys.prefix}/bin")
assert entrypoint, "the esmfold entrypoint is not in the environment"
print("entrypoint", entrypoint)
PY
}

# Record what was resolved so `vizfold status` and the DB commands see this install. Only what
# something actually settled: OPENFOLD_PREFIX is written by vizfold::settle_site when a site or the
# caller chose one, and recording this backend's private fallback instead would hand every later
# install a prefix nobody picked. ESMFOLD_ENV_PREFIX is absolute, so the env is findable regardless.
esmfold::config_save() {
    log config
    export OPENFOLD_HOME=$REPO ESMFOLD_ENV_PREFIX=$ENV
    export VIZFOLD_DB=${VIZFOLD_DB:-${OPENFOLD_PREFIX:+$OPENFOLD_PREFIX/vizfold.db}}
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

Check it works -- fold the bundled example, onto a GPU node if one is configured
(downloads facebook/esmfold_v1 on first run):

  vizfold fold ${OPENFOLD_EXAMPLE:-6KWC_1} --backend esmfold

To drive the model yourself, the environment installs its own CLI:

  $ENV/bin/esmfold --help
EOF
}
main "$@"
