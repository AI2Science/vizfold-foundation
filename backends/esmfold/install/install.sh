#!/bin/bash

# Install the ESMFold backend: no AF2 databases, no CUDA build, weights from HuggingFace at run time.
# A pip install needs no compute node, but the site still decides where. Idempotent.
set -euo pipefail

. "${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/lib/slurm.sh"

# Cluster, allocation, prefix, saved config. Without it the prefix defaulted to a quota'd $HOME.
vizfold::settle_site

ESM=$REPO/backends/esmfold
PREFIX=$(vizfold::prefix)
STATE=$(vizfold::state esmfold)
ENV=${ESMFOLD_ENV_PREFIX:-$(vizfold::env esmfold)}
# A cluster login node's python3 is routinely older than the package needs, so we bring our own.
PYTHON_VERSION=3.11
# Its own state dir: else the 2.5-3 GB torch wheel lands in the ~/.cache/pip quota this install avoids.
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STATE/pip}
test -f "$ESM/pyproject.toml" || die "no esmfold project at $ESM; is $REPO a vizfold checkout?"

esmfold::present() { "$ENV/bin/python" -c 'import torch, transformers, esmfold' 2>/dev/null; }

esmfold::env() {
    log "env $ENV (python $PYTHON_VERSION)"
    rm -rf "$ENV"   # clear a partial env; create fails on a non-empty dir
    export MAMBA_ROOT_PREFIX=$PREFIX/mamba
    mkdir -p "$(dirname "$ENV")"
    # --no-rc so a user ~/.condarc envs_dirs/channels cannot redirect it.
    micromamba create -y --no-rc -p "$ENV" -c conda-forge "python=$PYTHON_VERSION" pip
}

esmfold::install() {
    # torch first, off its own index when set: pyproject asks only for torch>=2.1, so this build stands.
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

# Only what something settled: recording this backend's own fallback would pin a prefix nobody picked.
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

  vizfold run ${OPENFOLD_EXAMPLE:-6KWC_1} --backend esmfold

To drive the model yourself, use its own CLI:

  micromamba run -p $ENV esmfold --help
EOF
}
main "$@"
