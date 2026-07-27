#!/bin/bash

# Install the ESMFold backend: no AF2 databases, no CUDA build, weights from HuggingFace at run time.
# A pip install needs no compute node, but the site still decides where. Idempotent.
set -euo pipefail

. "${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/lib/slurm.sh"

# A cluster login node's python3 is routinely older than the package needs, so we bring our own.
PYTHON_VERSION=3.11

esmfold::config() {
    # Cluster, allocation, prefix, saved config. Without it the prefix defaulted to a quota'd $HOME.
    vizfold::settle_site
    ESM=$REPO/backends/esmfold
    PREFIX=$(vizfold::prefix)
    STATE=$(vizfold::state esmfold)
    ENV=${ESMFOLD_ENV_PREFIX:-$(vizfold::env esmfold)}
    # Its own state dir: else the 2.5-3 GB torch wheel lands in the ~/.cache/pip quota this install avoids.
    export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STATE/pip}
    test -f "$ESM/pyproject.toml" || die "no esmfold project at $ESM; is $REPO a vizfold checkout?"
}

# PyPI's torch is built against the newest CUDA, and an older driver refuses that build outright
# ("the NVIDIA driver on your system is too old"), leaving a GPU fold with no GPU. Pick the newest
# wheel index the driver accepts -- the pip analogue of the env's `cuda-version<=$MAX_CUDA`. The list
# is the cu* indexes download.pytorch.org publishes; extend it as more appear.
esmfold::torch_cuda() {
    export OPENFOLD_DRIVER_CUDA=${OPENFOLD_DRIVER_CUDA:-$(vizfold::driver_cuda)}
    local tag
    TORCH_CUDA=0
    for tag in 118 121 124 126 128 129 130 132; do
        if [ -n "$OPENFOLD_DRIVER_CUDA" ] && [ "$tag" -le "${OPENFOLD_DRIVER_CUDA//./}" ]; then
            TORCH_CUDA=$tag
        fi
    done
    # Assignment, not :-, so ESMFOLD_PIP_INDEX_URL= opts out of the pin entirely.
    [ "$TORCH_CUDA" = 0 ] ||
        ESMFOLD_PIP_INDEX_URL=${ESMFOLD_PIP_INDEX_URL-https://download.pytorch.org/whl/cu$TORCH_CUDA}
    [ -n "${ESMFOLD_PIP_INDEX_URL:-}" ] || TORCH_CUDA=0
    echo "driver CUDA ${OPENFOLD_DRIVER_CUDA:-unknown}, torch index ${ESMFOLD_PIP_INDEX_URL:-PyPI default}"
}

# A torch built for a newer CUDA than the driver is not "present": it imports, then cannot reach the GPU.
esmfold::present() {
    "$ENV/bin/python" -c "
import torch, transformers, esmfold
assert not $TORCH_CUDA or int((torch.version.cuda or '0').replace('.', '')) <= $TORCH_CUDA" 2>/dev/null
}

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
    # --force-reinstall: pip would otherwise call an already-installed torch satisfied and keep the
    # build the driver rejects, so a re-install could not repair the very thing it exists to fix.
    "$ENV/bin/pip" install --force-reinstall ${index[@]+"${index[@]}"} "${ESMFOLD_TORCH_SPEC:-torch}"
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
    esmfold::config
    esmfold::torch_cuda
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

# Sourced (tests/torch_index.sh) this file is just its definitions; only an execution installs.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
main "$@"
