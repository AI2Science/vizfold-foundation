#!/bin/bash

# Install the ESMFold backend: no AF2 databases, no CUDA compilation, weights from HuggingFace at run
# time -- so the env solves on any node. Idempotent.
set -euo pipefail

. "${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/lib/slurm.sh"

esmfold::config() {
    # Without settle_site the prefix falls back to a quota'd $HOME.
    vizfold::settle_site
    ESM=$REPO/backends/esmfold
    PREFIX=$(vizfold::prefix)
    STATE=$(vizfold::state esmfold)
    ENV=${ESMFOLD_ENV_PREFIX:-$(vizfold::env esmfold)}
    export OPENFOLD_DRIVER_CUDA=${OPENFOLD_DRIVER_CUDA:-$(vizfold::driver_cuda)}
    # Its own state dir: else wheels land in the quota'd ~/.cache/pip.
    export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STATE/pip}
    test -f "$ESM/environment.yml" || die "no esmfold project at $ESM; is $REPO a vizfold checkout?"
    echo "driver CUDA ${OPENFOLD_DRIVER_CUDA:-unknown}"
}

# Where a driver exists the env must carry a torch built for a CUDA major it can load -- a newer one
# imports but cannot reach the GPU, a CPU-only one never tries. Without this a re-install keeps the wrong build.
esmfold::present() {
    local driver=${OPENFOLD_DRIVER_CUDA:-}; driver=${driver%%.*}
    "$ENV/bin/python" -c "
import torch, transformers, esmfold
assert not ${driver:-0} or 0 < int((torch.version.cuda or '0').split('.')[0]) <= ${driver:-0}" 2>/dev/null
}

# No driver -> no specs, and the solver picks the CPU (or on a Mac, MPS) build. Both specs are needed:
# unqualified pytorch resolves to a CPU build even with a GPU, and __cuda bounds only the CUDA major.
esmfold::cuda_specs() {
    [ -n "${OPENFOLD_DRIVER_CUDA:-}" ] || return 0
    echo "pytorch=*=cuda* cuda-version<=$OPENFOLD_DRIVER_CUDA"
}

esmfold::env() {
    log "env $ENV"
    rm -rf "$ENV"   # clear a partial env; create fails on a non-empty dir
    export MAMBA_ROOT_PREFIX=$PREFIX/mamba
    mkdir -p "$(dirname "$ENV")"
    # A driver the config knows but this node cannot see must still reach the solver, or the specs below have no __cuda.
    [ -z "${OPENFOLD_DRIVER_CUDA:-}" ] || export CONDA_OVERRIDE_CUDA=$OPENFOLD_DRIVER_CUDA
    # Unquoted on purpose: cuda_specs emits two specs or none. --no-rc so a user ~/.condarc cannot redirect it.
    micromamba create -y --no-rc -p "$ENV" -f "$ESM/environment.yml" $(esmfold::cuda_specs)
}

esmfold::install() {
    log package
    "$ENV/bin/pip" install "$ESM"
}

esmfold::verify() {
    log verify
    "$ENV/bin/python" - <<'PY'
import shutil, sys, torch, transformers
from esmfold.cli import main
print("python", sys.version.split()[0])
# torch.version.cuda, not just is_available(): a login node has the driver but no GPU, so the resolved build is all we can check.
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
entrypoint = shutil.which("esmfold", path=f"{sys.prefix}/bin")
assert entrypoint, "the esmfold entrypoint is not in the environment"
print("entrypoint", entrypoint)
PY
}

# Only what something settled: recording our own fallback would pin a prefix nobody picked.
esmfold::config_save() {
    log config
    export OPENFOLD_HOME=$REPO ESMFOLD_ENV_PREFIX=$ENV
    export VIZFOLD_DB=${VIZFOLD_DB:-${OPENFOLD_PREFIX:+$OPENFOLD_PREFIX/vizfold.db}}
    config::save
}

main() {
    esmfold::config
    if esmfold::present; then
        log "already installed at $ENV"
    else
        esmfold::env      # no resume: micromamba re-links from its own package cache, so this is cheap
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

# Sourced (tests/esmfold_env.sh) this file is just its definitions; only an execution installs.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
main "$@"
