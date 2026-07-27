#!/bin/bash

# Install the ESMFold backend: no AF2 databases, no CUDA compilation, weights from HuggingFace at run
# time. Solving the env needs no compute node, but the site still decides where. Idempotent.
set -euo pipefail

. "${OPENFOLD_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}/lib/slurm.sh"

esmfold::config() {
    # Cluster, allocation, prefix, saved config. Without it the prefix defaulted to a quota'd $HOME.
    vizfold::settle_site
    ESM=$REPO/backends/esmfold
    PREFIX=$(vizfold::prefix)
    STATE=$(vizfold::state esmfold)
    ENV=${ESMFOLD_ENV_PREFIX:-$(vizfold::env esmfold)}
    export OPENFOLD_DRIVER_CUDA=${OPENFOLD_DRIVER_CUDA:-$(vizfold::driver_cuda)}
    # Its own state dir: else pip's wheels land in the ~/.cache/pip quota this install avoids.
    export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$STATE/pip}
    test -f "$ESM/environment.yml" || die "no esmfold project at $ESM; is $REPO a vizfold checkout?"
    echo "driver CUDA ${OPENFOLD_DRIVER_CUDA:-unknown}"
}

# Where a driver exists the env must carry a torch built for a CUDA major it can load -- a newer one
# imports and then cannot reach the GPU, a CPU-only one never tries. Without this a re-install would
# keep the very build it exists to replace.
esmfold::present() {
    local driver=${OPENFOLD_DRIVER_CUDA:-}; driver=${driver%%.*}
    "$ENV/bin/python" -c "
import torch, transformers, esmfold
assert not ${driver:-0} or 0 < int((torch.version.cuda or '0').split('.')[0]) <= ${driver:-0}" 2>/dev/null
}

# What a GPU driver justifies asking for, and nothing where there is none: the solver then picks the
# CPU (or on a Mac, MPS) build the host can run. Two specs the driver alone does not get us -- the
# unqualified pytorch resolves to a CPU build even where a GPU is present, and __cuda bounds only the
# CUDA major while the driver is the real ceiling.
esmfold::cuda_specs() {
    [ -n "${OPENFOLD_DRIVER_CUDA:-}" ] || return 0
    echo "pytorch=*=cuda* cuda-version<=$OPENFOLD_DRIVER_CUDA"
}

esmfold::env() {
    log "env $ENV"
    rm -rf "$ENV"   # clear a partial env; create fails on a non-empty dir
    export MAMBA_ROOT_PREFIX=$PREFIX/mamba
    mkdir -p "$(dirname "$ENV")"
    # A driver the config knows but this node cannot see (detected on a compute node) still has to
    # reach the solver, or the CUDA specs below have no __cuda to satisfy them.
    [ -z "${OPENFOLD_DRIVER_CUDA:-}" ] || export CONDA_OVERRIDE_CUDA=$OPENFOLD_DRIVER_CUDA
    # Unquoted on purpose: cuda_specs emits two specs or none. --no-rc so a user ~/.condarc
    # envs_dirs/channels cannot redirect it.
    micromamba create -y --no-rc -p "$ENV" -f "$ESM/environment.yml" $(esmfold::cuda_specs)
}

# The env resolved torch and transformers already; this adds the `esmfold` package and its entrypoint.
esmfold::install() {
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
# torch.version.cuda, not just is_available(): a login node carries the driver but no GPU, so the
# build it resolved to is the only thing this can actually check.
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
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
