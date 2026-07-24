#!/bin/bash

# Install the ESMFold backend: a plain venv with PyTorch + Transformers. No SLURM/site machinery
# and no AF2 databases -- ESMFold needs no CUDA build and pulls its weights from HuggingFace at
# run time. Invoked by `vizfold install esmfold`. Idempotent: skips the pip work if the env
# already imports torch + transformers.
set -euo pipefail

# OPENFOLD_HOME is exported by `vizfold install`; config.sh also fills unset vars from an existing
# ~/.config/vizfold/vizfold.json (so an openfold install's PREFIX etc. carry over here). config.sh
# is shared install infrastructure that lives with the openfold backend.
CFG=${OPENFOLD_HOME:+$OPENFOLD_HOME/backends/openfold/install/config.sh}
. "${CFG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../openfold/install" && pwd)/config.sh}"

log() { echo "== $* (+$((SECONDS))s)"; }

REPO=${OPENFOLD_HOME:-$REPO}
ESM=$REPO/backends/esmfold
PREFIX=${OPENFOLD_PREFIX:-$HOME/openfold}
ENV=${ESMFOLD_ENV_PREFIX:-$PREFIX/esmfold-venv}
test -f "$ESM/pyproject.toml" || die "no esmfold project at $ESM; is $REPO a vizfold checkout?"
command -v python3 >/dev/null || die "python3 is required to create the ESMFold venv"

esmfold::present() { "$ENV/bin/python" -c 'import torch, transformers, esmfold' 2>/dev/null; }

esmfold::install() {
    log "venv $ENV"
    mkdir -p "$(dirname "$ENV")"
    [ -x "$ENV/bin/python" ] || python3 -m venv "$ENV"
    "$ENV/bin/python" -m pip install --upgrade pip
    # torch first (own wheel index, e.g. a CUDA build); transformers from PyPI reads requirements.
    log torch
    local index=()
    [ -n "${ESMFOLD_PIP_INDEX_URL:-}" ] && index=(--index-url "$ESMFOLD_PIP_INDEX_URL")
    "$ENV/bin/pip" install ${index[@]+"${index[@]}"} "${ESMFOLD_TORCH_SPEC:-torch}"
    # the esmfold project pulls its deps (transformers) from pyproject and installs the `esmfold`
    # package itself, so `import esmfold` resolves in the venv the run entrypoint executes under.
    log package
    "$ENV/bin/pip" install "$ESM"
}

esmfold::verify() {
    log verify
    "$ENV/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
}

# Record what was resolved so `vizfold status` and the DB commands see this install.
esmfold::config_save() {
    log config
    export OPENFOLD_HOME=$REPO OPENFOLD_PREFIX=$PREFIX ESMFOLD_ENV_PREFIX=$ENV
    export VIZFOLD_DB=${VIZFOLD_DB:-$PREFIX/vizfold.db}
    config::save OPENFOLD_HOME OPENFOLD_PREFIX ESMFOLD_ENV_PREFIX VIZFOLD_DB
}

main() {
    if esmfold::present; then
        log "already installed at $ENV"
    else
        esmfold::install
    fi
    esmfold::verify
    esmfold::config_save
    cat <<EOF
== ready (+$((SECONDS))s)

ESMFold env: $ENV

Fold the bundled example (downloads facebook/esmfold_v1 on first run):

  $ENV/bin/python $ESM/run_pretrained_esmf.py \\
    --fasta $REPO/examples/monomer/fasta_dir_6KWC/6KWC.fasta \\
    --out $PREFIX/outputs/esmf_6KWC --trace_mode none
EOF
}
main "$@"
