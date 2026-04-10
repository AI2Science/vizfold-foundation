#!/usr/bin/env bash
set -euo pipefail

# Ensure module/conda work even in shells where `module` isn't preloaded
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi

module load anaconda3 >/dev/null 2>&1 || true
source "$(conda info --base)/etc/profile.d/conda.sh"

SCR="${SCR:-/storage/ice1/2/0/$USER}"
ENV="${ENV:-$SCR/conda/envs/boltz_clean}"
mkdir -p "$(dirname "$ENV")" "$SCR/pip-cache"

echo "[INFO] SCR=$SCR"
echo "[INFO] ENV=$ENV"

# Create OR update env from YAML (idempotent)
if [ -d "$ENV" ]; then
  echo "[INFO] Env exists; updating: $ENV"
  conda env update -p "$ENV" -f environment_boltz.yml --prune
else
  echo "[INFO] Creating env: $ENV"
  conda env create -p "$ENV" -f environment_boltz.yml
fi

# Activate safely (avoid activate.d 'set -u' issues)
set +u
conda activate "$ENV"

# Ensure pip installs never go to ~/.local (and never outside this conda env)
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_CACHE_DIR="$SCR/pip-cache"
mkdir -p "$PIP_CACHE_DIR"

echo "[INFO] which python: $(which python)"
echo "[INFO] python: $(python -V)"

EXTRAS="scripts/boltz/boltz_pip_extras.txt"
if [ ! -f "$EXTRAS" ]; then
  echo "[ERROR] Missing $EXTRAS" >&2
  echo "[HINT] Create it and commit it." >&2
  exit 2
fi

# Keep pip modern (helps wheels)
python -m pip install -U pip

# If an old env had conda rdkit, remove it so pip rdkit can be imported
conda remove -y rdkit >/dev/null 2>&1 || true

# Install pip-only deps first (no dependency solving)
python -m pip install --no-deps -r "$EXTRAS"

# Install boltz itself without letting pip resolve/replace dependencies
python -m pip install --no-deps boltz==2.2.1

# Re-enable strict undefined vars after installs
set -u

# ---- Smoke checks (robust RDKit version check) ----
python - <<'PY'
import sys
from rdkit import rdBase

print("python:", sys.executable)
print("rdkit:", rdBase.rdkitVersion)

# Hard guard: on ICE we rely on pip RDKit matching boltz canonicals.
# If you update rdkit in boltz_pip_extras.txt, update this string too.
assert rdBase.rdkitVersion.startswith("2025.09.6"), f"Unexpected RDKit: {rdBase.rdkitVersion}"
PY

python - <<'PY'
import torch, boltz, pytorch_lightning as pl, numpy as np
import ihm, frozendict, msgpack
from rdkit import rdBase

print("torch", torch.__version__, "cuda?", torch.cuda.is_available(), "torch.version.cuda", torch.version.cuda)
print("numpy", np.__version__)
print("pl", pl.__version__)
print("boltz", boltz.__file__)
print("rdkit", rdBase.rdkitVersion)
print("ihm", ihm.__version__)
print("frozendict", frozendict.__version__)
print("msgpack", msgpack.__version__)
PY

command -v boltz >/dev/null 2>&1 || { echo "[ERROR] boltz CLI not found in PATH after install" >&2; exit 2; }
echo "[INFO] boltz CLI: $(command -v boltz)"

echo "[OK] Environment ready."
echo "[NEXT] Run: sbatch --export=ALL scripts/boltz/run_boltz_trace.sbatch"