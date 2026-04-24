#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import sys
import shutil
import time


def run_bash(script: str) -> None:
    """Run a bash script, streaming output live and raising on failure."""
    # Note: -u (nounset) is intentionally omitted because conda activation
    # scripts (e.g. libblas_mkl_activate.sh) reference unbound variables.
    result = subprocess.run(["bash", "-eo", "pipefail", "-c", script],
                            env=os.environ.copy())
    if result.returncode != 0:
        sys.exit(result.returncode)


def step(name: str) -> None:
    """Print a timestamped section header."""
    timestamp = time.strftime("%H:%M:%S")
    banner = f"\n{'='*60}\n[{timestamp}]  {name}\n{'='*60}"
    print(banner)


def _check_storage_hint() -> None:
    """Print a storage hint if disk space is low (called after any failure)."""
    disk = shutil.disk_usage(os.environ['ROOT_DIR'])
    free_gb = disk.free / (1024 ** 3)
    if free_gb < 50:
        print(f"\n[HINT] Only {free_gb:.1f} GB free in {os.environ['ROOT_DIR']}. "
              "Low disk space may have contributed to this failure. "
              "Try freeing space or pointing ROOT_DIR / CONDA_INSTALL_DIR "
              "to a partition with more room.")


# ---------------------------------------------------------------------------
# Setting Up the Directory
# ---------------------------------------------------------------------------

# Root directory for cloned repositories and the conda environment
os.environ['ROOT_DIR'] = '~/scratch'

# Path to AlphaFold data (pre-downloaded on PACE ICE)
os.environ['DATA_DIR'] = '/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data'

# Directory where the conda environment will be stored
os.environ['CONDA_INSTALL_DIR'] = '~/scratch'

# ---------------------------------------------------------------------------
# Fix paths to be expanded from user to exact paths
# ---------------------------------------------------------------------------

# Resolve ~ to the absolute home directory path for use in bash cells
os.environ['ROOT_DIR'] = os.path.expanduser(os.environ['ROOT_DIR'])
os.environ['DATA_DIR'] = os.path.expanduser(os.environ['DATA_DIR'])
os.environ['CONDA_INSTALL_DIR'] = os.path.expanduser(os.environ['CONDA_INSTALL_DIR'])

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

step("Pre-flight checks")

errors = []

# Check conda is available (hard requirement)
if shutil.which("conda") is None:
    errors.append("conda is not available. Load a conda module first (e.g. 'module load anaconda3').")
else:
    print("[OK] conda found")

# Check GPU is available via nvidia-smi (hard requirement)
gpu_result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                             "--format=csv,noheader"],
                            capture_output=True, text=True)
if gpu_result.returncode != 0:
    errors.append("No NVIDIA GPU detected. Run this on a compute node with GPU access.")
else:
    for i, line in enumerate(gpu_result.stdout.strip().splitlines()):
        print(f"[OK] GPU {i}: {line.strip()}")

# Check disk space — warn only, do not abort
root_dir = os.environ['ROOT_DIR']
os.makedirs(root_dir, exist_ok=True)
disk = shutil.disk_usage(root_dir)
free_gb = disk.free / (1024 ** 3)
if free_gb < 50:
    print(f"[WARN] Only {free_gb:.1f} GB free in {root_dir}. "
          "At least 50 GB recommended — proceeding anyway.")
else:
    print(f"[OK] {free_gb:.0f} GB free in {root_dir}")

# Check AlphaFold data directory exists (hard requirement)
data_dir = os.environ['DATA_DIR']
if not os.path.isdir(data_dir):
    errors.append(f"AlphaFold data directory not found: {data_dir}")
else:
    print(f"[OK] AlphaFold data found at {data_dir}")

if errors:
    print("\n*** Pre-flight checks FAILED ***")
    for e in errors:
        print(f"  ✗ {e}")
    sys.exit(1)

print("\nAll pre-flight checks passed.\n")


# ---------------------------------------------------------------------------
# Installation steps — storage hint is printed on any failure
# ---------------------------------------------------------------------------

try:

    # -----------------------------------------------------------------------
    # Clone the Vizfold-Foundation, and OpenFold repository.
    # -----------------------------------------------------------------------

    step("Cloning repositories")

    run_bash("""
    # Clone VizFold-Foundation and OpenFold into the root directory (skip if already cloned)
    if [ ! -d "$ROOT_DIR/vizfold-foundation/.git" ]; then
        git clone https://github.com/AI2Science/vizfold-foundation.git $ROOT_DIR/vizfold-foundation
    else
        echo "vizfold-foundation already cloned — skipping"
    fi

    if [ ! -d "$ROOT_DIR/openfold/.git" ]; then
        git clone https://github.com/aqlaboratory/openfold.git $ROOT_DIR/openfold
    else
        echo "openfold already cloned — skipping"
    fi
    """)

    # -----------------------------------------------------------------------
    # Overwrite environment.yml with a PACE ICE–compatible version
    # -----------------------------------------------------------------------

    step("Writing environment.yml")

    run_bash("""
    # Overwrite the upstream OpenFold environment.yml with a version that is
    # compatible with PACE ICE (CUDA 12.4, system GCC, no conda-bundled pip packages).
    cd $ROOT_DIR/openfold
    cat > environment.yml << 'EOF'
name: openfold_env
channels:
  - pytorch
  - nvidia
  - conda-forge
  - bioconda
dependencies:
  - python=3.10
  - pytorch=2.5.1
  - pytorch-cuda=12.4
  # Build tools — use conda cross-compiler; system GCC is used at build time
  - gxx_linux-64
  - gcc_linux-64
  - libstdcxx-ng
  - sysroot_linux-64=2.17
  - make
  - ninja
  # CUDA development headers and libraries
  - nvidia::cuda-nvcc=12.4
  - nvidia::cuda-libraries-dev=12.4
  - nvidia::cuda-cudart-dev=12.4
  - nvidia::cuda-driver-dev=12.4
  # Core scientific stack
  - numpy
  - pandas
  - scipy
  - tqdm
  - pyyaml
  - requests
  - typing-extensions
  # ML / framework dependencies
  - wandb
  - pytorch-lightning
  # Structural biology tools
  - openmm
  - pdbfixer
  - biopython
  - modelcif==0.7
  # Sequence search databases and tools
  - bioconda::hmmer
  - bioconda::hhsuite
  - bioconda::kalign2
  # Utilities
  - awscli
  - ml-collections
  - aria2
  - git
  - pip
  - packaging
EOF
    echo "environment.yml written successfully"
    """)

    # -----------------------------------------------------------------------
    # Create and activate the OpenFold conda environment
    # -----------------------------------------------------------------------

    step("Creating conda environment")

    run_bash("""
    cd $ROOT_DIR/openfold
    export PYTHONNOUSERSITE=1
    export MAX_JOBS=4
    source "$(conda info --base)/etc/profile.d/conda.sh"
    export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
    export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs

    # Remove any previous (possibly partial) environment
    echo "Removing old environment if it exists..."
    conda env remove -n openfold_env -y || true

    # Create the environment (use libmamba solver for faster dependency resolution)
    echo "Creating conda environment..."
    conda env create --solver=libmamba -f environment.yml --force

    # Install pip-only packages after the conda environment is ready.
    conda activate openfold_env
    echo "Installing pip dependencies..."
    pip install deepspeed==0.14.5 dm-tree==0.1.6 git+https://github.com/NVIDIA/dllogger.git
    pip install flash-attn --no-build-isolation --no-cache-dir

    echo "openfold_env created successfully"
    """)

    # -----------------------------------------------------------------------
    # Set up compiler and library paths, then build OpenFold
    # -----------------------------------------------------------------------

    step("Building OpenFold")

    run_bash("""
    source "$(conda info --base)/etc/profile.d/conda.sh"
    export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
    export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs
    conda activate openfold_env
    cd $ROOT_DIR/openfold

    # Use the PACE ICE system GCC 12.3.0 instead of the conda-bundled compiler.
    # The conda GCC toolchain causes linker issues when building CUDA extensions on RHEL 9.
    export CC=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/bin/gcc
    export CXX=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/bin/g++

    # Point the build at the conda CUDA 12.4 toolkit, not the system CUDA 13.0.
    export CUDA_HOME=$CONDA_PREFIX
    export PATH=$CONDA_PREFIX/bin:$PATH

    export CFLAGS="-I${CONDA_PREFIX}/include"
    export CXXFLAGS="-I${CONDA_PREFIX}/include"
    export LDFLAGS="-L${CONDA_PREFIX}/lib"
    export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH}
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
    export PYTHONNOUSERSITE=1
    export MAX_JOBS=4

    echo "Cleaning previous build artifacts..."
    rm -rf build/ openfold.egg-info/ dist/

    echo "Building OpenFold with system GCC..."
    pip install -e . --no-build-isolation

    echo "Installing third-party dependencies..."
    bash scripts/install_third_party_dependencies.sh

    echo "OpenFold installation complete"
    """)

    # -----------------------------------------------------------------------
    # Set up Vizfold-Foundation
    # -----------------------------------------------------------------------

    step("Setting up VizFold-Foundation")

    run_bash("""
    source "$(conda info --base)/etc/profile.d/conda.sh"
    export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
    export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs
    conda activate openfold_env

    cd $ROOT_DIR/vizfold-foundation/openfold
    mkdir -p resources

    # Symlink to the shared AlphaFold parameters on PACE ICE
    ln -sfn $DATA_DIR/params resources/params

    # Download stereo chemical properties file (required by OpenFold relaxation)
    if [ ! -f "resources/stereo_chemical_props.txt" ]; then
        wget -q --no-check-certificate -P resources \\
            https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
    fi

    echo "VizFold-Foundation setup complete"
    """)

    # -----------------------------------------------------------------------
    # Install additional visualization tools
    # -----------------------------------------------------------------------

    step("Installing visualization tools")

    run_bash("""
    source "$(conda info --base)/etc/profile.d/conda.sh"
    export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
    export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs
    conda activate openfold_env

    # Matplotlib for plotting attention maps
    conda install --solver=libmamba -y conda-forge::matplotlib

    # PyMOL for 3D molecular structure visualization
    conda install --solver=libmamba -y -c conda-forge pymol-open-source

    echo "Visualization tools installed successfully"
    """)

    # -----------------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------------

    step("Verifying installation")

    run_bash("""
    source "$(conda info --base)/etc/profile.d/conda.sh"
    export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
    export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs
    conda activate openfold_env

    echo "--- Python & PyTorch ---"
    python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
"

    echo ""
    echo "--- OpenFold ---"
    python -c "import openfold; print('  openfold imported successfully')"

    echo ""
    echo "--- Visualization tools ---"
    # Force a non-interactive backend; Jupyter sets MPLBACKEND=matplotlib_inline
    # which is invalid outside of an ipykernel process.
    export MPLBACKEND=Agg
    python -c "
import matplotlib; print(f'  matplotlib {matplotlib.__version__}')
import pymol; print(f'  pymol imported successfully')
"

    echo ""
    echo "--- Flash Attention ---"
    python -c "import flash_attn; print(f'  flash_attn {flash_attn.__version__}')"

    echo ""
    echo "--- Key paths ---"
    echo "  Params symlink: $(readlink -f $ROOT_DIR/vizfold-foundation/openfold/resources/params)"
    echo "  Stereo props:   $ROOT_DIR/vizfold-foundation/openfold/resources/stereo_chemical_props.txt"
    ls $ROOT_DIR/vizfold-foundation/openfold/resources/stereo_chemical_props.txt > /dev/null 2>&1 \\
        && echo "    [OK] exists" || echo "    [MISSING]"

    echo ""
    echo "============================================"
    echo "  Installation verified successfully!"
    echo "============================================"
    """)

except (SystemExit, Exception):
    _check_storage_hint()
    raise
