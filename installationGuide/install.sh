#!/usr/bin/env bash
set -x

# Set the directory you wish put OpenFold and Vizfold-Foundation into, please change this to your desired root directory
export ROOT_DIR=~/scratch
# Set the directory that contains OpenFold data
export DATA_DIR=/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data
# Set the directory you wish put Conda
export CONDA_INSTALL_DIR=~/scratch
# Set the module that loads a base environment with Conda and Mamba
export CONDA_MODULE=miniforge
# Set the Mamba command to run (Conda can be used as an alternative)
export MAMBA_CMD=mamba

# Expand paths
export ROOT_DIR="$(realpath -m "$ROOT_DIR")"
export DATA_DIR="$(realpath -m "$DATA_DIR")"
export CONDA_INSTALL_DIR="$(realpath -m "$CONDA_INSTALL_DIR")"

# Load module and resolve mamba path
module load "$CONDA_MODULE"
export MAMBA_CMD="$(which "$MAMBA_CMD")"

# Clone the Vizfold-Foundation repository
git clone https://github.com/AI2Science/vizfold-foundation.git $ROOT_DIR/vizfold-foundation
# Clone the OpenFold repository
git clone https://github.com/aqlaboratory/openfold.git $ROOT_DIR/openfold

# Load Miniforge module (present 3 times to flush all instances of other conda-based modules out of the module path)
module load $CONDA_MODULE
module load $CONDA_MODULE
module load $CONDA_MODULE
# Change directory to OpenFold
cd $ROOT_DIR/openfold
# Activate Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs
conda init
# Create the OpenFold conda environment
$MAMBA_CMD env create -n openfold_env -f environment.yml -y
# Activate the OpenFold conda environment
$MAMBA_CMD activate openfold_env
# Add pip dependencies not installed by Mamba
pip install -v deepspeed==0.14.5 dm-tree==0.1.6 git+https://github.com/NVIDIA/dllogger.git https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl --no-build-isolation
echo "openfold_env created and activated"

# Load Miniforge module
module load $CONDA_MODULE
module load $CONDA_MODULE
module load $CONDA_MODULE
# Activate Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs
conda init
$MAMBA_CMD activate openfold_env
# Change directory to OpenFold
cd $ROOT_DIR/openfold
# Set up compiler and library paths
mkdir -p $CONDA_PREFIX/x86_64-conda-linux-gnu/lib
ln -s $(realpath $CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/12.4.0/cc1plus) $CONDA_PREFIX/bin/cc1plus
ln -s $(realpath $CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0/crtbeginS.o) $CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtbeginS.o
ln -s $(realpath $CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0/crtendS.o) $CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtendS.o
ln -s $(realpath $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/crti.o) $CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crti.o
ln -s $(realpath $CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/crtn.o) $CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtn.o
# Install gcc and libgcc-ng
$MAMBA_CMD install -y gcc_linux-64 libgcc-ng
# Set up environment variables
export GCC_LTO_PLUGIN="$CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/12.4.0/liblto_plugin.so"
export CFLAGS="-O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
export CXXFLAGS="$CXXFLAGS -fno-use-linker-plugin -O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
export CFLAGS="$CFLAGS -fno-use-linker-plugin -O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
export LDFLAGS="$LDFLAGS -fno-use-linker-plugin -O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
export LDFLAGS="$LDFLAGS -L$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0 -L$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64"
export CPATH="$CONDA_PREFIX/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# Install OpenFold
pip install . --no-build-isolation
# Install third-party dependencies
scripts/install_third_party_dependencies.sh

# Load Miniforge module
module load $CONDA_MODULE
module load $CONDA_MODULE
module load $CONDA_MODULE
# Activate Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs
conda init
$MAMBA_CMD activate openfold_env
# Install additional required packages
$MAMBA_CMD install -y ipykernel
python -m ipykernel install --user --name=openfold_env \\
    --env PATH "$CONDA_INSTALL_DIR/.conda/envs/openfold_env/bin:/usr/local/cuda/bin:/usr/bin:/bin" \\
    --env LD_LIBRARY_PATH "$CONDA_INSTALL_DIR/.conda/envs/openfold_env/lib:/opt/slurm/current/lib"
# Set up Vizfold-Foundation
mkdir -p $ROOT_DIR/vizfold-foundation/openfold
ln -s $(realpath $ROOT_DIR/openfold/openfold/data) $ROOT_DIR/vizfold-foundation/openfold/data
# Create necessary directories and symlinks
mkdir -p $ROOT_DIR/vizfold-foundation/openfold/resources
ln -s $(realpath $DATA_DIR/params) $ROOT_DIR/vizfold-foundation/openfold/resources/params
wget -N --no-check-certificate -P $ROOT_DIR/vizfold-foundation/openfold/resources https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# Load Miniforge module
module load $CONDA_MODULE
module load $CONDA_MODULE
module load $CONDA_MODULE
# Activate Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=$CONDA_INSTALL_DIR/.conda/envs
export CONDA_PKGS_DIRS=$CONDA_INSTALL_DIR/.conda/pkgs
conda init
$MAMBA_CMD activate openfold_env
# Install matplotlib
$MAMBA_CMD install -y conda-forge::matplotlib
# Set strict channel priority for consistent package resolution
conda config --set channel_priority strict 
# Install PyMOL for molecular visualization
$MAMBA_CMD install -y -c conda-forge -c pytorch -c nvidia pymol-open-source
# Reset channel priority
conda config --remove-key channel_priority