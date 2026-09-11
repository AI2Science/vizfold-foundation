# Vizfold Foundations

This repository has two main components:

1. Model inference & feature extraction: Run protein structure prediction models and extract intermediate activations (hidden representations) and attention maps from any chosen layer.
2. Visualization & analysis: Explore, visualize, and analyze the extracted activations and attention maps.

---

Link to Openfold implimentation - [README_vizfold_openfold.md](https://github.com/vizfold/vizfold-foundation/blob/main/README_vizfold_openfold.md)

---

## PACE ICE (GT HPC) Setup

> Installs to **SCRATCH** (not HOME), builds OpenFold, and sets up the attention demo. Run on a **GPU node** you allocate with Slurm.

### 0) Get a GPU node
```bash
# from a login node
salloc -p ice-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G -t 02:00:00
# when the shell drops onto the compute node, continue below
```

### 1) Initialize tools and clone repos

```bash
module load mamba
source /usr/local/pace-apps/manual/packages/miniforge/24.3.0-0/etc/profile.d/conda.sh

git clone https://github.com/vizfold/attention-viz-demo.git ~/scratch/attention-viz-demo
git clone https://github.com/aqlaboratory/openfold.git       ~/scratch/openfold

cd ~/scratch/openfold
```

### 2) Use SCRATCH for env & cache, then create env with Mamba

```bash
export SCRATCH_ROOT=/storage/ice1/2/0/$USER
mkdir -p "$SCRATCH_ROOT/conda-pkgs" "$SCRATCH_ROOT/envs"
export CONDA_PKGS_DIRS="$SCRATCH_ROOT/conda-pkgs"
export ENV_PREFIX="$SCRATCH_ROOT/envs/openfold_env"

mamba env create -p "$ENV_PREFIX" -f environment.yml
conda activate "$ENV_PREFIX"
```

### 3) Compiler & library paths for CUDA extension build

```bash
mkdir -p "$CONDA_PREFIX/x86_64-conda-linux-gnu/lib"
ln -sf "$CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/12.4.0/cc1plus" "$CONDA_PREFIX/bin/"
ln -sf "$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0/crtbeginS.o" "$CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtbeginS.o"
ln -sf "$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0/crtendS.o"   "$CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtendS.o"
ln -sf "$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/crti.o"   "$CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crti.o"
ln -sf "$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/crtn.o"   "$CONDA_PREFIX/x86_64-conda-linux-gnu/lib/crtn.o"

mamba install -y -c conda-forge gcc_linux-64 libgcc-ng

export GCC_LTO_PLUGIN="$CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/12.4.0/liblto_plugin.so"
export CFLAGS="-O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
export CXXFLAGS="$CXXFLAGS -fno-use-linker-plugin -O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
export LDFLAGS="$LDFLAGS -fno-use-linker-plugin -O2 -fno-lto --sysroot=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot"
export LDFLAGS="$LDFLAGS -L$CONDA_PREFIX/lib/gcc/x86_64-conda-linux-gnu/12.4.0 -L$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64"

# Optional: target your node’s GPU arch (A100: 8.0, A40/L40S: 8.6, V100: 7.0, RTX 6000 (Turing): 7.5)
export TORCH_CUDA_ARCH_LIST="8.0"
```

### 4) Install OpenFold (build CUDA kernels) and third-party assets

```bash
cd ~/scratch/openfold
pip install -e .
scripts/install_third_party_dependencies.sh
```

### 5) Jupyter kernel & data links for the demo

```bash
mamba install -y ipykernel
python -m ipykernel install --user --name=openfold_env

cd ~/scratch/attention-viz-demo
ln -s ~/scratch/openfold/openfold/data ./openfold/

mkdir -p openfold/resources
ln -s /storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data/params ./openfold/resources/

wget -N --no-check-certificate -P openfold/resources \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

### 6) Visualization extras

```bash
mamba install -y conda-forge::matplotlib

# Optional: make solves reproducible
conda config --set channel_priority strict

mamba install -y pytorch==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
mamba install -y -c conda-forge -c pytorch -c nvidia pymol-open-source

# Optional: undo the strict setting
conda config --remove-key channel_priority || true
```

**Notes**

* Run installs/builds **on a GPU node** (the environment needs CUDA to compile OpenFold’s extension).
* Env and package caches live under `/storage/ice1/2/0/$USER` to avoid HOME quota issues.
* If a build fails, re-`conda activate "$ENV_PREFIX"` and re-run steps **3–4**.

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](./LICENSE) file for details.

---
