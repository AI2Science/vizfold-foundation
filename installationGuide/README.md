# OpenFold and VizFold-Foundation Setup

This guide explains how to install OpenFold and VizFold-Foundation on an HPC cluster (tested on Georgia Tech PACE ICE).

## Prerequisites

- Conda package manager available via `module load anaconda3` (or equivalent on your cluster)
- Access to the AlphaFold data directory:
  - **PACE ICE**: `/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data`
  - **Purdue Anvil**: `/anvil/datasets/alphafold/db`
- At least 50 GB of free disk space
- A compute node with GPU access (required for building CUDA extensions)

## Installation

Two equivalent options are provided — choose whichever fits your workflow. Both include automatic pre-flight checks and a verification step at the end.

### Option A — Jupyter Notebook (`install.ipynb`)

1. Open `install.ipynb` in JupyterLab or a compatible IDE.
2. Select a kernel: click **Kernel** (top right) → **Python Kernel** → choose the kernel starting with **base**. If it doesn't appear, click the refresh button.
3. In the **first code cell**, update the three directory variables to match your environment:
   ```python
   os.environ['ROOT_DIR'] = '~/scratch'        # where repos will be cloned
   os.environ['DATA_DIR'] = '/storage/...'     # path to AlphaFold data
   os.environ['CONDA_INSTALL_DIR'] = '~/scratch'  # where the conda env is stored
   ```
4. Click **Run All**. The notebook will:
   - Run **pre-flight checks** (GPU, conda, disk space, data directory)
   - Clone VizFold-Foundation and OpenFold (skips if already cloned)
   - Write a PACE ICE–compatible `environment.yml`
   - Create the `openfold_env` environment (using the fast libmamba solver)
   - Build OpenFold's CUDA extensions using the system GCC 12.3.0
   - Install third-party dependencies
   - Set up the VizFold-Foundation directory structure
   - Install visualization tools (matplotlib, PyMOL)
   - **Verify** the installation (PyTorch + CUDA, OpenFold import, all dependencies)

### Option B — Python Script (`install.py`)

1. Transfer the script to your cluster if needed.
2. Load conda and start an interactive GPU job:
   ```bash
   module load anaconda3
   srun --partition=gpu-ice --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash
   ```
3. Edit the three directory variables near the top of `install.py` to match your environment:
   ```python
   os.environ['ROOT_DIR'] = '~/scratch'
   os.environ['DATA_DIR'] = '/storage/...'
   os.environ['CONDA_INSTALL_DIR'] = '~/scratch'
   ```
4. Run the script:
   ```bash
   python install.py
   ```
   The script prints timestamped progress headers for each step and exits immediately on any error, so failures are easy to locate.

## Notes

- Run the installation on a **compute node with a GPU**, not the login node — building OpenFold's CUDA extensions requires GPU access.
- If your conda environment is stored in a non-default location (i.e. `CONDA_INSTALL_DIR` is not `~`), activate it with:
  ```bash
  conda activate [CONDA_INSTALL_DIR]/.conda/envs/openfold_env
  ```
  Otherwise:
  ```bash
  conda activate openfold_env
  ```
- To use this environment in another Jupyter notebook (e.g. `viz_attention_demo_base.ipynb`), select the `openfold_env` Jupyter kernel.
- The installer uses the **libmamba solver** (`conda --solver=libmamba`) for faster dependency resolution. This is included with recent conda versions and requires no additional installation.
- Installation time varies depending on your internet connection and cluster load; expect 30–90 minutes.
