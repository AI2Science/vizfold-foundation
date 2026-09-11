# OpenFold and Attention Visualization Demo Setup

This document provides instructions for setting up OpenFold and the Attention Visualization Demo in your HPC environment.

## Prerequisites

- Conda package manager (typically loaded using `module load anaconda3` or similar on a HPC environment)
- Access to the AlphaFold data directory (typically at `/storage/ice1/shared/d-pace_community/alphafold/alphafold_2.3.2_data` for Georgia Tech PACE ICE, and `/anvil/datasets/alphafold/db` for Purdue Anvil)
- Sufficient disk space, estimated at least 50GB of free disk space
- Access to a compute cluster with Slurm workload manager
- GPU allocation for running the visualization demo (must be run as a job on a compute node with GPU access)

## Installation

1. Open the `install.ipynb` notebook in Jupyter, or other compatible IDE
2. Click on the button labeled kernel (Top right)
3. Select "Python Kernel", then the kernel starting with "base". If it doesn't appear, click the refresh button. 
4. For the third line of the first cell (`os.environ['_DIR'] = '~/scratch'`), change the directory to where you want to store your Openfold and Attention Visualization Demo. 
5. For the ninth line of the first cell (`os.environ['CONDA_MODULE'] = 'miniforge'`), change the module which loads Conda and Mamba on your HPC (use `module spider` and search for modules which set up a mini Conda/Mamba environment - change `os.environ['MAMBA_CMD']` if your module doesn't provide Mamba)
6. Click "Run All" to execute the installation process. This will switch to a conda environment, install all dependences, and install OpenFold and the Attention Visualization Demo in your environment.

## Notes

- Ensure you have sufficient disk space in your root and conda install directories
- The installation process may take some time depending on your internet connection and system resources
- Make sure the AlphaFold data directory is accessible and contains the required model parameters
- If you chose to put your Conda environment in a different directory (that is, `CONDA_DIR` is not `~`), activate the Openfold environment using `conda activate [CONDA_DIR]/.conda/envs/openfold_env`. Otherwise use `conda activate openfold_env`.
- To use this environment in another Jupyter notebook (like `viz_attention_demo_base.ipynb`), selected the `openfold_env` Jupyter kernel.
