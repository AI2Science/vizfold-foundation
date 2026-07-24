# Running VizFold (ESMFold) on ICE

This guide covers running the ESMFold backend and trace export on an ICE cluster via SLURM.

## Prerequisites

- Access to ICE with GPU nodes
- Conda (or module environment) with PyTorch (CUDA), and `transformers` installed

## Environment

### Option A: Conda (simpler)

```bash
conda create -n vizfold python=3.10
conda activate vizfold
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers
# From repo root:
pip install -e .
```

### Option B: Container (if ICE supports Apptainer/Singularity)

Use a image that includes PyTorch + fair-esm. Not required; mention in job script if available.

## Interactive GPU session (debugging)

Request an interactive GPU node:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00
```

Then:

```bash
module load cuda   # or your site’s module
conda activate vizfold
cd /path/to/vizfold-foundation
python -c "import torch; print(torch.cuda.is_available())"
```

## Submitting a batch job

1. Set environment variables (or edit `scripts/hpc/ice/run_esmf_ice.slurm`):

   - `FASTA` – path to input FASTA (single sequence)
   - `OUTDIR` – where to write outputs (e.g. `outputs/esmf_6KWC`)
   - `TRACE_MODE` – `none`, `attention`, `activations`, or `attention+activations`

2. Submit:

```bash
export FASTA=examples/monomer/fasta_dir_6KWC/6KWC.fasta
export OUTDIR=outputs/esmf_6KWC
sbatch scripts/hpc/ice/run_esmf_ice.slurm
```

3. Monitor:

```bash
squeue -u $USER
```

4. Logs and outputs:

- SLURM stdout/stderr: `outputs/logs/esmf_<jobid>.out` and `.err`
- Run outputs: under `OUTDIR`: `meta.json`, `structure/`, `trace/`, `logs.txt`

## Minimal smoke test (<5 minutes)

To verify the job runs without using much GPU time:

```bash
# Create a tiny FASTA (e.g. 50 residues)
echo -e ">tiny\nMKFLKFSLLTAVLLSVVFAFSSCGDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD" > /tmp/tiny.fasta
export FASTA=/tmp/tiny.fasta
export OUTDIR=outputs/esmf_smoke
export TRACE_MODE=none
sbatch scripts/hpc/ice/run_esmf_ice.slurm
```

Then run with trace:

```bash
export TRACE_MODE=attention
# Optional: limit layers to speed up
# (add --layers 0,1 to the script or run CLI manually)
```

## Common issues

| Issue | What to check |
|-------|----------------|
| CUDA not found | Load correct `cuda` module; `nvidia-smi` on the node |
| `torch` not seeing GPU | Install PyTorch with CUDA: `conda install pytorch pytorch-cuda=...` |
| Missing packages | `pip install transformers`; run from repo root or `pip install -e .` |
| Disk quota | Use `OUTDIR` on scratch or project space, not home if limited |
| Job killed (OOM) | Increase `--mem` or use shorter sequence / `--trace_mode none` |

## Partition and resources

- Use your site’s GPU partition name in the script if different (e.g. `#SBATCH --partition=gpu`).
- Adjust `--time`, `--mem`, and `--cpus-per-task` to match queue limits and job size.
