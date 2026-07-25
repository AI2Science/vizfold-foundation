# Running VizFold (ESMFold) on ICE

This guide covers running the ESMFold backend and trace export on an ICE cluster via SLURM.

## Prerequisites

- Access to ICE with GPU nodes
- The `vizfold` CLI on PATH; `vizfold install esmfold` builds the PyTorch (CUDA) + `transformers` environment for you

## Environment

```bash
vizfold install esmfold
vizfold status          # prints OPENFOLD_HOME and ESMFOLD_ENV_PREFIX
```

The install creates the environment (its own Python, PyTorch, Transformers) at `<VIZFOLD_ENV_BASE>/vizfold-esmfold` and
records `ESMFOLD_ENV_PREFIX` in `~/.config/vizfold/vizfold.json`. See [esmfold.md](esmfold.md) for
the manual pip path and the CUDA wheel overrides (`ESMFOLD_TORCH_SPEC`, `ESMFOLD_PIP_INDEX_URL`).

## Interactive GPU session (debugging)

Request an interactive GPU node:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00
```

Then:

```bash
module load cuda   # or your site’s module
source "$ESMFOLD_ENV_PREFIX/bin/activate"    # the prefix `vizfold status` reports
cd "$OPENFOLD_HOME"
python -c "import torch; print(torch.cuda.is_available())"
```

## Submitting a batch job

1. Set environment variables (or edit `scripts/esmfold/run_esmf_ice.slurm`):

   - `FASTA` – path to input FASTA (single sequence)
   - `OUTDIR` – where to write outputs (e.g. `outputs/esmf_6KWC`)
   - `TRACE_MODE` – `none`, `attention`, `activations`, or `attention+activations`
   - `DEVICE` – torch device (default `cuda`)
   - `OPENFOLD_HOME` – checkout the job `cd`s into before running; relative `FASTA`/`OUTDIR`
     resolve there (unset means the submit directory)

2. Submit:

```bash
export FASTA=examples/monomer/fasta_dir_6KWC/6KWC.fasta
export OUTDIR=outputs/esmf_6KWC
sbatch scripts/esmfold/run_esmf_ice.slurm
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
sbatch scripts/esmfold/run_esmf_ice.slurm
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
| `torch` not seeing GPU | The venv holds a CPU wheel; reinstall from a CUDA index: `ESMFOLD_PIP_INDEX_URL=https://download.pytorch.org/whl/cu126 vizfold install esmfold` |
| Missing packages | The job runs a bare `python`; activate the venv (`source $ESMFOLD_ENV_PREFIX/bin/activate`) before `sbatch` |
| Disk quota | Use `OUTDIR` on scratch or project space, not home if limited |
| Job killed (OOM) | Increase `--mem` or use shorter sequence / `--trace_mode none` |

## Partition and resources

- Use your site’s GPU partition name in the script if different (e.g. `#SBATCH --partition=gpu`).
- Adjust `--time`, `--mem`, and `--cpus-per-task` to match queue limits and job size.
