# ESMFold Backend Reproducibility Guide

Verifying the ESMFold backend end to end: inference, trace extraction, and the archive it writes.

## Environment Setup

```bash
vizfold install base                    # the checkout the installer lives in
vizfold install esmfold
vizfold status                          # prints ESMFOLD_ENV_PREFIX

# it is not exported into your shell -- set it from what status printed
export ESMFOLD_ENV_PREFIX=...
micromamba run -p "$ESMFOLD_ENV_PREFIX" esmfold --help
```

The installer brings its own Python 3.11 — a login node's `python3` is routinely too old — and
solves torch against the host's GPU driver. For a manual pip install instead, see Option B in
[esmfold.md](esmfold.md#install).

## Structure-Only Inference Test

```bash
micromamba run -p "$ESMFOLD_ENV_PREFIX" esmfold \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/test_run \
  --trace_mode none \
  --device cpu
```

Expected outputs:

- `outputs/test_run/meta.json`
- `outputs/test_run/logs.txt` — written for every run, including `--trace_mode none`
- `outputs/test_run/structure/predicted.pdb`
- `outputs/test_run/structure/predicted.pt` — when the model returns coordinates

## Trace Extraction Test (Attention + Activations)

```bash
micromamba run -p "$ESMFOLD_ENV_PREFIX" esmfold \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/test_trace \
  --trace_mode attention+activations \
  --device cpu
```

Adds `trace/attention/`, `trace/activations/`, `trace/trunk/`, `trace/index.json`,
`trace/summary.json`, and top-k text exports in `attention/` at the archive root. The complete
layout is in [esmfold.md](esmfold.md#output-layout) — check the run against that, not a copy here.

`attention/msa_row_attn_layer*.txt` uses OpenFold's own `save_attention_topk` when `openfold` is
importable, and otherwise reproduces the same format.

## Through the Executor

The same run, recorded in the run database. `--no-exec` prints the run id to follow with `run`;
note that vizfold spells its flags with dashes (`--trace-mode`, `--structure-traces`) where the
script uses underscores.

```bash
vizfold run examples/monomer/fasta_dir_6KWC/6KWC.fasta --backend esmfold --structure-traces --no-exec
vizfold run <RUN_ID>
vizfold show run <RUN_ID>          # the run and its registered artifacts
```

`vizfold run 6KWC_1 --backend esmfold` records and folds the bundled example in one step. Preflight
checks the GPU, the base command, `input_id`, that the FASTA is a readable file, and the output dir.

## Running on ICE Cluster (PACE)

```bash
ssh <gt_username>@login-ice.pace.gatech.edu
vizfold install base
vizfold install esmfold
vizfold run 6KWC_1 --backend esmfold
```

The site is `ice-slurm`: `ice-cpu` / `ice-gpu` partitions, `gpu:a100:1`. Two gotchas:

- `OPENFOLD_GPU_*` governs ESMFold folds too — those settings are what an ESMFold run is `srun`'d
  onto. The names are OpenFold-prefixed for historical reasons only.
- The env is solved with micromamba against the GPU driver's CUDA, so a fold on the GPU partition
  gets a torch the driver can load. The installer prints the driver version it detected, and the
  verify step prints the torch build it ended up with.

## Verification Checklist

Counts from a full `attention+activations` trace of the 6KWC example:

- 36 attention tensors in `trace/attention/` (one per ESM-2 layer)
- 36 + 2 per captured recycling iteration in `trace/activations/` — the extras are
  `recycle_<i>_s_s` and `recycle_<i>_s_z`, which land in `activations/`, not `trunk/`
- ~98 Evoformer trunk tensors in `trace/trunk/` (48 blocks × seq/pair, plus final `s_s` and `s_z`)
- 36 text files in `attention/`

Shapes:

- attention: `[B, H, N, N]` — `<cls>`/`<eos>` sliced off, so `N` is the sequence length
- activations: `[B, N, D]` — sliced the same way, so residue indices line up with attention
- pair representations (`s_z`): `[L, L, C_z]`

With `--structure_traces`, also check:

- `trace/structure_module/ipa_attention/recycle_NN_block_NN.pt`
- `trace/structure_module/backbone/recycle_NN_positions.pt` and `recycle_NN_states.pt`
