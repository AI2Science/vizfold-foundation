# ESMFold backend

The ESMFold backend runs [ESMFold](https://github.com/facebookresearch/esm) via **HuggingFace Transformers** (`EsmForProteinFolding`) and writes **VizFold-compatible** trace archives: structure + optional attention and activation tensors with metadata. Using Transformers avoids the OpenFold build dependency (no CUDA compilation on cluster).

## Install

**Option A – `vizfold install` (recommended)**  
The executor CLI provisions a self-contained environment — its own Python 3.11, PyTorch,
Transformers, and the `esmfold` package with its `esmfold` entrypoint — and records it in
`~/.config/vizfold/vizfold.json`. It brings its own interpreter rather than building on the host's,
which on a cluster login node is routinely older than the package needs:

```bash
vizfold install esmfold
vizfold status          # shows resolved config + which backends are installed
```

Override the torch wheel when a specific CUDA build is needed:

```bash
ESMFOLD_TORCH_SPEC=torch \
ESMFOLD_PIP_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  vizfold install esmfold
```

**Option B – pip (manual, into a Python ≥3.10 environment)**  
`backends/esmfold/pyproject.toml` declares everything the backend imports, so one command is
enough. Install PyTorch first only when you need a specific CUDA build — pip then finds the
`torch>=2.1` requirement already satisfied and leaves that build alone:

```bash
pip install ./backends/esmfold          # torch, transformers, numpy, and the esmfold package
```



## Run locally

The executor runs the same entrypoint and records the run and its outputs:
`vizfold queue-run esmfold --input-id 6KWC_1 --input-sequence <SEQ> --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta`,
then `vizfold execute-run <id>`. The commands below call it directly, through the environment's own
`esmfold` (`$ESMFOLD_ENV_PREFIX/bin/esmfold`, from `vizfold status`; `python -m esmfold` is the same) — it needs
nothing from the checkout. `scripts/esmfold/run_pretrained_esmf.py` runs the same function by path,
under the same interpreter.

**Structure only (fast):**

```bash
esmfold \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode none
```

**Structure + attention + activations:**

```bash
esmfold \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --model facebook/esmfold_v1 \
  --device cuda \
  --trace_mode attention+activations \
  --layers all \
  --save_fp16
```

**Limit layers/heads (saves memory and disk):**

```bash
esmfold \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode attention \
  --layers 0,1,2,5 \
  --heads 0,1,2
```

**Structure + IPA attention + per-recycle backbone (structure module traces):**

```bash
esmfold \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode attention+activations \
  --structure_traces \
  --save_fp16
```

## Output layout

After a run, `--out` contains:

```
outputs/esmf_6KWC/
  meta.json              # Run metadata (backend, model, shapes, seed, etc.)
  structure/
    predicted.pdb        # Predicted structure (PDB)
    predicted.pt          # Optional coordinate tensor
  trace/
    attention/
      layer_000.pt
      layer_001.pt
      ...
    activations/
      layer_000.pt
      ...
    trunk/                # Evoformer intermediates (per-block + final)
      block_000_seq.pt    # [L, C_s]  per-block sequence state (last recycle)
      block_000_pair.pt   # [L, L, C_z]  per-block pair state (last recycle)
      ...
      s_s.pt              # [L, C_s]  final trunk single representations
      s_z.pt              # [L, L, C_z]  final trunk pair representations
    structure_module/     # Only with --structure_traces
      ipa_attention/
        recycle_00_block_00.pt   # IPA attention [H, N, N]
        ...
      backbone/
        recycle_00_positions.pt  # Per-recycle backbone coords
        recycle_00_states.pt     # Per-recycle single representations
        ...
    summary.json          # Per-layer attention entropy, sparsity, norms
    index.json            # Maps layer/head to path, dtype, shape
  attention/
    msa_row_attn_layer0.txt   # VizFold text format (top-k per head)
    ...
  logs.txt               # Log lines from the run
```

## meta.json

Includes:

- `backend`, `model_name`, `date_time`, `device`, `dtype`
- `sequence_length`, `input_fasta_hash`, `input_fasta_path`
- `layer_count`, `head_count`, `trace_mode`, `tensor_format` (fp16/fp32)
- `trace_formats`: which output formats were produced (`pt`, `txt`)
- `shapes_recorded`: per-file shapes for attention, activations, trunk, and structure module
- `seed`, `deterministic` (if set)
- `repo_commit` (if run from a git repo)

## Reproducibility

- `--seed 42` fixes the PyTorch RNG.
- `--deterministic` sets CuDNN deterministic mode (can be slower).

Both are recorded in `meta.json`.

## Long sequences

Attention storage is O(N²). For long proteins the script warns and suggests:

- `--trace_mode activations` (no attention), or
- `--layers 0,1,2` to save only a few layers.

## Running on a cluster

`vizfold install esmfold` picks the cluster, allocation and install prefix the same way
`vizfold install openfold` does, so there is no batch script to edit and nothing to submit by hand.
Folds go to a GPU node whenever the site settled a GPU partition:

```bash
vizfold install esmfold
vizfold fold 6KWC_1 --backend esmfold
```

Two things worth knowing:

- **`OPENFOLD_GPU_*` governs ESMFold folds too.** The GPU partition, account, gres, resources and
  time that `vizfold status` shows are what an ESMFold run is `srun`'d onto — the names are
  OpenFold-prefixed for historical reasons, but nothing about them is OpenFold-specific.
- **The environment installs a CPU torch build by default.** For a CUDA build, point the installer
  at the matching wheel index:

  ```bash
  ESMFOLD_PIP_INDEX_URL=https://download.pytorch.org/whl/cu126 vizfold install esmfold
  ```

  `ESMFOLD_TORCH_SPEC` pins the spec itself (default `torch`), e.g. `torch==2.5.1`. Neither is part
  of the saved config: they describe how to build the environment, not what the install settled, so
  pass them again on a re-install. The installer's verify step prints which build you ended up with
  (`torch 2.5.1 cuda True`).
