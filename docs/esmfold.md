# ESMFold backend

Runs [ESMFold](https://github.com/facebookresearch/esm) via **HuggingFace Transformers**
(`EsmForProteinFolding`) and writes **VizFold-compatible** trace archives: structure plus optional
attention and activation tensors with metadata. No CUDA compilation, no AlphaFold2 databases.

## Install

**Option A – `vizfold install` (recommended).** Provisions a self-contained environment (its own
Python 3.11, PyTorch, Transformers, the `esmfold` package) and records `ESMFOLD_ENV_PREFIX` in
`~/.config/vizfold/vizfold.json`.

```bash
vizfold install source    # once per machine: the checkout the installers live in
vizfold install esmfold
vizfold status          # resolved config + which backends are installed
```

**torch is matched to the machine it lands on.** The env is solved from
[`backends/esmfold/environment.yml`](../backends/esmfold/environment.yml) with micromamba, because
conda-forge's CUDA packages declare the driver they need and the solver honours it. PyPI wheels carry
no such metadata: `pip install torch` takes whatever CUDA is newest, and on a cluster whose driver
lags that, the fold reaches a GPU node and dies with "the NVIDIA driver on your system is too old".

The installer prints the driver version it detected, asks for a CUDA build and caps `cuda-version` at
that driver. Where it finds no driver it asks for neither, so a CPU node or a laptop resolves to the
build it can actually run rather than failing or dragging in a GPU stack. An env already holding a
torch this machine cannot use is rebuilt rather than kept. `OPENFOLD_DRIVER_CUDA` in
`~/.config/vizfold/vizfold.json` overrides the detection — set it on a GPU cluster whose login node
carries no driver.

**Option B – pip**, into a Python ≥3.10 environment. Install PyTorch first if you need a specific
CUDA build — pip then leaves it alone.

```bash
pip install ./backends/esmfold   # torch>=2.1, transformers>=4.36,<5, numpy>=1.24, + the esmfold package
```

## Fold through vizfold

```bash
vizfold run 6KWC_1 --backend esmfold        # bundled example
vizfold run ./my.fasta --backend esmfold    # your own sequence
```

`--backend` defaults to the only backend installed, else openfold — so pass `--backend esmfold`
whenever OpenFold is installed too. Trace settings are flags on `run`; add `--no-exec` to record
the run now and fold it later:

```bash
vizfold run ./my.fasta --backend esmfold --trace-mode attention --layers 0,1,2 --no-exec
vizfold run <run-id>
```

ESMFold folds one target at a time — it reads a single file and loads its model inside the fold —
so more than one target is refused. Batches are OpenFold's.

The first fold downloads `facebook/esmfold_v1` (~2.6 GB). `vizfold run` points `HF_HOME` at
`<env>/hf` when it is unset, keeping the weights out of a quota'd `$HOME`. Driving the model
yourself gets no such redirect — set `HF_HOME` first.

## Drive the model directly

The entrypoint exists only inside the backend env, so run it the way `vizfold install esmfold` prints:

```bash
micromamba run -p <env> esmfold \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/esmf_6KWC \
  --trace_mode attention+activations \
  --layers all \
  --save_fp16 \
  --structure_traces
```

(`python -m esmfold` and `scripts/esmfold/run_pretrained_esmf.py` are the same function.)

Flags, and their spelling on `vizfold run`:

| raw CLI | vizfold run | notes |
|---|---|---|
| `--fasta` | `<TARGET>` | the positional; required on both |
| `--out` | — | the run's output workspace |
| `--model` | `--model` | default `facebook/esmfold_v1` |
| `--device` | `--model-device` | default: cuda if visible, else cpu |
| `--dtype` | `--dtype` | `float32` \| `float16` |
| `--trace_mode` | `--trace-mode` | `none` \| `attention` \| `activations` \| `attention+activations` |
| `--layers` | `--layers` | `all`, `0,1,2`, or `0:12` |
| `--save_fp16` | `--save-fp16` | |
| `--structure_traces` | `--structure-traces` | IPA attention + per-recycle backbone |
| `--heads`, `--top_k` | — | raw CLI only; runs through vizfold use `all` and `50` |
| `--seed`, `--deterministic` | — | raw CLI only; recorded in `meta.json` when set |
| — | `--input-id` | name recorded for the run |

## Output layout

After a run, `--out` contains:

```
outputs/esmf_6KWC/
  meta.json
  structure/
    predicted.pdb
    predicted.pt          # optional coordinate tensor
  trace/
    attention/
      layer_000.pt
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
    structure_module/     # only with --structure_traces
      ipa_attention/
        recycle_00_block_00.pt   # IPA attention [H, N, N]
      backbone/
        recycle_00_positions.pt  # per-recycle backbone coords
        recycle_00_states.pt     # per-recycle single representations
    summary.json          # per-layer attention entropy, sparsity, norms
    index.json            # maps layer/head to path, dtype, shape
  attention/
    msa_row_attn_layer0.txt   # VizFold text format (top-k per head)
    ...
  logs.txt
```

## meta.json

- `backend`, `model_name`, `date_time`, `device`, `dtype`
- `sequence_length`, `input_fasta_hash`, `input_fasta_path`
- `layer_count`, `head_count`, `trace_mode`, `tensor_format` (fp16/fp32), `top_k`
- `trace_formats`: formats produced — `pt`, `txt`, or `["none"]`
- `shapes_recorded`: per-file shapes for attention, activations, trunk, and structure module
- `seed`, `deterministic` (only when set)
- `repo_commit` (if run from a git repo)

## Long sequences

Attention storage is O(N²). Above 400 residues, a run with attention tracing warns and suggests
`--layers 0,1` or `--trace_mode activations`.

## Running on a cluster

`vizfold install esmfold` picks the cluster, allocation and install prefix the same way
`vizfold install openfold` does — no batch script to edit, nothing to submit by hand.

**`OPENFOLD_GPU_*` governs ESMFold folds too.** The GPU partition, account, gres, resources and time
that `vizfold status` shows are what an ESMFold run is `srun`'d onto. The names are OpenFold-prefixed
for historical reasons; nothing about them is OpenFold-specific.
