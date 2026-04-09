# ESMFold Backend Reproducibility Guide

This document describes how to run the HuggingFace-based ESMFold backend with
VizFold-compatible trace export using the shared `feature/esmfold-backend`
integration branch.

It provides instructions for verifying structure inference, attention extraction,
activation extraction, and expected archive outputs locally and on the ICE cluster.

## Environment Setup (Local)

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements-esmfold.txt
pip install torch
pip install -e . --no-build-isolation
```

## Structure-Only Inference Test

Run:

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/test_run \
  --trace_mode none \
  --device cpu
```

Expected outputs:

- `outputs/test_run/meta.json`
- `outputs/test_run/structure/predicted.pdb`

Successful execution confirms:

- model loads correctly
- inference pipeline runs end-to-end
- archive metadata generation works

## Trace Extraction Test (Attention + Activations)

Run:

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/test_trace \
  --trace_mode attention+activations \
  --device cpu
```

Expected outputs:

- `outputs/test_trace/meta.json`
- `outputs/test_trace/structure/predicted.pdb`
- `outputs/test_trace/trace/`

Trace directory should contain:

- `trace/attention/`
- `trace/activations/`

## Verified Tensor Outputs (Local Validation)

Successful execution produces:

- 36 attention tensors
- 36 activation tensors
- 72 total `.pt` trace tensors

Attention tensors follow expected shape:

`[B, H, N, N]`

where:

- B = batch size
- H = number of attention heads
- N = sequence length (after special-token slicing)

This confirms compatibility with VizFold's visualization pipeline.

## Archive Structure Validation

Expected archive layout:

```
outputs/test_trace/
├── meta.json
├── structure/
│   └── predicted.pdb
└── trace/
    ├── attention/
    └── activations/
```

This structure matches the OpenFold-compatible VizFold archive schema.

## Running on ICE Cluster (PACE)

Login:

```bash
ssh <gt_username>@login-ice.pace.gatech.edu
```

Navigate to the repository:

```bash
cd attention-viz-demo
```

Activate environment:

```bash
source .venv/bin/activate
```

Run structure inference:

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/test_run \
  --trace_mode none \
  --device cuda
```

Run trace extraction:

```bash
python run_pretrained_esmf.py \
  --fasta examples/monomer/fasta_dir_6KWC/6KWC.fasta \
  --out outputs/test_trace \
  --trace_mode attention+activations \
  --device cuda
```

Expected outputs:

- `structure/predicted.pdb`
- `meta.json`
- `trace/attention/`
- `trace/activations/`

GPU execution confirms cluster compatibility for larger inference workloads.


## Additional Intermediate Output Validation

The latest shared `feature/esmfold-backend` branch now exports additional intermediate outputs beyond the original encoder attention and activation traces.

Verified local outputs include:

- 36 attention tensors in `trace/attention/`
- 36 activation tensors in `trace/activations/`
- ~98 Evoformer trunk intermediate tensors in `trace/trunk/`
- 36 VizFold attention text files in `attention_files/`

Expected tensor shapes include:

- attention tensors: `[B, H, N, N]`
- activation tensors: `[B, N, D]`
- pair representations (`s_z`): `[N, N, D]`

If recycling outputs are enabled, they are expected to appear under `trace/trunk/` with keys such as:

- `recycle_*_s_s`
- `recycle_*_s_z`

If structure-module / IPA outputs are enabled, they should also be saved as `.pt` tensors in the trace archive and can be validated separately for expected attention dimensions.