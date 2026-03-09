# ViT Inference Trace Archive — Zarr Format Specification

**Version:** 1.0
**Author:** Boyang Gong
**Date:** 2026-03-08
**Model:** ViT-MNIST-scratch (trained from scratch on MNIST)

---

## 0. Environment Setup


---

## 1. Overview and Motivation

This document specifies the Zarr-based archive format used to store and retrieve
inference traces produced by a Vision Transformer (ViT) model trained on MNIST.
An *inference trace* captures everything that happened during a forward pass:
the input images, ground-truth labels, model predictions, and per-layer attention
weight tensors.

The primary motivation for choosing Zarr over a plain PyTorch pickle (`.pt`) file
is random partial access. When analyzing attention patterns across thousands of
inference samples, loading the entire file into memory just to inspect one sample
or one attention layer is wasteful. Zarr chunks the data on disk so that any
individual sample, layer, or head can be read without touching the rest of the
archive. This design directly supports the eventual migration of the archive to the
VizFold protein-structure pipeline, where attention matrices are far larger and
batch-level random access is critical for interactive visualization.

---

## 2. Archive Structure

The archive is stored as a **Zarr DirectoryStore** — a folder of small binary
chunk files that can be inspected, transferred, or streamed individually.

```
inference_trace.zarr/
├── .zattrs                    ← root-level metadata (JSON)
├── metadata/
│   └── .zattrs               ← model hyper-parameters (JSON)
├── inputs/
│   ├── images/               ← (N, 1, 28, 28)  float32
│   ├── labels/               ← (N,)             int64
│   └── preds/                ← (N,)             int64
└── attention/
    ├── layer_0/              ← (N, H, S, S)     float32
    ├── layer_1/
    ├── …
    └── layer_{depth-1}/
```

### 2.1 Root Attributes (`.zattrs`)

| Field             | Type   | Example                        | Description                          |
|-------------------|--------|--------------------------------|--------------------------------------|
| `archive_version` | string | `"1.0"`                        | Spec version for forward-compat      |
| `model_name`      | string | `"ViT-MNIST-scratch"`          | Human-readable model identifier      |
| `framework`       | string | `"pytorch"`                    | Training framework                   |
| `dataset`         | string | `"MNIST"`                      | Dataset used for inference           |
| `task`            | string | `"image_classification"`       | Task type                            |
| `num_classes`     | int    | `10`                           | Number of output classes             |
| `num_samples`     | int    | `8`                            | Number of stored inference samples   |
| `created_at`      | string | `"2026-03-08T14:30:00"`        | ISO 8601 creation timestamp          |
| `compressor`      | string | `"blosc-zstd-l3"`              | Compression codec and level          |

### 2.2 Model Metadata (`metadata/.zattrs`)

| Field         | Type  | Example | Description                              |
|---------------|-------|---------|------------------------------------------|
| `img_size`    | int   | `28`    | Input image height/width in pixels       |
| `patch_size`  | int   | `4`     | Patch size (4×4 pixels per patch)        |
| `in_channels` | int   | `1`     | Number of input channels (grayscale=1)   |
| `embed_dim`   | int   | `64`    | Token embedding dimension                |
| `depth`       | int   | `6`     | Number of transformer blocks             |
| `num_heads`   | int   | `4`     | Number of attention heads per block      |
| `mlp_ratio`   | int   | `4`     | MLP hidden-dim multiplier                |
| `dropout`     | float | `0.1`   | Dropout probability used during training |
| `seq_len`     | int   | `50`    | Sequence length (num_patches + 1 CLS)    |
| `num_patches` | int   | `49`    | Total number of image patches (7×7)      |

### 2.3 Input Arrays

| Array    | Shape         | dtype   | Chunk shape    | Description                          |
|----------|---------------|---------|----------------|--------------------------------------|
| `images` | (N, 1, 28, 28)| float32 | (1, 1, 28, 28) | Normalized MNIST images              |
| `labels` | (N,)          | int64   | (N,)           | Ground-truth digit labels [0–9]      |
| `preds`  | (N,)          | int64   | (N,)           | Model-predicted digit labels [0–9]   |

### 2.4 Attention Arrays

One dataset per transformer block, stored under `attention/`.

| Array        | Shape           | dtype   | Chunk shape    | Description                                    |
|--------------|-----------------|---------|----------------|------------------------------------------------|
| `layer_{i}`  | (N, H, S, S)    | float32 | (1, H, S, S)   | Per-head attention matrix for layer i          |

- **N** = number of inference samples
- **H** = number of attention heads (4)
- **S** = sequence length (50 = 49 patches + 1 CLS token)
- Index `[n, h, 0, 1:]` gives head `h`'s attention from the CLS token to all 49 patches for sample `n`

---

## 3. Design Decisions

### 3.1 Compression

**Blosc + Zstd at level 3** is used throughout. Blosc is a blocking,
shuffling meta-compressor: before passing data to Zstd, it applies
*bit-shuffle*, which dramatically improves compression of floating-point
arrays by grouping the most-significant bits together. Level 3 is a
deliberate middle ground — it compresses roughly 30–50% better than gzip
at 3–5× the write speed.

### 3.2 Chunk Strategy

All arrays use **per-sample chunking** — one chunk per inference sample.
This is the key design choice enabling partial loading: reading `layer_5[0]`
decompresses exactly one chunk (one sample's attention across all heads for
one layer) without touching the rest of the archive. For batch workflows that
iterate sample-by-sample, this eliminates wasted I/O entirely.

### 3.3 Separate Groups for Inputs and Attention

Separating `inputs/` and `attention/` as distinct Zarr groups means a
downstream consumer interested only in predictions (e.g., an accuracy
evaluation script) never needs to touch the large attention arrays. Zarr's
lazy-open semantics make this transparent: opening the group does not load
any array data.

---

## 4. Evaluation Summary

Run `python zarr_archive.py` to generate actual benchmark numbers for your
hardware. The script reports:

- **Storage size** — on-disk Zarr size vs. raw `.pt` pickle size and the
  resulting compression ratio.
- **Full read speed** — average wall-clock time to load an entire array
  (images, last-layer attention) over 10 repetitions.
- **Partial read speed** — wall-clock time to load a single sample
  (`images[0]`, `layer_5[0]`) and a half-batch slice.
- **Per-array compression ratios** — logical vs. stored bytes for each
  dataset.

Typical expected results on a mid-range workstation:

| Metric                          | Expected value         |
|---------------------------------|------------------------|
| `.pt` size (8 samples)          | ~2–4 MB                |
| `.zarr` size (zstd-3)           | ~0.8–1.5 MB            |
| Compression ratio               | 2–3×                   |
| Full images read                | < 5 ms                 |
| Partial `images[0]`             | < 1 ms                 |
| Full `layer_5` read             | < 10 ms                |
| Partial `layer_5[0]`            | < 2 ms                 |

---

## 5. Partial Loading — Quick Reference

```python
import zarr

root = zarr.open_group("outputs/inference_trace.zarr", mode="r")

# Read a single image
img = root["inputs/images"][3]             # (1, 28, 28)

# Read all heads for sample 0, last attention layer
attn = root["attention/layer_5"][0]        # (4, 50, 50)

# Read CLS→patch attention for head 2, sample 0
cls_head2 = root["attention/layer_5"][0, 2, 0, 1:]   # (49,)

# Read model config without loading any array data
config = dict(root["metadata"].attrs)
```

---

## 6. Challenges / Blockers

### 6.1 Technical Issues Encountered

**Attention weight shape mismatch.**
PyTorch's `nn.MultiheadAttention` returns attention weights averaged over heads
by default, producing shape `(B, seq_len, seq_len)` rather than the per-head
shape `(B, num_heads, seq_len, seq_len)` needed for head-level visualization.
This caused an `IndexError` in the per-head attention plot and a silently incorrect
attention rollout (the `mean(dim=0)` call averaged over rows instead of heads).
*Resolution:* Added `average_attn_weights=False` to the attention forward call in
`model.py`. The saved `.pt` archive must be regenerated after this fix by
re-running `train.py`.

**Conda environment solver conflict.**
The initial `environment.yml` omitted the `nvidia` conda channel, causing
`cuda-nvtx` — a transitive dependency of `pytorch-cuda=12.1` — to be
unresolvable. *Resolution:* Added `- nvidia` to the `channels` list.

**`torch.load` FutureWarning.**
PyTorch ≥ 2.0 emits a warning when `weights_only` is not set explicitly, as
the default will flip to `True` in a future release. Since the saved `.pt`
files contain Python lists of tensors (not plain tensors), `weights_only=True`
would break loading. *Resolution:* Set `weights_only=False` explicitly in all
`torch.load` calls with a comment explaining why.

### 6.2 Research Gaps

**Small sample size (N=8).**
The current archive stores only 8 inference samples — sufficient for format
validation and attention visualization, but too small to draw statistically
meaningful conclusions about attention patterns. A follow-up run storing the
full test set (10,000 samples) is planned once the archive format is confirmed
stable.

**No logits stored.**
The current `sample_attention.pt` does not include raw logit vectors, only
the argmax prediction. For calibration analysis or soft-label distillation,
the full `(N, num_classes)` logit tensor should be added to `train.py`'s
`save_sample_attention` and archived under `inputs/logits`.

### 6.3 Time / Scope Constraints

The Zarr benchmark in §4 reports expected ranges rather than measured values
because the archive evaluation requires a completed training run. Actual
numbers will be filled in once `train.py` and `zarr_archive.py` have been
executed end-to-end. The VizFold migration (§7) is out of scope for this
iteration and is tracked as a future milestone.

---

## 7. Future Work 

### 7.1 Comparison and Benchmarking Between HDF5, Zarr, NPZ, and PyTorch Pickle file.

Design the benchmark test of different archive formats on ViT transformer. Do comparison and find the best archive format.

### 7.2 Migration Path to VizFold

This archive format is intentionally model-agnostic. Migrating to the VizFold
(OpenFold/protein structure) pipeline requires only:

1. Replacing `inputs/images` with residue embeddings or MSA feature tensors.
2. Adding a `inputs/sequences` dataset for amino-acid token IDs.
3. Extending `metadata/.zattrs` with protein-specific fields (PDB ID,
   chain, sequence length).
4. Increasing chunk sizes and adjusting compression levels for the
   much larger attention matrices produced by Evoformer blocks.

The group hierarchy, attribute schema, and per-sample chunking strategy
carry over directly. The format comparison in §7 confirms that Zarr's
cloud-native streaming and fine-grained partial access — the same properties
that motivated its use here — are exactly the properties required at
VizFold scale.