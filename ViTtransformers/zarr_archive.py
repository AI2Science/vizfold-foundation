"""
Build a Zarr inference-trace archive from a ViT sample_attention.pt file,
then benchmark storage size, write/read speed, and partial-loading support.

Usage:
    python zarr_archive.py
    python zarr_archive.py --input outputs/sample_attention.pt \
                           --out-dir outputs/ \
                           --clevel 3
"""

import argparse
import os
import time

import numcodecs
import numpy as np
import torch
import zarr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def dir_size(path):
    """Recursively sum file sizes under a directory."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


# ---------------------------------------------------------------------------
# Archive builder
# ---------------------------------------------------------------------------

def build_archive(data, archive_path, clevel=3):
    """
    Write the Zarr inference-trace archive.

    Archive layout
    --------------
    /                           root group
    ├── .zattrs                 root-level metadata (model, dataset, timestamp)
    ├── metadata/
    │   └── .zattrs            model hyper-parameters
    ├── inputs/
    │   ├── images              (N, 1, 28, 28)  float32
    │   ├── labels              (N,)             int64
    │   └── preds               (N,)             int64
    └── attention/
        ├── layer_0             (N, H, S, S)    float32
        ├── layer_1             …
        └── layer_{depth-1}    …
    """
    compressor = numcodecs.Blosc(cname="zstd", clevel=clevel, shuffle=numcodecs.Blosc.BITSHUFFLE)

    images = data["images"].numpy().astype(np.float32)   # (N,1,28,28)
    labels = data["labels"].numpy().astype(np.int64)     # (N,)
    preds  = data["preds"].numpy().astype(np.int64)      # (N,)
    attn   = data["attn_weights"]                        # list[Tensor (N,H,S,S)]

    N      = images.shape[0]
    depth  = len(attn)
    H      = attn[0].shape[1]   # num_heads
    S      = attn[0].shape[2]   # seq_len (50)

    store = zarr.DirectoryStore(archive_path)
    root  = zarr.open_group(store, mode="w")

    # ── Root metadata ────────────────────────────────────────────────────────
    root.attrs.update({
        "archive_version": "1.0",
        "model_name":      "ViT-MNIST-scratch",
        "framework":       "pytorch",
        "dataset":         "MNIST",
        "task":            "image_classification",
        "num_classes":     10,
        "num_samples":     N,
        "created_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compressor":      f"blosc-zstd-l{clevel}",
    })

    # ── Model hyper-parameters ───────────────────────────────────────────────
    meta = root.require_group("metadata")
    meta.attrs.update({
        "img_size":    28,
        "patch_size":  4,
        "in_channels": 1,
        "embed_dim":   64,
        "depth":       depth,
        "num_heads":   H,
        "mlp_ratio":   4,
        "dropout":     0.1,
        "seq_len":     S,          # num_patches + 1 (CLS)
        "num_patches": S - 1,
    })

    # ── Inputs ───────────────────────────────────────────────────────────────
    inp = root.require_group("inputs")
    # Chunk per sample so a single image can be read without loading all N
    inp.create_dataset("images", data=images,
                       chunks=(1, 1, 28, 28), compressor=compressor, dtype="f4")
    inp.create_dataset("labels", data=labels,
                       chunks=(N,), compressor=compressor, dtype="i8")
    inp.create_dataset("preds",  data=preds,
                       chunks=(N,), compressor=compressor, dtype="i8")

    # ── Attention weights ─────────────────────────────────────────────────────
    attn_grp = root.require_group("attention")
    for layer_idx, layer_attn in enumerate(attn):
        arr = layer_attn.numpy().astype(np.float32)   # (N, H, S, S)
        # Chunk: one sample, all heads, full attention matrix
        attn_grp.create_dataset(
            f"layer_{layer_idx}",
            data=arr,
            chunks=(1, H, S, S),
            compressor=compressor,
            dtype="f4",
        )

    return root


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(archive_path, pt_path):
    print("\n" + "=" * 60)
    print("ZARR ARCHIVE EVALUATION")
    print("=" * 60)

    # ── Storage size ─────────────────────────────────────────────────────────
    pt_size  = os.path.getsize(pt_path)
    zar_size = dir_size(archive_path)
    ratio    = pt_size / zar_size if zar_size > 0 else float("inf")
    print(f"\n[Storage]")
    print(f"  .pt  (pickle)  : {fmt_bytes(pt_size)}")
    print(f"  .zarr (zstd-3) : {fmt_bytes(zar_size)}")
    print(f"  Compression ratio (pt/zarr): {ratio:.2f}x")

    store = zarr.DirectoryStore(archive_path)
    root  = zarr.open_group(store, mode="r")

    # ── Full read speed ───────────────────────────────────────────────────────
    print(f"\n[Full Read Performance]")
    REPS = 10

    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = root["inputs/images"][:]
    img_read_ms = (time.perf_counter() - t0) / REPS * 1000
    print(f"  images   full read (avg {REPS}x): {img_read_ms:.2f} ms")

    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = root["attention/layer_5"][:]
    attn_read_ms = (time.perf_counter() - t0) / REPS * 1000
    print(f"  layer_5  full read (avg {REPS}x): {attn_read_ms:.2f} ms")

    # ── Partial (single-sample) read speed ───────────────────────────────────
    print(f"\n[Partial Load Performance — single sample, index 0]")

    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = root["inputs/images"][0]
    partial_img_ms = (time.perf_counter() - t0) / REPS * 1000
    print(f"  images[0]        : {partial_img_ms:.2f} ms")

    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = root["attention/layer_5"][0]
    partial_attn_ms = (time.perf_counter() - t0) / REPS * 1000
    print(f"  layer_5[0]       : {partial_attn_ms:.2f} ms")

    # ── Partial slice (multi-sample) ─────────────────────────────────────────
    n = root["inputs/images"].shape[0]
    half = max(1, n // 2)
    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = root["attention/layer_0"][:half]
    slice_ms = (time.perf_counter() - t0) / REPS * 1000
    print(f"  layer_0[:{half}] slice  : {slice_ms:.2f} ms")

    # ── Metadata read ─────────────────────────────────────────────────────────
    print(f"\n[Metadata]")
    print(f"  Root attrs     : {dict(root.attrs)}")
    print(f"  Model config   : {dict(root['metadata'].attrs)}")

    # ── Array summaries ───────────────────────────────────────────────────────
    print(f"\n[Array Info]")
    for path in ["inputs/images", "inputs/labels", "inputs/preds",
                 "attention/layer_0", "attention/layer_5"]:
        arr = root[path]
        nbytes_stored = arr.nbytes_stored
        nbytes_logical = arr.nbytes
        cr = nbytes_logical / nbytes_stored if nbytes_stored > 0 else float("inf")
        print(f"  {path:<30s}  shape={arr.shape}  "
              f"logical={fmt_bytes(nbytes_logical)}  "
              f"stored={fmt_bytes(nbytes_stored)}  ratio={cr:.1f}x")

    print("\n" + "=" * 60)
    print("Evaluation complete.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    default="./outputs/sample_attention.pt")
    p.add_argument("--out-dir",  default="./outputs")
    p.add_argument("--clevel",   type=int, default=3,
                   help="Blosc zstd compression level 1-9 (default 3)")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run train.py first.")
        return

    archive_path = os.path.join(args.out_dir, "inference_trace.zarr")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    data = torch.load(args.input, map_location="cpu", weights_only=False)

    print(f"Building Zarr archive → {archive_path}")
    t0 = time.perf_counter()
    build_archive(data, archive_path, clevel=args.clevel)
    write_ms = (time.perf_counter() - t0) * 1000
    print(f"Write complete in {write_ms:.1f} ms")

    benchmark(archive_path, args.input)


if __name__ == "__main__":
    main()