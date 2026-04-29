"""
Writes ESMFold traces into VizFold-compatible archive layout.

Layout:
  out_dir/
    meta.json
    structure/
      predicted.pdb
      predicted.pt   (optional)
    trace/
      attention/
        layer_000.pt
        ...
      activations/
        layer_000.pt
        ...
      trunk/
        block_000_seq.pt       [L, C_s]  per-block sequence state (last recycle)
        block_000_pair.pt      [L, L, C_z]  per-block pair state (last recycle)
        ...
        s_s.pt                 [L, C_s]  final trunk single representations
        s_z.pt                 [L, L, C_z]  final trunk pair representations
      index.json
    attention_files/
      msa_row_attn_layer0.txt   (VizFold text format)
      ...
    logs.txt (caller appends)
"""
import os
import re
from typing import Any, Dict, Optional, Tuple

import torch

from vizfold.backends.esmfold.schema import (
    build_meta,
    write_meta,
    write_trace_index,
)
from vizfold.backends.esmfold.hooks import ESMFoldTraceCollector


def _save_tensor(path: str, t: torch.Tensor, save_fp16: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if save_fp16 and t.dtype == torch.float32:
        t = t.half()
    torch.save(t.cpu(), path)


def write_structure(out_dir: str, pdb_str: Optional[str], coords: Optional[torch.Tensor]) -> Dict[str, str]:
    base = os.path.join(out_dir, "structure")
    os.makedirs(base, exist_ok=True)
    paths = {}
    if pdb_str:
        pdb_path = os.path.join(base, "predicted.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_str)
        paths["pdb"] = pdb_path
    if coords is not None:
        pt_path = os.path.join(base, "predicted.pt")
        torch.save(coords.cpu(), pt_path)
        paths["coords"] = pt_path
    return paths


def write_traces(
    out_dir: str,
    collector: ESMFoldTraceCollector,
    save_fp16: bool = False,
    layer_indices: Optional[list] = None,
    head_indices: Optional[list] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Write attention and activations from collector to trace/attention/ and trace/activations/.
    Returns (attention_index, activations_index) for index.json.
    """
    trace_root = os.path.join(out_dir, "trace")
    attn_dir = os.path.join(trace_root, "attention")
    act_dir = os.path.join(trace_root, "activations")
    os.makedirs(attn_dir, exist_ok=True)
    os.makedirs(act_dir, exist_ok=True)

    attention_index: Dict[str, Any] = {}
    activations_index: Dict[str, Any] = {}

    for key, t in collector.attention.items():
        path = os.path.join(attn_dir, f"{key}.pt")
        # Head filtering is done in the hook (ESMFoldTraceCollector), so
        # tensors here are already filtered if head_indices was specified.
        _save_tensor(path, t, save_fp16=save_fp16)
        attention_index[key] = {
            "path": os.path.relpath(path, out_dir),
            "dtype": str(t.dtype),
            "shape": list(t.shape),
        }

    for key, t in collector.activations.items():
        path = os.path.join(act_dir, f"{key}.pt")
        _save_tensor(path, t, save_fp16=save_fp16)
        activations_index[key] = {
            "path": os.path.relpath(path, out_dir),
            "dtype": str(t.dtype),
            "shape": list(t.shape),
        }

    write_trace_index(out_dir, attention_index, activations_index)
    return attention_index, activations_index


def write_attention_txt(
    out_dir: str,
    collector: ESMFoldTraceCollector,
    top_k: int = 50,
) -> Optional[str]:
    """
    Write VizFold-compatible text-file attention maps from collector.

    Converts each [B, H, N, N] attention tensor into the standard
    msa_row_attn_layer*.txt format used by VizFold visualization tools
    (PyMOL scripts, arc diagrams, etc.).

    Does not require OpenFold to be installed — uses a self-contained
    implementation of the top-k writing logic with numpy only.

    Args:
        out_dir: Root output directory. Files go to out_dir/attention_files/.
        collector: ESMFoldTraceCollector with populated .attention dict.
        top_k: Number of top attention values to save per head.

    Returns:
        Path to the attention_files directory, or None if no attention data.
    """
    import numpy as np

    if not collector.attention:
        return None

    attn_dir = os.path.join(out_dir, "attention_files")
    os.makedirs(attn_dir, exist_ok=True)

    # Try to use OpenFold's implementation for format consistency;
    # fall back to the self-contained version if openfold is not installed.
    try:
        from openfold.model.evoformer import save_attention_topk as _of_save

        def _save(arr: "np.ndarray", layer_idx: int) -> None:
            key = f"layer_{layer_idx:03d}"
            _of_save(
                attention_dict={key: arr},
                save_dir=attn_dir,
                layer_name=key,
                layer_idx=layer_idx,
                attn_type="msa_row_attn",
                triangle_residue_idx=None,
                top_k=top_k,
            )

    except ImportError:
        # Standalone implementation — same output format as save_attention_topk.
        # arr shape: [B, H, N, N]; msa_row_attn slices batch dim → [H, N, N].
        def _save(arr: "np.ndarray", layer_idx: int) -> None:  # type: ignore[misc]
            heads = arr[0]  # [H, N, N]
            path = os.path.join(attn_dir, f"msa_row_attn_layer{layer_idx}.txt")
            with open(path, "w") as f:
                for head_idx, attn_map in enumerate(heads):
                    S = attn_map.shape[0]
                    k = min(top_k, S * S)
                    flat = np.argsort(attn_map.flatten())[::-1][:k]
                    rows, cols = np.unravel_index(flat, (S, S))
                    scores = attn_map[rows, cols]
                    f.write(f"Layer {layer_idx}, Head {head_idx}\n")
                    for r, c, s in zip(rows, cols, scores):
                        f.write(f"{r} {c} {s:.6f}\n")
            print(f"[Done] Saved top {top_k} entries for msa_row_attn to {path}")

    for key, t in collector.attention.items():
        m = re.search(r"\d+", key)
        layer_idx = int(m.group()) if m else 0
        arr = t.float().cpu().numpy()
        _save(arr, layer_idx)

    return attn_dir


def write_trace_summary(
    out_dir: str,
    collector: "ESMFoldTraceCollector",
) -> Optional[str]:
    """
    Write trace/summary.json with per-layer attention entropy, mean/std, sparsity proxy,
    and activation norms. Cheap to compute; useful for analysis.
    """
    import json
    import numpy as np
    summary = {"attention": {}, "activations": {}}
    for key, t in collector.attention.items():
        a = t.float().cpu().numpy()
        # a: [..., heads, N, N]
        if a.size == 0:
            continue
        if a.ndim == 3:
            a = a[np.newaxis, ...]
        # Per-layer (first dim after batch) stats
        for i in range(a.shape[0]):
            layer_key = f"{key}_slice{i}" if a.shape[0] > 1 else key
            block = a[i]
            if block.ndim >= 3:
                # block: [heads, N, N] — each row is a distribution over keys
                # Entropy per row (axis=-1), averaged over all rows and heads
                ent = -np.sum(block * np.log(block + 1e-12), axis=-1).mean()
                summary["attention"][layer_key] = {
                    "mean": float(block.mean()),
                    "std": float(block.std()),
                    "entropy_proxy": float(ent),
                    "sparsity_proxy": float((block < 1e-5).mean()),
                }
    for key, t in collector.activations.items():
        h = t.float().cpu().numpy()
        if h.size == 0:
            continue
        summary["activations"][key] = {
            "norm_mean": float(np.sqrt((h ** 2).mean())),
            "mean": float(h.mean()),
            "std": float(h.std()),
        }
    path = os.path.join(out_dir, "trace", "summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def build_and_write_meta(
    out_dir: str,
    model_name: str,
    fasta_path: str,
    device: str,
    dtype: str,
    seq_len: int,
    fasta_hash: str,
    layer_count: int,
    head_count: int,
    trace_mode: str,
    shapes_recorded: Dict[str, Any],
    seed: Optional[int] = None,
    deterministic: bool = False,
    save_fp16: bool = False,
    top_k: int = 50,
) -> str:
    # Determine which formats were actually produced
    formats = []
    if os.path.isdir(os.path.join(out_dir, "trace", "attention")):
        formats.append("pt")
    if os.path.isdir(os.path.join(out_dir, "attention_files")):
        formats.append("txt")
    if not formats:
        formats = ["none"]

    meta = build_meta(
        backend="esmfold",
        model_name=model_name,
        out_dir=out_dir,
        fasta_path=fasta_path,
        device=device,
        dtype=dtype,
        sequence_length=seq_len,
        fasta_hash=fasta_hash,
        layer_count=layer_count,
        head_count=head_count,
        trace_mode=trace_mode,
        trace_formats=formats,
        shapes_recorded=shapes_recorded,
        seed=seed,
        deterministic=deterministic,
        save_fp16=save_fp16,
        top_k=top_k,
    )
    return write_meta(meta, out_dir)
