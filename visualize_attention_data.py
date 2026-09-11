"""
Shared attention-map loading helpers for visualization modules.

All visualization types should consume the parsed ``heads`` structure from this
module instead of re-parsing attention text files independently.
"""

import os
from typing import Dict, List, Optional, Tuple


AttentionEdge = Tuple[int, int, float]
AttentionHeads = Dict[int, List[AttentionEdge]]


def filter_top_k_edges(heads: AttentionHeads, top_k: Optional[int] = None) -> AttentionHeads:
    """Sort each head by descending weight and optionally keep only top-k edges."""
    filtered = {}
    for head_idx, conns in heads.items():
        sorted_conns = sorted(conns, key=lambda x: x[2], reverse=True)
        filtered[head_idx] = sorted_conns[:top_k] if top_k is not None else sorted_conns
    return filtered


def load_attention_map(connections_file: str, top_k: Optional[int] = None) -> AttentionHeads:
    """
    Load a combined attention text file into ``head_idx -> [(res_i, res_j, weight)]``.

    Expected format:
        layer 47 head 0
        12 39 0.41
        8 14 0.32
        layer 47 head 1
        ...
    """
    heads = {}
    current_head = None

    with open(connections_file, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("layer"):
                parts = line.replace(",", "").split()
                try:
                    current_head = int(parts[-1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Could not parse head index on line {line_number}: {line}"
                    ) from exc
                heads[current_head] = []
                continue

            if current_head is None:
                raise ValueError(
                    f"Found attention edge before any head header on line {line_number}: {line}"
                )

            try:
                res1, res2, weight = map(float, line.split())
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse attention edge on line {line_number}: {line}"
                ) from exc
            heads[current_head].append((int(res1), int(res2), weight))

    return filter_top_k_edges(heads, top_k=top_k)


def get_attention_file_path(
    attention_dir: str,
    attention_type: str,
    layer_idx: int,
    residue_idx: Optional[int] = None,
) -> str:
    """Return the canonical attention text-file path for a visualization request."""
    if attention_type == "msa_row":
        return os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")

    if attention_type == "triangle_start":
        if residue_idx is None:
            raise ValueError("residue_idx is required for triangle_start attention")
        return os.path.join(
            attention_dir,
            f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_idx}.txt",
        )

    raise ValueError(f"Unsupported attention_type: {attention_type}")


def parse_fasta_sequence(fasta_path: str) -> str:
    """Parse a single-entry FASTA file and return the sequence string."""
    with open(fasta_path, "r") as f:
        return "".join(line.strip() for line in f if not line.startswith(">"))


# Backward-compatible name used by existing notebooks and scripts.
load_all_heads = load_attention_map
