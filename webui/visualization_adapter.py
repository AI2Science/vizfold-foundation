"""
visualization_adapter.py

Middle-layer adapter for Issue #41.

This file converts stored offline trace data from the archive/reader layer
into visualization-ready formats for the Streamlit UI.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

Connection = Tuple[int, int, float]
AttentionMap = Dict[int, List[Connection]]


def flatten_attention_heads(attn_data: AttentionMap) -> List[Connection]:
    """
    Converts:
        {head: [(res1, res2, weight), ...]}

    into:
        [(res1, res2, weight), ...]

    This is the standardized format expected by Dev's visualization components.
    """
    connections: List[Connection] = []

    for head_connections in attn_data.values():
        connections.extend(head_connections)

    return sorted(connections, key=lambda x: x[2], reverse=True)


def dense_attention_to_connections(
    matrix: np.ndarray,
    top_k: int = 50,
    symmetric: bool = True,
) -> List[Connection]:
    """
    Converts a dense N x N attention matrix into top-k residue connections.

    This supports archive outputs that store full tensors instead of sparse files.
    """
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D attention matrix.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Attention matrix must be square.")

    n = matrix.shape[0]

    if symmetric:
        rows, cols = np.triu_indices(n, k=1)
        weights = (matrix[rows, cols] + matrix[cols, rows]) / 2.0
    else:
        rows, cols = np.where(~np.eye(n, dtype=bool))
        weights = matrix[rows, cols]

    top_indices = np.argsort(weights)[::-1][:top_k]

    return [
        (int(rows[i]), int(cols[i]), float(weights[i]))
        for i in top_indices
    ]


def residue_attention_scores(
    connections: List[Connection],
    n_residues: int,
) -> np.ndarray:
    """
    Aggregates connection weights into one score per residue.

    Used for coloring protein structure or plotting residue-level importance.
    """
    scores = np.zeros(n_residues)

    for r1, r2, weight in connections:
        if 0 <= r1 < n_residues:
            scores[r1] += weight
        if 0 <= r2 < n_residues:
            scores[r2] += weight

    return scores


def build_visualization_payload(
    fasta_seq: str,
    pdb_path: Optional[str],
    connections: List[Connection],
    layer_idx: int,
    head_label: str,
) -> dict:
    """
    Creates one clean payload Dev's UI can consume.
    """
    n_residues = len(fasta_seq)

    return {
        "fasta_seq": fasta_seq,
        "pdb_path": pdb_path,
        "connections": connections,
        "n_residues": n_residues,
        "layer_idx": layer_idx,
        "head_label": head_label,
        "residue_scores": residue_attention_scores(connections, n_residues),
    }