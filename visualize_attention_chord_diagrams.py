"""
Chord-style attention visualizations.

This module renders circular residue-residue attention diagrams from the shared
``heads`` structure produced by ``visualize_attention_data.load_attention_map``.
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np

from visualize_attention_data import get_attention_file_path, load_attention_map


AttentionEdge = Tuple[int, int, float]


def aggregate_chord_edges(
    heads: Dict[int, List[AttentionEdge]],
    aggregation: str = "mean",
) -> List[AttentionEdge]:
    """
    Aggregate per-head attention edges for an all-head chord diagram.

    Args:
        heads: Dict mapping head_idx -> list of (res_i, res_j, weight).
        aggregation: "mean" or "sum" over weights for each residue pair.

    Returns:
        List of aggregated (res_i, res_j, weight), sorted by descending weight.
    """
    edge_to_weights = defaultdict(list)
    for conns in heads.values():
        for res_i, res_j, weight in conns:
            edge_to_weights[(int(res_i), int(res_j))].append(float(weight))

    aggregated = []
    for (res_i, res_j), weights in edge_to_weights.items():
        if aggregation == "sum":
            value = np.sum(weights)
        else:
            value = np.mean(weights)
        aggregated.append((res_i, res_j, float(value)))

    aggregated.sort(key=lambda x: x[2], reverse=True)
    return aggregated


def _residue_positions(n_residues: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n_residues, endpoint=False) + (np.pi / 2)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _draw_chord(ax, p1, p2, weight, w_min, w_max, color):
    if w_max <= w_min:
        norm_weight = 0.5
    else:
        norm_weight = (weight - w_min) / (w_max - w_min)

    path = Path(
        [p1, (0.0, 0.0), p2],
        [Path.MOVETO, Path.CURVE3, Path.CURVE3],
    )
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=color,
        linewidth=0.4 + 3.0 * norm_weight,
        alpha=0.25 + 0.55 * norm_weight,
        zorder=1,
    )
    ax.add_patch(patch)


def plot_chord_diagram(
    connections: List[AttentionEdge],
    n_residues: int,
    residue_sequence: Optional[str],
    layer_idx: int,
    protein: str,
    output_path: str,
    head_idx: Optional[int] = None,
    title: Optional[str] = None,
    highlight_residue_index: Optional[int] = None,
    max_edges: Optional[int] = 80,
    figsize: Tuple[float, float] = (9, 9),
    show_plot: bool = False,
) -> str:
    """
    Draw one circular chord diagram for a set of attention connections.

    Args:
        connections: List of (res_i, res_j, weight) attention edges.
        n_residues: Number of residues in the sequence.
        residue_sequence: Optional sequence labels for small proteins.
        layer_idx: Attention layer index.
        protein: Protein name for plot title.
        output_path: PNG output path.
        head_idx: Optional head index for titles.
        title: Optional explicit title.
        highlight_residue_index: Optional residue index to highlight.
        max_edges: Cap edges drawn by descending weight.
        figsize: Matplotlib figure size.
        show_plot: If True, call plt.show().

    Returns:
        Path to saved PNG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if max_edges is not None:
        connections = sorted(connections, key=lambda x: x[2], reverse=True)[:max_edges]

    pos = _residue_positions(n_residues)
    fig, ax = plt.subplots(figsize=figsize)

    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", linewidth=1.0, alpha=0.7)
    ax.add_patch(circle)

    if connections:
        weights = [weight for _, _, weight in connections]
        w_min, w_max = min(weights), max(weights)
        for res_i, res_j, weight in connections:
            if 0 <= res_i < n_residues and 0 <= res_j < n_residues:
                _draw_chord(
                    ax,
                    pos[res_i],
                    pos[res_j],
                    weight,
                    w_min,
                    w_max,
                    color="royalblue",
                )

    node_colors = ["lightgray"] * n_residues
    if highlight_residue_index is not None and 0 <= highlight_residue_index < n_residues:
        node_colors[highlight_residue_index] = "coral"

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=24,
        c=node_colors,
        edgecolors="dimgray",
        linewidths=0.5,
        zorder=2,
    )

    show_labels = residue_sequence is not None and len(residue_sequence) == n_residues and n_residues <= 90
    if show_labels:
        for idx, (x, y) in enumerate(pos):
            label_x, label_y = 1.08 * x, 1.08 * y
            ax.text(
                label_x,
                label_y,
                residue_sequence[idx],
                ha="center",
                va="center",
                fontsize=5,
                color="darkred" if idx == highlight_residue_index else "black",
            )

    if title is None:
        head_label = f"Head {head_idx}" if head_idx is not None else "Aggregated Heads"
        title = f"{protein} - Layer {layer_idx} - {head_label} chord diagram"

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"[Saved] Chord diagram: {output_path}")
    return output_path


def plot_chord_diagrams_per_head(
    heads: Dict[int, List[AttentionEdge]],
    n_residues: int,
    residue_sequence: Optional[str],
    layer_idx: int,
    protein: str,
    output_dir: str,
    max_edges_per_head: int = 80,
    show_plot: bool = False,
) -> List[str]:
    """Save one chord diagram per attention head."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for head_idx, connections in sorted(heads.items()):
        output_path = os.path.join(
            output_dir,
            f"chord_head_{head_idx}_layer_{layer_idx}_{protein}.png",
        )
        saved_paths.append(
            plot_chord_diagram(
                connections,
                n_residues=n_residues,
                residue_sequence=residue_sequence,
                layer_idx=layer_idx,
                protein=protein,
                output_path=output_path,
                head_idx=head_idx,
                max_edges=max_edges_per_head,
                show_plot=show_plot,
            )
        )

    return saved_paths


def plot_chord_diagram_grid(
    heads: Dict[int, List[AttentionEdge]],
    n_residues: int,
    layer_idx: int,
    protein: str,
    output_dir: str,
    max_edges_per_head: int = 50,
    cols: int = 4,
    figsize_per_subplot: float = 3.0,
    show_plot: bool = False,
) -> List[str]:
    """Save a small-multiples grid of chord diagrams, one subplot per head."""
    os.makedirs(output_dir, exist_ok=True)
    head_indices = sorted(heads.keys())
    if not head_indices:
        return []

    rows = (len(head_indices) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_subplot, rows * figsize_per_subplot),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    pos = _residue_positions(n_residues)

    for idx, head_idx in enumerate(head_indices):
        ax = axes_flat[idx]
        conns = sorted(heads[head_idx], key=lambda x: x[2], reverse=True)[:max_edges_per_head]
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="gray", linewidth=0.7, alpha=0.6))
        if conns:
            weights = [weight for _, _, weight in conns]
            w_min, w_max = min(weights), max(weights)
            for res_i, res_j, weight in conns:
                if 0 <= res_i < n_residues and 0 <= res_j < n_residues:
                    _draw_chord(ax, pos[res_i], pos[res_j], weight, w_min, w_max, "royalblue")
        ax.scatter(pos[:, 0], pos[:, 1], s=6, c="lightgray", edgecolors="gray", linewidths=0.2, zorder=2)
        ax.set_title(f"Head {head_idx}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)

    for idx in range(len(head_indices), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"{protein} - Layer {layer_idx} - Chord diagrams per head", fontsize=12, weight="bold", y=1.02)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"chord_heads_layer_{layer_idx}_{protein}_grid.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"[Saved] Chord grid: {output_path}")
    return [output_path]


def generate_chord_diagrams(
    attention_dir: str,
    residue_sequence: str,
    output_dir: str,
    protein: str,
    attention_type: str = "msa_row",
    residue_indices: Optional[List[int]] = None,
    top_k: int = 50,
    layer_idx: int = 47,
    save_individual: bool = True,
    save_grid: bool = True,
    save_aggregated: bool = True,
) -> List[str]:
    """
    Generate chord diagrams from saved attention text files.

    For MSA row attention, this can save per-head diagrams, a per-head grid, and
    an aggregated mean diagram. For triangle-start attention, diagrams are
    generated separately for each requested residue index.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_residues = len(residue_sequence)
    saved_paths = []

    if attention_type == "msa_row":
        file_path = get_attention_file_path(attention_dir, attention_type, layer_idx)
        heads = load_attention_map(file_path, top_k=top_k)

        if save_individual:
            saved_paths.extend(
                plot_chord_diagrams_per_head(
                    heads,
                    n_residues,
                    residue_sequence,
                    layer_idx,
                    protein,
                    output_dir,
                    max_edges_per_head=top_k,
                )
            )
        if save_grid:
            saved_paths.extend(
                plot_chord_diagram_grid(
                    heads,
                    n_residues,
                    layer_idx,
                    protein,
                    output_dir,
                    max_edges_per_head=top_k,
                )
            )
        if save_aggregated:
            aggregated = aggregate_chord_edges(heads, aggregation="mean")
            output_path = os.path.join(output_dir, f"chord_aggregated_layer_{layer_idx}_{protein}.png")
            saved_paths.append(
                plot_chord_diagram(
                    aggregated,
                    n_residues,
                    residue_sequence,
                    layer_idx,
                    protein,
                    output_path,
                    title=f"{protein} - Layer {layer_idx} - Mean attention chord diagram",
                    max_edges=top_k,
                )
            )

    elif attention_type == "triangle_start":
        if residue_indices is None:
            raise ValueError("residue_indices is required for triangle_start attention")

        for res_idx in residue_indices:
            file_path = get_attention_file_path(
                attention_dir, attention_type, layer_idx, residue_idx=res_idx
            )
            if not os.path.exists(file_path):
                print(f"[Warning] Missing file for residue {res_idx}")
                continue
            heads = load_attention_map(file_path, top_k=top_k)
            for head_idx, connections in sorted(heads.items()):
                output_path = os.path.join(
                    output_dir,
                    f"chord_tri_start_res_{res_idx}_head_{head_idx}_layer_{layer_idx}_{protein}.png",
                )
                saved_paths.append(
                    plot_chord_diagram(
                        connections,
                        n_residues,
                        residue_sequence,
                        layer_idx,
                        protein,
                        output_path,
                        head_idx=head_idx,
                        highlight_residue_index=res_idx,
                        max_edges=top_k,
                    )
                )
    else:
        raise ValueError(f"Unsupported attention_type: {attention_type}")

    return saved_paths
