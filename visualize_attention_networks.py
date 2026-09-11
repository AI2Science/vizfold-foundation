"""
Network-style attention visualization.

Aggregates per-head attention edges into a single weighted graph and plots
2D residue networks (circular or linear layout) to highlight hub residues
and compare attention patterns across heads.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def build_aggregated_graph(
    heads: Dict[int, List[tuple]],
    aggregation: str = "mean",
    normalize_by_heads: bool = True,
) -> List[Tuple[int, int, float]]:
    """
    Aggregate all heads' edges into a single weighted graph.

    Args:
        heads: Dict mapping head_idx -> list of (res_i, res_j, weight).
        aggregation: "mean" or "sum" over heads for each (i, j).
        normalize_by_heads: If True and aggregation is "sum", divide by number
                            of heads that have that edge (so we get mean).

    Returns:
        List of (res_i, res_j, aggregated_weight), sorted by weight descending.
    """
    from collections import defaultdict

    edge_to_weights = defaultdict(list)
    for head_idx, conns in heads.items():
        for res_i, res_j, w in conns:
            key = (int(res_i), int(res_j))
            edge_to_weights[key].append(w)

    aggregated = []
    for (res_i, res_j), weights in edge_to_weights.items():
        if aggregation == "mean":
            agg_w = np.mean(weights)
        elif aggregation == "sum":
            agg_w = np.sum(weights)
            if normalize_by_heads:
                agg_w /= len(weights)
        else:
            agg_w = np.mean(weights)
        aggregated.append((res_i, res_j, float(agg_w)))

    aggregated.sort(key=lambda x: x[2], reverse=True)
    return aggregated


def _layout_circular(n_residues: int) -> np.ndarray:
    """Place n_residues nodes on a circle (radius 1)."""
    angles = np.linspace(0, 2 * np.pi, n_residues, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _layout_linear(n_residues: int) -> np.ndarray:
    """Place n_residues nodes in a line (x = 0..1, y = 0)."""
    x = np.linspace(0, 1, n_residues)
    return np.column_stack([x, np.zeros(n_residues)])


def plot_residue_network(
    edges: List[Tuple[int, int, float]],
    n_residues: int,
    residue_sequence: Optional[str],
    layer_idx: int,
    protein: str,
    output_dir: str,
    layout: str = "circular",
    threshold: Optional[float] = None,
    max_edges: Optional[int] = 200,
    top_k_hubs: int = 10,
    figsize: Tuple[float, float] = (10, 10),
    show_plot: bool = False,
) -> str:
    """
    Draw a 2D network: residues as nodes, attention as weighted edges.
    Optionally highlight top-k hub residues by total incident weight.

    Args:
        edges: List of (res_i, res_j, weight) from build_aggregated_graph().
        n_residues: Number of residues (nodes).
        residue_sequence: Optional sequence for node labels (if short).
        layer_idx: Layer index for title.
        protein: Protein name for title and filename.
        output_dir: Where to save the PNG.
        layout: "circular" or "linear".
        threshold: If set, only draw edges with weight >= threshold.
        max_edges: Cap number of edges drawn (take top by weight).
        top_k_hubs: Number of hub nodes to highlight (by total incident weight).
        figsize: Figure size in inches.
        show_plot: If True, call plt.show().

    Returns:
        Path to saved PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    if layout == "circular":
        pos = _layout_circular(n_residues)
    else:
        pos = _layout_linear(n_residues)

    # Filter and limit edges
    if threshold is not None:
        edges = [(i, j, w) for i, j, w in edges if w >= threshold]
    if max_edges is not None:
        edges = edges[:max_edges]

    if not edges:
        print("[Warning] No edges to draw for network.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"{protein} — Layer {layer_idx} — Aggregated attention (no edges)")
        path = os.path.join(output_dir, f"network_layer_{layer_idx}_{protein}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    weights = [w for _, _, w in edges]
    w_min, w_max = min(weights), max(weights)
    if w_max <= w_min:
        w_max = w_min + 1e-6

    # Hub strength: sum of incident edge weights
    hub_strength = np.zeros(n_residues)
    for res_i, res_j, w in edges:
        hub_strength[res_i] += w
        hub_strength[res_j] += w
    hub_rank = np.argsort(hub_strength)[::-1]
    hub_set = set(hub_rank[:top_k_hubs])

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges
    for res_i, res_j, weight in edges:
        xi, yi = pos[res_i]
        xj, yj = pos[res_j]
        norm_w = (weight - w_min) / (w_max - w_min)
        lw = 0.3 + 2.0 * norm_w
        alpha = 0.4 + 0.5 * norm_w
        ax.plot([xi, xj], [yi, yj], color="steelblue", linewidth=lw, alpha=alpha, zorder=1)

    # Draw nodes
    node_size = 20
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=node_size,
        c="lightgray",
        edgecolors="gray",
        linewidths=0.5,
        zorder=2,
    )
    if hub_set:
        hub_pos = pos[list(hub_set)]
        ax.scatter(
            hub_pos[:, 0],
            hub_pos[:, 1],
            s=80,
            c="coral",
            edgecolors="darkred",
            linewidths=1.5,
            zorder=3,
            label=f"Top-{top_k_hubs} hubs",
        )
        ax.legend(loc="upper right", fontsize=8)

    show_labels = residue_sequence is not None and n_residues <= 60
    if show_labels and len(residue_sequence) == n_residues:
        for i in range(n_residues):
            ax.annotate(
                residue_sequence[i],
                (pos[i, 0], pos[i, 1]),
                fontsize=4,
                ha="center",
                va="center",
            )
    else:
        ax.set_xlabel("Layout x")
        ax.set_ylabel("Layout y")

    ax.set_title(
        f"{protein} — Layer {layer_idx} — Aggregated attention network ({len(edges)} edges)"
    )
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    path = os.path.join(output_dir, f"network_layer_{layer_idx}_{protein}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()
    print(f"[Saved] Network: {path}")
    return path


def plot_residue_network_per_head(
    heads: Dict[int, List[tuple]],
    n_residues: int,
    residue_sequence: Optional[str],
    layer_idx: int,
    protein: str,
    output_dir: str,
    layout: str = "circular",
    max_edges_per_head: int = 50,
    cols: int = 4,
    figsize_per_subplot: float = 3.0,
    show_plot: bool = False,
) -> List[str]:
    """
    Draw one small network per head in a grid (small multiples).

    Args:
        heads: From load_attention_map() in visualize_attention_data.
        n_residues, residue_sequence, layer_idx, protein, output_dir: As in plot_residue_network.
        layout: "circular" or "linear".
        max_edges_per_head: Max edges to draw per head.
        cols: Grid columns.
        figsize_per_subplot: Inches per subplot.
        show_plot: If True, plt.show().

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    head_indices = sorted(heads.keys())
    if not head_indices:
        return []

    if layout == "circular":
        pos = _layout_circular(n_residues)
    else:
        pos = _layout_linear(n_residues)

    n_heads = len(head_indices)
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_subplot, rows * figsize_per_subplot),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, head_idx in enumerate(head_indices):
        ax = axes_flat[idx]
        conns = heads[head_idx][:max_edges_per_head]
        if not conns:
            ax.set_title(f"Head {head_idx}")
            ax.axis("off")
            continue
        w_min = min(w for _, _, w in conns)
        w_max = max(w for _, _, w in conns)
        if w_max <= w_min:
            w_max = w_min + 1e-6
        for res_i, res_j, weight in conns:
            xi, yi = pos[res_i]
            xj, yj = pos[res_j]
            norm_w = (weight - w_min) / (w_max - w_min)
            lw = 0.2 + 1.2 * norm_w
            alpha = 0.3 + 0.5 * norm_w
            ax.plot([xi, xj], [yi, yj], color="steelblue", linewidth=lw, alpha=alpha)
        ax.scatter(
            pos[:, 0], pos[:, 1], s=8, c="lightgray", edgecolors="gray", linewidths=0.3
        )
        ax.set_title(f"Head {head_idx}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")

    for idx in range(n_heads, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"{protein} — Layer {layer_idx} — Per-head networks",
        fontsize=12,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"network_per_head_layer_{layer_idx}_{protein}.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()
    print(f"[Saved] Per-head networks: {path}")
    return [path]
