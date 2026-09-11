"""
Head-level attention heatmap visualization.

Builds dense per-head attention matrices from the same text files used by
arc diagrams and 3D overlays, and plots them as a grid of heatmaps for
comparing all heads in one layer at once.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def build_head_matrices(
    heads: Dict[int, List[tuple]],
    n_residues: Optional[int] = None,
    symmetrize: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Convert sparse per-head connection lists into dense residue x residue matrices.

    Args:
        heads: Dict mapping head_idx -> list of (res_i, res_j, weight).
               From load_attention_map() in visualize_attention_data.
        n_residues: Number of residues (sequence length). If None, inferred from
                    max index appearing in any head.
        symmetrize: If True, set A[i,j] = A[j,i] = max(weight(i,j), weight(j,i)).

    Returns:
        Dict mapping head_idx -> 2D array of shape (n_residues, n_residues).
    """
    if n_residues is None:
        n_residues = 0
        for conns in heads.values():
            for res1, res2, _ in conns:
                n_residues = max(n_residues, res1 + 1, res2 + 1)
        if n_residues == 0:
            return {}

    out = {}
    for head_idx, conns in heads.items():
        A = np.zeros((n_residues, n_residues), dtype=np.float64)
        for res_i, res_j, weight in conns:
            i, j = int(res_i), int(res_j)
            if 0 <= i < n_residues and 0 <= j < n_residues:
                if symmetrize:
                    A[i, j] = max(A[i, j], weight)
                    A[j, i] = max(A[j, i], weight)
                else:
                    A[i, j] = weight
        out[head_idx] = A
    return out


def plot_head_heatmaps(
    head_mats: Dict[int, np.ndarray],
    residue_sequence: Optional[str],
    layer_idx: int,
    protein: str,
    output_dir: str,
    cols: int = 4,
    cmap: str = "viridis",
    mask_zeros: bool = True,
    save_combined: bool = True,
    save_individual: bool = False,
    figsize_per_subplot: float = 3.0,
    show_plot: bool = False,
) -> List[str]:
    """
    Plot a grid of heatmaps, one per head, for comparing all heads in a layer.

    Args:
        head_mats: From build_head_matrices().
        residue_sequence: Optional sequence string for axis labels (e.g. from parse_fasta_sequence).
        layer_idx: Layer index (for titles and filenames).
        protein: Protein name (for filenames).
        output_dir: Directory to save PNGs.
        cols: Number of columns in the grid.
        cmap: Matplotlib colormap name.
        mask_zeros: If True, plot zeros as transparent / masked so pattern is clearer.
        save_combined: Save one combined multi-head panel.
        save_individual: If True, also save one PNG per head.
        figsize_per_subplot: Size per subplot in inches.
        show_plot: If True, call plt.show().

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    head_indices = sorted(head_mats.keys())
    if not head_indices:
        print("[Warning] No head matrices to plot.")
        return saved_paths

    n_heads = len(head_indices)
    n_res = next(iter(head_mats.values())).shape[0]

    # Optional: show residue labels only for small sequences
    show_residue_labels = residue_sequence is not None and n_res <= 80
    if show_residue_labels and len(residue_sequence) != n_res:
        show_residue_labels = False

    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * figsize_per_subplot, rows * figsize_per_subplot),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    vmin = min(np.min(A) for A in head_mats.values() if A.size > 0)
    vmax = max(np.max(A) for A in head_mats.values() if A.size > 0)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    for idx, head_idx in enumerate(head_indices):
        ax = axes_flat[idx]
        A = head_mats[head_idx]
        plot_mat = np.ma.masked_where(A == 0, A) if mask_zeros else A
        im = ax.imshow(
            plot_mat,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"Head {head_idx}", fontsize=10)
        if show_residue_labels:
            ax.set_xticks(np.arange(n_res))
            ax.set_xticklabels(list(residue_sequence), fontsize=5, rotation=90)
            ax.set_yticks(np.arange(n_res))
            ax.set_yticklabels(list(residue_sequence), fontsize=5)
        else:
            ax.set_xlabel("Residue j")
            ax.set_ylabel("Residue i")
        ax.tick_params(axis="both", labelsize=6)

    for idx in range(n_heads, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"{protein} — Layer {layer_idx} — All heads (residue–residue attention)",
        fontsize=12,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()

    combined_path = os.path.join(
        output_dir, f"head_heatmaps_layer_{layer_idx}_{protein}.png"
    )
    if save_combined:
        plt.savefig(combined_path, dpi=150, bbox_inches="tight")
        saved_paths.append(combined_path)
        print(f"[Saved] Combined heatmaps: {combined_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    if save_individual:
        for head_idx in head_indices:
            fig1, ax1 = plt.subplots(figsize=(6, 5))
            A = head_mats[head_idx]
            plot_mat = np.ma.masked_where(A == 0, A) if mask_zeros else A
            ax1.imshow(
                plot_mat,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax1.set_title(f"{protein} — Layer {layer_idx} — Head {head_idx}")
            ax1.set_xlabel("Residue j")
            ax1.set_ylabel("Residue i")
            path = os.path.join(
                output_dir, f"head_heatmap_layer_{layer_idx}_head_{head_idx}_{protein}.png"
            )
            plt.savefig(path, dpi=150, bbox_inches="tight")
            saved_paths.append(path)
            plt.close()
        print(f"[Saved] {n_heads} individual heatmap(s) to {output_dir}")

    return saved_paths
