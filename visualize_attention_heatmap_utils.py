"""
Utilities for generating attention heatmap visualizations.

This module consolidates the heatmap functionality from the heatmap PRs:

1. Static Matplotlib PNG heatmaps for all heads.
2. Static Matplotlib PNG combined heatmaps for MSA Row + Triangle Start.
3. Interactive Plotly HTML heatmap grids.

The heatmaps operate on exported attention text files and complement the
existing arc diagram / PyMOL visualization workflow. This module does not
import or invoke PyMOL.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualize_attention_arc_diagram_demo_utils import load_all_heads


# =============================================================================
# Shared helpers
# =============================================================================


def reconstruct_attention_matrix(connections, seq_length):
    """
    Reconstruct a full attention matrix from sparse top-K connections.
    """
    matrix = np.zeros((seq_length, seq_length))

    for res1, res2, weight in connections:
        if 0 <= res1 < seq_length and 0 <= res2 < seq_length:
            matrix[res1, res2] = weight

    return matrix


def get_sequence_length_from_fasta(fasta_path):
    """
    Get sequence length from FASTA file.
    """
    with open(fasta_path, "r") as f:
        lines = f.readlines()

    seq_lines = [line.strip() for line in lines if not line.startswith(">")]
    sequence = "".join(seq_lines)
    return len(sequence)


def _resolve_seq_length(seq_length=None, fasta_path=None):
    """
    Resolve sequence length from an explicit value or FASTA path.

    This intentionally avoids a hard-coded fallback length, because using the
    wrong sequence length can produce misleading heatmaps.
    """
    if seq_length is not None:
        return seq_length

    if fasta_path and os.path.exists(fasta_path):
        return get_sequence_length_from_fasta(fasta_path)

    raise ValueError("seq_length is required when fasta_path is not provided or does not exist")


def _get_grid_shape(num_heads):
    """
    Choose subplot grid dimensions.
    """
    if num_heads <= 4:
        return 1, num_heads
    if num_heads <= 8:
        return 2, 4
    if num_heads <= 12:
        return 3, 4
    if num_heads <= 16:
        return 4, 4

    cols = min(4, int(np.ceil(np.sqrt(num_heads))))
    rows = (num_heads + cols - 1) // cols
    return rows, cols


def _normalize_matrix(matrix, normalization, global_min=None, global_max=None):
    """
    Normalize an attention matrix using global or per-head normalization.
    """
    if normalization == "global":
        min_val = global_min
        max_val = global_max
    elif normalization == "per_head":
        min_val = np.nanmin(matrix)
        max_val = np.nanmax(matrix)
    else:
        raise ValueError(f"Invalid normalization: {normalization}")

    if max_val > min_val:
        return (matrix - min_val) / (max_val - min_val)

    return matrix


def _get_attention_file(attention_dir, attention_type, layer_idx, residue_idx=None):
    """
    Build the attention file path for a given attention type.
    """
    if attention_type == "msa_row":
        return os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")

    if attention_type == "triangle_start":
        if residue_idx is None:
            raise ValueError("residue_idx is required for triangle_start attention")

        return os.path.join(
            attention_dir,
            f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_idx}.txt",
        )

    raise ValueError(f"Unknown attention_type: {attention_type}")


def _load_heads_for_attention_type(attention_dir, attention_type, layer_idx, residue_indices=None):
    """
    Load heads for msa_row or triangle_start attention.

    For triangle_start, residue_indices is a list of candidate residue indices.
    The first existing file with valid data is used, matching the behavior from
    the second PR.
    """
    if attention_type == "msa_row":
        file_path = _get_attention_file(attention_dir, attention_type, layer_idx)

        if not os.path.exists(file_path):
            print(f"[Warning] Missing file: {file_path}")
            return {}, None

        return load_all_heads(file_path, top_k=None), None

    if attention_type == "triangle_start":
        if residue_indices is None:
            raise ValueError("residue_indices required for triangle_start attention")

        for residue_idx in residue_indices:
            file_path = _get_attention_file(
                attention_dir,
                attention_type,
                layer_idx,
                residue_idx=residue_idx,
            )

            if not os.path.exists(file_path):
                print(f"[Warning] Missing file for residue {residue_idx}: {file_path}")
                continue

            heads = load_all_heads(file_path, top_k=None)

            if heads:
                return heads, residue_idx

            print(f"[Warning] No attention data found in {file_path}")

        print(f"[Warning] No valid attention data found for residues {residue_indices}")
        return {}, None

    raise ValueError(f"Invalid attention_type: {attention_type}")


def _build_attention_matrices(heads, seq_length):
    """
    Convert loaded attention heads into dense matrices.
    """
    attention_matrices = {}

    for head_idx, connections in heads.items():
        if not connections:
            continue

        matrix = reconstruct_attention_matrix(connections, seq_length)
        attention_matrices[head_idx] = matrix

    return attention_matrices


# =============================================================================
# Static Matplotlib PNG heatmaps from PR 2
# =============================================================================


def plot_all_heads_heatmap(
    attention_dir,
    output_dir,
    protein,
    attention_type="msa_row",
    layer_idx=47,
    seq_length=None,
    fasta_path=None,
    normalization="global",
    colormap="viridis",
    figsize_per_head=(2.0, 2.0),
    dpi=300,
    save_to_png=True,
    residue_indices=None,
):
    """
    Generate heatmap grid showing all attention heads for a layer.

    Args:
        attention_dir: Directory containing attention text files.
        output_dir: Directory to save output PNG.
        protein: Protein identifier, e.g. "6KWC".
        attention_type: "msa_row" or "triangle_start".
        layer_idx: Layer number to visualize.
        seq_length: Sequence length. If None, infer from fasta_path.
        fasta_path: Path to FASTA file for sequence length detection.
        normalization: "global" or "per_head".
        colormap: Matplotlib colormap name.
        figsize_per_head: Size of each subplot, in inches.
        dpi: Output resolution.
        save_to_png: Whether to save to PNG file.
        residue_indices: List of residue indices for triangle_start.

    Returns:
        Output path if save_to_png=True, otherwise the Matplotlib figure.
        Returns None if no valid heads are found.
    """
    if attention_type not in ["msa_row", "triangle_start"]:
        raise ValueError(f"Invalid attention_type: {attention_type}")

    if normalization not in ["global", "per_head"]:
        raise ValueError(f"Invalid normalization: {normalization}")

    if attention_type == "triangle_start" and residue_indices is None:
        raise ValueError("residue_indices required for triangle_start attention")

    seq_length = _resolve_seq_length(seq_length, fasta_path)

    heads, selected_residue_idx = _load_heads_for_attention_type(
        attention_dir=attention_dir,
        attention_type=attention_type,
        layer_idx=layer_idx,
        residue_indices=residue_indices,
    )

    attention_matrices = _build_attention_matrices(heads, seq_length)
    num_heads = len(attention_matrices)

    if num_heads == 0:
        print("[Warning] No valid attention heads to visualize")
        return None

    rows, cols = _get_grid_shape(num_heads)

    fig_width = cols * figsize_per_head[0]
    fig_height = rows * figsize_per_head[1]

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if num_heads == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    all_values = []
    for matrix in attention_matrices.values():
        all_values.extend(matrix.flatten())

    all_values = np.array(all_values)
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    for i, (head_idx, matrix) in enumerate(sorted(attention_matrices.items())):
        ax = axes[i]

        normalized_matrix = _normalize_matrix(
            matrix=matrix,
            normalization=normalization,
            global_min=global_min,
            global_max=global_max,
        )

        im = ax.imshow(
            normalized_matrix,
            cmap=colormap,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(f"Head {head_idx}", fontsize=10, weight="bold")
        ax.set_xlabel("Residue Position", fontsize=8)
        ax.set_ylabel("Residue Position", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=6)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

    for i in range(num_heads, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if rows == 1:
        plt.subplots_adjust(top=0.75)
        title_y = 0.95
    elif rows <= 2:
        plt.subplots_adjust(top=0.85)
        title_y = 0.98
    else:
        plt.subplots_adjust(top=0.90)
        title_y = 0.99

    title = f"{protein} {attention_type.replace('_', ' ').title()} Attention - Layer {layer_idx}"
    if selected_residue_idx is not None:
        title += f" - Residue {selected_residue_idx}"

    fig.suptitle(title, fontsize=14, weight="bold", y=title_y)

    if save_to_png:
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"{attention_type}_heatmap_layer_{layer_idx}_{protein}"
        if selected_residue_idx is not None:
            output_filename += f"_residue_{selected_residue_idx}"
        output_filename += ".png"

        output_path = os.path.join(output_dir, output_filename)

        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"[Saved] Heatmap visualization to {output_path}")
        return output_path

    return fig


def plot_combined_attention_heatmap(
    attention_dir,
    output_dir,
    protein,
    layer_idx=47,
    seq_length=None,
    fasta_path=None,
    normalization="global",
    colormap="viridis",
    figsize_per_head=(1.5, 1.5),
    dpi=300,
    save_to_png=True,
    residue_indices=None,
):
    """
    Generate combined heatmap showing both MSA Row and Triangle Start attention.

    Args:
        attention_dir: Directory containing attention text files.
        output_dir: Directory to save output PNG.
        protein: Protein identifier, e.g. "6KWC".
        layer_idx: Layer number to visualize.
        seq_length: Sequence length. If None, infer from fasta_path.
        fasta_path: Path to FASTA file for sequence length detection.
        normalization: "global" or "per_head".
        colormap: Matplotlib colormap name.
        figsize_per_head: Size of each subplot, in inches.
        dpi: Output resolution.
        save_to_png: Whether to save to PNG file.
        residue_indices: List of residue indices for triangle_start.

    Returns:
        Output path if save_to_png=True, otherwise the Matplotlib figure.
        Returns None if no valid heads are found.
    """
    if normalization not in ["global", "per_head"]:
        raise ValueError(f"Invalid normalization: {normalization}")

    seq_length = _resolve_seq_length(seq_length, fasta_path)

    msa_heads, _ = _load_heads_for_attention_type(
        attention_dir=attention_dir,
        attention_type="msa_row",
        layer_idx=layer_idx,
    )

    if residue_indices is None:
        residue_indices = [18]

    tri_heads, selected_residue_idx = _load_heads_for_attention_type(
        attention_dir=attention_dir,
        attention_type="triangle_start",
        layer_idx=layer_idx,
        residue_indices=residue_indices,
    )

    msa_matrices = _build_attention_matrices(msa_heads, seq_length)
    tri_matrices = _build_attention_matrices(tri_heads, seq_length)

    plot_items = []

    for head_idx, matrix in sorted(msa_matrices.items()):
        plot_items.append(("MSA", head_idx, matrix))

    for head_idx, matrix in sorted(tri_matrices.items()):
        plot_items.append(("Tri", head_idx, matrix))

    total_heads = len(plot_items)

    if total_heads == 0:
        print("[Warning] No valid attention heads to visualize")
        return None

    rows, cols = _get_grid_shape(total_heads)

    fig_width = cols * figsize_per_head[0]
    fig_height = rows * figsize_per_head[1]

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    if total_heads == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    all_values = []
    for _, _, matrix in plot_items:
        all_values.extend(matrix.flatten())

    all_values = np.array(all_values)
    global_min = np.min(all_values)
    global_max = np.max(all_values)

    for i, (label, head_idx, matrix) in enumerate(plot_items):
        ax = axes[i]

        normalized_matrix = _normalize_matrix(
            matrix=matrix,
            normalization=normalization,
            global_min=global_min,
            global_max=global_max,
        )

        im = ax.imshow(
            normalized_matrix,
            cmap=colormap,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_title(f"{label} Head {head_idx}", fontsize=10, weight="bold")
        ax.set_xlabel("Residue", fontsize=8)
        ax.set_ylabel("Residue", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=6)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

    for i in range(total_heads, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if rows == 1:
        plt.subplots_adjust(top=0.75)
        title_y = 0.95
    elif rows <= 2:
        plt.subplots_adjust(top=0.85)
        title_y = 0.98
    else:
        plt.subplots_adjust(top=0.90)
        title_y = 0.99

    title = f"{protein} Combined Attention Heatmaps - Layer {layer_idx}"
    if selected_residue_idx is not None:
        title += f" - Triangle Residue {selected_residue_idx}"

    fig.suptitle(title, fontsize=14, weight="bold", y=title_y)

    if save_to_png:
        os.makedirs(output_dir, exist_ok=True)

        output_filename = f"combined_attention_heatmap_layer_{layer_idx}_{protein}"
        if selected_residue_idx is not None:
            output_filename += f"_triangle_residue_{selected_residue_idx}"
        output_filename += ".png"

        output_path = os.path.join(output_dir, output_filename)

        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"[Saved] Combined heatmap visualization to {output_path}")
        return output_path

    return fig


# =============================================================================
# Interactive Plotly HTML heatmaps from PR 1
# =============================================================================


def create_heatmap_grid(
    attention_file,
    seq_len,
    layer_idx=47,
    attention_type="msa_row",
    output_html="heatmap_grid.html",
    threshold=None,
):
    """
    Create an interactive Plotly heatmap grid for all heads in one attention file.

    This preserves the first PR's HTML heatmap behavior.
    """
    heads = load_all_heads(attention_file, top_k=None)
    num_heads = len(heads)

    if num_heads == 0:
        print(f"No heads found in {attention_file}")
        return None

    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    all_weights = [
        weight
        for head_idx in sorted(heads.keys())
        for _, _, weight in heads[head_idx]
    ]

    if threshold is not None:
        all_weights = [weight for weight in all_weights if weight >= threshold]

    global_min = min(all_weights) if all_weights else 0
    global_max = max(all_weights) if all_weights else 1

    per_head_mins = []
    per_head_maxs = []

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Head {i}" for i in sorted(heads.keys())],
        horizontal_spacing=0.05,
        vertical_spacing=0.15,
    )

    for idx, head_idx in enumerate(sorted(heads.keys())):
        row = idx // cols + 1
        col = idx % cols + 1

        matrix = reconstruct_attention_matrix(heads[head_idx], seq_len)

        if threshold is not None:
            matrix[matrix < threshold] = np.nan

        head_weights = [weight for _, _, weight in heads[head_idx]]

        if threshold is not None:
            head_weights = [weight for weight in head_weights if weight >= threshold]

        head_min = min(head_weights) if head_weights else 0
        head_max = max(head_weights) if head_weights else 1

        per_head_mins.append(head_min)
        per_head_maxs.append(head_max)

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale="Blues",
                zmin=global_min,
                zmax=global_max,
                showscale=(idx == 0),
                colorbar=dict(x=1.02, len=0.3, title="Weight") if idx == 0 else None,
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            title_text="Residue",
            row=row,
            col=col,
            showticklabels=False,
        )
        fig.update_yaxes(
            title_text="Residue",
            row=row,
            col=col,
            showticklabels=False,
        )

    title_text = f"{attention_type.upper()} Layer {layer_idx} - All Heads"
    if threshold is not None:
        title_text += f" (Threshold > {threshold})"

    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        height=350 * rows,
        width=1200,
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.6,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label="Global Norm",
                        method="restyle",
                        args=[
                            {
                                "zmin": [global_min],
                                "zmax": [global_max],
                                "showscale": [True] + [False] * (num_heads - 1),
                            }
                        ],
                    ),
                    dict(
                        label="Per-Head Norm",
                        method="restyle",
                        args=[
                            {
                                "zmin": per_head_mins,
                                "zmax": per_head_maxs,
                                "showscale": [False] * num_heads,
                            }
                        ],
                    ),
                ],
            )
        ],
    )

    fig.write_html(output_html)
    print(f"Saved: {output_html}")

    return fig


def visualize_layer_attention(
    attention_dir,
    seq_len,
    layer_idx=47,
    attention_type="msa_row",
    residue_idx=None,
    output_dir="./outputs/attention_heatmaps",
    threshold=None,
):
    """
    Visualize layer-specific attention as an interactive Plotly heatmap grid.

    This preserves the first PR's notebook-facing helper.
    """
    os.makedirs(output_dir, exist_ok=True)

    if attention_type == "msa_row":
        attention_file = os.path.join(attention_dir, f"msa_row_attn_layer{layer_idx}.txt")
        output_html = os.path.join(output_dir, f"msa_row_layer{layer_idx}_heatmap_grid.html")

    elif attention_type == "triangle_start":
        if residue_idx is None:
            raise ValueError("residue_idx required for triangle_start")

        attention_file = os.path.join(
            attention_dir,
            f"triangle_start_attn_layer{layer_idx}_residue_idx_{residue_idx}.txt",
        )
        output_html = os.path.join(
            output_dir,
            f"triangle_start_layer{layer_idx}_res{residue_idx}_heatmap_grid.html",
        )

    else:
        raise ValueError(f"Unknown attention_type: {attention_type}")

    if not os.path.exists(attention_file):
        print(f"File not found: {attention_file}")
        return None

    print(f"Processing: {attention_file}")

    return create_heatmap_grid(
        attention_file=attention_file,
        seq_len=seq_len,
        layer_idx=layer_idx,
        attention_type=attention_type,
        output_html=output_html,
        threshold=threshold,
    )
