"""
Generate example head heatmap and network visualizations using synthetic attention data.

Use this to produce sample outputs without running full OpenFold inference.
Reads a FASTA for sequence length/labels and writes a minimal attention file
in the same format as run_pretrained_openfold.py --demo_attn, then runs
the new visualization utilities.

Usage:
  python scripts/run_head_heatmap_network_demo.py

Outputs are written to examples/monomer/sample_attention_viz_outputs/
"""

import os
import sys

# Allow importing from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
from visualize_attention_data import load_attention_map, parse_fasta_sequence
from visualize_attention_chord_diagrams import generate_chord_diagrams
from visualize_attention_head_heatmaps import build_head_matrices, plot_head_heatmaps
from visualize_attention_networks import (
    build_aggregated_graph,
    plot_residue_network,
    plot_residue_network_per_head,
)


def write_synthetic_msa_row_attention(
    output_path: str,
    n_residues: int,
    num_heads: int = 8,
    edges_per_head: int = 80,
    seed: int = 42,
) -> None:
    """Write a minimal msa_row_attn_layer{L}.txt with synthetic (i, j, weight) lines."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for h in range(num_heads):
            f.write(f"layer 47 head {h}\n")
            for _ in range(edges_per_head):
                i = rng.integers(0, n_residues)
                j = rng.integers(0, n_residues)
                if i == j:
                    j = (j + 1) % n_residues
                w = float(rng.uniform(0.05, 0.6))
                f.write(f"{i} {j} {w}\n")
    print(f"Wrote synthetic MSA row attention: {output_path}")


def main():
    fasta_path = os.path.join(
        REPO_ROOT, "examples", "monomer", "fasta_dir_6KWC", "6KWC.fasta"
    )
    if not os.path.exists(fasta_path):
        print(f"FASTA not found: {fasta_path}")
        return 1
    residue_seq = parse_fasta_sequence(fasta_path)
    n_residues = len(residue_seq)

    out_base = os.path.join(
        REPO_ROOT, "examples", "monomer", "sample_attention_viz_outputs"
    )
    attn_dir = os.path.join(out_base, "attention_files_6KWC_demo")
    heatmap_dir = os.path.join(out_base, "head_heatmaps")
    network_dir = os.path.join(out_base, "network_plots")
    chord_dir = os.path.join(out_base, "chord_diagrams")

    layer_idx = 47
    top_k = 50
    protein = "6KWC"

    write_synthetic_msa_row_attention(
        os.path.join(attn_dir, f"msa_row_attn_layer{layer_idx}.txt"),
        n_residues=n_residues,
        num_heads=8,
        edges_per_head=100,
    )

    msa_file = os.path.join(attn_dir, f"msa_row_attn_layer{layer_idx}.txt")
    heads = load_attention_map(msa_file, top_k=top_k)
    head_mats = build_head_matrices(heads, n_residues=n_residues)
    plot_head_heatmaps(
        head_mats,
        residue_sequence=residue_seq,
        layer_idx=layer_idx,
        protein=protein,
        output_dir=heatmap_dir,
        cols=4,
        save_combined=True,
        save_individual=False,
    )
    agg_edges = build_aggregated_graph(heads, aggregation="mean", normalize_by_heads=True)
    plot_residue_network(
        agg_edges,
        n_residues=n_residues,
        residue_sequence=residue_seq,
        layer_idx=layer_idx,
        protein=protein,
        output_dir=network_dir,
        layout="circular",
        max_edges=200,
        top_k_hubs=10,
    )
    plot_residue_network_per_head(
        heads,
        n_residues=n_residues,
        residue_sequence=residue_seq,
        layer_idx=layer_idx,
        protein=protein,
        output_dir=network_dir,
        layout="circular",
        max_edges_per_head=50,
        cols=4,
    )
    generate_chord_diagrams(
        attention_dir=attn_dir,
        residue_sequence=residue_seq,
        output_dir=chord_dir,
        protein=protein,
        attention_type="msa_row",
        top_k=top_k,
        layer_idx=layer_idx,
        save_individual=True,
        save_grid=True,
        save_aggregated=True,
    )
    print(f"Example outputs saved under {out_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
