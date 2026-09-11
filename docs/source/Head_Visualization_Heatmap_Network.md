# Head-Level Heatmap and Network Visualizations

This document describes the new attention-head visualizations (heatmaps and network-style plots) and how they compare to the existing arc diagrams and 3D PyMOL overlays.

## Overview

Arc diagrams and PyMOL overlays are useful but show one head at a time. The new tools let you:

- **Compare all heads in one layer at once** via a grid of heatmaps or a grid of small network plots.
- **See aggregated attention** across heads as a single network, with hub residues highlighted.
- **Use the same attention text files** produced by `run_pretrained_openfold.py --demo_attn` (no change to the inference pipeline).

## New Components

| Component | Purpose |
|-----------|---------|
| `visualize_attention_data.py` | Shared attention-map parser and FASTA reader used by every visualization module. |
| `visualize_attention_head_heatmaps.py` | Builds dense per-head matrices from shared parsed attention data and plots a grid of residue–residue heatmaps (one per head). |
| `visualize_attention_networks.py` | Aggregates heads into one weighted graph and/or draws one small network per head; supports circular or linear layout and hub highlighting. |
| `visualize_attention_chord_diagrams.py` | Renders circular chord diagrams for single heads, all-head grids, and aggregated mean attention. |
| Notebook cells (in `viz_attention_demo.ipynb` and `viz_attention_demo_base.ipynb`) | After running 3D and arc visualizations, a new cell runs heatmap and network code for MSA row and (optionally) triangle-start attention. |
| `scripts/run_head_heatmap_network_demo.py` | Standalone script that generates **synthetic** attention data and runs the new visualizations to produce example outputs without full OpenFold inference. |

## How to Run

1. **From the notebook (real data)**  
   Run the usual inference cell so that `ATTN_MAP_DIR` contains files like `msa_row_attn_layer47.txt`. Then run the new cell titled *"Head-level heatmap and network-style visualizations"*. Outputs go to:
   - `IMAGE_OUTPUT_DIR/head_heatmaps/` (combined heatmap panel, optional per-head heatmaps)
   - `IMAGE_OUTPUT_DIR/network_plots/` (aggregated network, per-head network grid)

2. **Example outputs without inference**  
   From the repo root:
   ```bash
   python scripts/run_head_heatmap_network_demo.py
   ```
   This writes synthetic attention into `examples/monomer/sample_attention_viz_outputs/` and runs the same heatmap and network functions. Use it to check that the pipeline runs and to get sample figures.

## Comparing All Heads at Once

- **Heatmap grid**: One subplot per head; each shows the full residue × residue attention matrix (or top-k filled). Shared color scale across heads makes it easy to see which heads focus on similar pairs and which are sparse or different.
- **Per-head network grid**: Same idea as heatmaps but each head is shown as a 2D network (nodes = residues, edges = attention). Good for seeing structural “clusters” and long-range links per head.
- **Aggregated network**: One graph where edge weight is the mean (or sum) of attention over heads. Top-k hubs by total incident weight are highlighted, so you see which residues are attended to most across the layer.

## What These Visualizations Capture That Arc Diagrams Might Miss

- **Global head similarity**: Arc diagrams are one-head-at-a-time. Heatmaps and small-multiples networks show the whole layer in one view, so you can quickly see redundant vs. diverse heads.
- **Node-centric importance**: The aggregated network plus hub highlighting shows which residues are “important” in the sense of total incoming/outgoing attention, which is harder to read from a single-head arc.
- **Dense pattern vs. sparse pattern**: Heatmaps make it obvious when a head is diffuse (many weak links) vs. focused (few strong links), and where on the sequence those links lie (diagonal vs. off-diagonal).
- **Layer-wise comparison**: The same functions can be called for multiple layers (change `layer_idx` and the attention file path); then comparing saved heatmap/network figures across layers shows how attention evolves with depth.

## Evaluation Summary

| Visualization | Best for |
|---------------|----------|
| Arc diagram | Single-head, sequence-linear view of top-k edges; easy to match residues to sequence. |
| 3D PyMOL overlay | Same head in 3D structure context; good for spatial interpretation. |
| **Heatmap grid** | Comparing all heads in one layer; seeing dense vs. sparse and similarity across heads. |
| **Aggregated network** | Which residues are hubs across the whole layer; one picture for “consensus” attention. |
| **Per-head network grid** | Same as heatmap grid but with a network layout; can be easier for seeing clusters and long-range ties. |
| **Chord diagrams** | Circular residue-residue attention view; useful for seeing long-range links without forcing residues into a straight line. |

The new visualizations do not replace arc or 3D views; they complement them by answering “what do all heads in this layer look like together?” and “which residues matter most when we aggregate heads?”

## File and Function Reference

- **Build matrices**: `build_head_matrices(heads, n_residues)` in `visualize_attention_head_heatmaps.py`.
- **Load attention data**: `load_attention_map(connections_file, top_k=...)` in `visualize_attention_data.py`.
- **Heatmap panel**: `plot_head_heatmaps(head_mats, residue_sequence, layer_idx, protein, output_dir, ...)`.
- **Aggregated graph**: `build_aggregated_graph(heads, aggregation="mean", normalize_by_heads=True)` in `visualize_attention_networks.py`.
- **Single network plot**: `plot_residue_network(edges, n_residues, residue_sequence, layer_idx, protein, output_dir, layout="circular", max_edges=200, top_k_hubs=10)`.
- **Per-head networks**: `plot_residue_network_per_head(heads, n_residues, ...)`.
- **Chord diagrams**: `generate_chord_diagrams(attention_dir, residue_sequence, output_dir, protein, ...)` in `visualize_attention_chord_diagrams.py`.

Input `heads` is the dict returned by `load_attention_map(connections_file, top_k=...)` from `visualize_attention_data.py`. The older `load_all_heads()` import path is still available for compatibility.
