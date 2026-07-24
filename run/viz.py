#!/usr/bin/env python3
"""Render attention visualizations for a completed fold.

Produces 3D PyMOL head renders, arc diagrams, and combined panels from the attention
maps and relaxed PDB a fold left behind. Running the fold is run/fold.sh's job; this
only consumes its outputs. Lifted from the visualization half of closed PR #67 (the
inference half duplicated run/fold.sh and was dropped).

Example:
  python run/viz.py \\
      --fasta-file examples/monomer/fasta_dir_6KWC/6KWC.fasta \\
      --output-dir "$OPENFOLD_PREFIX/outputs/6KWC_1" \\
      --protein 6KWC
"""
import argparse
import logging
import os
import sys

# run/ sits one level below the repo root that holds the visualize_* modules.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def derived_paths(output_dir, protein, tri_residue_idx, config_preset, attn_map_dir=None):
    """Output layout produced by a --demo_attn fold (attention_files_/images_ tags,
    predictions/<tag>_relaxed.pdb)."""
    tag = f"{protein}_demo_tri_{tri_residue_idx}"
    attn_map_dir = attn_map_dir or os.path.join(output_dir, f"attention_files_{tag}")
    image_dir = os.path.join(output_dir, f"attention_images_{tag}")
    pdb_file = os.path.join(output_dir, "predictions", f"{protein}_1_{config_preset}_relaxed.pdb")
    return attn_map_dir, image_dir, pdb_file


def run_visualizations(fasta_file, protein, attn_map_dir, image_dir, pdb_file,
                       layer_idx, top_k, tri_residue_idx):
    from visualize_attention_general_utils import (
        render_pdb_to_image,
        generate_combined_attention_panels,
    )
    from visualize_attention_3d_demo_utils import plot_pymol_attention_heads
    from visualize_attention_arc_diagram_demo_utils import (
        generate_arc_diagrams,
        parse_fasta_sequence,
    )

    residue_sequence = parse_fasta_sequence(fasta_file)
    msa_dir = os.path.join(image_dir, "msa_row_attention_plots")
    tri_dir = os.path.join(image_dir, "tri_start_attention_plots")
    combined_dir = os.path.join(image_dir, "combined")

    logging.info("rendering predicted structure...")
    render_pdb_to_image(pdb_file, image_dir, f"predicted_structure_{protein}_tri_{tri_residue_idx}.png")

    logging.info("rendering 3D attention heads...")
    plot_pymol_attention_heads(pdb_file=pdb_file, attention_dir=attn_map_dir, output_dir=msa_dir,
                               protein=protein, attention_type="msa_row",
                               top_k=top_k, layer_idx=layer_idx)
    plot_pymol_attention_heads(pdb_file=pdb_file, attention_dir=attn_map_dir, output_dir=tri_dir,
                               protein=protein, attention_type="triangle_start",
                               residue_indices=[tri_residue_idx], top_k=top_k, layer_idx=layer_idx)

    logging.info("rendering arc diagrams...")
    generate_arc_diagrams(attention_dir=attn_map_dir, residue_sequence=residue_sequence,
                          output_dir=msa_dir, protein=protein, attention_type="msa_row",
                          top_k=top_k, layer_idx=layer_idx)
    generate_arc_diagrams(attention_dir=attn_map_dir, residue_sequence=residue_sequence,
                          output_dir=tri_dir, protein=protein, attention_type="triangle_start",
                          residue_indices=[tri_residue_idx], top_k=top_k, layer_idx=layer_idx)

    logging.info("assembling combined panels...")
    generate_combined_attention_panels(attention_type="msa_row", protein=protein, layer_idx=layer_idx,
                                       output_dir_3d=msa_dir, output_dir_arc=msa_dir,
                                       combined_output_dir=combined_dir)
    generate_combined_attention_panels(attention_type="triangle_start", protein=protein, layer_idx=layer_idx,
                                       output_dir_3d=tri_dir, output_dir_arc=tri_dir,
                                       combined_output_dir=combined_dir, residue_indices=[tri_residue_idx])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta-file", required=True, help="Single-sequence FASTA used for the fold.")
    parser.add_argument("--output-dir", required=True,
                        help="The fold's output directory (holds predictions/ and the attention maps).")
    parser.add_argument("--protein", required=True, help="Protein tag used to name outputs (e.g. 6KWC).")
    parser.add_argument("--tri-residue-idx", type=int, default=18,
                        help="Residue index for triangle-start focus (default 18).")
    parser.add_argument("--layer-idx", type=int, default=47, help="Model layer to visualize (default 47).")
    parser.add_argument("--top-k", type=int, default=50, help="Max attention edges per head (default 50).")
    parser.add_argument("--config-preset", default="model_1_ptm",
                        help="Config preset used for the fold (names the relaxed PDB).")
    parser.add_argument("--attn-map-dir", default=None,
                        help="Override the attention-map directory (auto-derived otherwise).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")

    attn_map_dir, image_dir, pdb_file = derived_paths(
        args.output_dir, args.protein, args.tri_residue_idx, args.config_preset, args.attn_map_dir)

    for label, path in [("fasta file", args.fasta_file), ("attention map directory", attn_map_dir),
                        ("relaxed PDB", pdb_file)]:
        if not os.path.exists(path):
            parser.error(f"{label} not found: {path} (run the fold with --demo_attn first)")

    logging.info(f"protein:      {args.protein}")
    logging.info(f"attn_map_dir: {attn_map_dir}")
    logging.info(f"image_dir:    {image_dir}")
    run_visualizations(args.fasta_file, args.protein, attn_map_dir, image_dir, pdb_file,
                       args.layer_idx, args.top_k, args.tri_residue_idx)
    logging.info("done.")


if __name__ == "__main__":
    main()
