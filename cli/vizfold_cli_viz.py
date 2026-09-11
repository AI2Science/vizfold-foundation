import argparse
import json
import logging
import os
import subprocess
import sys

EXIT_SUCCESS = 0
EXIT_PARTIAL = 1


def derived_paths(args):
    tag = f"{args.protein}_demo_tri_{args.tri_residue_idx}"
    attn_map_dir = args.attn_map_dir or os.path.join(args.output_dir, f"attention_files_{tag}")
    image_dir = os.path.join(args.output_dir, f"attention_images_{tag}")
    pdb_file = os.path.join(
        args.output_dir, "predictions",
        f"{args.protein}_1_{args.config_preset}_relaxed.pdb"
    )
    return attn_map_dir, image_dir, pdb_file


def validate_args(args, parser):
    errors = []

    if not os.path.isfile(args.fasta_file):
        errors.append(f"--fasta_file not found: {args.fasta_file}")

    if not args.visualize_only and not args.base_data_dir:
        errors.append("--base_data_dir is required unless --visualize_only is set")

    if not args.visualize_only and args.base_data_dir and not os.path.isdir(args.base_data_dir):
        errors.append(f"--base_data_dir does not exist: {args.base_data_dir}")

    if args.alignment_dir and not os.path.isdir(args.alignment_dir):
        errors.append(f"--alignment_dir does not exist: {args.alignment_dir}")

    if args.visualize_only:
        attn_map_dir, _, pdb_file = derived_paths(args)
        if not os.path.isdir(attn_map_dir):
            errors.append(f"attention map directory not found (needed for --visualize_only): {attn_map_dir}")
        if not os.path.isfile(pdb_file):
            errors.append(f"PDB file not found (needed for --visualize_only): {pdb_file}")

    if errors:
        parser.error("\n  ".join(errors))


def save_manifest(args, attn_map_dir, image_dir):
    manifest = {
        "protein": args.protein,
        "fasta_file": args.fasta_file,
        "output_dir": args.output_dir,
        "attn_map_dir": attn_map_dir,
        "image_dir": image_dir,
        "tri_residue_idx": args.tri_residue_idx,
        "layer_idx": args.layer_idx,
        "top_k": args.top_k,
        "config_preset": args.config_preset,
        "model_device": args.model_device,
        "visualize_only": args.visualize_only,
    }
    path = os.path.join(args.output_dir, "vizfold_viz_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def _find_python():
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    candidate = os.path.join(conda_prefix, "bin", "python")
    if conda_prefix and os.path.isfile(candidate):
        return candidate
    fallback = os.path.expanduser("~/scratch/.conda/envs/openfold_env/bin/python")
    if os.path.isfile(fallback):
        return fallback
    return sys.executable


def run_inference(args, attn_map_dir):
    fasta_dir = os.path.dirname(os.path.abspath(args.fasta_file))
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    inference_script = os.path.abspath(os.path.join(repo_root, "run_pretrained_openfold.py"))
    mmcif_dir = os.path.join(args.base_data_dir, "pdb_mmcif", "mmcif_files")
    python = _find_python()
    logging.info(f"using python: {python}")

    cmd = [
        python, inference_script,
        fasta_dir,
        mmcif_dir,
        "--output_dir", args.output_dir,
        "--config_preset", args.config_preset,
        "--uniref90_database_path", os.path.join(args.base_data_dir, "uniref90", "uniref90.fasta"),
        "--mgnify_database_path", os.path.join(args.base_data_dir, "mgnify", "mgy_clusters_2022_05.fa"),
        "--pdb70_database_path", os.path.join(args.base_data_dir, "pdb70", "pdb70"),
        "--uniclust30_database_path", os.path.join(args.base_data_dir, "uniclust30", "uniclust30_2018_08"),
        "--bfd_database_path", os.path.join(args.base_data_dir, "bfd", "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"),
        "--save_outputs",
        "--model_device", args.model_device,
        "--attn_map_dir", attn_map_dir,
        "--num_recycles_save", str(args.num_recycles_save),
        "--triangle_residue_idx", str(args.tri_residue_idx),
        "--demo_attn",
    ]

    if args.alignment_dir:
        cmd += ["--use_precomputed_alignments", args.alignment_dir]

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    logging.info("running inference...")
    subprocess.run(cmd, check=True, cwd=os.path.abspath(repo_root), env=env)
    logging.info("inference done")


def run_visualizations(args, attn_map_dir, image_dir, pdb_file):
    from visualize_attention_general_utils import render_pdb_to_image, generate_combined_attention_panels
    from visualize_attention_3d_demo_utils import plot_pymol_attention_heads
    from visualize_attention_arc_diagram_demo_utils import generate_arc_diagrams, parse_fasta_sequence

    residue_sequence = parse_fasta_sequence(args.fasta_file)

    msa_3d_dir = os.path.join(image_dir, "msa_row_attention_plots")
    tri_3d_dir = os.path.join(image_dir, "tri_start_attention_plots")
    msa_arc_dir = msa_3d_dir
    tri_arc_dir = tri_3d_dir
    combined_dir = os.path.join(image_dir, "combined")

    logging.info("rendering PDB structure...")
    struct_fname = f"predicted_structure_{args.protein}_tri_{args.tri_residue_idx}.png"
    render_pdb_to_image(pdb_file, image_dir, struct_fname)

    logging.info("generating 3D attention visualizations...")
    plot_pymol_attention_heads(
        pdb_file=pdb_file,
        attention_dir=attn_map_dir,
        output_dir=msa_3d_dir,
        protein=args.protein,
        attention_type="msa_row",
        top_k=args.top_k,
        layer_idx=args.layer_idx,
    )
    plot_pymol_attention_heads(
        pdb_file=pdb_file,
        attention_dir=attn_map_dir,
        output_dir=tri_3d_dir,
        protein=args.protein,
        attention_type="triangle_start",
        residue_indices=[args.tri_residue_idx],
        top_k=args.top_k,
        layer_idx=args.layer_idx,
    )

    logging.info("generating arc diagram visualizations...")
    generate_arc_diagrams(
        attention_dir=attn_map_dir,
        residue_sequence=residue_sequence,
        output_dir=msa_arc_dir,
        protein=args.protein,
        attention_type="msa_row",
        top_k=args.top_k,
        layer_idx=args.layer_idx,
    )
    generate_arc_diagrams(
        attention_dir=attn_map_dir,
        residue_sequence=residue_sequence,
        output_dir=tri_arc_dir,
        protein=args.protein,
        attention_type="triangle_start",
        residue_indices=[args.tri_residue_idx],
        top_k=args.top_k,
        layer_idx=args.layer_idx,
    )

    logging.info("generating combined panels...")
    generate_combined_attention_panels(
        attention_type="msa_row",
        protein=args.protein,
        layer_idx=args.layer_idx,
        output_dir_3d=msa_3d_dir,
        output_dir_arc=msa_arc_dir,
        combined_output_dir=combined_dir,
    )
    generate_combined_attention_panels(
        attention_type="triangle_start",
        protein=args.protein,
        layer_idx=args.layer_idx,
        output_dir_3d=tri_3d_dir,
        output_dir_arc=tri_arc_dir,
        combined_output_dir=combined_dir,
        residue_indices=[args.tri_residue_idx],
    )


def main(args):
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    attn_map_dir, image_dir, pdb_file = derived_paths(args)

    logging.info(f"protein:       {args.protein}")
    logging.info(f"output_dir:    {args.output_dir}")
    logging.info(f"attn_map_dir:  {attn_map_dir}")
    logging.info(f"image_dir:     {image_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    save_manifest(args, attn_map_dir, image_dir)

    if not args.visualize_only:
        run_inference(args, attn_map_dir)

    run_visualizations(args, attn_map_dir, image_dir, pdb_file)
    logging.info("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run VizFold end-to-end: OpenFold inference + attention visualization. "
            "Produces 3D PyMOL renders, arc diagrams, and combined panels for each attention head."
        ),
        epilog=(
            "Example (full pipeline):\n"
            "  python cli/vizfold_cli_viz.py \\\n"
            "      --fasta_file examples/monomer/fasta_dir_6KWC/6KWC.fasta \\\n"
            "      --output_dir outputs/6KWC_viz \\\n"
            "      --base_data_dir /data/alphafold \\\n"
            "      --protein 6KWC \\\n"
            "      --alignment_dir examples/monomer/alignments\n\n"
            "Example (skip inference, visualize existing outputs):\n"
            "  python cli/vizfold_cli_viz.py \\\n"
            "      --fasta_file examples/monomer/fasta_dir_6KWC/6KWC.fasta \\\n"
            "      --output_dir outputs/6KWC_viz \\\n"
            "      --protein 6KWC \\\n"
            "      --visualize_only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io_group = parser.add_argument_group("Required I/O")
    io_group.add_argument("--fasta_file", type=str, required=True,
        help="Path to a single-sequence FASTA file.")
    io_group.add_argument("--output_dir", type=str, required=True,
        help="Root directory for all outputs (predictions, attention maps, images).")
    io_group.add_argument("--protein", type=str, required=True,
        help="Protein identifier (e.g. 6KWC). Used to name output files.")

    viz_group = parser.add_argument_group("Visualization parameters")
    viz_group.add_argument("--tri_residue_idx", type=int, default=18,
        help="Residue index for triangle start attention focus (default: 18).")
    viz_group.add_argument("--layer_idx", type=int, default=47,
        help="Model layer to visualize (default: 47, the last layer).")
    viz_group.add_argument("--top_k", type=int, default=50,
        help="Maximum number of attention edges to show per head (default: 50).")
    viz_group.add_argument("--attn_map_dir", type=str, default=None,
        help="Override path to attention map directory. Auto-derived from output_dir if not set.")

    inf_group = parser.add_argument_group("Inference options")
    inf_group.add_argument("--base_data_dir", type=str, default=None,
        help="Root directory for AlphaFold databases. Required unless --visualize_only.")
    inf_group.add_argument("--alignment_dir", type=str, default=None,
        help="Path to precomputed alignments directory.")
    inf_group.add_argument("--config_preset", type=str, default="model_1_ptm",
        help="OpenFold model config preset (default: model_1_ptm).")
    inf_group.add_argument("--model_device", type=str, default="cuda:0",
        help="CUDA device for inference (default: cuda:0).")
    inf_group.add_argument("--num_recycles_save", type=int, default=1,
        help="Number of recycles to save during inference (default: 1).")
    inf_group.add_argument("--visualize_only", action="store_true", default=False,
        help="Skip inference and generate visualizations from existing attention maps and PDB.")

    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args()
    validate_args(args, parser)

    try:
        main(args)
    except subprocess.CalledProcessError as e:
        logging.error(f"Inference subprocess failed with exit code {e.returncode}")
        sys.exit(EXIT_PARTIAL)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(EXIT_PARTIAL)
