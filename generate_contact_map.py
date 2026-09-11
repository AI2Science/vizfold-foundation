"""
CLI wrapper for contact_map.py.
Validates all inputs, then delegates to contact_map.py via subprocess.

Usage:
    python generate_contact_map.py FASTA_PATH ALIGNMENT_DIR [options]

Example:
    python generate_contact_map.py \
        examples/monomer/fasta_dir/6kwc.fasta \
        examples/monomer/alignments/6KWC_1 \
        --pdb_path outputs/predictions/6KWC_1_model_1_ptm_relaxed.pdb \
        --region 30 90
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

EVOFORMER_BLOCKS = 48   # layers 0–47
VALID_PRESETS = {
    "model_1", "model_1_ptm",
    "model_2", "model_2_ptm",
    "model_3", "model_3_ptm",
    "model_4", "model_4_ptm",
    "model_5", "model_5_ptm",
    "model_1_multimer_v3", "model_2_multimer_v3", "model_3_multimer_v3",
    "model_4_multimer_v3", "model_5_multimer_v3",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate inputs and run contact_map.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fasta_path", type=str, help="Path to the protein FASTA file")
    parser.add_argument("alignment_dir", type=str, help="Path to the precomputed alignment directory")

    parser.add_argument(
        "--pdb_path", type=str, default=None,
        help="Path to reference PDB file. Default: <output_dir>/predictions/<PROTEIN>_1_<preset>_relaxed.pdb",
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "outputs"),
        help="Directory for output plot",
    )
    parser.add_argument(
        "--config_preset", type=str, default="model_1_ptm",
        choices=sorted(VALID_PRESETS),
        help="Model config preset",
    )
    parser.add_argument(
        "--params_path", type=str, default=None,
        help="Path to JAX model parameters .npz. Default: openfold/resources/params/params_<preset>.npz",
    )
    parser.add_argument(
        "--layers", type=int, nargs=2, metavar=("START", "END"),
        default=[38, 47],
        help="Evoformer block range to capture attention from (inclusive, 0-based)",
    )
    parser.add_argument(
        "--region", type=int, nargs=2, metavar=("START", "END"),
        default=None,
        help="Residue index range to plot (0-based, end-exclusive). Omit for the full map.",
    )
    parser.add_argument(
        "--contact_threshold", type=float, default=8.0,
        help="Cα contact distance threshold in Å",
    )
    parser.add_argument(
        "--chain_id", type=str, default="A",
        help="Chain ID for Cα distance computation",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device string",
    )
    return parser.parse_args()


def error(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def validate(args):
    # FASTA
    if not os.path.isfile(args.fasta_path):
        error(f"FASTA file not found: {args.fasta_path}")

    # Alignment dir
    if not os.path.isdir(args.alignment_dir):
        error(f"Alignment directory not found: {args.alignment_dir}")

    # Params path
    if args.params_path is None:
        args.params_path = os.path.join(
            SCRIPT_DIR, "openfold", "resources", "params",
            f"params_{args.config_preset}.npz",
        )
    if not os.path.isfile(args.params_path):
        error(f"Params file not found: {args.params_path}")

    # PDB — only validate if explicitly provided; auto-detection is contact_map.py's job
    if args.pdb_path is not None and not os.path.isfile(args.pdb_path):
        error(f"PDB file not found: {args.pdb_path}")

    # Output dir — create if needed
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.access(args.output_dir, os.W_OK):
        error(f"Output directory is not writable: {args.output_dir}")

    # Layer range
    layer_start, layer_end = args.layers
    if not (0 <= layer_start <= layer_end < EVOFORMER_BLOCKS):
        error(
            f"--layers {layer_start} {layer_end} is invalid. "
            f"Both values must be in [0, {EVOFORMER_BLOCKS - 1}] with start <= end."
        )

    # Region
    if args.region is not None:
        r0, r1 = args.region
        if r0 < 0:
            error(f"--region start must be >= 0, got {r0}")
        if r1 <= r0:
            error(f"--region end ({r1}) must be greater than start ({r0})")

    # Contact threshold
    if args.contact_threshold <= 0:
        error(f"--contact_threshold must be positive, got {args.contact_threshold}")

    # Chain ID
    if len(args.chain_id) != 1:
        error(f"--chain_id must be a single character, got '{args.chain_id}'")


def build_env(args):
    env = os.environ.copy()
    env["CM_FASTA_PATH"]        = args.fasta_path
    env["CM_ALIGNMENT_DIR"]     = args.alignment_dir
    env["CM_OUTPUT_DIR"]        = args.output_dir
    env["CM_CONFIG_PRESET"]     = args.config_preset
    env["CM_PARAMS_PATH"]       = args.params_path
    env["CM_CHAIN_ID"]          = args.chain_id
    env["CM_LAYER_START"]       = str(args.layers[0])
    env["CM_LAYER_END"]         = str(args.layers[1])
    env["CM_CONTACT_THRESHOLD"] = str(args.contact_threshold)
    if args.pdb_path is not None:
        env["CM_PDB_PATH"] = args.pdb_path
    if args.region is not None:
        env["CM_REGION_START"] = str(args.region[0])
        env["CM_REGION_END"]   = str(args.region[1])
    return env


def main():
    args = parse_args()
    validate(args)

    contact_map_script = os.path.join(SCRIPT_DIR, "contact_map.py")
    env = build_env(args)

    print("Validation passed. Launching contact_map.py...")
    result = subprocess.run([sys.executable, contact_map_script], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
