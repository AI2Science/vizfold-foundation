"""
vizfold cluster — Cluster protein sequences using mmseqs2 with PDB-style settings.

This CLI wraps fasta_to_clusterfile.py and provides a standardized interface
for VizFold workflows, HPC batch jobs, and Airavata pipelines.
"""

import argparse
import logging
from pathlib import Path
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from scripts.fasta_to_clusterfile import main as run_cluster


def validate_args(args, parser):
    errors = []

    if not args.input_fasta.is_file():
        errors.append(f"Input FASTA not found: {args.input_fasta}")

    if not args.output_file.parent.exists():
        errors.append(f"Output directory does not exist: {args.output_file.parent}")

    if not Path(args.mmseqs_binary_path).is_file():
        errors.append(f"mmseqs2 binary not found: {args.mmseqs_binary_path}")

    if args.seq_id <= 0 or args.seq_id > 1:
        errors.append(f"--seq-id must be in (0, 1], got {args.seq_id}")

    if errors:
        parser.error("\n  ".join(errors))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="vizfold cluster",
        description=(
            "Cluster protein sequences from a FASTA file using mmseqs2 with "
            "PDB-style parameters. Produces a reformatted cluster file where "
            "each line lists all {PDB_ID}_{CHAIN_ID} entries in a cluster."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_fasta",
        type=Path,
        help="Input FASTA file. Headers must be >{PDB_ID}_{CHAIN_ID}.",
    )

    parser.add_argument(
        "output_file",
        type=Path,
        help="Output text file containing clusters (one cluster per line).",
    )

    parser.add_argument(
        "mmseqs_binary_path",
        type=str,
        help="Path to the mmseqs2 binary.",
    )

    parser.add_argument(
        "--seq-id",
        type=float,
        default=0.4,
        help="Sequence identity threshold.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return parser


def cli():
    parser = build_parser()
    args = parser.parse_args()

    validate_args(args, parser)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logging.info("Starting clustering run")
    logging.info(f"  input_fasta: {args.input_fasta}")
    logging.info(f"  output_file: {args.output_file}")
    logging.info(f"  mmseqs2:     {args.mmseqs_binary_path}")
    logging.info(f"  seq_id:      {args.seq_id}")

    run_cluster(args)

    logging.info("Clustering complete")


if __name__ == "__main__":
    cli()