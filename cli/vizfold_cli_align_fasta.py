"""
vizfold fasta — Generate a FASTA file from alignment directories or alignment DB indices.

This CLI wraps alignment_data_to_fasta.py and provides a standardized interface
for VizFold workflows, HPC batch jobs, and Airavata pipelines.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import argparse
import logging
from pathlib import Path

from scripts.alignment_data_to_fasta import main as generate_fasta


def validate_args(args, parser):
    errors = []

    # Output directory must exist
    if not args.output_path.parent.exists():
        errors.append(f"Output directory does not exist: {args.output_path.parent}")

    # Exactly one source must be provided
    if (args.alignment_dir is None) and (args.alignment_db_index is None):
        errors.append("You must provide either alignment_dir or alignment_db_index.")

    if (args.alignment_dir is not None) and (args.alignment_db_index is not None):
        errors.append("Only one of alignment_dir or alignment_db_index may be provided.")

    # Validate alignment_dir
    if args.alignment_dir is not None and not args.alignment_dir.is_dir():
        errors.append(f"alignment_dir does not exist or is not a directory: {args.alignment_dir}")

    # Validate alignment_db_index
    if args.alignment_db_index is not None and not args.alignment_db_index.is_file():
        errors.append(f"alignment_db_index not found: {args.alignment_db_index}")

    if errors:
        parser.error("\n  ".join(errors))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="vizfold fasta",
        description=(
            "Generate a FASTA file from alignment directories or alignment DB indices. "
            "Each chain becomes a FASTA entry using the first available alignment file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional output path (matches teammate’s style)
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path to write the output FASTA file