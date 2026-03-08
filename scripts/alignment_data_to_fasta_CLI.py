"""
CLI for alignment_data_to_fasta.py

This CLI delegates all core logic to the underlying fasta_generator module,
providing a clean, stable interface for VizFold workflows and HPC batch jobs.
"""

from argparse import ArgumentParser
from pathlib import Path

from fasta_generator import main as generate_fasta


def build_cli():
    parser = ArgumentParser(
        prog="vizfold fasta",
        description="Generate a FASTA file from alignment directories or alignment DB indices."
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the output FASTA file."
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--alignment-dir",
        type=Path,
        help="Directory containing chain subdirectories with alignment files."
    )

    source.add_argument(
        "--alignment-db-index",
        type=Path,
        help="Path to alignment DB index JSON file."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of worker threads (default: number of CPU cores)."
    )

    return parser


def cli():
    parser = build_cli()
    args = parser.parse_args()

    # Delegates to existing script
    generate_fasta(
        output_path=args.output,
        alignment_db_index=args.alignment_db_index,
        alignment_dir=args.alignment_dir,
    )


if __name__ == "__main__":
    cli()