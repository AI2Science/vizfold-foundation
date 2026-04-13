"""
CLI wrapper for clustering FASTA sequences using mmseqs2 with PDB-style settings.

This provides a standardized, reproducible interface for VizFold workflows,
HPC batch jobs, and Airavata pipelines.
"""

from argparse import ArgumentParser
from pathlib import Path

# Import your existing logic
from cluster_fasta import main as run_cluster


def build_cli():
    parser = ArgumentParser(
        prog="vizfold cluster",
        description=(
            "Cluster protein sequences from a FASTA file using mmseqs2 with "
            "PDB-style parameters. Produces a reformatted cluster file where "
            "each line lists all {PDB_ID}_{CHAIN_ID} entries in a cluster."
        )
    )

    parser.add_argument(
        "--input-fasta",
        required=True,
        type=Path,
        help="Input FASTA file. Headers must be >{PDB_ID}_{CHAIN_ID}."
    )

    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Output text file containing clusters (one cluster per line)."
    )

    parser.add_argument(
        "--mmseqs-binary",
        required=True,
        type=str,
        help="Path to the mmseqs2 binary."
    )

    parser.add_argument(
        "--seq-id",
        type=float,
        default=0.4,
        help="Sequence identity threshold (default: 0.4)."
    )

    return parser


def cli():
    parser = build_cli()
    args = parser.parse_args()

    # Convert to the argument names expected by your original script
    class WrappedArgs:
        input_fasta = args.input_fasta
        output_file = args.output_file
        mmseqs_binary_path = args.mmseqs_binary
        seq_id = args.seq_id

    run_cluster(WrappedArgs())


if __name__ == "__main__":
    cli()