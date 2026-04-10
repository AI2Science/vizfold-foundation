#!/usr/bin/env python3
"""
Generate AlphaFold feature dictionaries from FASTA sequences.

See USAGE.md for CLI documentation.
"""

import argparse
import os
import pickle
import sys

from alphafold.data import pipeline, pipeline_multimer, templates
from alphafold.data.tools import hmmsearch, hhsearch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import add_data_args


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI."""
    
    parser = argparse.ArgumentParser(
        prog="generate_feature_dict_cli",
        description=(
            "Generate AlphaFold feature dictionaries from protein sequences. "
            "Processes FASTA file(s) and produces feature_dict.pickle for structure prediction."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_argument_group(
        "Input (required)",
    )
    input_group.add_argument(
        "fasta_path",
        type=str,
        help="Path to FASTA file containing protein sequence(s).",
    )
    input_group.add_argument(
        "mmcif_dir",
        type=str,
        help="Path to directory containing template mmCIF files.",
    )

    output_group = parser.add_argument_group(
        "Output (required)",
    )
    output_group.add_argument(
        "output_dir",
        type=str,
        help="Directory where MSA results and feature_dict.pickle are written.",
    )

    mode_group = parser.add_argument_group(
        "Mode selection",
    )
    mode_group.add_argument(
        "--multimer",
        action="store_true",
        default=False,
        help=(
            "Use multimer pipeline (for protein complexes). "
            "Default: monomer pipeline."
        ),
    )

    # Add data/database arguments
    add_data_args(parser)

    return parser


def main(args: argparse.Namespace) -> None:
    """Execute the feature dictionary generation pipeline."""
    
    # Validate input paths exist
    if not os.path.isfile(args.fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {args.fasta_path}")
    
    if not os.path.isdir(args.mmcif_dir):
        raise FileNotFoundError(f"mmCIF directory not found: {args.mmcif_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize template searcher and featurizer based on mode
    if args.multimer:
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=args.hmmsearch_binary_path,
            hmmbuild_binary_path=args.hmmbuild_binary_path,
            database_path=args.pdb_seqres_database_path,
        )

        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=args.mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=20,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )
    else:
        template_searcher = hhsearch.HHSearch(
            binary_path=args.hhsearch_binary_path,
            databases=[args.pdb70_database_path],
        )

        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=20,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=None,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )

    # Initialize monomer data pipeline
    data_pipeline = pipeline.DataPipeline(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniref30_database_path=args.uniref30_database_path,
        small_bfd_database_path=None,
        template_featurizer=template_featurizer,
        template_searcher=template_searcher,
        use_small_bfd=False,
    )

    # Wrap with multimer pipeline if requested
    if args.multimer:
        data_pipeline = pipeline_multimer.DataPipeline(
            monomer_data_pipeline=data_pipeline,
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            uniprot_database_path=args.uniprot_database_path
        )

    # Process the FASTA file
    print(f"Processing FASTA: {args.fasta_path}")
    print(f"Template mmCIF directory: {args.mmcif_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'multimer' if args.multimer else 'monomer'}")
    print()

    feature_dict = data_pipeline.process(
        input_fasta_path=args.fasta_path,
        msa_output_dir=args.output_dir,
    )

    # Save feature dictionary
    feature_dict_path = os.path.join(args.output_dir, "feature_dict.pickle")
    with open(feature_dict_path, "wb") as fp:
        pickle.dump(feature_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Feature dictionary saved to: {feature_dict_path}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)