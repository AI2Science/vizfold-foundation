#!/usr/bin/env python3
"""
CLI for generating AlphaFold feature dictionaries from sequences and templates.

This tool processes FASTA sequences and template structures to generate feature
dictionaries compatible with AlphaFold / OpenFold inference pipelines.

Usage:
    python vizfold_cli_feature_dict.py \\
        sequences.fasta templates_dir/ output_dir/ \\
        --uniref90_database_path /data/uniref90.fasta
"""

import argparse
import logging
import os
import pickle
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import add_data_args

# Exit codes
EXIT_SUCCESS = 0
EXIT_PARTIAL = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_args(args, parser):
    """Validate command-line arguments and paths."""
    errors = []

    if not os.path.isfile(args.fasta_path):
        errors.append(f"--fasta_path (input file) does not exist: {args.fasta_path}")

    if not os.path.isdir(args.template_mmcif_dir):
        errors.append(f"--template_mmcif_dir does not exist: {args.template_mmcif_dir}")

    # Database validation for monomer mode
    if not args.multimer:
        if not args.uniref90_database_path or not os.path.isfile(args.uniref90_database_path):
            errors.append(f"--uniref90_database_path required for monomer mode, not found: {args.uniref90_database_path}")
        if not args.pdb70_database_path or not os.path.isfile(args.pdb70_database_path):
            errors.append(f"--pdb70_database_path required for monomer mode (hhsearch), not found: {args.pdb70_database_path}")
    
    # Database validation for multimer mode
    if args.multimer:
        if not args.uniref90_database_path or not os.path.isfile(args.uniref90_database_path):
            errors.append(f"--uniref90_database_path required for multimer mode, not found: {args.uniref90_database_path}")
        if not args.pdb_seqres_database_path or not os.path.isfile(args.pdb_seqres_database_path):
            errors.append(f"--pdb_seqres_database_path required for multimer mode (hmmsearch), not found: {args.pdb_seqres_database_path}")
        if not args.uniprot_database_path or not os.path.isfile(args.uniprot_database_path):
            errors.append(f"--uniprot_database_path required for multimer mode, not found: {args.uniprot_database_path}")

    # Binary validation
    if not os.path.isfile(args.jackhmmer_binary_path):
        errors.append(f"--jackhmmer_binary_path not found: {args.jackhmmer_binary_path}")

    if not args.multimer:
        if not os.path.isfile(args.hhsearch_binary_path):
            errors.append(f"--hhsearch_binary_path not found (required for monomer): {args.hhsearch_binary_path}")
    else:
        if not os.path.isfile(args.hmmsearch_binary_path):
            errors.append(f"--hmmsearch_binary_path not found (required for multimer): {args.hmmsearch_binary_path}")
        if not os.path.isfile(args.hmmbuild_binary_path):
            errors.append(f"--hmmbuild_binary_path not found (required for multimer): {args.hmmbuild_binary_path}")

    if args.hhblits_binary_path and not os.path.isfile(args.hhblits_binary_path):
        errors.append(f"--hhblits_binary_path provided but not found: {args.hhblits_binary_path}")

    if args.kalign_binary_path and not os.path.isfile(args.kalign_binary_path):
        errors.append(f"--kalign_binary_path provided but not found: {args.kalign_binary_path}")

    if errors:
        parser.error("Validation errors:\n  " + "\n  ".join(errors))


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for feature dict generation."""
    
    parser = argparse.ArgumentParser(
        prog="vizfold_cli_feature_dict",
        description=(
            "Generate AlphaFold feature dictionaries from protein sequences and templates. "
            "Processes FASTA input through the full data pipeline (MSA search, template search) "
            "and outputs a pickle file containing the feature dictionary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_argument_group("Input (positional arguments)")
    input_group.add_argument(
        "fasta_path",
        type=str,
        help="Path to input FASTA file containing the protein sequence(s).",
    )
    input_group.add_argument(
        "template_mmcif_dir",
        type=str,
        help="Path to directory containing template mmCIF files for template search.",
    )
    input_group.add_argument(
        "output_dir",
        type=str,
        help="Directory where feature_dict.pickle and intermediate MSA files are written.",
    )

    model_group = parser.add_argument_group("Model options")
    model_group.add_argument(
        "--multimer",
        action="store_true",
        default=False,
        help=(
            "Use multimer data pipeline. When enabled, expects multimer-mode "
            "proteins and uses different template search (HMMsearch vs HHsearch) "
            "and UniProt database."
        ),
    )

    # Add all data_args (handles CONDA_PREFIX defaults for binaries)
    add_data_args(parser)

    return parser


def main(args):
    """Generate feature dictionary from FASTA and templates."""
    try:
        from alphafold.data import pipeline, pipeline_multimer, templates
        from alphafold.data.tools import hmmsearch, hhsearch
    except ImportError as e:
        logger.error(f"Failed to import AlphaFold modules: {e}")
        logger.error("Ensure alphafold is installed and accessible in the Python environment.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Processing: {args.fasta_path}")
    logger.info(f"Templates: {args.template_mmcif_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Mode: {'multimer' if args.multimer else 'monomer'}")

    try:
        if args.multimer:
            logger.info("Setting up multimer pipeline...")
            template_searcher = hmmsearch.Hmmsearch(
                binary_path=args.hmmsearch_binary_path,
                hmmbuild_binary_path=args.hmmbuild_binary_path,
                database_path=args.pdb_seqres_database_path,
            )

            template_featurizer = templates.HmmsearchHitFeaturizer(
                mmcif_dir=args.template_mmcif_dir,
                max_template_date=args.max_template_date,
                max_hits=20,
                kalign_binary_path=args.kalign_binary_path,
                release_dates_path=args.release_dates_path,
                obsolete_pdbs_path=args.obsolete_pdbs_path
            )
        else:
            logger.info("Setting up monomer pipeline...")
            template_searcher = hhsearch.HHSearch(
                binary_path=args.hhsearch_binary_path,
                databases=[args.pdb70_database_path],
            )

            template_featurizer = templates.HhsearchHitFeaturizer(
                mmcif_dir=args.template_mmcif_dir,
                max_template_date=args.max_template_date,
                max_hits=20,
                kalign_binary_path=args.kalign_binary_path,
                release_dates_path=None,
                obsolete_pdbs_path=args.obsolete_pdbs_path
            )

        logger.info("Initializing data pipeline...")
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

        if args.multimer:
            logger.info("Wrapping with multimer pipeline...")
            data_pipeline = pipeline_multimer.DataPipeline(
                monomer_data_pipeline=data_pipeline,
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                uniprot_database_path=args.uniprot_database_path
            )

        logger.info("Processing sequence and generating features...")
        feature_dict = data_pipeline.process(
            input_fasta_path=args.fasta_path,
            msa_output_dir=args.output_dir,
        )

        output_path = os.path.join(args.output_dir, "feature_dict.pickle")
        logger.info(f"Saving feature dictionary to {output_path}...")
        with open(output_path, "wb") as fp:
            pickle.dump(feature_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Successfully generated feature dictionary!")
        return EXIT_SUCCESS

    except Exception as e:
        logger.error(f"Failed to generate feature dictionary: {e}")
        import traceback
        traceback.print_exc()
        return EXIT_PARTIAL


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    sys.exit(main(args))
