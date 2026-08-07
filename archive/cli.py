"""
Manual CLI wrapper for Archive Utils.

Examples:
  python3 "archive.cli.py" validate --archive trace.zarr --strict
  python3 "archive.cli.py" ingest-pkl --archive trace.zarr --pkl out.pkl --overwrite
  python3 "archive.cli.py" ingest-attn-txt --archive trace.zarr --txt attn.txt --layer 0 --tokens 128 --type triangle_start
  python3 "archive.cli.py" store-metadata --archive trace.zarr --model-version openfold --config-version v1 --sequence ACDE --num-residues 4 --num-recycles 1
"""

import argparse
import json

try:
    from .core import validate_archive
    from .load import ingest_attention_txt, ingest_output_pkl
    from .store import store_metadata
except ImportError:  # Allow running as `python archive/cli.py`.
    from core import validate_archive
    from load import ingest_attention_txt, ingest_output_pkl
    from store import store_metadata


def _cmd_validate(args):
    report = validate_archive(args.archive, strict=args.strict)
    print(json.dumps(report, indent=2, default=str))


def _cmd_ingest_pkl(args):
    summary = ingest_output_pkl(args.archive, args.pkl, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, default=str))


def _cmd_ingest_attn_txt(args):
    summary = ingest_attention_txt(
        archive_path=args.archive,
        txt_file=args.txt,
        layer_index=args.layer,
        num_tokens=args.tokens,
        attention_type=args.type,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, default=str))


def _cmd_store_metadata(args):
    store_metadata(
        path=args.archive,
        model_version=args.model_version,
        config_version=args.config_version,
        sequence=args.sequence,
        num_residues=args.num_residues,
        num_recycles=args.num_recycles,
        overwrite=args.overwrite,
    )
    print("metadata stored")


def build_parser():
    parser = argparse.ArgumentParser(description="Archive Utils manual CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_validate = subparsers.add_parser("validate", help="Validate a trace archive")
    p_validate.add_argument("--archive", required=True, help="Path to archive (e.g., trace.zarr)")
    p_validate.add_argument("--strict", action="store_true", default=False, help="Enable strict validation")
    p_validate.set_defaults(func=_cmd_validate)

    p_ingest_pkl = subparsers.add_parser("ingest-pkl", help="Ingest OpenFold output_dict.pkl")
    p_ingest_pkl.add_argument("--archive", required=True, help="Path to archive")
    p_ingest_pkl.add_argument("--pkl", required=True, help="Path to .pkl file")
    p_ingest_pkl.add_argument("--overwrite", action="store_true", default=False)
    p_ingest_pkl.set_defaults(func=_cmd_ingest_pkl)

    p_ingest_txt = subparsers.add_parser("ingest-attn-txt", help="Ingest attention .txt export")
    p_ingest_txt.add_argument("--archive", required=True, help="Path to archive")
    p_ingest_txt.add_argument("--txt", required=True, help="Path to attention text file")
    p_ingest_txt.add_argument("--layer", type=int, required=True, help="Layer index")
    p_ingest_txt.add_argument("--tokens", type=int, required=True, help="Number of tokens/residues")
    p_ingest_txt.add_argument("--type", default="pairwise", help="Attention type")
    p_ingest_txt.add_argument("--overwrite", action="store_true", default=False)
    p_ingest_txt.set_defaults(func=_cmd_ingest_attn_txt)

    p_meta = subparsers.add_parser("store-metadata", help="Store minimal metadata")
    p_meta.add_argument("--archive", required=True, help="Path to archive")
    p_meta.add_argument("--model-version", required=True)
    p_meta.add_argument("--config-version", required=True)
    p_meta.add_argument("--sequence", required=True)
    p_meta.add_argument("--num-residues", type=int, required=True)
    p_meta.add_argument("--num-recycles", type=int, required=True)
    p_meta.add_argument("--overwrite", action="store_true", default=False)
    p_meta.set_defaults(func=_cmd_store_metadata)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
