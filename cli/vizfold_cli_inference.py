#!/usr/bin/env python3
"""
See USAGE.md for CLI documentation
"""

import argparse
import logging
import os
import subprocess
import sys
import tempfile


# Exit codes
EXIT_SUCCESS = 0
EXIT_PARTIAL = 1


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="vizfold_cli_inference",
        description=(
            "Standardized CLI for running pre-trained OpenFold structure "
            "prediction inference on protein sequences."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_argument_group(
        "Input (positional)",
        "Provide EITHER fasta_dir + template_mmcif_dir, "
        "OR use --sequence / --sequence_file together with --template_mmcif_dir.",
    )
    input_group.add_argument(
        "fasta_dir",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to a directory of FASTA files (.fasta / .fa), one sequence "
            "per file, or a single FASTA file path. "
            "Omit when using --sequence or --sequence_file."
        ),
    )
    input_group.add_argument(
        "template_mmcif_dir",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to directory containing template mmCIF files. "
            "Omit from the positional position when using --template_mmcif_dir flag."
        ),
    )

    seq_group = parser.add_argument_group(
        "Sequence input (alternative to fasta_dir positional arg)",
        "Pass a protein sequence directly without creating a FASTA file first.",
    )
    seq_group.add_argument(
        "--sequence",
        type=str,
        default=None,
        metavar="AMINO_ACIDS",
        help=(
            "Raw amino-acid sequence string (single-letter codes). "
            "A temporary FASTA file is created automatically. "
            "Use with --sequence_id to set the sequence name."
        ),
    )
    seq_group.add_argument(
        "--sequence_id",
        type=str,
        default="query",
        metavar="ID",
        help="FASTA header/tag used when --sequence is provided (default: query).",
    )
    seq_group.add_argument(
        "--sequence_file",
        type=str,
        default=None,
        metavar="FASTA_PATH",
        help=(
            "Path to a single FASTA file. Shorthand for passing a single-file "
            "path as fasta_dir."
        ),
    )
    seq_group.add_argument(
        "--template_mmcif_dir",
        type=str,
        default=None,
        dest="template_mmcif_dir_flag",
        metavar="DIR",
        help=(
            "Path to directory containing template mmCIF files "
            "(flag alternative to the positional argument)."
        ),
    )

    out_group = parser.add_argument_group("Output options")
    out_group.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        metavar="DIR",
        help=(
            "Directory where predicted structures and auxiliary files are written. "
            "Created automatically if it does not exist (default: current directory)."
        ),
    )
    out_group.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        metavar="SUFFIX",
        help="String appended to output file names (e.g. '_run1').",
    )
    out_group.add_argument(
        "--cif_output",
        action="store_true",
        default=False,
        help="Write predicted structures in ModelCIF format instead of PDB format.",
    )
    out_group.add_argument(
        "--save_outputs",
        action="store_true",
        default=False,
        help=(
            "Save all raw model outputs (embeddings, logits, etc.) as a pickle "
            "file (.pkl) alongside the structure prediction."
        ),
    )

    model_group = parser.add_argument_group("Model options")
    model_group.add_argument(
        "--model_device",
        type=str,
        default="cpu",
        metavar="DEVICE",
        help=(
            'PyTorch device for inference, e.g. "cpu", "cuda:0", "cuda:1". '
            "Default: cpu (a warning is printed if a GPU is available but unused)."
        ),
    )
    model_group.add_argument(
        "--config_preset",
        type=str,
        default="model_1",
        metavar="PRESET",
        help=(
            "Model configuration preset defined in openfold/config.py. "
            "Monomer presets: model_1 … model_5, model_1_ptm … model_5_ptm. "
            "Single-sequence presets: seq_model_esm1b, seq_model_esm1b_ptm, "
            "seqemb_initial_training, seqemb_finetuning. "
            "Multimer: any name containing 'multimer'. "
            "Default: model_1."
        ),
    )
    model_group.add_argument(
        "--openfold_checkpoint_path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to an OpenFold PyTorch checkpoint (.pt file) or a DeepSpeed "
            "checkpoint directory. Mutually exclusive with --jax_param_path. "
            "If neither is given, parameters are auto-selected from "
            "openfold/resources/params/ based on --config_preset."
        ),
    )
    model_group.add_argument(
        "--jax_param_path",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to JAX/AlphaFold2 model parameters (.npz). "
            "Mutually exclusive with --openfold_checkpoint_path."
        ),
    )
    model_group.add_argument(
        "--long_sequence_inference",
        action="store_true",
        default=False,
        help=(
            "Enable memory-saving options at the cost of speed. "
            "Use for sequences that exhaust GPU VRAM in the default mode."
        ),
    )
    model_group.add_argument(
        "--use_deepspeed_evoformer_attention",
        action="store_true",
        default=False,
        help=(
            "Use the DeepSpeed evoformer attention kernel. "
            "Requires deepspeed to be installed."
        ),
    )
    model_group.add_argument(
        "--enable_chunking",
        action="store_true",
        default=False,
        help="Enable activation chunking to reduce peak memory usage.",
    )
    model_group.add_argument(
        "--experiment_config_json",
        type=str,
        default="",
        metavar="JSON",
        help=(
            "Path to a JSON file containing flattened config key/value pairs "
            "that override the selected preset. "
            "Example: {\"model.evoformer_stack.no_blocks\": 24}"
        ),
    )

    align_group = parser.add_argument_group(
        "Alignment options",
        "Control how MSAs and templates are generated. "
        "Specify --use_precomputed_alignments to skip database searches entirely.",
    )
    align_group.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Path to a directory of pre-computed alignments. "
            "When provided, all MSA database searches are skipped and the "
            "database path arguments below are ignored."
        ),
    )
    align_group.add_argument(
        "--use_single_seq_mode",
        action="store_true",
        default=False,
        help="Use single-sequence embeddings (ESM) instead of MSAs.",
    )
    align_group.add_argument(
        "--use_custom_template",
        action="store_true",
        default=False,
        help=(
            "Use the mmCIF files from template_mmcif_dir directly as template "
            "input, bypassing the HHsearch/HMMsearch template search step."
        ),
    )
    align_group.add_argument(
        "--cpus",
        type=int,
        default=4,
        metavar="N",
        help="Number of CPU threads passed to alignment tools (default: 4).",
    )
    align_group.add_argument(
        "--preset",
        type=str,
        default="full_dbs",
        choices=("reduced_dbs", "full_dbs"),
        help=(
            "Database search preset. 'full_dbs' (default) uses all databases for "
            "maximum accuracy; 'reduced_dbs' is faster but less sensitive."
        ),
    )
    align_group.add_argument(
        "--max_template_date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "Latest allowed template release date. Templates released after "
            "this date are excluded. Defaults to today's date."
        ),
    )

    db_group = parser.add_argument_group(
        "Database paths",
        "Required for alignment unless --use_precomputed_alignments is set.",
    )
    db_group.add_argument(
        "--uniref90_database_path", type=str, default=None,
        metavar="PATH",
        help="UniRef90 FASTA database for jackhmmer (monomer + multimer).",
    )
    db_group.add_argument(
        "--mgnify_database_path", type=str, default=None,
        metavar="PATH",
        help="MGnify FASTA database for jackhmmer (monomer).",
    )
    db_group.add_argument(
        "--pdb70_database_path", type=str, default=None,
        metavar="PATH",
        help="PDB70 database for HHsearch template search (monomer).",
    )
    db_group.add_argument(
        "--pdb_seqres_database_path", type=str, default=None,
        metavar="PATH",
        help="PDB seqres database for HMMsearch template search (multimer).",
    )
    db_group.add_argument(
        "--uniref30_database_path", type=str, default=None,
        metavar="PATH",
        help="UniRef30 database for HHblits (monomer).",
    )
    db_group.add_argument(
        "--uniclust30_database_path", type=str, default=None,
        metavar="PATH",
        help="UniClust30 database for HHblits (monomer; alternative to UniRef30).",
    )
    db_group.add_argument(
        "--uniprot_database_path", type=str, default=None,
        metavar="PATH",
        help="UniProt FASTA database for jackhmmer (multimer only).",
    )
    db_group.add_argument(
        "--bfd_database_path", type=str, default=None,
        metavar="PATH",
        help=(
            "BFD database for HHblits (monomer). "
            "If omitted, the script uses the 'small BFD' mode automatically."
        ),
    )
    db_group.add_argument(
        "--obsolete_pdbs_path", type=str, default=None,
        metavar="PATH",
        help="Path to obsolete PDB entries list (optional).",
    )
    db_group.add_argument(
        "--release_dates_path", type=str, default=None,
        metavar="PATH",
        help="Path to PDB release dates file (optional).",
    )

    tools_group = parser.add_argument_group(
        "Alignment tool binary paths",
        "Override auto-detection; defaults come from the active conda environment.",
    )
    tools_group.add_argument(
        "--jackhmmer_binary_path", type=str, default=None, metavar="PATH",
        help="Path to jackhmmer binary.",
    )
    tools_group.add_argument(
        "--hhblits_binary_path", type=str, default=None, metavar="PATH",
        help="Path to hhblits binary.",
    )
    tools_group.add_argument(
        "--hhsearch_binary_path", type=str, default=None, metavar="PATH",
        help="Path to hhsearch binary.",
    )
    tools_group.add_argument(
        "--hmmsearch_binary_path", type=str, default=None, metavar="PATH",
        help="Path to hmmsearch binary.",
    )
    tools_group.add_argument(
        "--hmmbuild_binary_path", type=str, default=None, metavar="PATH",
        help="Path to hmmbuild binary.",
    )
    tools_group.add_argument(
        "--kalign_binary_path", type=str, default=None, metavar="PATH",
        help="Path to kalign binary.",
    )

    trace_group = parser.add_argument_group(
        "Intermediate trace and attention capture",
        "Flags that control saving of intermediate model state for visualization "
        "and debugging.",
    )
    trace_group.add_argument(
        "--trace_model",
        action="store_true",
        default=False,
        help=(
            "Convert model to TorchScript before inference. "
            "Significantly speeds up runtime for large batches at the cost of "
            "a one-time compilation delay. Requires fixed_size mode in the config."
        ),
    )
    trace_group.add_argument(
        "--attn_map_dir",
        type=str,
        default="",
        metavar="DIR",
        help=(
            "Directory to write attention map files. "
            "Leave empty (default) to disable attention map saving."
        ),
    )
    trace_group.add_argument(
        "--num_recycles_save",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of recycling iterations whose intermediate outputs are saved. "
            "Defaults to the value in the model config."
        ),
    )
    trace_group.add_argument(
        "--demo_attn",
        action="store_true",
        default=False,
        help="Enable demo attention visualization mode.",
    )
    trace_group.add_argument(
        "--triangle_residue_idx",
        type=int,
        default=None,
        metavar="IDX",
        help="Residue index for triangle-attention demo visualization.",
    )

    post_group = parser.add_argument_group("Post-processing options")
    post_group.add_argument(
        "--skip_relaxation",
        action="store_true",
        default=False,
        help=(
            "Skip OpenMM Amber relaxation of the predicted structure. "
            "Only the unrelaxed PDB/CIF file is written."
        ),
    )
    post_group.add_argument(
        "--subtract_plddt",
        action="store_true",
        default=False,
        help=(
            "Write (100 - pLDDT) in the B-factor column instead of pLDDT. "
            "Useful for downstream tools that interpret low B-factor as high confidence."
        ),
    )
    post_group.add_argument(
        "--multimer_ri_gap",
        type=int,
        default=200,
        metavar="N",
        help=(
            "Residue index gap inserted between chains in multimer mode "
            "(default: 200)."
        ),
    )

    repro_group = parser.add_argument_group("Reproducibility")
    repro_group.add_argument(
        "--data_random_seed",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "Integer seed for NumPy and PyTorch random number generators. "
            "If not set, a random seed is sampled at runtime."
        ),
    )

    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    hpc_group = parser.add_argument_group(
        "HPC / Slurm options",
        "Generate a ready-to-submit Slurm batch script or preview the execution "
        "plan without running inference.",
    )
    hpc_group.add_argument(
        "--generate-slurm",
        action="store_true",
        default=False,
        help=(
            "Generate a Slurm batch script (vizfold_job.sh) in --output_dir "
            "instead of (or in addition to) running inference directly. "
            "Use --slurm-account, --slurm-partition, --slurm-gpus, and "
            "--slurm-walltime to populate the #SBATCH directives."
        ),
    )
    hpc_group.add_argument(
        "--slurm-account",
        type=str,
        default=None,
        metavar="ACCOUNT",
        help="Slurm account / project to charge (sets #SBATCH --account).",
    )
    hpc_group.add_argument(
        "--slurm-partition",
        type=str,
        default=None,
        metavar="PARTITION",
        help=(
            "Slurm partition (queue) to submit to, e.g. 'gpu-v100' "
            "(sets #SBATCH --partition)."
        ),
    )
    hpc_group.add_argument(
        "--slurm-gpus",
        type=str,
        default=None,
        metavar="GPUS",
        help=(
            "GPU resource request forwarded to #SBATCH --gpus, "
            "e.g. '1', 'v100:2', or 'a100:1'."
        ),
    )
    hpc_group.add_argument(
        "--slurm-walltime",
        type=str,
        default=None,
        metavar="HH:MM:SS",
        help=(
            "Maximum wall-clock time for the job in HH:MM:SS format, "
            "e.g. '4:00:00' (sets #SBATCH --time)."
        ),
    )
    hpc_group.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Validate all arguments and print the full execution plan—directories "
            "that would be created, sequences that would be processed, and any "
            "Slurm script that would be generated—without loading models or "
            "running inference. Ideal for testing Airavata pipeline configurations "
            "without consuming compute hours."
        ),
    )

    return parser

def _add_str(cmd: list, flag: str, value) -> None:
    if value is not None and value != "":
        cmd.extend([flag, str(value)])


def _add_flag(cmd: list, flag: str, condition: bool) -> None:
    if condition:
        cmd.append(flag)


def build_command(script_path: str, fasta_dir: str, template_mmcif_dir: str,
                  args: argparse.Namespace) -> list:
    cmd = [sys.executable, script_path, fasta_dir, template_mmcif_dir]

    cmd.extend(["--output_dir", args.output_dir])
    _add_str(cmd, "--output_postfix", args.output_postfix)
    _add_flag(cmd, "--cif_output", args.cif_output)
    _add_flag(cmd, "--save_outputs", args.save_outputs)

    cmd.extend(["--model_device", args.model_device])
    cmd.extend(["--config_preset", args.config_preset])
    _add_str(cmd, "--openfold_checkpoint_path", args.openfold_checkpoint_path)
    _add_str(cmd, "--jax_param_path", args.jax_param_path)
    _add_flag(cmd, "--long_sequence_inference", args.long_sequence_inference)
    _add_flag(cmd, "--use_deepspeed_evoformer_attention",
              args.use_deepspeed_evoformer_attention)
    _add_flag(cmd, "--enable_chunking", args.enable_chunking)
    if args.experiment_config_json:
        cmd.extend(["--experiment_config_json", args.experiment_config_json])

    _add_str(cmd, "--use_precomputed_alignments", args.use_precomputed_alignments)
    _add_flag(cmd, "--use_single_seq_mode", args.use_single_seq_mode)
    _add_flag(cmd, "--use_custom_template", args.use_custom_template)
    cmd.extend(["--cpus", str(args.cpus)])
    cmd.extend(["--preset", args.preset])
    if args.max_template_date:
        cmd.extend(["--max_template_date", args.max_template_date])

    for db_arg in (
        "uniref90_database_path", "mgnify_database_path", "pdb70_database_path",
        "pdb_seqres_database_path", "uniref30_database_path",
        "uniclust30_database_path", "uniprot_database_path",
        "bfd_database_path", "obsolete_pdbs_path", "release_dates_path",
    ):
        _add_str(cmd, f"--{db_arg}", getattr(args, db_arg, None))

    for tool_arg in (
        "jackhmmer_binary_path", "hhblits_binary_path", "hhsearch_binary_path",
        "hmmsearch_binary_path", "hmmbuild_binary_path", "kalign_binary_path",
    ):
        _add_str(cmd, f"--{tool_arg}", getattr(args, tool_arg, None))

    _add_flag(cmd, "--trace_model", args.trace_model)
    if args.attn_map_dir:
        cmd.extend(["--attn_map_dir", args.attn_map_dir])
    _add_str(cmd, "--num_recycles_save", args.num_recycles_save)
    _add_flag(cmd, "--demo_attn", args.demo_attn)
    _add_str(cmd, "--triangle_residue_idx", args.triangle_residue_idx)

    _add_flag(cmd, "--skip_relaxation", args.skip_relaxation)
    _add_flag(cmd, "--subtract_plddt", args.subtract_plddt)
    cmd.extend(["--multimer_ri_gap", str(args.multimer_ri_gap)])

    _add_str(cmd, "--data_random_seed", args.data_random_seed)

    return cmd


def generate_slurm_script(args: argparse.Namespace, cmd: list) -> str:
    """Write a Slurm batch script to <output_dir>/vizfold_job.sh and return its path."""
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=vizfold_inference",
    ]
    if args.slurm_account:
        lines.append(f"#SBATCH --account={args.slurm_account}")
    if args.slurm_partition:
        lines.append(f"#SBATCH --partition={args.slurm_partition}")
    if args.slurm_gpus:
        lines.append(f"#SBATCH --gpus={args.slurm_gpus}")
    if args.slurm_walltime:
        lines.append(f"#SBATCH --time={args.slurm_walltime}")
    lines += [
        f"#SBATCH --output={args.output_dir}/vizfold_%j.out",
        f"#SBATCH --error={args.output_dir}/vizfold_%j.err",
        "",
        "# === VizFold OpenFold Inference Job ===",
        "# Auto-generated by vizfold_cli_inference.py",
        "# Submit with: sbatch vizfold_job.sh",
        "",
        " ".join(cmd),
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    slurm_path = os.path.join(args.output_dir, "vizfold_job.sh")
    with open(slurm_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return slurm_path


def resolve_inputs(args: argparse.Namespace, dry_run: bool = False):
    tmpl_dir = args.template_mmcif_dir or args.template_mmcif_dir_flag
    if tmpl_dir is None:
        print(
            "error: a template_mmcif_dir is required. "
            "Supply it as the second positional argument or via --template_mmcif_dir.",
            file=sys.stderr,
        )
        sys.exit(2)

    explicit_count = sum([
        args.fasta_dir is not None,
        args.sequence is not None,
        args.sequence_file is not None,
    ])
    if explicit_count == 0:
        print(
            "error: one of fasta_dir (positional), --sequence, or --sequence_file "
            "is required.",
            file=sys.stderr,
        )
        sys.exit(2)
    if explicit_count > 1:
        print(
            "error: fasta_dir, --sequence, and --sequence_file are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.sequence_file is not None:
        if not os.path.isfile(args.sequence_file):
            print(
                f"error: --sequence_file path does not exist: {args.sequence_file}",
                file=sys.stderr,
            )
            sys.exit(2)
        return args.sequence_file, tmpl_dir

    if args.fasta_dir is not None:
        if not os.path.exists(args.fasta_dir):
            print(
                f"error: fasta_dir path does not exist: {args.fasta_dir}",
                file=sys.stderr,
            )
            sys.exit(2)
        return args.fasta_dir, tmpl_dir

    if dry_run:
        placeholder = os.path.join(args.output_dir, f"{args.sequence_id}_<tmp>.fasta")
        return placeholder, tmpl_dir

    os.makedirs(args.output_dir, exist_ok=True)
    tmp_fasta = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".fasta",
        dir=args.output_dir,
        delete=False,
        prefix=f"{args.sequence_id}_",
    )
    tmp_fasta.write(f">{args.sequence_id}\n{args.sequence}\n")
    tmp_fasta.close()
    print(f"[VizFold CLI] Wrote sequence to temporary FASTA: {tmp_fasta.name}", flush=True)
    return tmp_fasta.name, tmpl_dir


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    # Locate run_pretrained_openfold.py relative to this file
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(cli_dir)
    script_path = os.path.join(repo_root, "run_pretrained_openfold.py")

    if not os.path.isfile(script_path):
        log.error("run_pretrained_openfold.py not found at: %s", script_path)
        sys.exit(EXIT_PARTIAL)

    fasta_dir, template_mmcif_dir = resolve_inputs(args, dry_run=args.dry_run)
    cmd = build_command(script_path, fasta_dir, template_mmcif_dir, args)

    if args.dry_run:
        log.info("=== DRY RUN — no models will be loaded, no inference will run ===")
        log.info("Output directory : %s", args.output_dir)
        log.info("  Would create if missing: %s", args.output_dir)
        log.info("Template mmCIF directory : %s", template_mmcif_dir)
        log.info("Input FASTA path : %s", fasta_dir)
        if os.path.isdir(fasta_dir):
            seqs = [
                f for f in os.listdir(fasta_dir)
                if f.endswith((".fasta", ".fa"))
            ]
            log.info("Sequences to process: %d file(s) in %s", len(seqs), fasta_dir)
            for s in seqs:
                log.info("  - %s", s)
        else:
            log.info("Sequences to process: 1 — %s", fasta_dir)
        log.info("Inference command that would run:")
        log.info("  %s", " ".join(cmd))
        if args.generate_slurm:
            slurm_path = os.path.join(args.output_dir, "vizfold_job.sh")
            log.info("Slurm script that would be generated: %s", slurm_path)
            log.info("  #SBATCH --account    = %s", args.slurm_account or "(not set)")
            log.info("  #SBATCH --partition  = %s", args.slurm_partition or "(not set)")
            log.info("  #SBATCH --gpus       = %s", args.slurm_gpus or "(not set)")
            log.info("  #SBATCH --time       = %s", args.slurm_walltime or "(not set)")
        return

    if args.generate_slurm:
        slurm_path = generate_slurm_script(args, cmd)
        log.info("Slurm batch script written to: %s", slurm_path)
        log.info("Submit with: sbatch %s", slurm_path)

    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode == EXIT_SUCCESS:
        log.info("Successfully ran pretrained OpenFold")
        sys.exit(EXIT_SUCCESS)
    else:
        log.error("OpenFold exited with error code %d", result.returncode)
        sys.exit(EXIT_PARTIAL)


if __name__ == "__main__":
    main()
