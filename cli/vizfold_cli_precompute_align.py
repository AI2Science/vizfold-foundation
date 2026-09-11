import argparse
import json
import logging
import os
import sys
import threading
import tempfile
from functools import partial
from multiprocessing import cpu_count
from shutil import copyfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.utils import add_data_args


# Exit codes
EXIT_SUCCESS = 0       # all alignments completed
EXIT_PARTIAL = 1       # completed with some sequences skipped due to errors
# EXIT_ARGS = 2        # bad arguments — argparse uses 2 by default


class _FailureCounter:
    """Thread-safe counter for tracking skipped/failed sequences."""
    def __init__(self):
        self._lock = threading.Lock()
        self.value = 0

    def increment(self):
        with self._lock:
            self.value += 1


def validate_args(args, parser):
    errors = []

    if not os.path.isdir(args.input_dir):
        errors.append(f"--input_dir does not exist: {args.input_dir}")

    if args.checkpoint is not None and not os.path.isfile(args.checkpoint):
        errors.append(f"--checkpoint not found: {args.checkpoint}")

    if not args.uniref90_database_path or not os.path.isfile(args.uniref90_database_path):
        errors.append(f"--uniref90_database_path not found: {args.uniref90_database_path}")

    if args.hmmsearch_binary_path and os.path.isfile(args.hmmsearch_binary_path) and not args.pdb_seqres_database_path:
        errors.append("--hmmsearch_binary_path requires --pdb_seqres_database_path")

    if args.hhsearch_binary_path and os.path.isfile(args.hhsearch_binary_path) and not args.pdb70_database_path:
        errors.append("--hhsearch_binary_path requires --pdb70_database_path")

    if args.num_threads < 1:
        errors.append(f"--num_threads must be >= 1, got: {args.num_threads}")

    if errors:
        parser.error("\n  ".join(errors))


def save_run_manifest(args, intermediates_dir):
    manifest = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "checkpoint": args.checkpoint,
        "capture_intermediates": args.capture_intermediates,
        "intermediates_layers": args.intermediates_layers if args.capture_intermediates else [],
        "intermediates_dir": intermediates_dir,
        "num_threads": args.num_threads,
        "cpus_per_task": args.cpus_per_task,
        "slurm_nodes": os.environ.get("SLURM_JOB_NUM_NODES", "N/A"),
        "slurm_node_id": os.environ.get("SLURM_NODEID", "N/A"),
    }
    manifest_path = os.path.join(args.output_dir, "vizfold_run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_seq_group_alignments(seq_groups, alignment_runner, args, failure_counter=None):
    dirs = set(os.listdir(args.output_dir))
    for seq, names in seq_groups:
        first_name = names[0]
        alignment_dir = os.path.join(args.output_dir, first_name)

        try:
            os.makedirs(alignment_dir)
        except Exception as e:
            logging.warning(f"Failed to create directory for {first_name} with exception {e}...")
            if failure_counter is not None:
                failure_counter.increment()
            continue

        fd, fasta_path = tempfile.mkstemp(suffix=".fasta")
        with os.fdopen(fd, 'w') as fp:
            fp.write(f'>query\n{seq}')

        try:
            alignment_runner.run(fasta_path, alignment_dir)
        except Exception as e:
            logging.warning(e)
            logging.warning(f"Failed to run alignments for {first_name}. Skipping...")
            os.remove(fasta_path)
            os.rmdir(alignment_dir)
            if failure_counter is not None:
                failure_counter.increment()
            continue

        os.remove(fasta_path)

        for name in names[1:]:
            if name in dirs:
                logging.warning(f'{name} has already been processed. Skipping...')
                continue
            cp_dir = os.path.join(args.output_dir, name)
            os.makedirs(cp_dir, exist_ok=True)
            for f in os.listdir(alignment_dir):
                copyfile(os.path.join(alignment_dir, f), os.path.join(cp_dir, f))


def parse_and_align(files, alignment_runner, args, failure_counter=None):
    import openfold.data.mmcif_parsing as mmcif_parsing
    from openfold.data.parsers import parse_fasta
    from openfold.np import protein, residue_constants

    for f in files:
        path = os.path.join(args.input_dir, f)
        file_id = os.path.splitext(f)[0]
        seq_group_dict = {}

        if f.endswith('.cif'):
            with open(path, 'r') as fp:
                mmcif_str = fp.read()
            mmcif = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_str)
            if mmcif.mmcif_object is None:
                logging.warning(f'Failed to parse {f}...')
                if args.raise_errors:
                    raise list(mmcif.errors.values())[0]
                else:
                    if failure_counter is not None:
                        failure_counter.increment()
                    continue
            mmcif = mmcif.mmcif_object
            for chain_letter, seq in mmcif.chain_to_seqres.items():
                chain_id = '_'.join([file_id, chain_letter])
                l = seq_group_dict.setdefault(seq, [])
                l.append(chain_id)
        elif f.endswith('.fasta') or f.endswith('.fa'):
            with open(path, 'r') as fp:
                fasta_str = fp.read()
            input_seqs, _ = parse_fasta(fasta_str)
            if len(input_seqs) != 1:
                msg = f'More than one input_sequence found in {f}'
                if args.raise_errors:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
            input_sequence = input_seqs[0]
            seq_group_dict[input_sequence] = [file_id]
        elif f.endswith('.core'):
            with open(path, 'r') as fp:
                core_str = fp.read()
            core_prot = protein.from_proteinnet_string(core_str)
            aatype = core_prot.aatype
            seq = ''.join([
                residue_constants.restypes_with_x[aatype[i]]
                for i in range(len(aatype))
            ])
            seq_group_dict[seq] = [file_id]
        else:
            continue

        seq_group_tuples = [(k, v) for k, v in seq_group_dict.items()]
        run_seq_group_alignments(seq_group_tuples, alignment_runner, args, failure_counter)


def main(args):
    import openfold.data.mmcif_parsing as mmcif_parsing
    from openfold.data.data_pipeline import AlignmentRunner
    from openfold.data.parsers import parse_fasta
    from openfold.data.tools import hhsearch, hmmsearch
    from openfold.np import protein, residue_constants

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logging.info("starting alignment run")
    logging.info(f"  input:    {args.input_dir}")
    logging.info(f"  output:   {args.output_dir}")
    logging.info(f"  threads:  {args.num_threads}  cpus/task: {args.cpus_per_task}")
    logging.info(f"  uniref90: {args.uniref90_database_path}")
    logging.info(f"  capture intermediates: {'yes' if args.capture_intermediates else 'no'}")

    os.makedirs(args.output_dir, exist_ok=True)

    intermediates_dir = None
    if args.capture_intermediates:
        run_name = os.path.basename(args.input_dir.rstrip("/")) or "run"
        intermediates_dir = os.path.join(args.output_dir, f"{run_name}_intermediates")
        os.makedirs(intermediates_dir, exist_ok=True)

    save_run_manifest(args, intermediates_dir)

    if args.hmmsearch_binary_path is not None and args.pdb_seqres_database_path is not None:
        template_searcher = hmmsearch.Hmmsearch(
            binary_path=args.hmmsearch_binary_path,
            hmmbuild_binary_path=args.hmmbuild_binary_path,
            database_path=args.pdb_seqres_database_path,
        )
    elif args.hhsearch_binary_path is not None and args.pdb70_database_path is not None:
        template_searcher = hhsearch.HHSearch(
            binary_path=args.hhsearch_binary_path,
            databases=[args.pdb70_database_path],
        )
    else:
        template_searcher = None

    alignment_runner = AlignmentRunner(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniref30_database_path=args.uniref30_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        uniprot_database_path=args.uniprot_database_path,
        template_searcher=template_searcher,
        use_small_bfd=args.bfd_database_path is None,
        no_cpus=args.cpus_per_task,
    )

    files = list(os.listdir(args.input_dir))

    if args.mmcif_cache is not None:
        with open(args.mmcif_cache, "r") as fp:
            cache = json.load(fp)
    else:
        cache = None

    dirs = []
    if cache is not None and args.filter:
        dirs = set(os.listdir(args.output_dir))
        def prot_is_done(f):
            prot_id = os.path.splitext(f)[0]
            if prot_id in cache:
                chain_ids = cache[prot_id]["chain_ids"]
                for c in chain_ids:
                    full_name = prot_id + "_" + c
                    if not full_name in dirs:
                        return False
            else:
                return False
            return True
        files = [f for f in files if not prot_is_done(f)]

    def split_up_arglist(arglist):
        if os.environ.get("SLURM_JOB_NUM_NODES", 0):
            num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
            if num_nodes > 1:
                node_id = int(os.environ["SLURM_NODEID"])
                logging.info(f"SLURM: num_nodes={num_nodes}, node_id={node_id}")
                arglist = arglist[node_id::num_nodes]

        t_arglist = []
        for i in range(args.num_threads):
            t_arglist.append(arglist[i::args.num_threads])

        return t_arglist

    failure_counter = _FailureCounter()

    if cache is not None and "seqs" in next(iter(cache.values())):
        seq_group_dict = {}
        for f in files:
            prot_id = os.path.splitext(f)[0]
            if prot_id in cache:
                prot_cache = cache[prot_id]
                chains_seqs = zip(prot_cache["chain_ids"], prot_cache["seqs"])
                for chain, seq in chains_seqs:
                    chain_name = prot_id + "_" + chain
                    if chain_name not in dirs:
                        l = seq_group_dict.setdefault(seq, [])
                        l.append(chain_name)

        func = partial(run_seq_group_alignments, alignment_runner=alignment_runner, args=args, failure_counter=failure_counter)
        seq_groups = [(k, v) for k, v in seq_group_dict.items()]
        seq_groups = sorted(seq_groups, key=lambda x: len(x[1]))
        task_arglist = [[a] for a in split_up_arglist(seq_groups)]
    else:
        func = partial(parse_and_align, alignment_runner=alignment_runner, args=args, failure_counter=failure_counter)
        task_arglist = [[a] for a in split_up_arglist(files)]

    threads = []
    for i, task_args in enumerate(task_arglist):
        logging.info(f"Started thread {i}...")
        t = threading.Thread(target=func, args=task_args)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if failure_counter.value > 0:
        logging.warning(f"{failure_counter.value} sequence(s) failed or were skipped due to errors.")
        sys.exit(EXIT_PARTIAL)

    logging.info("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run MSA alignments for a directory of protein sequences. "
            "Each sequence gets its own output folder with .a3m/.sto/.hhr files "
            "that can be passed directly to OpenFold inference."
        ),
        epilog=(
            "Example (minimal):\n"
            "  python cli/vizfold_cli_precompute_align.py \\\n"
            "      --input_dir fasta_dir/ --output_dir alignments/ \\\n"
            "      --uniref90_database_path /data/uniref90/uniref90.fasta\n\n"
            "Example (full databases, 4 threads):\n"
            "  python cli/vizfold_cli_precompute_align.py \\\n"
            "      --input_dir fasta_dir/ --output_dir alignments/ \\\n"
            "      --uniref90_database_path /data/uniref90/uniref90.fasta \\\n"
            "      --mgnify_database_path /data/mgnify/mgy_clusters.fa \\\n"
            "      --bfd_database_path /data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \\\n"
            "      --pdb70_database_path /data/pdb70/pdb70 \\\n"
            "      --num_threads 4 --cpus_per_task 8\n\n"
            "To save attention maps and hidden states during inference, add --capture_intermediates.\n"
            "Then pass the alignments folder to inference:\n"
            "  python run_pretrained_openfold.py ... --use_precomputed_alignments alignments/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io_group = parser.add_argument_group("Required I/O")
    io_group.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing input files (.fasta, .fa, .cif, or .core).",
    )
    io_group.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where alignment subdirectories will be written.",
    )

    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--num_threads", type=int, default=1,
        help="Number of parallel alignment threads (default: 1).",
    )
    exec_group.add_argument(
        "--no_tasks", type=int, default=None,
        help=argparse.SUPPRESS,
    )
    exec_group.add_argument(
        "--cpus_per_task", type=int, default=cpu_count(),
        help=f"How many CPU cores each alignment tool gets (default: {cpu_count()}).",
    )
    exec_group.add_argument(
        "--filter", action=argparse.BooleanOptionalAction, default=True,
        help="Skip sequences already present in output_dir when using --mmcif_cache (default: --filter).",
    )
    exec_group.add_argument(
        "--mmcif_cache", type=str, default=None,
        help="Path to a mmCIF cache JSON. Used with --filter to skip already-processed entries.",
    )
    exec_group.add_argument(
        "--raise_errors", action="store_true", default=False,
        help="Stop on the first parse error instead of skipping bad files and continuing.",
    )
    exec_group.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    vizfold_group = parser.add_argument_group("VizFold options")
    vizfold_group.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Path to a pretrained OpenFold model checkpoint. "
            "Gets written to the run manifest so the inference step knows which model to load."
        ),
    )
    vizfold_group.add_argument(
        "--capture_intermediates", action="store_true", default=False,
        help=(
            "Flag the run manifest to capture attention maps and hidden states during inference. "
            "Alignment itself doesn't run the model — this just saves the setting for the inference step."
        ),
    )
    vizfold_group.add_argument(
        "--intermediates_layers", type=int, nargs="+", default=None,
        help=(
            "Layer indices to pull intermediates from, e.g. --intermediates_layers 4 8 12. "
            "Only used when --capture_intermediates is set."
        ),
    )

    db_group = parser.add_argument_group("Database and binary paths")
    add_data_args(db_group)
    parser._option_string_actions["--uniref90_database_path"].required = True
    parser._option_string_actions["--uniref90_database_path"].help = (
        "Path to UniRef90 FASTA database (required for jackhmmer MSA search)."
    )

    args = parser.parse_args()

    if args.no_tasks is not None:
        logging.warning("--no_tasks is deprecated, use --num_threads instead. using --no_tasks value for now.")
        args.num_threads = args.no_tasks

    validate_args(args, parser)
    try:
        main(args)
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(EXIT_PARTIAL)
