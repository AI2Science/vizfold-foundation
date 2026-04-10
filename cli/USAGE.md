# VizFold CLI

This directory contains standardized CLI wrappers for common computational workflows.

## Available Commands

- [`generate_feature_dict_cli.py`](#generate-feature-dict-cli) — Generate AlphaFold feature dictionaries from sequences
- [`run_pretrained_openfold_cli.py`](#openfold-cli) — Run pre-trained OpenFold structure prediction

---

## Generate Feature Dict CLI

Generate AlphaFold feature dictionaries from FASTA sequences. This is typically the first preprocessing step before running structure prediction.

### Quick Start

```bash
# From the repository root
python cli/generate_feature_dict_cli.py <fasta_path> <mmcif_dir> <output_dir> [OPTIONS]
```

### Summary

The feature dict CLI processes a FASTA file through the AlphaFold data pipeline to generate:
- Multiple Sequence Alignments (MSAs) from sequence databases
- Template structures from the template library
- A serialized feature dictionary (`feature_dict.pickle`) for consumption by the structure prediction model

### Examples

#### Monomer mode (default)

```bash
python cli/generate_feature_dict_cli.py \
  ./proteins/protein_A.fasta \
  /path/to/templates/ \
  ./features/protein_A/ \
  --uniref90_database_path /data/uniref90/uniref90.fasta \
  --mgnify_database_path /data/mgnify/mgnify.fasta \
  --pdb70_database_path /data/pdb70/pdb70 \
  --bfd_database_path /data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
```

#### Multimer mode

```bash
python cli/generate_feature_dict_cli.py \
  ./complexes/complex_AB.fasta \
  /path/to/templates/ \
  ./features/complex_AB/ \
  --multimer \
  --uniref90_database_path /data/uniref90/uniref90.fasta \
  --pdb_seqres_database_path /data/pdb_seqres/pdb_seqres.txt \
  --uniprot_database_path /data/uniprot/uniprot.fasta
```

### Arguments

| Argument | Required | Type | Description |
|---|---|---|---|
| `fasta_path` | **Yes** | `str` | Path to FASTA file with protein sequence(s). |
| `mmcif_dir` | **Yes** | `str` | Directory containing template mmCIF files. |
| `output_dir` | **Yes** | `str` | Output directory for MSAs and `feature_dict.pickle`. |
| `--multimer` | No | flag | Use multimer pipeline for protein complexes. |

### Database & Tool Paths

See [Database Paths](#database-paths) and [Binary Paths](#alignment-tool-binary-paths) sections below.

### Output

- `feature_dict.pickle` — Serialized feature dictionary for structure prediction
- `a3m/` — Multiple sequence alignments (one per input sequence)
- `sto/` — Stockholm-format alignments (if applicable)

---

## OpenFold CLI

<openfold-cli>

### Quick Start

```bash
# From the repository root
python cli/run_pretrained_openfold_cli.py <fasta_dir> <template_mmcif_dir> [OPTIONS]
```

## Input Modes

The CLI accepts protein input in three ways. Exactly **one**
must be provided.

### 1. FASTA directory (positional)

Pass a directory containing one or more `.fasta` / `.fa` files, plus a
directory of template mmCIF files.

```bash
python cli/run_pretrained_openfold_cli.py ./sequences/ ./templates/
```

### 2. Raw sequence string (`--sequence`)

Pass an amino-acid sequence directly. A temporary FASTA file is created
automatically in `--output_dir`.

```bash
python cli/run_pretrained_openfold_cli.py 
--sequence MKTAYIAKQRQISFVK... 
--sequence_id MY_PROTEIN 
--template_mmcif_dir ./templates/ 
--output_dir ./results/
```

### 3. Single FASTA file (`--sequence_file`)

Point to a single FASTA file instead of a directory.

```bash
python cli/run_pretrained_openfold_cli.py 
--sequence_file ./my_protein.fasta 
--template_mmcif_dir ./templates/ 
--output_dir ./results/
```

> **Note:** When using `--sequence` or `--sequence_file`, the template mmCIF
> directory must be provided via the `--template_mmcif_dir` flag (not as a
> positional argument).

---

## Argument Reference

### Input (positional)

| Argument | Required | Default | Description |
|---|---|---|---|
| `fasta_dir` | One of the three input modes | — | Path to a directory of FASTA files or a single FASTA file. Omit when using `--sequence` or `--sequence_file`. |
| `template_mmcif_dir` | **Yes** (positional or flag) | — | Directory containing template mmCIF files. |

### Sequence Input (alternative)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--sequence` | Mutually exclusive with `fasta_dir` and `--sequence_file` | — | Raw amino-acid sequence (single-letter codes). A temp FASTA is created automatically. |
| `--sequence_id` | No | `query` | FASTA header/tag used when `--sequence` is provided. |
| `--sequence_file` | Mutually exclusive with `fasta_dir` and `--sequence` | — | Path to a single FASTA file. |
| `--template_mmcif_dir` | **Yes** when using `--sequence` or `--sequence_file` | — | Template mmCIF directory (flag form of the positional argument). |

### Output Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--output_dir` | No | Current directory | Directory for predicted structures and auxiliary files. Created automatically if missing. |
| `--output_postfix` | No | — | Suffix appended to output file names (e.g. `_run1`). |
| `--cif_output` | No | `false` | Write structures in ModelCIF format instead of PDB. |
| `--save_outputs` | No | `false` | Save all raw model outputs (embeddings, logits, etc.) as a `.pkl` file. |

### Model Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model_device` | No | `cpu` | PyTorch device string: `cpu`, `cuda:0`, `cuda:1`, etc. |
| `--config_preset` | No | `model_1` | Model configuration preset (see [Presets](#model-presets) below). |
| `--openfold_checkpoint_path` | No | Auto-selected | Path to an OpenFold `.pt` checkpoint or DeepSpeed checkpoint directory. Mutually exclusive with `--jax_param_path`. |
| `--jax_param_path` | No | — | Path to JAX/AlphaFold2 `.npz` parameters. Mutually exclusive with `--openfold_checkpoint_path`. |
| `--long_sequence_inference` | No | `false` | Enable memory-saving mode for very long sequences. Slower but uses less VRAM. |
| `--use_deepspeed_evoformer_attention` | No | `false` | Use the DeepSpeed evoformer attention kernel. Requires DeepSpeed. |
| `--enable_chunking` | No | `false` | Enable activation chunking to reduce peak memory. |
| `--experiment_config_json` | No | — | Path to a JSON file with config overrides (flattened key/value pairs). |

### Alignment Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--use_precomputed_alignments` | No | — | Path to pre-computed alignments directory. When set, all database searches are skipped. |
| `--use_single_seq_mode` | No | `false` | Use single-sequence ESM embeddings instead of MSAs. |
| `--use_custom_template` | No | `false` | Use mmCIF files from `template_mmcif_dir` directly, bypassing template search. |
| `--cpus` | No | `4` | Number of CPU threads for alignment tools. |
| `--preset` | No | `full_dbs` | Database search preset: `full_dbs` (maximum accuracy) or `reduced_dbs` (faster). |
| `--max_template_date` | No | Today's date | Latest allowed template release date (`YYYY-MM-DD`). |

### Database Paths

Required for alignment unless `--use_precomputed_alignments` is set.

| Argument | Used By | Description |
|---|---|---|
| `--uniref90_database_path` | Monomer + Multimer | UniRef90 FASTA database for jackhmmer. |
| `--mgnify_database_path` | Monomer | MGnify FASTA database for jackhmmer. |
| `--pdb70_database_path` | Monomer | PDB70 database for HHsearch template search. |
| `--pdb_seqres_database_path` | Multimer | PDB seqres database for HMMsearch template search. |
| `--uniref30_database_path` | Monomer | UniRef30 database for HHblits. |
| `--uniclust30_database_path` | Monomer | UniClust30 database for HHblits (alternative to UniRef30). |
| `--uniprot_database_path` | Multimer | UniProt FASTA database for jackhmmer. |
| `--bfd_database_path` | Monomer | BFD database for HHblits. If omitted, small-BFD mode is used. |
| `--obsolete_pdbs_path` | Optional | Obsolete PDB entries list. |
| `--release_dates_path` | Optional | PDB release dates file. |

### Alignment Tool Binary Paths

Override auto-detected paths from the conda environment. All optional.

| Argument | Description |
|---|---|
| `--jackhmmer_binary_path` | Path to jackhmmer binary. |
| `--hhblits_binary_path` | Path to hhblits binary. |
| `--hhsearch_binary_path` | Path to hhsearch binary. |
| `--hmmsearch_binary_path` | Path to hmmsearch binary. |
| `--hmmbuild_binary_path` | Path to hmmbuild binary. |
| `--kalign_binary_path` | Path to kalign binary. |

### Intermediate Trace and Attention Capture

| Argument | Required | Default | Description |
|---|---|---|---|
| `--trace_model` | No | `false` | Convert model to TorchScript before inference. Speeds up large batches but has a one-time compilation cost. |
| `--attn_map_dir` | No | — | Directory for attention map output files. Leave empty to disable. |
| `--num_recycles_save` | No | Config default | Number of recycling iterations whose intermediate outputs are saved. |
| `--demo_attn` | No | `false` | Enable demo attention visualization mode. |
| `--triangle_residue_idx` | No | — | Residue index for triangle-attention demo visualization. |

### Post-processing Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--skip_relaxation` | No | `false` | Skip Amber relaxation. Only unrelaxed structures are written. |
| `--subtract_plddt` | No | `false` | Write `(100 - pLDDT)` in B-factor column instead of pLDDT. |
| `--multimer_ri_gap` | No | `200` | Residue index gap between chains in multimer mode. |

### Reproducibility

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_random_seed` | No | Random | Integer seed for NumPy and PyTorch RNGs. |

### HPC / Slurm Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--generate-slurm` | No | `false` | Generate a ready-to-submit `vizfold_job.sh` batch script in `--output_dir`. Populates `#SBATCH` directives from the flags below. |
| `--slurm-account` | No | — | Slurm account / project to charge (`#SBATCH --account`). |
| `--slurm-partition` | No | — | Slurm partition (queue) to submit to, e.g. `gpu-v100` (`#SBATCH --partition`). |
| `--slurm-gpus` | No | — | GPU resource request, e.g. `1`, `v100:2`, or `a100:1` (`#SBATCH --gpus`). |
| `--slurm-walltime` | No | — | Maximum wall-clock time in `HH:MM:SS`, e.g. `4:00:00` (`#SBATCH --time`). |
| `--dry-run` | No | `false` | Validate all arguments and print the execution plan (directories, sequences, Slurm script) without loading models or running inference. |

---

## Model Presets

The `--config_preset` argument selects a model architecture and weight set
defined in `openfold/config.py`.

| Preset | Description |
|---|---|
| `model_1` through `model_5` | Standard monomer models. |
| `model_1_ptm` through `model_5_ptm` | Monomer models with pTM (predicted TM-score) head. |
| `model_1_multimer_v3` through `model_5_multimer_v3` | Multimer models. |
| `seq_model_esm1b` | Single-sequence model using ESM-1b embeddings. |
| `seq_model_esm1b_ptm` | Single-sequence model with pTM head. |
| `seqemb_initial_training` | Sequence-embedding initial training config. |
| `seqemb_finetuning` | Sequence-embedding fine-tuning config. |

If neither `--openfold_checkpoint_path` nor `--jax_param_path` is provided,
the CLI auto-selects matching parameters from `openfold/resources/params/`
based on the chosen preset.

---

## Example Commands

### 1. Basic monomer inference with precomputed alignments

```bash
python cli/run_pretrained_openfold_cli.py examples/monomer/fasta_dir/ /path/to/mmcifs/
--use_precomputed_alignments examples/monomer/alignments/
--output_dir ./results/
```

### 2. GPU inference with a specific checkpoint

```bash
python cli/run_pretrained_openfold_cli.py examples/monomer/fasta_dir/ /path/to/mmcifs/ 
--model_device cuda:0
--config_preset model_1_ptm 
--openfold_checkpoint_path ./checkpoints/model.pt 
--use_precomputed_alignments examples/monomer/alignments/ 
--output_dir ./results/
```

### 3. Direct sequence input (no FASTA file needed)

```bash
python cli/run_pretrained_openfold_cli.py 
--sequence GSTIQPGTGYNNGYFYSYWNDGHGGVTYTNGPGGQFSVNWSNSGEFVGGKGWQPGTKNKVINFSG 
--sequence_id 6KWC 
--template_mmcif_dir /path/to/mmcifs/ 
--model_device cuda:0 
--use_precomputed_alignments examples/monomer/alignments/ 
--output_dir ./results/
```

### 4. Trace capture and attention maps for visualization

```bash
python cli/run_pretrained_openfold_cli.py examples/monomer/fasta_dir/ /path/to/mmcifs/ 
--use_precomputed_alignments examples/monomer/alignments/ 
--trace_model 
--save_outputs 
--attn_map_dir ./attention_maps/ 
--num_recycles_save 3 
--output_dir ./results/
```

### 5. Multimer inference with databases

```bash
python cli/run_pretrained_openfold_cli.py ./multimer_sequences/ /path/to/mmcifs/ 
--config_preset model_1_multimer_v3 
--model_device cuda:0 
--uniref90_database_path /data/uniref90/uniref90.fasta 
--pdb_seqres_database_path /data/pdb_seqres/pdb_seqres.txt 
--uniprot_database_path /data/uniprot/uniprot.fasta 
--output_dir ./results/
```

### 6. Reproducible run with seed and no relaxation

```bash
python cli/run_pretrained_openfold_cli.py examples/monomer/fasta_dir/ /path/to/mmcifs/ 
--use_precomputed_alignments examples/monomer/alignments/ 
--data_random_seed 42 
--skip_relaxation 
--output_dir ./results/
```

### 7. Generate a Slurm batch script (HPC / PACE)

Generate a `vizfold_job.sh` ready to submit with `sbatch`. The script contains
all `#SBATCH` directives and the exact inference command.

```bash
python cli/run_pretrained_openfold_cli.py examples/monomer/fasta_dir/ /path/to/mmcifs/ 
--use_precomputed_alignments examples/monomer/alignments/ 
--model_device cuda:0 
--output_dir ./results/ 
--generate-slurm 
--slurm-account GT-xyz123 
--slurm-partition gpu-v100 
--slurm-gpus v100:1 
--slurm-walltime 4:00:00
```

Submit the generated script:

```bash
sbatch ./results/vizfold_job.sh
```

### 8. Dry-run (validate arguments without running inference)

Use `--dry-run` to check that all paths and flags are valid and to preview
exactly what the CLI would do. No models are loaded and no files are written. Good for not consuming GPU hours.

```bash
python cli/run_pretrained_openfold_cli.py examples/monomer/fasta_dir/ /path/to/mmcifs/ 
--use_precomputed_alignments examples/monomer/alignments/ 
--output_dir ./results/ 
--generate-slurm 
--slurm-account GT-xyz123 
--slurm-partition gpu-v100 
--slurm-gpus v100:1 
--slurm-walltime 4:00:00 
--dry-run
```

---

## Output Files

Depending on the options used, the CLI writes the following to `--output_dir`:

| File | When |
|---|---|
| `*_unrelaxed.pdb` | Always (unless `--cif_output`) |
| `*_relaxed.pdb` | When relaxation is not skipped |
| `*_unrelaxed.cif` / `*_relaxed.cif` | When `--cif_output` is set |
| `*_output.pkl` | When `--save_outputs` is set |
| `<attn_map_dir>/*.npy` | When `--attn_map_dir` is set |
| `vizfold_job.sh` | When `--generate-slurm` is set |
| `vizfold_<jobid>.out` / `vizfold_<jobid>.err` | Slurm stdout/stderr after `sbatch` submission |

---

## Help

For the Generate Feature Dict CLI:
```bash
python cli/generate_feature_dict_cli.py --help
```

For the OpenFold CLI:
```bash
python cli/run_pretrained_openfold_cli.py --help
```

These print the full argument reference with defaults and descriptions for every option.
