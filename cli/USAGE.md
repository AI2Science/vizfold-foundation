# VizFold CLI

This directory contains the standardized CLI wrappers for the VizFold pipeline. The three scripts represent the three stages of the pipeline and are designed to be used in sequence.

## Available Commands

| Script | Stage | Purpose |
|---|---|---|
| `vizfold_cli_fasta_to_clusterfile.py` | 0 — Pre-process (optional) | Cluster FASTA sequences using mmseqs2 (PDB-style settings) |
| [`vizfold_cli_precompute_align.py`](#precompute-alignments-cli) | 1 — Align | Batch MSA alignment precomputation |
| `vizfold_cli_align_fasta.py` | 1.5 — Post-align (optional) | Extract representative sequences from alignment dirs or alignment-DBs |
| [`vizfold_cli_inference.py`](#openfold-inference-cli) | 2 — Infer | Pre-trained OpenFold structure prediction |
| [`vizfold_cli_viz.py`](#visualization-cli) | 3 — Visualize | Attention map and 3D structure visualization |


---

## Full Pipeline Overview

The VizFold workflow consists of three primary stages, with two optional utility stages that can be used before or after alignment generation.

### End-to-End Example (protein 6KWC on PACE)

```bash
# Stage 0 — optional preprocessing (FASTA clustering)
python cli/vizfold_cli_fasta_to_clusterfile.py \
    sequences.fasta clusters.txt /path/to/mmseqs \
    --seq-id 0.4

# Stage 1 — precompute MSA alignments
python cli/vizfold_cli_precompute_align.py \
    --input_dir examples/monomer/fasta_dir_6KWC/ \
    --output_dir outputs/6KWC_alignments/ \
    --uniref90_database_path /data/uniref90/uniref90.fasta \
    --mgnify_database_path /data/mgnify/mgy_clusters_2022_05.fa \
    --bfd_database_path /data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --num_threads 4 --cpus_per_task 8

# Stage 1.5 — optional post-alignment utility (extract FASTA from alignments)
python cli/vizfold_cli_align_fasta.py \
    outputs/6KWC_sequences_from_alignments.fasta \
    --alignment-dir outputs/6KWC_alignments/

# Stage 2 — run OpenFold inference with precomputed alignments
python cli/vizfold_cli_inference.py \
    examples/monomer/fasta_dir_6KWC/ /data/pdb_mmcif/mmcif_files/ \
    --output_dir outputs/6KWC_predictions/ \
    --use_precomputed_alignments outputs/6KWC_alignments/ \
    --model_device cuda:0 \
    --config_preset model_1_ptm \
    --attn_map_dir outputs/6KWC_attn_maps/ \
    --num_recycles_save 1 \
    --demo_attn \
    --triangle_residue_idx 18 \
    --save_outputs

# Stage 3 — visualize attention maps
python cli/vizfold_cli_viz.py \
    --fasta_file examples/monomer/fasta_dir_6KWC/6KWC.fasta \
    --output_dir outputs/6KWC_predictions/ \
    --protein 6KWC \
    --tri_residue_idx 18 \
    --visualize_only
```

### Running on PACE via Slurm

Stage 2 supports generating a ready-to-submit Slurm batch script:

```bash
python cli/vizfold_cli_inference.py \
    examples/monomer/fasta_dir_6KWC/ /data/pdb_mmcif/mmcif_files/ \
    --output_dir outputs/6KWC_predictions/ \
    --use_precomputed_alignments outputs/6KWC_alignments/ \
    --model_device cuda:0 \
    --generate-slurm \
    --slurm-account GT-xyz123 \
    --slurm-partition gpu-v100 \
    --slurm-gpus v100:1 \
    --slurm-walltime 4:00:00

sbatch outputs/6KWC_predictions/vizfold_job.sh
```

---

## Precompute Alignments CLI

Run MSA alignments for a directory of protein sequences. Each sequence gets its own output folder with `.a3m` / `.sto` / `.hhr` files that can be passed directly to Stage 2.

### Quick Start

```bash
python cli/vizfold_cli_precompute_align.py \
    --input_dir fasta_dir/ \
    --output_dir alignments/ \
    --uniref90_database_path /data/uniref90/uniref90.fasta
```

### Examples

#### Minimal (uniref90 only)

```bash
python cli/vizfold_cli_precompute_align.py \
    --input_dir fasta_dir/ --output_dir alignments/ \
    --uniref90_database_path /data/uniref90/uniref90.fasta
```

#### Full databases, 4 threads

```bash
python cli/vizfold_cli_precompute_align.py \
    --input_dir fasta_dir/ --output_dir alignments/ \
    --uniref90_database_path /data/uniref90/uniref90.fasta \
    --mgnify_database_path /data/mgnify/mgy_clusters.fa \
    --bfd_database_path /data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --pdb70_database_path /data/pdb70/pdb70 \
    --num_threads 4 --cpus_per_task 8
```

#### Resume from mmCIF cache (skip already-processed entries)

```bash
python cli/vizfold_cli_precompute_align.py \
    --input_dir fasta_dir/ --output_dir alignments/ \
    --uniref90_database_path /data/uniref90/uniref90.fasta \
    --mmcif_cache cache.json --filter
```

### Argument Reference

#### Required I/O

| Argument | Required | Description |
|---|---|---|
| `--input_dir` | **Yes** | Directory containing input files (`.fasta`, `.fa`, `.cif`, or `.core`). |
| `--output_dir` | **Yes** | Directory where per-sequence alignment subdirectories are written. |

#### Execution

| Argument | Default | Description |
|---|---|---|
| `--num_threads` | `1` | Number of parallel alignment threads. |
| `--cpus_per_task` | All CPUs | CPU cores allocated to each alignment tool invocation. |
| `--filter` | `true` | Skip sequences already in `output_dir` when using `--mmcif_cache`. |
| `--mmcif_cache` | — | Path to mmCIF cache JSON. Used with `--filter` to skip processed entries. |
| `--raise_errors` | `false` | Stop on the first parse error instead of skipping bad files. |
| `--log_level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

#### VizFold Options

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | — | Path to a pretrained model checkpoint. Written to the run manifest for the inference step. |
| `--capture_intermediates` | `false` | Flag the run manifest to capture attention maps during inference. |
| `--intermediates_layers` | — | Layer indices to pull intermediates from, e.g. `--intermediates_layers 4 8 12`. Only used with `--capture_intermediates`. |

#### Database Paths

See [Database Paths](#database-paths) and [Binary Paths](#alignment-tool-binary-paths) sections below.

### Output

- One subdirectory per sequence under `--output_dir`, containing `.a3m`, `.sto`, and/or `.hhr` alignment files.
- `vizfold_run_manifest.json` — records all run settings for reproducibility.

---

## OpenFold Inference CLI

Run pre-trained OpenFold structure prediction on protein sequences. Wraps `run_pretrained_openfold.py` with a standardized interface.

### Quick Start

```bash
python cli/vizfold_cli_inference.py <fasta_dir> <template_mmcif_dir> [OPTIONS]
```

### Input Modes

The CLI accepts protein input in three ways. Exactly **one** must be provided.

#### 1. FASTA directory (positional)

```bash
python cli/vizfold_cli_inference.py ./sequences/ ./templates/
```

#### 2. Raw sequence string (`--sequence`)

```bash
python cli/vizfold_cli_inference.py \
    --sequence MKTAYIAKQRQISFVK... \
    --sequence_id MY_PROTEIN \
    --template_mmcif_dir ./templates/ \
    --output_dir ./results/
```

#### 3. Single FASTA file (`--sequence_file`)

```bash
python cli/vizfold_cli_inference.py \
    --sequence_file ./my_protein.fasta \
    --template_mmcif_dir ./templates/ \
    --output_dir ./results/
```

> **Note:** When using `--sequence` or `--sequence_file`, the template mmCIF directory must be provided via the `--template_mmcif_dir` flag (not as a positional argument).

### Argument Reference

#### Input (positional)

| Argument | Required | Default | Description |
|---|---|---|---|
| `fasta_dir` | One of the three input modes | — | Path to a directory of FASTA files or a single FASTA file. Omit when using `--sequence` or `--sequence_file`. |
| `template_mmcif_dir` | **Yes** (positional or flag) | — | Directory containing template mmCIF files. |

#### Sequence Input (alternative)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--sequence` | Mutually exclusive | — | Raw amino-acid sequence (single-letter codes). A temp FASTA is created automatically. |
| `--sequence_id` | No | `query` | FASTA header/tag used when `--sequence` is provided. |
| `--sequence_file` | Mutually exclusive | — | Path to a single FASTA file. |
| `--template_mmcif_dir` | **Yes** with `--sequence` / `--sequence_file` | — | Template mmCIF directory (flag form). |

#### Output Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--output_dir` | No | Current directory | Directory for predicted structures and auxiliary files. Created automatically if missing. |
| `--output_postfix` | No | — | Suffix appended to output file names (e.g. `_run1`). |
| `--cif_output` | No | `false` | Write structures in ModelCIF format instead of PDB. |
| `--save_outputs` | No | `false` | Save all raw model outputs (embeddings, logits, etc.) as a `.pkl` file. |

#### Model Options

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

#### Alignment Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--use_precomputed_alignments` | No | — | Path to pre-computed alignments directory. When set, all database searches are skipped. |
| `--use_single_seq_mode` | No | `false` | Use single-sequence ESM embeddings instead of MSAs. |
| `--use_custom_template` | No | `false` | Use mmCIF files from `template_mmcif_dir` directly, bypassing template search. |
| `--cpus` | No | `4` | Number of CPU threads for alignment tools. |
| `--preset` | No | `full_dbs` | Database search preset: `full_dbs` (maximum accuracy) or `reduced_dbs` (faster). |
| `--max_template_date` | No | Today's date | Latest allowed template release date (`YYYY-MM-DD`). |

#### Database Paths

See [Database Paths](#database-paths) section below.

#### Alignment Tool Binary Paths

See [Binary Paths](#alignment-tool-binary-paths) section below.

#### Intermediate Trace and Attention Capture

| Argument | Required | Default | Description |
|---|---|---|---|
| `--trace_model` | No | `false` | Convert model to TorchScript before inference. Speeds up large batches with a one-time compilation cost. |
| `--attn_map_dir` | No | — | Directory for attention map output files. Leave empty to disable. Pass this path to Stage 3. |
| `--num_recycles_save` | No | Config default | Number of recycling iterations whose intermediate outputs are saved. |
| `--demo_attn` | No | `false` | Enable demo attention visualization mode. |
| `--triangle_residue_idx` | No | — | Residue index for triangle-attention demo visualization. |

#### Post-processing Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--skip_relaxation` | No | `false` | Skip Amber relaxation. Only unrelaxed structures are written. |
| `--subtract_plddt` | No | `false` | Write `(100 - pLDDT)` in B-factor column instead of pLDDT. |
| `--multimer_ri_gap` | No | `200` | Residue index gap between chains in multimer mode. |

#### Reproducibility

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_random_seed` | No | Random | Integer seed for NumPy and PyTorch RNGs. |

#### Execution

| Argument | Required | Default | Description |
|---|---|---|---|
| `--log_level` | No | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

#### HPC / Slurm Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--generate-slurm` | No | `false` | Generate a ready-to-submit `vizfold_job.sh` batch script in `--output_dir`. |
| `--slurm-account` | No | — | Slurm account / project to charge (`#SBATCH --account`). |
| `--slurm-partition` | No | — | Slurm partition (queue), e.g. `gpu-v100` (`#SBATCH --partition`). |
| `--slurm-gpus` | No | — | GPU resource request, e.g. `1`, `v100:2`, or `a100:1` (`#SBATCH --gpus`). |
| `--slurm-walltime` | No | — | Maximum wall-clock time in `HH:MM:SS`, e.g. `4:00:00` (`#SBATCH --time`). |
| `--dry-run` | No | `false` | Validate all arguments and print the execution plan without loading models or running inference. |

### Example Commands

#### Basic monomer inference with precomputed alignments

```bash
python cli/vizfold_cli_inference.py examples/monomer/fasta_dir/ /path/to/mmcifs/ \
    --use_precomputed_alignments examples/monomer/alignments/ \
    --output_dir ./results/
```

#### GPU inference with a specific checkpoint

```bash
python cli/vizfold_cli_inference.py examples/monomer/fasta_dir/ /path/to/mmcifs/ \
    --model_device cuda:0 \
    --config_preset model_1_ptm \
    --openfold_checkpoint_path ./checkpoints/model.pt \
    --use_precomputed_alignments examples/monomer/alignments/ \
    --output_dir ./results/
```

#### Direct sequence input (no FASTA file needed)

```bash
python cli/vizfold_cli_inference.py \
    --sequence GSTIQPGTGYNNGYFYSYWNDGHGGVTYTNGPGGQFSVNWSNSGEFVGGKGWQPGTKNKVINFSG \
    --sequence_id 6KWC \
    --template_mmcif_dir /path/to/mmcifs/ \
    --model_device cuda:0 \
    --use_precomputed_alignments examples/monomer/alignments/ \
    --output_dir ./results/
```

#### Trace capture and attention maps for visualization

```bash
python cli/vizfold_cli_inference.py examples/monomer/fasta_dir/ /path/to/mmcifs/ \
    --use_precomputed_alignments examples/monomer/alignments/ \
    --save_outputs \
    --attn_map_dir ./attention_maps/ \
    --num_recycles_save 3 \
    --demo_attn --triangle_residue_idx 18 \
    --output_dir ./results/
```

#### Generate a Slurm batch script (HPC / PACE)

```bash
python cli/vizfold_cli_inference.py examples/monomer/fasta_dir/ /path/to/mmcifs/ \
    --use_precomputed_alignments examples/monomer/alignments/ \
    --model_device cuda:0 \
    --output_dir ./results/ \
    --generate-slurm \
    --slurm-account GT-xyz123 \
    --slurm-partition gpu-v100 \
    --slurm-gpus v100:1 \
    --slurm-walltime 4:00:00

sbatch ./results/vizfold_job.sh
```

#### Dry-run (validate arguments without running inference)

```bash
python cli/vizfold_cli_inference.py examples/monomer/fasta_dir/ /path/to/mmcifs/ \
    --use_precomputed_alignments examples/monomer/alignments/ \
    --output_dir ./results/ \
    --dry-run
```

### Output Files

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

## Visualization CLI

Run attention visualization from OpenFold inference outputs. Produces 3D PyMOL renders, arc diagrams, and combined panels for each attention head.

Can run the full pipeline (inference + visualization) or skip inference and generate visuals from existing attention maps (`--visualize_only`).

### Quick Start

```bash
# Full pipeline (inference + visualization)
python cli/vizfold_cli_viz.py \
    --fasta_file examples/monomer/fasta_dir_6KWC/6KWC.fasta \
    --output_dir outputs/6KWC_viz \
    --base_data_dir /data/alphafold \
    --protein 6KWC

# Visualization only (skip inference)
python cli/vizfold_cli_viz.py \
    --fasta_file examples/monomer/fasta_dir_6KWC/6KWC.fasta \
    --output_dir outputs/6KWC_viz \
    --protein 6KWC \
    --visualize_only
```

### Argument Reference

#### Required I/O

| Argument | Required | Description |
|---|---|---|
| `--fasta_file` | **Yes** | Path to a single-sequence FASTA file. |
| `--output_dir` | **Yes** | Root directory for all outputs (predictions, attention maps, images). |
| `--protein` | **Yes** | Protein identifier (e.g. `6KWC`). Used to name output files. |

#### Visualization Parameters

| Argument | Default | Description |
|---|---|---|
| `--tri_residue_idx` | `18` | Residue index for triangle start attention focus. |
| `--layer_idx` | `47` | Model layer to visualize (default: `47`, the last layer). |
| `--top_k` | `50` | Maximum number of attention edges to show per head. |
| `--attn_map_dir` | Auto-derived | Override path to attention map directory. Auto-derived from `output_dir` if not set. |

#### Inference Options

| Argument | Default | Description |
|---|---|---|
| `--base_data_dir` | — | Root directory for AlphaFold databases. Required unless `--visualize_only`. |
| `--alignment_dir` | — | Path to precomputed alignments directory (output of Stage 1). |
| `--config_preset` | `model_1_ptm` | OpenFold model config preset. |
| `--model_device` | `cuda:0` | CUDA device for inference. |
| `--num_recycles_save` | `1` | Number of recycles to save during inference. |
| `--visualize_only` | `false` | Skip inference and generate visualizations from existing attention maps and PDB. |

#### Execution

| Argument | Default | Description |
|---|---|---|
| `--log_level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. |

### Output

All outputs are written under `--output_dir`:

| Path | Contents |
|---|---|
| `predictions/` | Predicted PDB structures from inference |
| `attention_files_<tag>/` | Raw attention map `.npy` files |
| `attention_images_<tag>/msa_row_attention_plots/` | MSA row attention 3D + arc diagrams |
| `attention_images_<tag>/tri_start_attention_plots/` | Triangle start attention 3D + arc diagrams |
| `attention_images_<tag>/combined/` | Combined panel images |
| `vizfold_viz_manifest.json` | Run settings for reproducibility |
---

## Additional Utility CLIs

In addition to the three primary VizFold pipeline stages (alignment → inference → visualization), VizFold includes two auxiliary CLI tools for working with FASTA files and alignment directories. These tools are optional but useful for dataset preparation, large-scale analysis, and reproducible workflows.

---

## FASTA Clustering CLI

**Script:** `vizfold_cli_fasta_to_clusterfile.py`  
**Stage:** 0 — Pre-process (optional)  
**Purpose:** Cluster protein sequences using mmseqs2 with PDB-style parameters.

This tool clusters a FASTA file of `{PDB_ID}_{CHAIN_ID}` sequences using the same mmseqs2 settings used by the PDB for official sequence clusters. It produces a clean, reformatted cluster file where each line lists all chains belonging to a cluster.

### When Should I Use This?

Use this tool when you want to:

- Reduce redundancy in a large FASTA dataset  
- Group homologous chains before running large-scale inference  
- Reproduce PDB-style sequence clusters for benchmarking or analysis  
- Prepare clustered datasets for training or evaluation  

This CLI is **not required** for the main VizFold pipeline, but it is extremely useful for dataset curation and large-scale workflows.

### Quick Start

```bash
python cli/vizfold_cli_fasta_to_clusterfile.py \
    sequences.fasta clusters.txt /path/to/mmseqs \
    --seq-id 0.4
```

### Output

- `clusters.txt` — each line contains a space-separated list of `{PDB_ID}_{CHAIN_ID}` belonging to the same cluster  
- Temporary mmseqs2 files are cleaned automatically  

---

## Alignment-to-FASTA CLI

**Script:** `vizfold_cli_align_fasta.py`  
**Stage:** 1.5 — Post-align (optional)  
**Purpose:** Extract representative sequences from alignment directories or alignment-DB index files and write them to a FASTA file.

This tool converts either:

- a directory of per-chain alignment folders (e.g., outputs from Stage 1), or  
- a compressed alignment-DB index  

into a single FASTA file containing one sequence per chain.

### When Should I Use This?

Use this tool when you want to:

- Reconstruct FASTA files from alignment directories  
- Extract sequences from alignment-DBs for dataset creation  
- Prepare FASTA inputs for clustering or downstream analysis  
- Inspect or validate the sequences used during alignment  

This CLI is **not part of the core 3-stage VizFold pipeline**, but it is valuable for dataset inspection, preprocessing, and analysis workflows.

### Quick Start (alignment directory)

```bash
python cli/vizfold_cli_align_fasta.py output.fasta \
    --alignment-dir alignments/
```

### Quick Start (alignment-DB index)

```bash
python cli/vizfold_cli_align_fasta.py output.fasta \
    --alignment-db-index alignment_index.json
```

### Output

- `output.fasta` — one FASTA entry per chain, using the first available alignment file (`mgnify_hits.a3m`, `uniref90_hits.a3m`, or `bfd_uniclust_hits.a3m`)  

---

## How These Tools Fit Into the VizFold Pipeline

These CLIs are **optional utilities** that complement the main VizFold workflow:

- **Before Stage 1 (Precompute Alignments):**  
  - `vizfold_cli_fasta_to_clusterfile.py` can cluster input sequences to reduce redundancy or group homologs.

- **After Stage 1 (Alignment Generation):**  
  - `vizfold_cli_align_fasta.py` can reconstruct FASTA files from alignment directories for dataset inspection or downstream processing.

They are not required for the core alignment → inference → visualization pipeline, but they provide useful functionality for dataset preparation, large-scale analysis, and reproducible research workflows.

---

## Shared Reference

### Database Paths

Required for alignment unless `--use_precomputed_alignments` (Stage 2) is set or `--visualize_only` (Stage 3) is used.

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

Override auto-detected paths from the active conda environment. All optional.

| Argument | Description |
|---|---|
| `--jackhmmer_binary_path` | Path to jackhmmer binary. |
| `--hhblits_binary_path` | Path to hhblits binary. |
| `--hhsearch_binary_path` | Path to hhsearch binary. |
| `--hmmsearch_binary_path` | Path to hmmsearch binary. |
| `--hmmbuild_binary_path` | Path to hmmbuild binary. |
| `--kalign_binary_path` | Path to kalign binary. |

### Model Presets

The `--config_preset` argument selects a model architecture defined in `openfold/config.py`.

| Preset | Description |
|---|---|
| `model_1` through `model_5` | Standard monomer models. |
| `model_1_ptm` through `model_5_ptm` | Monomer models with pTM (predicted TM-score) head. |
| `model_1_multimer_v3` through `model_5_multimer_v3` | Multimer models. |
| `seq_model_esm1b` | Single-sequence model using ESM-1b embeddings. |
| `seq_model_esm1b_ptm` | Single-sequence model with pTM head. |
| `seqemb_initial_training` | Sequence-embedding initial training config. |
| `seqemb_finetuning` | Sequence-embedding fine-tuning config. |

---

## Help

```bash
# Stage 1
python cli/vizfold_cli_precompute_align.py --help

# Stage 2
python cli/vizfold_cli_inference.py --help

# Stage 3
python cli/vizfold_cli_viz.py --help
```
