# Boltz inference and tracing (VizFold)

This doc describes how to run **Boltz-2** inference and extract **attention-style traces** in the same text format used by VizFold’s arc visualization utilities. The repo includes a minimal example input for validation, which can be overriden IN_YAML/IN_FASTA to run on any target.

## Environment notes (ICE/H100)
- Boltz runs on GPU.
- On ICE/H100, run with `--no_kernels` to avoid CUDA kernel / cuBLAS symbol issues.

## Output layout
A run produces:
- `pred/` : structure prediction outputs (e.g., CIF, pLDDT, PAE)
- `attn_txt/` : attention-style trace text files
- `arc_png/` : arc diagram PNGs generated from `attn_txt/`

## Trace text formats

### MSA row attention
- File: `msa_row_attn_layer{L}.txt`
- Contains multiple heads:
  - header line: `Layer {L} Head {h}`
  - then edges: `i j weight`

### Triangle attention (per residue)
- Files:
  - `triangle_start_attn_layer{L}_residue_idx_{r}.txt`
  - `triangle_end_attn_layer{L}_residue_idx_{r}.txt`
- Each file contains multiple heads:
  - header line: `Layer {L} Head {h}`
  - then edges: `i j weight` (with `i == r`)

## Running on SLURM

Run:
```bash
sbatch scripts/boltz/run_boltz_trace.sbatch
sbatch --export=ALL,BOLTZ_TRACE_LAYERS=0,1,2,3,BOLTZ_TRACE_RESIDUES=18,39,51,79,BOLTZ_TRACE_TOPK=50,BOLTZ_TRACE_HEAD=all \
  scripts/boltz/run_boltz_trace.sbatch
python -m py_compile visualize_attention_arc_diagram_demo_utils.py && echo "utils OK"
```
