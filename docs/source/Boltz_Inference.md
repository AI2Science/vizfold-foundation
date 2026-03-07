# Boltz inference and tracing (VizFold)

This doc describes how to run **Boltz-2** inference and extract **attention-style traces** in the same text format used by VizFold’s arc visualization utilities.

The repo ships a small example input under `scripts/boltz/inputs/` for validation. To run on your own target, override `IN_YAML` / `IN_FASTA` (see below).

## Environment notes (ICE/H100)
- Boltz runs on GPU.
- On ICE/H100, run with `--no_kernels` to avoid CUDA kernel / cuBLAS symbol issues (the provided runner already uses this).

## Output layout
A run produces:
- `pred/` : structure prediction outputs (e.g., CIF, pLDDT, PAE)
- `attn_txt/` : attention-style trace text files
- `arc_png/` : arc diagram PNGs generated from `attn_txt/`
- `act_npz/` : lightweight activation summaries (optional; produced by the tracer)

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

## Configuration knobs

### Inputs
Default example inputs live here:
- `scripts/boltz/inputs/input.yaml`
- `scripts/boltz/inputs/input.fasta`

To run on your own target, override:
```bash
export IN_YAML=/path/to/your_input.yaml
export IN_FASTA=/path/to/your_input.fasta
```
