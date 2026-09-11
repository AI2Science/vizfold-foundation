# Heatmap Visualizations

This page outlines how to generate the per-head attention heatmap grids that
appear in the demo notebook (`viz_attention_demo_base.ipynb`). The same helper
functions can be invoked from a shell to visualize attention outside the
notebook environment.

## Requirements

- Attention text files exported by OpenFold, located in a directory such as
  `./outputs/attention_files_<PROT>_demo_tri_<IDX>`.
- The FASTA file that was used for inference (needed to determine the sequence
  length). If you followed the notebook, this lives at
  `./examples/monomer/fasta_dir_<PROT>/<PROT>.fasta`.
- A Python environment with the project dependencies (Plotly, NumPy, etc.).

## Generating a Heatmap Grid

Use the existing `visualize_attention_heatmap_grid.visualize_layer_attention`
function. Example command:

- You can also run the helper outside the notebook by dropping the snippet
  below into a Python file or one-off script; it will parse the FASTA, render
  the heatmap, and save the Plotly HTML output just like the notebook cell.

```bash
python - <<'PY'
from visualize_attention_heatmap_grid import visualize_layer_attention
from visualize_attention_arc_diagram_demo_utils import parse_fasta_sequence

PROT = "6KWC"
TRI_RESIDUE_IDX = 18
ATTN_MAP_DIR = f"./outputs/attention_files_{PROT}_demo_tri_{TRI_RESIDUE_IDX}"
FASTA_PATH = f"./examples/monomer/fasta_dir_{PROT}/{PROT}.fasta"
IMAGE_OUTPUT_DIR = f"./outputs/attention_images_{PROT}_demo_tri_{TRI_RESIDUE_IDX}"

seq = parse_fasta_sequence(FASTA_PATH)
fig = visualize_layer_attention(
    attention_dir=ATTN_MAP_DIR,
    seq_len=len(seq),
    layer_idx=47,
    attention_type="msa_row",
    output_dir=f"{IMAGE_OUTPUT_DIR}/attention_heatmap_grids",
    threshold=None,
)

if fig:
    fig.show()
PY
```

The output HTML file is written to the specified `output_dir`. When run in a
notebook, calling `fig.show()` renders the interactive Plotly grid inline.

## Supported Attention Types

`visualize_layer_attention` supports two kinds of OpenFold attention dumps:

- `msa_row`: Shows row attention over the multiple sequence alignment.
- `triangle_start`: Shows triangle-start attention for a specific residue.

Triangle attention requires a `residue_idx` argument to target the residue of
interest. Supply whichever type matches the attention files you generated.
