# `viz` — residue-indexed plot utilities

Generic plot functions for visualising OpenFold representations, scoped to
issue #8 deliverable: *"Provide functionalities for different types of
visualizations - including image and line plots."*

Every function takes plain numpy arrays in and returns a
`matplotlib.figure.Figure`, so the same code can be embedded in notebooks,
served by the Flask UI in [`web_interface.py`](web_interface.py), or
composed with the existing `visualize_attention_*` helpers without changes.

## Public API

```python
from viz import (
    plot_heatmap,
    plot_heatmap_grid,
    plot_line,
    plot_lines,
    plot_layer_trajectory,
    plot_histogram,
)
```

| Function | Input shape | What it shows |
|---|---|---|
| `plot_heatmap` | `(R, C)` | one residue-by-residue map (attention slice, pair channel, MSA channel). |
| `plot_heatmap_grid` | `(K, R, C)` or list of `(R, C)` | K heatmaps in a grid (e.g. all heads of one layer). |
| `plot_line` | `(N,)` | a single residue-indexed signal (one channel of `s`, one row of an attention map, pLDDT, ...). |
| `plot_lines` | `(K, N)` or list of `(N,)` | K residue-indexed signals overlaid on one axis. |
| `plot_layer_trajectory` | `(L, N)` | one channel's value across `L` layers, drawing one line per chosen residue. |
| `plot_histogram` | any shape | distribution of values (flattened internally), useful for activation summaries. |

## Examples

### Heatmap

```python
import numpy as np
from viz import plot_heatmap

attn = np.random.rand(64, 64)
fig = plot_heatmap(
    attn,
    title="MSA-row attention, layer 47, head 2",
    colorbar_label="attention weight",
    highlight_residues=[18],
    save_path="outputs/heatmap_msa_row_layer47.png",
)
```

![heatmap example](examples/heatmap_demo.png)

### Heatmap grid

```python
from viz import plot_heatmap_grid
from viz._fakes import fake_attention_heads

heads = fake_attention_heads(N=64, H=8)
plot_heatmap_grid(
    heads,
    titles=[f"head {i}" for i in range(8)],
    ncols=4,
    suptitle="MSA-row attention, layer 47",
    colorbar_label="attention weight",
    save_path="outputs/heatmap_grid_layer47.png",
)
```

![heatmap grid example](examples/heatmap_grid_demo.png)

### Line plot

```python
import numpy as np
from viz import plot_line

channel = np.random.randn(64)
fig = plot_line(
    channel,
    title="single representation, channel 12",
    ylabel="activation",
    highlight_residues=[18],
    save_path="outputs/lineplot_single_ch12.png",
)
```

![lineplot example](examples/lineplot_demo.png)

### Multi-line overlay

```python
from viz import plot_lines
from viz._fakes import fake_single_channels

s = fake_single_channels(N=64, C=6)
plot_lines(
    s.T,
    labels=[f"channel {c}" for c in range(6)],
    title="single representation, 6 channels",
    save_path="outputs/lines_single.png",
)
```

![lines example](examples/lines_demo.png)

### Layer trajectory

```python
from viz import plot_layer_trajectory
from viz._fakes import fake_layer_trajectory

traj = fake_layer_trajectory(L=48, N=64)
plot_layer_trajectory(
    traj,
    residue_indices=[0, 16, 32, 48],
    title="one channel across 48 layers",
    save_path="outputs/layer_trajectory.png",
)
```

![layer trajectory example](examples/layer_trajectory_demo.png)

### Histogram

```python
from viz import plot_histogram
from viz._fakes import fake_distribution

plot_histogram(
    fake_distribution(N=20000),
    bins=60,
    title="layer 47 activation distribution",
    save_path="outputs/hist_layer47.png",
)
```

![histogram example](examples/histogram_demo.png)

## End-to-end: raw OpenFold tensors → figure

Once the model is producing real `pair`, `msa`, and `single` representations,
the cleanest path is to call the bridge helpers in
[`viz/integrations.py`](integrations.py). They wrap Priyavi's
[`representation_tensor_utils`](../representation_tensor_utils.py) so the
same call site works for any representation kind:

```python
from viz import (
    heatmap_from_representation,
    line_from_representation,
    lines_from_representation,
    pair_channel_grid,
)

# pair: shape (..., N, N, C_z)  -> residue × residue heatmap of one channel
heatmap_from_representation(
    z, kind="pair", channel=12,
    save_path="outputs/pair_ch12.png",
)

# single: shape (N, C_s)        -> per-residue line for one channel
line_from_representation(
    s, kind="single", channel=4, ylabel="activation",
)

# single: shape (N, C_s)        -> overlay several channels
lines_from_representation(
    s, kind="single", channels=[0, 4, 8, 11],
    save_path="outputs/single_overlay.png",
)

# pair: render the first 8 channels in a 4-column grid
pair_channel_grid(
    z, channels=list(range(8)), ncols=4,
    suptitle="pair representation: 8 channels",
    save_path="outputs/pair_grid.png",
)
```

Behavior delegated to `representation_tensor_utils`:

- Validation / shape checks (`pair` is square, leading batch dims squeezed).
- Channel selection vs. `mean`/`max`/`l2` aggregation across the channel axis.
- `minmax` / `zscore` / `none` normalization for display.
- For `kind="msa"` line plots: depth-averaged per residue.
- For `kind="pair"` line plots: diagonal slice at the chosen channel.

| Bridge function | Calls upstream | Then plots with |
|---|---|---|
| `heatmap_from_representation` | `prepare_heatmap_data` | `plot_heatmap` |
| `line_from_representation` | `prepare_lineplot_data` | `plot_line` |
| `lines_from_representation` | `prepare_lineplot_data` (per channel) | `plot_lines` |
| `pair_channel_grid` | `prepare_heatmap_data` (per channel) | `plot_heatmap_grid` |

![pair channel example](examples/from_representation_pair.png)
![pair grid example](examples/from_representation_pair_grid.png)
![single overlay example](examples/from_representation_lines.png)

## End-to-end: `EvoformerRunArtifact` → figure

The instrumentation in
[`openfold/utils/evoformer_instrumentation.py`](../openfold/utils/evoformer_instrumentation.py)
captures intermediate `m` / `pair` tensors and sparse top-K attention text
files during a model run. The wrapper around those outputs is
[`EvoformerRunArtifact`](../openfold/utils/evoformer_run_artifact.py).
The bridge in `viz/integrations.py` reads from that artifact directly:

```python
from openfold.utils.evoformer_run_artifact import EvoformerRunArtifact
from viz import (
    attention_heatmap_from_artifact,
    representation_heatmap_from_artifact,
    representation_line_from_artifact,
    figure_from_artifact,
)

art = EvoformerRunArtifact(
    run_dir="outputs/run_6KWC",
    attention_dir="outputs/attention_files_6KWC_demo_tri_18",
    reps_path="outputs/run_6KWC/reps.pt",
)

# attention path: text file -> N x N matrix -> heatmap
attention_heatmap_from_artifact(
    art, "msa_row_attn", layer=24, mean_across_heads=True,
    save_path="outputs/heatmap_msa_row_layer24.png",
)

# representation path: reps.pt[layer_LL.pair] -> prepare_heatmap_data -> plot_heatmap
representation_heatmap_from_artifact(
    art, "pair", layer=47, channel=12, normalize="minmax",
)

# residue-indexed line: reps.pt[layer_LL.msa] -> prepare_lineplot_data -> plot_line
representation_line_from_artifact(art, "msa", layer=47, channel=4)

# unified dispatcher: pick attn_kind= XOR rep_kind=
figure_from_artifact(art, attn_kind="triangle_start_attn", layer=47, residue_idx=18)
figure_from_artifact(art, rep_kind="pair", layer=47, aggregate="l2")
```

| Bridge function | Source on disk | Then renders with |
|---|---|---|
| `attention_heatmap_from_artifact` | `attention_dir/{kind}_layer{L}*.txt` | `plot_heatmap` |
| `representation_heatmap_from_artifact` | `reps.pt[layer_{LL:02d}.{kind}]` | `heatmap_from_representation` |
| `representation_line_from_artifact` | `reps.pt[layer_{LL:02d}.{kind}]` | `line_from_representation` |
| `representation_tensor_from_artifact` | `reps.pt[...]` | (raw numpy, for custom plots) |
| `figure_from_artifact` | dispatches to one of the above | — |

The artifact bridge requires `torch` (for loading `reps.pt`); the rest of
`viz` does not. End-to-end correctness is covered by
[`tests/test_viz_artifact_bridge.py`](../tests/test_viz_artifact_bridge.py),
which builds a synthetic on-disk artifact and checks that what gets painted
on the Figure matches `artifact.get_attention_matrix(...)` and
`prepare_*` bit for bit.

## Conventions

- All functions return a `Figure` and never call `plt.show()`.
- Pass `save_path=...` to persist a PNG (parent dirs created on demand).
- Axes default to residue indexing (`residue i`, `residue j`, `residue`).
- Wrong-rank inputs raise `ValueError` early with a message that includes the actual shape and what was expected.
- `plot_lines` receiving a 1-D array raises `ValueError` directing the caller to use `plot_line()` instead.

### Flask UI configuration

`web_interface.py` reads its defaults from environment variables so it can be pointed at any run without editing the source:

| Variable | Default | Effect |
|---|---|---|
| `VIZFOLD_IMAGE_DIR` | `./outputs/attention_images_6KWC_demo_tri_18` | Directory of heatmap PNGs to serve |
| `VIZFOLD_PROT` | `6KWC` | Protein name shown in the page heading |
| `VIZFOLD_TRI_IDX` | `18` | Triangle-attention query residue index |
| `VIZFOLD_NUM_LAYERS` | `48` | Number of layers in the layer dropdown |

```bash
VIZFOLD_IMAGE_DIR=outputs/attention_images_7ABC_demo_tri_22 \
VIZFOLD_PROT=7ABC VIZFOLD_TRI_IDX=22 \
    python viz/web_interface.py
```

## Synthetic tensors (`viz._fakes`)

While the model-side extraction layer is still in progress,
[`viz/_fakes.py`](_fakes.py) provides shape-matching synthetic tensors so the
plot functions (and the `*_from_representation` bridge above) can be
exercised end-to-end. Once real `m` / `z` / `s` tensors arrive from the
extraction step, the bridge functions consume them directly through
`representation_tensor_utils` -- no code change here.

| Helper | Shape | Stand-in for |
|---|---|---|
| `fake_attention_heads(N, H)` | `(H, N, N)` | per-head attention map |
| `fake_pair_channel(N)` | `(N, N)` | one channel of the pair representation `z` |
| `fake_single_channels(N, C)` | `(N, C)` | multi-channel single representation `s` |
| `fake_layer_trajectory(L, N)` | `(L, N)` | one channel across layers |
| `fake_distribution(N)` | `(N,)` | flat values for histograms |

## Demo

[`notebooks/viz_plot_demo.ipynb`](../notebooks/viz_plot_demo.ipynb) exercises
every plot function and writes the example PNGs above.

## Wiring into the Flask UI

[`web_interface.py`](web_interface.py) serves PNGs from
`outputs/attention_images_<PROT>_demo_tri_<R>/` named:

```
heatmap_msa_row_layer{N}.png
heatmap_triangle_start_layer{N}.png
```

`viz/render_for_ui.py` is the script that **populates** this directory by
calling `plot_heatmap` for every (attention type, layer) pair. The
attention-map source is auto-detected:

```bash
# Real attention maps from a save_attention_topk run:
python -m viz.render_for_ui --mode real \
    --attn-dir outputs/attention_files_6KWC_demo_tri_18

# Synthetic placeholders (so the UI works before any model run):
python -m viz.render_for_ui --mode demo --n-res 96

# Auto-detect: real if files exist, otherwise demo.
python -m viz.render_for_ui
```

The script writes 96 PNGs (48 layers × 2 attention types). Real mode
parses the sparse top-K text files written by `save_attention_topk`,
reconstructs an `N x N` matrix per layer (mean over heads), and renders
through `plot_heatmap`. No changes to `web_interface.py` are required
for either mode — only the contents of the image directory differ.
