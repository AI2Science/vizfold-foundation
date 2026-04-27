"""Visualization helpers for OpenFold residue-level representations.

Public API:
    plot_heatmap:           2-D image plot.
    plot_heatmap_grid:      grid of 2-D heatmaps (e.g. all heads of a layer).
    plot_line:              1-D line plot over residue index.
    plot_lines:             multi-channel residue-indexed overlay.
    plot_layer_trajectory:  one channel's value across layers (line per residue).
    plot_histogram:         value distribution of a tensor slice.

Both kinds of functions accept plain numpy arrays and return a matplotlib
``Figure``, so they can be embedded in notebooks, web frontends, or saved to
disk via the ``save_path`` keyword argument.
"""

from viz.plots.distribution import plot_histogram
from viz.plots.heatmap import plot_heatmap, plot_heatmap_grid
from viz.plots.lineplot import plot_layer_trajectory, plot_line, plot_lines

__all__ = [
    "plot_heatmap",
    "plot_heatmap_grid",
    "plot_line",
    "plot_lines",
    "plot_layer_trajectory",
    "plot_histogram",
]
