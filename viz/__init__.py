"""Visualization helpers for OpenFold residue-level representations.

Public API:
    plot_heatmap: 2-D image plot (e.g. attention map, pair representation channel).
    plot_line:    1-D line plot over residue index (e.g. single-representation channel).

Both functions accept plain numpy arrays and return a matplotlib ``Figure``,
so they can be embedded in notebooks, web frontends, or saved to disk via
the ``save_path`` keyword argument.
"""

from viz.plots.heatmap import plot_heatmap
from viz.plots.lineplot import plot_line

__all__ = ["plot_heatmap", "plot_line"]
