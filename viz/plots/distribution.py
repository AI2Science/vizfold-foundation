"""Distribution plots over numpy tensors.

Currently exposes :func:`plot_histogram`. The input tensor can be any shape
(it's flattened internally) so this works equally well for "all attention
weights of one head", "one channel of the single representation across all
residues", or "all activations in a layer".
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.figure import Figure

from viz.plots.common import new_figure, save_or_return


def plot_histogram(
    values: np.ndarray,
    *,
    bins: int = 50,
    title: Optional[str] = None,
    xlabel: str = "value",
    ylabel: str = "count",
    log: bool = False,
    color: str = "tab:blue",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot the value distribution of a numpy tensor.

    Parameters
    ----------
    values:
        Any-shape numpy array. NaNs are dropped before binning.
    bins:
        Number of histogram bins.
    log:
        If True, the y-axis is plotted on a log scale.
    """
    arr = np.asarray(values).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("plot_histogram received no finite values")

    fig, ax = new_figure(figsize=(7.0, 3.5))
    ax.hist(arr, bins=int(bins), color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=11)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    return save_or_return(fig, save_path)
