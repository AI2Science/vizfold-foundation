"""Residue-indexed line plot for 1-D OpenFold representations.

Typical inputs:
    - A single channel of the single-representation: ``s[:, c]``, shape ``(N,)``.
    - A row of an attention matrix (one residue's outgoing weights), shape ``(N,)``.
    - Per-residue scalar metrics (pLDDT, attention entropy, etc.).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from matplotlib.figure import Figure

from viz.plots.common import add_residue_axes, new_figure, save_or_return


def plot_line(
    values: np.ndarray,
    *,
    x: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    xlabel: str = "residue",
    ylabel: str = "value",
    color: str = "tab:blue",
    highlight_residues: Optional[Iterable[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a 1-D residue-indexed signal.

    Parameters
    ----------
    values:
        1-D numpy array of length ``N``.
    x:
        Optional residue index vector. Defaults to ``np.arange(len(values))``
        so the x-axis is residue index by default.
    title, xlabel, ylabel:
        Standard text labels.
    color:
        Forwarded to ``ax.plot``.
    highlight_residues:
        Iterable of residue indices to mark with vertical reference lines.
    save_path:
        If provided, the figure is written to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
        The constructed figure.
    """
    arr = np.asarray(values).squeeze()
    if arr.ndim != 1:
        raise ValueError(
            f"plot_line expects a 1-D array, got shape {np.asarray(values).shape!r}"
        )

    n = arr.shape[0]
    xs = np.arange(n) if x is None else np.asarray(x)
    if xs.shape != arr.shape:
        raise ValueError(
            f"x has shape {xs.shape!r} but values has shape {arr.shape!r}"
        )

    fig, ax = new_figure(figsize=(8.0, 3.5))
    ax.plot(xs, arr, color=color, linewidth=1.4)
    ax.set_xlim(xs[0], xs[-1] if n > 1 else xs[0] + 1)

    add_residue_axes(ax, n_rows=n, n_cols=n)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=11)

    if highlight_residues is not None:
        for r in highlight_residues:
            r_int = int(r)
            if 0 <= r_int < n:
                ax.axvline(r_int, color="red", linewidth=0.8, alpha=0.7)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    return save_or_return(fig, save_path)
