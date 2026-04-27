"""Residue-indexed heatmap (image plot) for OpenFold representations.

Typical inputs:
    - Attention map slice ``(N, N)`` for a chosen layer/head.
    - Pair-representation channel ``z[:, :, c]`` of shape ``(N, N)``.
    - MSA-representation channel ``m[:, :, c]`` of shape ``(S, N)`` (rectangular allowed).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from matplotlib.figure import Figure

from viz.plots.common import (
    add_colorbar,
    add_residue_axes,
    new_figure,
    normalize,
    save_or_return,
)


def plot_heatmap(
    matrix: np.ndarray,
    *,
    title: Optional[str] = None,
    xlabel: str = "residue j",
    ylabel: str = "residue i",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: Optional[str] = None,
    highlight_residues: Optional[Iterable[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Plot a 2-D matrix as a residue-indexed heatmap.

    Parameters
    ----------
    matrix:
        2-D numpy array. Square ``(N, N)`` for attention / pair channels;
        rectangular ``(R, C)`` is also accepted (e.g. MSA channel ``(S, N)``).
    title, xlabel, ylabel:
        Standard text labels. Axis defaults assume residue indexing.
    cmap, vmin, vmax:
        Forwarded to ``imshow``. ``vmin`` / ``vmax`` default to the array's
        own min / max via :func:`viz.plots.common.normalize`.
    colorbar_label:
        Optional label for the colorbar (e.g. ``"attention weight"``).
    highlight_residues:
        Iterable of residue indices to mark with red gridlines (used for
        triangle-attention "query residue" overlays).
    save_path:
        If provided, the figure is written to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
        The constructed figure. Callers can embed or further modify it.
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(
            f"plot_heatmap expects a 2-D array, got shape {arr.shape!r}"
        )

    lo, hi = normalize(arr, vmin=vmin, vmax=vmax)

    fig, ax = new_figure(figsize=(6.0, 5.0))
    im = ax.imshow(
        arr,
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
        aspect="auto",
        origin="lower",
    )

    n_rows, n_cols = arr.shape
    add_residue_axes(ax, n_rows=n_rows, n_cols=n_cols)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title is not None:
        ax.set_title(title, fontsize=11)

    if highlight_residues is not None:
        for r in highlight_residues:
            r_int = int(r)
            if 0 <= r_int < n_cols:
                ax.axvline(r_int, color="red", linewidth=0.8, alpha=0.7)
            if 0 <= r_int < n_rows:
                ax.axhline(r_int, color="red", linewidth=0.8, alpha=0.7)

    add_colorbar(fig, im, label=colorbar_label)
    return save_or_return(fig, save_path)
