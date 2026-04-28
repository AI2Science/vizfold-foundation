"""Residue-indexed heatmap (image plot) for OpenFold representations.

Typical inputs:
    - Attention map slice ``(N, N)`` for a chosen layer/head.
    - Pair-representation channel ``z[:, :, c]`` of shape ``(N, N)``.
    - MSA-representation channel ``m[:, :, c]`` of shape ``(S, N)`` (rectangular allowed).

Also provides :func:`plot_heatmap_grid` for drawing K matrices side-by-side
(e.g. all heads of one attention layer).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from viz.plots.common import (
    add_colorbar,
    add_residue_axes,
    draw_highlight_lines,
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

    draw_highlight_lines(ax, highlight_residues, n_cols=n_cols, n_rows=n_rows)
    add_colorbar(fig, im, label=colorbar_label)
    return save_or_return(fig, save_path)


def _coerce_matrix_stack(
    matrices: Union[np.ndarray, Sequence[np.ndarray]],
) -> List[np.ndarray]:
    """Normalize ``(K, R, C)`` arrays or lists of 2-D arrays to a list of 2-D arrays."""
    if isinstance(matrices, np.ndarray):
        if matrices.ndim == 3:
            if matrices.shape[0] == 0:
                raise ValueError("plot_heatmap_grid received a (0, R, C) array")
            return [matrices[i] for i in range(matrices.shape[0])]
        if matrices.ndim == 2:
            return [matrices]
        raise ValueError(
            f"plot_heatmap_grid expects a 3-D array (K, R, C) or list of 2-D arrays; "
            f"got shape {matrices.shape!r}"
        )

    arrs = [np.asarray(m) for m in matrices]
    if not arrs:
        raise ValueError("plot_heatmap_grid received an empty matrices list")
    if any(m.ndim != 2 for m in arrs):
        raise ValueError("plot_heatmap_grid expects each matrix to be 2-D")
    return arrs


def plot_heatmap_grid(
    matrices: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    titles: Optional[Sequence[str]] = None,
    ncols: int = 4,
    suptitle: Optional[str] = None,
    cmap: str = "viridis",
    shared_clim: bool = True,
    colorbar_label: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Draw K residue-indexed heatmaps in a grid.

    Parameters
    ----------
    matrices:
        Either a 3-D array ``(K, R, C)`` or a sequence of 2-D arrays. Common
        case: all H heads of an attention layer, ``(H, N, N)``.
    titles:
        Optional per-cell titles, length must equal K.
    ncols:
        Number of columns in the grid; rows are computed automatically.
    shared_clim:
        If True (default), all cells share the same colormap range so they're
        visually comparable, and a single colorbar is placed alongside.
    """
    mats = _coerce_matrix_stack(matrices)
    K = len(mats)
    if titles is not None and len(titles) != K:
        raise ValueError(f"titles length {len(titles)} does not match K {K}")

    ncols = max(1, min(int(ncols), K))
    nrows = math.ceil(K / ncols)

    if shared_clim:
        stacked = np.stack([np.asarray(m).ravel() for m in mats])
        vmin, vmax = normalize(stacked)
    else:
        vmin = vmax = None

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(2.6 * ncols + 1.0, 2.4 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = axes.flatten()

    last_im = None
    for i, ax in enumerate(flat_axes):
        if i >= K:
            ax.axis("off")
            continue
        m = mats[i]
        lo, hi = (vmin, vmax) if shared_clim else normalize(m)
        im = ax.imshow(
            m,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
            aspect="auto",
            origin="lower",
        )
        last_im = im
        if titles is not None:
            ax.set_title(titles[i], fontsize=9)
        else:
            ax.set_title(f"#{i}", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

    if shared_clim and last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12)

    return save_or_return(fig, save_path)
