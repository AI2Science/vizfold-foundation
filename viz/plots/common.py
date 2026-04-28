"""Shared utilities for the viz plot functions.

Kept tiny and side-effect-free: no ``plt.show()``, no global state. Every
helper is meant to be composed inside ``plot_heatmap`` / ``plot_line``.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage


def save_or_return(fig: Figure, save_path: Optional[str]) -> Figure:
    """Persist ``fig`` to disk if ``save_path`` is given, then return it.

    The figure is always returned so callers (notebooks, web frontends) can
    embed it directly. Parent directories are created on demand.
    """
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def normalize(
    matrix: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve colorbar limits, falling back to the array's own min/max.

    Returns ``(vmin, vmax)``. If the resolved range is degenerate (vmin == vmax)
    the bounds are nudged apart so matplotlib doesn't divide by zero.
    """
    arr = np.asarray(matrix)
    lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
    hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
    if lo == hi:
        hi = lo + max(1e-9, abs(lo) * 1e-6)
    return lo, hi


def add_residue_axes(
    ax: Axes,
    n_rows: int,
    n_cols: Optional[int] = None,
    *,
    max_ticks: int = 20,
) -> None:
    """Place integer residue ticks on the axes.

    Heatmaps pass both ``n_rows`` and ``n_cols``; line plots pass only
    ``n_rows`` (treated as residue count along x). At most ``max_ticks`` are
    rendered per axis to keep long sequences legible.
    """
    if n_cols is None:
        n_cols = n_rows

    def _ticks(n: int) -> np.ndarray:
        if n <= max_ticks:
            return np.arange(n)
        step = max(1, n // max_ticks)
        return np.arange(0, n, step)

    x_ticks = _ticks(n_cols)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(t) for t in x_ticks], fontsize=8)

    if ax.images or ax.collections:
        y_ticks = _ticks(n_rows)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(t) for t in y_ticks], fontsize=8)


def draw_highlight_lines(
    ax: Axes,
    highlight_residues: Optional[Iterable[int]],
    *,
    n_cols: int,
    n_rows: Optional[int] = None,
) -> None:
    """Draw red reference lines for each residue index in *highlight_residues*.

    Vertical lines are drawn when the index is within ``[0, n_cols)``.
    Horizontal lines are drawn when ``n_rows`` is given and the index is within
    ``[0, n_rows)`` — used by heatmaps to cross-hair a query residue.
    """
    if highlight_residues is None:
        return
    for r in highlight_residues:
        r_int = int(r)
        if 0 <= r_int < n_cols:
            ax.axvline(r_int, color="red", linewidth=0.8, alpha=0.7)
        if n_rows is not None and 0 <= r_int < n_rows:
            ax.axhline(r_int, color="red", linewidth=0.8, alpha=0.7)


def add_colorbar(fig: Figure, im: AxesImage, label: Optional[str] = None) -> None:
    """Attach a thin colorbar to ``im`` on ``fig``."""
    cbar = fig.colorbar(im, ax=im.axes, fraction=0.046, pad=0.04)
    if label is not None:
        cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def new_figure(figsize: Tuple[float, float] = (6.0, 5.0)) -> Tuple[Figure, Axes]:
    """Construct a matplotlib ``(fig, ax)`` pair with sane defaults."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax
