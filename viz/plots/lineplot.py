"""Residue-indexed line plots for 1-D OpenFold representations.

Typical inputs:
    - A single channel of the single-representation: ``s[:, c]``, shape ``(N,)``.
    - A row of an attention matrix (one residue's outgoing weights), shape ``(N,)``.
    - Per-residue scalar metrics (pLDDT, attention entropy, etc.).
    - Multiple channels overlaid: ``s[:, [c1, c2, ...]]``, shape ``(N, K)``.
    - One channel's value across all layers: ``(L, N)`` per-layer per-residue.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

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


def _stack_series(series: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:
    """Coerce ``(K, N)`` array or list of 1-D arrays into a single 2-D array."""
    if isinstance(series, np.ndarray):
        if series.ndim != 2:
            raise ValueError(
                f"plot_lines expects a 2-D array (n_lines, N) when given an "
                f"ndarray; got shape {series.shape!r}"
            )
        return series

    arrs = [np.asarray(s).squeeze() for s in series]
    if not arrs:
        raise ValueError("plot_lines received an empty series list")
    if any(a.ndim != 1 for a in arrs):
        raise ValueError("plot_lines expects each entry of `series` to be 1-D")
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise ValueError("plot_lines entries must share the same length")
    return np.stack(arrs, axis=0)


def plot_lines(
    series: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    labels: Optional[Sequence[str]] = None,
    x: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    xlabel: str = "residue",
    ylabel: str = "value",
    highlight_residues: Optional[Iterable[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Overlay several residue-indexed signals on one axis.

    Parameters
    ----------
    series:
        Either a 2-D array of shape ``(n_lines, N)`` or a sequence of 1-D
        arrays of equal length.
    labels:
        Optional legend labels, one per line. Length must match ``n_lines``.
    x:
        Optional shared residue-index vector. Defaults to ``np.arange(N)``.
    highlight_residues:
        Iterable of residue indices to mark with red vertical reference lines.
    save_path:
        If provided, the figure is written to this path before being returned.
    """
    arr = _stack_series(series)
    n_lines, n = arr.shape
    xs = np.arange(n) if x is None else np.asarray(x)
    if xs.shape[0] != n:
        raise ValueError(f"x has length {xs.shape[0]} but each line has length {n}")
    if labels is not None and len(labels) != n_lines:
        raise ValueError(
            f"labels length {len(labels)} does not match n_lines {n_lines}"
        )

    fig, ax = new_figure(figsize=(8.0, 3.5))
    for i in range(n_lines):
        ax.plot(
            xs,
            arr[i],
            linewidth=1.2,
            alpha=0.85,
            label=labels[i] if labels is not None else f"line {i}",
        )

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
    if labels is not None or n_lines <= 12:
        ax.legend(fontsize=8, loc="best", ncol=min(4, max(1, n_lines // 2)))
    return save_or_return(fig, save_path)


def plot_layer_trajectory(
    matrix: np.ndarray,
    *,
    residue_indices: Optional[Iterable[int]] = None,
    title: Optional[str] = None,
    xlabel: str = "layer",
    ylabel: str = "value",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot one channel's value across layers, one line per chosen residue.

    Parameters
    ----------
    matrix:
        2-D array of shape ``(L, N)`` -- layer index by residue index.
    residue_indices:
        Which residues to draw. Defaults to an evenly spaced sample of up to
        eight residues so the legend stays legible.
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(
            f"plot_layer_trajectory expects a 2-D (L, N) array, got shape {arr.shape!r}"
        )
    L, N = arr.shape

    if residue_indices is None:
        k = min(8, N)
        residue_indices = np.linspace(0, N - 1, k).round().astype(int).tolist()
    residues: List[int] = [int(r) for r in residue_indices]
    for r in residues:
        if not 0 <= r < N:
            raise ValueError(f"residue index {r} out of range [0, {N})")

    series = arr[:, residues].T  # (n_lines, L)
    labels = [f"residue {r}" for r in residues]
    xs = np.arange(L)

    fig = plot_lines(
        series,
        labels=labels,
        x=xs,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=None,
    )
    fig.axes[0].set_xlim(0, L - 1 if L > 1 else 1)
    return save_or_return(fig, save_path)
