"""End-to-end glue: raw OpenFold representation tensors -> matplotlib Figure.

This module is the contract between Priyavi's
``representation_tensor_utils`` (validation / channel selection /
aggregation / normalization) and the residue-indexed plot functions in
``viz``.

Typical caller:

    >>> from viz import heatmap_from_representation
    >>> fig = heatmap_from_representation(z, kind="pair", channel=12)

Once Priyavi's extraction layer lands, the same calls work against the
real ``m`` / ``z`` / ``s`` tensors -- no changes to plotting code required.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Optional, Sequence, Union

import numpy as np
from matplotlib.figure import Figure

from representation_tensor_utils import (
    AggregateMethod,
    NormalizeMethod,
    RepresentationKind,
    convert_to_numpy,
    prepare_heatmap_data,
    prepare_lineplot_data,
    validate_representation,
)
from viz.plots.heatmap import plot_heatmap, plot_heatmap_grid
from viz.plots.lineplot import plot_line, plot_lines

__all__ = [
    "heatmap_from_representation",
    "line_from_representation",
    "lines_from_representation",
    "pair_channel_grid",
]


def _default_heatmap_axis_labels(kind: RepresentationKind) -> tuple[str, str]:
    if kind == "pair":
        return ("residue j", "residue i")
    if kind == "single":
        return ("channel", "residue")
    return ("residue", "MSA depth")  # msa


def heatmap_from_representation(
    x: Union[np.ndarray, Any],
    kind: RepresentationKind,
    *,
    channel: Optional[int] = None,
    aggregate: Optional[AggregateMethod] = None,
    normalize: NormalizeMethod = "minmax",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = None,
    highlight_residues: Optional[Iterable[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Render a heatmap from a raw ``pair`` / ``single`` / ``msa`` tensor.

    Routes through :func:`representation_tensor_utils.prepare_heatmap_data`
    (validation + channel/aggregation + normalization), then hands the
    resulting 2-D array to :func:`viz.plot_heatmap`.

    Pass either ``channel=`` or ``aggregate=``, not both. The default for
    ``pair`` and ``msa`` is mean-aggregation when neither is given.
    """
    matrix = prepare_heatmap_data(
        x, kind, channel=channel, aggregate=aggregate, normalize=normalize
    )

    if title is None:
        if channel is not None:
            title = f"{kind} representation, channel {channel}"
        elif aggregate is not None:
            title = f"{kind} representation, {aggregate}-aggregated"
        else:
            title = f"{kind} representation"

    auto_x, auto_y = _default_heatmap_axis_labels(kind)
    return plot_heatmap(
        matrix,
        title=title,
        xlabel=xlabel if xlabel is not None else auto_x,
        ylabel=ylabel if ylabel is not None else auto_y,
        cmap=cmap,
        colorbar_label=colorbar_label,
        highlight_residues=highlight_residues if kind == "pair" else None,
        save_path=save_path,
    )


def line_from_representation(
    x: Union[np.ndarray, Any],
    kind: RepresentationKind,
    *,
    channel: int,
    title: Optional[str] = None,
    ylabel: str = "value",
    highlight_residues: Optional[Iterable[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Render a 1-D residue-indexed line plot from a raw representation tensor.

    Uses :func:`representation_tensor_utils.prepare_lineplot_data` so the
    sampling rule for each kind matches Priyavi's contract:

    - ``single``: ``s[:, channel]``
    - ``pair``: diagonal ``z[i, i, channel]``
    - ``msa``: mean over depth of ``m[:, :, channel]``
    """
    xs, ys = prepare_lineplot_data(x, kind, channel=channel)

    if title is None:
        title = f"{kind} representation, channel {channel}"

    fig = plot_line(
        ys,
        title=title,
        xlabel="residue",
        ylabel=ylabel,
        highlight_residues=highlight_residues,
        save_path=None,
    )
    fig.axes[0].set_xlim(int(xs[0]), int(xs[-1]) if len(xs) > 1 else int(xs[0]) + 1)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def lines_from_representation(
    x: Union[np.ndarray, Any],
    kind: RepresentationKind,
    *,
    channels: Sequence[int],
    title: Optional[str] = None,
    ylabel: str = "value",
    highlight_residues: Optional[Iterable[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Overlay several channels of one representation on a single residue axis.

    Internally calls :func:`prepare_lineplot_data` once per channel so the
    sampling rule per ``kind`` is identical to :func:`line_from_representation`.
    """
    if len(channels) == 0:
        raise ValueError("lines_from_representation requires at least one channel")

    series = []
    xs_ref: Optional[np.ndarray] = None
    for c in channels:
        xs, ys = prepare_lineplot_data(x, kind, channel=int(c))
        if xs_ref is None:
            xs_ref = xs
        series.append(ys)
    assert xs_ref is not None

    if title is None:
        title = f"{kind} representation, {len(channels)} channels"

    return plot_lines(
        np.stack(series, axis=0),
        labels=[f"channel {c}" for c in channels],
        x=xs_ref,
        title=title,
        xlabel="residue",
        ylabel=ylabel,
        highlight_residues=highlight_residues,
        save_path=save_path,
    )


def pair_channel_grid(
    x: Union[np.ndarray, Any],
    *,
    channels: Optional[Sequence[int]] = None,
    normalize: NormalizeMethod = "minmax",
    ncols: int = 4,
    suptitle: Optional[str] = None,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Render multiple pair-channel heatmaps in a grid.

    Convenience for "show me channels 0..K of the pair representation."
    Each panel is built independently via :func:`prepare_heatmap_data` so
    the per-channel ``minmax``/``zscore`` normalization matches the rest
    of the pipeline.
    """
    arr = convert_to_numpy(x, dtype=np.float64)
    info = validate_representation(arr, "pair")

    if channels is None:
        n_ch = int(info["n_channels"])
        k = min(8, n_ch)
        channels = list(range(k))

    mats = [
        prepare_heatmap_data(arr, "pair", channel=int(c), normalize=normalize)
        for c in channels
    ]
    titles = [f"channel {c}" for c in channels]

    return plot_heatmap_grid(
        mats,
        titles=titles,
        ncols=ncols,
        suptitle=suptitle if suptitle is not None else "pair representation",
        cmap=cmap,
        shared_clim=False,  # per-channel minmax already happened upstream
        colorbar_label=colorbar_label,
        save_path=save_path,
    )
