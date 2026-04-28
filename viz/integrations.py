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
    "attention_heatmap_from_artifact",
    "representation_tensor_from_artifact",
    "representation_heatmap_from_artifact",
    "representation_line_from_artifact",
    "figure_from_artifact",
]

AttentionKind = Literal["msa_row_attn", "triangle_start_attn"]


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


# ---------------------------------------------------------------------------
# EvoformerRunArtifact bridge (Pranav's extraction layer -> viz)
# ---------------------------------------------------------------------------
#
# A run captured by ``openfold.utils.evoformer_instrumentation`` produces an
# ``EvoformerRunArtifact`` (Pranav) that exposes:
#   - get_attention_matrix(kind, layer, head, residue_idx, mean_across_heads)
#   - reps: dict keyed by ``"layer_{LL:02d}.{msa|pair}"`` (or top-level "msa"/"pair")
#
# These four helpers turn that artifact directly into a Figure. They never
# touch the model -- only the on-disk artifact and the existing
# representation-tensor / plot pipeline.


def _format_attention_label(
    kind: str, layer: int, head: int, mean: bool, residue_idx: Optional[int]
) -> str:
    pretty = kind.replace("_", " ").title()
    extra = "mean over heads" if mean else f"head {head}"
    if residue_idx is not None:
        return f"{pretty} — layer {layer}, residue {residue_idx} ({extra})"
    return f"{pretty} — layer {layer} ({extra})"


def attention_heatmap_from_artifact(
    artifact: Any,
    kind: AttentionKind,
    layer: int,
    *,
    head: int = 0,
    mean_across_heads: bool = False,
    residue_idx: Optional[int] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = "attention weight",
    highlight_residues: Optional[Iterable[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Render an attention heatmap from an ``EvoformerRunArtifact``.

    Routes through :meth:`EvoformerRunArtifact.get_attention_matrix` (Pranav)
    to reconstruct the N x N matrix from the sparse top-K text file, then
    paints it with :func:`viz.plot_heatmap`.
    """
    matrix = artifact.get_attention_matrix(
        kind,
        layer=layer,
        head=head,
        residue_idx=residue_idx,
        mean_across_heads=mean_across_heads,
    )
    if title is None:
        title = _format_attention_label(
            kind, layer, head, mean_across_heads, residue_idx
        )
    return plot_heatmap(
        np.asarray(matrix),
        title=title,
        xlabel="residue j",
        ylabel="residue i",
        cmap=cmap,
        colorbar_label=colorbar_label,
        highlight_residues=highlight_residues,
        save_path=save_path,
    )


def representation_tensor_from_artifact(
    artifact: Any,
    rep_kind: RepresentationKind,
    layer: Optional[int] = None,
) -> np.ndarray:
    """Pull a per-layer representation tensor out of ``artifact.reps``.

    Looks up ``"layer_{LL:02d}.{rep_kind}"`` first (the key format written by
    :class:`EvoformerRecorder`), falls back to a top-level ``rep_kind`` key
    (the format expected by Pranav's ``plot_pair_mean`` etc.).

    Lazily calls :meth:`EvoformerRunArtifact.load_reps` if the artifact
    hasn't been loaded yet.
    """
    if getattr(artifact, "reps", None) is None:
        artifact.load_reps()
    reps = artifact.reps
    if reps is None:
        raise RuntimeError("artifact.load_reps() returned None")

    if layer is not None:
        key = f"layer_{int(layer):02d}.{rep_kind}"
        if key in reps:
            return convert_to_numpy(reps[key], dtype=np.float64)

    if rep_kind in reps:
        return convert_to_numpy(reps[rep_kind], dtype=np.float64)

    available = sorted(reps.keys())
    raise KeyError(
        f"Neither 'layer_{int(layer):02d}.{rep_kind}' nor '{rep_kind}' "
        f"found in artifact.reps. Available keys: {available[:8]}"
        + ("..." if len(available) > 8 else "")
    )


def representation_heatmap_from_artifact(
    artifact: Any,
    rep_kind: RepresentationKind,
    *,
    layer: Optional[int] = None,
    channel: Optional[int] = None,
    aggregate: Optional[AggregateMethod] = None,
    normalize: NormalizeMethod = "minmax",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **plot_kwargs: Any,
) -> Figure:
    """Render a heatmap of ``pair`` / ``msa`` / ``single`` from an artifact.

    Composition: artifact.reps -> :func:`heatmap_from_representation`.
    """
    tensor = representation_tensor_from_artifact(artifact, rep_kind, layer=layer)
    if title is None:
        layer_str = f" — layer {layer}" if layer is not None else ""
        if channel is not None:
            title = f"{rep_kind} representation{layer_str}, channel {channel}"
        elif aggregate is not None:
            title = f"{rep_kind} representation{layer_str}, {aggregate}-aggregated"
        else:
            title = f"{rep_kind} representation{layer_str}"
    return heatmap_from_representation(
        tensor,
        kind=rep_kind,
        channel=channel,
        aggregate=aggregate,
        normalize=normalize,
        title=title,
        save_path=save_path,
        **plot_kwargs,
    )


def representation_line_from_artifact(
    artifact: Any,
    rep_kind: RepresentationKind,
    *,
    channel: int,
    layer: Optional[int] = None,
    title: Optional[str] = None,
    ylabel: str = "value",
    highlight_residues: Optional[Iterable[int]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """Render a residue-indexed line plot from an artifact tensor.

    Composition: artifact.reps -> :func:`line_from_representation`.
    """
    tensor = representation_tensor_from_artifact(artifact, rep_kind, layer=layer)
    if title is None:
        layer_str = f" — layer {layer}" if layer is not None else ""
        title = f"{rep_kind} representation{layer_str}, channel {channel}"
    return line_from_representation(
        tensor,
        kind=rep_kind,
        channel=channel,
        title=title,
        ylabel=ylabel,
        highlight_residues=highlight_residues,
        save_path=save_path,
    )


def figure_from_artifact(
    artifact: Any,
    *,
    attn_kind: Optional[AttentionKind] = None,
    rep_kind: Optional[RepresentationKind] = None,
    layer: Optional[int] = None,
    head: int = 0,
    mean_across_heads: bool = False,
    residue_idx: Optional[int] = None,
    channel: Optional[int] = None,
    aggregate: Optional[AggregateMethod] = None,
    normalize: NormalizeMethod = "minmax",
    plot: Literal["heatmap", "line"] = "heatmap",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    **plot_kwargs: Any,
) -> Figure:
    """One-call dispatcher: artifact -> Figure for any supported view.

    Provide *either* ``attn_kind`` (attention path) *or* ``rep_kind``
    (representation path); not both. ``plot`` selects between heatmap and
    line for representations; attention is always rendered as a heatmap.
    """
    if (attn_kind is None) == (rep_kind is None):
        raise ValueError(
            "figure_from_artifact requires exactly one of attn_kind= or rep_kind="
        )

    if attn_kind is not None:
        if layer is None:
            raise ValueError("attention path requires layer=")
        return attention_heatmap_from_artifact(
            artifact,
            attn_kind,
            layer=layer,
            head=head,
            mean_across_heads=mean_across_heads,
            residue_idx=residue_idx,
            title=title,
            save_path=save_path,
            **plot_kwargs,
        )

    assert rep_kind is not None
    if plot == "heatmap":
        return representation_heatmap_from_artifact(
            artifact,
            rep_kind,
            layer=layer,
            channel=channel,
            aggregate=aggregate,
            normalize=normalize,
            title=title,
            save_path=save_path,
            **plot_kwargs,
        )
    if plot == "line":
        if channel is None:
            raise ValueError("line path requires channel=")
        return representation_line_from_artifact(
            artifact,
            rep_kind,
            channel=channel,
            layer=layer,
            title=title,
            save_path=save_path,
            **plot_kwargs,
        )
    raise ValueError(f"Unknown plot type: {plot!r}")
