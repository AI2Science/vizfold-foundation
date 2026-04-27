"""
Tensor and array utilities for OpenFold representation visualization (Issue #8).

Sits between extraction (model hooks) and rendering (matplotlib / Flask /
notebooks). Typical trunk shapes after squeezing leading batch dimensions:

- pair: [..., N_res, N_res, C_z]
- msa: [..., N_seq, N_res, C_m]
- single: [..., N_res, C_s]
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

RepresentationKind = Literal["pair", "msa", "single"]
AggregateMethod = Literal["mean", "max", "l2"]
NormalizeMethod = Literal["minmax", "zscore", "none"]


def convert_to_numpy(
    x: Union[np.ndarray, "torch.Tensor"],
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Convert torch tensor or array to a host NumPy array (detached if torch)."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return np.asarray(x, dtype=dtype)


def summarize_tensor(x: Union[np.ndarray, Any]) -> Dict[str, Any]:
    """Debug summary: shape, dtype, finite stats, NaN/Inf counts."""
    arr = convert_to_numpy(x, dtype=np.float64)
    finite = np.isfinite(arr)
    return {
        "shape": tuple(int(s) for s in arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr[finite])) if finite.any() else None,
        "max": float(np.max(arr[finite])) if finite.any() else None,
        "mean": float(np.mean(arr[finite])) if finite.any() else None,
        "std": float(np.std(arr[finite])) if finite.any() else None,
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
    }


def validate_representation(
    x: Union[np.ndarray, Any],
    kind: RepresentationKind,
    *,
    n_res: Optional[int] = None,
    n_msa: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validate trailing axis layout for ``pair`` / ``msa`` / ``single``.

    Leading dimensions (batch, recycling) are allowed. ``pair`` must be square
    on the last two spatial axes.
    """
    arr = convert_to_numpy(x, dtype=np.float64)

    if kind == "single":
        if arr.ndim < 2:
            raise ValueError(
                f"single representation expects at least 2 dimensions, got {arr.shape}"
            )
        n, c = int(arr.shape[-2]), int(arr.shape[-1])
        if n_res is not None and n != n_res:
            raise ValueError(f"single: expected N_res={n_res}, got {n}")
        return {"kind": kind, "n_res": n, "n_channels": c, "trailing": (n, c)}

    if arr.ndim < 3:
        raise ValueError(
            f"{kind} representation expects at least 3 dimensions, got {arr.shape}"
        )

    if kind == "msa":
        s, n, c = int(arr.shape[-3]), int(arr.shape[-2]), int(arr.shape[-1])
        if n_res is not None and n != n_res:
            raise ValueError(f"msa: expected N_res={n_res}, got {n}")
        if n_msa is not None and s != n_msa:
            raise ValueError(f"msa: expected N_seq={n_msa}, got {s}")
        return {
            "kind": kind,
            "n_msa": s,
            "n_res": n,
            "n_channels": c,
            "trailing": (s, n, c),
        }

    if kind == "pair":
        n0, n1, c = int(arr.shape[-3]), int(arr.shape[-2]), int(arr.shape[-1])
        if n0 != n1:
            raise ValueError(f"pair: expected square N×N, got ({n0}, {n1})")
        if n_res is not None and n0 != n_res:
            raise ValueError(f"pair: expected N_res={n_res}, got {n0}")
        return {"kind": kind, "n_res": n0, "n_channels": c, "trailing": (n0, n0, c)}

    raise ValueError(f"Unknown kind: {kind!r}")


def select_channel(x: Union[np.ndarray, Any], channel: int) -> np.ndarray:
    """Select one channel from the last dimension."""
    arr = convert_to_numpy(x, dtype=np.float64)
    if channel < 0 or channel >= arr.shape[-1]:
        raise IndexError(f"channel {channel} out of bounds for shape {arr.shape}")
    return arr[..., channel]


def aggregate_channels(
    x: Union[np.ndarray, Any],
    method: AggregateMethod = "mean",
) -> np.ndarray:
    """Collapse the last dimension with mean, max, or L2 norm."""
    arr = convert_to_numpy(x, dtype=np.float64)
    if method == "mean":
        return np.mean(arr, axis=-1)
    if method == "max":
        return np.max(arr, axis=-1)
    if method == "l2":
        return np.sqrt(np.sum(arr * arr, axis=-1))
    raise ValueError(f"Unknown aggregate method: {method!r}")


def normalize_array(
    arr: np.ndarray,
    method: NormalizeMethod = "minmax",
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize for display: ``minmax`` to [0, 1], ``zscore``, or ``none``.
    Non-finite values are masked to 0 in the output for minmax/zscore.
    """
    out = np.array(arr, dtype=np.float64, copy=True)
    if method == "none":
        return out

    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros_like(out)

    if method == "minmax":
        lo = float(np.min(out[finite]))
        hi = float(np.max(out[finite]))
        denom = max(hi - lo, eps)
        out = (out - lo) / denom
        out[~finite] = 0.0
        return out

    if method == "zscore":
        mu = float(np.mean(out[finite]))
        sigma = float(np.std(out[finite]))
        sigma = max(sigma, eps)
        out = (out - mu) / sigma
        out[~finite] = 0.0
        return out

    raise ValueError(f"Unknown normalize method: {method!r}")


def _squeeze_to_pair_map(arr: np.ndarray) -> np.ndarray:
    x = arr
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(
            f"pair heatmap expects rank 3 after squeezing singleton batch dims, "
            f"got {arr.shape} -> {x.shape}"
        )
    return x


def _squeeze_to_msa(arr: np.ndarray) -> np.ndarray:
    x = arr
    while x.ndim > 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"msa expects rank 3 (S, N, C), got {x.shape}")
    return x


def _squeeze_to_single(arr: np.ndarray) -> np.ndarray:
    x = arr
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"single expects rank 2 (N, C), got {x.shape}")
    return x


def prepare_heatmap_data(
    x: Union[np.ndarray, Any],
    kind: RepresentationKind,
    *,
    channel: Optional[int] = None,
    aggregate: Optional[AggregateMethod] = None,
    normalize: NormalizeMethod = "minmax",
) -> np.ndarray:
    """
    Build a 2D NumPy array for ``imshow``-style heatmaps.

    - **pair**: residue × residue (one channel or aggregated over channels).
    - **single**: residue × channel (rows = residues).
    - **msa**: depth × residue after collapsing channels (shape ``(N_seq, N_res)``).
    """
    if channel is not None and aggregate is not None:
        raise ValueError("Specify at most one of channel= and aggregate=")

    arr = convert_to_numpy(x, dtype=np.float64)
    validate_representation(arr, kind)

    if kind == "pair":
        h = _squeeze_to_pair_map(arr)
        if channel is not None:
            plane = select_channel(h, channel)
        elif aggregate is not None:
            plane = aggregate_channels(h, aggregate)
        else:
            plane = aggregate_channels(h, "mean")
        return normalize_array(plane, normalize)

    if kind == "single":
        h = _squeeze_to_single(arr)
        return normalize_array(h, normalize)

    if kind == "msa":
        h = _squeeze_to_msa(arr)
        if channel is not None:
            plane = select_channel(h, channel)
        elif aggregate is not None:
            plane = aggregate_channels(h, aggregate)
        else:
            plane = aggregate_channels(h, "mean")
        if plane.ndim != 2:
            raise ValueError(f"internal: expected 2D plane for msa, got {plane.shape}")
        return normalize_array(plane, normalize)

    raise ValueError(f"Unknown kind: {kind!r}")


def prepare_lineplot_data(
    x: Union[np.ndarray, Any],
    kind: RepresentationKind,
    *,
    channel: int,
    pair_slice: Literal["diagonal"] = "diagonal",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(x_index, y_values)`` for ``plot`` — residue index vs signal.

    **pair** uses the diagonal ``z[i, i, channel]``. **msa** averages the
    chosen channel over the sequence (depth) axis per residue.
    """
    arr = convert_to_numpy(x, dtype=np.float64)
    validate_representation(arr, kind)

    if kind == "single":
        h = _squeeze_to_single(arr)
        n = h.shape[0]
        y = select_channel(h, channel)
        return np.arange(n, dtype=np.int64), y

    if kind == "pair":
        if pair_slice != "diagonal":
            raise ValueError(f"Unsupported pair_slice: {pair_slice!r}")
        h = _squeeze_to_pair_map(arr)
        n = h.shape[0]
        idx = np.arange(n, dtype=np.int64)
        y = h[idx, idx, channel]
        return idx, y

    if kind == "msa":
        h = _squeeze_to_msa(arr)
        n = h.shape[1]
        y = np.mean(h[:, :, channel], axis=0)
        return np.arange(n, dtype=np.int64), y

    raise ValueError(f"Unknown kind: {kind!r}")
