"""Synthetic tensor helpers used by demos and example PNGs.

These shapes intentionally mirror what the extraction layer
(Priyavi/Pranav) is expected to hand off, so the visualization functions
can be exercised end-to-end before extraction lands.

REPLACE WITH REAL TENSORS ONCE AVAILABLE FROM viz extraction layer.
"""

from __future__ import annotations

import numpy as np


def fake_attention_heads(N: int = 64, H: int = 8, seed: int = 0) -> np.ndarray:
    """Return ``(H, N, N)`` per-head attention-style maps.

    Each head gets a diagonal band plus one or two off-diagonal blobs at
    head-dependent positions, with a small noise floor.
    """
    rng = np.random.default_rng(seed)
    ii, jj = np.indices((N, N))
    out = np.empty((H, N, N), dtype=np.float32)
    for h in range(H):
        sigma = 2.5 + 0.5 * h
        m = np.exp(-(ii - jj) ** 2 / (2 * sigma ** 2))
        cx = (h * 7) % N
        cy = (N - 1 - cx) % N
        m += 0.5 * np.exp(-((ii - cx) ** 2 + (jj - cy) ** 2) / (2 * 4.0 ** 2))
        if h % 2 == 0:
            cx2 = (h * 11 + 5) % N
            cy2 = (cx2 + N // 4) % N
            m += 0.4 * np.exp(-((ii - cx2) ** 2 + (jj - cy2) ** 2) / (2 * 3.0 ** 2))
        m += 0.03 * rng.random((N, N))
        out[h] = m
    return out


def fake_pair_channel(N: int = 64, seed: int = 0) -> np.ndarray:
    """Return ``(N, N)`` mimicking one channel of the pair representation z."""
    rng = np.random.default_rng(seed)
    ii, jj = np.indices((N, N))
    base = np.cos((ii - jj) / 6.0) * np.exp(-np.abs(ii - jj) / 20.0)
    base += 0.1 * rng.standard_normal((N, N))
    return base.astype(np.float32)


def fake_single_channels(N: int = 64, C: int = 8, seed: int = 0) -> np.ndarray:
    """Return ``(N, C)`` mimicking the single representation s.

    Each channel is a smooth low-frequency signal plus noise so a multi-line
    overlay actually looks like distinct channels.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(N)
    out = np.empty((N, C), dtype=np.float32)
    for c in range(C):
        freq = 0.05 + 0.04 * c
        phase = 0.7 * c
        amp = 1.0 + 0.15 * c
        out[:, c] = amp * np.sin(2 * np.pi * freq * x + phase) + 0.15 * rng.standard_normal(N)
    return out


def fake_layer_trajectory(L: int = 48, N: int = 64, seed: int = 0) -> np.ndarray:
    """Return ``(L, N)`` for "one channel value across layers".

    Layer index is the row; residue index is the column. Each residue's
    trajectory across layers is a smooth curve, so plotting a few residues
    gives interpretable lines.
    """
    rng = np.random.default_rng(seed)
    layers = np.arange(L)
    out = np.empty((L, N), dtype=np.float32)
    for r in range(N):
        center = 6 + (r % 7) * 5
        width = 8 + (r % 4) * 2
        amp = 1.0 + 0.02 * r
        bell = amp * np.exp(-((layers - center) ** 2) / (2 * width ** 2))
        drift = 0.005 * (r - N / 2) * layers
        out[:, r] = bell + drift + 0.05 * rng.standard_normal(L)
    return out


def fake_distribution(N: int = 10_000, seed: int = 0) -> np.ndarray:
    """Return a 1-D mixture-of-gaussians sample for histogram demos."""
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-1.0, scale=0.6, size=N // 2)
    b = rng.normal(loc=1.5, scale=0.9, size=N - N // 2)
    return np.concatenate([a, b]).astype(np.float32)
