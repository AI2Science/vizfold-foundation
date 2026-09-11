"""
Interactive N×N attention heatmap rendered with Plotly.
Builds a dense matrix from sparse (res1, res2, weight) connections.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_heatmap(
    connections: List[Tuple[int, int, float]],
    fasta_seq: str,
    head_label: str,
) -> None:
    n = len(fasta_seq)
    if n == 0 or not connections:
        st.info("No attention data to display.")
        return

    matrix = _build_matrix(connections, n)

    tick_step = max(1, n // 20)
    tick_vals = list(range(0, n, tick_step))
    tick_text = [fasta_seq[i] for i in tick_vals]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            colorscale="Blues",
            colorbar=dict(title="Attention", thickness=14),
            hovertemplate="Source %{y} → Target %{x}<br>Score: %{z:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"Attention Map — {head_label}", font=dict(size=14)),
        xaxis=dict(
            title="Target residue",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Source residue",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10),
            autorange="reversed",
        ),
        margin=dict(l=60, r=20, t=50, b=60),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_matrix(
    connections: List[Tuple[int, int, float]], n: int
) -> np.ndarray:
    matrix = np.zeros((n, n))
    for r1, r2, w in connections:
        if r1 < n and r2 < n:
            matrix[r1, r2] += w
            matrix[r2, r1] += w
    return matrix
